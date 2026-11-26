"""Compile and measure one slice. Launch each shape in a fresh Python process."""
import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE.parent / "assets"
sys.path.insert(0, str(ASSETS))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case", choices=["attention", "mlp-prefill", "mlp-decode"])
parser.add_argument("--checkpoint", type=Path, default=ROOT / ".local/qwen38/checkpoint")
parser.add_argument("--scratch", type=Path, default=ROOT / ".local/tools/torchinductor")
parser.add_argument("--out", type=Path, default=HERE / "evidence")
args = parser.parse_args()
args.scratch.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix=args.case + "-", dir=args.scratch)).resolve()
out = args.out / args.case
out.mkdir(parents=True, exist_ok=True)
os.environ.update(TORCH_COMPILE_DEBUG="1", TORCH_COMPILE_DEBUG_DIR=str(run),
                  TORCHINDUCTOR_CACHE_DIR=str(run / "cache"), TORCHINDUCTOR_COMPILE_THREADS="1",
                  OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1")

import numpy as np
import torch
from safetensors import safe_open
from workload import AttentionCore, SPEC, load_replay

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def mlp(x, gate, up, down):
    linear = torch.nn.functional.linear
    return linear(torch.nn.functional.silu(linear(x, gate)) * linear(x, up), down)


def tensor_info(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "bytes": tensor.numel() * tensor.element_size()}


def get_case():
    if args.case == "attention":
        q, k, v, reference = load_replay()
        return AttentionCore().eval(), [q, k, v], ["query", "keys", "values"], reference, {}
    phase = args.case.split("-")[1]
    with np.load(ASSETS / "layer3-mlp.npz", allow_pickle=False) as capture:
        activation = torch.from_numpy(capture[phase + "_input"]).bfloat16()
        reference = torch.from_numpy(capture[phase + "_output"]).bfloat16()
    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights, names, sources = [], [], {}
    for role in ("gate", "up", "down"):
        name = f"model.language_model.layers.3.mlp.{role}_proj.weight"
        shard = weight_map[name]
        with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as handle:
            weights.append(handle.get_tensor(name))
        names.append(name)
        sources[name] = shard
    return mlp, [activation, *weights], ["hidden", *names], reference, sources


def measure(function, inputs, repeats, batch):
    wall_us, cpu_us = [], []
    for _ in range(repeats):
        start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
        for _ in range(batch):
            function(*inputs)
        cpu_us.append((time.process_time_ns() - start_cpu) / 1000 / batch)
        wall_us.append((time.perf_counter_ns() - start_wall) / 1000 / batch)
    return {"batches": repeats, "calls_per_batch": batch,
            "wall_us_per_call": wall_us, "median_wall_us": statistics.median(wall_us),
            "process_cpu_us_per_call": cpu_us, "median_process_cpu_us": statistics.median(cpu_us)}


def errors(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "rmse": delta.square().mean().sqrt().item(), "num_values": delta.numel(),
            "count_exact": int((delta == 0).sum()), "reference_abs_max": expected.abs().max().item(),
            "expected_rms": expected.float().square().mean().sqrt().item()}


def sanitize(source):
    """Only compiler comments contain machine/source paths. Keep executable text intact."""
    lines = []
    for line in source.splitlines(keepends=True):
        if "#" in line:
            pre, comment = line.split("#", 1)
            comment = comment.replace(str(ROOT) + "/", "")
            comment = re.sub(r"(?:/private)?/var/folders/[^\s,]+", "<temporary-path>", comment)
            comment = re.sub(r"(?:\.local|/Users|/private|/opt|/usr)/[^\s,:]+/site-packages/", "site-packages/", comment)
            if comment.lstrip().startswith("kernel path:"):
                comment = " kernel path: <compiler-cache>"
            line = pre + "#" + comment
            if not line.endswith("\n"):
                line += "\n"
        lines.append(line)
    return "".join(lines).rstrip("\n") + "\n"


def record_calls(compiled, inputs, source):
    """Wrap only generated C++ and external ATen call sites for one separate execution."""
    tree = ast.parse(source)
    refs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.startswith(("cpp_fused_", "extern_kernels.", "torch.ops.aten.")) and name not in refs:
            refs.append(name)
    from torch._inductor.codecache import PyCodeCache
    module = next(m for m in reversed(PyCodeCache.modules) if hasattr(m, "call") and
                  all(hasattr(m, name.split(".")[0]) for name in refs))
    events = []
    start = 0
    def observe(name, original):
        def timed(*call_args, **kwargs):
            at = time.perf_counter_ns()
            returned = original(*call_args, **kwargs)
            end = time.perf_counter_ns()
            events.append(dict(name=name, start_us=(at-start)/1000, end_us=(end-start)/1000,
                               elapsed_us=(end-at)/1000, thread=threading.get_ident(),
                               positional=[tensor_info(v) for v in call_args if torch.is_tensor(v)],
                               named={k:tensor_info(v) for k,v in kwargs.items() if torch.is_tensor(v)},
                               returned=[tensor_info(v) for v in (returned if isinstance(returned, tuple) else (returned,))
                                         if torch.is_tensor(v)]))
            return returned
        return timed
    from unittest.mock import patch
    with contextlib.ExitStack() as stack:
        for name in refs:
            owner = module
            for component in name.split(".")[:-1]:
                owner = getattr(owner, component)
            attr = name.split(".")[-1]
            stack.enter_context(patch.object(owner, attr, observe(name, getattr(owner, attr))))
        start = time.perf_counter_ns()
        compiled(*inputs)
    lanes = {}
    for event in events:
        ident = event["thread"]
        event["thread"] = lanes.setdefault(ident, len(lanes))
    if not events:
        raise RuntimeError("No instrumented calls were observed")
    return events


def capture_backend(graph, example_inputs):
    from torch._dynamo.backends.registry import lookup_backend
    (out / "dynamo_fx.txt").write_text(sanitize(graph.print_readable(print_output=False)))
    return lookup_backend("inductor")(graph, example_inputs)


with torch.inference_mode():
    function, inputs, names, reference, sources = get_case()
    eager = function(*inputs)
    compiled = torch.compile(function, backend=capture_backend, fullgraph=True, dynamic=False)
    started = time.perf_counter()
    result = compiled(*inputs)
    first_call_s = time.perf_counter() - started
    debug = next(run.glob("torch_compile_debug/run_*/torchinductor/*/output_code.py")).parent
    for filename in ("fx_graph_readable.py", "fx_graph_transformed.py", "ir_pre_fusion.txt",
                     "ir_post_fusion.txt", "output_code.py"):
        (out / filename).write_text(sanitize((debug / filename).read_text()))
    code = (debug / "output_code.py").read_text()
    events = record_calls(compiled, inputs, code)
    compiled(*inputs)
    batch = 40 if args.case == "attention" else 1
    times = measure(compiled, inputs, repeats=11, batch=batch)
    fixture = ASSETS / ("layer3-decode.npz" if args.case == "attention" else "layer3-mlp.npz")
    options = {"rtol": 1e-5, "atol": 1e-6} if args.case == "attention" else {"rtol": .016, "atol": .003}
    close = torch.isclose(result.float(), reference.float(), **options)
    report = {"case": args.case, "revision": SPEC["revision"], "torch": str(torch.__version__),
              "python": platform.python_version(), "host": platform.platform(), "threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(), "inputs": dict(zip(names, map(tensor_info, inputs))),
              "output": tensor_info(result), "weight_shards": sources, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "first_compiled_call_s": first_call_s, "timing": times, "wrapper_calls": events,
              "generated_openmp_parallel_directives": re.findall(r"^\s*#pragma omp parallel.*$", code, re.M),
              "compiled_vs_reference": errors(result, reference), "eager_vs_reference": errors(eager, reference),
              "compiled_vs_eager": errors(result, eager), "check": {**options, "pass": bool(close.all()),
                "out_of_tolerance": int((~close).sum())}}
    (out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.case, json.dumps({k:report[k] for k in ("compiled_vs_reference","check")}), flush=True)
    torch.testing.assert_close(result.float(), reference.float(), **options)
