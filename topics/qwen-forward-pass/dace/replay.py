"""Compile and inspect one Qwen3.8 block-3 decode attention replay with DaCe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
LOCAL = HERE.parents[2] / ".local/tools/dace"
LOCAL.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("DACE_CONFIG", str(LOCAL / "dace.conf"))
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
os.environ["OMP_NUM_THREADS"] = "1"

import dace
import numpy as np

SPEC = json.loads((ASSETS / "workload.json").read_text())
H, D = SPEC["q_shape"][1::2]
G, T = SPEC["k_shape"][1:3]
SHAPES = {"q": (H, D), "k": (G, T, D), "v": (G, T, D), "scores": (H, T),
          "peak": (H,), "exps": (H, T), "denom": (H,), "probs": (H, T), "out": (H, D)}


def make_graph():
    graph = dace.SDFG("qwen_block3_decode")
    for name, shape in SHAPES.items():
        graph.add_array(name, shape, dace.float32)
    previous = None

    def stage(name, ranges, inputs, code, output):
        nonlocal previous
        state = graph.add_state(name)
        if previous is not None:
            graph.add_edge(previous, state, dace.InterstateEdge())
        previous = state
        state.add_mapped_tasklet(
            name, ranges, {key: dace.Memlet(value) for key, value in inputs.items()},
            code, {"result": dace.Memlet(output)}, external_edges=True,
            schedule=dace.ScheduleType.Sequential, language=dace.Language.CPP,
        )

    heads = {"h": f"0:{H}"}
    positions = {**heads, "t": f"0:{T}"}
    features = {**heads, "d": f"0:{D}"}
    group = f"h//{H // G}"
    stage("scaled_dot", positions, {"query": f"q[h, 0:{D}]", "keys": f"k[{group}, t, 0:{D}]"},
          f"""float acc = 0.0f;
for (int j = 0; j < {D}; ++j) acc += query[j] * keys[j];
result = acc * {SPEC['scale']}f;""", "scores[h, t]")
    stage("row_max", heads, {"row": f"scores[h, 0:{T}]"},
          f"""float maximum = row[0];
for (int j = 1; j < {T}; ++j) maximum = std::max(maximum, row[j]);
result = maximum;""", "peak[h]")
    stage("shift_exp", positions, {"score": "scores[h, t]", "maximum": "peak[h]"},
          "result = std::exp(score - maximum);", "exps[h, t]")
    stage("row_sum", heads, {"row": f"exps[h, 0:{T}]"},
          f"""float acc = 0.0f;
for (int j = 0; j < {T}; ++j) acc += row[j];
result = acc;""", "denom[h]")
    stage("normalize", positions, {"weight": "exps[h, t]", "total": "denom[h]"},
          "result = weight / total;", "probs[h, t]")
    # The value connector starts at V[group,0,d]. In this C-contiguous cache,
    # moving to the next token advances D FP32 values.
    stage("value_mix", features, {"weights": f"probs[h, 0:{T}]", "values": f"v[{group}, 0:{T}, d]"},
          f"""float acc = 0.0f;
for (int j = 0; j < {T}; ++j) acc += weights[j] * values[j * {D}];
result = acc;""", "out[h, d]")
    graph.validate()
    return graph


def inspect_stages(graph):
    stages = []
    for state in graph.states():
        tasklet = next(node for node in state.nodes() if isinstance(node, dace.nodes.Tasklet))
        entry = state.scope_dict()[tasklet]
        copies = int(entry.map.range.num_elements())
        entry_data = {}
        for key, edges in [("reads", state.in_edges(tasklet)), ("writes", state.out_edges(tasklet))]:
            items = []
            for edge in sorted(edges, key=lambda e: e.data.data):
                memlet = edge.data
                elements = int(memlet.subset.num_elements())
                size = graph.arrays[memlet.data].dtype.bytes
                items.append(dict(array=memlet.data, subset=str(memlet.subset),
                                  elements_per_iteration=elements, bytes=copies * elements * size))
            entry_data[key] = items
        stages.append(dict(name=state.label, map=dict(zip(entry.map.params, map(str, entry.map.range))),
                           iterations=copies, code=tasklet.code.as_string, **entry_data))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ASSETS / "layer3-decode.npz")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dace.Config.set("default_build_folder", value=str(LOCAL / "build"))
    # Preserve ordinary FP32 math for comparison to the common reference.
    strict_args = dace.Config.get("compiler", "cpu", "args").replace("-ffast-math", "")
    if platform.system() == "Darwin":
        # DaCe 1.0.2 reduction.h uses a dependent template call rejected by
        # Apple Clang 17 under its default compatibility diagnostic.
        strict_args += " -Wno-missing-template-arg-list-after-template-kw"
    dace.Config.set("compiler", "cpu", "args", value=strict_args)
    if platform.system() == "Darwin" and shutil.which("brew"):
        omp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(filter(None, [omp, os.getenv("CMAKE_PREFIX_PATH")]))
    graph = make_graph()
    stages = inspect_stages(graph)
    # Remove generator host paths in source-location tags before distributing the graph.
    doc = graph.to_json()
    def clean_locations(obj):
        if isinstance(obj, list):
            for value in obj:
                clean_locations(value)
        elif isinstance(obj, dict):
            if isinstance(obj.get("filename"), str):
                obj["filename"] = Path(obj["filename"]).name
            for value in obj.values():
                clean_locations(value)
    clean_locations(doc)
    sdfg_path = args.output / "decode.sdfg"
    sdfg_path.write_text(json.dumps(doc, indent=2) + "\n")
    graph = dace.SDFG.from_file(str(sdfg_path))
    graph.validate()
    compiled = graph.compile()
    generated_path = Path(graph.build_folder) / "src/cpu" / (graph.name + ".cpp")
    shutil.copyfile(generated_path, args.output / "decode.cpp")

    with np.load(args.fixture, allow_pickle=False) as tensors:
        for name in ("q", "k", "v"):
            assert list(tensors[name].shape) == SPEC[name + "_shape"]
            assert tensors[name].dtype == np.float32
        buffers = {name: np.empty(shape, dtype=np.float32) for name, shape in SHAPES.items()}
        for name in ("q", "k", "v"):
            np.copyto(buffers[name], tensors[name].reshape(SHAPES[name]))
        expected = tensors["expected"].reshape(H, D)
    compiled(**buffers)
    np.testing.assert_allclose(buffers["out"], expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(buffers["probs"].sum(axis=1), 1, rtol=1e-6, atol=1e-6)

    # Time the public Python entry point. Intermediate and output buffers are preallocated.
    warmups, samples = 15, 151
    for _ in range(warmups):
        compiled(**buffers)
    wall, cpu = [], []
    for _ in range(samples):
        start_cpu = time.process_time_ns()
        start = time.perf_counter_ns()
        compiled(**buffers)
        wall.append((time.perf_counter_ns() - start) / 1_000)
        cpu.append((time.process_time_ns() - start_cpu) / 1_000)

    graph.name = "qwen_block3_states"
    for state in graph.states():
        state.instrument = dace.InstrumentationType.Timer
    # Retain one native timer event per state, in a separate profiled invocation.
    dace.Config.set("instrumentation", "report_each_invocation", value=False)
    profiled = graph.compile()
    profiled(**buffers)
    profiled.finalize()
    latest = max((Path(graph.build_folder) / "perf").glob("report-*.json"), key=lambda p: p.stat().st_mtime_ns)
    trace = json.loads(latest.read_text())
    state_events = {e["name"]: e for e in trace["traceEvents"] if e.get("ph") == "X"}
    for stage in stages:
        stage["trace_us"] = state_events["State " + stage["name"]]["dur"]
    shutil.copyfile(latest, args.output / "state-trace.json")
    chip = platform.machine()
    if platform.system() == "Darwin":
        hardware = subprocess.check_output(["system_profiler", "SPHardwareDataType", "-json"], text=True)
        chip = json.loads(hardware)["SPHardwareDataType"][0].get("chip_type", chip)

    result = dict(cpu_model=chip,
        model=SPEC["model"], revision=SPEC["revision"], fixture=args.fixture.name,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        versions=dict(dace=dace.__version__, numpy=np.__version__, python=platform.python_version(),
                      system=platform.system(), system_release=platform.release(), arch=platform.machine()),
        schedule="Sequential", omp_threads=1, warmups=warmups, samples=samples,
        wall_us=dict(zip(("p10", "median", "p90"), np.percentile(wall, [10, 50, 90]).tolist())),
        cpu_us_median=float(np.median(cpu)),
        max_abs_error=float(np.max(np.abs(expected - buffers["out"]))),
        rms_error=float(np.sqrt(np.mean((expected - buffers["out"]) ** 2))),
        stages=stages, unique_bytes={name: arr.nbytes for name, arr in buffers.items()},
        head_values={name: buffers[name].tolist() for name in ("scores", "peak", "exps", "denom", "probs")},
        # Feature zero across all heads is enough to inspect value mixing for each head.
        value_feature0=buffers["v"][:, :, 0].tolist(),
        output_feature0=buffers["out"][:, 0].tolist(),
    )
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Same data for offline file:// viewing without fetch or CORS restrictions.
    (args.output / "results.js").write_text(
        "window.DACE_REPLAY = " + json.dumps(result, separators=(",", ":"), allow_nan=False) + ";\n")
    print(json.dumps({name:result[name] for name in ("fixture","wall_us","cpu_us_median","max_abs_error","rms_error")}, indent=2))


if __name__ == "__main__":
    main()
