"""Capture the shared numerical replay with TorchLens and package its evidence."""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import torchlens as tl
from torchlens.options import CaptureOptions

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
from workload import ASSETS, AttentionCore, SPEC, load_replay


def fact_shape(facts):
    return [list(fact.shape) for fact in facts]


def operation(op):
    source = next((f for f in reversed(op.source.frames) if Path(f.file).name == "workload.py"), None)
    out = op.out
    flat = out.flatten()
    return {
        "id": op.layer_label, "label": op.label, "function": op.func_name,
        "parents": list(op.parents), "shape": list(op.shape), "dtype": str(op.dtype),
        "tensor_bytes": int(op.cost.activation_memory), "input_shapes": [list(s) for s in op.input_shapes],
        "read_bytes": int(op.cost.bytes_read), "write_bytes": int(op.cost.bytes_written),
        "macs": None if op.cost.macs_forward is None else int(op.cost.macs_forward),
        "call_us": round(1e6 * float(op.cost.duration), 2),
        "source": None if source is None else {"line": source.line_number, "code": source.call_line.strip()},
        "first_four": flat[:4].tolist(), "minimum": float(flat.min()), "maximum": float(flat.max()),
        "aten": [{"sequence": a.sequence, "name": a.namespace + "::" + a.operator + "." + a.overload,
                  "inputs": fact_shape(a.input_tensor_facts), "outputs": fact_shape(a.output_tensor_facts),
                  "view_copy_kind": a.view_copy_kind} for a in op.aten_ops],
    }


def capture(inp, expected, source_hash, output_dir):
    torch.set_num_threads(SPEC["torch_threads_replay"])
    torch.set_num_interop_threads(1)
    with torch.no_grad():
        trace = tl.trace(AttentionCore().eval(), inp, save="all",
                         capture=CaptureOptions(aten_ops=True, unwrap_when_done=True))
        error = float((trace.output - expected).abs().max())
        torch.testing.assert_close(trace.output, expected, atol=1e-6, rtol=1e-5)
    records = [operation(op) for op in trace]
    by_function = [r["function"] for r in records]
    assert by_function == ["none", "none", "none", "repeat_interleave", "repeat_interleave",
                           "transpose", "matmul", "__mul__", "softmax", "matmul", "none"]
    scores = trace[records[7]["id"]].out.squeeze(0).squeeze(1)
    weights = trace[records[8]["id"]].out.squeeze(0).squeeze(1)
    data = {
        "model": SPEC["model"], "revision": SPEC["revision"], "fixture_sha256": source_hash,
        "torchlens": tl.__version__, "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "dtype": "float32", "max_abs_error": error,
        "verified_elements": trace.output.numel(), "output_shape": list(trace.output.shape),
        "compute_nodes": len(trace.ops), "edges": trace.num_edges,
        "operator_attributions": sum(len(op.aten_ops) for op in trace.ops),
        "token_ids": SPEC["prompt_ids"] + [SPEC["decode_id"]],
        "scores": scores.tolist(), "probabilities": weights.tolist(),
        "max_softmax_row_sum_error": float((weights.sum(dim=-1) - 1).abs().max()), "nodes": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trace.json").write_text(json.dumps(data, indent=2) + "\n")
    tl.export.svg(trace, output_dir / "torchlens.svg")
    # The native report offers a schematic when a Graphviz layout is unavailable.
    tl.export.html(trace, output_dir / "native-report.html")
    return data


def refresh_page(page, data, structure):
    html = page.read_text()
    topology = base64.b64encode((HERE / "text-topology.json.gz").read_bytes()).decode("ascii")
    for name, obj in [("capture", data), ("structure", structure), ("topology", topology)]:
        tag = f'<script id="{name}" type="application/json">'
        payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).replace("<", r"\u003c")
        html, count = re.subn(re.escape(tag) + r".*?</script>", lambda _: tag + payload + "</script>",
                              html, count=1, flags=re.DOTALL)
        assert count == 1, f"Missing embedded {name} data slot"
    page.write_text(html)


if __name__ == "__main__":
    q, k, v, expected = load_replay()
    digest = hashlib.sha256((ASSETS / "layer3-decode.npz").read_bytes()).hexdigest()
    data = capture((q, k, v), expected, digest, HERE)
    refresh_page(HERE / "index.html", data, json.loads((HERE / "structure.json").read_text()))
    print(f"{data['compute_nodes']} compute nodes, {data['verified_elements']} elements, "
          f"maximum absolute error {data['max_abs_error']}")
