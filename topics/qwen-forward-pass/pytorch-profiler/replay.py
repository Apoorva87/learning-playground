"""Profile one actual layer-3 grouped-query attention replay on one CPU thread."""
import gzip
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.profiler import ExecutionTraceObserver, ProfilerActivity, profile, record_function

from collect import HERE, save_trace, tensors
sys.path.insert(0, str(HERE.parent / "assets"))
from workload import ASSETS, AttentionCore, load_replay


def capture():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    q, k, v, expected = load_replay()
    model = AttentionCore().eval()
    with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
        for _ in range(5):
            model(q, k, v)
        et = Path(temp) / "execution-trace.json"
        kineto = Path(temp) / "replay.json"
        observer = ExecutionTraceObserver().register_callback(str(et))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
                     with_flops=True, execution_trace_observer=observer) as prof:
            with record_function("Qwen/block.3/attention-replay"):
                out = model(q, k, v)
        observer.unregister_callback()
        prof.export_chrome_trace(str(kineto))
        trace = json.loads(kineto.read_text())
        trace.pop("traceName", None)
        native_et = json.loads(et.read_text())
    save_trace(HERE / "replay-operators.json.gz", trace)
    nodes = {a["value"]: n for n in native_et["nodes"] for a in n["attrs"] if a["name"] == "rf_id"}
    spans = [e for e in trace["traceEvents"] if e.get("cat") in {"cpu_op", "user_annotation"} and e["ph"] == "X"]
    spans.sort(key=lambda e: (e["ts"], -e["dur"]))
    root = spans[0]["ts"]
    rows, stack = [], []
    for event in spans:
        start = round(event["ts"]-root, 3)
        duration = event["dur"]
        while stack and rows[stack[-1]][1] + rows[stack[-1]][2] <= start + .001:
            stack.pop()
        rf = event["args"]["Record function id"]
        node = nodes[rf]
        if stack:
            parent = rows[stack[-1]]
            assert start + duration <= parent[1] + parent[2] + .002
            parent[4] = round(parent[4] - duration, 3)
        rows.append([event["name"], start, duration, len(stack), duration, tensors(node["inputs"]),
                     tensors(node["outputs"]), rf])
        stack.append(len(rows)-1)
    counters = [e for e in trace["traceEvents"] if e["name"] == "[memory]"]
    allocations = [[round(e["ts"]-root, 3), e["args"]["Bytes"]] for e in counters]
    actual = out.detach().numpy().reshape(24, 256)
    reference = expected.numpy().reshape(24, 256)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    with np.load(ASSETS / "layer3-decode.npz") as fixture:
        native = fixture["native"].reshape(24, 256)
    # Capture all 6144 output comparisons and BF16 baseline for an interactive head/feature probe.
    result = dict(rows=rows, host_tid=spans[0]["tid"], count=len(rows), trace_us=rows[0][2],
                  allocations=allocations, actual=actual.tolist(), expected=reference.tolist(), native=native.tolist(),
                  max_error=float(np.abs(actual-reference).max()), exact=int(np.equal(actual,reference).sum()),
                  native_error=float(np.abs(actual-native).max()), threads=torch.get_num_threads(), torch=torch.__version__)
    (HERE / "replay-data.js").write_text("window.profileData=window.profileData||{};window.profileData.replay="
                                         + json.dumps(result, separators=(",", ":")) + ";\n")
    print(f'{len(rows)} host ranges, {len(allocations)} memory events, {result["exact"]}/6144 equal, '
          f'maximum absolute error {result["max_error"]}')
    return result


if __name__ == "__main__":
    capture()
