"""Collect one FP32 replay as a paired PyTorch ET and Kineto CPU capture."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ExecutionTraceObserver, ProfilerActivity, profile, record_function

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
from workload import AttentionCore, SPEC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=HERE.parent / "assets/layer3-decode.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.fixture, allow_pickle=False) as fixture:
        q, k, v, expected = [torch.from_numpy(fixture[key].copy()) for key in ("q", "k", "v", "expected")]
    torch.set_num_threads(SPEC["torch_threads_replay"])
    core = AttentionCore().eval()
    host = args.output / "host.json"
    kineto = args.output / "kineto.json"
    with torch.no_grad():
        core(q, k, v)
        observer = ExecutionTraceObserver().register_callback(str(host))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, execution_trace_observer=observer) as trace:
            with record_function("ProfilerStep#0"), record_function("layer3.decode.attention_core"):
                result = core(q, k, v)
        trace.export_chrome_trace(str(kineto))
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
    metrics = dict(torch=torch.__version__, threads=torch.get_num_threads(),
                   max_abs_error=float((result-expected).abs().max()),
                   values_compared=result.numel())
    (args.output / "replay-check.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)


if __name__ == "__main__":
    main()
