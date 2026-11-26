# PyTorch Profiler

[Tool guide](index.html) · [Interactive Qwen trial](../../topics/qwen-forward-pass/pytorch-profiler/index.html) · [Trial notes](../../topics/qwen-forward-pass/pytorch-profiler/notes.md)

PyTorch 2.11.0 supplies `torch.profiler`, its Kineto-backed CPU capture, execution-trace observer and Chrome-trace exporter. The Qwen replay used NumPy 2.4.4, Python 3.13 and one CPU thread. The full-checkpoint capture used eight CPU threads, BF16 and Transformers 5.24.0. [Upstream API](https://docs.pytorch.org/docs/2.11/profiler.html)

## Install and verify

Run from the workspace root. The recipe uses an existing matching playground Python environment when available. A fresh environment needs uv and access to the pinned packages. The install verifies that the normal PyTorch profiler captures a dot product. The replay uses the checkpoint-derived tensor fixture shipped with this workspace.

```sh
./install.sh pytorch-profiler
.local/tools/pytorch-profiler/.venv/bin/python topics/qwen-forward-pass/pytorch-profiler/replay.py
```

Open the saved trial directly in a browser. To inspect a native trace outside the study, download its gzip file, decompress it, and open the resulting JSON in a Chrome-trace-compatible viewer such as Perfetto. [Perfetto UI](https://ui.perfetto.dev/)
