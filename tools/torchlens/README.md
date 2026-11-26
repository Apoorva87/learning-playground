# TorchLens

Trace tensor-producing operations in an eager PyTorch forward pass, inspect their dependencies and saved values, and list ATen operations inside each Python call. [Illustrated guide](index.html) · [Qwen investigation](../../topics/qwen-forward-pass/torchlens/index.html) · [Upstream repository](https://github.com/johnmarktaylor91/torchlens) · [TorchLens 3.0.3](https://pypi.org/project/torchlens/3.0.3/)

## Installation

The tested environment used Python 3.13.12, TorchLens 3.0.3, torch 2.11.0 and transformers 5.24.0. From the repository root run the installer with uv available. It creates a dedicated environment in the ignored tool directory and runs a captured ReLU through the normal Python API. Two consecutive installer runs succeeded.

```sh
./install.sh torchlens
```

The installer can reuse the existing HuggingArch environment’s Python packages through a read-only import path in the TorchLens environment. On a fresh workspace it installs pinned PyTorch and Transformers in the new environment. The TorchLens dependencies particular to this installation include graphviz 0.21 and Pillow 13.0.0. An optional local `.local/tools/torchlens/wheels/` directory can hold wheels for these two packages and TorchLens when reusing the compatible base offline. The optional Graphviz system executable enables the native HTML graph layout. Without it the upstream report supplies a labeled schematic.

## Reproduce the study

```sh
python=.local/tools/torchlens/.venv/bin/python
study=topics/qwen-forward-pass/torchlens
"$python" "$study/structure.py"
"$python" "$study/replay.py"
```

The structure run uses the checked-in model configuration. The numerical run reads the checked-in NPZ containing a real Qwen layer’s Q, K and V, and its FP32 reference output. The scripts write outputs in the trial directory and refresh the study’s embedded records. Repeating a capture can change instrumented timings. [Run notes and evidence](../../topics/qwen-forward-pass/torchlens/notes.md)
