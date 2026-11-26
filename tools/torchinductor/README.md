# TorchInductor

TorchInductor lowers PyTorch tensor graphs to executable CPU or GPU programs. In this trial its CPU debug artifacts connect a real Qwen expression to scheduler dependencies, allocated buffers, generated C++ and ATen library calls. [Tool guide](index.html) · [Interactive trial](../../topics/qwen-forward-pass/torchinductor/index.html) · [PyTorch 2.11 source](https://github.com/pytorch/pytorch/tree/v2.11.0/torch/_inductor) · [PyTorch CPU debugging guide](https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html)

## Install and verify

The tested Apple M4 Max environment used Python 3.13.12, torch 2.11.0, numpy 2.4.4, safetensors 0.8.0, Apple Clang and Homebrew libomp. Install a C++ compiler and OpenMP for your host. Use uv for the standalone package installation. From the repository root run:

```sh
./install.sh torchinductor
```

The installer creates a dedicated environment under `.local/tools/torchinductor/.venv`. If an existing HuggingArch environment already has exactly the three tested packages, the new environment imports those via a path file. Otherwise it installs the pinned packages. Two successive installer runs each compiled a CPU shape and compared the output to eager PyTorch.

## Run the trial

```sh
study=topics/qwen-forward-pass/torchinductor
python=.local/tools/torchinductor/.venv/bin/python
"$python" "$study/capture.py" attention
```

The attention replay needs only the published tensors. The MLP replay also needs the checkpoint index and its first pinned weight shard. [Run all three captures and rebuild the browser data](../../topics/qwen-forward-pass/torchinductor/README.md)
