# DaCe

[Tool guide](index.html) · [Qwen attention explorer](../../topics/qwen-forward-pass/dace/index.html) · [Trial record](../../topics/qwen-forward-pass/dace/README.md)

DaCe provides the Stateful Dataflow Multigraph (SDFG), a Python graph API, native backends, instrumentation and graph transformations. The tested release is [DaCe 1.0.2](https://pypi.org/project/dace/1.0.2/) from [SPCL](https://github.com/spcl/dace). The trial compiled with Apple Clang on Apple Silicon using Python 3.11.14, NumPy 1.26.4, CMake 3.31.10 and Homebrew libomp.

## Install and verify

From the repository root, run the commands below. You need uv, Python 3.11, a C++14-capable compiler, and an OpenMP runtime for that compiler. On a Mac with Xcode command line tools installed, `brew install libomp` provides the runtime. On Linux, an OpenMP-enabled GCC toolchain is suitable. [DaCe installation instructions](https://spcldace.readthedocs.io/en/stable/setup/installation.html)

```sh
./install.sh dace
.local/tools/dace/.venv/bin/sdfgcc --help
.local/tools/dace/.venv/bin/python topics/qwen-forward-pass/dace/replay.py
```

The [recipe](install.sh) creates an isolated environment and cache under `.local/tools/dace/`. It reuses an existing environment on subsequent runs. Set `DACE_PYTHON` to a local Python 3.11 interpreter if needed. When PyPI is inaccessible to uv, `DACE_WHEELHOUSE` can point to an absolute directory containing the upstream packages and their dependencies. This trial used that route with packages retrieved from PyPI. The published case already includes its small checkpoint activation fixture.

The benchmark builds and validates a graph in Python, saves the native SDFG, then reloads and compiles the saved file. It executes the six states on actual model activations and compares all 6,144 outputs with the shared FP32 reference. The browser lesson opens directly from its relative files.
