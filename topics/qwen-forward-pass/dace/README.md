# DaCe · One token through six states

[Interactive attention explorer](index.html) · [Full-model investigation](../index.html) · [Tool guide and setup](../../../tools/dace/index.html)

## Run of November 25, 2026

The shared task follows one 16-token prefill and one fixed decode token through Qwen/Qwen3.8-27B at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. This program replays the grouped-query attention core of zero-based block 3 on activations captured during that decode. The model’s BF16 query has passed normalization and rotary embedding. K and V include the newest token. The comparison casts these values to FP32. [Shared workload](../assets/workload.json) · [Checkpoint activation file](../assets/layer3-decode.npz) · [Full-pass record](../assets/whole-pass.json)

The 64-block hybrid decoder has 48 recurrent mixers with convolution and recurrent state, 16 full-attention mixers with growing KV caches, and a gated MLP in every block. The shared full-model run exercises them all. This SDFG implements scaled QK, stable softmax and value mixing, ending before the attention output gate. It uses compact KV arrays indexed by `h // 6`. The shared PyTorch reference repeats the cached values across six query heads. Each implementation produces the same output shape. [Model configuration](../assets/config.json) · [Common FP32 reference](../assets/workload.py)

## Reproduce

From the repository root, run the two commands below. The [setup record](../../../tools/dace/README.md) lists prerequisites. Running the Python command rebuilds the published copies of the graph, generated code, native timer trace and numerical results. Pass `--output /path/to/results` for a separate copy.

```sh
./install.sh dace
.local/tools/dace/.venv/bin/python topics/qwen-forward-pass/dace/replay.py
```

The [Python generator](replay.py) constructs the six SDFG states with the public DaCe API. C++ tasklets specify the arithmetic inside each map. It saves the native graph, reloads and validates it, then invokes DaCe’s CPU backend. Five intermediate buffers are preallocated and exposed as graph arguments so each state’s outputs remain inspectable. In-benchmark calls use the compiled graph through its ordinary Python interface. The run strips DaCe’s default fast-math flag. On this host, Apple Clang 17 needed a compatibility diagnostic disabled for the 1.0.2 runtime header. The source sets that flag on macOS and points CMake at Homebrew libomp.

- [Compiled graph input](decode.sdfg) and [DaCe-generated CPU implementation](decode.cpp)
- [Native per-state timer trace](state-trace.json) and [benchmark, tensor sizes and saved activations](results.json)
- [Browser controls](explorer.js), [styling](explorer.css) and [offline copy of the results](results.js)

## Measurements

Python 3.11.14, DaCe 1.0.2 and NumPy 1.26.4 ran on an Apple M4 Max CPU. Batch size is one. The new query has 24 heads of width 256. The cache has 4 heads and 17 positions. The 6,144 output elements agree with the shared FP32 reference within `rtol=1e-4, atol=1e-5`. The maximum absolute difference is **3.69549 × 10⁻⁶** and the RMS difference is **2.57713 × 10⁻⁷**. The fixture file SHA-256 is `be8abdcda8cdd3b43d8269083087c5893017b736fe9c37d5d3566e4d277ad8d7`.

After correctness validation and 15 warmups, the median over 151 individual Python-to-native calls was **85.417 µs elapsed**. The 10th and 90th percentiles were 83.625 and 88.542 µs. Median process CPU time was 86 µs. These times include Python argument validation and dispatch. Buffer allocation, compilation, loading the fixture, the correctness test and tracing happen outside this timing loop.

| State | Independent map iterations | Read bytes | Written bytes | Instrumented example, µs |
| --- | ---: | ---: | ---: | ---: |
| Scaled dot product | 408 | 835,584 | 1,632 | 34 |
| Row maximum | 24 | 1,632 | 96 | 1 |
| Shift and exponentiate | 408 | 3,264 | 1,632 | 1 |
| Row sum | 24 | 1,632 | 96 | 0 |
| Normalize | 408 | 3,264 | 1,632 | 0 |
| Value mix | 6,144 | 835,584 | 24,576 | 7 |
| **Logical payload total** | | **1,680,960** | **29,664** | |

The timer column comes from one separate native invocation instrumented with DaCe’s state Timer. Its timestamps have 1 µs resolution, so a zero can mean both timestamp samples fell in the same microsecond. The uninstrumented benchmark above includes a different scope. All six states are ordered and all maps use a sequential schedule with `OMP_NUM_THREADS=1`. Independent map indices identify legal parallel work. Each tasklet reduces its inner dimension with a local accumulator where a reduction is required.

The byte counts sum the inner read/write memlet payloads across iterations, one count per logical FP32 access in the tasklet. One physical K buffer occupies 69,632 bytes, and the score tasklets request 417,792 key bytes because six queries read each key. Summing all distinct allocated buffers, including Q, K, V, every intermediate and the output, gives 193,504 bytes. Hardware cache reuse and compiler transformations affect actual movement. No DRAM traffic counters were collected.

## Agent observations

Selecting query head 6 and token 16 in **Value mix** follows a concrete read. That head uses KV head 1. The probability at position 16 is approximately 0.0241977 and the cached value for feature zero is −0.0869141. Their product is one of 17 signed contributions. The compiled final output for that feature is approximately 0.0813723. The native score and mixing tasklets have equal logical read counts and different durations in the separate trace.

## Learner feedback

Pending. Did following one concrete array slice through its map explain the memory reuse more clearly than the operation traces in the other tools?
