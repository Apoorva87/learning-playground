# From Python to a CPU kernel

Use the [interactive compiler walkthrough](index.html) to compare data dependencies, generated code and measured call order. [TorchInductor guide](../../../tools/torchinductor/index.html) · [Full-model investigation](../index.html) · [Shared workload and exact checkpoint revision](../assets/workload.json)

## November 25, 2026 trial

The 64-block Qwen3.8-27B forward pass produced these inputs. The attention case replays one FP32 grouped-query attention core using the shared tensors. The gated MLP cases run the complete BF16 gate, up and down projections of decoder block 3 for both the prompt and the cached token. The model's post-attention norm supplies the MLP inputs. The reference outputs precede residual addition. The MLP runner opens exactly three tensors in the first pinned shard using safetensors. The site distributes activation fixtures, compiler outputs and summary measurements.

| Workload | Warm median wall per call | Warm median process CPU | Max absolute error | Exact entries |
| --- | ---: | ---: | ---: | ---: |
| 1-token attention, 17 cached positions | 55.10 µs | 54.70 µs | 3.5763 × 10⁻⁷ | 1,689 / 6,144 |
| 1-token MLP | 15.17 ms | 15.12 ms | 0.000244140625 | 2,248 / 5,120 |
| 16-token MLP | 211.01 ms | 210.74 ms | 0.001953125 | 41,105 / 81,920 |

Each interval was collected with torch 2.11.0, Python 3.13.12, Apple M4 Max CPU, one PyTorch intra-op thread and one inter-op thread. We set the OpenMP, Accelerate, MKL and OpenBLAS thread limits to one. The warm benchmark used separate repeated calls with instrumentation off. The displayed timeline comes from a separate invocation with Python wrappers around emitted call sites. Those wrappers recorded monotonic timestamps around each original function and tensor shapes after it returned. The library and hardware can use vector units. No hardware worker trace or memory controller counters were collected.

The dependency diagram gives the buffer flow recorded by the compiler. The selected call’s byte labels give tensor payloads. For the attention copy loops, 24 × 17 × 256 FP32 elements are copied for each of K and V. The two 69,632-byte source tensors are each read six times at the logical element level, and each expanded destination spans 417,792 bytes. How many bytes cross the DRAM interface depends on the memory hierarchy. The source score and probability arrays each have 1,632 logical bytes. The wrapper delegates the calculation to CPU flash attention. It returns a 24,576-byte context and discards the 96-byte log-sum-exp auxiliary tensor.

The MLP matrices each span 178,257,920 bytes (170 MiB). Gate and up matrix multiplies both depend on the input and their own weights. They ran successively. The SiLU and product are one generated vector loop with a BF16 in-place output. It processed 17,408 elements for decode and 278,528 for prefill. The down projection then consumed that result. Each pre-fusion IR file is identical to its post-fusion counterpart in this captured run. Some transformations had already happened before the scheduler emitted those files. The emitted wrapper and C++ establish the actual code boundaries.

Eager execution matched every saved value in all three cases. Compiled attention passed the FP32 check with relative tolerance 10⁻⁵ and absolute tolerance 10⁻⁶. For the BF16 cases the chosen pointwise condition was |actual − expected| ≤ 0.003 + 0.016 × |expected|. All entries passed. The fused C++ keeps the SiLU intermediate in FP32 until after multiplication, whereas the eager implementation rounds SiLU to BF16 before the multiply. The run JSON records exact-match counts, root mean square differences, maxima and means for both eager and compiled outputs.

## Evidence

| Shape | TorchDynamo FX | Transformed FX | Pre-fusion IR | Post-fusion IR | CPU wrapper and C++ | Runtime and checks |
| --- | --- | --- | --- | --- | --- | --- |
| Attention | [FX](evidence/attention/dynamo_fx.txt) | [FX](evidence/attention/fx_graph_transformed.py) | [IR](evidence/attention/ir_pre_fusion.txt) | [IR](evidence/attention/ir_post_fusion.txt) | [Code](evidence/attention/output_code.py) | [JSON](evidence/attention/run.json) |
| MLP decode | [FX](evidence/mlp-decode/dynamo_fx.txt) | [FX](evidence/mlp-decode/fx_graph_transformed.py) | [IR](evidence/mlp-decode/ir_pre_fusion.txt) | [IR](evidence/mlp-decode/ir_post_fusion.txt) | [Code](evidence/mlp-decode/output_code.py) | [JSON](evidence/mlp-decode/run.json) |
| MLP prefill | [FX](evidence/mlp-prefill/dynamo_fx.txt) | [FX](evidence/mlp-prefill/fx_graph_transformed.py) | [IR](evidence/mlp-prefill/ir_pre_fusion.txt) | [IR](evidence/mlp-prefill/ir_post_fusion.txt) | [Code](evidence/mlp-prefill/output_code.py) | [JSON](evidence/mlp-prefill/run.json) |

## Reproduce

Install TorchInductor from the repository root. Each capture creates a fresh compilation cache below the ignored local tool folder. On a different PyTorch version or CPU the compiler may choose a different lowering. The call instrumentation and browser annotations target the tested 2.11 CPU lowering.

```sh
./install.sh torchinductor
study=topics/qwen-forward-pass/torchinductor
python=.local/tools/torchinductor/.venv/bin/python
"$python" "$study/capture.py" attention
```

For the MLP, put the pinned checkpoint’s index and shard 1 in `.local/qwen38/checkpoint`, or point the runner to the directory with `--checkpoint`. The full-model capture verified all 18 SHA-256 digests; the two files below suffice to rerun this MLP trial. [Pinned weight index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/model.safetensors.index.json) · [Pinned first shard](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/model-00001-of-00018.safetensors) · [Reference model capture](../assets/capture.py)

```sh
"$python" "$study/capture.py" mlp-decode
"$python" "$study/capture.py" mlp-prefill
"$python" "$study/build.py"
```

[Capture and backend hook](capture.py) · [Browser data builder](build.py) · [Attention source](../assets/workload.py) · [PyTorch compiler API](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)

## Learner feedback

Pending. Which view helped you decide whether the two gate and up projections actually ran at the same time?
