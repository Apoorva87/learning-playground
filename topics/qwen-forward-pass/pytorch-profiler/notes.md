# PyTorch Profiler trial

[Open the interactive result](index.html) · [Tool guide](../../../tools/pytorch-profiler/index.html) · [Shared study](../index.html)

## Experiment

The BF16 full-model run covers all 64 decoder blocks of Qwen/Qwen3.8-27B at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. A 16-token unpadded text prompt is followed by one forced token using the actual cache. Both calls project the final hidden position to vocabulary scores. Hardware was an Apple M4 Max CPU with 128 GB unified memory, eight intra-op threads and one inter-op thread. PyTorch 2.11.0 and Transformers 5.24.0 ran in eager mode. A warm pair, module-clock pair and separate profiler pair used the same token IDs. [Capture script](../assets/capture.py) · [Manifest](../assets/workload.json)

## Measurements

| Quantity | Prefill | Cached decode |
| --- | ---: | ---: |
| Native CPU operator and annotation ranges | 6,051,615 | 5,930,735 |
| Ranges retained in the timeline | 3,461 | 2,885 |
| Native instrumented phase range | 84.432 s | 104.366 s |
| Module-clock elapsed time, separate run | 22.434 s | 19.139 s |
| Module-clock process CPU time | 55.334 s | 35.438 s |
| Inclusive time inside 48 grouped convolutions | 79.129 s | 76.374 s |
| Fraction of instrumented phase in grouped convolutions | 93.7% | 73.2% |
| Single-channel slow-convolution calls | 491,520 | 491,520 |

The reduced Chrome traces keep major operator families, their recorded ancestors and the first 21 host records within each grouped convolution. The streaming reader aggregates every native CPU operator and annotation before filtering. It verifies the complete event count, nested interval containment and the equality between summed self time and the outer phase's duration. The two timing runs quantify instrumentation overhead. One host lane captures operator nesting. Process CPU divided by wall time in the separate module-clock run averages 2.47 busy cores for prefill and 1.85 for decode. Worker schedules and DRAM traffic are unmeasured. [Raw module clocks](../assets/whole-pass.json) · [Process CPU sampling view](../index.html#cpu-occupancy)

The FP32 replay runs the shared `AttentionCore` on the fourth decoder block’s real Q/K/V tensors. Five warmups precede one measured CPU call with a single intra-op thread and a single inter-op thread. The observer, sampling scope and allocator recording influence durations. This capture has 466 CPU ranges and 15 memory events. Two `repeat_interleave` calls each allocate 417,792 bytes. Their original FP32 K and V tensors each hold 69,632 logical bytes. The profiled output and saved FP32 reference match at all 6,144 positions. Maximum absolute difference from the model’s BF16 result is 0.01483192295. Compare every head and feature in the replay graph. [Shared function](../assets/workload.py) · [Fixture](../assets/layer3-decode.npz)

## Reproduction

From the repository root, profile only the small attention fixture with the following commands.

```sh
./install.sh pytorch-profiler
.local/tools/pytorch-profiler/.venv/bin/python topics/qwen-forward-pass/pytorch-profiler/replay.py
```

For full-model profiling, follow the shared checkpoint setup and capture instructions first. Once the native `*-kineto.json` and `*-et.json` files exist locally, reduce them without loading the entire traces into Python objects.

```sh
study=topics/qwen-forward-pass
store=.local/qwen38
python3 "$study/assets/select_traces.py" "$store" prefill
python3 "$study/assets/select_traces.py" "$store" full
python3 "$study/pytorch-profiler/collect.py" "$store" prefill
python3 "$study/pytorch-profiler/collect.py" "$store" decode
```

Browser data bundles store event name, relative start (µs), inclusive duration (µs), stack depth, self duration (µs), input descriptors, output descriptors and record-function ID. Each descriptor includes shape, dtype and logical bytes. Kineto supplies timing and input shapes. The execution trace supplies input and output tensor metadata for retained major calls. Output shapes are unavailable for the supplemental channel samples. Native allocation events count allocator bytes. Tensor shapes quantify payload size. Neither value directly measures transferred bytes. [Whole-model capture instructions](../README.md) · [Profiler API](https://docs.pytorch.org/docs/2.11/profiler.html)

## Observations and feedback

The full-model profile makes the grouped-convolution fallback visible, while the smaller replay exposes materialization and true tensor dependencies. The native call intervals heavily amplify the cost of this particular instrumented CPU fallback.

Learner feedback on this tool is pending. Did the per-channel convolution drill-down or the replay allocation graph help you connect a PyTorch call to the work it caused?
