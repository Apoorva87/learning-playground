# MLCommons Chakra on Qwen text inference

[Open the study](index.html) · [Tool guide](../../../tools/chakra/index.html) · [Common workload](../assets/workload.json)

## Run and provenance

The host ran the pinned BF16 Qwen3.8-27B checkpoint on an Apple M4 Max CPU. The whole-model comparison is a single text sequence of 16 prefill tokens, followed by one fixed cached decode token. The fourth block’s attention replay uses the full model’s saved Q, K and V promoted to FP32. The replay produced 6,144 output values with a maximum absolute difference of zero against the shared expected tensor. The full pass used eight intra-op threads. The replay used one.

The trial used PyTorch 2.11.0, MLCommons’ `mlc-chakra` 1.0.0 at `065445b5d48ffb8796c2dbf4f0540cc608551f8d`, and the host schema `1.1.1-chakra.0.0.4`. The [original source](https://github.com/mlcommons/chakra/tree/065445b5d48ffb8796c2dbf4f0540cc608551f8d) documents the standard converter and schema.

## Agent observations

The replay’s native ET held 469 nodes. Exactly 467 matched Kineto CPU intervals by record-function ID and operation name. The other two describe the process and thread. The upstream linker's HTA critical-path stage reached a kernel queue-length assertion on this CPU-only recording, even with a real `ProfilerStep#0` region. The supported host-only conversion then produced 468 nodes and 467 scheduling predecessors. Its encoded durations are zero because its input lacks joined timing. The bytes of this standard conversion are available as [native-replay.et](native-replay.et).

The teaching export retains a region and its seven direct children. A child’s control predecessor identifies the region. The children’s six data edges match an output tensor ID to a later sibling input. The published ET attaches measured start offsets and inclusive durations from a separately checked, one-to-one join to Kineto and rounds to microseconds. The release's protobuf reader and jsonizer both decoded the export. [Projected binary](attention.et) · [Decoded messages](attention.json) · [Build and analysis](analyze.py)

For example, the final value mix receives values from ET 23 and probabilities from ET 318. The transpose at ET 41 has different input and output tensor IDs with the same backing storage. The score multiply consumes the transpose’s output tensor ID. In this recording the key and value expansions occupy disjoint intervals on one host track despite their independent tensor inputs. This evidence concerns the PyTorch operator track. Worker-level CPU concurrency requires another instrument, while the aggregate process CPU / elapsed ratio estimates average CPU cores across a separate hook-timed pass.

The complete prefill ET contained 6,051,617 nodes. The selection kept 2,551 and matched 2,549 Kineto intervals. The complete decode ET contained 5,930,737 nodes. The selection kept 1,975 and matched 1,973 intervals. Both sets include all 64 decoder blocks.

The full-model block selector uses the bounded trace selection shared with the profiler study. It retains important operator families and their ancestor scopes from both complete recordings. Reported descendant counts and grouped inclusive times apply to this selection. The original observer truncated literal lists longer than 4,096 entries. Every operator from the much smaller FP32 replay remains available in the [replay ET](replay-host.json.gz) and [replay Kineto file](replay-kineto.json.gz).

The graph walk, tensor-byte arithmetic, real-time comparison and interactive block selector were added for this study. Hardware simulation, distributed communication and GPU kernel tracing were outside this run.

## Reproduce

From the repository root, install Chakra using `./install.sh chakra`. The whole-model [capture](../assets/capture.py) produces the pinned fixture and paired native traces. Generate compact phase files with the [selection script](../assets/select_traces.py), then run the [study script](run.sh).

```sh
python3 topics/qwen-forward-pass/assets/select_traces.py .local/qwen38 prefill
python3 topics/qwen-forward-pass/assets/select_traces.py .local/qwen38 full
topics/qwen-forward-pass/chakra/run.sh
```

`TORCH_PYTHON` can point to another PyTorch 2.11.0 interpreter. `FULL_TRACE_DIR` selects the compact trace directory. Rerunning retimes the workload and creates fresh observer IDs. The exported files and CPython script are sufficient to inspect the original run without a model download.

## Learner feedback

Pending. While checking whether the two expansions could run concurrently, which evidence helped you decide what happened in this recording?
