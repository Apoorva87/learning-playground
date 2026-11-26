# Inside one Qwen forward pass

[Open the interactive investigation](index.html) or [compare the five visual results](compare/index.html). Compare five ways to study the same text-only forward pass of [Qwen3.8-27B at the pinned revision](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0).

[Download the five diagrams as one image](assets/five-views.png).

## Shared experiment

The [input manifest](assets/workload.json) fixes all 18 checkpoint shard hashes, 16 prompt token IDs, the teacher-forced next token, and the shapes for the common replay. The input is one text sequence without chat formatting, images or padding. Predicting the next-token distribution after the 16-token prefill and after one cached decode exercises all 64 decoder blocks and the vocabulary projection.

For a like-for-like numerical comparison across frameworks, replay the attention core from decoder block 3 (the fourth block). Capture Q after normalization and rotary position encoding, K and V after the cache update, before the output gate. There are 24 query heads, four KV heads, 17 cached positions including the new token and 256 features per head. The BF16 model supplies the values. All tool replays use the same exported tensors promoted to FP32 and compare against the same expected result. The full model and its replay have different scopes and precisions.

The [shared Python workload](assets/workload.py) defines the replay. Measured host elapsed time, process CPU time, tensor payload sizes, static read/write counts and hardware traffic counters answer different questions. Report the quantity and its unit beside every figure. Dependency order alone establishes no hardware overlap.

## Two observations from the measured run

The 16-token prefill took 22.43 s on the M4 Max CPU. The single cached decode took 19.14 s. The 48 recurrent mixers occupied 18.83 s of that decode using the available reference convolution and recurrent implementations. The full-attention mixers together took 0.047 s. The 16-token prefill used 781.8 GFLOPs in its linear projections versus 51.2 GFLOPs in decode. Elapsed time also reflects scheduling, memory access and the reference kernels.

The measured cache tensor payload grew by exactly 65,536 bytes. Each of 16 full-attention blocks appended one BF16 key and value across four KV heads with 256 features. That adds 4,096 bytes per such block. The shapes of the 48 recurrent states stayed fixed.

The [CPU occupancy plot](index.html#cpu-occupancy) uses a separate warmed run. A sampler accumulated CPU time across process threads and subtracted its own CPU time. In a 54.66 ms prefill window, the runtime used an average of 7.41 cores. Across the full sampled decode the average was 1.86, with a peak of 6.70 during its final 19.68 ms window. The timing resolution identifies intervals containing concurrent work. It cannot identify a particular worker or assign its instructions to one operator. [Raw samples and block entries](assets/cpu-windows.json)

## Reproduce the shared inputs

The capture ran on an Apple M4 Max with 128 GB of unified memory, PyTorch 2.11.0 and Transformers 5.24.0. Allow about 56 GB of disk for the checkpoint and additional room for traces. The published attention replay needs only its small fixture. From the playground root, using the [capture environment](assets/requirements.txt) in Python 3.13, run:

```sh
study=topics/qwen-forward-pass/assets
store=.local/qwen38
python "$study/get_checkpoint.py" "$store/checkpoint"
python "$study/inspect_weights.py" "$store/checkpoint"
HF_HUB_OFFLINE=1 python "$study/capture.py" "$store/checkpoint" "$store"
HF_HUB_OFFLINE=1 python "$study/sample_cpu.py" "$store/checkpoint"
python "$study/render_data.py"
```

The downloader requests the pinned Hub revision. The capture verifies each shard against the manifest. The initial no-gradient forward/decode pair warms the process and exports the shared fixture. A second pair records monotonic elapsed time and process CPU time in module hooks. A third collects separate native PyTorch operator and execution traces for the two phases. Within each pair the second call receives the first call’s actual cache. Each measured pass requests only the final-position vocabulary scores. The warm and measured passes check that the top token IDs agree.

The raw native captures recorded over six million prefill nodes and nearly six million decode nodes, including the reference convolution’s per-channel operations. To build a smaller view, run `python "$study/select_traces.py" "$store" prefill` and repeat with `full` for decode. It retains selected operator families and all their enclosing scopes, then joins timings by the original record-function ID. The source records and the retention counts allow you to change or audit the filter.

The [checkpoint tensor inventory](assets/weights.json) accounts for every one of the 1,199 BF16 tensors. The normal text path uses the decoder, token embeddings, final norm and separate output head. Vision and multi-token prediction have their own checkpoint weights. The embedding stage looks up the specified rows. Counting the entire embedding table as traffic would overestimate those reads. The [Apache 2.0 license](assets/checkpoint-LICENSE.txt) accompanies the checkpoint metadata. We keep the full weights and raw machine traces under the ignored local directory.

## Feedback

Pending. Which view helped you determine whether an intermediate must be written to memory or two branches can run at the same time?
