# TorchLens on the Qwen3.8-27B forward pass

[Interactive result](index.html) · [Tool guide](../../../tools/torchlens/index.html) · [Shared workload](../assets/workload.json)

## Run

TorchLens 3.0.3, torch 2.11.0, Transformers 5.24.0. One 16-token text prefill, then one cached teacher-forced token. Model revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. The numerical attention capture uses the actual block 3 Q/K/V file with SHA-256 `be8abdcda8cdd3b43d8269083087c5893017b736fe9c37d5d3566e4d277ad8d7`. The FP32 replay used one intra-op thread and one inter-op thread. All 6,144 outputs exactly matched the FP32 shared reference. The largest absolute softmax row-sum error was 1.2 × 10⁻⁷. [Input fixture](../assets/layer3-decode.npz) · [Replay script](replay.py) · [Recorded output](trace.json)

The independent structure run instantiated the exact conditional-generation model class on the meta device, using the same eager attention setting, text shapes, cache update and final-token logits projection. It recorded 17,464 tensor operations and 20,996 dependency edges. The full 17,469 records include input/output and buffer records. Both output shapes were [1, 1, 248320]. No checkpoint weight values were loaded in the structure run. The built-in PyTorch reference implementations handled the linear-attention layers. [Structure script](structure.py) · [Per-block records](structure.json) · [Full graph](text-topology.json.gz)

## Observations

- Seven Python compute records in the replay own 23 ATen dispatch records. Source lines connect both native matrix multiplications to their batched `aten::bmm` operations. The synthetic input/output graph records sit outside that compute count.
- Each repeat expands a 68 KiB cached tensor into 408 KiB. The transpose has 408 KiB of logical output and a zero payload read/write estimate. The shape-derived numbers cannot establish physical traffic. TorchLens call durations were taken under instrumentation.
- The two expansions can proceed independently at the graph level. The value mix directly depends on both the softmax weights and expanded values. Each matrix multiplication requires 104,448 multiply-accumulates.
- The captured Transformers class inventory is 27,356,728,560 parameters. The checkpoint contains another 424,699,392 BF16 parameters in the multi-token-prediction branch. Its 849,398,784 stored bytes account for the difference from the checkpoint’s 27,781,427,952 parameters. [Shard inventory](../assets/weights.json)
- The schema-preserving SVG is a native TorchLens export. The native HTML report used TorchLens’s fallback schematic because the Graphviz system executable was unavailable. The teaching page draws its own layout over actual TorchLens dependency labels and includes the full meta topology for offline navigation. [Native SVG](torchlens.svg) · [Native HTML report](native-report.html)
- Qwen’s output gate, projection, residual, MLP, other layers and vocabulary head fall outside the seven-call numerical replay. The structure browser covers their text-path geometry. The shared whole-model evidence records the measured BF16 run. [Whole-model data](../assets/whole-pass.json)

## Learner feedback

Pending. Which lookup helped you explain the 408 KiB transpose and the two branches that meet at the value mix?
