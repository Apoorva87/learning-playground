# Claude implementation: the whole-model dependency graph

[2D graph](index.html) · [3D WebGL coil](3d.html) · rebuild the data with `python3 -I build.py` (reads `../torchlens/text-topology.json.gz`, writes `data.js`).

## The question

Show one Qwen3.8-27B forward pass as a dependency graph over time. Node size follows compute, memory traffic or arithmetic intensity. Node colour follows the transformer stage. The picture should show where the work is parallel and where it is serial.

## What I found in the existing study

- The dependency pictures cover **7 nodes**: the attention core of block 3. Five tools each answer one question about that core. The whole-model evidence is a block-level timeline (64 bars) and a CPU occupancy plot.
- A whole-model graph already existed in the repo and was not used as one: `torchlens/text-topology.json.gz` holds 17,469 ops with parents, shapes and dtypes. Shape-derived FLOPs and bytes need nothing else.
- The timeline is measured on a CPU where 18.8 of 19.1 decode seconds are the reference recurrent kernels. That says more about those kernels than about the model, and it hides the model's structure.
- Colour distinguishes recurrent from full attention only. FLOPs are counted for linear layers only (the README says so), so there is no arithmetic intensity and no memory-bound or compute-bound classification.
- CPU occupancy cannot say whether two ops *could* overlap. The Chakra page itself concludes that its observed order and its dependencies differ.

## What I did differently

1. **Used the full graph, not a sample.** 4,999 drawn ops for decode, 6,042 for prefill, after removing metadata-only ops (drawn as edges) and weight-only ops (listed as left out).
2. **Cost every op from its shapes.** A linear layer is 2·tokens·K·N FLOPs and reads K·N weights. Everything else reads its inputs and writes its output once, unfused. Totals agree with the other agent's linear-only counts (807 vs 782 GFLOP prefill, 51.5 vs 51.2 decode) and with the checkpoint size (53.7 GB moved vs 55.6 GB of weights).
3. **Separate structure from hardware.** The layout (dependency step, lane) depends on the graph alone. A roofline machine model (vendor peaks, optional launch overhead) turns costs into modeled time and a critical path. No kernel quality enters.
4. **Colour by stage, size by a choice of FLOPs, bytes or FLOPs per byte**, log-scaled so a 2.5 GFLOP matmul and a 5 kFLOP elementwise op can share one canvas.
5. **Quantify serial vs parallel** with work and span: serial time, critical-path time, FLOPs on the critical path, ops per step.

## What the graph says (decode, H100 peaks)

- 4,999 ops in a chain **3,418 steps deep**. Never more than 5 ops are ready at once. 73% of steps hold exactly one op.
- **71% of all FLOPs sit on the critical path.** Graph-level parallelism is about 1.4×. The extra width is the gate/up pair of each MLP, the QKV projections and a few small branches.
- Every op is memory-bound, because a decode step has one token. Prefill with 16 tokens is also memory-bound (about 16 FLOP/B against a ridge of about 295). Compute-bound work needs hundreds of tokens in flight.
- The parallelism worth having is **inside** the big nodes, not between them. A column cannot show that, so node size carries it.

## Limits

- FLOPs and bytes are logical counts from shapes, not counters. Fusion and caching change real traffic. Elementwise ops count 1 FLOP per element (4 for exp, sigmoid, silu, rsqrt).
- Machine peaks are vendor numbers. The laptop preset's 16 TFLOP/s is an assumption.
- The topology comes from a meta-device structure run of the same architecture. Values were not loaded, so this is structure and cost, not measured behaviour.
- Operations that depend only on weights or constants are left out (476 in decode, 714 in prefill). They are not token-dependent work.
- The 3D coil is a layout choice: one turn per block, angle = step within the block. It does not change any dependency.
- Tested in headless Chrome (software GL): scripted click, hover, drag, zoom, play, phase, machine and stage-filter changes with no page errors. Not tested on a physical GPU or in Safari and Firefox.

## Feedback

Pending. Which view told you more about what can run at the same time: the 2D steps or the 3D coil?
