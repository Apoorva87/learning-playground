# Count a dense KV cache

One token in the example MHA model adds 512 KiB. At 4,096 tokens the cache holds 2 GiB for one sequence. A dense decoding step still reads keys and values from the history.

Multiply batch × tokens × layers × KV heads × head dimension × 2 arrays × bytes per number. Here one token costs 1 × 32 layers × 32 heads × 128 numbers × 2 arrays × 2 bytes = 524,288 bytes. At 8,192 tokens the total reaches 4 GiB. This counts the ideal KV tensors. Weights, temporary storage, and allocator overhead have separate budgets.

Try [[shared-heads]] to change just one factor. Use [[global-budget]] for an architecture with a different storage layout. [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)
