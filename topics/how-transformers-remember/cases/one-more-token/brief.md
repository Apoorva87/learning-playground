# One more token

## Learner’s original prompt

> Teach me things like what is a KV cash over the years how have KV cash enhancements help? How have we gone from a basic transformer like architecture KV cash to something like very advanced DeepSeek with 4.1 flashes KV cache, which is much smaller explain me the math explain with the general understanding and how it also working.

## Shared teaching task

Teach an English-language lesson called **How Transformers Remember**. Start with a single decoding step. Explain why it is safe to cache a prefix’s keys and values, how a new query uses them, and what work still scales with context length. Trace the ideas from multi-head attention to multi-query and grouped-query attention, DeepSeek’s latent attention, V4’s sequence compression, and V4.1-Flash’s cross-layer reuse and FP4 main cache. Use the numerical example below throughout. Finish with a prediction the learner can check.

Hold one sequence, 32 layers, 32 query heads, 128 numbers per head, and two bytes per number fixed. At 4,096 tokens compare 32, 8, and 1 KV heads. Predict the result at 8,192 tokens and explain why the number of query heads can stay fixed. Compare those *hypothetical* layouts with DeepSeek’s *reported* global cache figure, 890 bytes per token. Account separately for local attention state, weights, and serving overhead. Use [the source notes](../../assets/field-guide.md) for the history and the memory accounting.
