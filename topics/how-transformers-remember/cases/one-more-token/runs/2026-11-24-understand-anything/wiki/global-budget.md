# Reconstruct 890 bytes per token

DeepSeek reports 890 bytes of global KV per token for V4.1‑Flash. That is about one quarter of V4‑Flash’s corresponding global footprint at equal context length.

The two ingredients are [[entry-precision]] (288 + 68 = 356 bytes per global entry) and [[shared-layers]] (3 × ½ + 1 = 2.5 entries per token). Multiply them: 356 × 2.5 = 890 bytes per token. At exactly one million tokens this term is 890,000,000 bytes, or 890 decimal MB (about 849 MiB). The local window, weights, allocator overhead, and temporary activations consume additional memory. The reported one-eighth figure concerns persistent storage in a different comparison.

Return to [[dense-budget]] to compare what each formula counts before using the numbers. [Technical report](https://arxiv.org/html/2609.19969v1) · [Release announcement](https://api-docs.deepseek.com/news/news260910/)
