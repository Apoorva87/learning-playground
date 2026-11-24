# How Transformers Remember

## Predict one more token

Suppose the prompt says “The spare key is in the blue drawer. Open the …”. During **prefill**, a causal transformer processes the prompt and, at each attention layer, saves a key and a value for every position. A key describes what might make that position relevant. A value carries what the position can contribute if selected. Think of an entry in an index and the passage the entry points to. In the model, both are learned vectors, so a position can match a question by degrees.

During **decode**, the newest position makes a fresh query. It scores the cached keys, turns those scores into weights, and blends the corresponding values. The model also computes and appends the new position’s key and value. Why is reuse valid? Causal attention prevents a later word from changing the hidden state at an earlier position. With the same prefix, weights, positions and inference settings, those earlier projections can be reused. The newest query still reads the history in dense attention, so adding a token to a long prompt involves more memory traffic than adding one to a short prompt. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)

## Do one attention calculation by hand

For a single head, let q be the new query, kᵢ and vᵢ the key and value at position i, and d the key dimension. The score is sᵢ = q·kᵢ / √d. Convert scores to weights with aᵢ = exp(sᵢ) / Σⱼ exp(sⱼ). The output is Σᵢ aᵢvᵢ. Later layers use that result when predicting a token.

Try an attention row with two eligible positions. Set d = 1, q = 1, keys to 0 and ln(3), and values to 2 and 10. Exponentiating the two scores gives 1 and 3, so the weights are ¼ and ¾. The output is ¼ × 2 + ¾ × 10 = **8**. A different query changes those weights even when it reads the exact same cache. For q = 0, both weights become ½ and the output becomes **6**. A real decoder also admits its current position to the attention row. [Transformer attention, §3.2](https://arxiv.org/abs/1706.03762)

## Count the bytes

For dense attention with equal key and value dimensions, ideal tensor storage is

**bytes = batch × tokens × layers × KV heads × head dimension × 2 arrays × bytes per element.**

With one sequence, 32 layers, 32 KV heads, head dimension 128, and 2-byte elements, one new token adds 32 × 32 × 128 × 2 × 2 = **524,288 bytes**, or 512 KiB. At 4,096 tokens that is **2 GiB**. This toy model lets us change a single variable at a time.

| Design, holding 32 query heads fixed | KV heads | At 4,096 tokens | At 8,192 tokens |
| --- | ---: | ---: | ---: |
| Multi-head attention (MHA) | 32 | 2 GiB | 4 GiB |
| Grouped-query attention (GQA) | 8 | 512 MiB | 1 GiB |
| Multi-query attention (MQA) | 1 | 64 MiB | 128 MiB |

**Try predicting before reading on.** Double the prompt length and halve the KV head count. What happens to storage? Those factors cancel. The number of query heads can stay the same because several queries can use the same bank of keys and values. Each query still computes its own scores. These are architectures learned during training or adaptation. Changing the head count of a trained MHA model is a model change. [Multi-query attention, 2019](https://arxiv.org/abs/1911.02150) · [Grouped-query attention, 2023](https://arxiv.org/abs/2305.13245)

## The route from separate heads to shared history

**2017–2023 · Share across heads.** The original Transformer assigns different projections to attention heads. MQA gives all query heads one set of keys and values. GQA splits the query heads into groups, providing one KV set per group. This reduces stored bytes and the amount of KV data that must move during decode. Query heads can still attend to different positions. [Transformer](https://arxiv.org/abs/1706.03762) · [MQA](https://arxiv.org/abs/1911.02150) · [GQA](https://arxiv.org/abs/2305.13245)

**2024 · Store a learned latent.** DeepSeek‑V2’s multi-head latent attention (MLA) compresses the hidden state h into a compact vector c = W↓h. Heads learn different projections of c. In the content part of an attention score, qᵀ(W↑c) = (W↑ᵀq)ᵀc. That identity lets an implementation score cached latents directly. A small positional key is cached separately for rotary position information. A latent costs fewer numbers per position while still supporting different heads. [DeepSeek‑V2, §2.1](https://arxiv.org/html/2405.04434v5)

**2026 · Compress along the sequence.** V4‑Flash keeps a detailed recent window and represents older context with learned summary entries. Compressed Sparse Attention (CSA) forms summaries at a finer stride and chooses relevant ones for a query. Heavily Compressed Attention (HCA) forms much coarser summaries and can attend to all of them. Two costs are at play. Packing more tokens into each entry reduces how much history is stored. Selecting fewer stored entries reduces the work of reading them for the next token. [DeepSeek‑V4 model card and technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

**2026 · Share across layers and shrink each entry.** DeepSeek‑V4.1‑Flash uses CSA2. A Full layer writes a global cache and selects positions. A Reindex layer reads that cache and picks positions for its own query. A Reuse layer reads the cache and a preceding layer’s selection. Each layer still computes its own attention query and keeps its local 128-token window. The encoder’s global entries cover pairs of positions. The decoder’s cover single positions, with many layers reading the same store. [DeepSeek‑V4.1‑Flash, §§2.3 and 4.2.1](https://arxiv.org/html/2609.19969v1)

**Follow the rounding.** The main global cache uses 4-bit numbers with a one-byte scale per group of 16. Sixteen such values cost 8 + 1 = **9 bytes**. Storing 16 BF16 values takes 32 bytes. That small example explains one ingredient. The model’s overall cache also includes the indexer and benefits from sharing entries across layers. Its local sliding-window cache uses FP8. [DeepSeek‑V4.1‑Flash, §2.4.4](https://arxiv.org/html/2609.19969v1)

## What fits in 890 MB?

Here is one way to reconstruct the reported global total from the released configuration. A 512-channel main entry takes 512/2 bytes of FP4 payload plus 512/16 scale bytes, or **288 bytes**. Its 128-channel index key takes 128/2 payload bytes plus 128/32 scale bytes, or **68 bytes**. Together they cost **356 bytes per entry**. Three encoder cache writers each create one entry per two tokens. One decoder writer creates one per token. Over a long sequence, the average is **(3 × ½ + 1) × 356 = 890 bytes per token** across the network. The other layers can use those entries again. [Model configuration and inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py) · [Released architecture and dimensions, §4.2.1](https://arxiv.org/html/2609.19969v1)

DeepSeek reports **890 bytes of global KV cache per token**, about one quarter of V4‑Flash’s corresponding footprint. At exactly 1,000,000 tokens that is **890 MB**, about **849 MiB**. Weights, the bounded local window, temporary activations, and allocator overhead require additional memory. For persistent storage on SSD or host memory, DeepSeek reports about one eighth of its predecessor’s footprint, aided by reconstructing missing sliding-window state when a session resumes. These ratios have different denominators. [DeepSeek technical report, §§2–3](https://arxiv.org/html/2609.19969v1) · [Release announcement](https://api-docs.deepseek.com/news/news260910/)

**Explain it back.** Two designs both read 512 global entries per query. One stores an entry for every old token and one stores an entry per group of four. Would their long-context caches occupy the same memory? With equal entry sizes and all else fixed, the second stores about a quarter as many entries. The read limit alone tells you little about the total inventory.
