# How Transformers Remember

## One Decoding Step by Hand

Imagine you're at position 5 of a sequence. The prompt so far is:

> "The spare key is in the blue drawer. Open the …"

During **prefill**, the model processed positions 1–4 and saved their keys (K) and values (V) in cache. At each attention layer, this looks like:

| Position | Key $k_i$ | Value $v_i$ |
|----------|-----------|-------------|
| 1 | learned vector | learned vector |
| 2 | learned vector | learned vector |
| 3 | learned vector | learned vector |
| 4 | learned vector | learned vector |

Now at **decode**, position 5 arrives. It computes a fresh query $q_5$ and scores against the cached keys:

$$s_i = \frac{q_5 \cdot k_i}{\sqrt{d}}$$

Then softmax to get attention weights:

$$a_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$

Finally, blend the values:

$$\text{output}_5 = \sum_{i=1}^4 a_i v_i$$

**Why is this safe?** Causal attention means position 5 can only attend to positions 1–5 (including itself). Positions 1–4 never change once written. Their keys and values are projections of fixed hidden states, so reusing them gives identical results as if you recomputed everything. The new query still reads history in dense attention, but the *computation* of earlier K/V pairs is saved.

---

## Do One Attention Calculation by Hand

Let's work through a single head with $d=1$:

| Position | Key $k_i$ | Value $v_i$ |
|----------|-----------|-------------|
| 1 | $0$ | $2$ |
| 2 | $\ln(3)$ | $10$ |

**Query A:** $q = 1$

$$s_1 = \frac{1 \cdot 0}{\sqrt{1}} = 0, \quad s_2 = \frac{1 \cdot \ln(3)}{\sqrt{1}} = \ln(3)$$

Exponentiate: $\exp(0)=1$, $\exp(\ln(3))=3$. Weights: $a_1=\frac{1}{4}$, $a_2=\frac{3}{4}$.

Output: $\frac{1}{4}\times 2 + \frac{3}{4}\times 10 = \mathbf{8}$

**Query B:** $q = 0$

$$s_1 = s_2 = 0 \implies a_1=a_2=\frac{1}{2}$$

Output: $\frac{1}{2}\times 2 + \frac{1}{2}\times 10 = \mathbf{6}$

The same cache, different queries produce different outputs. The cache stores *what* to attend to; the query decides *how*.

---

## Counting Bytes: The Memory Cost

For dense attention with equal key and value dimensions:

$$\text{bytes} = \text{batch} \times \text{tokens} \times \text{layers} \times \text{KV heads} \times \text{head dim} \times 2 \times \text{bytes per element}$$

**Our toy model:**
- Batch = 1 (one sequence)
- Layers = 32
- Query heads = 32 (fixed)
- Head dimension = 128 numbers
- Bytes per number = 2 (BF16 or FP16)

One new token adds: $32 \times 32 \times 128 \times 2 \times 2 = \mathbf{524,288\ bytes}$ = **512 KiB**

| Design | KV heads | At 4,096 tokens | At 8,192 tokens |
|--------|----------|-----------------|-----------------|
| Multi-head attention (MHA) | 32 | 2 GiB | 4 GiB |
| Grouped-query attention (GQA) | 8 | 512 MiB | 1 GiB |
| Multi-query attention (MQA) | 1 | 64 MiB | 128 MiB |

**Try predicting before reading on:** Double the prompt length and halve the KV head count. What happens to storage? Those factors cancel. The number of query heads can stay the same because several queries can use the same bank of keys and values. Each query still computes its own scores. These are architectures learned during training or adaptation. Changing the head count of a trained MHA model is a model change.

---

## The Route from Separate Heads to Shared History

### 2017–2023: Share Across Heads
The original Transformer assigns different projections to each attention head. **MQA** gives all query heads one set of keys and values. **GQA** splits the query heads into groups, providing one KV set per group. This reduces stored bytes and the amount of KV data that must move during decode. Query heads can still attend to different positions.

### 2024: Store a Learned Latent
**DeepSeek‑V2's multi-head latent attention (MLA)** compresses the hidden state $h$ into a compact vector $c = W_\downarrow h$. Heads learn different projections of $c$. In the content part of an attention score, $q^\top(W_\uparrow c) = (W_\uparrow^\top q)^\top c$. That identity lets an implementation score cached latents directly. A small positional key is cached separately for rotary position information. A latent costs fewer numbers per position while still supporting different heads.

### 2026: Compress Along the Sequence
**V4‑Flash** keeps a detailed recent window and represents older context with learned summary entries. **Compressed Sparse Attention (CSA)** forms summaries at a finer stride and chooses relevant ones for a query. **Heavily Compressed Attention (HCA)** forms much coarser summaries and can attend to all of them. Two costs are at play:
1. Packing more tokens into each entry reduces how much history is stored.
2. Selecting fewer stored entries reduces the work of reading them for the next token.

### 2026: Share Across Layers and Shrink Each Entry
**DeepSeek‑V4.1‑Flash** uses **CSA2**:
- A **Full** layer writes a global cache and selects positions.
- A **Reindex** layer reads that cache and picks positions for its own query.
- A **Reuse** layer reads the cache and a preceding layer's selection.

Each layer still computes its own attention query and keeps its local 128-token window. The encoder's global entries cover pairs of positions. The decoder's cover single positions, with many layers reading the same store.

---

## Follow the Rounding: FP4 Main Cache

The main global cache uses **4-bit numbers** with a one-byte scale per group of 16:
- Sixteen such values cost $8 + 1 = \mathbf{9\ bytes}$
- Storing 16 BF16 values takes **32 bytes**

That small example explains one ingredient. The model's overall cache also includes the indexer and benefits from sharing entries across layers. Its local sliding-window cache uses FP8.

---

## What Fits in 890 MB?

Here is one way to reconstruct the reported global total:

| Component | Calculation | Bytes per entry |
|-----------|-------------|-----------------|
| Main entry (512-channel) | $512/2 + 512/16$ | **288 bytes** |
| Index key (128-channel) | $128/2 + 128/32$ | **68 bytes** |
| **Total per entry** | $288 + 68$ | **356 bytes** |

Three encoder cache writers each create one entry per two tokens. One decoder writer creates one per token. Over a long sequence, the average is:

$$(3 \times \tfrac{1}{2} + 1) \times 356 = \mathbf{890\ bytes\ per\ token}$$

Across the network, other layers can use those entries again. DeepSeek reports **890 bytes of global KV cache per token**, about one quarter of V4‑Flash's corresponding footprint. At exactly 1,000,000 tokens that is **890 MB**, about **849 MiB**. Weights, the bounded local window, temporary activations, and allocator overhead require additional memory.

---

## Compare Hypothetical vs. DeepSeek's Reported Figure

| Design | KV heads | 4,096 tokens | 8,192 tokens |
|--------|----------|--------------|--------------|
| MHA (32 KV heads) | 32 | 2 GiB | 4 GiB |
| GQA (8 KV heads) | 8 | 512 MiB | 1 GiB |
| MQA (1 KV head) | 1 | 64 MiB | 128 MiB |
| **DeepSeek V4.1-Flash** | — | ~3.6 MB | ~7.3 MB |

**Why the gap?** DeepSeek's architecture differs fundamentally:
1. **Latent compression** reduces per-entry size from 128 numbers to a compressed representation.
2. **Cross-layer reuse** means one global cache serves all 32 layers instead of 32 separate caches.
3. **FP4 main cache** shrinks each entry to ~9 bytes for 16 values (vs. 32 bytes for BF16).

---

## Account Separately: Local State, Weights, Serving Overhead

| Component | Estimate | Notes |
|-----------|----------|-------|
| **Global KV cache** | 890 bytes/token | DeepSeek's reported figure |
| **Local sliding window** | FP8, ~128 tokens | Per-layer, not shared |
| **Model weights** | Gigabytes | Not part of KV cache |
| **Temporary activations** | Variable | During forward pass |
| **Allocator overhead** | 5–10% | Fragmentation, alignment |

For persistent storage on SSD or host memory, DeepSeek reports about one eighth of its predecessor's footprint, aided by reconstructing missing sliding-window state when a session resumes. These ratios have different denominators.

---

## Check Your Prediction

**Question:** If you double the prompt length from 4,096 to 8,192 tokens while keeping KV heads at 8 (GQA), what happens to storage?

**Answer:** Storage doubles from **512 MiB to 1 GiB**. The number of query heads can stay fixed because several queries can use the same bank of keys and values. Each query still computes its own scores, so attention quality is preserved even with fewer KV heads.

**Question:** Why does DeepSeek's cache scale differently?

**Answer:** Because it uses latent compression and cross-layer reuse, not per-layer KV caches. The 890 bytes/token figure already accounts for sharing across all 32 layers, which is why it appears so much smaller than the hypothetical MHA/GQA/MQA layouts.

---

## Final Prediction to Check

**Predict:** At 16,384 tokens with DeepSeek V4.1-Flash's architecture, what is the approximate global KV cache size?

**Check:** $16,384 \times 890\ bytes \approx \mathbf{14.6\ MB}$ (global only; add local window and overhead for total).

This lesson shows how transformers evolved from storing every key-value pair separately to sharing compressed representations across layers—enabling much longer contexts with the same memory budget.
