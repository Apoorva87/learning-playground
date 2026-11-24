# Store a learned latent

DeepSeek‑V2’s multi-head latent attention stores a compact learned vector per position plus a positional key. Head-specific projections can use that shared vector.

Let c = W↓h compress the token’s hidden state h. The content score qᵀ(W↑c) equals (W↑ᵀq)ᵀc. Moving the projection onto the query allows scoring directly against the stored latent. The positional key preserves rotary position information. This reduces how many numbers each position needs while retaining head-specific transformations.

Compare this representation with the explicit KV banks in [[shared-heads]]. Another dimension available for compression is the number of positions represented, explored in [[compressed-history]]. [DeepSeek‑V2, §2.1](https://arxiv.org/html/2405.04434v5)
