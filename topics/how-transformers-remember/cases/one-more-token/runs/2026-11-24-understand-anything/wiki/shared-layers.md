# Let several layers read one global store

In DeepSeek‑V4.1‑Flash, a Full layer writes global KV and selects positions. A Reindex layer reads shared KV and chooses new positions. A Reuse layer also reuses a preceding selection.

Every layer computes its own attention query and retains its own 128-token local attention window. Of the global writers, three encoder layers each write one entry per pair of tokens; one decoder layer writes one entry per token. Averaged over a long sequence that is 3 × ½ + 1 = 2.5 entries per token across the network. A fractional average counts entries over many tokens.

Combine that rate with [[entry-precision]] to reconstruct the [[global-budget]]. Earlier [[compressed-history]] addresses how far back positions are represented. [CSA2 and model configuration, §§2.3 and 4.2.1](https://arxiv.org/html/2609.19969v1)
