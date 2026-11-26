# One sentence through a modern transformer — design

Date: November 26, 2026 · Status: draft for the learner's approval

## Intent

The learner asked, in their words: explain "transformer architecture stage by stage where you show how a sentence gets flows through the model for forward pass, and then ultimately generates one token", in the same style as Articles A and B (story, intuition and math, with interactive WebGL diagrams that move in sync with the text). Use "the transformer with GQA, maybe MLA, some of the newer features", and "make sure that you don't abstract out things like feed forward network and layer normalization. Show the math, show what is happening in terms of compute, how does the vector get transformed". Follow "a small sentence" through the prefill pass "with real numbers", then the decode pass that generates one token: "what we are storing, the KV cache, and then in the decode pass how we are reading up that KV cache". Show how big the final vector is, how a token is picked from it, and how the token is looked up in the vocabulary. Link it to the same site.

It is **Article C**, in `topics/looped-transformers/` beside A and B, with its own page `transformer-walkthrough.html`. The landing page, A and B link to it, and it is the recommended first read for anyone who wants the plain machinery before looping.

## Goals

- One real sentence, **"The capital of France is"** (about 5 tokens), through a real small modern model, with **real recorded numbers** at every stage.
- Prefill: tokenize, embed, then one decoder layer in full: RMSNorm, Q/K/V projections, QK-norm, RoPE, scores, causal mask, softmax, weighted values, output projection, residual, second RMSNorm, SwiGLU feed-forward, residual. Then the layer stack, the final norm, the LM head, softmax, sampling, and the vocabulary lookup of the chosen token.
- Decode: one new token, with the KV cache written at the end of prefill and **read** during decode, and the cost of that read.
- **GQA** explained on the model's real heads. **MLA** explained with real numbers from a real MLA model, including the latent cache and the "absorbed" decode trick. Other new features that the real model has (QK-norm, tied embeddings, RoPE base, SwiGLU) are shown in place, and sliding-window attention, mixture of experts and attention residuals are pointed to in Article B.
- Compute and bytes at each stage: shapes, FLOPs, parameters touched, and bytes moved.
- Same engine as A and B: one `(beat, progress)` state, Scroll, Step and Play, linked terms, narration, light and dark.

## Non-goals

- Training, tokenizer training, batching, multi-GPU, or quantisation (B covers the last).
- Every optimisation (FlashAttention, paged caches). They get a sentence each.
- A model that is good at the task: we show one sentence flowing, not a benchmark.

## Anti-goals: fails even if it technically works when…

- It abstracts away what the learner asked to see: RMSNorm, the FFN and its gate, the residuals and the head must each be a stage with their math and real numbers.
- A displayed number is invented or rounded in a way that contradicts the recording. Every number comes from the capture or is recomputed in the browser from captured tensors and must agree within tolerance.
- The decode pass is shown without the cache being read, or the cache sizes disagree with the real shapes.
- MLA is shown with fabricated numbers. If real MLA numbers cannot be captured, the MLA chapter must say it uses the config shapes only.

## Constraints

Same as A and B: static files for GitHub Pages, plain WebGL2, pinned CDN KaTeX only, light and dark, phone width, keyboard, reduced motion, English only. Weights stay in the ignored `.local/`. Feedback recorded as pending. Commit and push when ready.

## The models (my calls, to be approved)

1. **Main walkthrough: Qwen3-0.6B-Base** (`Qwen/Qwen3-0.6B-Base`, Apache 2.0, about 1.2 GB in BF16). It has 28 layers, hidden size 1024, 16 query heads and 8 key/value heads (GQA 2:1), head dimension 128, QK-norm, RoPE, RMSNorm, SwiGLU of width 3072, and tied embeddings. It is the same family as the Qwen model in the earlier study. Hidden vectors are 1024 numbers, so real vectors fit on screen with a few shown and the rest summarised. If its top prediction for the sentence is not sensible, the sentence is changed, not the numbers.
2. **MLA chapter: DeepSeek-V2-Lite, layer 0 attention only** (`deepseek-ai/DeepSeek-V2-Lite`; model files are 31 GB in total, but I would fetch only the one shard that holds the embedding and layer 0, about 4 GB, into `.local/`). Layer 0 is dense, so only attention weights are needed. I run the real attention module on the same sentence under that model's tokenizer and record the latent `c_KV`, the decoupled RoPE key, the up-projected keys and values, and the absorbed-matmul scores. The tokenization differs from Qwen's, which the page states.
3. A reference implementation is recomputed in the browser for the small steps (RMSNorm, RoPE, scores, softmax, the SwiGLU product) from captured full tensors, and unit tests require agreement with the recording to 1e-4.

## Outline (about 22 beats)

1. **The map.** The sentence's journey and the config numbers.
2. **Tokens.** Text to pieces to IDs, real IDs.
3. **Embedding.** IDs pick rows of the table: `[5] → [5, 1024]`, with real values and norms.
4. **The layer, zoomed out.** Pre-norm, attention, residual, norm, FFN, residual.
5. **RMSNorm.** The formula worked on the real vector of one token: mean square, scale, gain.
6. **Q, K, V.** The three projections, shapes `[5,2048]`, `[5,1024]`, `[5,1024]`, split into heads.
7. **GQA.** 16 query heads share 8 key/value heads; which share which; what it saves.
8. **QK-norm.** RMSNorm on each query and key head (a newer feature), with real numbers.
9. **RoPE.** Rotating pairs of features by position-dependent angles: real frequencies, real before and after.
10. **Scores and the mask.** `q·k/√d`, causal mask, softmax: a real 5×5 for a chosen head, as a heat map.
11. **Mixing values and the output projection.** Weighted values, concatenated heads `[5,2048]`, `W_o`, and the residual add.
12. **The feed-forward network.** Second norm, gate and up projections, SiLU, the product, the down projection, the residual. Real shapes, parameter and FLOP counts.
13. **The stack.** 28 layers: per-layer hidden norm and the "logit lens" prediction at each layer, real.
14. **The head.** Final norm, `W_vocab`, the logits vector `[151936]`, softmax, temperature, picking a token.
15. **From ID to word.** The chosen ID, looked up in the vocabulary, with the top candidates.
16. **Prefill, by the numbers.** FLOPs, parameters and bytes per stage.
17. **The KV cache.** What prefill leaves behind: `[8, 5, 128]` per layer for K and for V, its bytes.
18. **Decode, one token.** One row through the layers, the new K and V appended, the query reading the whole cache, with a real attention pattern.
19. **Decode, by the numbers.** Why it is memory-bound: weight bytes and cache bytes read per token.
20. **GQA versus MHA versus MQA.** Cache bytes per token on the real config.
21. **MLA.** The latent, the decoupled RoPE key, the up-projections, and the real numbers for the same sentence.
22. **MLA at decode.** Absorbing `W_UK` into the query so attention runs in latent space, real scores matching the explicit ones, and cache bytes against GQA.
23. **Where the other new features live.** Sliding window, mixture of experts, attention residuals: one paragraph each, with links to Article B.
24. **The whole thing in one picture,** with every shape.

## Scenes

All micro scenes are data-driven by a recorded `walkthrough-data.json`, drawn with the existing cell-grid style (signed values as brightness), plus new scene kinds: token columns, a head-sharing diagram for GQA, a rotation diagram for RoPE, attention heat maps, a vocabulary bar chart for the logits, a KV cache that grows, and the latent-versus-full cache comparison. The engine, narration, linked terms and tests are reused.

## Verification

- `facts.json` entries for the config values, the Hugging Face source lines of `modeling_qwen3.py` and `modeling_deepseek_v2.py` at the pinned Transformers commit, and recorded numbers.
- The capture scripts are committed and reproducible. Their outputs are small JSON files. Unit tests recompute the browser-side math from captured tensors and compare it with the recording. Arithmetic for cache and parameter counts is recomputed from config in tests, as in B.
- Browser tests for sync, narration and phone layout run on the new page too.

## Left to the builder

Exact beat splits and prose, the visuals' look, which heads and layers to show, and how many dimensions of each vector to print.
