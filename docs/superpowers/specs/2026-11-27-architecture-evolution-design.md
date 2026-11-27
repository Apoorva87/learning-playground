# Article C, second pass: how the choices evolved, and diagrams you can poke — design

Date: November 27, 2026 · Status: draft for the learner's approval

## Intent

The learner, after reading Article C, asked for two things. First, in the spirit of Jay Alammar's *The Illustrated Transformer*, each architectural choice must be explained **comparatively and historically**: what it replaced, why the field moved on as research progressed, and what each choice does to **computation, the KV cache and memory**, using the sentence that is already flowing through the model as the example. Second, the WebGL diagrams should be **interactive**: hover over a thing to see its detail, click it to see more. (The message was cut off after "I think that"; I read the rest as the interactivity request.)

An honest answer to "do the figures follow the literature?": they use the standard components and the real numbers of a modern model, but they do not yet show the lineage, and they are look-only.

## Goals

- For every stage that has a family of designs, a **"How it changed"** panel beside the stage's beat with: the older design, what broke or cost too much, the successor, the date and citation, and a **what-if table for our example** with the impact on (a) parameters, (b) FLOPs for the five-token prompt and for one decode step, (c) bytes of KV cache per token and at 32,768 tokens, (d) bytes read per decode step.
- A **variant switch** in the scene for the families below. Flipping it redraws the diagram (for example MHA, MQA, GQA, MLA) and updates the table and the numbers in the beat.
- **Hover and click** on every data-driven scene object: hover shows a tooltip (what it is, its shape, its exact value); click pins it in an inspector with the arithmetic behind it (a score cell shows the dot product of that query and key; a vocabulary bar shows token, ID, logit and probability; a head shows which key/value head it shares).
- Keyboard and screen-reader access to the same details.
- Every claim about history cites a source checked against its page, as in A and B. Numbers for designs the real model does not use are labelled **computed from the config**, not recorded.

## Families and transitions to cover

1. **Position:** sinusoidal added to the embedding (Vaswani 2017) → learned absolute (BERT, GPT-2) → relative (Shaw 2018; T5) → rotary (RoPE, Su 2021) and ALiBi (2021). Impact: where position enters, parameters, and what is cached (RoPE-rotated keys).
2. **Normalisation:** post-LayerNorm (2017) → pre-LayerNorm (stability at depth) → RMSNorm (Zhang and Sennrich 2019) → QK-norm. Impact: FLOPs and memory traffic of the norm, scale of the residual stream, score range.
3. **Attention heads:** MHA → MQA (Shazeer 2019) → GQA (Ainslie 2023) → MLA (DeepSeek-V2, 2024). Impact table from the real config: K/V projection parameters, cache bytes per token and per context length, decode bytes read, and the quality trade-off as the papers state it. Sliding-window attention and FlashAttention get a short note each (what changes and what does not).
4. **Feed-forward:** ReLU (2017) → GELU (BERT/GPT) → gated variants and SwiGLU (Shazeer 2020) → mixture of experts (Shazeer 2017, Switch, Mixtral and later). Impact: parameters and FLOPs per token at matched width, the 8/3 width rule, active versus stored parameters.
5. **Embeddings:** untied versus tied (Press and Wolf 2016). Impact: 155,582,464 parameters counted once.

## Approach

- One "variant" model in `walkthrough/variants.js`: pure functions from the real config (heads, kv heads, dims, layers, context) to parameters, FLOPs, cache bytes and decode bytes for each variant, unit-tested against arithmetic already in `walk-arithmetic.test.cjs`.
- Scenes read a `variant` field from the scene data (set by the switch). New scenes: a side-by-side head-sharing diagram, a position-scheme diagram (add versus rotate), a norm diagram (post versus pre), an FFN width diagram (ReLU versus gated), and the comparison bar charts.
- Interactivity in the engine: objects carry `info: {title, lines[]}`; `scene.js` gets `hit(x, y)`, a hover tooltip layer in the DOM, a click-to-pin inspector panel under the scene, and keyboard stepping through objects (arrow keys on the focused canvas, Enter to pin). Existing term linking, narration and Scroll/Step/Play keep working.
- Citations verified by a fetch pass before use (Vaswani, Shaw, RoPE, ALiBi, post/pre-LN, RMSNorm, MQA, GQA, DeepSeek-V2, SwiGLU, Switch/Mixtral, Press and Wolf, FlashAttention, Alammar's post).
- New beats are added to Article C after the matching stage (or as a "How it changed" sub-block of the beat), keeping the 24 beats' flow and the narration, equations-in-words and tests.

## Non-goals

Retraining other variants (every alternative is computed from sizes, not recorded), benchmarking quality, or a survey of every attention variant.

## Anti-goals

Presenting computed what-if numbers as recorded; history stated without a verified source; interactions that only work with a mouse; tooltips that cover the thing being inspected.

## My calls to confirm

1. Put the history inside Article C, beside each stage, and not in a new article.
2. What-if numbers are computed from config and labelled, since training other variants is out of scope.
3. Build the interactivity engine-wide (so A and B scenes can use it later) but fill in the details only for Article C's scenes now.
