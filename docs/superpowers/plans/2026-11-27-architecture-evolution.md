# Architecture Evolution and Interactive Diagrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add to Article C course-style "how it changed" explanations (position, normalisation, attention heads, feed-forward, embeddings) with math, history and the impact on computation, KV cache and memory for the running example, plus hover and click detail on every data-driven WebGL object.

**Architecture:** A pure `walkthrough/variants.js` computes parameters, FLOPs, cache bytes and decode bytes for each design variant from the real configs. New `walk-var-*` micro scenes draw a chosen variant, and a switch in each evolution beat changes it and refills a comparison table. The engine gains object `info`, a hover tooltip, a click-to-pin inspector and keyboard stepping.

**Tech Stack:** Existing: plain WebGL2/Canvas 2D engine, KaTeX 0.16.11, `node:test`, `playwright-core`.

**Spec:** `docs/superpowers/specs/2026-11-27-architecture-evolution-design.md` (approved November 27, 2026; the learner added: descriptive, academic-course-style explanations with the math, why each transition happened and what changed in the industry).

## Global Constraints

- Same as the earlier plans: English only, static, plain WebGL2, KaTeX only from the pinned jsdelivr URL, light and dark, 390 px, keyboard, reduced motion, spoken text for every display equation (`data-speak`), every number in the prose either from the recording (`data-walk`), a fact (`data-fact`) or recomputed by a unit test.
- History claims cite only sources verified by a fetch pass (list below). Numbers for variants the real model does not use are labelled "computed from the config".
- Verified sources (arXiv v1 dates): Vaswani 1706.03762 (Jun 2017); Shaw 1803.02155 (Mar 2018); RoPE/RoFormer Su 2104.09864 (Apr 2021); ALiBi Press 2108.12409 (Aug 2021); Xiong 2002.04745 (Feb 2020); RMSNorm Zhang 1910.07467 (Oct 2019; abstract hypothesises that re-centering invariance is dispensable); LayerNorm Ba 1607.06450 (Jul 2016); GELU Hendrycks 1606.08415 (Jun 2016); QKNorm Henry 2010.04245 (Oct 2020); MQA Shazeer 1911.02150 (Nov 2019); GQA Ainslie 2305.13245 (May 2023); DeepSeek-V2 2405.04434 (May 2024; KV cache reduced 93.3%, throughput 5.76x); GLU variants Shazeer 2002.05202 (Feb 2020; abstract does not name SwiGLU); Dauphin 1612.08083 (Dec 2016; the gating mechanism later called GLU); MoE Shazeer 1701.06538 (Jan 2017); Switch Fedus 2101.03961 (Jan 2021); Mixtral 2401.04088 (Jan 2024; 13B active of 47B); Press and Wolf 1608.05859 (Aug 2016; perplexity down, and translation models less than half the size); FlashAttention Dao 2205.14135 (May 2022); Longformer 2004.05150 (Apr 2020); Mistral 7B 2310.06825 (Oct 2023); Alammar, The Illustrated Transformer, 27 June 2018. Do not cite Dehghani 2023 for QK-norm (its abstract does not mention it).
- One commit and push at the end (the learner's standing instruction). Trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A what-if number on the page that does not equal the function that computes it (tests recompute each table cell).
2. A hover or click that works only with a mouse (keyboard test) or leaves a stuck tooltip when the scene changes.
3. A history claim stronger than its source (the evolution beats cite only verified wording).
4. A computed number mislabelled as recorded (every what-if table and scene label says "computed").
5. Narration reading TeX or table markup from the new beats.

## File Structure (under `topics/looped-transformers/`)

| Path | Responsibility |
| --- | --- |
| `walkthrough/variants.js` | Pure: `attention(kind, cfg)`, `position(kind, cfg)`, `norm(kind, cfg)`, `ffn(kind, cfg)`, `embedding(kind, cfg)` and `table(family, cfg)`. |
| `engine/micro-walk.js` | Add `info` to objects of the existing scenes, and the `walk-var-attention`, `walk-var-position`, `walk-var-norm`, `walk-var-ffn`, `walk-var-embed` scenes. |
| `engine/inspect.js` | Hover tooltip, click-to-pin inspector, keyboard stepping, bound to the scene. |
| `engine/scene.js`, `engine/micro.js` | `hit(x, y)` returning the object, `infoList()`, grid `info` option. |
| `beats/beats-c.js`, `transformer-walkthrough.html` | Five new beats: `evo-norm`, `evo-position`, `evo-attention`, `evo-ffn`, `evo-embed`. |
| `tests/unit/walk-variants.test.cjs`, `tests/unit/inspect.test.cjs`, `tests/e2e/inspect.test.cjs` | Tests. |

**Contracts**

- `variants.js` cfg: `{ layers, hidden, heads, kvHeads, headDim, ffn, vocab, maxPos, tokens, context, bytes }` (Qwen3-0.6B: 28, 1024, 16, 8, 128, 3072, 151936, 32768, 5, 32768, 2). Each function returns plain objects: attention `{ kvHeads, params (K and V projections per layer), cachePerTokenPerLayer, cachePerToken, cacheAtContext, decodeReadAtContext }`; position `{ params, ops, where }`; norm `{ opsPerVector, params, placement }`; ffn `{ params, flopsPerToken, activeParams }`; embedding `{ params }`.
- Object info: `info: { title: string, lines: string[] }` on any scene object. `scene.hit(px, py)` returns the topmost object with `info` under a CSS-pixel point, preferring the smallest.
- DOM: `#inspector` (in `aside#stage`), `#tip` (in `.stage-box`), `[data-variants="<family>"]` button groups with `data-variant`, and `<div data-variant-table="<family>">`.

---

### Task 1: The variant model

- [ ] **Step 1: Failing tests** `tests/unit/walk-variants.test.cjs`: for the Qwen3 cfg, `attention('mha')` gives cache 8,192 B/token/layer, `gqa` 4,096, `mqa` 512, and `cacheAtContext` for 32,768 tokens over 28 layers 7,516,192,768 / 3,758,096,384 / 469,762,048; K and V projection params per layer 4,194,304 (MHA), 2,359,296 (GQA, 8 heads: 2×1024×1024... computed), 262,144 (MQA); `mla` with the DeepSeek-V2-Lite numbers gives 1,152 B; `position('learned')` params `maxPos × hidden` = 33,554,432 and `rope` 0; `norm('layernorm')` has more ops per vector and parameters than `rmsnorm` (counts from the formulas in the beat); `ffn('relu')` at width 4×hidden has 8,388,608 params per layer, `swiglu` at 3072 has 9,437,184, and `swiglu` at the 8/3 rule (2730) is within 0.1% of the ReLU count; `moe` with 8 experts and 2 active has stored = 8× and active = 2× the dense FFN params; `embedding('untied')` has `vocab × hidden` more than `tied`.
- [ ] **Step 2:** Run; expect FAIL. **Step 3:** Implement `walkthrough/variants.js` (UMD). **Step 4:** Run; expect PASS.

### Task 2: Interactivity engine

- [ ] **Step 1: Failing tests** `tests/unit/inspect.test.cjs` (pure parts: choosing the smallest object under a point, ordering `infoList()` by screen position) and `tests/e2e/inspect.test.cjs` on the gallery fixture: hovering a score cell shows a tooltip containing its recorded probability; clicking pins it in `#inspector` with the dot product line; Escape clears; Tab to the canvas, ArrowRight/ArrowLeft move through objects and Enter pins (and the tooltip text equals the hover text); changing beat clears the tooltip; the inspector is `aria-live="polite"`; touch tap pins.
- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement `scene.hit`, `scene.infoList`, grid `info(r, c)` in `micro.js`, `engine/inspect.js`, CSS, and the markup in both article shells. **Step 4:** Run; PASS.

### Task 3: Info on the existing walkthrough scenes

- [ ] **Step 1: Failing tests** in `walk-scenes.test.cjs`: every data-driven cell of `walk-embed`, `walk-norm`, `walk-qkv`, `walk-qknorm`, `walk-rope`, `walk-scores`, `walk-mix`, `walk-ffn`, `walk-stack`, `walk-head`, `walk-decode`, `walk-mla`, `walk-mla-absorb` has `info` whose first line names the quantity and whose lines contain the exact recorded value (scores: `raw`, `weight`, and the query-key dot product recomputed with `ref.dot`).
- [ ] **Step 2:** Run; FAIL. **Step 3:** Add `info` in `micro-walk.js`. **Step 4:** Run; PASS.

### Task 4: Variant scenes and the switch

- [ ] **Step 1: Failing tests:** each `walk-var-*` scene returns objects for every variant of its family with labels carrying the computed numbers (matching `variants.js`); a browser test that clicking `[data-variant="mqa"]` in the attention beat changes the scene's labels and the comparison table cells to the `variants.js` values, and that the table says "computed from the config".
- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement the scenes, `engine/variants-ui.js` (button groups, table rendering from `variants.js`), and wire `scene.setData('walk', {…, variant: {attention: 'gqa', …}})`. **Step 4:** Run; PASS.

### Task 5: The course-style beats

Five beats, each with `.story`, `.intuition`, `.math`, a variant switch and a what-if table, inserted after their stage: `evo-norm` (after `qknorm`), `evo-position` (after `rope`), `evo-ffn` (after `ffn`), `evo-embed` (after `head`), `evo-attention` (after `mla-absorb`). Each states: the problem the earlier design had (with the math), what the field changed and when (cited), why it worked, the impact on computation, KV cache and memory with our example's numbers, and what it cost.

- [ ] **Step 1: Failing checks:** `check-article`, speech-coverage and facts checks extended to the new beats; beats-c ordering test; narration test that an evolution beat is read without TeX.
- [ ] **Step 2:** Run; FAIL. **Step 3:** Write the beats (add the verified citations to `facts.json` as `url` facts first). **Step 4:** Run; PASS.

### Task 6: Integration, review, publish

- [ ] **Step 1:** Update the landing page, README, notes and Article A and B nav text for the interactive diagrams and the new beats.
- [ ] **Step 2:** Full suite (unit then e2e serially), the three checkers, screenshots of the new beats in light/dark at desktop and phone and fix what looks wrong, Pages build and the Pages test.
- [ ] **Step 3:** Commit, dispatch a fresh reviewer with this plan's Review Focus, fix findings test-first, rerun, push, and confirm the live site.
