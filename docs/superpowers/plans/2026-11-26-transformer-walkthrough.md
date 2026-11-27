# One Sentence Through a Modern Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Article C, `topics/looped-transformers/transformer-walkthrough.html`, which follows "The capital of France is" through a real Qwen3-0.6B-Base forward pass and one decode step with recorded numbers, then explains MLA with real numbers from DeepSeek-V2-Lite.

**Architecture:** Python capture scripts record real tensors into small JSON files. A JavaScript reference (`walkthrough/ref.js`) recomputes the small steps from captured tensors, and tests require agreement with the recording. New data-driven micro scenes draw those tensors with the existing engine. The page reuses the engine, narration, linked terms and tests from Articles A and B.

**Tech Stack:** The existing venv (`.local/inkling2/venv`, Transformers 5.25.0.dev0 at the pinned commit), numpy, plain JS UMD modules, plain WebGL2, KaTeX 0.16.11, `node:test`, `playwright-core`.

**Spec:** `docs/superpowers/specs/2026-11-26-transformer-walkthrough-design.md` (approved November 26, 2026).

## Global Constraints

- English only. Static files for GitHub Pages. Plain WebGL2. KaTeX only from jsdelivr at `katex@0.16.11` (pinned, with the existing SRI hashes).
- Light and dark via CSS tokens, 390 px width with no horizontal scroll, keyboard reachable, `prefers-reduced-motion` respected.
- Every number in the prose is either wrapped in `data-fact` (checked by `tests/check-facts.cjs`) or recomputed in a unit test. Display equations are wrapped `<span class="eq" data-speak="…">` with a spoken version (`tests/unit/speech.test.cjs` enforces it).
- Pinned sources: model `Qwen/Qwen3-0.6B-Base` (record the revision hash used); `deepseek-ai/DeepSeek-V2-Lite` (record its revision hash); Transformers `b62646bbda836d467060b7e3d64028202079b77d`.
- Weights stay in the ignored `.local/walkthrough/`. Only the small recorded JSON files and the capture scripts are committed.
- The MLA chapter says plainly that DeepSeek's tokenizer splits the sentence differently from Qwen's, and uses only recorded numbers.
- The learner asked for one commit and push at the end. Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Plain `git push`.

## Review Focus

1. **Recorded numbers disagree with the browser's recomputation.** Every `ref.js` function is tested against the recording at 1e-4. A mismatch fails the unit tests.
2. **A cache size on the page contradicts the config.** Recomputed from config in `arithmetic.test.cjs`.
3. **The sentence tokenizes differently on another machine.** The tokens are recorded, and the capture asserts the expected IDs.
4. **A model prediction that is not sensible.** The capture fails if the top-1 next token is not in a short allowed list for the sentence, so the sentence is changed rather than the numbers.
5. **The page is long.** Narration, Step and Play, phone layout and dark mode are exercised on the new page by the existing e2e files.

## File Structure

All paths under `topics/looped-transformers/` unless stated.

| Path | Responsibility |
| --- | --- |
| `assets/walkthrough/capture_qwen.py` | Runs Qwen3-0.6B-Base with hooks on the sentence (prefill, then one decode step) and writes `walkthrough-data.json`. |
| `assets/walkthrough/capture_mla.py` | Fetches only the embedding and layer-0 tensors of DeepSeek-V2-Lite by HTTP range requests, runs the real MLA attention on the sentence, writes `walkthrough-mla.json`. |
| `assets/walkthrough/requirements.txt` | Pinned extras (numpy, requests). |
| `walkthrough/ref.js` | Pure math from captured tensors: `rmsnorm`, `rope`, `softmax`, `scores`, `swiglu`, `silu`, `mlaScores`, `mlaAbsorbed`. |
| `engine/micro-walk.js` | Data-driven micro scenes for Article C (registered into `LoopLayout.MICRO`). |
| `beats/beats-c.js`, `transformer-walkthrough.html` | Beat data and prose for the article. |
| `tests/unit/walk-*.test.cjs`, `tests/e2e/walkthrough.test.cjs` | Parity, scene, arithmetic and browser tests. |

**Contracts**

- `walkthrough-data.json` top-level keys: `meta` (model, revision, transformers_commit, dtype `float32`, sentence, `tokens: [{id, text}]`), `config` (hidden, layers, heads, kv_heads, head_dim, ffn, vocab, rope_theta, eps, tied), `embed` (`h0` as 5×1024, `row_norms`), `layer0` (`norm1`, `qkv`, `qknorm`, `rope`, `scores`, `attn_out`, `resid1`, `norm2`, `ffn`, `out`), `stack` (per-layer last-token norm and logit-lens top token), `head` (final norm, logits summary, top-20, chosen token), `cache` (shapes), `decode` (new token, layer-0 q/k/v, probabilities over 6 keys for all 16 heads, logits top-10). Full tensors are stored where `ref.js` needs them; the rest are first 16 dims plus statistics.
- `walkthrough-mla.json`: `meta`, `config` (hidden 2048, heads 16, kv_lora_rank 512, qk_nope 128, qk_rope 64, v 128, yarn), `tokens`, `latent` (`c_kv` T×512, `k_rope` T×64), `explicit` (`k_nope`, `v`, scores for one head) and `absorbed` (scores for the same head).
- `LoopWalk.ref` functions take and return plain arrays of numbers: `rmsnorm(x: number[], gain: number[], eps: number): number[]`, `rope(x: number[], pos: number, theta: number): number[]` (half-split pairs, as Qwen3 does), `softmax(v: number[]): number[]`, `silu(x: number): number`, `swiglu(gate: number[], up: number[]): number[]`.
- Micro scene names (all take `data` as the second argument of `microObjects(name, data)`): `walk-tokens`, `walk-embed`, `walk-layer`, `walk-norm`, `walk-qkv`, `walk-gqa`, `walk-rope`, `walk-scores`, `walk-mix`, `walk-ffn`, `walk-stack`, `walk-head`, `walk-cache`, `walk-decode`, `walk-mla`, `walk-mla-absorb`, `walk-cachesize`.

---

### Task 1: Capture the Qwen3 forward pass

**Files:** Create `assets/walkthrough/capture_qwen.py`, `assets/walkthrough/requirements.txt`, `assets/walkthrough/walkthrough-data.json`, `tests/unit/walk-data.test.cjs`

- [ ] **Step 1: Write the failing test** `walk-data.test.cjs`: loads the JSON and asserts the schema in the contract above (keys present, `tokens.length === 5` and `meta.sentence === "The capital of France is"`, `embed.h0` is 5×1024, `layer0.scores` has 16 heads of 5×5 with each row summing to 1 within 1e-5 and zeros above the diagonal, `decode.probs` is 16×6 rows summing to 1, `head.top20[0].prob` is the largest, `cache.k_shape` equals `[8, 5, 128]`, and the file is under 2 MB).
- [ ] **Step 2:** Run `cd topics/looped-transformers/tests && node --test unit/walk-data.test.cjs`. Expected: FAIL (file missing).
- [ ] **Step 3: Implement the capture** in `capture_qwen.py` with the model from `.local/walkthrough/qwen3` (download: `hf download Qwen/Qwen3-0.6B-Base --local-dir .local/walkthrough/qwen3`). Load in float32 with `attn_implementation="eager"`. Hook layer 0 to record every stage in the contract. Recompute RMSNorm, RoPE, scores, softmax and SwiGLU in numpy float64 from the recorded inputs and assert they match the model's recorded outputs within 1e-4 before writing. Record the per-layer last-token hidden norm and the logit-lens top token. Run the decode step with the model's cache and record the new token's layer-0 q/k/v and probabilities over all 6 keys. Assert that the greedy next token is in `[" Paris", " the", " a", " located", " one"]`; if not, print the top 10 and exit non-zero.
- [ ] **Step 4:** Run `.local/inkling2/venv/bin/python topics/looped-transformers/assets/walkthrough/capture_qwen.py`, then the test again. Expected: PASS.

### Task 2: Capture the MLA layer

**Files:** Create `assets/walkthrough/capture_mla.py`, `assets/walkthrough/walkthrough-mla.json`, `tests/unit/walk-mla-data.test.cjs`

- [ ] **Step 1: Write the failing test:** schema from the contract; `latent.c_kv` is T×512 and `latent.k_rope` T×64; the explicit and absorbed scores for the recorded head agree within 1e-3 relative; cache bytes per token from the config (`(512 + 64) × 2` against `16 × (128 + 64 + 128) × 2`) equal the values stored in `meta`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3: Implement.** Read `model.safetensors.index.json`, find the shard of `model.embed_tokens.weight` and `model.layers.0.*`, then fetch only those tensors with HTTP `Range` requests on the resolve URL (read the 8-byte header length and JSON header first). Build a one-layer `DeepseekV2` attention module from the real config, load the fetched weights, and run it on the DeepSeek tokenization of the sentence. Capture `c_kv` (after `kv_a_layernorm`), `k_rope` (after RoPE), the explicit per-head `k_nope` and `v`, the scores for head 0, and the absorbed scores `(q_nope W_UK) · c_kv + q_rope · k_rope`, with the YaRN scaling taken from the module.
- [ ] **Step 4:** Run the script and the test. Expected: PASS.

### Task 3: The JavaScript reference and parity tests

**Files:** Create `walkthrough/ref.js`, `tests/unit/walk-ref.test.cjs`

**Interfaces:** Consumes the two JSON files. Produces `LoopWalk.ref` as in the contract plus `scores(q: number[][], k: number[][], headDim: number): number[][]` (causal mask applied, softmax per row) and `mlaScores`, `mlaAbsorbed`.

- [ ] **Step 1: Write the failing tests:** for each stage in `layer0` recompute from the recorded inputs with `ref.js` and compare to the recorded output within 1e-4: `rmsnorm` of each of the 5 `h0` rows, QK-norm of head 0, `rope` of q and k head 0 at positions 0–4, the 5×5 softmax of head 0, the SwiGLU product, and the decode probabilities over 6 keys. For MLA, `mlaScores` and `mlaAbsorbed` agree with each other and with the recording.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement. **Step 4:** Run; expected PASS.

### Task 4: Data-driven scenes

**Files:** Create `engine/micro-walk.js`, `tests/unit/walk-scenes.test.cjs`; modify `tests/e2e/micro.test.cjs` and `tests/fixtures/micro-gallery.html` to include the new scenes with the recorded data.

**Interfaces:** Registers the scene names in the contract into `LoopLayout.MICRO` and takes the recorded data as `data`. Reuses `grid(...)`'s cell style and signed values. Every scene works without data (placeholder grids) so beats can use `LoopLayout.microCamera(name, margin, {placeholder: true})`.

- [ ] **Step 1: Write the failing unit tests:** every `walk-*` scene returns non-empty objects with unique ids, positive sizes and a labelled `tensor.*` frame with a shape label like `[5, 1024]`; `walk-scores` cell values equal the recorded probabilities and are zero above the diagonal; `walk-gqa` has 16 query-head objects and 8 key/value-head objects and each query head `h` points at kv head `⌊h/2⌋`; `walk-head` has a bar for each of the recorded top-20 tokens with heights in proportion to probability; `walk-cachesize` bar heights match the arithmetic in `arithmetic.test.cjs`.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement the scenes. **Step 4:** Run; expected PASS, then add the browser gallery test (non-blank, fits at four window sizes, both renderers) and run it.

### Task 5: Facts and arithmetic

**Files:** Modify `assets/facts.json`, `tests/unit/arithmetic.test.cjs`

- [ ] **Step 1: Write failing tests** that recompute from the config in the recording: the KV cache bytes per token per layer for GQA (`2 × 8 × 128 × 2`), for the MHA equivalent (16 kv heads) and MQA (1), the total for 28 layers, the parameters per layer split into attention and FFN, the FLOPs of each prefill stage for T = 5, the bytes of weights read in one decode step, and the MLA numbers from the MLA recording.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Add `config` facts for Qwen3's `config.json` values (a new `file` pointing into `.local/walkthrough/qwen3/config.json`), `source` facts with line numbers from `modeling_qwen3.py` and `modeling_deepseek_v2.py` (found by quote with a helper script like the one used for Article B), and `json` facts into the two recordings. **Step 4:** Run the tests and `node tests/check-facts.cjs --transformers <site-packages>/transformers` ; expected PASS.

### Task 6: Article C, beats 1–9 (map to QK-norm and RoPE)

**Files:** Create `transformer-walkthrough.html`, `beats/beats-c.js`; modify `engine/boot.js` only if a data loader is needed.

Beats (ids): `map`, `tokens`, `embed`, `layer-overview`, `rmsnorm`, `qkv`, `gqa`, `qknorm`, `rope`. Each has `.story`, `.intuition`, `.math` (with shape annotations and spoken equations), real numbers cited from the recording by `data-fact` or from a data attribute resolved at load, and a scene from Task 4.

- [ ] **Step 1: Write the failing checks:** `node tests/check-article.cjs transformer-walkthrough.html` (structure), `tests/unit/speech.test.cjs` equation coverage extended to the new page, and an e2e test in `walkthrough.test.cjs` that the page loads with no console errors, every beat's scene draws (non-blank, in bounds) and Scroll, Step and Play keep text and scene together.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Write the shell (copy the controls block, scripts and styles from `inside-inkling.html`), the beats and the prose. Load `walkthrough-data.json` at runtime into the scenes (`scene.setData('walk', data)`). **Step 4:** Run the checks and the e2e file; expected PASS.

### Task 7: Article C, beats 10–17 (scores to the KV cache)

Beats: `scores`, `mix`, `ffn`, `stack`, `head`, `token-pick`, `prefill-cost`, `kv-cache`. Same structure as Task 6.

- [ ] **Step 1:** Extend the failing checks to these beats (structure, speech coverage, scene fits). **Step 2:** Run; FAIL. **Step 3:** Write them. Required content: the real 5×5 matrix as a heat map; SwiGLU shown as gate, up, SiLU, product, down with shapes and FLOPs; the per-layer norm and logit-lens chart (chart via `engine/charts.js`, palette validated with the dataviz validator); the logits vector `[151936]` with the top-20 bar chart, softmax with temperature, the chosen ID looked up in the vocabulary; prefill FLOPs, parameters and bytes per stage; the cache written at the end of prefill. **Step 4:** Run; PASS.

### Task 8: Article C, beats 18–24 (decode, GQA vs MHA vs MQA, MLA, recap)

Beats: `decode`, `decode-cost`, `gqa-vs-mha`, `mla`, `mla-absorb`, `other-features`, `recap`.

- [ ] **Step 1:** Extend the failing checks. **Step 2:** Run; FAIL. **Step 3:** Write them. Required content: one new row through layer 0 with the new K and V appended and the query reading all 6 cached keys (real probabilities for all 16 heads); bytes of weights read per decode step against bytes of cache read; cache bytes per token for MHA, GQA and MQA on the real config, and as a chart against context length; MLA with the real latent `[5, 512]`, the decoupled RoPE key `[5, 64]`, the up-projections, the cache comparison, and the absorbed-score check with real numbers; one paragraph each on sliding window, mixture of experts and attention residuals linking to Article B. **Step 4:** Run; PASS.

### Task 9: Integration, review and publish

**Files:** Modify `index.html` (landing), `why-loop.html` and `inside-inkling.html` (nav links), `README.md`, `notes.md`, `topics/README.md`, root `index.html`, `scripts` only if the build drops files; create `tests/e2e/walkthrough-*.test.cjs` additions.

- [ ] **Step 1:** Add the landing card, nav links and README/notes sections (status, sources, limits, feedback pending with a grounded question such as "At the RoPE beat, could you predict which way the rotation turns the query before the text said it?").
- [ ] **Step 2:** Run the whole suite (unit, then e2e serially) and the three checkers (`check-article`, `check-facts` with `--transformers`, `check-links --tracked`). Expected: all pass.
- [ ] **Step 3:** Screenshots of every beat in light and dark at 1400×900 and 390×844 (`tests/e2e/screenshots.cjs`), read them, and fix overlaps and clipped labels.
- [ ] **Step 4:** Build the site (`python3 scripts/build-site.py`) and run `tests/e2e/pages.test.cjs`; extend it to open Article C by the landing card.
- [ ] **Step 5:** Commit, dispatch a fresh reviewer on the commit with this plan's Review Focus, fix findings test-first, run the suite again, then `git push`. Report the commit hashes and the live URL.
