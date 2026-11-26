# Looped transformers, stage by stage — design

Date: November 26, 2026 · Status: approved by the learner on November 26, 2026; built (see `docs/superpowers/plans/2026-11-26-looped-transformers.md` and `topics/looped-transformers/notes.md`)

## Intent

The learner wants to understand **looped transformers**: why someone would run the same layers more than once, what the math does at every stage, and what it costs and buys. Inkling 2 Mini (Thinking Machines, released November 20, 2026) is the worked example.

In their words: “I want to do stage by stage understanding of what is the math that is happening the intuition behind that and you know the story and the web animation should move in SYNC.” They want blog-style explanatory articles like Jay Alammar’s *The Illustrated Transformer*, with diagrams like Brendan Bycroft’s *LLM Visualization*.

Every stage gets three things together:

1. **Story.** Why this stage exists and what problem it solves.
2. **Intuition.** A picture or analogy that makes the mechanism feel obvious.
3. **Math.** The actual equation, with real shapes and, where possible, real numbers.

The 3D scene must show the part the reader is reading about, at the moment they read it.

**Who it is for.** The learner already worked through the Qwen3.8-27B code walkthrough, measured profiles, and the whole-pass dependency graph (`topics/qwen-forward-pass/`). They are comfortable with tensor shapes, attention, FLOPs, memory bandwidth, rooflines, and serial vs parallel work. New mechanisms (message gates, depth-wise attention residuals) need full explanation; standard attention needs a reminder, not a lesson.

## Goals

- **Article A, “Why loop a transformer?”** The general idea: motivation, lineage, the math of a loop, how Inkling implements it, what it costs (parameters, FLOPs, bandwidth, KV cache, serial depth), and what more loops buy, using published evidence.
- **Article B, “Inside Inkling 2 Mini.”** The full anatomy, component by component, from token IDs to logits. Looping is one major chapter among the others.
- One shared WebGL engine drives both articles. Text and animation stay in sync under **all three navigation modes** the learner asked for (see Approaches).
- Every factual claim traces to a pinned source.
- It works on GitHub Pages without local services, and it is linked from the root index as a story, following the repo's conventions in `CODEX.md`.

## Non-goals

- The vision tower (23-layer ViT). It gets one sentence in B.
- Training the real model, or reproducing Thinking Machines' benchmarks.
- A general survey of every looped-model paper. The lineage is a short story, not a literature review.
- Running the real 26.4B model in the browser.

## Anti-goals: it fails even if it technically works when…

- Text and animation drift. The reader reads about pass 2 while the scene shows pass 1, or scrolling back doesn't rewind.
- It turns into slides. The learner asked for flowing blog-style prose; scrolling must feel like reading an article, with the scene following.
- The math is decorative: equations without real shapes, or symbols the scene never shows.
- It makes unsupported claims. If Thinking Machines only states something as intuition, such as “middle layers carry more abstract reasoning”, the text must say it is their intuition. Their results back it only partly (looping the middle 12 layers beat looping the last 12 on 27 of 28 evals), and the text must say that too.
- Looping drowns in Inkling's other features in Article A, or Article B skips them.

## Constraints

- English only (`CODEX.md`, `preferences.md`).
- It must be static and run on GitHub Pages. External scripts may only come from pinned CDN versions (cdnjs, jsdelivr, unpkg). Most existing pages use none, and plain WebGL2 is the house style (`claude-graph/graph3d.js`).
- Light and dark mode, phone width with no horizontal scroll, keyboard reachable, and respect `prefers-reduced-motion`.
- The learner's feedback is recorded as **pending** until they respond (`CODEX.md` step 5–6).
- Commit and push only when the learner asks.

## What is already known (verified November 26, 2026)

Sources are pinned so the builder can line-check them:

- **Model card and config:** `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, and `thinkingmachines/TML-d24-varloop-research` at `91a57fe2b779bbd12c0f74d1f518cbd5b84e8d0d`.
- **Code:** the Hugging Face Transformers source `src/transformers/models/inkling2_loop/` and `models/inkling2/` at commit `b62646bbda836d467060b7e3d64028202079b77d`. The model was added in #51050 on November 21, 2026.
- **Thinking Machines announcement:** https://thinkingmachines.ai/news/inkling-2-mini/
- **Hugging Face blog:** https://huggingface.co/blog/inkling2mini
- **Attention residuals:** Kimi Team, *Attention Residuals*, arXiv 2603.15031.

### Shape of the model (from the config)

- **Hidden size and attention:** hidden 2304, 24 query heads, 8 KV heads, head dim 128.
- **Vocabulary:** 201,024, of which 200,058 entries are real.
- **Layers:** 24 physical layers in 4 blocks of 6 (`n_layers_per_attn_res_block` 6).
- **Attention types:** 20 sliding-window layers (window 1024) and 4 full-attention layers. The full-attention layers are 5, 11, 17 and 23, the complement of `local_layer_ids`.
- **Mixture of experts:** 128 routed experts, 8 active, no shared expert. Experts use SwiGLU with intermediate size 1152, `route_scale` 8, and softcaps of 4 (gate) and 25 (up).
- **Position information:** learned relative-position logits (`d_rel` 16, `rel_extent` 1024).
- **Other:** final logit softcap 30, and context length up to 1,048,576.
- **Loop config:** `{core_start 6, core_end 18, num_passes 3, gate_rank 64}`.

### Counts (from the model card)

- 26.4B parameters in total.
- 3.4B active for one pass, about 1.0B more per extra pass, and 5.4B per token at 3 passes.
- Running layers 6–17 three times gives 6 + 3×12 + 6 = **48 layer visits from 24 layers of weights**.

### The loop as implemented (from `Inkling2LoopTextModel.forward`)

1. **Prelude.** Layers 0–5 run once. Save `pre_loop_hidden_states` and `pre_loop_block_states`.
2. **Pass 1.** Layers 6–17 run normally, with no messages.
3. **Passes 2 and later.** Each pass restarts from the prelude's hidden state and block states. It does **not** take the previous pass's output as its input. Information from the previous pass arrives only as **messages**:
   - For each core layer *j*, `message_input` forms 4 messages: for the query, key, value and MLP inputs. Each message attends with a learned query (initialised to zero, so it starts as a uniform average) over the previous pass's block states plus its final hidden state.
   - Each message enters through a **message gate**, `out = (1 − tanh(W_f z)) ⊙ h + tanh(W_m z) ⊙ m`, where `z = W_down [rms(h); rms(m)]` has rank 64.
   - Thinking Machines says the gate “initially blocks the entire message” in training, so the looped model starts out as the unlooped one. That is a training-recipe claim, not visible in the inference code.
4. **Separate caches.** Each pass uses **its own copy of the KV and convolution caches** (`cache.layers += deepcopy(...)`), so the cache grows like a 48-layer model.
5. **Coda.** Layers 18–23 run once. Then `final_input` (a depth-wise attention residual), the final norm, the LM head and the logit softcap.

### Attention residuals (from `Inkling2BlockAttentionResidual`)

This is depth-wise attention. Within a block of 6, a normal residual stream accumulates. At each block boundary the stream's state is appended to `block_states`, and the stream restarts.

The input to each attention and MLP sublayer is formed by a learned, zero-initialised query. It attends over the RMS-normalised stack [block states…, current partial state].

### What more loops buy (from the variable-loop research model's card and the announcement)

- **DeepSWE:** 0.7% → 28.3% → 47.4% → 46.9% → 48.2% for 1–5 passes.
- **GraphWalks BFS 256K:** 21.1 → 54.5 → 63.2 → 64.3 → 64.9.
- **SimpleQA Verified:** stays at 6–8% (looping doesn't add knowledge).
- **Mean reasoning tokens:** 3,703 at one pass, 1,731 at three and 1,661 at five.
- **N-hop lookup:** keeps improving out to ten loops.
- **Loop placement:** looping the middle layers beat looping the last layers on 27 of 28 evals.
- **Monitorability:** “at this scale we do not see a meaningful reduction.”
- **Caveat:** the research card warns against comparing the research model directly with Mini.

## Approaches considered

**How text and animation stay in sync.** Three options were laid out:

1. **Scroll-scrubbed.** The scene is a pure function of (current beat, progress through it). Scrolling back rewinds.
2. **Stepper.** Next, Prev and Play, as in the existing Qwen walkthrough (`topics/qwen-forward-pass/index.html`, `walkthrough/lesson.js`). Its reading edition has no animation.
3. **Scroll-triggered.** Reaching a paragraph starts a timed animation.

**The learner chose all three.** They become three ways to drive one source of truth, so they cannot drift. The design is option 1's engine with option 2's and option 3's controls layered on:

- **Scroll mode (default).** Scroll position sets (beat, progress).
- **Step mode.** Next, Prev and the ←/→ keys move to the adjacent beat. The article scrolls there, and the scene tweens that beat's progress from 0 to 1.
- **Play mode.** The beats advance on a timer at reading pace, with an adjustable speed. The article scrolls with them. Any manual scroll, click or key pauses Play.

The text and the scene always show the same beat, because there is exactly one `(beat, progress)` state and every mode writes to it.

**Two articles, or one?** Both, as the learner asked: “I want both A and B.” A is the conceptual article. B is the full anatomy. They share the engine and link to each other at matching beats; for example, A's message-gate beat links to B's gate internals.

## Decisions

**Decided by the learner:**

- Cover both the general idea (A) and Inkling's full anatomy (B).
- Teach stage by stage, with story, intuition and math at each stage.
- Text and WebGL move in sync.
- Support all three navigation modes.
- Inkling 2 Mini is the worked example.

**My calls, which the learner has not explicitly approved:**

- **Numbers in Article A come from a tiny looped model that runs live in the browser.**
  - The equations show actual vectors flowing, and the reader can change the loop count and watch the result change.
  - The task is pointer chasing (k-hop lookup). It mirrors Thinking Machines' N-hop result, and the intuition is clean: each pass follows one more hop.
  - It is trained offline with a seeded, committed PyTorch script, and the weights are shipped as JSON.
- **What the toy implements.** One seeded script trains two variants on the same task:
  - **Carry-state loop:** `s_t = R(e, s_{t−1})`, Huginn-style.
  - **Inkling-style loop:** each pass restarts from the prelude output, one message per core layer, built from the previous pass's states with a learned query, enters through the same `(1 − tanh(W_f z)) ⊙ h + tanh(W_m z) ⊙ m` gate.

  Both are deliberately simplified: plain residuals instead of attention residuals, a dense MLP instead of the mixture of experts, no short convolutions, and one or two core layers.
  - **Which beats it feeds.** The variants power beat 6 (one hop per pass) and beat 7 (side by side: carry vs restart-plus-message). The Inkling-style variant's real gate activations drive beat 8's dimmers.
  - **Fallbacks.**
    - If the Inkling-style variant learns the task but not a clean one-hop-per-pass pattern, show what it actually learned and say so.
    - If neither variant learns the task, use hand-constructed weights for the carry-state loop only, label them as such, and show beat 8's gates without numbers.
    - No fallback may present invented numbers as learned.
- **Labels.** Where toy vectors flow, the scene and the equations carry the toy's sizes, for example `[T, 16]`. A secondary label gives the matching Inkling size, for example "in Inkling: `[T, 2304]`". Everywhere else, labels use Inkling's real config sizes.
- **The real model runs now, in BF16 (decided by the learner on November 26: “yes convert to bf16 … run whatever you can run here”).**
  - No official BF16 checkpoint exists; the community copies are GGUF or MLX quantisations, or modified fine-tunes.
  - We download the official checkpoint at its pinned revision. Transformers at the pinned commit dequantises the NVFP4 and MXFP8 experts while loading (`Inkling2LoopDequantizeExperts` in `core_model_loading.py`), so loading with `dtype=torch.bfloat16` on the M4 Max CPU (128 GB) gives a BF16 model in memory.
  - **What we capture** for one short prefill and one cached decode:
    - top next tokens;
    - per-visit hidden-state norms;
    - attention-residual weights over block states;
    - message-attention weights over the previous pass;
    - forget and message gate values for each core layer and pass;
    - router choices per pass, including whether a token picks the same experts in passes 1, 2 and 3;
    - KV and convolution cache sizes per visit;
    - wall time per pass.

    These fill B's "in the recorded run" panels and give A's cost beats measured numbers.
  - **Prompt and other loop counts.** The prompt is a short multi-hop question, so the passes have something to do. Running Mini at 1 or 2 passes is outside what it was trained for. If we show it, we label it as such. The variable-loop research model is the right tool for comparing loop counts, and it is a possible second download.
  - **Where files go.** Large files (checkpoint, environment, raw captures) stay in the ignored `.local/inkling2/`. A small summarised capture JSON is committed with the topic.
  - **If the capture fails,** the articles fall back to the toy model and config shapes. Record the failure in the notes.
  - **Capture result (November 26, done).** The model loaded in BF16 in 82 s, with a peak memory footprint of 51 GB. See `topics/looped-transformers/assets/` for the scripts, `inkling-capture.json` and its summary, and `hop-probe.json`. Findings the articles must use:
    - **Each pass mostly picks different experts.** In the same core layer, a token shares about 4.8 of its 8 experts between passes 1 and 2, about 4.3 between passes 2 and 3, and only about 3 across all three. In layer 17, the last core layer, it shares about 0.7 across all three passes (3.1 between passes 1 and 2). For the last prompt token, the three passes together touch 13–20 distinct experts. So the cost beat must **not** say that looping re-reads the same weights. For a single token, the core's expert bytes grow with the number of distinct experts, not with one shared set.
    - **The trained gates are open.** Across layers, the message's norm matches the kept state's norm (`‖g⊙m‖ / ‖(1−f)⊙h‖` is about 1.0–1.4 for the query, key and value inputs, and about 0.6 for the MLP). The mean forget value is negative (down to −0.75 on the MLP input), so `1−f` amplifies the restarted state rather than shrinking it. The "starts closed" story describes training at initialisation, not the released weights.
    - **Each message reads 2 sources from the previous pass:** its middle block boundary and its final state. For example, the first read gives about 0.9 of the weight to the final state.
    - **Time grows with the number of visits.** For a 43-token CPU prefill: prelude 2.1 s, each core pass about 4.3 s, coda 2.1 s. The totals were 7.9, 11.8 and 17.3 s for 1, 2 and 3 passes.
    - **The KV cache grows by 196,608 B per token,** which is 48 visits × 2 × 8 KV heads × 128 × 2 B. The sliding-window visits stop growing at 1024 positions.
    - **Tokenisation quirk:** "attic" is split into a space token followed by "attic", so first-token comparisons mislead. `probe_hops.py` scores whole answers instead.
    - **Hop probe (`hop-probe.json`).** Two container chains, each told in forward and reversed order. Two- and three-hop answers were correct in 2 of 8 prompts at 1 pass, 5 of 8 at 2 passes and 8 of 8 at 3 passes. The right answer's average share among the candidate containers was 0.36, 0.60 and 0.99. At one pass the model usually names the first container.
      - **Caveats to state on the page.** Only 3 passes is the trained setting. The 3-hop ring prompt succeeds at every count via a shortcut: "garage" is the only place-like candidate. This is the real-model counterpart to the toy's one-hop-per-pass beat.
- **The scene has two zoom levels.**
  - *Macro:* the layer stack and loop.
  - *Micro:* the tensors inside one layer, in the spirit of the Qwen tensor view.
- **The signature visual is a morph between the stack *folded* and *unrolled*.**
  - Folded: 24 physical layers, with the core drawn as a loop that returns to its start.
  - Unrolled: 48 visits in a row, where each core visit is tinted and linked to the physical layer whose weights it reuses.
  - Scrolling through the "what does a loop cost" beats morphs one into the other.
- **Terms in the text are linked to scene objects.** Hovering or focusing a term such as *gate*, *m₁* or *pass 2* highlights the object in the scene, and hovering an object highlights the term.
- **Each beat has its own URL hash,** so links can jump straight to it.

## Article A outline — “Why loop a transformer?”

Each line is roughly one beat. The builder may split or merge beats.

1. **Hook.** Chain-of-thought makes a model think by *writing*. What if it could think more *before* writing?
2. **Depth is serial computation.** A transformer's depth limits how many dependent steps it can do per token. Tie this to the learner's Qwen dependency-graph finding: depth is the critical path.
3. **Two ways to add depth: more layers (more parameters) or the same layers again (no new parameters).** Math: `h_{ℓ+1} = h_ℓ + F_ℓ(h_ℓ)` with distinct `F_ℓ`, versus `F_ℓ = F` shared.
4. **Lineage, as a story.** Candidate citations, each to be verified before use:
   - Universal Transformer (2018), with adaptive computation time (ACT) halting;
   - ALBERT's cross-layer weight sharing (2019), which saves parameters, not compute;
   - looped transformers as programmable computers (2023);
   - length generalisation (2024);
   - Huginn's recurrent depth (2025);
   - Ouro (2025);
   - Mixture-of-Recursions, HRM and TRM (2025);
   - iso-depth scaling (2026);
   - Inkling 2 Mini (2026).
5. **Prelude, core, coda.** Why not loop everything? The first layers translate from embeddings and the last layers translate back. This is Thinking Machines' stated intuition, backed by the middle-vs-last result.
6. **The toy model, pass by pass.** One hop per pass. Change the loop count and see when the answer appears.
7. **Two ways to feed a pass.**
   - Carry the state forward, `s_t = R(e, s_{t−1})`, as in Huginn.
   - Restart from the prelude output and pass a message, as Inkling does.
   - Why restarting plus a gate is stable: the gate starts closed.
8. **The message gate math,** with the scene showing the forget and message gates as dimmers.
9. **Stability.** Why naive loops blow up; the dynamical-systems view, pointing to Parcae.
10. **Costs, beat by beat, with the folded/unrolled morph:**
    - parameters stored: 24 layers;
    - weights read per token: 5.4B, versus 3.4B for one pass;
    - KV cache: grows like 48 layers, since every visit keeps its own;
    - FLOPs: 48 layer visits;
    - serial depth: 48 dependent steps.

    Key point: looping saves *storage*, not *bandwidth* or *latency*. The capture shows that routed experts mostly differ from pass to pass, so in this mixture of experts the passes often don't even read the same expert weights. Use the measured overlap.
11. **What loops buy.** The varloop results as a small chart; fewer reasoning tokens; knowledge stays flat (SimpleQA). Looping helps multi-step computation, not recall.
12. **Monitorability.** "Think more, write less" means less reasoning is visible as text. Thinking Machines sees no meaningful reduction at this scale, and Mira Murati says it "may not stay the case."
13. **Bridge to Article B.**

## Article B outline — “Inside Inkling 2 Mini”

One chapter per component. Each has story, intuition and math (with config shapes) and a link to the exact source lines at the pinned commit.

1. **Map.** 4 blocks × 6 layers, sliding vs full layers, the loop region, and 48 visits.
2. **Tokens to vectors.** Vocabulary 201,024, width 2304.
3. **Attention residuals.** The depth-wise attention that builds each sublayer's input from block states. Why it starts as a uniform average. How it differs from a plain residual stream. Cite the Kimi paper.
4. **A sliding-window layer:**
   - input norm;
   - query, key and value projections (24 query heads, 8 KV heads, head dim 128, grouped-query attention 3:1);
   - learned relative-position logits;
   - query/key L2 clipping and log scaling;
   - window 1024;
   - learned sink;
   - gated output projection;
   - post-attention short convolution (kernel 4);
   - residual, or a block-boundary append.
5. **A full-attention layer.** What changes: separate query, key and value input norms; dynamic key/value short convolutions; no window.
6. **The mixture of experts:**
   - router: top 8 of 128, normalised and scaled by 8, with a gate bias;
   - softcapped SwiGLU experts of size 1152;
   - post-MLP short convolution;
   - the history of pruning from 384 experts to 128 during training.
7. **Prelude (layers 0–5).**
8. **The loop, in full.** Pass 1, then passes 2–3: restart, `message_input`, the four message gates per layer, and separate caches. Link back to A's beats 7–8.
9. **Coda, the final depth-attention, the norm, the LM head and the logit softcap of 30.**
10. **Memory and bytes.**
    - KV and convolution state per visit: the sliding-window visits are capped at 1024 positions, while the 8 full-attention visits grow with the context.
    - Precision: NVFP4 experts and MXFP8 for the last four layers.
    - Compute every number from the source and config. Don't trust this design's arithmetic.
11. **Speculative-decoding drafter.** It has 3 layers and reads visits 5, 23, 41 and the final state (48). Brief.
12. **Vision.** One sentence, out of scope.

## The engine

- **The scene state** is a pure function `render(beat, progress)`. Each beat declares:
  - which objects are lit;
  - the camera pose;
  - the macro or micro level;
  - fold amount (0 = folded, 1 = unrolled);
  - which tensor or vector flows, and where;
  - the highlighted term IDs.

  Between beats, values are interpolated. Beats are data (a JS or JSON array), separate from rendering, so prose and beats can be edited independently.
- **The page layout.** On desktop, the article is on the left and the scene is pinned on the right. On a phone, the scene is pinned above the text at about 40% of the screen height. A mode switch (Scroll / Step / Play) and Prev/Next sit with the scene.
- **Drivers.** Scroll, Step and Play, as above. Use `prefers-reduced-motion` to replace tweens with cuts.
- **Math rendering.** Equations show real shapes, for example `[T, 2304]`. Use KaTeX from a pinned CDN, or render to static HTML/MathML at build time; the builder decides.
- **Fallback.** Without WebGL, each beat shows a static image or SVG of its scene, so the article still reads as an illustrated blog post.
- **Reuse.** Reuse or extend `topics/qwen-forward-pass/walkthrough/tensor-view.js` and `claude-graph/graph3d.js` where they fit. The builder decides how much.

## Site integration (per `CODEX.md`)

- **New topic** `topics/looped-transformers/`, containing:
  - `README.md`, stating the learning goal and linking to both articles;
  - `assets/`, holding the toy-model script and weights, the pinned source excerpts and `facts.json`;
  - a landing `index.html` that leads to both articles.
- **Links:** from `topics/README.md`, from the root `index.html` in the same story style as the Qwen topic, and from the preference entry dated November 26.
- **`notes.md`:** what was built, sources, limits, and feedback marked **pending**. The feedback question should be grounded in a stage. For example: “At the message-gate beat, could you predict what pass 3 receives before the text said it? Which mode did you end up using: scroll, step or play?”
- **Publish** under GitHub Pages. Test one learner action through the Pages URL, and check links under the repository prefix.

## How we know it works

- **Facts.** A `facts.json` maps every number and quoted claim in the prose to its source: URL plus section, or file plus line at the pinned commit. A build or check script fails if prose cites a fact ID that has no source, or if linked source lines no longer match (the pattern of the Qwen `walkthrough/build.cjs`).
- **Toy model.** The Python reference and the JavaScript implementation agree within 1e-5 on saved test vectors. The page's loop-count control reproduces the offline accuracy-versus-passes table.
- **Sync.** Automated browser tests (playwright-core, as used before) check three things:
  - scrolling to each beat puts the scene in that beat's state;
  - Step and Play keep the text and the scene on the same beat;
  - scrolling back rewinds, and opening a hash deep link restores the state.
- **Look.** Headless Chrome screenshots of every beat, in light and dark mode, at desktop and phone width, are reviewed by eye.
- **Links.** All local links resolve.

## Left to the builder

- **Beat granularity and the exact prose,** within the outlines above.
- **Rendering stack:** plain WebGL2 or a pinned three.js; KaTeX or MathML.
- **The toy model:** its exact size and task details.
- **How much to reuse** of the Qwen tensor view and the 3D graph code.
- **Visual styling,** within the repo's existing look and theme tokens.
- **The order of building A and B.** Suggestion: the engine and Article A first, since A defines the engine's needs, then B.
