# Looped Transformers Articles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two blog-style articles, "Why loop a transformer?" (A) and "Inside Inkling 2 Mini" (B), whose prose and a shared WebGL2 scene move in sync under scroll, step and play.

**Architecture:** One store holds `(beat, progress)`. Scroll, Step and Play are three writers to it. The scene is a pure function `sceneAt(beats, beat, progress)` that a WebGL2 renderer (with a Canvas 2D fallback) draws. Beats are data files, separate from prose. A tiny looped model trained offline powers A's numbers, and the committed real-model capture powers B's. A facts checker ties every cited number to a source.

**Tech Stack:** Plain WebGL2 and vanilla JS (UMD modules, no bundler), KaTeX from jsdelivr at a pinned version, PyTorch for the toy model (existing pinned `.local/inkling2/venv`), `node:test` for unit tests, `playwright-core` driving system Chrome for browser tests.

**Spec:** `docs/superpowers/specs/2026-11-26-looped-transformers-design.md` (approved November 26, 2026). Executors read it as well as this plan.

## Global Constraints

- English only, in every learner-facing string and every record.
- Static files only, runnable on GitHub Pages. External scripts only from cdnjs, jsdelivr or unpkg, each at an exact pinned version. KaTeX is `katex@0.16.11` from `cdn.jsdelivr.net/npm/`.
- Plain WebGL2, no three.js. Reuse math helpers from `topics/qwen-forward-pass/claude-graph/graph3d.js` (`mul`, `persp`, `look`) by copying them into `engine/gl.js`.
- Light and dark mode via CSS tokens, layout works at 390 px width with no horizontal scroll, controls reachable by keyboard, `prefers-reduced-motion` replaces tweens with cuts.
- One source of truth: exactly one `(beat, progress)` state. Every driver writes it. The scene never reads the DOM.
- Pinned sources: model `thinkingmachines/Inkling-2-Mini` @ `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`; research model `thinkingmachines/TML-d24-varloop-research` @ `91a57fe2b779bbd12c0f74d1f518cbd5b84e8d0d`; Transformers @ `b62646bbda836d467060b7e3d64028202079b77d`; Attention Residuals arXiv 2603.15031.
- Layer arithmetic comes from config, never typed by hand: 24 layers, core layers 6–17, 3 passes, 48 visits. `physicalOf(v)` in `engine/layout.js` is the one mapping.
- Claims that are Thinking Machines' stated intuition must say so. The training-time "gate starts closed" is not what the released weights show (gates are open, see `inkling-capture-summary.json`).
- The learner's feedback stays **pending** until they respond. The learner asked for one commit and push when everything is ready, so only the last task commits and pushes. Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Use plain `git push`.
- Weights, checkpoints and environments stay in the ignored `.local/`.

## Review Focus

Inputs and conditions the spec implies but its tests do not otherwise cover, most likely first:

1. **Deep link while in Step or Play mode.** `…#b-gate` opens at that beat, mode Scroll, scene matching. Pinned in Task 4.
2. **Manual scroll, click or key during Play.** Play pauses and never fights the reader. Pinned in Task 4.
3. **Zero-height or very tall sections, and resize or phone rotation.** Progress never becomes NaN or leaves 0–1, and the scene refits. Pinned in Tasks 1 and 3.
4. **No WebGL, or context lost.** The Canvas 2D fallback draws the same scene and the article still reads. Pinned in Task 3.
5. **`prefers-reduced-motion`.** No tweens, the same final states. Pinned in Tasks 1 and 4.

---

## File Structure

All paths are under `topics/looped-transformers/` unless stated.

| Path | Responsibility |
| --- | --- |
| `engine/state.js` | Pure: `clamp01`, `ease`, `createStore`, `beatFromScroll`, `sceneAt`. UMD. |
| `engine/layout.js` | Pure: `physicalOf`, `macroObjects(fold)`, `microObjects(name)`, `fitCamera`, `project`. UMD. |
| `engine/gl.js`, `engine/canvas2d.js` | Two renderers with one interface `{ draw(frame), resize(), destroy() }`. |
| `engine/scene.js` | Turns a `SceneState` into a `frame` (objects, camera, labels) and drives the renderer and DOM labels. |
| `engine/drivers.js` | Scroll, Step, Play, hash, controls, reduced motion. |
| `engine/terms.js` | Links `.term[data-term]` in the text to scene objects, both ways. |
| `engine/boot.js` | `LoopArticle.start({ beats, mount })`. Exposes `window.LOOP` for tests. |
| `toy/toy.js` | Plain JS inference for the toy model and its trace. |
| `beats/beats-a.js`, `beats/beats-b.js` | Beat data (`window.LOOP_BEATS_A/B`). |
| `why-loop.html`, `inside-inkling.html`, `index.html`, `loop.css` | Articles, landing page and shared styles. |
| `assets/toy/train_toy.py`, `toy-weights.json`, `toy-vectors.json`, `toy-accuracy.json` | Offline training and results. |
| `assets/facts.json` | Every cited number or quote with its source. |
| `tests/` | `package.json` (playwright-core), `unit/*.test.cjs`, `e2e/*.test.cjs`, `check-facts.cjs`, `check-article.cjs`, `check-links.cjs`, `serve.cjs`. |

**Contracts used across tasks**

- **Beat spec** (an element of `window.LOOP_BEATS_*`): `{ id, title, kind: 'math' | 'interlude', scene: { level: 'macro'|'micro', micro?: string, fold: number, pass: 0|1|2|3, camera: { yaw, pitch, dist, tx, ty, tz }, lit: string[], flow: { kind: 'none'|'token'|'message'|'hop', t0?: number }, toy?: object }, terms: { [termId]: string[] } }`.
- **Section DOM:** `<section class="beat" id="b-<id>" data-beat="<id>">` containing `.story`, `.intuition`, `.math` blocks (interludes may omit them). Terms: `<span class="term" data-term="<termId>">`. Facts: `<span data-fact="<factId>">`.
- **Object ids:** `V0…V47` visits, `L0…L23` physical layers, `msg.q.<layer>`/`msg.k.<layer>`/`msg.v.<layer>`/`msg.mlp.<layer>`, `gate.<role>.<layer>`, `tensor.<name>`, `hop.<n>`.
- **`window.LOOP`** (test hook): `{ state(): {beat, progress, mode}, scene(): {beatId, lit: string[], fold, level}, store, drivers, config }`. `?test=1` shortens tweens to 150 ms and Play dwell to 300 ms.
- **Beat anchor:** the viewport line at 40% of its height. `beatFromScroll` uses it.

---

### Task 1: Pure state core

**Files:**
- Create: `engine/state.js`, `tests/package.json` (devDependency `playwright-core` pinned exactly, `"scripts": {"test": "node --test unit/"}`), `tests/unit/state.test.cjs`
- Modify: `/.gitignore` (add `node_modules/`)

**Interfaces:**
- Produces:
  - `clamp01(x: number): number`, `ease(x: number): number` (smoothstep).
  - `beatFromScroll(tops: number[], anchorY: number, docEnd: number): { beat: number, progress: number }`.
  - `createStore(nBeats: number): { get(): {beat, progress}, set(next: {beat, progress}, source: string): void, subscribe(fn: (state, source) => void): () => void }`.
  - `sceneAt(beats: BeatSpec[], beat: number, progress: number, reduced: boolean): SceneState` where `SceneState = { beatId, fold, pass, camera, lit: {[id]: alpha}, flow: {kind, t}, layers: [{level, micro, alpha}] }`. `ARRIVE = 0.35` is exported: numeric fields lerp from `beats[beat-1].scene` to `beats[beat].scene` over `progress ∈ [0, ARRIVE]` with `ease`; `flow.t = clamp01((progress-ARRIVE)/(1-ARRIVE))`; at `progress = 1` the state equals the beat's own spec. `reduced` returns the target immediately for any `progress > 0`, and the previous beat's spec at exactly 0.

- [ ] **Step 1: Write failing tests** in `tests/unit/state.test.cjs`: `beatFromScroll([-100, 300, 900], 400, 1400)` → `{beat: 1, progress: (400-300)/(900-300)}`; the last beat uses `docEnd` as its bottom edge; anchor above the first top → beat 0, progress 0; zero-height section (`tops [0,0,500]`) → progress 0, no NaN; `createStore` clamps beat to `[0,n-1]`, rounds, clamps progress, and does not notify on an identical state; `sceneAt` at progress 0 equals the previous beat's lit set, at 1 equals this beat's, `flow.t` is 0 at `ARRIVE` and 1 at 1, and `reduced=true` at 0.01 equals the target.
- [ ] **Step 2:** Run `cd topics/looped-transformers/tests && node --test unit/`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `engine/state.js`** as a UMD module (`module.exports` in Node, `window.LoopState` in the browser). Lerp numbers by key, take non-numeric fields from the target when `progress > 0`.
- [ ] **Step 4:** Rerun the tests. Expected: PASS.

### Task 2: Layout, camera and projection

**Files:**
- Create: `engine/layout.js`, `tests/unit/layout.test.cjs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `physicalOf(v: number): number`. For `v < 6` return `v`; for `6 ≤ v < 42` return `6 + (v-6) % 12`; otherwise `v - 24`. Also `passOf(v): 0|1|2|3` (0 outside the core, otherwise `floor((v-6)/12)+1`).
  - `macroObjects(fold: number): Obj[]` with `Obj = { id, kind: 'visit'|'ghost', x, y, z, w, h, d, tint: 'prelude'|'core'|'coda', pass, full: boolean }`. 48 visits `V0…V47`. At `fold = 0`, a core visit of pass 2 or 3 sits at its physical layer's position, offset only along z (a translucent stacked ghost). At `fold = 1`, visits lie on a line with increasing `x`. `full` marks layers 5, 11, 17, 23 (full attention).
  - `microObjects(name: string): Obj[]`, defined for each micro scene named in beats (see Tasks 8, 10, 12, 13). Returns `[]` for an unknown name.
  - `fitCamera(objects: Obj[], aspect: number): Camera` (so a narrow or tall canvas still fits everything), `project(camera, aspect, p: [x,y,z]): [sx, sy, depth]`.

- [ ] **Step 1: Write failing tests:** `physicalOf` for v = 0, 5, 6, 17, 18, 29, 30, 41, 42, 47 → 0, 5, 6, 17, 6, 17, 6, 17, 18, 23; `passOf(18) = 2`; `macroObjects(0).length === 48` and the count of distinct `(x,y)` is 24; `macroObjects(1)` has strictly increasing `x` over `V0…V47`; exactly 4 physical layers are `full`; `fitCamera` for aspect 0.5 and aspect 2 both project every object inside the clip box.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3: Implement.** Folded picture: prelude as a vertical stack, core as a ring of 12 slabs seen from the front with a return arrow object `hop.return`, coda continuing upward. Interpolating `fold` morphs the ring into the line.
- [ ] **Step 4:** Run; expect PASS.

### Task 3: Renderers, scene glue, page shell

**Files:**
- Create: `engine/gl.js`, `engine/canvas2d.js`, `engine/scene.js`, `engine/boot.js` (render only for now), `loop.css`, `why-loop.html` (three placeholder beats), `beats/beats-a.js` (the same three beats), `tests/serve.cjs`, `tests/e2e/smoke.test.cjs`

**Interfaces:**
- Consumes: `LoopState.sceneAt`, `LoopLayout.macroObjects/microObjects/fitCamera/project`.
- Produces:
  - Renderer interface `{ draw(frame: {objects: Obj[], alpha: {[id]: number}, camera, flow, size}), resize(w, h), destroy() }`. `gl.js` draws instanced boxes and lines. `canvas2d.js` draws the same projected objects as rectangles and lines.
  - `Scene.create({ canvas, labelLayer, beats, renderer: 'gl'|'2d'|'auto' }): { show(sceneState), getState() }`. `?renderer=2d` forces the fallback. A `webglcontextlost` event switches to the 2D renderer.
  - Labels are DOM elements in `labelLayer`, positioned from `project`, so they stay accessible. Each carries the object's label and the Inkling size where the beat gives one.
  - `tests/serve.cjs`: `serve(rootDir): Promise<{url, close}>` static server on a random port.
  - Page layout: article left, scene pinned right on desktop; scene pinned above the text at 40vh on a phone. Colours come from CSS tokens, with `@media (prefers-color-scheme: dark)` overrides.

- [ ] **Step 1: Write failing e2e smoke test** (`playwright-core`, `executablePath` from `process.env.CHROME` defaulting to `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`, args `--use-angle=swiftshader --enable-unsafe-swiftshader`). Assert: no console errors; the canvas has more than 5% non-background pixels in light and dark (`emulateMedia`); with `?renderer=2d` it is also non-blank; at 390 px width `document.documentElement.scrollWidth <= 390`; after a `resize` to 1400×900 the objects still lie inside the canvas (compare `LOOP.scene()` object bounds).
- [ ] **Step 2:** `cd topics/looped-transformers/tests && npm install && node e2e/smoke.test.cjs`. Expected: FAIL.
- [ ] **Step 3: Implement the renderers and scene glue.** Keep draw state out of the renderers: `scene.js` converts a `SceneState` into a `frame`.
- [ ] **Step 4:** Rerun. Expected: PASS.

### Task 4: Drivers (Scroll, Step, Play), hash, controls

**Files:**
- Create: `engine/drivers.js`, `tests/e2e/sync.test.cjs`
- Modify: `engine/boot.js` (start drivers, expose `window.LOOP`), `why-loop.html` (controls markup)

**Interfaces:**
- Consumes: `createStore`, `beatFromScroll`, `Scene.show`.
- Produces: `Drivers.create({ store, sections: HTMLElement[], beats, config }): { mode, setMode(m), next(), prev(), play(), pause(), goTo(id), destroy() }`.
  - Controls in the DOM: radio group `#mode` with `#mode-scroll`, `#mode-step`, `#mode-play`; buttons `#prev`, `#next`; `<select id="speed">` with 0.5, 1, 2, 4.
  - Scroll mode: the scroll handler writes `beatFromScroll(...)`, throttled with `requestAnimationFrame`. Step/Play ignore scroll events.
  - Step: `next()` scrolls the section to the anchor and tweens progress 0→1 over 2200 ms ÷ speed. ←/→ keys when focus is not in a form control.
  - Play: after each tween it waits `max(2000, words/4 s) ÷ speed`, then moves on, and stops at the last beat.
  - Any `wheel`, `touchstart`, `pointerdown` or `keydown` on the article pauses Play. A mode change to Scroll recomputes the state from the scroll position.
  - Hash: `#b-<id>`. On load, scroll to it. On beat change in any mode, `history.replaceState`.
  - Reduced motion: tweens become cuts.

- [ ] **Step 1: Write failing e2e tests** (`sync.test.cjs`): (a) for every beat, scroll its section to the anchor → `LOOP.state().beat` and `LOOP.scene().beatId` match the section and `LOOP.scene().lit` contains that beat's declared `lit` ids; (b) scroll to the last beat and back to the first → the scene rewinds exactly (compare `scene()` JSON before and after); (c) Step: 6 clicks on `#next` keep the section at the anchor equal to `scene().beatId` after every click; `←` returns; (d) Play at speed 4 with `?test=1`: starts, reaches the next beat on its own, text and scene stay on the same beat at every sample of 100 ms; (e) **Review Focus 2:** a `wheel` event during Play sets `mode` to a paused state and the beat stops changing for 1 s; (f) **Review Focus 1:** open `…#b-<id>` in a fresh page, with and without `?mode=play` stored, → correct beat, `mode === 'scroll'`; (g) switching Step → Scroll with the page scrolled elsewhere adopts the scroll position; (h) **Review Focus 5:** `emulateMedia({reducedMotion:'reduce'})`, `next()` → `progress` reaches 1 within 50 ms.
- [ ] **Step 2:** Run `node tests/e2e/sync.test.cjs`. Expected: FAIL.
- [ ] **Step 3: Implement `engine/drivers.js`.** The tween is a function of `performance.now()` so a test hook can fast-forward.
- [ ] **Step 4:** Rerun. Expected: PASS.

### Task 5: Linked terms, KaTeX, article structure check

**Files:**
- Create: `engine/terms.js`, `tests/check-article.cjs`, `tests/e2e/terms.test.cjs`
- Modify: `engine/boot.js`, `loop.css`, `why-loop.html`

**Interfaces:**
- Produces:
  - `Terms.bind({ root, beats, scene })`: hovering or focusing `.term[data-term=t]` in the current beat sets `Scene.highlight(ids)` using `beat.terms[t]`; hovering an object (pointer over its projected box, or focusing its label) adds `.term-active` to every term with that object. Terms are `tabindex=0`.
  - Math: KaTeX auto-render on `.math`, delimiters `$$…$$` and `\(…\)`. Failures leave the TeX source visible.
  - `check-article.cjs <html>` exits non-zero when: a `kind:'math'` beat lacks `.story`, `.intuition` or `.math`; a `.math` block has no `[T,…]`-style shape annotation (text matching `/\[[^\]]*\d+[^\]]*\]|\[T,/`); a `data-term` has no entry in the beat's `terms`; a `terms` id is not an object id the layout can produce.

- [ ] **Step 1: Write failing tests:** the e2e test hovers a term and expects `scene().highlight` to equal the declared ids, and hovering the object adds `.term-active`. A node test feeds `check-article.cjs` a fixture with a missing `.intuition` block and expects exit code 1, and the valid fixture exit code 0.
- [ ] **Step 2:** Run both; expect FAIL. **Step 3: Implement.** **Step 4:** Rerun; expect PASS.

### Task 6: Toy model, trained offline and run in the browser

**Files:**
- Create: `assets/toy/train_toy.py`, `assets/toy/toy-weights.json`, `assets/toy/toy-vectors.json`, `assets/toy/toy-accuracy.json`, `toy/toy.js`, `tests/unit/toy.test.cjs`

**Interfaces:**
- Task (my call, within "left to the builder"): N = 8 nodes, token ids 0–7. Input is 8 `(src, dst)` pairs in random order, where `dst = f(src)` for a random map `f` with no fixed points, then a query node: 17 tokens. The readout at the last position after pass `t` is trained to equal `f^t(query)` for `t = 1…4` (loss summed over passes, so each pass advances one hop). Sizes: `d = 16`, 2 heads, 1 prelude layer, 2 core layers, readout linear to 8, gate rank 4. Seed fixed, float64 evaluation.
- Two variants in one script, selected by `--variant carry|inkling`: **carry** feeds pass `t`'s final state to pass `t+1` as its input state, `s_t = R(e, s_{t-1})`. **inkling** restarts each pass from the prelude output and, per core layer, forms one message by attention with a learned zero-initialised query over the previous pass's per-layer outputs, then gates it `(1 − tanh(W_f z)) ⊙ h + tanh(W_m z) ⊙ m`, `z = W_down[rms(h); rms(m)]`. No attention residuals, a dense MLP, no convolutions.
- `toy-accuracy.json`: `{ variant: { passes: [1..6], accuracy: [...], target_hop: [...] } }` on 2,000 held-out sequences, with passes 5 and 6 labelled extrapolation. `toy-vectors.json`: 5 fixed sequences per variant with logits, pointer attention and gate values per pass, float64.
- `toy.js` exports `Toy.load(weights)`, `Toy.run(model, variant, tokens, passes): { logits: number[][], argmax: number[], trace: [{pass, hidden: number[][], attn: number[][], gate: {f: number[], g: number[]}|null}] }`.
- **Fallbacks, as the spec says:** if inkling learns the task but not one hop per pass, ship what it learned and say so in `toy-accuracy.json` notes; if neither learns it, ship hand-constructed carry-state weights with `"hand_built": true` and no gate numbers. Never present invented numbers as learned.

- [ ] **Step 1: Write failing test** (`toy.test.cjs`): for each variant and each saved sequence and pass, `Toy.run` logits match `toy-vectors.json` within `1e-5`; the argmax at pass `t` on the saved sequences equals `f^t(query)` for at least the accuracy file's claim; the file's table reproduces by running 200 of the held-out sequences (tolerance ±0.03 of the stored value).
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Train with `.local/inkling2/venv/bin/python topics/looped-transformers/assets/toy/train_toy.py --variant carry` and `--variant inkling`, then implement `toy/toy.js`. Report the final accuracy table in the task summary.
- [ ] **Step 4:** Run; expect PASS.

### Task 7: Facts and links checker

**Files:**
- Create: `assets/facts.json`, `tests/check-facts.cjs`, `tests/check-links.cjs`, `tests/unit/facts.test.cjs`

**Interfaces:**
- `facts.json` is `{ [factId]: { claim, value?, source } }`, where `source` is one of `{ kind:'url', url, section }`, `{ kind:'json', file, path, expected }` (path into a committed JSON file, value compared with tolerance 1e-9), `{ kind:'config', file, path, expected }` (path into `.local/inkling2/checkpoint/config.json`, skipped with a warning when absent), `{ kind:'source', file, line, quote }` (a line in the Transformers file at the pinned commit; verified when `--transformers <dir>` is given, otherwise skipped with a warning).
- `check-facts.cjs [--transformers <dir>] <html…>` fails when an HTML file cites `data-fact` ids that are missing from `facts.json`, when a fact has no source, or when a `json`/`config`/`source` source does not match. It also prints the facts no page uses.
- `check-links.cjs <html…>`: every relative `href`/`src` resolves to a tracked file, and anchor targets exist.

- [ ] **Step 1: Write failing unit tests** with fixtures: missing id, id with no source, wrong `expected`, quote not on the line, good case.
- [ ] **Step 2:** Run; expect FAIL. **Step 3: Implement.** **Step 4:** Rerun; expect PASS.
- [ ] **Step 5: Seed `facts.json`** with every number the spec lists under "What is already known" and "Capture result" (config shapes, 26.4B / 3.4B / 5.4B, 48 visits, varloop results, hop table, overlap 4.8 / 4.3 / 3, gate ratios, timings, 196,608 B per token). Capture figures use `json` sources into `assets/inkling-capture-summary.json` and `assets/hop-probe.json`. Open each URL source and confirm it says what the claim says before adding it.

### Task 8: Micro zoom

**Files:**
- Modify: `engine/layout.js`, `engine/gl.js`, `engine/canvas2d.js`, `engine/scene.js`, `tests/unit/layout.test.cjs`, `tests/e2e/smoke.test.cjs`

**Interfaces:**
- `microObjects(name)` is defined for these names, each with stable `tensor.*` ids so beats can light them: `toy-core` (the toy's tokens, state, message, gate, readout, with its `[T,16]` sizes), `message-gate` (h, m, rms(h), rms(m), z of rank 64, f, g, output), `depth-attn` (block states and the learned query), `sliding-layer`, `full-layer`, `moe` (router, 128 expert cells, the 8 chosen). Each tensor is a grid of cells whose width and height are proportional to its dimensions (capped for large dimensions with a "…" marker), labelled with the shape.
- `Scene.show` renders `layers` with alpha, so a macro↔micro change crossfades over `ARRIVE`.

- [ ] **Step 1: Write failing tests:** every named micro scene returns non-empty objects with unique ids; every `tensor.*` id used in beats resolves; the smoke test renders each micro scene non-blank in both renderers.
- [ ] **Step 2:** Run; expect FAIL. **Step 3: Implement.** **Step 4:** Rerun; expect PASS.

### Task 9: Article A, part 1 — beats 1–5

**Files:**
- Modify: `why-loop.html`, `beats/beats-a.js`, `assets/facts.json`

**Content (one beat each; ids `hook`, `serial-depth`, `two-depths`, `lineage`, `prelude-core-coda`):** the spec's Article A outline items 1–5, as flowing blog prose with `.story`, `.intuition`, `.math`. Required math: `h_{ℓ+1} = h_ℓ + F_ℓ(h_ℓ)` versus shared `F`, with shapes `[T, 2304]`; the critical-path link to the Qwen dependency graph (`../qwen-forward-pass/claude-graph/index.html`). Scene: macro, `fold` 0, prelude/core/coda lit in turn. The lineage list is verified before use: open each paper's page and cite by `facts.json` URL source; drop any item that cannot be verified, and say so in the final report. The middle-versus-last result is stated as "27 of 28 evals", and the intuition is attributed to Thinking Machines.

- [ ] **Step 1:** Write the prose and beats. **Step 2:** `node tests/check-article.cjs why-loop.html && node tests/check-facts.cjs why-loop.html`. Expect exit 0. **Step 3:** `node tests/e2e/sync.test.cjs` still passes with the real beats.

### Task 10: Article A, part 2 — beats 6–9 (the toy and the gate)

**Files:**
- Modify: `why-loop.html`, `beats/beats-a.js`, `engine/boot.js` (toy widget), `engine/scene.js`, `loop.css`, `assets/facts.json`

**Content (ids `toy-hops`, `two-ways`, `gate-math`, `stability`):**
- `toy-hops`: a loop-count control (1–6) re-runs `Toy.run` live. The scene shows the pointer attention landing on the next pair each pass, with `hop.n` objects, and the page prints the accuracy-versus-passes table from `toy-accuracy.json` and marks passes 5–6 as extrapolation. Labels carry `[T,16]` and "in Inkling: `[T, 2304]`".
- `two-ways`: carry-state versus restart-plus-message, side by side, using each variant's trace.
- `gate-math`: `micro: 'message-gate'`, with dimmers for `f` and `g` driven by the inkling variant's real gate values. The text states the trained-from-closed intuition as Thinking Machines' and then shows that the released Mini's measured gates are open (norm ratio 1.0–1.4, cited facts).
- `stability`: why naive loops blow up (spectral radius of the state update above 1), pointing to Parcae. Verify the paper before citing.

- [ ] **Step 1: Write a failing e2e test** in `sync.test.cjs`: changing the loop-count control to 2 changes `LOOP.scene().toy.pass` and the displayed hop; the printed table equals `toy-accuracy.json`. **Step 2:** Run; expect FAIL. **Step 3:** Implement. **Step 4:** Rerun the test, `check-article`, and `check-facts`; expect PASS.

### Task 11: Article A, part 3 — beats 10–13 (costs, benefits, bridge)

**Files:**
- Modify: `why-loop.html`, `beats/beats-a.js`, `assets/facts.json`, `loop.css`

**Content (ids `costs`, `benefits`, `monitorability`, `bridge`):**
- `costs`: scroll-driven morph (`fold` 0→1) while the text counts parameters stored (24 layers), weights read per token (5.4B versus 3.4B), KV cache (196,608 B per token), FLOPs (48 visits) and serial depth (48 steps). It uses the measured expert overlap (about 4.8 of 8 shared between passes 1 and 2), so it does not claim the passes re-read the same experts. It states "saves storage, not bandwidth or latency".
- `benefits`: a chart of the varloop results (DeepSWE, GraphWalks, SimpleQA flat, reasoning tokens) and the real-model hop table (2/8, 5/8, 8/8 with its caveats). Load the `dataviz` skill first, validate the palette with its script, add the hover layer and a table view.
- `monitorability`: Thinking Machines' observation and Murati's "may not stay the case" quote, attributed and cited. `bridge`: links to Article B at matching chapters.

- [ ] **Step 1:** Write them. **Step 2:** `check-article`, `check-facts`, `check-links`, `sync.test.cjs` all pass. **Step 3:** Add an e2e check that scrolling through `costs` takes `LOOP.scene().fold` from 0 to 1 monotonically.

### Task 12: Article B, part 1 — chapters 1–6

**Files:**
- Create: `inside-inkling.html`, `beats/beats-b.js`
- Modify: `assets/facts.json`, `tests/e2e/sync.test.cjs` (run over both articles)

**Content (ids `map`, `embed`, `attn-res`, `sliding-layer`, `full-layer`, `moe`):** the spec's Article B chapters 1–6. Each chapter gives story, intuition and math with config shapes, plus a link to the exact source lines at the pinned commit (a `source` fact). Required details: grouped-query attention 3:1, learned relative-position logits (`d_rel` 16, `rel_extent` 1024), window 1024, sink, output gate, short convolution kernel 4; full-attention layers 5, 11, 17, 23; router top 8 of 128 with `route_scale` 8, softcaps 4 and 25. Use micro scenes `depth-attn`, `sliding-layer`, `full-layer`, `moe`. Read the source before writing each equation, and check shapes with the downloaded source in `.local/inkling2/venv`.

- [ ] **Step 1:** Write the page and beats. **Step 2:** `check-article`, `check-facts --transformers <site-packages/transformers>`, and the sync tests pass for both articles.

### Task 13: Article B, part 2 — chapters 7–12

**Files:**
- Modify: `inside-inkling.html`, `beats/beats-b.js`, `assets/facts.json`

**Content (ids `prelude`, `loop-full`, `coda-head`, `bytes`, `drafter`, `vision`):** the spec's chapters 7–12. `loop-full` includes "In the recorded run" panels from `inkling-capture-summary.json`: per-layer gate ratios, message weights, expert overlap, per-pass timing, and a link back to A's `two-ways` and `gate-math` beats. `bytes` computes every number from config and source (KV and convolution state per visit; sliding-window visits capped at 1024 positions; the 8 full-attention visits grow with context; NVFP4 experts and MXFP8 for the last four layers). A node test recomputes 196,608 B per token from config and compares it with the figure in the prose.

- [ ] **Step 1:** Write them. **Step 2:** `check-article`, `check-facts`, `check-links`, sync tests pass for both articles.

### Task 14: Landing page, links, notes, Pages build

**Files:**
- Create: `index.html` (topic landing), `notes.md`
- Modify: `README.md` (topic), `topics/README.md`, root `index.html`, `preferences.md`, `scripts/build-site.py` only if the build drops the new files

**Content:** The landing page leads to both articles with the story arc and the three modes explained. The root `index.html` entry follows the Qwen topic's story style and links the result with one concrete thing to try. `notes.md` records what was built, sources, limits (toy simplifications, one short prompt, Mini run outside its trained pass count at 1 and 2 passes), and feedback **pending** with the grounded question from the spec. Update the topic README status from "design waiting for approval" to built.

- [ ] **Step 1:** Run `python3 scripts/build-site.py`; confirm `_site/topics/looped-transformers/` holds the pages, engine, beats, toy files and assets, and check the large JSON size is acceptable.
- [ ] **Step 2:** Serve `_site` under a repository-style prefix and run `check-links.cjs` and `sync.test.cjs` against it (base URL as `BASE` env var). Expect PASS.

### Task 15: Look, review, commit and push

**Files:**
- Create: `tests/e2e/screenshots.cjs` (output to `.local/shots/`, not committed)

- [ ] **Step 1:** Screenshot every beat of both articles in light and dark at 1400×900 and 390×844. Read them and fix overlaps, clipped labels and unreadable contrast.
- [ ] **Step 2:** Run the whole suite: `node --test tests/unit/`, `node tests/e2e/smoke.test.cjs`, `node tests/e2e/sync.test.cjs`, `node tests/e2e/terms.test.cjs`, `check-article`, `check-facts`, `check-links`. Report the results as run, and any skipped check with the reason.
- [ ] **Step 3:** Run `git status`, confirm no weights or `node_modules` are staged, then `git add` the topic files, `docs/superpowers/plans/…`, index and preference updates.
- [ ] **Step 4:** Commit with the message `Teach looped transformers with articles that move in sync with a WebGL scene` and the Co-Authored-By trailer, then plain `git push`. Report the commit hash and where to open the pages locally.
