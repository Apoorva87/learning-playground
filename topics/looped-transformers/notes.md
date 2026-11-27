# Looped transformers: notes

Built November 26, 2026. Open [the landing page](index.html).

## What was built

- **Article C, [One sentence through a modern transformer](transformer-walkthrough.html).** Twenty-nine beats (five of them “how it changed”) following “The capital of France is” through Qwen3-0.6B-Base (28 layers, 16 query heads sharing 8 key/value heads, QK-norm, RoPE, SwiGLU, tied embeddings): tokens, embedding, one layer from above, RMSNorm, Q/K/V, GQA, QK-norm, RoPE, scores and softmax, mixing and the output projection, the feed-forward network, the 28-layer stack with a logit-lens view, the head and its 151,936 scores, picking a token and looking it up, the prefill bill, the KV cache, one decode step, why decode is memory-bound, what GQA buys, and multi-head latent attention with its absorbed decode trick on a real DeepSeek-V2-Lite layer.

- **Article A, [Why loop a transformer?](why-loop.html).** Eighteen beats: depth as a chain, two ways to add depth, a short lineage, prelude/core/coda, a toy model that follows one hop per pass (runs live in the page), carry-the-state versus restart-and-message, the message gate on the toy's real vectors, why loops can blow up, five costs (storage, compute, weights read, cache, waiting) while the stack unrolls from 24 layers to 48 visits, what the bill buys (small multiples plus a table), monitorability, and a bridge.
- **Article B, [Inside Inkling 2 Mini](inside-inkling.html).** Fifteen chapters: map, embedding, attention residuals, a sliding-window layer, a full-attention layer, the mixture of experts, prelude, the loop in full, messages, the four gates, the recorded run, coda and head, memory and bytes, the drafter, and one sentence on vision.
- **One engine.** A single `(beat, progress)` state, three drivers (Scroll, Step, Play), a WebGL2 scene with a Canvas 2D fallback, linked terms, a URL hash per beat, `prefers-reduced-motion` support, and light and dark themes.

## Sources and how the numbers were made

- Model and card: `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`. Source: Transformers at `b62646bbda836d467060b7e3d64028202079b77d`, `models/inkling2_loop/`. Every cited line is checked by `tests/check-facts.cjs`.
- The recorded run: the official checkpoint on an Apple M4 Max CPU, dequantised to BF16 by Transformers, one 43-token prompt. See [README.md](README.md).
- The toy: trained offline by `assets/toy/train_toy.py`, ported to JavaScript, and checked to 1e-5 against PyTorch on saved vectors.
- Publication checks run: unit and browser tests, structure, facts and link checkers, and a Pages-prefix build.

## Voice narration

Both articles have a **Narrate** switch, **Voice** and **Rate** pickers and a Pause/Read button under the scene. It uses the browser's built-in `speechSynthesis`, so voices differ by browser and system (on macOS they are the system voices). Each display equation carries a hand-written `data-speak` sentence, and inline math is read by a small TeX-to-words helper (`engine/speech.js`). Tests drive it with a fake `speechSynthesis`, so the real voices were not exercised in headless Chrome. Try it by ear and tell me which beats read badly.

## Article C: sources and limits

- The numbers come from `assets/walkthrough/walkthrough-data.json` (Qwen3-0.6B-Base, float32) and `walkthrough-mla.json` (DeepSeek-V2-Lite layer 0). Layer 0 of Qwen3 was recomputed by hand in float64 and matched the model to 1e-5; the page's own recomputation (`walkthrough/ref.js`) is tested against the recording to 1e-4, and every number in the text is written from the recording by `tests/fill-walk.cjs`.
- Only layer 0 is shown in full. The other 27 layers appear through their hidden-state norms and the logit-lens guesses. One sentence, one model, one head per attention picture.
- MLA is real but separate: layer 0 of a different model (DeepSeek-V2-Lite), read by range requests. Its tokenizer cuts the sentence into the same five pieces with different IDs. Explicit and absorbed scores agree to 2e-7, and the attention weights match the model's to the same precision.
- Counts of parameters, FLOPs and bytes are recomputed from config in unit tests, and the parameter count equals the sum of the checkpoint's tensor sizes. The 500 GB/s in the decode beat is an illustrative figure, not a measurement.
- The five “how it changed” beats compare designs by arithmetic on Qwen3-0.6B’s sizes (and DeepSeek-V2-Lite’s for MLA). Nothing about MHA, MQA, learned positions, post-norm, ReLU or mixtures was trained or run: those rows are computed from the config and labelled so. History claims quote the abstracts or body text of the cited papers (arXiv v1 dates); the cited sentences are in `assets/facts.json`. The operation counts for norms and RoPE are our own tally, stated in the text.
- Hover, click, tap and arrow-key detail exists on every data-driven object of Article C’s scenes. Articles A and B load the same layer but their scenes carry no detail yet.
- Why QK-norm exists, what the feed-forward network stores, and why blanks are strong runners-up for this base model are interpretations, and the text says so.

## Limits

- One prompt, one machine, and a BF16 copy of the quantised model. The gate values, message weights, expert overlap and timings show the mechanism, not a benchmark. The one- and two-pass runs of Mini are outside what it was trained for.
- The hop probe has eight prompts, and one three-hop story succeeds at every pass count through a "garage" shortcut.
- Both toy variants reach essentially 100%, including at passes 5 and 6 which they never trained on, so the toy cannot say which design is better. One hop per pass is by construction of its training target, and which arrow the attention picks is what the toy actually learned.
- The variable-loop research model's numbers come from its model card and are shown beside, not against, Mini's. Thinking Machines warns against comparing them.
- Second-hand: Mira Murati's "may not stay the case", quoted from The Inference. The design's figure "1,731 reasoning tokens at three passes" was not found on any page and is not used. Our own arithmetic (parameters per expert, an extra pass at about 1.08B, 34.5 GB of keys and values at 1,048,576 tokens, the gate machinery at about 28.4M) is labelled as ours and pinned by unit tests.
- Taps and keys on the controls themselves do not take control back from Step or Play, but a click or scroll in the article does.
- Step and Play cut the scroll instead of animating it, so that text and scene always change in the same moment.
- Two corrections made on the way: layer 17 shares about 0.7 of its 8 experts across all three passes (3.1 between passes 1 and 2), not "almost none"; and the gates are open in the released weights, not closed. A review then showed that the first version of that finding (a message about as large as the kept state, 1.0–1.4 times) was a mean pulled up by layer 6: the typical core layer has a median of about 0.7, and layer 6 alone is 4.4–7.3 with forget values of 0.81–0.98. The pages now say both.

## Feedback

**Pending.** Question for the learner, grounded in a stage: at the message-gate beat in Article A, could you predict what pass 3 receives before the text said it? And which mode did you end up using, Scroll, Step or Play?

When feedback arrives, record their words here, add a dated row to [preferences.md](../../preferences.md) naming the use case, the component and the reason, and link this file.
