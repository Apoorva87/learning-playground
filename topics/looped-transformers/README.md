# Looped transformers, stage by stage

Why would a model run the same layers more than once, and what does each stage compute? The worked example is [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195) from Thinking Machines, released November 20, 2026. It stores 24 layers and runs layers 6–17 three times, which gives 48 layer visits per token.

**Status, November 26, 2026.** Built. Open the [landing page](index.html), or go straight to [One sentence through a modern transformer](transformer-walkthrough.html) (29 beats: the plain machinery with real numbers, plus five “how it changed” beats on norms, positions, attention heads, feed-forward blocks and embeddings, each with a design switch and a computed cost table; hover or click any scene object for its value and arithmetic), [Why loop a transformer?](why-loop.html) (18 beats) or [Inside Inkling 2 Mini](inside-inkling.html) (15 chapters). The text and a WebGL2 scene move together under Scroll, Step and Play. The design is in [docs/superpowers/specs/2026-11-26-looped-transformers-design.md](../../docs/superpowers/specs/2026-11-26-looped-transformers-design.md), and [notes.md](notes.md) lists the limits and the pending feedback question.

## What the real model did

We ran the official checkpoint on an Apple M4 Max CPU. Transformers dequantized its 4-bit NVFP4 and 8-bit MXFP8 experts to BF16 while loading. It took 82 s and peaked at 51 GB of memory.

### Each pass mostly picks different experts

In the same core layer, a prompt token shares on average about 4.8 of its 8 routed experts between passes 1 and 2. It shares about 4.3 between passes 2 and 3, about 3 across all three, and in layer 17, the last core layer, about 0.7 across all three (3.1 between passes 1 and 2). Looping reuses the stored weights, but each pass reads a largely different set of them.

### The trained message gates are open

Passes 2 and 3 restart from the prelude output and receive a gated message from the previous pass. The message is of the same order as the restarted state it is mixed with: the norm ratio `‖g⊙m‖ / ‖(1−f)⊙h‖` is about 0.7 for the query, key and value inputs in the typical core layer (a median), and about 0.6 for the MLP input. Layer 6, the first loop layer, is an outlier at 4.4–7.3, with forget values of 0.81–0.98 that drop most of the restarted state.

### More passes, more hops

The probe scored whole answers for two chains of containers, stated in forward and reversed order. The model was trained with three passes. Its one- and two-pass results are therefore a probe outside its training setting.

| Loop passes | Two- and three-hop prompts answered correctly | Average share of the right answer among the containers |
| --- | --- | --- |
| 1 | 2 of 8 | 0.36 |
| 2 | 5 of 8 | 0.60 |
| 3 (trained) | 8 of 8 | 0.99 |

With one pass, the model usually names the first container in the chain. In the ring story, three hops succeed at every pass count, because the garage is the only location among the candidates.

### Cost

A 43-token CPU prefill took about 2.1 s in the prelude, about 4.3 s for each core pass and 2.1 s in the coda. Every visit keeps its own cache. Keys and values therefore grow by 196,608 bytes per token, the same as a 48-layer model. Sliding-window visits stop growing at 1,024 positions.

## Files

| File | What it holds |
| --- | --- |
| [capture_inkling.py](assets/capture_inkling.py) | Loads the checkpoint in BF16 and hooks the gates, messages, depth-wise attention residuals, router and cache. Records a prefill at 1, 2 and 3 passes and one cached decode. |
| [inkling-capture.json](assets/inkling-capture.json) | Raw recorded values. |
| [inkling-capture-summary.json](assets/inkling-capture-summary.json) | Compact figures for the pages, made by [summarize_capture.py](assets/summarize_capture.py). |
| [probe_hops.py](assets/probe_hops.py) | Scores whole multi-hop answers at each pass count. |
| [hop-probe.json](assets/hop-probe.json), [hop-summary.json](assets/hop-summary.json) | The probe's results, and the table the articles quote (made by [summarize_hops.py](assets/summarize_hops.py)). |
| [toy/](assets/toy/) | The toy looped model: [train_toy.py](assets/toy/train_toy.py), its weights, test vectors, held-out puzzles and accuracy tables. |
| [facts.json](assets/facts.json) | Every number or quotation in the prose, with its source. |
| `engine/`, `toy/toy.js`, `beats/` | The shared scene engine (state, layout, WebGL2 and Canvas 2D renderers, drivers, linked terms, charts, recorded tables), the JavaScript toy model, and the beat data for each article. |
| `engine/narrator.js`, `engine/speech.js` | Voice narration with the browser's speech synthesis: voice and rate pickers, sentence highlight, scene follows the voice. |
| [assets/walkthrough/](assets/walkthrough/) | Article C's recordings and capture scripts: [capture_qwen.py](assets/walkthrough/capture_qwen.py) (Qwen3-0.6B-Base on “The capital of France is”: prefill, then one decode step) and [capture_mla.py](assets/walkthrough/capture_mla.py) (layer 0 of DeepSeek-V2-Lite read by HTTP range requests, about 100 MB instead of 31 GB). |
| `walkthrough/variants.js`, `engine/micro-var.js`, `engine/variants-ui.js`, `engine/inspect.js` | The design-variant arithmetic (parameters, operations, cache and bytes read for MHA/MQA/GQA/MLA, positions, norms, feed-forward blocks, embeddings, all computed from the config), the scenes that draw each design, the switches and tables, and the hover/click/keyboard detail layer. |
| `walkthrough/ref.js`, `engine/micro-walk.js`, `tests/fill-walk.cjs` | Browser-side reference math checked against the recordings, the data-driven scenes for Article C, and the tool that writes the recorded numbers into the page. |
| `tests/` | Unit tests (`node --test`), browser tests (`playwright-core` with system Chrome), and the fact, structure and link checkers. |

## Reproduce

Use Python 3.13 with the [pinned environment](assets/requirements.txt), including Transformers at the commit that added Inkling 2. You need about 20 GB of disk for the checkpoint and 64 GB of memory. From the playground root:

```sh
hf download thinkingmachines/Inkling-2-Mini --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195 \
  --local-dir .local/inkling2/checkpoint
python topics/looped-transformers/assets/capture_inkling.py .local/inkling2/checkpoint .local/inkling2/capture
python topics/looped-transformers/assets/probe_hops.py .local/inkling2/checkpoint .local/inkling2/capture/hop-probe.json
python topics/looped-transformers/assets/summarize_capture.py .local/inkling2/capture/inkling-capture.json summary.json
```

The capture script checks the weights file against its published SHA-256 before loading. The model's code is the Transformers source in [models/inkling2_loop](https://github.com/huggingface/transformers/tree/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/models/inkling2_loop). The checkpoint is Apache 2.0. The weights stay in the ignored `.local/` directory.

## Check everything

```sh
cd topics/looped-transformers/tests && npm install      # playwright-core only
node --test unit/*.test.cjs                              # state, layout, micro scenes, toy parity, facts, arithmetic, charts
node --test --test-concurrency=1 e2e/*.test.cjs          # needs Google Chrome; set CHROME=... for another path
node check-article.cjs ../why-loop.html ../inside-inkling.html
node check-facts.cjs --transformers <site-packages>/transformers ../why-loop.html ../inside-inkling.html
node check-links.cjs --tracked ../index.html ../why-loop.html ../inside-inkling.html
```

Reproduce Article C's recordings with `hf download Qwen/Qwen3-0.6B-Base --local-dir .local/walkthrough/qwen3` then the two capture scripts above, and rewrite the page's numbers with `node tests/fill-walk.cjs transformer-walkthrough.html`.

Retrain the toy with `.local/inkling2/venv/bin/python topics/looped-transformers/assets/toy/train_toy.py` (about 5 minutes on CPU, seeded).

## Feedback

Pending. See [notes.md](notes.md) for the question to answer after reading.
