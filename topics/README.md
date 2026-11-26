# Topics

- [Inside one Qwen forward pass](qwen-forward-pass/index.html) — Trace a real 27B model and explore its workloads, storage and data dependencies with five tools.
- [How Transformers Remember](how-transformers-remember/README.md) — What is a KV cache, and how did we get from multi-head attention to DeepSeek‑V4.1‑Flash’s 890 bytes of global cache per token?
- [Inside Prism · From screen to Swift](inside-prism/README.md) — Which Swift creates each visible Apple TV element, and what happens when you use the remote?

## Inside a topic

```text
<topic>/
  README.md                         Learning goal and links to cases
  assets/                           Shared readings, data, images, or other inputs
  cases/<case>/
    brief.md                        Question, prompt, inputs, and desired outcome
    runs/<date>-<tool>/              Notes and outputs from one attempt
```

Keep a copy of the exact prompt in the brief and link to the inputs in the topic’s assets. Give repeat attempts distinct folders, for example by appending `-2`. Link the runs in the topic README so the same example can be reviewed across tools.

In each run’s `notes.md`, record the tool and version, how it was invoked, any input changes, and links to the output. Keep the agent’s observations and the learner’s feedback under separate headings. Mark feedback as pending until the learner responds.

[Workspace guide](../README.md) · [Tool catalog](../tools/README.md) · [Preferences](../preferences.md)
