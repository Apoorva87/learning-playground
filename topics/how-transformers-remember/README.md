# How Transformers Remember

What does a language model remember between tokens, and how did DeepSeek‑V4.1‑Flash make that memory so small?

Start with [the worked guide](assets/field-guide.md). Change one query by hand to see the same cache produce a different answer. Then count the bytes in a toy transformer and follow the design changes all the way to DeepSeek’s reported 890 bytes per token.

## One more token

| Learning material | What to do |
| --- | --- |
| [OpenMAIC interactive lesson](cases/one-more-token/runs/2026-11-23-openmaic/lesson.html) | Read the worked examples, change the calculator inputs, and answer four questions. |
| [Understand‑Anything knowledge graph](cases/one-more-token/runs/2026-11-24-understand-anything/map.html) | Follow the links from the 890-byte total to the entry size and writer rate, or search the four-stop tour. |
| [Socratic Workshop opening](cases/one-more-token/runs/2026-11-24-socratic-workshop/opening.html) | Explain the prefix in your own words, then continue the live workshop in chat. [Trial notes](cases/one-more-token/runs/2026-11-24-socratic-workshop/notes.md) |
| [DeepTutor lesson](cases/one-more-token/runs/2026-11-24-deeptutor/lesson.html) | Read one tutor response with corrections and compare it with the original answer. [Trial notes](cases/one-more-token/runs/2026-11-24-deeptutor/notes.md) |
| [Visual Explainer](cases/one-more-token/runs/2026-11-24-visual-explainer/one-more-token.html) | Play one attention step, predict a different output, and trace DeepSeek’s byte budget. [Trial notes](cases/one-more-token/runs/2026-11-24-visual-explainer/notes.md) |
| [Understand‑Anything trial notes](cases/one-more-token/runs/2026-11-24-understand-anything/notes.md) | Compare both tools on the same example and open the official dashboard launcher. |
| [Shared case](cases/one-more-token/brief.md) | Reuse this exact prompt and numerical example with the next tool. |
| [OpenMAIC trial notes](cases/one-more-token/runs/2026-11-23-openmaic/notes.md) | See how the lesson was generated and where to record feedback. |

## Inspect self-attention in 3D

LLM Visualization runs a small letter-sorting model locally. Choose **Self Attention** to follow queries, keys and values in the model. [Open local app](http://127.0.0.1:3002/llm) · [Startup instructions](../../tools/llm-viz/README.md) · [Trial notes](cases/attention-walkthrough/runs/2026-11-24-llm-viz/notes.md)

## Compare the models in HuggingArch

Inspect the local CLI’s results for Llama and DeepSeek, and separate the per-token slope from the bounded cache state. [Worked result](cases/huggingarch-cache/runs/2026-11-24-huggingarch/result.html) · [Tool guide](../../tools/huggingarch/index.html)
