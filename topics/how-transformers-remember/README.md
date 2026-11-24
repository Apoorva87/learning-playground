# How Transformers Remember

What does a language model remember between tokens, and how did DeepSeek‑V4.1‑Flash make that memory so small?

Start with [the worked guide](assets/field-guide.md). Change one query by hand to see the same cache produce a different answer. Then count the bytes in a toy transformer and follow the design changes all the way to DeepSeek’s reported 890 bytes per token.

## One more token

| Learning material | What to do |
| --- | --- |
| [OpenMAIC interactive lesson](cases/one-more-token/runs/2026-11-23-openmaic/lesson.html) | Read the worked examples, change the calculator inputs, and answer four questions. |
| [Shared case](cases/one-more-token/brief.md) | Reuse this exact prompt and numerical example with the next tool. |
| [OpenMAIC trial notes](cases/one-more-token/runs/2026-11-23-openmaic/notes.md) | See how the lesson was generated and where to record feedback. |
