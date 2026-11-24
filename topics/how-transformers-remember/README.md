# How Transformers Remember

What does a language model remember between tokens, and how did DeepSeek‑V4.1‑Flash make that memory so small?

Start with [the worked guide](assets/field-guide.md). Change one query by hand to see the same cache produce a different answer. Then count the bytes in a toy transformer and follow the design changes all the way to DeepSeek’s reported 890 bytes per token.

## One more token

| Learning material | What to do |
| --- | --- |
| [OpenMAIC interactive lesson](cases/one-more-token/runs/2026-11-23-openmaic/lesson.html) | Read the worked examples, change the calculator inputs, and answer four questions. |
| [Understand‑Anything knowledge graph](cases/one-more-token/runs/2026-11-24-understand-anything/map.html) | Follow the links from the 890-byte total to the entry size and writer rate, or search the four-stop tour. |
| [Understand‑Anything trial notes](cases/one-more-token/runs/2026-11-24-understand-anything/notes.md) | Compare both tools on the same example and open the official dashboard launcher. |
| [Shared case](cases/one-more-token/brief.md) | Reuse this exact prompt and numerical example with the next tool. |
| [OpenMAIC trial notes](cases/one-more-token/runs/2026-11-23-openmaic/notes.md) | See how the lesson was generated and where to record feedback. |
