# Understand‑Anything trial · How Transformers Remember

November 24, 2026. [Explore the offline map](map.html) · [Open the 890-byte calculation](map.html#article:global-budget) · [Official dashboard launcher](Open%20dashboard.command) · [OpenMAIC lesson](../2026-11-23-openmaic/lesson.html)

## Try the same case

Start at **What can the next token reuse?** and follow **Read the same cache two ways**. Predict the result for a zero query. Then open **Reconstruct 890 bytes per token**. Follow its links to the entry size and the number of entries written. Try reconstructing the total again from memory. [Shared case and original prompt](../../brief.md)

## Inputs and outputs

The source material is the same [worked guide](../../../../assets/field-guide.md) used for the OpenMAIC case. I adapted it into [ten English wiki articles](wiki/index.md), adding the index, headings, and explicit wiki links the tool needs. The numerical examples and primary references are preserved. The guide and brief remain the shared originals. Upstream papers are linked in the wiki articles.

The pinned knowledge skill parsed ten articles, four index topics, and 23 resolved wiki links. The upstream article-analyzer instructions were applied to one batch by a Codex subagent. It added five entities, ten claims, and eighteen connections. The upstream merger produced 29 nodes, 51 edges, four layers, and four tour stops. The [saved graph](wiki/.ua/knowledge-graph.json) and [agent batch](analysis-batch-1.json) record the run. The [tool record](../../../../../../tools/understand-anything/README.md) includes both tested versions and replay commands.

The tool produced the graph, relationships, and tour. The project’s offline HTML export presents the graph’s article text and relationships with clickable links, a search box, and an article map. The official v2.9.8 viewer’s CLI loaded, but this session denied it permission to bind its local web port. We then bundled the official browser dashboard with the saved graph and verified in Safari that it displayed the KV-cache material.

## Comparison with OpenMAIC

| Same question | OpenMAIC | Understand‑Anything |
| --- | --- | --- |
| What happens if I change the number of heads or tokens? | The calculator updates the memory result as you change inputs. | The dense-budget and head-sharing articles connect the formula with the six fixed examples. |
| Why does V4.1‑Flash report 890 bytes per token? | The worked guide explains it in sequence and a quiz checks the arithmetic. | The global-budget node links directly to the 356-byte entry and the 2.5-entry writer rate. |
| What did the tool contribute? | Its SDK and local Qwen model produced the original calculator and quiz. | Its parser, Codex article analysis, and merger turned a prepared wiki into a searchable graph. |

The host model and preparation differ between the two trials. This comparison describes what each setup produced. For the original English-only preference and OpenMAIC corrections, see [the earlier run](../2026-11-23-openmaic/notes.md).

## Learner feedback

Pending. When explaining 890 bytes per token to someone else, did following the two links help more than reading the walkthrough or answering the quiz? Which format would you keep for a new topic?
