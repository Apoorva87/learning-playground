# Understand‑Anything

The knowledge-base skill turns linked Markdown articles into a graph of topics, claims, entities, and their relationships. The official viewer supports search, selecting a node, and guided tours. [Upstream repository](https://github.com/Egonex-AI/Understand-Anything) · [Pinned skill instructions](https://github.com/Egonex-AI/Understand-Anything/blob/8ec75de1de8c9d2eef685d8d6cbc50782ad31fbb/understand-anything-plugin/skills/understand-knowledge/SKILL.md)

## Set up the trial

Run `./install.sh understand-anything` from the playground root. The recipe keeps source revision `8ec75de1de8c9d2eef685d8d6cbc50782ad31fbb` and the checksum-verified v2.9.8 viewer under `.local/tools/understand-anything/`. It requires Git, curl, tar, Python 3, and Node.js 18 or newer. The analysis scripts require Python 3.10 or newer. The skill can be read directly from the checkout for project-local use.

## How Transformers Remember

Open the [trial notes](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-understand-anything/notes.md) for the graph and comparison with OpenMAIC. On macOS, double-click `Open dashboard.command` in the run folder. From a terminal:

```sh
run=topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-understand-anything
"$run/Open dashboard.command"
```

The official viewer opens a local browser tab and prints its session URL. Keep the terminal open while exploring. The saved graph can be viewed without a model or an API key.

## Replay the analysis

The trial used the upstream parser, one agent following the upstream article-analyzer instructions, and the upstream merger. For an exact replay of the language-model step, reuse the saved batch. These commands write to a temporary copy so the recorded trial is preserved. Set `py` to a Python 3.10+ executable on your machine.

```sh
py=python3.11
skill=.local/tools/understand-anything/source/understand-anything-plugin/skills/understand-knowledge
run=topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-understand-anything
replay=$(mktemp -d)
cp "$run"/wiki/*.md "$replay/"
"$py" "$skill/parse-knowledge-base.py" "$replay"
cp "$run/analysis-batch-1.json" "$replay/.ua/intermediate/"
"$py" "$skill/merge-knowledge-graph.py" "$replay"
cp "$replay/.ua/intermediate/assembled-graph.json" "$replay/.ua/knowledge-graph.json"
node .local/tools/understand-anything/viewer-v2.9.8/bin/viewer.mjs "$replay"
```

## Offline reading view

The workspace also includes a small HTML exporter for the saved graph. It presents linked article text, a searchable node directory, the graph’s tour, and connections in both directions. It runs without dependencies beyond Python 3. Use it to regenerate the trial’s portable reading view.

```sh
python3 tools/understand-anything/export-map.py "$run/wiki/.ua/knowledge-graph.json" "$run/map.html"
```
