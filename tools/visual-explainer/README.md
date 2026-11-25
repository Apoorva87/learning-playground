# Visual Explainer

Give an agent a question and the material it should explain. The skill helps the agent choose figures, write a page source, and render it as browser-ready HTML. It also supports slide decks, system diagrams, code and plan reviews, and optional PowerPoint or video export. [Upstream repository at the tested revision](https://github.com/nicobailon/visual-explainer/tree/9d5f880b7e8e0a9a80c890b8e1bb6918b68481ad)

## Use the trial

Open [One more token](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-visual-explainer/one-more-token.html) and press **Play**. The original page and the kit license are kept in this project. [Trial notes](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-visual-explainer/notes.md) · [Kit license](LICENSE)

## Reproduce locally

From the playground root, run `./install.sh visual-explainer` with Git and a modern Node.js runtime (tested with 25.8.1). It checks out version 0.15.0 under `.local/tools/visual-explainer/source` and runs the project's version check. The agent can read `plugins/visual-explainer/SKILL.md` directly from this checkout.

To render the saved source again, run these commands. The renderer writes the sibling HTML file.

```sh
skill=.local/tools/visual-explainer/source/plugins/visual-explainer
run=topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-visual-explainer
node "$skill/kit/render.mjs" "$run/one-more-token.src.html"
```

The render needs no model. To create a different page, ask the agent to follow the skill's `generate-web-diagram` command with your question and sources. The optional automated layout check also needs Playwright Core and a browser it can launch.
