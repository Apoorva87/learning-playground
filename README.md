# Learning playground

This is my workbench for trying repositories, skills, agents, and other tools on topics I want to learn. The question after each trial is **what helped me learn this particular thing, and which parts would I use again?**

Read the [public learning site](https://apoorva87.github.io/learning-playground/) or open [index.html](index.html) locally. Each tool has a detailed guide with examples, upstream sources, and a record of what we tried.

**Start learning.** [How Transformers Remember](https://apoorva87.github.io/learning-playground/topics/how-transformers-remember/) · [Inside Prism](https://apoorva87.github.io/learning-playground/topics/inside-prism/) · [HuggingArch guide](https://apoorva87.github.io/learning-playground/tools/huggingarch/)

## Use the master skill

Ask your agent to use [learn-with-tools](skills/learn-with-tools/SKILL.md) with a learning goal, your starting knowledge, and any materials or tools you want to use. For example, “Teach me why a KV cache grows. I can follow algebra and I want to see the calculation for a real model.” The skill chooses an approach using the catalog and trial records, checks evidence, and incorporates learner feedback. To make the skill discoverable in Codex, link the skill folder into your skills directory from this checkout:

```sh
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$PWD/skills/learn-with-tools" "${CODEX_HOME:-$HOME/.codex}/skills/learn-with-tools"
```

You can also name the repository skill file directly in a session. If you install a standalone copy, point it to this checkout with `LEARNING_PLAYGROUND_ROOT`.

| Where | What belongs there |
| --- | --- |
| [Topics](topics/README.md) | One folder per topic, with its source material, examples, tool outputs, and my reactions. |
| [Tools](tools/README.md) | One record per tool, with its upstream source, tested version, setup recipe, and links to trials. |
| [Preferences](preferences.md) | My decisions by use case, including parts worth borrowing and combinations worth repeating. |
| [Agent instructions](CODEX.md) | How to run a trial with me and keep the records current. |

## Who keeps this organized

The agent working in this repository is responsible for maintaining the index, tool recipes, topic folders, trial outputs, and feedback log. After each trial, the agent asks what helped, records the learner’s judgment in context, and uses that history to help the learner decide which tools or components to try again. A tool can be useful for one learning task and a poor fit for another.

Use English only for communication, lessons, interface labels, and trial records.

**New code walkthrough.** Inside Prism connects the Apple TV interface to the Swift that draws it. Pick a poster, move the remote focus, or open any of the six tabs. [Screen explorer](topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code/index.html) · [Teachable Code walkthrough](topics/inside-prism/cases/swiftui-elements/runs/2026-11-24-teachable-code/walkthrough.html)

**KV-cache comparison.** Five tools have tried the same KV-cache example. OpenMAIC generated a simulation and quiz. Understand-Anything mapped the notes. Socratic Workshop started a live diagnostic, DeepTutor generated a worked lesson, and Visual Explainer rendered a visual walkthrough. Open any of the five from the [workspace index](index.html) or [topic page](topics/how-transformers-remember/README.md).

**Local app.** LLM Visualization runs the bundled letter-sorting model and includes a Self Attention chapter. Start it with `./tools/llm-viz/start.sh`. [Setup](tools/llm-viz/README.md)

## Rate a tool

Use **Rate or comment** on any tool in the [index](index.html). Save a 1–5 rating, a comment, or both. Each save adds a dated entry in that browser. **Download all feedback** exports the entries as JSON so you can bring them to another session and have the agent add them to the preference log. Favorites named in conversation are recorded in the [preference log](preferences.md).

## Start a session

Tell the agent the tool or repository and what you want to learn. A request might be “Try these two tutors on the same probability problem. I want to see which one helps me explain my reasoning.” The agent inspects each tool, records its setup, and works through the example with you. Open the outputs, tell it what helped or got in the way, and it records your feedback alongside the work. You can revisit a decision when a different example changes your mind.

## Rebuild the tools

From a checkout of this project, run the following in a terminal with a POSIX shell.

```sh
./install.sh
```

This runs every registered tool recipe. To install a single tool, pass its directory name, as in `./install.sh tool-name`. Each tool record lists any prerequisites and the command to verify the installation.

Git keeps the installation steps and the learning material. Downloaded repositories and installed environments live under the ignored `.local/tools/` directory. Save outputs you want to keep inside the corresponding topic.


## Publish learning assets

Commit the tool guide, setup recipe, case brief, trial notes and generated learning assets with each experiment. Keep upstream repositories, dependencies, local services and private working data in the ignored `.local/` directory. Attribute upstream visuals and include their license notices when redistributing them. The trial documentation distinguishes original learning material, tool output and upstream assets.

A push to `main` builds and deploys the public site through GitHub Actions. The build reads tracked files. Standalone lessons and their data keep their paths, and Markdown gets a separate HTML reading view so an original tutor response can coexist with an edited lesson. Browser feedback stays on the visitor’s device until exported and intentionally shared.

To preview what Pages will publish, stage the intended site assets, install `markdown-it-py==3.0.0` in a Python environment, then run:

```sh
python3 scripts/build-site.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory _site
```
