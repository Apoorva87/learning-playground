# Learning playground

This is my workbench for trying repositories, skills, agents, and other tools on topics I want to learn. The question after each trial is **what helped me learn this particular thing, and which parts would I use again?**

Open the [workspace index](index.html) in a browser, or use these links from Git.

| Where | What belongs there |
| --- | --- |
| [Topics](topics/README.md) | One folder per topic, with its source material, examples, tool outputs, and my reactions. |
| [Tools](tools/README.md) | One record per tool, with its upstream source, tested version, setup recipe, and links to trials. |
| [Preferences](preferences.md) | My decisions by use case, including parts worth borrowing and combinations worth repeating. |
| [Agent instructions](CODEX.md) | How to run a trial with me and keep the records current. |

## Who keeps this organized

The agent working in this repository is responsible for maintaining the index, tool recipes, topic folders, trial outputs, and feedback log. After each trial, the agent asks what helped, records the learner’s judgment in context, and uses that history to help the learner decide which tools or components to try again. A tool can be useful for one learning task and a poor fit for another.

Use English only for communication, lessons, interface labels, and trial records.

**First trial** · [How Transformers Remember with OpenMAIC](topics/how-transformers-remember/README.md).

## Start a session

Tell the agent the tool or repository and what you want to learn. A request might be “Try these two tutors on the same probability problem. I want to see which one helps me explain my reasoning.” The agent inspects each tool, records its setup, and works through the example with you. Open the outputs, tell it what helped or got in the way, and it records your feedback alongside the work. You can revisit a decision when a different example changes your mind.

## Rebuild the tools

From a checkout of this project, run the following in a terminal with a POSIX shell.

```sh
./install.sh
```

This runs every registered tool recipe. To install a single tool, pass its directory name, as in `./install.sh tool-name`. Each tool record lists any prerequisites and the command to verify the installation.

Git keeps the installation steps and the learning material. Downloaded repositories and installed environments live under the ignored `.local/tools/` directory. Save outputs you want to keep inside the corresponding topic.
