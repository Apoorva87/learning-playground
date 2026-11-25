# DeepTutor

DeepTutor’s Chat can teach from a prompt and continue the conversation in the same session. Its core package also includes Ask Questions, which asks the learner for input. The full workspace adds guided learning with mastery checks, source-linked reading and retrieval, practice questions, books, research, and visualization. The terminal, SDK and web interface share a turn engine. [Pinned upstream overview](https://github.com/HKUDS/DeepTutor/blob/1284116f92368480aef3a47f91b8df9e9b105062/README.md) · [CLI examples](https://github.com/HKUDS/DeepTutor/blob/1284116f92368480aef3a47f91b8df9e9b105062/SKILL.md)

| Trial detail | Value |
| --- | --- |
| Tested source | [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor/tree/1284116f92368480aef3a47f91b8df9e9b105062), core version 1.7.4, commit `1284116f92368480aef3a47f91b8df9e9b105062` |
| Model | Local Ollama `qwen3.5:35b`, model digest prefix `3460ffeede54` |
| Component | One complete English `run chat` turn with a directly supplied source guide |
| Example | [How Transformers Remember](../../topics/how-transformers-remember/README.md) · [Corrected reading edition](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-deeptutor/lesson.html) · [Original tutor answer and trial notes](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-deeptutor/notes.md) |

## Install the tested core

From the playground root, run `./install.sh deeptutor`. The recipe uses Git, uv and Python 3.11 to build DeepTutor’s core wheel from the pinned checkout. Its dependencies are locked. The checkout, environment, settings and sessions stay under `.local/tools/deeptutor/`. Startup checks use `deeptutor doctor --format json`. A fresh runtime selects the local Ollama profile, while a previously selected model is preserved. Replaying the trial requires Ollama running at its normal localhost endpoint with the named model already available. The static reader opens without installation.

## Talk to the tutor or rerun the case

For live tutoring, run these commands from the playground root. The first line selects this trial’s runtime for the current shell.

```sh
export DEEPTUTOR_HOME="$PWD/.local/tools/deeptutor/runtime"
dt_cli="$PWD/.local/tools/deeptutor/environment/.venv/bin/deeptutor"
"$dt_cli" chat --language en
```

For a fresh one-shot lesson, use a new scratch directory. The helper takes the shared teaching task from the case brief and appends the complete source guide. It extracts the public answer after the CLI reports completion.

```sh
scratch=$(mktemp -d "$PWD/.local/tools/deeptutor/replay.XXXXXX")
python3 tools/deeptutor/case.py prompt > "$scratch/prompt.md"
"$dt_cli" run chat "$(cat "$scratch/prompt.md")" --language en --format json > "$scratch/events.ndjson"
python3 tools/deeptutor/case.py answer "$scratch/events.ndjson" > "$scratch/lesson.md"
```

The raw NDJSON stream stays in local scratch and may include model reasoning. The saved trial exports only its final public response and selected completion fields. To continue a live session, pass the `session_id` from its `done` event using `--session`.

### Transport used in this session

The direct Python-to-Ollama connection failed here. A file exchange kept DeepTutor’s actual CLI, prompts, agent loop and provider parsing in use. For the recorded run we posted the exact request bytes to Ollama’s local OpenAI-compatible endpoint using curl, then returned its complete response to the waiting CLI. This included one tutoring request and a separate automatic title request. The file exchange and curl are optional on a host where the normal CLI reaches Ollama.

To repeat this transport, set `DEEPTUTOR_EXCHANGE_DIR` to a fresh empty directory. In the replay command above, substitute `"$PWD/.local/tools/deeptutor/environment/.venv/bin/python" tools/deeptutor/file-exchange.py` for `"$dt_cli"`. In a second terminal, for each numbered request the adapter writes, set `file` to its prefix (for example, `/path/to/exchange/001`) and run:

```sh
curl -sS --fail-with-body -H 'Content-Type: application/json' \
  --data-binary "@$file.request.json" http://localhost:11434/v1/chat/completions \
  -o "$file.downloading" && mv "$file.downloading" "$file.response"
```

## Explore the larger product

The full web app needs additional modules and frontend prerequisites. An indexed knowledge base also needs an embedding provider. The actual CLI trial used core Chat and pasted the supplied guide directly. [Upstream installation options](https://github.com/HKUDS/DeepTutor/blob/1284116f92368480aef3a47f91b8df9e9b105062/docs-for-dev/AGENT_SETUP.md) · [Module guide](https://github.com/HKUDS/DeepTutor/blob/1284116f92368480aef3a47f91b8df9e9b105062/docs-for-dev/MODULES.md)
