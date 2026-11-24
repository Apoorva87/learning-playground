# OpenMAIC

OpenMAIC builds an interactive classroom from a topic or source materials. For our first trial we used its published scene generator to create an interactive lesson with a local model. The full web application also supports teacher and student agents, classroom chat, and course playback. [Upstream repository](https://github.com/THU-MAIC/OpenMAIC) · [Pinned SDK documentation](https://github.com/THU-MAIC/OpenMAIC/tree/c09bb0e2a9f32c2ce330817767eda4697c635351/packages/@openmaic/generation)

| Component | Trial setup |
| --- | --- |
| Source | OpenMAIC 1.5.0, commit `c09bb0e2a9f32c2ce330817767eda4697c635351` |
| Lesson generator | `@openmaic/generation@0.3.34`, with its npm lockfile |
| Model | Local Ollama, `qwen3.5:35b`, model ID prefix `3460ffeede54` |
| Trial | [How Transformers Remember](../../topics/how-transformers-remember/README.md) · [Run notes](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-23-openmaic/notes.md) |

## Install and replay a scene

The recipe needs Git, Node.js 22.19 or newer, and npm. Run `./install.sh openmaic` from the playground root. It retrieves the pinned source and installs the scene SDK under `.local/tools/openmaic/`. The saved lesson opens directly in a browser. Regenerating it also needs curl, a running Ollama service, and the model listed above.

From the project root:

```sh
run=topics/how-transformers-remember/cases/one-more-token/runs/2026-11-23-openmaic
scratch=.local/tools/openmaic
node tools/openmaic/scene.mjs request "$run/scene-outline.json" > "$scratch/request.json"
curl --fail-with-body --silent --show-error -H 'Content-Type: application/json' \
  --data-binary @"$scratch/request.json" http://localhost:11434/api/chat -o "$scratch/reply.json"
node tools/openmaic/scene.mjs render "$run/scene-outline.json" "$scratch/reply.json" > "$scratch/replay.html"
```

The first command asks the SDK to build its simulation prompts. The last feeds the model reply to the same SDK for HTML extraction, JavaScript syntax checks, and post-processing. Replay writes a fresh file beside the installation, so the original trial stays available for comparison. To recreate the quiz, use the same three commands with `quiz-outline.json` as the outline and `replay-quiz.json` as the final output. The SDK returns structured questions, options, answers, and explanations.

## Full classroom with Docker Desktop

For live classroom chat and the multi-agent experience, use the upstream app and PostgreSQL. The supplied model configuration points to Ollama on the Docker Desktop host. After running the installer and making `qwen3.5:35b` available to Ollama, run the following from the playground root. Ensure Ollama accepts connections from Docker Desktop.

```sh
source_dir=.local/tools/openmaic/source
touch "$source_dir/.env.local"
export OPENMAIC_PORT=4319
export OPENMAIC_MODEL_CONFIG="$PWD/tools/openmaic/openmaic.yml"
docker compose -p learning-openmaic -f "$source_dir/docker-compose.yml" \
  -f tools/openmaic/compose.yml up -d --build
curl --fail http://localhost:4319/api/health
```

Open the classroom at `http://localhost:4319`. The app and database use this project’s named Docker volumes. The first trial used the SDK because the session could not start this stack. [Upstream deployment instructions](https://github.com/THU-MAIC/OpenMAIC/tree/c09bb0e2a9f32c2ce330817767eda4697c635351#docker-deployment)
