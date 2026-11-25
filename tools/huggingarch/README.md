# HuggingArch

Trace a model’s architecture, calculate the cache needed at a chosen context and inspect the tensors behind the total. Start with the [illustrated tool guide](index.html), then [compare Llama with DeepSeek](../../topics/how-transformers-remember/cases/huggingarch-cache/runs/2026-11-24-huggingarch/result.html).

## Source and setup

Version 0.2.0 at commit `2c31e91ebcc83ee0f32d5ba26e4e13d14ee24f1a`. The tested Python version was 3.13.12. The package accepts Python 3.13 and 3.14. The workspace recipe requests 3.13 on a fresh installation. Git, uv, Node.js and npm are required. First-time setup downloads the checkout, Python packages and web dependencies. The two models in the trial ship with local configuration, checkpoint metadata and specs. [Pinned repository](https://github.com/shenh10/HuggingArch/tree/2c31e91ebcc83ee0f32d5ba26e4e13d14ee24f1a) · [Author’s English article](https://shenhan.cc/blog/huggingarch.html)

Run these commands from the workspace root. The installer reuses a matching checkout and invokes the upstream Python and npm lockfiles. The launcher binds to the loopback interface. Open the Inspector in a browser on that computer after starting the app.

```sh
./install.sh huggingarch
./tools/huggingarch/start.sh
```

Local address `http://127.0.0.1:8016/inspector.html`. Set `HUGGINGARCH_PORT` to choose another port. For access without installation, open the [hosted Inspector](https://www.huggingarch.com/inspector.html).

## Repeat the CLI example

From the workspace root, these two official CLI invocations write full JSON analyses inside the ignored tool directory. The upstream CLI prints some translated terminal labels, so the English trial uses the structured output. `HF_HUB_OFFLINE=1` restricts the Hugging Face client to bundled or cached evidence.

```sh
tool=.local/tools/huggingarch
for model in meta-llama/Llama-3.2-1B deepseek-ai/DeepSeek-V4.1-Flash; do
    HF_HUB_OFFLINE=1 "$tool/source/.venv/bin/huggingarch" "$model" \
        --batch-size 1 --seq-len 4096 --dtype bfloat16 --no-trust-remote-code \
        --output "$tool/${model##*/}.json" > "$tool/${model##*/}.log"
done
```

The standard CLI summary includes `composition.totals.kv_per_token_bytes` and `composition.totals.kv_constant_bytes`. The trial’s small Python script also evaluates exact stored bytes at the selected context lengths and retains the engine’s per-node storage rows. Run it to display the English JSON, or redirect it to a new filename to compare with the saved result.

```sh
run=topics/how-transformers-remember/cases/huggingarch-cache/runs/2026-11-24-huggingarch
.local/tools/huggingarch/source/.venv/bin/python "$run/measure.py"
```

## Trial and scope

Both official CLI runs succeeded. We ran the install recipe twice, built the production frontend and verified the Inspector page and Llama `POST /analyze` through an in-process ASGI test client. This host restricted binding a local socket, so the launcher was verified through application startup and the web routes were exercised in-process. Upstream also offers the Gallery, Inference Planner, Fusion view, module Playground, agent-driven spec authoring and Kernel Bench. The trial exercised the architecture and cache evaluator using bundled specs. [Case brief](../../topics/how-transformers-remember/cases/huggingarch-cache/brief.md) · [Run notes](../../topics/how-transformers-remember/cases/huggingarch-cache/runs/2026-11-24-huggingarch/notes.md)

The architecture figure comes unchanged from `docs/assets/architecture.svg` in the pinned repository, © 2026 shenh10, under the MIT license. [Local original figure](assets/architecture.svg) · [Full license](assets/LICENSE.txt)
