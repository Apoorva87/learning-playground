# HuggingArch cache trial · November 24, 2026

[Open the worked result](result.html) · [Inspect the engine data](result.json) · [Exact prompt and inputs](../../brief.md) · [Tool guide](../../../../../../tools/huggingarch/index.html)

## Run

HuggingArch 0.2.0 at `2c31e91ebcc83ee0f32d5ba26e4e13d14ee24f1a`, Python 3.13.12. We ran the official `huggingarch` CLI with `--seq-len 4096 --dtype bfloat16 --no-trust-remote-code` for each model, using the default batch size of one and `HF_HUB_OFFLINE=1`. The model-native cache formats are BF16 for Llama and a mix of FP8, FP4 and declared FP32 state for DeepSeek. The CLI dtype argument alone does not specify all the cache formats. The local checkpoint snapshots and specs supplied all model inputs. Weights, inference requests to a model provider and GPU hardware were unnecessary. [Install and CLI recipe](../../../../../../tools/huggingarch/README.md)

The standard CLI returned `success=true` for both models. `composition.totals.kv_per_token_bytes` gave 32,768 and 890. `composition.totals.kv_constant_bytes` gave 0 and 2,727,936. The modeled checkpoint byte difference was zero against each bundled header. The Llama snapshot records a header obtained from the upstream-listed `unsloth/Llama-3.2-1B` mirror. The full CLI logs are machine-local. The English JSON retains the relevant numbers, both snapshot paths, both header revisions and the engine’s per-node output.

The extraction script calls `analyze_model`, constructs the upstream `Program` and `Deployment`, then asks `cache_view_from_analysis` for `storage_bytes_at(length)` and `storage_rows(4096)`. It cross-checks the growth and bounded totals against the analyzer. Each result covers one sequence on one rank, model backbone only, with native formats. [Rerunnable script](measure.py) · [Selected engine records](result.json)

## Agent observations

At 4,096 positions, Llama stores 67,108,864 bytes of keys and the same number of value bytes. DeepSeek’s 4,096-position rows include 2,703,360 bytes of window data and 24,576 bytes of fixed compressor state. The half-rate storage rules use `ceiling(s_kv/2)`. This makes the first token of each pair count a new slot for each of three half-rate writers. The observed 1,424- and 356-byte increments average to 890 bytes per token. The saved HTML adds an English explanation, a context selector and prediction checks around the extracted records.

The installer was exercised twice. The production frontend built using the pinned web lockfile. The built Inspector page and the Llama `POST /analyze` request returned HTTP 200 via an in-process test client. The response included a block graph, formulas and the expected 32,768 B/token. A local server booted but this host restricted listening on a loopback socket. The hosted UI, new-spec authoring, custom modules, real GPU timings and multi-GPU layouts were not exercised. The browser result opens directly from disk.

## Learner feedback

Pending. Which view helped you predict the second step: counting the writers, reading the per-tensor records, or comparing adjacent totals?
