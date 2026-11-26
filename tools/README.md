# Tool catalog

- [OpenMAIC](openmaic/README.md) — Trying its interactive lesson generator on transformer KV caches.
- [Understand‑Anything](understand-anything/README.md) — Turning the same KV-cache material into a linked knowledge graph and guided tour.
- [Socratic Workshop](socratic-workshop/README.md) asks one diagnostic question at a time and adapts to the learner’s reply. We started a source-based KV-cache workshop.
- [DeepTutor](deeptutor/README.md) offers tutor chat and optional study modes. We ran its chat command and reviewed its generated KV-cache lesson.
- [Visual Explainer](visual-explainer/README.md) turns an agent’s figure source into a browser page. We rendered an attention replay, cache-size charts, and a byte breakdown.
- [Teachable Code](teachable-code/README.md) connects a guided diagram to verified code excerpts. We used its app lesson on Prism’s SwiftUI interface.
- [LLM Visualization](llm-viz/README.md) runs a small transformer in the browser. We started it locally and explored its Self Attention chapter.
- [HuggingArch](huggingarch/index.html) computes architecture costs from a model specification. We measured Llama and DeepSeek KV growth with its local engine.

- [PyTorch Profiler](pytorch-profiler/index.html) — Capture operators, their nested durations and tensor shapes in a real forward pass. [Qwen study](../topics/qwen-forward-pass/pytorch-profiler/index.html)
- [TorchLens](torchlens/index.html) — Trace a forward call to inspect which tensor operations produced each value. [Qwen study](../topics/qwen-forward-pass/torchlens/index.html)
- [DaCe](dace/index.html) — Lower numerical Python into a dataflow graph and inspect the array regions and parallel maps. [Qwen study](../topics/qwen-forward-pass/dace/index.html)
- [TorchInductor](torchinductor/index.html) — Compile tensor operations and compare the captured graph with the buffers and kernels it generates. [Qwen study](../topics/qwen-forward-pass/torchinductor/index.html)
- [MLCommons Chakra](chakra/index.html) — Link execution events and inspect their control, timing and tensor relationships. [Qwen study](../topics/qwen-forward-pass/chakra/index.html)

## Register a tool

Create `tools/<tool>/README.md` with the upstream link, what the tool does, the tested commit or package version, prerequisites, and the command used to verify it. Link back to the topic runs that used this version.

Put the setup commands in `tools/<tool>/install.sh`. Use a lowercase directory name made of letters, digits, hyphens, or underscores, starting with a letter or digit. The root installer discovers these recipes automatically and runs each with `sh` from the recipe directory. Each recipe receives two absolute paths.

| Variable | Purpose |
| --- | --- |
| `TOOL_HOME` | An existing, Git-ignored directory at `.local/tools/<tool>/`. Keep source checkouts and local installs here. |
| `PLAYGROUND_ROOT` | The root of this workspace, for finding other project files when needed. |

Use `set -eu`, quote paths, and pin the commit or package version used in the trial. Include the steps needed to make the tool usable, such as installing dependencies or registering a skill with its host. Document any interaction or credentials the host requires. Repeated runs should reuse matching installations and preserve local edits.

Test a new recipe with `./install.sh <tool>` from the project root, then check the same command a second time. To recreate all registered tools, run `./install.sh`.

[Workspace guide](../README.md) · [Topics](../topics/README.md) · [Preferences](../preferences.md)
