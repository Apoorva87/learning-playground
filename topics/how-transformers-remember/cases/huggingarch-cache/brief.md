# Counting the next token’s cache

**Learning goal.** Distinguish the long-run average number of bytes added per token from the number of bytes added at a particular context length. Locate the tensors that explain the difference.

**Exact trial prompt.** “Using the bundled HuggingArch specifications for meta-llama/Llama-3.2-1B and deepseek-ai/DeepSeek-V4.1-Flash, compare backbone KV storage for one sequence at lengths 4,096, 4,097 and 4,098. Identify the cached tensors responsible for each change and explain how DeepSeek’s reported 890 bytes per token relates to the bounded cache.”

This is a numerical architecture exercise using the existing topic’s DeepSeek example. We chose Llama 3.2 1B as a grouped-query baseline and added the two odd/even transitions so the learner could test the word “average.” The prompt records the task. The engine received model IDs and numeric parameters through the CLI and Python API. [Shared field guide](../../assets/field-guide.md) · [HuggingArch tool guide](../../../../tools/huggingarch/index.html) · [Case result](runs/2026-11-24-huggingarch/result.html) · [Run notes](runs/2026-11-24-huggingarch/notes.md)
