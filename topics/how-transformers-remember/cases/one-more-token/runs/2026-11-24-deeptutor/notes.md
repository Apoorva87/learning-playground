# DeepTutor · How Transformers Remember

November 24, 2026. [Open the corrected lesson](lesson.html) · [Read DeepTutor’s original answer](lesson.md) · [Inspect the completion record](receipt.json)

Try the final prediction with the answer closed. At 16,384 tokens, how many global KV bytes would DeepSeek-V4.1-Flash use at the reported average? What else would a server need memory for?

## What ran

One English turn through DeepTutor 1.7.4’s core Chat CLI used Ollama `qwen3.5:35b`. We supplied the exact shared teaching task, followed by a separator and the full field guide. That guide already included worked answers and primary-source links. DeepTutor had no indexed knowledge base or selected tools in this run. [Case brief](../../brief.md) · [Field guide](../../../../assets/field-guide.md) · [Pinned tool and replay instructions](../../../../../../tools/deeptutor/README.md)

The direct `deeptutor run chat` attempt ended with a provider transport error. This session’s Python process could not connect to localhost. The successful attempt ran the same upstream CLI with a file-backed HTTP transport. We sent the CLI’s exact request body to local Ollama using curl, and the original provider parser consumed its response. The agent loop finished in one round with zero tool calls. The final event reported completion, 5,219 prompt tokens and 2,923 completion tokens. An additional model request generated the session title. Buffered delivery makes stream-speed and first-token timings from this run unsuitable for comparison.

The reconstructed tutor message consists of the visible response, plus a final newline for the Markdown file. Its contents also matched the concatenated public `content` events. The receipt uses the prompt as supplied to the CLI, with the terminal newline removed by shell command substitution. The archived answer and receipt contain no internal reasoning.

## Agent observations

- The tutor walked through the same two cached values under two queries. Its outputs of 8 and 6 make it possible to see which parts of attention are reused and which change.
- The MHA, GQA and MQA table correctly gives 2 GiB, 512 MiB and 64 MiB at 4,096 tokens, with all three doubling at 8,192. The 890-byte calculation and million-token unit conversion also match the supplied guide. Several passages closely follow that guide. This turn shows how Chat presented supplied material. Independent retrieval and unaided recall were untested.
- The opening equation accidentally summed over only four positions, even though the tutor described the fifth position attending to itself. The reading edition appends the newest key/value and sums over all five eligible positions.
- The tutor applied the toy model’s 32 layers and 128-number head size to DeepSeek’s explanation. DeepSeek-V4.1-Flash has **40 transformer layers** and four global cache writers. Its main global entries have 512 channels plus a separate index key. Global cache bytes still grow approximately linearly with sequence length, at a smaller per-token cost. [DeepSeek technical report, §§2.3 and 4.2.1](https://arxiv.org/html/2609.19969v1)
- Fixed query-head count lets each query compute its own attention scores. It gives no quality guarantee. The GQA experiments reported quality close to MHA after uptraining. The tutor also guessed 5–10% allocator overhead without evidence for a specific serving system. The reading edition corrects both claims. [GQA paper](https://arxiv.org/abs/2305.13245)
- DeepTutor wrote the Markdown lesson. We added an offline HTML reading edition with visible corrections, typeset equations, source links and three answers that open on request. The web classroom, retrieval, mastery checks, quiz generation and multi-turn adaptation were outside this single-turn trial.

## Learner feedback

**November 24, 2026.** The learner named DeepTutor among their three favorites so far. Did doing the two-query example and then estimating the global cache help you explain what the cache saves? Which part of this tutoring format would you reuse?
