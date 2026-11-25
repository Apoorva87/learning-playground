# Socratic Workshop on one more token

**November 24, 2026 · Opening diagnostic · Awaiting the learner’s first answer**

Open the [first tutor turn](opening.html) and answer its question in the Codex conversation. If you reopen this case in a fresh chat at the playground root, paste the prompt below and add your answer.

```text
Use .local/tools/socratic-workshop/source/SKILL.md to continue the One more token case in English. Resume from topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-socratic-workshop/state.txt. Ask one question at a time. My answer to the opening question is …
```

## What ran

The active Codex agent read Socratic Workshop at `b74b71c2d4476cdbaa76cabd8f2d113cc171b6a8` and followed its session opening. Inputs were the existing English teaching task, including the fixed 32-layer numerical example, and the shared source notes. The tutor took the spare-key sentence from the guide’s “Predict one more token” section. The only change to the teaching task was the delivery format. The learner gets one question per turn and supplies the explanation first. [Shared case](../../brief.md) · [Field guide](../../../../assets/field-guide.md) · [Pinned skill](https://github.com/smashburger-dev/socratic-workshop/blob/b74b71c2d4476cdbaa76cabd8f2d113cc171b6a8/SKILL.md)

The tutor proposed a target, introduced the diagnostic, named the trial folder as the session location, and asked for an explanation of one decoding step. The workspace added the offline [copy of the opening](opening.html) and a compact [resume block](state.txt). The opening contains no answer key.

## Agent observations

The first question gives the learner room to explain what persists between tokens. Their words will determine which idea the next probe should test. The tutor needs that answer before it can place subskills or show the first map. Hints, Workbench figures, delayed recall, final assessment, and the later comparisons with DeepSeek are still untested in this run.

The project installer was run twice. Its local registration resolved to the pinned skill. An additional clean installation and repeat were checked using the downloaded checkout as a local Git source because the sandbox blocked a second network fetch from inside the shell recipe. Independent discovery in a fresh Codex CLI process was unavailable under sandbox permissions. [Setup and replay](../../../../../../tools/socratic-workshop/README.md)

## Learner feedback

Pending. After the first follow-up, ask whether explaining the decoding step unaided helped expose a gap, and whether the learner would want this question before or after the OpenMAIC lesson or Understand‑Anything graph.
