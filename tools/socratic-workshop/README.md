# Socratic Workshop

The learner explains an idea in their own words. The tutor asks one question at a time and uses the answers to select follow-ups. After a short diagnostic, it shows a skill map. Later, independent answers, transfer to a new example, and connections between ideas can raise the learner’s level. The tutor can give a small hint, build a figure, or save a session for another day. [Upstream repository](https://github.com/smashburger-dev/socratic-workshop) · [Tested skill instructions](https://github.com/smashburger-dev/socratic-workshop/blob/b74b71c2d4476cdbaa76cabd8f2d113cc171b6a8/SKILL.md)

## Try it on one token

Read the [opening question](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-socratic-workshop/opening.html) and answer it in the Codex conversation. The HTML preserves the first turn for offline reading. Codex runs the live tutor. The [trial notes](../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-socratic-workshop/notes.md) link the materials and show how to continue.

## Local setup

Run this from the playground root with Git and a POSIX shell. The recipe needs network access for the initial checkout. It pins revision `b74b71c2d4476cdbaa76cabd8f2d113cc171b6a8` and registers a link under `.local/tools/socratic-workshop/.agents/skills/`. Reruns reuse the checkout and preserve local edits.

```sh
./install.sh socratic-workshop
```

To use this trial from the playground root, ask Codex to read `.local/tools/socratic-workshop/source/SKILL.md` and tutor you with the shared case. That is how the opening was generated. To use the project registration in a fresh Codex session, launch the agent from the tool home.

```sh
cd .local/tools/socratic-workshop
codex
```

In that session, enter the following. Project paths in the handoff are relative to the playground root, three directories above the tool home.

```text
Use $socratic-workshop. Resume from ../../../topics/how-transformers-remember/cases/one-more-token/runs/2026-11-24-socratic-workshop/state.txt. Take one answer at a time in English and keep any learning files in that trial folder.
```

We checked that the registration resolves to the pinned instructions, and ran the installer twice. A separate Codex CLI 0.171.0 process could not open its state database under the sandbox. The fresh-session skill picker remains unverified. The active Codex agent read the pinned skill directly.
