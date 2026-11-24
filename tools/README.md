# Tool catalog

- [OpenMAIC](openmaic/README.md) — Trying its interactive lesson generator on transformer KV caches.
- [Understand‑Anything](understand-anything/README.md) — Turning the same KV-cache material into a linked knowledge graph and guided tour.

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
