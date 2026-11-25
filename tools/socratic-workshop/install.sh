#!/bin/sh
set -eu

repo=https://github.com/smashburger-dev/socratic-workshop.git
revision=b74b71c2d4476cdbaa76cabd8f2d113cc171b6a8
checkout="$TOOL_HOME/source"
skills="$TOOL_HOME/.agents/skills"
registration="$skills/socratic-workshop"

if [ ! -e "$checkout" ] && [ ! -L "$checkout" ]; then
    staging=$(mktemp -d "$TOOL_HOME/install.XXXXXX")
    trap 'rm -rf "$staging"' 0
    git clone -q --no-checkout "$repo" "$staging/source"
    git -C "$staging/source" checkout -q --detach "$revision"
    mv "$staging/source" "$checkout"
fi

if [ "$(git -C "$checkout" config --get remote.origin.url)" != "$repo" ] ||
   [ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]; then
    echo "The checkout differs from the trial revision. Your files have been preserved." >&2
    exit 1
fi

mkdir -p "$skills"
if [ ! -e "$registration" ] && [ ! -L "$registration" ]; then
    ln -s ../../source "$registration"
fi
if [ ! -L "$registration" ] || [ "$(readlink "$registration")" != ../../source ]; then
    echo "Another registration exists at $registration. Your files have been preserved." >&2
    exit 1
fi

test -r "$registration/SKILL.md"
test -r "$registration/references/codebase-learning.md"
test -r "$registration/agents/openai.yaml"
printf 'Socratic Workshop at %s\nStart Codex from %s to use $socratic-workshop.\n' "$revision" "$TOOL_HOME"
