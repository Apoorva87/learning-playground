#!/bin/sh
set -eu
upstream=https://github.com/THU-MAIC/OpenMAIC.git
revision=c09bb0e2a9f32c2ce330817767eda4697c635351
checkout="$TOOL_HOME/source"

if [ ! -e "$checkout" ]; then
    git clone --no-checkout --depth 1 "$upstream" "$checkout"
fi
if [ "$(git -C "$checkout" remote get-url origin)" != "$upstream" ]; then
    echo "Unexpected source repository at $checkout" >&2
    exit 1
fi
if ! git -C "$checkout" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$checkout" fetch --depth 1 origin "$revision"
fi
# A regular clone initializes its worktree. A new --no-checkout clone has no index.
if [ ! -e "$checkout/.git/index" ]; then
    git -C "$checkout" checkout --detach "$revision"
elif [ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]; then
    echo "A different revision is checked out at $checkout. Preserve it before reinstalling." >&2
    exit 1
fi

mkdir -p "$TOOL_HOME/sdk"
cp package.json package-lock.json "$TOOL_HOME/sdk/"
npm ci --prefix "$TOOL_HOME/sdk" --no-audit --no-fund
