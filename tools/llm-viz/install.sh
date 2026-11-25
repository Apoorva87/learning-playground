#!/bin/sh
set -eu

upstream=https://github.com/bbycroft/llm-viz.git
revision=9da93742382f1bf36c020c38a1ace454e82c4490
checkout="$TOOL_HOME/source"
if [ ! -e "$checkout" ] && [ ! -L "$checkout" ]; then
    stage=$(mktemp -d "$TOOL_HOME/checkout.XXXXXX")
    trap 'rm -rf "$stage"' 0
    git clone --quiet --no-checkout "$upstream" "$stage/source"
    git -C "$stage/source" checkout --quiet --detach "$revision"
    mv "$stage/source" "$checkout"
fi
if [ "$(git -C "$checkout" remote get-url origin)" != "$upstream" ] ||
   [ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]; then
    echo "Different checkout at $checkout. Your files are preserved." >&2
    exit 1
fi
cd "$checkout"
npm install --legacy-peer-deps --no-package-lock --no-audit --no-fund --prefer-offline --cache "$TOOL_HOME/npm-cache"
./node_modules/.bin/next --version
