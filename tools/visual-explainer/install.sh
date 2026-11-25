#!/bin/sh
set -eu
upstream=https://github.com/nicobailon/visual-explainer.git
revision=9d5f880b7e8e0a9a80c890b8e1bb6918b68481ad
source_dir="$TOOL_HOME/source"

if [ ! -e "$source_dir" ] && [ ! -L "$source_dir" ]; then
    staging=$(mktemp -d "$TOOL_HOME/checkout.XXXXXX")
    trap 'rm -rf "$staging"' 0
    git clone --quiet --no-checkout "$upstream" "$staging/source"
    git -C "$staging/source" checkout --quiet --detach "$revision"
    mv "$staging/source" "$source_dir"
fi
if [ "$(git -C "$source_dir" config --get remote.origin.url)" != "$upstream" ] ||
   [ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]; then
    echo "Unexpected checkout at $source_dir. Your files have been preserved." >&2
    exit 1
fi
node "$source_dir/scripts/check-versions.mjs"
