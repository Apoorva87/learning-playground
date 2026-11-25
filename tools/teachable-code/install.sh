#!/bin/sh
set -eu
repo=https://github.com/ywEdAi/teachable-code.git
revision=e74598d4b0d2a13cf67182ec897ad8f308894495
source_dir="$TOOL_HOME/source"

if [ ! -e "$source_dir" ] && [ ! -L "$source_dir" ]; then
    stage=$(mktemp -d "$TOOL_HOME/checkout.XXXXXX")
    trap 'rm -rf "$stage"' 0
    git clone --quiet --no-checkout "$repo" "$stage/source"
    git -C "$stage/source" checkout --quiet --detach "$revision"
    mv "$stage/source" "$source_dir"
fi
if [ "$(git -C "$source_dir" config --get remote.origin.url)" != "$repo" ] ||
   [ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]; then
    echo "Unexpected checkout at $source_dir. Your files have been preserved." >&2
    exit 1
fi
node --version
python3 "$source_dir/scripts/build_lesson.py" --help
