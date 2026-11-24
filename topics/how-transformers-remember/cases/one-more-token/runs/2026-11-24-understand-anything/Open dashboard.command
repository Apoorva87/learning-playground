#!/bin/sh
set -eu
run_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
project=$(git -C "$run_dir" rev-parse --show-toplevel)
viewer="$project/.local/tools/understand-anything/viewer-v2.9.8/bin/viewer.mjs"
if [ ! -f "$viewer" ]; then
    "$project/install.sh" understand-anything
fi
exec node "$viewer" "$run_dir/wiki" "$@"
