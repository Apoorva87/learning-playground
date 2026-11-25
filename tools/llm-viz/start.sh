#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
checkout="$root/.local/tools/llm-viz/source"
if [ ! -x "$checkout/node_modules/.bin/next" ]; then
    "$root/install.sh" llm-viz
fi
cd "$checkout"
printf 'Open http://127.0.0.1:3002/llm when the server is ready. Press Ctrl+C here to stop.\n'
exec npm run dev -- --hostname 127.0.0.1
