#!/bin/sh
# Rebuild from the saved real model inputs, using the existing full-model trace pair.
set -eu
here=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
root=$(CDPATH= cd -P "$here/../../.." && pwd)
chakra="$root/.local/tools/chakra/source/.venv/bin/python"
torch=${TORCH_PYTHON:-"$root/.local/tools/huggingarch/source/.venv/bin/python"}
traces=${FULL_TRACE_DIR:-"$root/.local/qwen38/compact"}
raw=${REPLAY_TRACE_DIR:-"$root/.local/tools/chakra/real-replay"}
"$torch" "$here/capture.py" --output "$raw"
"$chakra" "$here/analyze.py" --replay "$raw" --full-traces "$traces" --whole-pass "$here/../assets/whole-pass.json" --output "$here"
