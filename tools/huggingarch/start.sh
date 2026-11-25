#!/bin/sh
set -eu
root=$(CDPATH= cd -P "$(dirname "$0")/../.." && pwd)
cd "$root/.local/tools/huggingarch/source"
exec .venv/bin/python -m backend.api --host 127.0.0.1 --port "${HUGGINGARCH_PORT:-8016}"
