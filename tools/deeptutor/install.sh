#!/bin/sh
set -eu

repository=https://github.com/HKUDS/DeepTutor.git
revision=1284116f92368480aef3a47f91b8df9e9b105062
source_dir="$TOOL_HOME/source"
env_dir="$TOOL_HOME/environment"

if [ ! -e "$source_dir" ]; then
    stage=$(mktemp -d "$TOOL_HOME/checkout.XXXXXX")
    git -C "$stage" init -q
    git -C "$stage" remote add origin "$repository"
    git -C "$stage" fetch -q --depth 1 origin "$revision"
    git -C "$stage" checkout -q --detach "$revision"
    mv "$stage" "$source_dir"
fi
if [ "$(git -C "$source_dir" remote get-url origin)" != "$repository" ] ||
   [ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]; then
    echo "The checkout differs from the trial revision. Preserved $source_dir." >&2
    exit 1
fi

mkdir -p "$env_dir"
cp environment.toml "$env_dir/pyproject.toml"
cp uv.lock "$env_dir/uv.lock"
uv sync --locked --project "$env_dir" --python 3.11 --cache-dir "$TOOL_HOME/uv-cache"
export DEEPTUTOR_HOME="$TOOL_HOME/runtime"
dt_cli="$env_dir/.venv/bin/deeptutor"
"$dt_cli" init --cli --non-interactive
catalog="$DEEPTUTOR_HOME/data/user/settings/model_catalog.json"
if "$env_dir/.venv/bin/python" - "$catalog" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
model = json.loads(path.read_text())["services"]["llm"] if path.exists() else {}
sys.exit(bool(model.get("active_model_id")))
PY
then
    "$dt_cli" config apply "$PLAYGROUND_ROOT/tools/deeptutor/ollama.json"
fi
"$dt_cli" doctor --format json
