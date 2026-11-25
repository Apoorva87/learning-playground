#!/bin/sh
set -eu

upstream=https://github.com/shenh10/HuggingArch.git
revision=2c31e91ebcc83ee0f32d5ba26e4e13d14ee24f1a
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
    echo "Existing checkout has a different origin or revision. Your files are preserved." >&2
    exit 1
fi
cd "$checkout"
python_version=3.13
if [ -x .venv/bin/python ]; then
    python_version=.venv/bin/python
fi
export UV_PYTHON_INSTALL_DIR="$TOOL_HOME/python"
uv sync --locked --no-dev --python "$python_version" --cache-dir "$TOOL_HOME/uv-cache"
.venv/bin/huggingarch --help > /dev/null
# A Git checkout preserves the source of truth for both the web dependencies and the build.
npm ci --prefix web --prefer-offline --no-audit --no-fund --cache "$TOOL_HOME/npm-cache"
npm run build --prefix web
