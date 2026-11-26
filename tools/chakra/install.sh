#!/bin/sh
set -eu
revision=065445b5d48ffb8796c2dbf4f0540cc608551f8d
upstream=https://github.com/mlcommons/chakra.git
checkout="$TOOL_HOME/source"
if [ ! -e "$checkout" ]; then
    stage=$(mktemp -d "$TOOL_HOME/checkout.XXXXXX")
    trap 'rm -rf "$stage"' 0
    git clone --quiet --no-checkout "$upstream" "$stage/source"
    git -C "$stage/source" checkout --quiet --detach "$revision"
    mv "$stage/source" "$checkout"
fi
if [ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ] ||
   [ "$(git -C "$checkout" remote get-url origin)" != "$upstream" ]; then
    echo "The existing Chakra checkout differs from the tested source. Please inspect it." >&2
    exit 1
fi
python=3.13
if [ -x "$checkout/.venv/bin/python" ]; then
    python="$checkout/.venv/bin/python"
elif [ -x "$PLAYGROUND_ROOT/.local/tools/huggingarch/source/.venv/bin/python" ]; then
    python="$PLAYGROUND_ROOT/.local/tools/huggingarch/source/.venv/bin/python"
fi
export UV_CACHE_DIR="$TOOL_HOME/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TOOL_HOME/python"
uv sync --project "$checkout" --locked --no-dev --no-editable --python "$python"
"$checkout/.venv/bin/chakra_converter" --help >/dev/null
"$checkout/.venv/bin/chakra_trace_link" --help >/dev/null
"$checkout/.venv/bin/chakra_jsonizer" --help >/dev/null
