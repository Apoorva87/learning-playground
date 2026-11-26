#!/bin/sh
set -eu
: "${TOOL_HOME:?Run ./install.sh dace from the repository root}"
export DACE_CONFIG="$TOOL_HOME/dace.conf"
export UV_CACHE_DIR="$TOOL_HOME/uv-cache"
if [ ! -x "$TOOL_HOME/.venv/bin/python" ]; then
    uv venv --python "${DACE_PYTHON:-3.11}" "$TOOL_HOME/.venv"
fi
if [ -n "${DACE_WHEELHOUSE:-}" ]; then
    set -- --no-index --find-links "$DACE_WHEELHOUSE" --no-build-isolation
else
    set -- --no-build-isolation
fi
export PATH="$TOOL_HOME/.venv/bin:$PATH"
uv pip install --python "$TOOL_HOME/.venv/bin/python" "$@" 'setuptools==80.9.0' 'wheel==0.45.1' 'cmake==3.31.10'
uv pip install --python "$TOOL_HOME/.venv/bin/python" "$@" 'dace==1.0.2' 'numpy==1.26.4'
"$TOOL_HOME/.venv/bin/python" -c 'import dace; print("DaCe", dace.__version__)'
DACE_CONFIG="$TOOL_HOME/dace.conf" "$TOOL_HOME/.venv/bin/sdfgcc" --help >/dev/null
