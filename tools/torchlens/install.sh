#!/bin/sh
set -eu

: "${PLAYGROUND_ROOT:?Run ./install.sh torchlens from the workspace root.}"
: "${TOOL_HOME:?Run ./install.sh torchlens from the workspace root.}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TOOL_HOME/uv-cache}"
env_dir="$TOOL_HOME/.venv"
shared="$PLAYGROUND_ROOT/.local/tools/huggingarch/source/.venv/bin/python"
if [ ! -x "$env_dir/bin/python" ]; then
    if [ -x "$shared" ]; then
        "$shared" -m venv --without-pip "$env_dir"
        shared_lib=$("$shared" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        local_lib=$("$env_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        printf '%s\n' "$shared_lib" > "$local_lib/playground-base.pth"
    else
        uv venv --python 3.13 "$env_dir"
    fi
fi
python="$env_dir/bin/python"

if "$python" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.11.0" and transformers.__version__ == "5.24.0"' 2>/dev/null; then
    if ! "$python" -c 'from importlib.metadata import version; assert (version("torchlens"),version("pillow"),version("graphviz"))==("3.0.3","13.0.0","0.21")' 2>/dev/null; then
        if [ -d "$TOOL_HOME/wheels" ]; then
            uv pip install --python "$python" --no-index --find-links "$TOOL_HOME/wheels" --no-deps 'torchlens==3.0.3' 'graphviz==0.21' 'pillow==13.0.0'
        else
            uv pip install --python "$python" --no-deps 'torchlens==3.0.3' 'graphviz==0.21' 'pillow==13.0.0'
        fi
    fi
else
    uv pip install --python "$python" 'torch==2.11.0' 'transformers==5.24.0' 'torchlens==3.0.3' 'graphviz==0.21' 'pillow==13.0.0'
fi

"$python" - <<'PY'
import torch, torchlens as tl
with torch.no_grad():
    trace = tl.trace(torch.nn.ReLU(), torch.tensor([-1.0, 1.0]))
assert trace.output.tolist() == [0.0, 1.0]
print(f"TorchLens {tl.__version__} captured ReLU with PyTorch {torch.__version__}.")
PY
