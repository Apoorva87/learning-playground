#!/bin/sh
set -eu
: "${PLAYGROUND_ROOT:?Run ./install.sh pytorch-profiler from the repository root}"
: "${TOOL_HOME:?Run ./install.sh pytorch-profiler from the repository root}"
env_dir="$TOOL_HOME/.venv"
shared="$PLAYGROUND_ROOT/.local/tools/huggingarch/source/.venv/bin/python"
versions='import numpy,torch; assert numpy.__version__=="2.4.4" and torch.__version__.split("+")[0]=="2.11.0"'
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TOOL_HOME/uv-cache}"
if [ ! -x "$env_dir/bin/python" ]; then
    if [ -x "$shared" ] && "$shared" -c "$versions"; then
        "$shared" -m venv --without-pip "$env_dir"
        src=$("$shared" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        dest=$("$env_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        printf '%s\n' "$src" > "$dest/playground-base.pth"
    else
        uv venv --python 3.13 "$env_dir"
    fi
fi
if ! "$env_dir/bin/python" -c "$versions" 2>/dev/null; then
    uv pip install --python "$env_dir/bin/python" "torch==2.11.0" "numpy==2.4.4"
fi
"$env_dir/bin/python" - <<'PY'
import torch
from torch.profiler import profile, ProfilerActivity
x = torch.ones(4)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as capture:
    y = torch.dot(x, x)
assert y.item() == 4 and any(e.name == "aten::dot" for e in capture.events())
print(f"PyTorch {torch.__version__} captured a CPU dot product.")
PY
