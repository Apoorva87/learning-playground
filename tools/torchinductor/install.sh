#!/bin/sh
set -eu
: "${TOOL_HOME:?Run ./install.sh torchinductor from the repository root}"
: "${PLAYGROUND_ROOT:?Run ./install.sh torchinductor from the repository root}"
venv="$TOOL_HOME/.venv"
shared="$PLAYGROUND_ROOT/.local/tools/huggingarch/source/.venv/bin/python"
versions='import numpy,safetensors,torch; assert (numpy.__version__,safetensors.__version__,torch.__version__.split("+")[0]) == ("2.4.4","0.8.0","2.11.0")'
if [ ! -x "$venv/bin/python" ]; then
    if [ -x "$shared" ] && "$shared" -c "$versions"; then
        "$shared" -m venv --without-pip "$venv"
        source_lib=$("$shared" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        local_lib=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        printf '%s\n' "$source_lib" > "$local_lib/playground-base.pth"
    else
        uv venv --python 3.13 "$venv"
    fi
fi
if ! "$venv/bin/python" -c "$versions" 2>/dev/null; then
    UV_CACHE_DIR="$TOOL_HOME/uv-cache" uv pip install --python "$venv/bin/python" 'torch==2.11.0' 'safetensors==0.8.0' 'numpy==2.4.4'
fi
TORCHINDUCTOR_CACHE_DIR="$TOOL_HOME/install-cache" TORCHINDUCTOR_COMPILE_THREADS=1 "$venv/bin/python" - <<'PY'
import torch
torch.set_num_threads(1)
x = torch.arange(8, dtype=torch.float32)
f = torch.compile(lambda x: x.square() + 1, backend="inductor", fullgraph=True)
torch.testing.assert_close(f(x), x.square() + 1)
print(f"PyTorch {torch.__version__} compiled and verified a CPU kernel.")
PY
