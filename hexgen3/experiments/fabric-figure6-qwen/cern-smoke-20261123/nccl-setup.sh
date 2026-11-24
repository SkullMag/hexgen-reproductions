#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hexgen2-qwen"
library="$PWD/nccl-runtime/nvidia/nccl/lib/libnccl.so.2"
if [[ ! -r "$library" ]]; then
    .venv/bin/pip install --no-deps --target ./nccl-runtime nvidia-nccl-cu12==2.26.5
fi
export LD_PRELOAD="$library"
export PYTHONPATH="$PWD/hexgen2/source"
.venv/bin/python - <<'PY' > nccl.ready.tmp
from hexgen2.nccl_runtime import runtime_version

version = runtime_version()
if version != (2, 26, 5):
    raise SystemExit(f"Expected NCCL 2.26.5 at runtime, got {version}")
print(".".join(map(str, version)))
PY
mv nccl.ready.tmp nccl.ready
