#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hexgen3-openr1"
python3 -m venv .venv
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
.venv/bin/pip install --upgrade pip uv
export UV_NO_CACHE=1
.venv/bin/uv pip install --python .venv/bin/python -e './runtime/python[srt,openai]' \
  'mooncake-transfer-engine==0.3.7' 'pybind11==2.13.6' 'netifaces==0.11.0'
.venv/bin/python -c 'import sglang, torch; print(sglang.__version__, torch.__version__, torch.version.cuda)'
date -u +%FT%TZ > runtime.ready
