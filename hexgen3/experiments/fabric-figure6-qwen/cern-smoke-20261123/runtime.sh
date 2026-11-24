#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/hexgen2-qwen"
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
.venv/bin/pip install 'transformers==4.51.3' 'huggingface-hub==0.36.2' \
  'safetensors==0.5.3' 'psutil==6.1.0' 'requests==2.32.3'
.venv/bin/pip freeze > runtime-version.txt
PYTHONPATH="$PWD/hexgen2/source" .venv/bin/python -c 'import torch; import hexgen2.qualify_qwen; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
date -u +%FT%TZ > runtime.ready
