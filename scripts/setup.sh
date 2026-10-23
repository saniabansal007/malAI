#!/usr/bin/env bash
# One-time setup (WSL2 Ubuntu or native Linux). Creates two virtualenvs so SGLang's
# heavy dependencies never clash with the app's.
#   LIGHT=1 ./scripts/setup.sh   -> skip sentence-transformers (keyword search only)
set -euo pipefail
cd "$(dirname "$0")/.."

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found. In WSL2 install the NVIDIA Windows driver (not a Linux one) and check 'nvidia-smi' works."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -m venv .venv-sglang
.venv-sglang/bin/pip install -U pip uv
.venv-sglang/bin/uv pip install --python .venv-sglang/bin/python "sglang[all]"

python3 -m venv .venv-app
.venv-app/bin/pip install -U pip
.venv-app/bin/pip install -r requirements.txt
if [ "${LIGHT:-0}" != "1" ]; then
  # CPU-only torch keeps the download small and leaves the GPU to SGLang
  .venv-app/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv-app/bin/pip install sentence-transformers
fi
echo "Setup complete. Next: ./run.sh"
