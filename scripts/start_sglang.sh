

#!/usr/bin/env bash
# Launch SGLang tuned for a 6 GB GPU (RTX 4050 laptop). Override with env vars, e.g.
#   MODEL=Qwen/Qwen3-1.7B CTX=4096 ./scripts/start_sglang.sh
#   EXTRA="--disable-radix-cache" ./scripts/start_sglang.sh     # baseline for the benchmark
# Flag names change between SGLang releases; if one is rejected, check:
#   python -m sglang.launch_server --help
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-30000}"
CTX="${CTX:-4096}"
MEM="${MEM:-0.75}"        # fraction of VRAM for weights + KV cache.
export SGLANG_IS_FLASHINFER_AVAILABLE=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source .venv-sglang/bin/activate
exec python -m sglang.launch_server \
  --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --context-length "$CTX" --mem-fraction-static "$MEM" \
  --attention-backend triton \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --max-running-requests 4 --chunked-prefill-size 1024 \
  --enable-metrics --enable-cache-report ${EXTRA:-}

