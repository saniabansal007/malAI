#!/usr/bin/env bash
# Launch SGLang server for Enchanted Emporium (Dobby) model on RTX 4050 GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d "models/dobbie_magic_store_merged" ]; then
    DEFAULT_MODEL="models/dobbie_magic_store_merged"
elif [ -d "models/dobbie_magic_store_lora" ]; then
    DEFAULT_MODEL="models/dobbie_magic_store_lora"
elif [ -d "models/shopping_model_merged" ]; then
    DEFAULT_MODEL="models/shopping_model_merged"
else
    DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
fi

MODEL="${MODEL:-$DEFAULT_MODEL}"
PORT="${PORT:-30000}"
CTX="${CTX:-4096}"
MEM="${MEM:-0.75}"

export SGLANG_IS_FLASHINFER_AVAILABLE=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source .venv-sglang/bin/activate
echo "================================================================="
echo "   STARTING SGLANG SERVER: ENCHANTED EMPORIUM (DOBBY)"
echo "   Model: $MODEL | Port: $PORT | Context: $CTX"
echo "================================================================="

exec python -m sglang.launch_server \
  --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --context-length "$CTX" --mem-fraction-static "$MEM" \
  --attention-backend triton \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --max-running-requests 4 --chunked-prefill-size 1024 \
  --enable-metrics --enable-cache-report ${EXTRA:-}
