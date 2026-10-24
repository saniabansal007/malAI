#!/usr/bin/env bash
# Launch SGLang with the Fine-Tuned Banking Compliance Model
# Tuned for RTX 4050 Laptop GPU (6 GB VRAM) on port 30000.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LORA_PATH="${LORA_PATH:-models/banking_lora_adapter}"
PORT="${PORT:-30000}"
CTX="${CTX:-4096}"
MEM="${MEM:-0.75}"

export SGLANG_IS_FLASHINFER_AVAILABLE=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source .venv-sglang/bin/activate

if [ -d "models/banking_model_merged" ]; then
    echo "=================================================================="
    echo "    STARTING SGLANG WITH MERGED BANKING COMPLIANCE MODEL"
    echo "=================================================================="
    echo "Model Path:     models/banking_model_merged"
    echo "Port:           $PORT"
    echo "Prefix Cache:   Enabled (Full RadixAttention)"
    echo "=================================================================="

    exec python -m sglang.launch_server \
      --model-path "models/banking_model_merged" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --context-length "$CTX" \
      --mem-fraction-static "$MEM" \
      --attention-backend triton \
      --disable-cuda-graph \
      --disable-overlap-schedule \
      --max-running-requests 4 \
      --chunked-prefill-size 1024 \
      --enable-metrics \
      --enable-cache-report ${EXTRA:-}
else
    echo "=================================================================="
    echo "    STARTING SGLANG WITH DYNAMIC BANKING LORA ADAPTER"
    echo "=================================================================="
    echo "Base Model:     $MODEL"
    echo "LoRA Adapter:   $LORA_PATH"
    echo "Port:           $PORT"
    echo "=================================================================="

    exec python -m sglang.launch_server \
      --model-path "$MODEL" \
      --lora-paths "banking_adapter=$LORA_PATH" \
      --max-loras-per-batch 1 \
      --disable-radix-cache \
      --host 127.0.0.1 \
      --port "$PORT" \
      --context-length "$CTX" \
      --mem-fraction-static "$MEM" \
      --attention-backend triton \
      --disable-cuda-graph \
      --disable-overlap-schedule \
      --max-running-requests 4 \
      --chunked-prefill-size 1024 \
      --enable-metrics \
      --enable-cache-report ${EXTRA:-}
fi
