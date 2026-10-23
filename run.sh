#!/usr/bin/env bash
# Start SGLang (background), wait until it's ready, then start the app on :8000.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PORT="${PORT:-30000}"
# Kill any lingering instances on 30000 or 8000
pkill -9 -f "fake_sglang|sglang.launch_server" 2>/dev/null || true
fuser -k "${PORT}/tcp" 8000/tcp 2>/dev/null || true
sleep 1
./scripts/start_sglang.sh > logs/sglang.log 2>&1 &
SG=$!
trap 'kill $SG 2>/dev/null || true' EXIT
echo "Starting SGLang (first run downloads the model; log: logs/sglang.log)…"
READY=0
for i in $(seq 1 1800); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    READY=1
    break
  fi
  kill -0 $SG 2>/dev/null || { echo "SGLang exited. Last log lines:"; tail -25 logs/sglang.log; exit 1; }
  if [ $((i % 30)) -eq 0 ]; then
    echo "Still initializing SGLang into GPU memory... ($((i * 2))s elapsed)"
  fi
  sleep 2
done
if [ "$READY" -ne 1 ]; then
  echo "SGLang failed to become healthy within timeout."
  exit 1
fi
echo "SGLang is up. Open http://localhost:8000"
source .venv-app/bin/activate
SGLANG_URL="http://127.0.0.1:$PORT" uvicorn backend.main:app --host 0.0.0.0 --port 8000

