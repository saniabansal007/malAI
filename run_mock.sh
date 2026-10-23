#!/usr/bin/env bash
# Develop the UI / test the backend with NO GPU, against a fake SGLang server.
set -euo pipefail
cd "$(dirname "$0")"
source .venv-app/bin/activate 2>/dev/null || true
uvicorn tests.mock_sglang:app --port 30000 --log-level warning &
MOCK=$!; trap 'kill $MOCK 2>/dev/null || true' EXIT
sleep 1
SGLANG_URL=http://127.0.0.1:30000 uvicorn backend.main:app --host 0.0.0.0 --port 8000

