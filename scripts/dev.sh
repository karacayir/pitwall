#!/bin/zsh
# Local dev: backend (uvicorn, replay mode) + frontend (next dev) together.
set -e
cd "$(dirname "$0")/.."
(cd backend && DATA_SOURCE="${DATA_SOURCE:-replay}" uv run uvicorn app.api:app --reload --port 8000) &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT INT TERM
cd frontend && npm run dev
