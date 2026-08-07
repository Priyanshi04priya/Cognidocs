#!/usr/bin/env bash
# Convenience launcher — starts API, Celery worker, and Streamlit.
# Prerequisites: Redis + Qdrant already running (docker compose up -d)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Starting FastAPI on :8000 ..."
uvicorn app.main:app --reload --port 8000 &
API_PID=$!

echo "Starting Celery worker ..."
celery -A app.celery_app.celery_app worker --loglevel=info &
CELERY_PID=$!

echo "Starting Streamlit on :8501 ..."
streamlit run streamlit_app.py --server.port 8501 &
UI_PID=$!

cleanup() {
  echo "Stopping..."
  kill "$API_PID" "$CELERY_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
