#!/usr/bin/env bash
# Start the Knowledge Explorer (backend + frontend)
# Usage: ./start.sh [--db /path/to/openaugi.db]

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
DB="${2:-$HOME/.openaugi/openaugi.db}"

# ── Backend ────────────────────────────────────────────────────────────────────
echo "Starting backend on :8000 (db: $DB)"
cd "$ROOT/backend"

if ! python3 -c "import fastapi, uvicorn, umap" 2>/dev/null; then
  echo "Installing backend deps..."
  pip install fastapi uvicorn umap-learn numpy
fi

python3 server.py --db "$DB" &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# ── Frontend ───────────────────────────────────────────────────────────────────
echo "Starting frontend on :5173"
cd "$ROOT/frontend"

if [ ! -d node_modules ]; then
  echo "Installing frontend deps..."
  npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# ── Cleanup ────────────────────────────────────────────────────────────────────
cleanup() {
  echo "Stopping..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "  Knowledge Explorer: http://localhost:5173"
echo "  Backend API:        http://localhost:8000/api"
echo "  (Ctrl-C to stop)"
echo ""

wait
