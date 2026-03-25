#!/usr/bin/env bash
# Run the same checks as CI locally. Run before pushing.
# Usage: ./scripts/check.sh

set -e

echo "=== Ruff lint ==="
.venv/bin/ruff check src tests

echo "=== Pyright type check ==="
.venv/bin/pyright src

echo "=== Tests ==="
.venv/bin/python -m pytest tests/ -v --tb=short

echo ""
echo "✓ All checks passed"
