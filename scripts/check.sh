#!/usr/bin/env bash
# Run the same checks as CI locally. Run before pushing.
# Usage: ./scripts/check.sh

set -e

echo "=== Build validation ==="
# The venv is uv-managed and has no pip in it, so shelling out to
# .venv/bin/pip failed every pre-push. Prefer pip when it is actually there
# (a plain `python -m venv` setup), fall back to `uv pip` otherwise. Neither
# path touches uv.lock.
if [ -x .venv/bin/pip ]; then
  .venv/bin/pip install -e ".[dev]" --quiet
else
  uv pip install -e ".[dev]" --quiet
fi

echo "=== Pre-commit hooks (lint, format, types) ==="
.venv/bin/pre-commit run --all-files

echo ""
echo "=== Tests ==="
.venv/bin/python -m pytest tests/ -v --tb=short

echo ""
echo "✓ All checks passed"
