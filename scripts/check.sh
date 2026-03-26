#!/usr/bin/env bash
# Run the same checks as CI locally. Run before pushing.
# Usage: ./scripts/check.sh

set -e

echo "=== Pre-commit hooks (lint, format, types) ==="
.venv/bin/pre-commit run --all-files

echo ""
echo "=== Tests ==="
.venv/bin/python -m pytest tests/ -v --tb=short

echo ""
echo "✓ All checks passed"
