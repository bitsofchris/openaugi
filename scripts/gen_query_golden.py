#!/usr/bin/env python
"""Regenerate the query-layer golden file.

Usage: .venv/bin/python scripts/gen_query_golden.py

Rebuilds the deterministic golden DB (tests/query_golden_corpus.py), runs
every case in the matrix against the MCP read tools, and writes the raw
outputs to tests/fixtures/golden/query_golden.json.

Regenerate ONLY when the MCP wire format is meant to change — the whole
point of the file is that refactors (docs/plans/query-layer.md) keep it
byte-identical. Say why in the commit when you do.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.query_golden_corpus import GOLDEN_PATH, build_store, run_all  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "golden.db"
        build_store(db)
        outputs = run_all(db)

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(outputs)} cases to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
