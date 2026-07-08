#!/usr/bin/env python3
"""Regenerate the cross-repo contract fixtures owned by this repo.

Source of truth for `tests/fixtures/contracts/context-pack.sample.json`: it is
produced by running the REAL context-pack builder over the deterministic corpus
in `tests.contract_corpus`, so the sample can never drift from the builder's
actual output shape. Run this whenever the ContextPack contract changes, then
run the sync script in the mobile repo (see docs/plans/task-contract-fixtures.md
and README "Contract fixtures").

    .venv/bin/python scripts/gen_contract_fixtures.py

The other two fixtures (`dashboard-nominations.md`, `capture-daily-note.md`) are
hand-authored sanitized samples — edit them directly; there is nothing to
generate. All three are pinned by tests/test_contract_fixtures.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.contract_corpus import SAMPLE_GENERATED_AT, build_reference_pack  # noqa: E402

CONTRACTS_DIR = REPO / "tests" / "fixtures" / "contracts"
SAMPLE_PATH = CONTRACTS_DIR / "context-pack.sample.json"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        pack = build_reference_pack(Path(tmp))
    pack["generatedAt"] = SAMPLE_GENERATED_AT  # keep the fixture byte-stable
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {SAMPLE_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
