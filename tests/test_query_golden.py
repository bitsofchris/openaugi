"""Golden regression for the read-tool wire format (query-layer step 1).

Every MCP read tool's raw JSON output over the deterministic corpus must be
BYTE-identical to tests/fixtures/golden/query_golden.json. This is the
contract that lets docs/plans/query-layer.md move query semantics out of
mcp/server.py into query/ without the wire format drifting.

If a test here fails, either the refactor changed behavior (fix the code)
or the wire format is changing on purpose (regenerate with
scripts/gen_query_golden.py and justify in the commit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.query_golden_corpus import CASES, GOLDEN_PATH, build_store, run_all


@pytest.fixture(scope="module")
def golden_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    db = tmp_path_factory.mktemp("golden") / "golden.db"
    build_store(db)
    return run_all(db)


@pytest.fixture(scope="module")
def golden_file() -> dict[str, str]:
    assert GOLDEN_PATH.exists(), (
        f"Golden file missing: {GOLDEN_PATH}. Generate with scripts/gen_query_golden.py"
    )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_case_matrix_matches_golden_file_keys(golden_file: dict[str, str]):
    assert set(golden_file) == set(CASES), (
        "Case matrix and golden file diverge — regenerate scripts/gen_query_golden.py"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden(case: str, golden_outputs: dict[str, str], golden_file: dict[str, str]):
    assert golden_outputs[case] == golden_file[case], (
        f"Wire format changed for {case!r}. If intentional, regenerate via "
        "scripts/gen_query_golden.py and explain in the commit."
    )


def test_build_is_deterministic(tmp_path: Path, golden_outputs: dict[str, str]):
    """Two independent builds produce identical outputs — no hidden clock/order."""
    db = tmp_path / "golden2.db"
    build_store(db)
    assert run_all(db) == golden_outputs
