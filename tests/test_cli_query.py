"""CLI smoke tests for `openaugi search` and `openaugi query`.

The CLI is the third adapter over the query engine (query-layer step 6):
same rules as MCP/HTTP — tags/time/task filters, bronze exclusion — where
the old CLI search was a filter-less duplicate implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from openaugi.cli.main import app
from openaugi.query import saved
from tests.query_golden_corpus import build_store

runner = CliRunner()


@pytest.fixture
def golden_db(tmp_path: Path) -> Path:
    db = tmp_path / "cli.db"
    build_store(db)
    return db


@pytest.fixture
def vault_with_queries(tmp_path: Path) -> Path:
    import importlib.resources

    vault = tmp_path / "vault"
    folder = saved.queries_dir(vault)
    folder.mkdir(parents=True)
    templates = importlib.resources.files("openaugi") / "templates" / "queries"
    for entry in templates.iterdir():
        (folder / entry.name).write_text(entry.read_text(encoding="utf-8"))
    return vault


class TestSearchCommand:
    def test_keyword_search(self, golden_db: Path):
        result = runner.invoke(app, ["search", "quantum", "--keyword", "--db", str(golden_db)])
        assert result.exit_code == 0, result.output
        assert "2026-06-01" in result.output  # b1's title (note stem)

    def test_task_filter_excludes_bronze(self, golden_db: Path):
        result = runner.invoke(
            app, ["search", "--task", "--after", "2026-01-01", "--db", str(golden_db)]
        )
        assert result.exit_code == 0, result.output
        assert "2026-06-02" in result.output  # open-checkbox task
        assert "2026-06-04" in result.output  # type/task tagged
        assert "2026-06-03" not in result.output  # bronze never counts

    def test_browse_filters_and_reference_note(self, golden_db: Path):
        result = runner.invoke(app, ["search", "--after", "2026-01-01", "--db", str(golden_db)])
        assert result.exit_code == 0, result.output
        assert "reference block(s) collapsed" in result.output

    def test_tag_filter(self, golden_db: Path):
        result = runner.invoke(
            app,
            ["search", "--tag", "idea", "--after", "2026-01-01", "--db", str(golden_db)],
        )
        assert result.exit_code == 0, result.output
        assert "2026-06-01" in result.output
        assert "2026-06-02" not in result.output

    def test_empty_invocation_errors(self, golden_db: Path):
        result = runner.invoke(app, ["search", "--db", str(golden_db)])
        assert result.exit_code == 1


class TestQueryCommand:
    def test_list_saved_queries(self, golden_db: Path, vault_with_queries: Path):
        result = runner.invoke(app, ["query", "--vault", str(vault_with_queries)])
        assert result.exit_code == 0, result.output
        assert "dashboard-task-shelf" in result.output
        assert "review-queue" in result.output

    def test_run_review_queue(self, golden_db: Path, vault_with_queries: Path):
        """$review-mark resolves against the DB's mark; post-mark blocks print."""
        result = runner.invoke(
            app,
            [
                "query",
                "review-queue",
                "--vault",
                str(vault_with_queries),
                "--db",
                str(golden_db),
            ],
        )
        assert result.exit_code == 0, result.output
        # Ingested before the 2026-06-05 mark → excluded (b1..b4, b6).
        assert "2026-06-02" not in result.output
        assert "quantum garden" not in result.output
        # Ingested after the mark → shown (b9/b10 live in the container note).
        assert "alpha container" in result.output
        # b5 is post-mark but under OpenAugi/ → excluded by path prefix.
        assert "derived artifact" not in result.output

    def test_unknown_query_fails(self, golden_db: Path, vault_with_queries: Path):
        result = runner.invoke(
            app, ["query", "nope", "--vault", str(vault_with_queries), "--db", str(golden_db)]
        )
        assert result.exit_code == 1
