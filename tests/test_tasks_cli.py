"""On-demand task dispatch: `[tasks] auto_dispatch` and `openaugi tasks`.

The default install runs no background agent. `zzz:` lines still become task
files (on `ingest` or `tasks`), and the user reviews and launches them by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import openaugi.config as config_mod
from openaugi.agents import task_watcher
from openaugi.agents.task_watcher import list_tasks
from openaugi.cli.main import app
from openaugi.config import auto_dispatch_enabled
from openaugi.pipeline.dispatch import queue_from_ingest
from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore

runner = CliRunner()

TASK = """---
status: {status}
{extra}---

# {title}

## Context

Some context.

## User instruction

> {instruction}

## Task

Do it.

## Results
"""


def _task(tasks_dir: Path, name: str, status="pending", title=None, instruction="do x", extra=""):
    tasks_dir.mkdir(parents=True, exist_ok=True)
    path = tasks_dir / f"{name}.md"
    path.write_text(
        TASK.format(status=status, extra=extra, title=title or name, instruction=instruction)
    )
    return path


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "old note.md").write_text("# Old\n\nAn idea.\n\nzzz: an instruction from last year\n")
    return v


@pytest.fixture
def configured(vault: Path, monkeypatch):
    """Point every load_config() at the temp vault."""
    cfg = {"vault": {"default_path": str(vault)}, "models": {"embedding": {"provider": "none"}}}
    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: cfg)
    return cfg


# ── config ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "config,expected",
    [
        (None, False),
        ({}, False),
        ({"tasks": {"zzz_settle_seconds": 10}}, False),
        ({"tasks": {"auto_dispatch": True}}, True),
        ({"tasks": {"auto_dispatch": False}}, False),
    ],
)
def test_auto_dispatch_is_opt_in(config, expected):
    assert auto_dispatch_enabled(config) is expected


# ── queue_from_ingest ──────────────────────────────────────────────────────


def test_first_ingest_skips_zzz_already_in_the_vault(vault: Path, tmp_path: Path):
    store = SQLiteStore(tmp_path / "db.sqlite")
    result = run_layer0(vault, store)
    written, skipped = queue_from_ingest(result, store, vault, 0.0, first_ingest=True)
    store.close()
    assert written == []
    assert skipped == 1
    assert not (vault / "OpenAugi" / "Tasks").exists()


def test_new_zzz_after_setup_becomes_a_task_file(vault: Path, tmp_path: Path):
    store = SQLiteStore(tmp_path / "db.sqlite")
    queue_from_ingest(run_layer0(vault, store), store, vault, 0.0, first_ingest=True)

    (vault / "today.md").write_text("# Today\n\nzzz: summarize this week\n")
    written, skipped = queue_from_ingest(run_layer0(vault, store), store, vault, 0.0)
    store.close()
    assert skipped == 0
    assert len(written) == 1
    assert "summarize this week" in written[0].read_text()


def test_unsettled_zzz_waits(vault: Path, tmp_path: Path):
    store = SQLiteStore(tmp_path / "db.sqlite")
    queue_from_ingest(run_layer0(vault, store), store, vault, 0.0, first_ingest=True)
    (vault / "today.md").write_text("# Today\n\nzzz: half typed\n")
    written, _ = queue_from_ingest(run_layer0(vault, store), store, vault, 3600.0)
    # A second pass with settle 0 (what `openaugi tasks` does) picks it up.
    later, _ = queue_from_ingest(run_layer0(vault, store), store, vault, 0.0)
    store.close()
    assert written == []
    assert len(later) == 1


# ── list_tasks ─────────────────────────────────────────────────────────────


def test_list_tasks_orders_pending_first_and_hides_history(tmp_path: Path):
    d = tmp_path / "Tasks"
    _task(d, "running", status="active", extra="task_id: TASK-1\n")
    _task(d, "finished", status="done")
    _task(d, "waiting", title="Waiting one", instruction="write the memo")

    tasks = list_tasks(d)
    assert [t.status for t in tasks] == ["pending", "active"]
    assert tasks[0].title == "Waiting one"
    assert tasks[0].instruction == "write the memo"
    assert tasks[1].task_id == "TASK-1"
    assert len(list_tasks(d, include_finished=True)) == 3


def test_list_tasks_missing_folder_is_empty(tmp_path: Path):
    assert list_tasks(tmp_path / "nope") == []


# ── CLI ────────────────────────────────────────────────────────────────────


def test_tasks_list_syncs_and_shows_new_zzz(configured, vault: Path, tmp_path: Path):
    db = str(tmp_path / "db.sqlite")
    first = runner.invoke(app, ["tasks", "list", "--db", db])
    assert first.exit_code == 0, first.output
    assert "Skipped 1 zzz" in first.output
    assert "Nothing waiting" in first.output

    (vault / "today.md").write_text("# Today\n\nzzz: draft the agenda\n")
    second = runner.invoke(app, ["tasks", "list", "--db", db])
    assert second.exit_code == 0, second.output
    assert "1 pending" in second.output
    assert "draft the agenda" in second.output


def test_bare_tasks_takes_list_options(configured, vault: Path):
    """`openaugi tasks --all` is `openaugi tasks list --all`."""
    _task(vault / "OpenAugi" / "Tasks", "finished", status="done", instruction="old job")
    hidden = runner.invoke(app, ["tasks", "--no-sync"])
    assert hidden.exit_code == 0, hidden.output
    assert "old job" not in hidden.output

    shown = runner.invoke(app, ["tasks", "--all", "--no-sync"])
    assert shown.exit_code == 0, shown.output
    assert "old job" in shown.output


def test_group_options_before_a_subcommand_are_refused(configured, vault: Path):
    _task(vault / "OpenAugi" / "Tasks", "waiting")
    result = runner.invoke(app, ["tasks", "--all", "show", "1"])
    assert result.exit_code == 2
    assert "after the subcommand" in result.output


def test_tasks_run_confirms_then_launches(configured, vault: Path, monkeypatch):
    path = _task(vault / "OpenAugi" / "Tasks", "memo", instruction="write the memo")
    launched = []

    def fake_run(filepath, vault_path, repos_note=None):
        launched.append(filepath)
        return "TASK-2026-01-01-memo"

    monkeypatch.setattr(task_watcher, "run_task", fake_run)

    declined = runner.invoke(app, ["tasks", "run", "1"], input="n\n")
    assert declined.exit_code == 0, declined.output
    assert "write the memo" in declined.output  # shown before asking
    assert "Left pending" in declined.output
    assert launched == []

    accepted = runner.invoke(app, ["tasks", "run", "1"], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert launched == [path]
    assert "tmux attach -t TASK-2026-01-01-memo" in accepted.output


def test_tasks_run_refuses_an_active_task(configured, vault: Path, monkeypatch):
    _task(vault / "OpenAugi" / "Tasks", "busy", status="active", extra="task_id: TASK-9\n")
    _task(vault / "OpenAugi" / "Tasks", "waiting")
    monkeypatch.setattr(task_watcher, "run_task", lambda *a, **k: pytest.fail("launched"))
    # Active tasks sort after pending ones, so #2 is the active one.
    result = runner.invoke(app, ["tasks", "run", "2", "--yes"])
    assert result.exit_code == 1
    assert "tmux attach -t TASK-9" in result.output


def test_tasks_run_reports_missing_tmux(configured, vault: Path, monkeypatch):
    _task(vault / "OpenAugi" / "Tasks", "memo")

    def no_tmux(*a, **k):
        raise FileNotFoundError("tmux not found. Install with: brew install tmux")

    monkeypatch.setattr(task_watcher, "run_task", no_tmux)
    result = runner.invoke(app, ["tasks", "run", "memo", "--yes"])
    assert result.exit_code == 1
    assert "tmux not found" in result.output


def test_tasks_run_with_nothing_pending(configured):
    result = runner.invoke(app, ["tasks", "run"])
    assert result.exit_code == 0
    assert "No pending tasks" in result.output


def test_ingest_skips_history_on_first_run(configured, tmp_path: Path):
    result = runner.invoke(app, ["ingest", "--db", str(tmp_path / "db.sqlite")])
    assert result.exit_code == 0, result.output
    assert "Skipped 1 zzz" in result.output
    assert "Embeddings off" in result.output
