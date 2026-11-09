"""Sync before read: a Light install's `openaugi kb` keeps its own index fresh.

Pinned here: a stale index is refreshed before the answer, a fresh one is
not touched, `up` running or the setting off means no sync, concurrent
callers never both sync, a failure never fails the read, and `zzz:` lines
indexed by a sync are queued like `openaugi ingest` queues them.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import openaugi.mcp.server as srv
from openaugi.cli.main import app
from openaugi.config import DEFAULT_CONFIG, _merge
from openaugi.pipeline import kb_sync
from openaugi.pipeline.kb_sync import sync_before_read
from openaugi.store.sqlite import SQLiteStore

runner = CliRunner()
NEVER_UP = lambda: False  # noqa: E731


def _config(**kb) -> dict:
    return _merge(
        DEFAULT_CONFIG,
        {"models": {"embedding": {"provider": "none"}}, "kb": {"auto_sync": True, **kb}},
    )


@pytest.fixture
def vault(tmp_path: Path, vault_path: Path) -> Path:
    v = tmp_path / "vault"
    shutil.copytree(vault_path, v)
    return v


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "kb.db"


def _blocks(db: Path) -> int:
    store = SQLiteStore(db)
    try:
        return store.get_stats()["total_blocks"]
    finally:
        store.close()


# ── sync_before_read ───────────────────────────────────────────────────────


def test_first_call_indexes_the_vault(vault: Path, db: Path):
    out = sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    assert out.status == "synced"
    assert _blocks(db) > 0


def test_fresh_index_is_left_alone(vault: Path, db: Path):
    sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    out = sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    assert out.status == "fresh"


def test_stale_index_picks_up_a_new_note(vault: Path, db: Path):
    sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    (vault / "Fresh idea.md").write_text("# Fresh idea\n\nzebracorn migration plan\n")
    later = os.path.getmtime(db.with_name(db.name + ".kb-sync")) + 120
    out = sync_before_read(db, str(vault), _config(), up_running=NEVER_UP, clock=lambda: later)
    assert out.status == "synced"
    assert out.changed > 0
    store = SQLiteStore(db)
    try:
        assert store.search_fts("zebracorn", limit=5)
    finally:
        store.close()


def test_sync_after_seconds_sets_the_window(vault: Path, db: Path):
    sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    out = sync_before_read(db, str(vault), _config(sync_after_seconds=0), up_running=NEVER_UP)
    assert out.status == "synced"


def test_off_by_config(vault: Path, db: Path):
    out = sync_before_read(db, str(vault), _config(auto_sync=False), up_running=NEVER_UP)
    assert out.status == "off"
    assert not db.exists()


def test_skipped_while_up_runs(vault: Path, db: Path):
    out = sync_before_read(db, str(vault), _config(), up_running=lambda: True)
    assert out.status == "up-running"
    assert not db.exists()


def test_no_vault_configured(db: Path, tmp_path: Path):
    assert sync_before_read(db, None, _config(), up_running=NEVER_UP).status == "no-vault"
    missing = str(tmp_path / "nope")
    assert sync_before_read(db, missing, _config(), up_running=NEVER_UP).status == "no-vault"


def test_concurrent_caller_reads_without_syncing(vault: Path, db: Path):
    db.parent.mkdir(parents=True, exist_ok=True)
    with open(db.with_name(db.name + ".kb-sync.lock"), "a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    assert out.status == "busy"


def test_failure_is_reported_not_raised(vault: Path, db: Path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(kb_sync, "_sync", boom)
    out = sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)
    assert out.status == "failed"


def test_zzz_lines_queue_like_ingest(vault: Path, db: Path, monkeypatch):
    sync_before_read(db, str(vault), _config(), up_running=NEVER_UP)  # first: history skipped
    (vault / "today.md").write_text("# Today\n\nzzz: draft the agenda\n")
    queued = []
    monkeypatch.setattr(
        "openaugi.pipeline.dispatch.record_zzz_changes",
        lambda new, removed, store, vault_path: queued.extend(new),
    )
    later = os.path.getmtime(db.with_name(db.name + ".kb-sync")) + 120
    sync_before_read(db, str(vault), _config(), up_running=NEVER_UP, clock=lambda: later)
    # The splitter moves `zzz:` instructions out of the body; zzz_blocks is
    # the same test ingest uses to pick them out.
    from openaugi.pipeline.dispatch import zzz_blocks

    assert len(zzz_blocks(queued)) == 1


def test_up_lock_probe(tmp_path: Path):
    from openaugi.singleton import is_held

    assert is_held("up", tmp_path) is False
    lock = tmp_path / "up.lock"
    with open(lock, "a+") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # flock is per open file description: a second open() in this process
        # conflicts with it exactly as another process would.
        assert is_held("up", tmp_path) is True
    assert is_held("up", tmp_path) is False


# ── through `openaugi kb` ──────────────────────────────────────────────────


@pytest.fixture
def light_install(vault: Path, db: Path, monkeypatch):
    monkeypatch.setenv("OPENAUGI_DB", str(db))
    monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(vault))
    monkeypatch.setattr(srv, "load_config", lambda *a, **k: _config())
    monkeypatch.setattr("openaugi.singleton.is_held", lambda *a, **k: False)
    srv._store = None
    srv._embedding_model = None
    yield vault
    srv._store = None
    srv._embedding_model = None


def test_kb_answers_from_a_never_ingested_vault(light_install: Path):
    """No `openaugi ingest` ever ran: the first kb call indexes, then answers."""
    result = runner.invoke(app, ["kb", "search", "--keyword", "career"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["count"] > 0


def test_kb_sees_a_note_written_a_moment_ago(light_install: Path, db: Path, monkeypatch):
    runner.invoke(app, ["kb", "search", "--keyword", "career"])
    (light_install / "Just now.md").write_text("# Just now\n\nquokka budget review\n")
    monkeypatch.setattr(kb_sync.time, "time", lambda: 10**10)  # well past the window
    result = runner.invoke(app, ["kb", "search", "--keyword", "quokka"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["count"] == 1


def test_no_sync_flag_answers_from_the_index_as_is(light_install: Path, db: Path):
    result = runner.invoke(app, ["kb", "--no-sync", "search", "--keyword", "career"])
    assert result.exit_code == 0, result.output
    assert not db.with_name(db.name + ".kb-sync").exists()
