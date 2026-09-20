"""Tests for stale-daemon detection.

The bug this exists to catch: `openaugi up` had been running since 2026-09-04
with a dispatch fix committed on the 11th sitting inert in the checkout. The
system looked healthy from every angle — the tests passed, the code was
correct, the ledger was being written — and none of that was running.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from openaugi.service_version import (
    SERVICE_STATE_COLLECTION,
    UP_RECORD_ID,
    head_sha,
    record_service_start,
    version_drift,
)
from openaugi.store.sqlite import SQLiteStore


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git checkout with one commit."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("one")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=tmp_path, check=True)
    return tmp_path


def _commit(repo: Path, text: str) -> str:
    (repo / "a.txt").write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", text], cwd=repo, check=True)
    sha = head_sha(repo)
    assert sha
    return sha


def test_no_drift_when_nothing_has_started(store: SQLiteStore):
    assert version_drift(store) is None


def test_no_drift_when_the_daemon_runs_head(store: SQLiteStore, repo: Path, monkeypatch):
    monkeypatch.setattr("openaugi.service_version.package_root", lambda: repo)
    record_service_start(store, "2026-09-13T21:50:00", pid=123)

    assert version_drift(store) is None


def test_drift_is_reported_once_head_moves(store: SQLiteStore, repo: Path, monkeypatch):
    """The 2026-09-13 case: daemon started, two commits landed, nothing said so."""
    monkeypatch.setattr("openaugi.service_version.package_root", lambda: repo)
    started = head_sha(repo)
    record_service_start(store, "2026-09-04T09:00:00", pid=81819)

    _commit(repo, "two")
    head = _commit(repo, "three")

    drift = version_drift(store)
    assert drift is not None
    assert drift["running_sha"] == started
    assert drift["head_sha"] == head
    assert drift["commits_behind"] == 2
    assert drift["started_at"] == "2026-09-04T09:00:00"
    assert drift["pid"] == 81819


def test_drift_survives_an_unreachable_running_sha(store: SQLiteStore, repo: Path, monkeypatch):
    """A rebase can orphan the SHA the daemon started from.

    That is still drift — it just cannot be counted, and an uncountable gap
    must not silently read as no gap.
    """
    monkeypatch.setattr("openaugi.service_version.package_root", lambda: repo)
    store.write_record(
        SERVICE_STATE_COLLECTION,
        UP_RECORD_ID,
        {"started_at": "2026-09-04T09:00:00", "pid": 1, "sha": "0" * 40},
        "2026-09-04T09:00:00",
    )

    drift = version_drift(store)
    assert drift is not None
    assert drift["commits_behind"] is None


def test_a_non_git_install_reports_no_drift(store: SQLiteStore, tmp_path: Path, monkeypatch):
    """A wheel install has no HEAD to compare against — silence, not noise."""
    monkeypatch.setattr("openaugi.service_version.package_root", lambda: tmp_path)
    record_service_start(store, "2026-09-13T21:50:00", pid=1)

    assert version_drift(store) is None


def test_liveness_is_none_before_any_start(store: SQLiteStore):
    from openaugi.service_version import service_liveness

    assert service_liveness(store) is None


def test_liveness_reports_a_running_pid(store: SQLiteStore, monkeypatch):
    import os

    from openaugi.service_version import record_service_start, service_liveness

    monkeypatch.setattr("openaugi.service_version.head_sha", lambda root=None: "abc")
    record_service_start(store, "2026-09-20T10:00:00", os.getpid())
    live = service_liveness(store)
    assert live == {"pid": os.getpid(), "started_at": "2026-09-20T10:00:00", "alive": True}


def test_liveness_reports_a_dead_pid(store: SQLiteStore, monkeypatch):
    from openaugi.service_version import record_service_start, service_liveness

    monkeypatch.setattr("openaugi.service_version.head_sha", lambda root=None: "abc")

    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr("openaugi.service_version.os.kill", gone)
    record_service_start(store, "2026-09-20T10:00:00", 424242)
    assert service_liveness(store)["alive"] is False
