"""What code the daemon is actually running, and whether it still matches HEAD.

A long-lived process loads its modules once. `openaugi up` runs for weeks
under launchd, so a fix committed today does not run until someone restarts
it — and nothing anywhere says so.

That is not hypothetical. On 2026-09-13 the service had been up since the 4th.
A dispatch fix committed on the 11th had therefore never executed once: the
ledger it writes to had 134 rows and not one carried the field that fix
introduced. The first visible symptom was three phantom tasks, and the cause
looked like a logic bug in code that was already correct.

So `up` stamps the SHA it started from, and anything that audits the system
compares it against what is committed now. An editable install means the two
diverge the moment you commit, which is exactly the window worth naming.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

# Service lifecycle state: which process started when, running what.
SERVICE_STATE_COLLECTION = "service_state"
UP_RECORD_ID = "up"

_GIT_TIMEOUT = 5.0


def package_root() -> Path:
    """The checkout this package is imported from.

    Only meaningful for an editable install, which is how the daemon runs.
    """
    return Path(__file__).resolve().parent.parent.parent


def head_sha(root: Path | None = None) -> str | None:
    """The commit currently checked out, or None if this is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root or package_root(),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def record_service_start(store: SQLiteStore, stamp: str, pid: int) -> None:
    """Stamp the SHA this process started from. Called once, by `up`."""
    store.write_record(
        SERVICE_STATE_COLLECTION,
        UP_RECORD_ID,
        {"started_at": stamp, "pid": pid, "sha": head_sha()},
        stamp,
    )


def service_liveness(store: SQLiteStore) -> dict | None:
    """Is the `up` process that last recorded a start still alive?

    None when no daemon has ever recorded a start. Otherwise the recorded
    pid and start time, with `alive` from a signal-0 probe — the cheapest
    honest answer to "is OpenAugi running?", and the one `status` prints
    first. launchd restarts the process when it dies; this line is for the
    moment before it has, or for a machine where launchd is not loaded.
    """
    rows = store.list_records(SERVICE_STATE_COLLECTION, limit=10)
    row = next((r for r in rows if r["id"] == UP_RECORD_ID), None)
    if row is None:
        return None
    pid = row.get("pid")
    alive = False
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
    return {"pid": pid, "started_at": row.get("started_at"), "alive": alive}


def version_drift(store: SQLiteStore) -> dict | None:
    """Describe the gap between the running daemon and the committed code.

    Returns None when there is nothing to say — no daemon has recorded a
    start, or it is running what is committed. Otherwise a dict with the
    running sha, the current head, and when the daemon started, so the
    caller can decide how loudly to say it.
    """
    rows = store.list_records(SERVICE_STATE_COLLECTION, limit=10)
    row = next((r for r in rows if r["id"] == UP_RECORD_ID), None)
    if row is None:
        return None
    running = row.get("sha")
    head = head_sha()
    if not running or not head or running == head:
        return None
    return {
        "running_sha": running,
        "head_sha": head,
        "started_at": row.get("started_at"),
        "pid": row.get("pid"),
        "commits_behind": _commits_between(running, head),
    }


def _commits_between(running: str, head: str) -> int | None:
    """How many commits the daemon is behind, or None if it cannot be counted.

    A force-push or a rebase can leave the running SHA unreachable. That is
    still drift worth reporting, just not drift worth counting.
    """
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"{running}..{head}"],
            cwd=package_root(),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None
