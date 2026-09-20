"""Single-instance guard for long-running commands.

`openaugi up` starts a vault watcher AND a task dispatcher. Two of them over
one vault is not a degraded state, it is a correctness bug: both watchers see
the same `zzz:` block land, both dispatch it, and two agents run the same task
concurrently against one database. Observed 2026-08-20 — a single "run the
review pass" produced two tmux sessions three seconds apart.

**Why an advisory lock and not a PID file.** A PID file has to answer "is that
process still alive?", and every answer races: the process can die between the
check and the decision, and PIDs are reused. `flock` has no such problem — the
kernel releases it when the holding process dies, however it dies. There is no
stale state to reason about, and no cleanup path to get wrong.

What this does NOT cover: two *machines* sharing a synced vault. A lock is
local to a filesystem, so a second Mac running its own watcher is invisible
here. That needs an atomic claim on the task file itself, which is a separate
piece of work.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

# The lock is held by keeping its file descriptor open. Module-level so it
# outlives the acquiring function and is not closed by garbage collection —
# closing the fd releases the lock, which would silently defeat the guard.
_held: list = []


class AlreadyRunning(RuntimeError):
    """Another instance holds the lock. Carries its pid when readable."""

    def __init__(self, lock_path: Path, pid: str | None):
        self.lock_path = lock_path
        self.pid = pid
        super().__init__(f"another instance is already running{f' (pid {pid})' if pid else ''}")


def acquire(name: str, lock_dir: Path | None = None) -> Path:
    """Take the named lock for the rest of this process, or raise.

    Returns the lock file path. Raises `AlreadyRunning` if held elsewhere.

    Call this ONCE, at startup. `flock` is per open file description rather
    than per process, so calling it twice in one process raises too — which is
    deliberate: a double-acquire is a caller bug, and failing loudly beats a
    silently reference-counted lock.
    """
    directory = lock_dir or Path.home() / ".openaugi"
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"{name}.lock"

    # Opened 'a+' so an existing lock file is never truncated before we know
    # whether we can have it — truncating first would erase the running
    # instance's pid and make the error message useless.
    # noqa SIM115 is the point, not an oversight: a context manager would
    # close this fd on the way out, and closing it releases the lock. The
    # descriptor has to outlive this function — that is what "held for the
    # process lifetime" means.
    fd = open(lock_path, "a+")  # noqa: SIM115
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.seek(0)
        pid = fd.read().strip() or None
        fd.close()
        raise AlreadyRunning(lock_path, pid) from None

    # We hold it. Now it is safe to replace the contents with our own pid.
    fd.seek(0)
    fd.truncate()
    fd.write(str(os.getpid()))
    fd.flush()
    _held.append(fd)
    return lock_path
