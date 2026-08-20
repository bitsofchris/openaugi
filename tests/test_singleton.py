"""Single-instance guard.

Two `openaugi up` processes over one vault is a correctness bug, not a
performance one: both dispatch the same `zzz:` and two agents race over one
database. These tests use REAL subprocesses, because the whole point of an
advisory lock is what the kernel does across process boundaries — an in-process
test would prove nothing.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from openaugi.singleton import AlreadyRunning, acquire


def _child(lock_dir, script: str) -> subprocess.CompletedProcess:
    """Run `script` in a separate interpreter with the package importable."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script).replace("__DIR__", str(lock_dir))],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_a_second_process_cannot_take_a_held_lock(tmp_path):
    acquire("guard", tmp_path)
    result = _child(
        tmp_path,
        """
        import pathlib
        from openaugi.singleton import acquire, AlreadyRunning
        try:
            acquire("guard", pathlib.Path("__DIR__"))
            print("ACQUIRED")
        except AlreadyRunning as exc:
            print("BLOCKED", exc.pid)
        """,
    )
    assert "BLOCKED" in result.stdout, result.stderr
    # The pid is what makes the error actionable — "already running" without
    # saying by whom leaves you running pkill blind.
    assert str(os.getpid()) in result.stdout


def test_the_lock_dies_with_its_holder(tmp_path):
    """The reason this is flock and not a PID file.

    A PID file left by a crashed process needs a liveness check, and every
    liveness check races. The kernel releases an flock when the process ends,
    however it ends — so there is no stale state to reason about.
    """
    holder = _child(
        tmp_path,
        """
        import pathlib
        from openaugi.singleton import acquire
        acquire("guard", pathlib.Path("__DIR__"))
        print("HELD")
        """,
    )
    assert "HELD" in holder.stdout, holder.stderr
    # That process has exited. The lock file still exists, with a dead pid in
    # it — and must not block anyone.
    assert (tmp_path / "guard.lock").exists()
    acquire("guard", tmp_path)


def test_a_blocked_attempt_does_not_erase_the_holders_pid(tmp_path):
    acquire("guard", tmp_path)
    before = (tmp_path / "guard.lock").read_text()
    for _ in range(3):
        _child(
            tmp_path,
            """
            import pathlib
            from openaugi.singleton import acquire, AlreadyRunning
            try:
                acquire("guard", pathlib.Path("__DIR__"))
            except AlreadyRunning:
                pass
            """,
        )
    # Opening the file 'w' instead of 'a+' would truncate it here, leaving the
    # running instance anonymous and every later error message useless.
    assert (tmp_path / "guard.lock").read_text() == before


def test_different_names_do_not_collide(tmp_path):
    acquire("alpha", tmp_path)
    acquire("beta", tmp_path)  # must not raise


def test_creates_the_lock_directory(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    path = acquire("guard", nested)
    assert path.exists()


def test_acquiring_twice_in_one_process_also_raises(tmp_path):
    """flock is per open file DESCRIPTION, not per process.

    A second `acquire` opens a second fd, which the kernel treats as a
    different owner — so it blocks, exactly as a second process would. That is
    stricter than "one process may re-enter", and it is the behaviour we want:
    a double-acquire is a bug in the caller, and failing loudly beats a
    silently reference-counted lock.
    """
    acquire("guard", tmp_path)
    with pytest.raises(AlreadyRunning):
        acquire("guard", tmp_path)


def test_already_running_carries_the_lock_path(tmp_path):
    acquire("guard", tmp_path)
    proc = _child(
        tmp_path,
        """
        import pathlib
        from openaugi.singleton import acquire, AlreadyRunning
        try:
            acquire("guard", pathlib.Path("__DIR__"))
        except AlreadyRunning as exc:
            print("PATH", exc.lock_path)
        """,
    )
    assert str(tmp_path / "guard.lock") in proc.stdout


def test_raises_the_documented_type(tmp_path):
    acquire("guard", tmp_path)
    # Callers catch this by name; it must stay a RuntimeError subclass so an
    # over-broad `except RuntimeError` upstream keeps working.
    assert issubclass(AlreadyRunning, RuntimeError)
    with pytest.raises(AlreadyRunning):
        raise AlreadyRunning(tmp_path / "guard.lock", "123")
