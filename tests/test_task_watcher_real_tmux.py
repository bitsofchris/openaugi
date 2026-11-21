"""launch_tmux against the real tmux binary, no mocks.

The other task-watcher tests use a fake tmux, which answers the way we
assume tmux answers. On 2026-11-20 that assumption was wrong: with a tmux
server already running, `display-message -t =<missing>:` exits 0 with
empty output, so a missing session read as "still running" and every
scheduled task was skipped. The fake said otherwise, so the tests passed.

These tests run real tmux on a private socket (TMUX_TMPDIR in a fresh
temp dir), so they never touch the user's own tmux server. The stand-in
for claude is a script that ignores its arguments and sleeps.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from openaugi.agents import task_watcher as tw

TMUX = shutil.which("tmux")

pytestmark = pytest.mark.skipif(TMUX is None, reason="tmux not installed")


@pytest.fixture
def real_tmux(monkeypatch):
    """An isolated tmux server namespace and a fake claude; torn down after."""
    # /tmp, not pytest's tmp_path: macOS caps a socket path at ~104 chars.
    sockdir = Path(tempfile.mkdtemp(prefix="oa-tmux-", dir="/tmp"))
    monkeypatch.setenv("TMUX_TMPDIR", str(sockdir))
    monkeypatch.delenv("TMUX", raising=False)  # inside tmux, $TMUX would win

    fake_claude = sockdir / "fake-claude"
    fake_claude.write_text("#!/bin/sh\nexec sleep 60\n")
    fake_claude.chmod(0o755)

    assert TMUX is not None
    yield TMUX, str(fake_claude)

    subprocess.run([TMUX, "kill-server"], capture_output=True, check=False)
    shutil.rmtree(sockdir, ignore_errors=True)


def session_exists(tmux: str, name: str) -> bool:
    r = subprocess.run([tmux, "has-session", "-t", f"={name}"], capture_output=True, check=False)
    return r.returncode == 0


def start_session(tmux: str, name: str) -> None:
    subprocess.run([tmux, "new-session", "-d", "-s", name, "sleep", "60"], check=True)


def test_launches_when_no_tmux_server_is_running(real_tmux):
    tmux, claude = real_tmux
    assert tw.launch_tmux(tmux, claude, "TASK-a", "prompt") is True
    assert session_exists(tmux, "TASK-a")


def test_launches_when_another_session_is_running(real_tmux):
    """The 2026-11-21 failure: an unrelated live session blocked every launch."""
    tmux, claude = real_tmux
    start_session(tmux, "TASK-yesterday-still-open")

    assert tw.launch_tmux(tmux, claude, "TASK-today", "prompt") is True
    assert session_exists(tmux, "TASK-today")


def test_skips_when_the_same_session_is_running(real_tmux):
    tmux, claude = real_tmux
    start_session(tmux, "TASK-a")

    assert tw.launch_tmux(tmux, claude, "TASK-a", "prompt") is False


def test_tmux_is_isolated_from_the_users_server(real_tmux):
    """Guard the fixture itself: the socket lives under the temp TMUX_TMPDIR."""
    tmux, _ = real_tmux
    start_session(tmux, "TASK-probe")
    r = subprocess.run(
        [tmux, "display-message", "-p", "-t", "=TASK-probe:", "#{socket_path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    # /tmp is a symlink to /private/tmp on macOS; compare resolved paths.
    socket = Path(r.stdout.strip()).resolve()
    assert socket.is_relative_to(Path(os.environ["TMUX_TMPDIR"]).resolve()), socket
