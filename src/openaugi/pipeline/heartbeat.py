"""The watcher's heartbeat — proof, on disk, that the tick is still running.

Since the launchd cutover there is one process. No `openaugi up` means no
ingest, no `zzz:` dispatch, no board, no habit parse, no janitor — and nothing
reports it. The failure mode is not an error; it is a morning where the board
simply isn't there, indistinguishable from a morning with nothing to say.

The alarm cannot be raised by the watcher, because the watcher is the thing
that is down. So the watcher writes a small view on every drain tick —
`OpenAugi/Views/View - System Heartbeat.md`, frontmatter `last_tick`, the
running commit, the pid, one row per scheduled lens — and the alarm is
rendered *at read time* from that file: a `dataviewjs` block on the Dashboard
compares `last_tick` to the clock, and `openaugi doctor` does the same in the
terminal. A stale tick is the signal; the file being absent is the loudest
form of stale.

Two guards keep the heartbeat from being its own problem:

* **Throttled to once per five minutes**, by reading the previous `last_tick`
  back off disk. No in-memory state, so a restarted watcher throttles
  correctly against the file its predecessor left.
* **Never ingested.** The path is in `adapters/vault.SYSTEM_EXCLUDE_PATTERNS`,
  which both the vault parser and the watcher's event handler apply
  unconditionally — a heartbeat that re-triggered the ingest that writes it
  would tick forever.

No LLM calls in this module.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from openaugi.adapters.vault import HEARTBEAT_VIEW
from openaugi.pipeline.context_pack import _FRONTMATTER_RE
from openaugi.pipeline.schedule import lens_status
from openaugi.service_version import SERVICE_STATE_COLLECTION, UP_RECORD_ID, head_sha

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

#: How often the file is rewritten while the watcher is alive.
HEARTBEAT_INTERVAL = timedelta(minutes=5)

#: A tick older than this means the watcher is down. Twice the interval, so
#: one slow tick (a long ingest riding the debounce) is not a false alarm.
STALE_AFTER = timedelta(minutes=10)


def heartbeat_path(vault: Path) -> Path:
    return vault / HEARTBEAT_VIEW


def read_heartbeat(vault: Path) -> dict[str, Any] | None:
    """The last heartbeat's frontmatter, or None if there has never been one.

    `last_tick` comes back as an aware datetime; an unreadable stamp is
    treated as no heartbeat at all, which is the safe direction — a file that
    cannot say when it was written cannot vouch for anything.
    """
    path = heartbeat_path(vault)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        logger.warning("Heartbeat frontmatter is unreadable; treating it as absent")
        return None
    if not isinstance(data, dict):
        return None
    tick = data.get("last_tick")
    if isinstance(tick, datetime):
        data["last_tick"] = tick if tick.tzinfo else tick.replace(tzinfo=UTC)
    elif isinstance(tick, str):
        try:
            data["last_tick"] = datetime.fromisoformat(tick)
        except ValueError:
            logger.warning("Heartbeat last_tick %r is unreadable; treating it as absent", tick)
            return None
    else:
        return None
    return data


def tick_age(vault: Path, now: datetime | None = None) -> timedelta | None:
    """How long ago the last tick was, or None if there has never been one."""
    beat = read_heartbeat(vault)
    if beat is None:
        return None
    return (now or datetime.now(UTC)) - beat["last_tick"]


def is_stale(vault: Path, now: datetime | None = None) -> bool:
    """Whether the watcher should be presumed down. Absent counts as stale."""
    age = tick_age(vault, now)
    return age is None or age > STALE_AFTER


def running_commit(store: SQLiteStore) -> str | None:
    """The SHA the daemon started from — what is actually executing.

    `up` stamps it into the `service_state` record. A watcher started some
    other way (`openaugi watch`, a test) has no record, and then the checkout
    is the best available answer.
    """
    rows = store.list_records(SERVICE_STATE_COLLECTION, limit=10)
    row = next((r for r in rows if r["id"] == UP_RECORD_ID), None)
    if row and row.get("sha"):
        return row["sha"]
    return head_sha()


def _stamp(when: datetime | None) -> str:
    return when.isoformat(timespec="seconds") if when else ""


def render_heartbeat(
    now: datetime, commit: str | None, pid: int, lenses: list[dict[str, Any]]
) -> str:
    """The view's text.

    Frontmatter is the machine-readable half — the Dashboard block reads
    `last_tick` and `lenses` from it — and the table is the same rows for a
    person.
    """
    interval_minutes = int(HEARTBEAT_INTERVAL.total_seconds() // 60)
    stale_minutes = int(STALE_AFTER.total_seconds() // 60)
    lines = [
        "---",
        "type: view",
        "description: >-",
        "  Liveness of the OpenAugi watcher — the last drain tick, the code it runs,",
        "  and every scheduled lens's last run and next due. Rewritten every",
        f"  {interval_minutes} minutes while `openaugi up` runs; a stale tick means",
        "  the watcher is down.",
        f"last_tick: {_stamp(now)}",
        f"commit: {commit or ''}",
        f"pid: {pid}",
        f"stale_after_minutes: {stale_minutes}",
        "lenses:",
    ]
    for row in lenses:
        lines += [
            f"  - name: {row['name']}",
            f"    trigger: {row['trigger']}",
            f"    period_seconds: {int(row['period'].total_seconds())}",
            f"    last_run: {_stamp(row['last_run'])}",
            f"    next_due: {_stamp(row['next_due'])}",
            f"    overdue: {'true' if row['overdue'] else 'false'}",
        ]
    lines += [
        "---",
        "",
        "# System Heartbeat",
        "",
        f"Last tick: `{_stamp(now)}` · commit `{(commit or 'unknown')[:12]}` · pid `{pid}`",
        "",
        "Regenerated by the watcher's drain tick. If `last_tick` is more than "
        f"{stale_minutes} minutes old, `openaugi up` is not running: "
        "`launchctl kickstart -k gui/$(id -u)/com.openaugi.up`.",
        "",
        "| Lens | Trigger | Last run | Next due | |",
        "|---|---|---|---|---|",
    ]
    for row in lenses:
        flag = "**overdue**" if row["overdue"] else ""
        lines.append(
            f"| {row['name']} | `{row['trigger']}` | {_stamp(row['last_run']) or '—'} "
            f"| {_stamp(row['next_due'])} | {flag} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_heartbeat(
    vault: Path,
    store: SQLiteStore,
    now: datetime | None = None,
    pid: int | None = None,
    force: bool = False,
) -> Path | None:
    """Write the heartbeat view if the last one is old enough. None if throttled.

    The throttle reads the previous file rather than remembering the last
    write, so it holds across a restart and costs nothing to test.
    """
    when = now or datetime.now(UTC)
    if not force:
        previous = read_heartbeat(vault)
        if previous is not None:
            age = when - previous["last_tick"]
            if timedelta(0) <= age < HEARTBEAT_INTERVAL:
                return None
    path = heartbeat_path(vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_heartbeat(
        when, running_commit(store), pid or os.getpid(), lens_status(vault, store, when)
    )
    path.write_text(text, encoding="utf-8")
    logger.debug("Heartbeat written: %s", path.name)
    return path
