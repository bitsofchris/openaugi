"""`openaugi doctor` — is the one process that runs everything actually running?

The terminal twin of the Dashboard's heartbeat block, for when Obsidian is
not open. Four questions, answered from state already on disk:

1. Is `openaugi up` alive? — the singleton lock, asked without taking it.
2. Is it running the committed code? — `service_state` SHA against HEAD.
3. When did the drain tick last run? — the heartbeat view's `last_tick`.
4. What has each scheduled lens done, and when is it next due?

The exit code is the point: **non-zero when the tick is stale** (or absent),
so a janitor can fire on `openaugi doctor` the way it already fires on code
drift. Everything here is a read; the command changes nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openaugi.pipeline.heartbeat import STALE_AFTER, read_heartbeat
from openaugi.pipeline.schedule import describe_anchor, lens_status
from openaugi.service_version import head_sha, version_drift
from openaugi.singleton import holder

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

RESTART = "launchctl kickstart -k gui/$(id -u)/com.openaugi.up"


def diagnose(
    vault: Path,
    store: SQLiteStore,
    now: datetime | None = None,
    lock_dir: Path | None = None,
    stale_after: timedelta = STALE_AFTER,
    tz: tzinfo | None = None,
) -> dict[str, Any]:
    """Everything `doctor` prints, as data. `healthy` is what the exit code follows.

    `tz` is the schedule's zone (`schedule.schedule_timezone(config)`), for
    the anchored lenses' next-due; None means the system zone.
    """
    when = now or datetime.now(UTC)
    beat = read_heartbeat(vault)
    age = (when - beat["last_tick"]) if beat else None
    stale = age is None or age > stale_after
    return {
        "now": when,
        "watcher_pid": holder("up", lock_dir),
        "drift": version_drift(store),
        "head": head_sha(),
        "last_tick": beat["last_tick"] if beat else None,
        "tick_commit": (beat or {}).get("commit"),
        "tick_pid": (beat or {}).get("pid"),
        "tick_age": age,
        "stale": stale,
        "stale_after": stale_after,
        "lenses": lens_status(vault, store, when, tz),
        "healthy": not stale,
    }


def _minutes(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h {(total % 3600) // 60}m"
    return f"{total // 86400}d {(total % 86400) // 3600}h"


def _stamp(when: datetime | None) -> str:
    return when.isoformat(timespec="minutes") if when else "never"


def render(report: dict[str, Any]) -> str:
    """The report as plain lines. Rich markup is kept out so tests and
    cron-mailed output read the same."""
    lines = []
    pid = report["watcher_pid"]
    lines.append(f"watcher:  {'running (pid ' + pid + ')' if pid else 'NOT RUNNING'}")

    drift = report["drift"]
    head = report["head"]
    if drift:
        behind = drift["commits_behind"]
        trailer = f", {behind} commit{'s' if behind != 1 else ''} behind" if behind else ""
        lines.append(
            f"code:     STALE — running {drift['running_sha'][:12]}, "
            f"HEAD {drift['head_sha'][:12]}{trailer}"
        )
    else:
        lines.append(f"code:     {head[:12] if head else 'not a git checkout'} (current)")

    if report["last_tick"] is None:
        lines.append("tick:     never — no heartbeat file")
    else:
        age = _minutes(report["tick_age"])
        verdict = "STALE" if report["stale"] else "ok"
        lines.append(f"tick:     {_stamp(report['last_tick'])} ({age} ago) {verdict}")

    if report["stale"]:
        lines.append(
            f"\nThe drain tick has not run for more than "
            f"{int(report['stale_after'].total_seconds() // 60)} minutes: no scheduled lens "
            f"will fire and no zzz will dispatch.\nRestart it: {RESTART}"
        )

    lenses = report["lenses"]
    if lenses:
        lines.append("")
        lines.append(
            f"{'lens':<20} {'trigger':<12} {'anchor':<10} {'last run':<26} {'next due':<26}"
        )
        for row in lenses:
            flag = "  OVERDUE" if row["overdue"] else ""
            anchor = describe_anchor(row.get("anchor")) or "-"
            lines.append(
                f"{row['name']:<20} {row['trigger']:<12} {anchor:<10} "
                f"{_stamp(row['last_run']):<26} {_stamp(row['next_due']):<26}{flag}"
            )
    else:
        lines.append("\nno scheduled lenses")
    return "\n".join(lines)
