---
name: heartbeat
description: How a dead watcher becomes visible. The drain tick writes `OpenAugi/Views/View - System Heartbeat.md` (last tick, running commit, pid, every scheduled lens's last run and next due); the Dashboard renders the alarm from it at read time and `openaugi doctor` prints the same in the terminal, exiting non-zero when the tick is stale.
---

# System heartbeat and `openaugi doctor`

## When to use this doc

- The board, the habit parse or a scheduled lens did not show up and you want
  to know whether anything is running at all
- You are wiring an alarm (Dashboard, janitor, cron) to the watcher's liveness
- You are changing what the drain tick writes, or how "stale" is decided

## The problem it answers

Since the launchd cutover there is **one process**. No `openaugi up` means no
ingest, no `zzz:` dispatch, no board, no habit parse, no janitor — and nothing
reported it. The failure mode was not an error; it was a morning where the
board simply wasn't there, indistinguishable from a morning with nothing to
say. The only check was a command you had to remember to type.

The design constraint that decides the shape: **the alarm cannot be raised by
the watcher**, because the watcher is the thing that is down. It has to be
rendered *at read time*, from state already on disk.

## The heartbeat file

`pipeline/heartbeat.py`. On every drain tick (`pipeline/watcher.py:_drain_tick`,
the same timer that runs the lens schedule) the watcher writes
`OpenAugi/Views/View - System Heartbeat.md` — Views are the one legal
overwrite target. **Throttled to once every 5 minutes** by reading the previous
file's `last_tick` back off disk, so a restarted watcher throttles correctly
against the file its predecessor left.

Frontmatter is the machine-readable half; the body is the same rows as a table:

```yaml
---
type: view
last_tick: 2026-09-20T12:55:27+00:00   # UTC, ISO 8601
commit: <sha the daemon started from>  # service_state record, else HEAD
pid: 12345
stale_after_minutes: 10
lenses:
  - name: currency-board
    trigger: every 1d
    period_seconds: 86400
    last_run: 2026-09-20T10:00:00+00:00
    next_due: 2026-09-21T10:00:00+00:00
    overdue: false
---
```

`overdue` means the next due time is more than one full period in the past —
a slot has been missed outright, not merely reached. That is the line the
Dashboard's overdue-lenses list draws. Lenses that never fire on a tick
(`on-demand`, `on-pass`, a malformed trigger) are not rows.

**It never triggers ingest.** The path is in
`adapters/vault.SYSTEM_EXCLUDE_PATTERNS`, which both the vault parser and the
watcher's event handler apply *unconditionally*. It is deliberately not a
config default: a `[vault] exclude_patterns` list in config replaces the
defaults wholesale, and one missing line would make the heartbeat re-trigger
the ingest that writes it, forever. Verify in `~/.openaugi/logs/up.err`: no
"Detected 1 changed file(s)" every five minutes.

## `openaugi doctor`

The same answer in the terminal, for when Obsidian is not open. Read-only.

```
$ openaugi doctor
watcher:  running (pid 12345)
code:     3c8d900a1b2c (current)
tick:     2026-09-20T12:55+00:00 (2m ago) ok

lens                 trigger      last run                   next due
currency-board       every 1d     2026-09-20T10:00+00:00     2026-09-21T10:00+00:00
substack-batch       every 7d     2026-09-18T10:30+00:00     2026-09-25T10:30+00:00
```

| Line | Source |
|---|---|
| `watcher` | the `up` singleton lock (`singleton.holder`) — the kernel's answer, not a pid file's |
| `code` | `service_state` SHA vs `HEAD` (`service_version.version_drift`), the same check `openaugi status` banners |
| `tick` | the heartbeat file's `last_tick` against the clock |
| lens rows | `schedule.lens_status` — live from the registry and the ledger, not from the heartbeat file |

**Exit code 1 when the tick is stale** (older than 10 minutes, or no file at
all), so a janitor can fire on it the way it already fires on code drift.
Fix: `launchctl kickstart -k gui/$(id -u)/com.openaugi.up`.

## The Dashboard block

The alarm on the Dashboard is a `dataviewjs` block that reads the heartbeat
page's frontmatter and compares `last_tick` to `Date.now()` — Dataview
re-renders on open, so it is live whether or not anything is running. The
Dashboard lives in the vault and is the user's; the block is documented in the
Command Deck, not shipped here. It shows one quiet line when the tick is
fresh, a red "watcher down since …" line with the restart command when it is
stale, and names any lens whose `overdue` is true.

## Tuning

| Knob | Where | Default |
|---|---|---|
| write interval | `heartbeat.HEARTBEAT_INTERVAL` | 5 min |
| stale threshold | `heartbeat.STALE_AFTER` (also written into the file as `stale_after_minutes`) | 10 min |
| tick frequency | `[tasks] zzz_settle_seconds / 4`, floor 5 s (`watcher._watch_loop`) | 30 s |

Code: `src/openaugi/pipeline/heartbeat.py`, `src/openaugi/doctor.py`,
`src/openaugi/singleton.py` (`holder`). Tests: `tests/test_heartbeat.py`,
`tests/test_doctor.py`, `tests/test_watcher.py::TestDrainTick`.

Related: [lenses.md](lenses.md) (scheduling), [task-dispatch.md](task-dispatch.md)
(the stale-code banner).
