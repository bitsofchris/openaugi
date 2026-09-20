"""Scheduling — the `trigger:` field becomes real.

Every lens spec has carried a `trigger:` since July (`on-demand | on-pass |
every <period>`), the contract has validated it, and nothing has ever acted on
it. Personal cadences went out-of-band instead: a shell script per surface and
a launchd agent per script, each one encoding a folder name, a time of day and
a prompt in the engine's own repo.

This module reads the field instead. A lens is three things — a trigger, a
scope, and a prompt — and all three already live in the vault file. So the
scheduler owns no vocabulary of its own: it lists the registry, asks each spec
whether it is due, and turns the due ones into pending task files that the task
watcher launches exactly as it launches a `zzz` dispatch. `zzz` is the same
mechanism with the trigger being "someone just typed it".

Nothing here is wired to the watcher; `pipeline/watcher.py` calls
`due_lenses` on its drain tick. Two guards keep a typo from becoming an agent
session:

* **The gate.** The loop runs only when `tasks.schedule_lenses` is on in
  config. Default off — a vault that has never heard of scheduling behaves
  exactly as before.
* **Malformed triggers are skipped, never guessed at.** A spec that already
  failed the contract, or whose period will not parse, is logged and dropped.
  Being unable to read a cadence is not a reason to invent one.

The ledger records the **slot, not the tick**. A late fire — the boundary
landed while the vault was busy, or the machine was asleep — is stamped as the
scheduled boundary it belongs to (`previous + n·period`, the latest one not
after now), never as the moment the tick happened to run. Otherwise every slip
would re-anchor the cadence and the drift only ever goes forward: a 06:00 board
becomes a 06:04 board, then 06:11. A three-day sleep produces one catch-up run,
not three, and lands the lens back on its own grid.

Deduplication is deliberately two-layered, because "already ran" has two
meanings. The schedule records the last run per lens, which answers *is it
due*; the optional `## Run` section's `dedupe:` line names an output path,
which answers *did the work already land* — a board that exists for today is
proof, whatever the records say about a re-imported database or a machine that
was asleep.

No LLM calls in this module.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openaugi.pipeline.context_pack import _FRONTMATTER_RE, LENSES_DIR, read_lens_specs
from openaugi.pipeline.dispatch import DEFAULT_TASKS_FOLDER

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

#: Last-run ledger, one row per lens name (docs/reference/records.md).
LENS_SCHEDULE_COLLECTION = "lens_schedule"

#: Triggers that never fire on a tick. `on-pass` is the review pass's to own.
_NEVER_DUE = ("", "on-demand", "on-pass")

#: `every 1d`, `every 12h`, `every: 7d` — the quoted colon form is legal YAML
#: and the contract accepts it, so parse both. Units are the ones a cadence is
#: actually written in; anything else is a typo, not a unit to guess at.
_EVERY_RE = re.compile(r"^every[:\s]\s*(\d+)\s*([smhdw])$", re.IGNORECASE)
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}

#: The optional `## Run` section — what a scheduled run needs that an
#: on-demand run gets from the person asking for it.
_RUN_SECTION_RE = re.compile(
    r"^##[ \t]+Run[ \t]*$\n(.*?)(?=^##[ \t]|\Z)", re.MULTILINE | re.DOTALL
)
#: `dedupe: OpenAugi/Board/{date} - Board.md` — one line inside `## Run`.
_DEDUPE_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?dedupe:[ \t]*(?P<path>\S.*?)[ \t]*$", re.MULTILINE)


def scheduling_enabled(config: dict[str, Any]) -> bool:
    """Whether the trigger loop runs at all. Off unless config says otherwise."""
    return bool(config.get("tasks", {}).get("schedule_lenses", False))


def parse_period(trigger: str) -> timedelta | None:
    """The cadence an `every <period>` trigger declares, or None if it has none.

    None means "do not schedule this", for every reason: `on-demand`,
    `on-pass`, an empty trigger, and a malformed one alike. Callers that need
    to tell a typo from a deliberate opt-out check the prefix themselves — see
    `due_lenses`, which logs the difference.
    """
    m = _EVERY_RE.match(trigger.strip())
    if not m:
        return None
    count = int(m.group(1))
    if count <= 0:
        return None
    return timedelta(**{_UNITS[m.group(2).lower()]: count})


def read_run_section(vault: Path, spec: dict) -> tuple[str, str | None]:
    """The lens body's `## Run` prose and the dedupe path pattern in it.

    Returns `("", None)` when the lens has no `## Run` section, which is the
    common case — the section only exists for the two things a scheduled run
    cannot ask a human for: what state to read first, and how to tell that
    today's run already happened.
    """
    path = vault / LENSES_DIR / spec.get("file", f"{spec['name']}.md")
    if not path.is_file():
        return "", None
    text = path.read_text(encoding="utf-8")
    body = text[m.end() :] if (m := _FRONTMATTER_RE.match(text)) else text
    section = _RUN_SECTION_RE.search(body)
    if not section:
        return "", None
    run = section.group(1).strip()
    dedupe = _DEDUPE_RE.search(run)
    return run, dedupe.group("path") if dedupe else None


def expand(pattern: str, when: datetime) -> str:
    """`OpenAugi/Board/{date} - Board.md` → today's path. `{date}` is ISO."""
    return pattern.replace("{date}", when.strftime("%Y-%m-%d"))


def task_filename(name: str, when: datetime) -> str:
    """The deterministic task filename for one lens on one day.

    Deterministic on purpose: the file's own existence is the cheapest
    "already queued today" check there is, and it survives a lost database.
    """
    return f"TASK-{when.strftime('%Y-%m-%d')}-{name}.md"


def last_run(store: SQLiteStore, name: str) -> datetime | None:
    """When this lens last had a task written for it, if ever."""
    rows = store.list_records(LENS_SCHEDULE_COLLECTION, where={"lens": name}, limit=1)
    if not rows:
        return None
    try:
        return datetime.fromisoformat(rows[0]["last_run"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Lens %s has an unreadable last_run; treating it as never run", name)
        return None


def slot_for(previous: datetime | None, period: timedelta, now: datetime) -> datetime:
    """The scheduled boundary a run at `now` belongs to.

    The latest point on the lens's own grid (`previous + n·period`) that is
    not after `now` — so a fire forty minutes late records the boundary it
    missed, and a fire three days late skips to the most recent boundary
    rather than replaying the ones in between. Without a previous run there
    is no grid yet, and `now` becomes its origin.
    """
    if previous is None or now < previous:
        return now
    missed = (now - previous) // period
    return previous + period * max(missed, 1)


def record_run(
    store: SQLiteStore,
    name: str,
    when: datetime,
    task: str,
    period: timedelta | None = None,
) -> None:
    """Stamp a lens's run into the ledger. One row per lens, replaced in place.

    With a `period`, what is stamped is the slot the run belongs to (see
    `slot_for`), not `when` itself; the tick is late, the grid is not. Without
    one, `when` is stamped verbatim — that is how a ledger is seeded by hand.
    """
    if period is not None:
        when = slot_for(last_run(store, name), period, when)
    stamp = when.isoformat()
    store.write_record(
        LENS_SCHEDULE_COLLECTION,
        name,
        {"lens": name, "last_run": stamp, "task": task},
        stamp,
    )


def due_lenses(
    vault: Path,
    store: SQLiteStore,
    now: datetime | None = None,
    tasks_folder: str = DEFAULT_TASKS_FOLDER,
) -> list[dict]:
    """Every lens whose cadence has come round, in registry order.

    Each returned spec carries what `write_lens_task` needs: the contract
    fields plus `period` and the `## Run` prose. A lens is skipped when it
    declares no cadence, when its spec already violates the contract, when its
    task file for today is already on disk, or when its `dedupe:` output
    exists — each for a different reason, all of them logged.
    """
    when = now or datetime.now(UTC)
    due = []
    for spec in read_lens_specs(vault):
        name = spec["name"]
        trigger = (spec.get("trigger") or "").strip()
        if trigger.lower() in _NEVER_DUE:
            continue
        if spec.get("error"):
            logger.warning(
                "Lens %s violates the contract, not scheduling it: %s", name, spec["error"]
            )
            continue
        period = parse_period(trigger)
        if period is None:
            logger.warning(
                "Lens %s has an unreadable trigger %r, not scheduling it", name, trigger
            )
            continue
        previous = last_run(store, name)
        if previous is not None and when - previous < period:
            continue
        if (vault / tasks_folder / task_filename(name, when)).exists():
            logger.debug("Lens %s already has a task file for today", name)
            continue
        run, dedupe = read_run_section(vault, spec)
        if dedupe and (vault / expand(dedupe, when)).exists():
            logger.debug("Lens %s already produced %s", name, expand(dedupe, when))
            continue
        due.append({**spec, "period": period, "run": run, "dedupe": dedupe})
    return due


def build_lens_task(spec: dict, vault: Path, when: datetime) -> str:
    """The pending task file for one scheduled lens run.

    Deliberately the same shape as a `zzz` dispatch and a board proposal, so
    the task watcher needs to know nothing about schedules. Every personal
    sentence in it — what to read first, what the scope is, what the bar is —
    comes out of the lens file, never out of this module.
    """
    day = when.strftime("%Y-%m-%d")
    name = spec["name"]
    description = (spec.get("description") or "").strip()
    scope = (spec.get("scope") or "").strip()
    run = (spec.get("run") or "").strip()

    context = f"Scheduled run: `{spec.get('trigger', '').strip()}`."
    if description:
        context += f" {description}"
    scope_line = f"\nDefault scope: {scope}\n" if scope else ""
    run_block = f"\n{run}\n" if run else ""

    return f"""---
status: pending
working_dir: {vault}
source: lens-schedule
lens: {name}
run: {day}
---

# {name} — {day}

## Context

{context}

## User instruction

> apply lens {name}

## Task

Apply the `{name}` lens (`{LENSES_DIR}/{name}.md`). Its body is the
instruction — follow its Intent, Process and Hard rules exactly, and write to
the target its `target:` field names.
{scope_line}{run_block}
## Human Todo

## Results
"""


def write_lens_task(
    spec: dict,
    vault: Path,
    now: datetime | None = None,
    tasks_folder: str = DEFAULT_TASKS_FOLDER,
) -> Path | None:
    """Write one scheduled lens's pending task file. None if it already exists.

    Returning None rather than overwriting is what makes a tick idempotent: a
    watcher that restarts mid-cadence re-reads the same registry and finds the
    work already queued.
    """
    when = now or datetime.now(UTC)
    tasks_dir = vault / tasks_folder
    tasks_dir.mkdir(parents=True, exist_ok=True)
    path = tasks_dir / task_filename(spec["name"], when)
    if path.exists():
        return None
    path.write_text(build_lens_task(spec, vault, when), encoding="utf-8")
    logger.info("Scheduled lens %s → %s", spec["name"], path.name)
    return path


def run_due_lenses(
    vault: Path,
    store: SQLiteStore,
    config: dict[str, Any],
    now: datetime | None = None,
) -> list[Path]:
    """The whole tick: what is due becomes a task file, and the run is recorded.

    Gated on `tasks.schedule_lenses`; an off gate returns an empty list without
    reading the registry at all.
    """
    if not scheduling_enabled(config):
        return []
    when = now or datetime.now(UTC)
    tasks_folder = config.get("tasks", {}).get("folder", DEFAULT_TASKS_FOLDER)
    written = []
    for spec in due_lenses(vault, store, when, tasks_folder):
        path = write_lens_task(spec, vault, when, tasks_folder)
        if path is None:
            continue
        record_run(store, spec["name"], when, path.name, period=spec["period"])
        written.append(path)
    return written
