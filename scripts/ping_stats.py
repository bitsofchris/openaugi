#!/usr/bin/env python3
"""ping_stats.py — count the structured check-in lines in the daily notes.

A phone-side prompt (see docs/reference/pings.md) appends one line per
check-in to the day's daily note. The grammar is generic:

    - [HH:MM] <kind>: key=value key=value ...

The time is optional; the last value may run to several words. Which kinds
exist, which keys they carry and what the values are called is the user's
own vocabulary and lives in the vault lens that invokes this script, never
here. The script discovers keys and values from the data.

Two roles are declared on the command line: the *scheduled* kinds (samples
taken on a timer) and the *on-demand* kinds (the user pressed a button).
One key is the *target*: the thing being rated. A line where the target is
present (not empty, not an "absent" word such as `none`) counts as a hit,
and the script cross-tabs the hit rate by time of day and by every other
key it finds. It only counts; it never opines.

Usage:
    python3 scripts/ping_stats.py --days 7                       # vault from config
    python3 scripts/ping_stats.py --vault ~/vault --days 7 --json
    python3 scripts/ping_stats.py --scheduled ping --ondemand event \\
        --target event --free note --days 7
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DAILY_DIR = "_private/0-Fleeting-Inbox"
DEFAULT_SCHEDULED = ("ping",)
DEFAULT_ONDEMAND = ("event",)
DEFAULT_TARGET = "event"
DEFAULT_ABSENT = ("none",)
TOP_N = 5

LINE_RE = re.compile(r"^\s*-?\s*(?:(?P<time>\d{1,2}:\d{2})\s+)?(?P<kind>[\w-]+):\s*(?P<rest>.*)$")
KV_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")


@dataclass
class Ping:
    day: str
    kind: str
    time: str | None = None
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def hour(self) -> int | None:
        return int(self.time.split(":")[0]) if self.time else None


@dataclass(frozen=True)
class Spec:
    """Which kinds play which role, and which key is being rated."""

    scheduled: tuple[str, ...] = DEFAULT_SCHEDULED
    ondemand: tuple[str, ...] = DEFAULT_ONDEMAND
    target: str = DEFAULT_TARGET
    free: str | None = None
    absent: tuple[str, ...] = DEFAULT_ABSENT

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self.scheduled) | frozenset(self.ondemand)

    def hit(self, p: Ping) -> bool:
        """True when the target key is present on the line with a non-absent value."""
        return p.fields.get(self.target, "") not in ("", *self.absent)


def parse_line(day: str, line: str, kinds: frozenset[str]) -> Ping | None:
    """One note line -> Ping, or None when it is prose or a kind we were not told about.

    Keys and values are lower-cased so the cross-tabs merge `Mood=Low` with
    `mood=low`. A line of the right kind with no `key=value` pairs at all
    (`- event: free text`) still counts as an event with no fields.
    """
    m = LINE_RE.match(line)
    if not m or m.group("kind").lower() not in kinds:
        return None
    fields = {k.lower(): v.strip().lower() for k, v in KV_RE.findall(m.group("rest"))}
    return Ping(day=day, kind=m.group("kind").lower(), time=m.group("time"), fields=fields)


def parse_note(day: str, text: str, kinds: frozenset[str]) -> list[Ping]:
    out = []
    for line in text.splitlines():
        p = parse_line(day, line, kinds)
        if p:
            out.append(p)
    return out


def load_window(
    vault: Path,
    days: int,
    kinds: frozenset[str],
    end: date | None = None,
    daily_dir: str = DEFAULT_DAILY_DIR,
) -> list[Ping]:
    end = end or date.today()
    pings: list[Ping] = []
    for i in range(days):
        d = end - timedelta(days=i)
        path = vault / daily_dir / f"{d.isoformat()}.md"
        if path.exists():
            pings.extend(parse_note(d.isoformat(), path.read_text(encoding="utf-8"), kinds))
    pings.sort(key=lambda p: (p.day, p.time or ""))
    return pings


def _rate(num: int, den: int) -> float:
    return round(num / den, 2) if den else 0.0


def _hour_bucket(h: int) -> str:
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    return "evening"


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _rate_table(groups: dict[str, list[Ping]], spec: Spec) -> dict[str, dict]:
    return {
        name: {"n": len(pool), "rate": _rate(sum(spec.hit(p) for p in pool), len(pool))}
        for name, pool in groups.items()
        if pool
    }


def summarize(pings: list[Ping], spec: Spec) -> dict:
    """Cross-tabs of the target's hit rate, with every key and value discovered from the data.

    Rates are computed over the scheduled samples only; on-demand lines are
    self-selected and would inflate them. Value counts run over every line.
    """
    scheduled = [p for p in pings if p.kind in spec.scheduled]
    ondemand = [p for p in pings if p.kind in spec.ondemand]
    days = sorted({p.day for p in pings})
    hits = [p for p in pings if spec.hit(p)]

    keys = sorted({k for p in pings for k in p.fields})
    crosstab_keys = [k for k in keys if k not in (spec.target, spec.free)]

    by_time: dict[str, list[Ping]] = defaultdict(list)
    for p in scheduled:
        if p.hour is not None:
            by_time[_hour_bucket(p.hour)].append(p)

    rate_by: dict[str, dict] = {}
    for key in crosstab_keys:
        groups: dict[str, list[Ping]] = defaultdict(list)
        for p in scheduled:
            if key in p.fields:
                groups[p.fields[key]].append(p)
        table = _rate_table(dict(sorted(groups.items())), spec)
        if table:
            rate_by[key] = table

    def counts(pool: list[Ping], key: str) -> dict[str, int]:
        return dict(Counter(p.fields[key] for p in pool if key in p.fields).most_common())

    numeric_by_target: dict[str, dict[str, float]] = {}
    for key in crosstab_keys:
        seen = [
            (p.fields.get(spec.target, "") or spec.absent[0], p.fields[key])
            for p in pings
            if key in p.fields
        ]
        if not seen or not all(_is_number(v) for _, v in seen):
            continue
        per_value: dict[str, list[float]] = defaultdict(list)
        for target_value, v in seen:
            per_value[target_value].append(float(v))
        numeric_by_target[key] = {
            tv: round(sum(vals) / len(vals), 1) for tv, vals in sorted(per_value.items())
        }

    return {
        "days": days,
        "spec": {
            "scheduled": list(spec.scheduled),
            "ondemand": list(spec.ondemand),
            "target": spec.target,
            "free": spec.free,
        },
        "scheduled": len(scheduled),
        "ondemand": len(ondemand),
        "per_day": _rate(len(scheduled), len(days)),
        "rate_overall": _rate(sum(spec.hit(p) for p in scheduled), len(scheduled)),
        "rate_by_time": _rate_table(dict(by_time), spec),
        "rate_by": rate_by,
        "target_values": dict(Counter(p.fields[spec.target] for p in hits).most_common()),
        "values": {k: counts(pings, k) for k in keys},
        "ondemand_values": {k: counts(ondemand, k) for k in keys if counts(ondemand, k)},
        "numeric_by_target": numeric_by_target,
        "free_before": dict(
            Counter(p.fields[spec.free] for p in hits if spec.free in p.fields).most_common(TOP_N)
        )
        if spec.free
        else {},
        "free_all": dict(
            Counter(p.fields[spec.free] for p in scheduled if spec.free in p.fields).most_common(
                TOP_N
            )
        )
        if spec.free
        else {},
    }


def render_md(s: dict) -> str:
    target = s["spec"]["target"]
    first = s["days"][0] if s["days"] else "—"
    last = s["days"][-1] if s["days"] else "—"
    lines = [
        f"**Window:** {first} → {last} · {s['scheduled']} scheduled "
        f"({s['per_day']}/day) · {s['ondemand']} on-demand",
        "",
        f"| cut | n | {target} rate |",
        "|---|---|---|",
        f"| all scheduled | {s['scheduled']} | {s['rate_overall']} |",
    ]
    for bucket, v in s["rate_by_time"].items():
        lines.append(f"| time={bucket} | {v['n']} | {v['rate']} |")
    for key, table in s["rate_by"].items():
        for value, v in table.items():
            lines.append(f"| {key}={value} | {v['n']} | {v['rate']} |")
    lines.append("")
    lines.append(f"- {target} values: {s['target_values'] or 'none'}")
    for key, table in s["values"].items():
        if key != target:
            lines.append(f"- {key} values: {table}")
    for key, table in s["numeric_by_target"].items():
        lines.append(f"- mean {key} by {target}: {table}")
    if s["spec"]["free"]:
        free = s["spec"]["free"]
        lines.append(f"- {free} right before a {target}: {s['free_before'] or '—'}")
        lines.append(f"- {free} at all scheduled: {s['free_all'] or '—'}")
    lines.append(f"- on-demand values: {s['ondemand_values'] or '—'}")
    return "\n".join(lines)


def resolve_vault(explicit: str | None) -> Path | None:
    """The vault root: the flag if given, else the openaugi config's default_path.

    The config lookup is optional so the script stays runnable as a plain
    file; without the package installed, --vault is simply required.
    """
    if explicit:
        return Path(explicit).expanduser()
    try:
        from openaugi.config import load_config, resolve_vault_path
    except ImportError:
        return None
    resolved = resolve_vault_path(None, load_config())
    return Path(resolved) if resolved else None


def _csv(values: list[str]) -> tuple[str, ...]:
    """Flatten repeated and comma-separated flag values into one lower-cased tuple."""
    return tuple(v.strip().lower() for chunk in values for v in chunk.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--vault",
        default=None,
        help="vault root (default: [vault] default_path from the openaugi config)",
    )
    ap.add_argument(
        "--daily-dir",
        default=DEFAULT_DAILY_DIR,
        help="folder of YYYY-MM-DD.md daily notes, relative to the vault",
    )
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--end", help="last day of the window, YYYY-MM-DD (default: today)")
    ap.add_argument(
        "--scheduled",
        action="append",
        default=[],
        metavar="KIND",
        help="line kind(s) written on a timer (repeatable or comma-separated; "
        f"default: {','.join(DEFAULT_SCHEDULED)})",
    )
    ap.add_argument(
        "--ondemand",
        action="append",
        default=[],
        metavar="KIND",
        help=f"line kind(s) written on demand (default: {','.join(DEFAULT_ONDEMAND)})",
    )
    ap.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        metavar="KEY",
        help=f"the key whose presence is the thing being rated (default: {DEFAULT_TARGET})",
    )
    ap.add_argument(
        "--free",
        default=None,
        metavar="KEY",
        help="a free-text key to report 'what preceded a hit' from (default: none)",
    )
    ap.add_argument(
        "--absent",
        action="append",
        default=[],
        metavar="VALUE",
        help=f"target value(s) that mean 'nothing' (default: {','.join(DEFAULT_ABSENT)})",
    )
    ap.add_argument(
        "--json", action="store_true", help="print the summary as JSON instead of markdown"
    )
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    vault = resolve_vault(args.vault)
    if vault is None:
        ap.error("no vault: pass --vault or set [vault] default_path in the openaugi config")
    spec = Spec(
        scheduled=_csv(args.scheduled) or DEFAULT_SCHEDULED,
        ondemand=_csv(args.ondemand) or DEFAULT_ONDEMAND,
        target=args.target.lower(),
        free=args.free.lower() if args.free else None,
        absent=_csv(args.absent) or DEFAULT_ABSENT,
    )
    end = date.fromisoformat(args.end) if args.end else None
    pings = load_window(vault, args.days, spec.kinds, end, args.daily_dir)
    s = summarize(pings, spec)
    print(json.dumps(s, indent=2) if args.json else render_md(s))


if __name__ == "__main__":
    main()
