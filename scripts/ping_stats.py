#!/usr/bin/env python3
"""ping_stats.py — count the ping lines the phone writes into daily notes.

The pinger (an iOS Shortcut, see docs/reference/pings.md) appends one line per
check-in to the daily note:

    - 14:35 ping: level=2 mood=low place=work event=food note=slack
    - 15:10 event: event=food acted=no level=2 mood=low place=work note=slack

This script reads those lines over a window and prints the cross-tabs the
Sunday `ping-read` lens interprets. It only counts; it never opines.

Usage:
    python3 scripts/ping_stats.py --vault "~/Documents/ZK Home" --days 7
    python3 scripts/ping_stats.py --days 7 --json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

DAILY_DIR = "_private/0-Fleeting-Inbox"

LINE_RE = re.compile(r"^\s*-?\s*(?P<time>\d{1,2}:\d{2})\s+(?P<kind>ping|event):\s*(?P<rest>.*)$")
KV_RE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")

MOOD = ("low", "ok", "high")
PLACE = ("home", "office", "errands", "others", "other")
EVENT = ("none", "bite", "snack", "play", "tap", "scroll", "sip")


@dataclass
class Ping:
    day: str
    time: str
    kind: str  # "ping" (scheduled) | "event" (on-demand)
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def hour(self) -> int:
        return int(self.time.split(":")[0])

    @property
    def event(self) -> str:
        return self.fields.get("event", "none").lower()

    @property
    def has_event(self) -> bool:
        return self.event not in ("", "none")


def parse_line(day: str, line: str) -> Ping | None:
    m = LINE_RE.match(line)
    if not m:
        return None
    fields = {k.lower(): v.strip() for k, v in KV_RE.findall(m.group("rest"))}
    return Ping(day=day, time=m.group("time"), kind=m.group("kind"), fields=fields)


def parse_note(day: str, text: str) -> list[Ping]:
    out = []
    for line in text.splitlines():
        p = parse_line(day, line)
        if p:
            out.append(p)
    return out


def load_window(vault: Path, days: int, end: date | None = None) -> list[Ping]:
    end = end or date.today()
    pings: list[Ping] = []
    for i in range(days):
        d = end - timedelta(days=i)
        path = vault / DAILY_DIR / f"{d.isoformat()}.md"
        if path.exists():
            pings.extend(parse_note(d.isoformat(), path.read_text(encoding="utf-8")))
    pings.sort(key=lambda p: (p.day, p.time))
    return pings


def _rate(num: int, den: int) -> float:
    return round(num / den, 2) if den else 0.0


def _hour_bucket(h: int) -> str:
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    return "evening"


def summarize(pings: list[Ping]) -> dict:
    scheduled = [p for p in pings if p.kind == "ping"]
    on_demand = [p for p in pings if p.kind == "event"]
    days = sorted({p.day for p in pings})

    def event_rate_by(key: str, values: tuple[str, ...], pool: list[Ping]) -> dict[str, dict]:
        out = {}
        for v in values:
            group = [p for p in pool if p.fields.get(key, "").lower() == v]
            if group:
                out[v] = {
                    "n": len(group),
                    "event_rate": _rate(sum(p.has_event for p in group), len(group)),
                }
        return out

    by_bucket: dict[str, list[Ping]] = defaultdict(list)
    for p in scheduled:
        by_bucket[_hour_bucket(p.hour)].append(p)

    urges_all = [p for p in pings if p.has_event]
    event_kinds = Counter(p.event for p in urges_all)
    note_before_event = Counter(
        p.fields.get("note", "").lower() for p in urges_all if p.fields.get("note")
    )
    note_all = Counter(
        p.fields.get("note", "").lower() for p in scheduled if p.fields.get("note")
    )
    level_at_food = [
        int(p.fields["level"])
        for p in urges_all
        if p.event == "food" and p.fields.get("level", "").isdigit()
    ]
    acted = Counter(p.fields.get("acted", "?").lower() for p in on_demand)
    place_away = sum(1 for p in scheduled if p.fields.get("place", "").lower() not in ("", "here"))

    return {
        "days": days,
        "scheduled_pings": len(scheduled),
        "on_demand_urges": len(on_demand),
        "pings_per_day": _rate(len(scheduled), len(days)),
        "event_rate_overall": _rate(sum(p.has_event for p in scheduled), len(scheduled)),
        "event_kinds": dict(event_kinds.most_common()),
        "event_rate_by_time": {
            b: {"n": len(v), "event_rate": _rate(sum(p.has_event for p in v), len(v))}
            for b, v in by_bucket.items()
        },
        "event_rate_by_mood": event_rate_by("mood", ENERGY, scheduled),
        "event_rate_by_place": event_rate_by("place", MIND, scheduled),
        "place_away_rate": _rate(place_away, len(scheduled)),
        "place_where": dict(
            Counter(
                p.fields.get("place", "").lower() for p in scheduled if p.fields.get("place")
            ).most_common()
        ),
        "note_before_event": dict(note_before_event.most_common(5)),
        "note_all": dict(note_all.most_common(5)),
        "level_at_food_urge_mean": round(sum(level_at_food) / len(level_at_food), 1)
        if level_at_food
        else None,
        "on_demand_acted": dict(acted),
    }


def render_md(s: dict) -> str:
    first = s["days"][0] if s["days"] else "—"
    last = s["days"][-1] if s["days"] else "—"
    level = s["level_at_food_urge_mean"]
    lines = [
        f"**Window:** {first} → {last} · {s['scheduled_pings']} pings "
        f"({s['pings_per_day']}/day) · {s['on_demand_urges']} event taps",
        "",
    ]
    lines += ["| cut | n | event rate |", "|---|---|---|"]
    lines.append(f"| all pings | {s['scheduled_pings']} | {s['event_rate_overall']} |")
    for name, table in (
        ("time", s["event_rate_by_time"]),
        ("mood", s["event_rate_by_mood"]),
        ("place", s["event_rate_by_place"]),
    ):
        for k, v in table.items():
            lines.append(f"| {name}={k} | {v['n']} | {v['event_rate']} |")
    lines.append("")
    lines.append(f"- events by kind: {s['event_kinds'] or 'none'}")
    lines.append(
        f"- mind somewhere other than here: {s['place_away_rate']} of pings"
        f" · where: {s['place_where'] or '—'}"
    )
    lines.append(f"- doing right before an event: {s['note_before_event'] or '—'}")
    lines.append(f"- doing at all pings: {s['note_all'] or '—'}")
    lines.append(f"- level when the event was food (1–5): {level if level is not None else '—'}")
    lines.append(f"- event taps acted on: {s['on_demand_acted'] or '—'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vault", default="~/Documents/ZK Home")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--end", help="last day of the window, YYYY-MM-DD (default: today)")
    ap.add_argument(
        "--json", action="store_true", help="print the summary as JSON instead of markdown"
    )
    args = ap.parse_args()
    end = date.fromisoformat(args.end) if args.end else None
    pings = load_window(Path(args.vault).expanduser(), args.days, end)
    s = summarize(pings)
    print(json.dumps(s, indent=2) if args.json else render_md(s))


if __name__ == "__main__":
    main()
