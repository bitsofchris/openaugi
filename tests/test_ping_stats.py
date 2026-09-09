"""Tests for scripts/ping_stats.py — the ping-line parser and cross-tabs."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ping_stats.py"
spec = importlib.util.spec_from_file_location("ping_stats", SCRIPT)
ps = importlib.util.module_from_spec(spec)
sys.modules["ping_stats"] = ps
spec.loader.exec_module(ps)

NOTE = """# 2026-09-09
Some prose that is not a ping.
- 07:40 ping: level=3 mood=ok place=here event=none note=coffee
- 09:20 ping: level=2 mood=low place=work event=food note=slack
- 14:35 ping: level=2 mood=low place=work event=game note=debugging
- 15:10 event: event=food acted=no level=2 mood=low place=work note=slack
- 19:40 ping: level=4 mood=ok place=money event=none note=dinner prep
- 09:15 pings are not this: level=1
"""


def test_parse_line_scheduled_and_on_demand():
    p = ps.parse_line(
        "2026-09-09", "- 09:20 ping: level=2 mood=low place=work event=food note=slack"
    )
    assert p.kind == "ping" and p.hour == 9 and p.event == "food" and p.has_event
    assert p.fields["note"] == "slack"
    u = ps.parse_line("2026-09-09", "- 15:10 event: event=food acted=no level=2")
    assert u.kind == "event" and u.fields["acted"] == "no"
    assert ps.parse_line("2026-09-09", "not a ping line") is None


def test_multiword_doing_is_kept():
    p = ps.parse_line(
        "d", "- 19:40 ping: level=4 mood=ok place=money event=none note=dinner prep"
    )
    assert p.fields["note"] == "dinner prep"
    assert not p.has_event


def test_parse_note_skips_prose_and_lookalikes():
    pings = ps.parse_note("2026-09-09", NOTE)
    assert [p.time for p in pings] == ["07:40", "09:20", "14:35", "15:10", "19:40"]


def test_summarize_cross_tabs():
    s = ps.summarize(ps.parse_note("2026-09-09", NOTE))
    assert s["scheduled_pings"] == 4 and s["on_demand_urges"] == 1
    assert s["event_rate_overall"] == 0.5
    assert s["event_rate_by_mood"]["low"] == {"n": 2, "event_rate": 1.0}
    assert s["event_rate_by_mood"]["ok"] == {"n": 2, "event_rate": 0.0}
    assert s["event_rate_by_time"]["morning"]["n"] == 2
    assert s["event_kinds"] == {"food": 2, "game": 1}
    assert s["place_away_rate"] == 0.75
    assert s["level_at_food_urge_mean"] == 2.0
    assert s["on_demand_acted"] == {"no": 1}
    assert list(s["note_before_event"])[0] == "slack"


def test_summarize_empty_is_safe():
    s = ps.summarize([])
    assert s["scheduled_pings"] == 0 and s["pings_per_day"] == 0.0
    assert "0 pings" in ps.render_md(s)


def test_load_window_reads_daily_notes(tmp_path):
    day_dir = tmp_path / ps.DAILY_DIR
    day_dir.mkdir(parents=True)
    (day_dir / "2026-09-09.md").write_text(NOTE)
    (day_dir / "2026-09-08.md").write_text(
        "- 12:00 ping: level=1 mood=high place=here event=none note=lift\n"
    )
    pings = ps.load_window(tmp_path, days=3, end=date(2026, 9, 9))
    assert len(pings) == 6
    assert pings[0].day == "2026-09-08"
    md = ps.render_md(ps.summarize(pings))
    assert "2026-09-08 → 2026-09-09" in md
    assert "| mood=high | 1 | 0.0 |" in md
