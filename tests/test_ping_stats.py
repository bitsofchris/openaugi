"""Tests for scripts/ping_stats.py — the generic check-in line parser and cross-tabs.

The vocabulary here is a placeholder (`mood=`, `place=`, `level=`, `note=`).
The real field names live in the vault lens that invokes the script.
"""

import importlib.util
import sys
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ping_stats.py"
spec = importlib.util.spec_from_file_location("ping_stats", SCRIPT)
ps = importlib.util.module_from_spec(spec)
sys.modules["ping_stats"] = ps
spec.loader.exec_module(ps)

SPEC = ps.Spec(scheduled=("ping",), ondemand=("event",), target="event", free="note")
KINDS = SPEC.kinds

NOTE = """# 2026-09-09
Some prose that is not a check-in.
- 07:40 ping: level=3 mood=ok place=home event=none note=coffee
- 09:20 ping: level=2 mood=low place=office event=snack note=email
- 14:35 ping: level=2 mood=low place=office event=scroll note=debugging
- 15:10 event: event=snack acted=no level=2 mood=low place=office note=email
- 19:40 ping: level=4 mood=ok place=out event=none note=dinner prep
- 09:15 pings are not this: level=1
- TODO: not a check-in either
"""


def test_parse_line_scheduled_and_on_demand():
    p = ps.parse_line(
        "2026-09-09", "- 09:20 ping: level=2 mood=low place=office event=snack note=email", KINDS
    )
    assert p.kind == "ping" and p.hour == 9 and SPEC.hit(p)
    assert p.fields["note"] == "email"
    e = ps.parse_line("2026-09-09", "- 15:10 event: event=snack acted=no level=2", KINDS)
    assert e.kind == "event" and e.fields["acted"] == "no"
    assert ps.parse_line("2026-09-09", "not a check-in line", KINDS) is None


def test_only_declared_kinds_count():
    assert ps.parse_line("d", "- 09:00 TODO: fix the thing", KINDS) is None
    assert ps.parse_line("d", "- Note: prose with a colon", KINDS) is None
    assert ps.parse_line("d", "- 09:00 todo: x=1", frozenset({"todo"})).kind == "todo"


def test_time_is_optional_and_free_form_line_still_counts():
    p = ps.parse_line("d", "- event: ate a cookie, felt low", KINDS)
    assert p is not None and p.kind == "event" and p.time is None and p.hour is None
    assert p.fields == {}
    assert not SPEC.hit(p)
    q = ps.parse_line("d", "ping: mood=ok", KINDS)
    assert q.time is None and q.fields == {"mood": "ok"}


def test_unknown_keys_and_multiword_values_are_kept():
    p = ps.parse_line(
        "d", "- 19:40 ping: level=4 Mood=OK place=out event=none note=dinner prep", KINDS
    )
    assert p.fields["note"] == "dinner prep"
    assert p.fields["mood"] == "ok"  # keys and values are lower-cased
    assert not SPEC.hit(p)
    r = ps.parse_line("d", "- 10:00 ping: weather=rain event=snack", KINDS)
    assert r.fields == {"weather": "rain", "event": "snack"}


def test_absent_values_are_configurable():
    p = ps.parse_line("d", "- 10:00 ping: event=-", KINDS)
    assert SPEC.hit(p)
    assert not ps.Spec(absent=("none", "-")).hit(p)


def test_parse_note_skips_prose_and_lookalikes():
    pings = ps.parse_note("2026-09-09", NOTE, KINDS)
    assert [p.time for p in pings] == ["07:40", "09:20", "14:35", "15:10", "19:40"]


def test_summarize_cross_tabs_by_discovered_values():
    s = ps.summarize(ps.parse_note("2026-09-09", NOTE, KINDS), SPEC)
    assert s["scheduled"] == 4 and s["ondemand"] == 1
    assert s["rate_overall"] == 0.5
    assert s["rate_by"]["mood"]["low"] == {"n": 2, "rate": 1.0}
    assert s["rate_by"]["mood"]["ok"] == {"n": 2, "rate": 0.0}
    assert s["rate_by"]["place"]["office"] == {"n": 2, "rate": 1.0}
    assert s["rate_by_time"]["morning"] == {"n": 2, "rate": 0.5}
    assert "event" not in s["rate_by"] and "note" not in s["rate_by"]
    assert s["target_values"] == {"snack": 2, "scroll": 1}
    assert s["values"]["place"] == {"office": 3, "home": 1, "out": 1}
    assert s["numeric_by_target"] == {"level": {"none": 3.5, "scroll": 2.0, "snack": 2.0}}
    assert s["ondemand_values"]["acted"] == {"no": 1}
    assert list(s["free_before"])[0] == "email"
    assert s["free_all"] == {"coffee": 1, "email": 1, "debugging": 1, "dinner prep": 1}


def test_on_demand_lines_do_not_inflate_rates():
    lines = "- 10:00 ping: mood=ok event=none\n- 10:30 event: event=snack mood=ok\n"
    s = ps.summarize(ps.parse_note("d", lines, KINDS), SPEC)
    assert s["rate_overall"] == 0.0 and s["rate_by"]["mood"]["ok"]["n"] == 1
    assert s["target_values"] == {"snack": 1}


def test_lines_without_time_are_left_out_of_the_time_buckets():
    lines = "- ping: mood=ok event=snack\n- 10:00 ping: mood=ok event=none\n"
    s = ps.summarize(ps.parse_note("d", lines, KINDS), SPEC)
    assert s["scheduled"] == 2 and s["rate_overall"] == 0.5
    assert s["rate_by_time"] == {"morning": {"n": 1, "rate": 0.0}}


def test_summarize_empty_is_safe():
    s = ps.summarize([], SPEC)
    assert s["scheduled"] == 0 and s["per_day"] == 0.0
    assert "0 scheduled" in ps.render_md(s)


def test_render_md_names_the_target_and_free_key():
    md = ps.render_md(ps.summarize(ps.parse_note("2026-09-09", NOTE, KINDS), SPEC))
    assert "| cut | n | event rate |" in md
    assert "| mood=low | 2 | 1.0 |" in md
    assert "- note right before a event:" in md
    assert "- mean level by event:" in md


def test_load_window_reads_daily_notes(tmp_path):
    day_dir = tmp_path / ps.DEFAULT_DAILY_DIR
    day_dir.mkdir(parents=True)
    (day_dir / "2026-09-09.md").write_text(NOTE)
    (day_dir / "2026-09-08.md").write_text(
        "- 12:00 ping: level=1 mood=high place=home event=none\n"
    )
    pings = ps.load_window(tmp_path, days=3, kinds=KINDS, end=date(2026, 9, 9))
    assert len(pings) == 6
    assert pings[0].day == "2026-09-08"
    md = ps.render_md(ps.summarize(pings, SPEC))
    assert "2026-09-08 → 2026-09-09" in md
    assert "| mood=high | 1 | 0.0 |" in md


def test_load_window_takes_a_custom_daily_dir(tmp_path):
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "2026-09-09.md").write_text("- 08:00 ping: event=none\n")
    pings = ps.load_window(tmp_path, days=1, kinds=KINDS, end=date(2026, 9, 9), daily_dir="daily")
    assert len(pings) == 1


def test_cli_flags_build_the_spec(tmp_path, capsys):
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "2026-09-09.md").write_text(
        "- 08:00 sample: mood=ok flag=none\n- 09:00 tap: flag=yes mood=low\n"
    )
    sys.argv = [
        "ping_stats",
        "--vault",
        str(tmp_path),
        "--daily-dir",
        "daily",
        "--days",
        "1",
        "--end",
        "2026-09-09",
        "--scheduled",
        "sample",
        "--ondemand",
        "tap",
        "--target",
        "flag",
        "--json",
    ]
    ps.main()
    out = capsys.readouterr().out
    assert '"scheduled": 1' in out and '"ondemand": 1' in out
    assert '"target": "flag"' in out
