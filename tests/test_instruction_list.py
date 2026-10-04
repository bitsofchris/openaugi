"""Tests for scripts/instruction_list.py — the deterministic instruction-marker lister."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "instruction_list.py"
spec = importlib.util.spec_from_file_location("instruction_list", SCRIPT)
il = importlib.util.module_from_spec(spec)
sys.modules["instruction_list"] = il
spec.loader.exec_module(il)

START, END = date(2026, 9, 28), date(2026, 10, 4)

DAILY = """# Journal
A thought about the thing.

Aaa: make a note of this

Another thought, aaa: move it to the backlog
 Aaa: indented one (done 2026-10-04)

How am I using aaa? No colon, so not a hit.
The `aaa:` marker in backticks is prose about the marker.
```
aaa: inside a code fence
```
"""

JOURNAL = """# Journal

### 2026-10-03 — role thinking
Some entry.
aaa: under a dated heading in the window

### 2026-09-01
aaa: under a dated heading outside the window

## Other section
aaa: under no dated heading
"""


def _vault(tmp_path: Path) -> Path:
    (tmp_path / "daily").mkdir()
    (tmp_path / "daily" / "2026-10-02.md").write_text(DAILY, encoding="utf-8")
    (tmp_path / "daily" / "2026-09-20.md").write_text("Aaa: too old\n", encoding="utf-8")
    (tmp_path / "daily" / "2026-10-03 - side note.md").write_text(
        "Aaa: in a dated side note\n", encoding="utf-8"
    )
    (tmp_path / "MOC - Role.md").write_text(JOURNAL, encoding="utf-8")
    (tmp_path / "Generated").mkdir()
    (tmp_path / "Generated" / "2026-10-02 - Board.md").write_text(
        "- aaa: feedback on a generated board\n", encoding="utf-8"
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "2026-10-02.md").write_text("aaa: not a note\n", encoding="utf-8")
    return tmp_path


def _texts(hits) -> list[str]:
    return [h.text for h in hits]


def test_finds_every_marker_line_in_a_dated_note(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    daily = [h for h in hits if h.path == "daily/2026-10-02.md"]
    assert _texts(daily) == [
        "make a note of this",
        "move it to the backlog",
        "indented one (done 2026-10-04)",
    ]
    assert [h.date for h in daily] == ["2026-10-02"] * 3


def test_skips_fences_backticks_and_bare_word(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    joined = " ".join(_texts(hits))
    assert "inside a code fence" not in joined
    assert "prose about the marker" not in joined
    assert "No colon" not in joined


def test_done_suffix_is_reported(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    done = [h for h in hits if h.done]
    assert _texts(done) == ["indented one (done 2026-10-04)"]


def test_filename_date_outside_window_is_dropped(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    assert "too old" not in _texts(hits)


def test_dated_side_note_counts(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    assert "in a dated side note" in _texts(hits)


def test_journal_heading_date_decides(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    journal = [h for h in hits if h.path == "MOC - Role.md"]
    assert _texts(journal) == ["under a dated heading in the window"]
    assert journal[0].date == "2026-10-03"


def test_heading_date_wins_over_filename_date():
    text = "### 2026-10-03\naaa: dated by the heading\n"
    hits = il.scan_text(text, "2026-09-01 - note.md", date(2026, 9, 1), "aaa")
    assert hits[0].date == "2026-10-03"


def test_exclude_and_skip_dirs(tmp_path):
    vault = _vault(tmp_path)
    hits = il.collect(vault, START, END, exclude=("Generated",))
    paths = {h.path for h in hits}
    assert not any(p.startswith("Generated/") or p.startswith(".git/") for p in paths)
    unfiltered = il.collect(vault, START, END, exclude=())
    assert any(h.path.startswith("Generated/") for h in unfiltered)


def test_context_is_the_line_above(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    first = next(h for h in hits if h.text == "make a note of this")
    assert first.context == "A thought about the thing."


def test_other_marker_and_order(tmp_path):
    (tmp_path / "2026-10-01.md").write_text("zzz: run this\naaa: not asked for\n", "utf-8")
    (tmp_path / "2026-09-29.md").write_text("Zzz: earlier one\n", "utf-8")
    hits = il.collect(tmp_path, START, END, marker="zzz", exclude=())
    assert _texts(hits) == ["earlier one", "run this"]


def test_render_lists_count_and_location(tmp_path):
    hits = il.collect(_vault(tmp_path), START, END, exclude=("Generated",))
    out = il.render(hits, START, END, "aaa")
    assert out.startswith(f"{len(hits)} `aaa:` lines, 2026-09-28 to 2026-10-04")
    assert "daily/2026-10-02.md:4" in out
    assert "[done]" in out
