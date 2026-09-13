"""Routing janitor — the master box, answers, aaa: overrides, extend, undo, feedback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.pipeline import augi_log, route, routing_janitor
from openaugi.pipeline.routing_janitor import (
    insert_extend,
    process_log,
    remove_extend,
)
from openaugi.pipeline.writeback import FEEDBACK_LOG

DAY = "2026-09-03"
DAILY = f"_private/0-Fleeting-Inbox/{DAY}.md"
PMOC = "PMOC - Audacity to take Action - Season 2 - Q2 2026"
PMOC_PATH = f"_private/3-MOCs and Projects/{PMOC}.md"
AMOC = "AMOC - OpenAugi Main"
AMOC_PATH = f"_private/3-MOCs and Projects/{AMOC}.md"
NOTE = "Tornado Antidote - Read When Tornado Spins"
NOTE_PATH = f"_private/0-Fleeting-Inbox/{NOTE}.md"

PMOC_TEXT = (
    "---\ndescription: The active season. Route build blocks here.\n---\n"
    "#status/active #note-type/pmoc\n\n### 2026-09-01\nfirst of the month\n\n"
    "### 2026-08-23\nolder\n"
)
AMOC_TEXT = (
    "---\ndescription: The product. Route product blocks here.\n---\n"
    "#note-type/amoc\n\n[[Links]]\n\n# Journal\n\n### 2026-01-14\nold entry\n"
)


def _block(content: str, bid: str, path: str = DAILY) -> Block:
    return Block(
        id=bid,
        kind="data_block",
        content=content,
        title=Path(path).stem,
        block_time=DAY,
        metadata={"source_path": path, "parent_note_title": Path(path).stem},
    )


def _doc(title: str, path: str) -> Block:
    return Block(
        id=Block.make_document_id(path),
        kind="context_block:document",
        title=title,
        metadata={"source_path": path},
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "_private/3-MOCs and Projects").mkdir(parents=True)
    (tmp_path / "_private/0-Fleeting-Inbox").mkdir(parents=True)
    (tmp_path / PMOC_PATH).write_text(PMOC_TEXT, encoding="utf-8")
    (tmp_path / AMOC_PATH).write_text(AMOC_TEXT, encoding="utf-8")
    (tmp_path / NOTE_PATH).write_text("# Tornado\n\nread this\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def seeded(store, vault):
    store.insert_blocks([_doc(PMOC, PMOC_PATH), _doc(AMOC, AMOC_PATH), _doc(NOTE, NOTE_PATH)])
    route._registry_cache.clear()
    return store


BLOCKS = {
    "a1": f"Board thought about the currency promise, see [[{AMOC}]] for it.",  # confident
    "b2": "Kids first day at school, cold plunge after, a quiet morning at home.",  # nothing
    "c3": f"Season thinking, close to what I wrote in [[{PMOC}]] on the first.",  # confident
}


@pytest.fixture
def log(seeded, vault, monkeypatch) -> Path:
    """A routed day: three rows, master box unticked."""
    monkeypatch.setattr(route, "_nearest", lambda *a, **k: [])
    blocks = [_block(text, bid) for bid, text in BLOCKS.items()]
    seeded.insert_blocks(blocks)
    route.run_routing(blocks, vault, seeded, None, {})
    return augi_log.log_path(vault, DAY)


def _tick(path: Path, block_id: str, label_prefix: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"<!-- route:{block_id} -->")
    head, body = text[:start], text[start:]
    needle = f"- [ ] {label_prefix}"
    assert needle in body, f"{label_prefix!r} not offered for {block_id}"
    body = body.replace(needle, f"- [x] {label_prefix}", 1)
    path.write_text(head + body, encoding="utf-8")


def _aaa(path: Path, block_id: str, answer: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index(f"<!-- route:{block_id} -->")
    head, body = text[:start], text[start:]
    body = body.replace("aaa:\n", f"aaa: {answer}\n", 1)
    path.write_text(head + body, encoding="utf-8")


def _master(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(routing_janitor.route.MASTER_BOX, "- [x] process this log"))


def _feedback(vault: Path) -> list[dict]:
    p = vault / FEEDBACK_LOG
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def _routes(store, block_id: str) -> list[str]:
    return store.get_routed_container_titles([block_id]).get(block_id, [])


class TestMasterBox:
    def test_nothing_happens_until_the_master_box(self, log, seeded, vault):
        _tick(log, "a1", f"**file under [[{AMOC}]]**")
        assert process_log(log, vault, seeded, {}) == 0
        assert _routes(seeded, "a1") == [] and not _feedback(vault)
        assert "- [x] **file under" in log.read_text()  # his tick is left alone

    def test_master_box_applies_ticks_autos_and_memory(self, log, seeded, vault):
        _tick(log, "a1", f"**file under [[{AMOC}]]**")
        _master(log)
        assert process_log(log, vault, seeded, {}) == 3
        text = log.read_text()
        assert "- ✓ processed" in text and "- [ ] process this log" not in text
        # a1: his tick, accepted
        assert f"- ✓ file under [[{AMOC}]] (you)" in text and _routes(seeded, "a1") == [AMOC]
        # b2: nothing proposed → memory, writes nothing
        assert "- ✓ memory (auto)" in text and _routes(seeded, "b2") == []
        # c3: untouched but confident (he linked it) → auto
        assert f"- ✓ file under [[{PMOC}]] (auto)" in text and _routes(seeded, "c3") == [PMOC]
        rows = text[text.index("<!-- route:") :]
        assert "- [ ] memory" not in rows and "- [ ] hold" not in rows and "aaa:" not in rows
        signals = {r["block_id"]: r["signal"] for r in _feedback(vault)}
        assert signals == {"a1": "accepted", "b2": "memory", "c3": "auto"}
        assert {r["block_id"]: r["by"] for r in _feedback(vault)} == {
            "a1": "you",
            "b2": "auto",
            "c3": "auto",
        }
        a1 = next(r for r in _feedback(vault) if r["block_id"] == "a1")
        assert a1["proposed"]["target"] == AMOC and a1["chosen"]["target"] == AMOC
        assert a1["features"]["folder"] == "_private/0-Fleeting-Inbox"
        assert not route.waiting_logs(seeded)

    def test_idempotent(self, log, seeded, vault):
        _master(log)
        assert process_log(log, vault, seeded, {}) == 3
        assert process_log(log, vault, seeded, {}) == 0
        assert len(_feedback(vault)) == 3

    def test_correction_is_recorded_as_such(self, log, seeded, vault):
        _tick(log, "a1", "memory")
        _master(log)
        process_log(log, vault, seeded, {})
        assert _routes(seeded, "a1") == []
        assert next(r for r in _feedback(vault) if r["block_id"] == "a1")["signal"] == "memory"

    def test_hold_parks_the_row(self, log, seeded, vault):
        _tick(log, "b2", "hold")
        _master(log)
        process_log(log, vault, seeded, {})
        assert "- ✓ hold (you)" in log.read_text()
        rec = routing_janitor._record(seeded, "b2")
        assert rec["status"] == "held"


class TestAaa:
    def test_aaa_overrides_a_tick(self, log, seeded, vault):
        _tick(log, "a1", "memory")
        _aaa(log, "a1", f"extend [[{PMOC}]]")
        _master(log)
        process_log(log, vault, seeded, {})
        text = log.read_text()
        assert f"- ✓ extend [[{PMOC}]] (your aaa:)" in text
        assert f"aaa: extend [[{PMOC}]]" in text  # his words stay
        assert _routes(seeded, "a1") == [PMOC]
        note = (vault / PMOC_PATH).read_text()
        assert "<!-- augi:routed a1 -->" in note
        assert next(r for r in _feedback(vault) if r["block_id"] == "a1")["signal"] == "corrected"

    def test_aaa_bare_words(self, log, seeded, vault):
        _aaa(log, "b2", "just a memory")
        _aaa(log, "c3", f"link to {NOTE}".replace(NOTE, f"[[{NOTE}]]"))
        _master(log)
        process_log(log, vault, seeded, {})
        text = log.read_text()
        assert "- ✓ memory (your aaa:)" in text
        assert f"- ✓ link [[{NOTE}]] (your aaa:)" in text and _routes(seeded, "c3") == [NOTE]


class TestExtend:
    def test_new_day_goes_on_top_of_the_dated_run(self):
        out = insert_extend(PMOC_TEXT, DAY, "a1", "today's block", DAY)
        assert out.index(f"### {DAY}") < out.index("### 2026-09-01") < out.index("### 2026-08-23")
        assert (
            "<!-- augi:routed a1 -->\ntoday's block\n— from [[2026-09-03]]\n<!-- /augi:routed -->"
            in out
        )
        assert out.index("#status/active") < out.index(f"### {DAY}")

    def test_journal_h1_is_the_anchor(self):
        out = insert_extend(AMOC_TEXT, DAY, "a1", "block", DAY)
        assert out.index("# Journal") < out.index(f"### {DAY}") < out.index("### 2026-01-14")
        assert out.index("[[Links]]") < out.index("# Journal")

    def test_same_day_appends_to_that_section(self):
        first = insert_extend(PMOC_TEXT, DAY, "a1", "one", DAY)
        second = insert_extend(first, DAY, "b2", "two", DAY)
        assert second.count(f"### {DAY}") == 1
        assert (
            second.index("routed a1") < second.index("routed b2") < second.index("### 2026-09-01")
        )

    def test_same_day_when_it_is_the_last_section(self):
        text = "# T\n\n### 2026-09-03\nhis words\n"
        out = insert_extend(text, DAY, "a1", "mine", DAY)
        assert out.index("his words") < out.index("routed a1")

    def test_no_journal_no_dates_appends_a_journal(self):
        out = insert_extend("# Tornado\n\nread this\n", DAY, "a1", "block", DAY)
        assert out.endswith(
            f"# Journal\n\n### {DAY}\n\n<!-- augi:routed a1 -->\nblock\n"
            f"— from [[{DAY}]]\n<!-- /augi:routed -->\n\n"
        )

    def test_remove_is_exact_and_drops_an_emptied_heading(self):
        text = insert_extend(PMOC_TEXT, DAY, "a1", "today's block", DAY)
        assert remove_extend(text, "a1") == PMOC_TEXT
        two = insert_extend(text, DAY, "b2", "second", DAY)
        left = remove_extend(two, "a1")
        assert "routed b2" in left and "routed a1" not in left and left.count(f"### {DAY}") == 1

    def test_extend_disabled_falls_back_to_link(self, log, seeded, vault):
        _aaa(log, "a1", f"extend [[{PMOC}]]")
        _master(log)
        process_log(log, vault, seeded, {"routing": {"extend_writes_note": False}})
        assert "(extend disabled)" in log.read_text()
        assert "augi:routed" not in (vault / PMOC_PATH).read_text() and _routes(seeded, "a1") == [
            PMOC
        ]

    def test_aaa_lines_are_stripped_from_the_inserted_text(self, seeded, vault, monkeypatch):
        monkeypatch.setattr(route, "_nearest", lambda *a, **k: [])
        block = _block(f"A thought worth keeping in the season log.\naaa: extend [[{PMOC}]]", "d4")
        seeded.insert_blocks([block])
        route.run_routing([block], vault, seeded, None, {})
        path = augi_log.log_path(vault, DAY)
        _master(path)
        process_log(path, vault, seeded, {})
        note = (vault / PMOC_PATH).read_text()
        assert "A thought worth keeping" in note and "aaa:" not in note


class TestNewNote:
    def test_new_note_is_written_and_linked(self, log, seeded, vault):
        _tick(log, "b2", "new note")
        _master(log)
        process_log(log, vault, seeded, {})
        notes = list((vault / "OpenAugi/Notes").glob("*.md"))
        assert len(notes) == 1 and notes[0].stem.startswith("Kids-first-day-at-school")
        body = notes[0].read_text()
        assert "- [ ] seen" in body and "<!-- augi:routed b2 -->" in body
        assert f"- ✓ new note → [[{notes[0].stem}]] (you)" in log.read_text()


class TestUndo:
    def test_undo_removes_link_and_inserted_text(self, log, seeded, vault):
        _aaa(log, "a1", f"extend [[{PMOC}]]")
        _master(log)
        process_log(log, vault, seeded, {})
        assert "<!-- augi:routed a1 -->" in (vault / PMOC_PATH).read_text()
        text = log.read_text()
        start = text.index("<!-- route:a1 -->")
        text = text[:start] + text[start:].replace("- [ ] undo", "- [x] undo", 1)
        log.write_text(text)
        assert process_log(log, vault, seeded, {}) == 1
        assert _routes(seeded, "a1") == []
        assert (vault / PMOC_PATH).read_text() == PMOC_TEXT
        assert (
            "- ✓ undone" in log.read_text()
            and "- [ ] undo"
            not in log.read_text().split("<!-- route:a1 -->")[1].split("<!-- route:")[0]
        )
        assert [r["signal"] for r in _feedback(vault) if r["block_id"] == "a1"] == [
            "corrected",
            "undo",
        ]
        assert routing_janitor._record(seeded, "a1")["status"] == "undone"

    def test_undo_on_an_unprocessed_log_is_ignored(self, log, seeded, vault):
        assert "- [ ] undo" not in log.read_text()


class TestGuards:
    def test_log_without_rows_is_skipped(self, vault, seeded):
        path = augi_log.log_path(vault, DAY)
        augi_log.ensure_log(path, DAY)
        assert process_log(path, vault, seeded, {}) == 0

    def test_block_gone_is_reported(self, log, seeded, vault):
        seeded.delete_block("a1")
        _master(log)
        process_log(log, vault, seeded, {})
        assert "block gone — nothing applied (auto)" in log.read_text()

    def test_process_changed_targets_augi_logs_only(self, log, seeded, vault):
        _master(log)
        other = vault / "x.md"
        other.write_text("- [x] process this log\n")
        assert routing_janitor.process_changed({str(log), str(other)}, vault, seeded, {}) == 3
