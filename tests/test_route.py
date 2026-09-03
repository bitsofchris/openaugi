"""Routing rows — eligibility, registry, proposals, rendering, ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.pipeline import augi_log, route
from openaugi.pipeline.route import (
    MASTER_BOX,
    Proposal,
    Suggestion,
    is_confident,
    is_routing_eligible,
    load_registry,
    propose,
    render_row,
    run_routing,
    waiting_logs,
)

DAILY = "_private/0-Fleeting-Inbox/2026-09-03.md"
PMOC_TITLE = "PMOC - Audacity to take Action - Season 2 - Q2 2026"
AMOC_TITLE = "AMOC - OpenAugi Main"
PMOC_PATH = f"_private/3-MOCs and Projects/{PMOC_TITLE}.md"
AMOC_PATH = f"_private/3-MOCs and Projects/{AMOC_TITLE}.md"
NOTE_TITLE = "Tornado Antidote - Read When Tornado Spins"
NOTE_PATH = f"_private/0-Fleeting-Inbox/{NOTE_TITLE}.md"


def _block(content: str, path: str = DAILY, bid: str = "abc123", **meta) -> Block:
    return Block(
        id=bid,
        kind="data_block",
        content=content,
        title=Path(path).stem,
        block_time="2026-09-03",
        metadata={"source_path": path, "parent_note_title": Path(path).stem, **meta},
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
    (tmp_path / PMOC_PATH).write_text(
        "---\ndescription: The active Q2 season. Route build/use blocks here.\n---\n"
        "#area-journal #status/active #note-type/pmoc\n\n### 2026-09-01\nx\n",
        encoding="utf-8",
    )
    (tmp_path / AMOC_PATH).write_text(
        "---\naugi_id: 1\ndescription: The OpenAugi product. Route product blocks here.\n---\n"
        "#area/openaugi #note-type/amoc\n\n# Journal\n\n### 2026-01-14\ny\n",
        encoding="utf-8",
    )
    # tagged but no description: NOT registered
    (tmp_path / "_private/3-MOCs and Projects/MOC - Unregistered.md").write_text(
        "---\ndescription:\n---\n#note-type/moc\n", encoding="utf-8"
    )
    (tmp_path / NOTE_PATH).write_text("# Tornado\n\nread when spinning\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def seeded(store, vault):
    store.insert_blocks(
        [
            _doc(PMOC_TITLE, PMOC_PATH),
            _doc(AMOC_TITLE, AMOC_PATH),
            _doc("MOC - Unregistered", "_private/3-MOCs and Projects/MOC - Unregistered.md"),
            _doc(NOTE_TITLE, NOTE_PATH),
        ]
    )
    route._registry_cache.clear()
    return store


class TestEligibility:
    def test_human_daily_prose(self):
        assert is_routing_eligible(_block("A working thought about the routing surface today."))

    def test_ai_and_reference_never_route(self):
        text = "A working thought about the routing surface today."
        assert not is_routing_eligible(_block(text, provenance="ai"))
        assert not is_routing_eligible(_block(text, provenance="reference"))

    def test_shares_the_capture_gate(self):
        assert not is_routing_eligible(_block("zzz: do this thing for me please, thoroughly"))
        assert not is_routing_eligible(_block("[[Only A Link]]"))


class TestRegistry:
    def test_tag_and_description_register(self, seeded, vault):
        registry = load_registry(seeded, vault)
        assert set(registry) == {PMOC_TITLE, AMOC_TITLE}
        assert registry[PMOC_TITLE].kind == "pmoc"
        assert registry[AMOC_TITLE].kind == "amoc"
        assert "OpenAugi product" in registry[AMOC_TITLE].description

    def test_cache_refreshes_on_edit(self, seeded, vault):
        load_registry(seeded, vault)
        path = vault / "_private/3-MOCs and Projects/MOC - Unregistered.md"
        path.write_text("---\ndescription: now registered\n---\n#note-type/moc\n")
        import os

        os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
        assert "MOC - Unregistered" in load_registry(seeded, vault)


class TestPropose:
    def _registry(self, seeded, vault):
        return load_registry(seeded, vault)

    def test_aaa_hint_wins(self, seeded, vault):
        block = _block(f"Thinking about the board again.\naaa: extend [[{PMOC_TITLE}]]")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert p.top is not None
        assert (p.top.verb, p.top.target, p.top.source) == ("extend", PMOC_TITLE, "aaa")
        assert p.had_aaa and is_confident(p)

    def test_aaa_bare_title_and_verb_words(self, seeded, vault):
        block = _block(f"A thought that belongs to the product.\naaa: route to {AMOC_TITLE}")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert p.top is not None and (p.top.verb, p.top.target) == ("file under", AMOC_TITLE)

    def test_aaa_memory_is_a_suggestion(self, seeded, vault):
        block = _block("Kids first day at school, cold plunge, coffee on the porch.\naaa: memory")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert p.top is not None and p.top.verb == "memory"

    def test_wikilink_to_container_files_under(self, seeded, vault):
        block = _block(f"Shipping the board today, see [[{AMOC_TITLE}]] for the thread.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert p.top is not None
        assert (p.top.verb, p.top.target, p.top.source) == ("file under", AMOC_TITLE, "link")
        assert is_confident(p)

    def test_wikilink_to_plain_note_links(self, seeded, vault):
        block = _block(f"Spinning a bit today, re-read [[{NOTE_TITLE}]] and calmed down.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert p.top is not None and (p.top.verb, p.top.target) == ("link", NOTE_TITLE)

    def test_nearest_writing_in_a_container_files_under(self, seeded, vault):
        older = _block(
            "the board is the one surface that promises currency",
            path=PMOC_PATH,
            bid="old1",
        )
        older.block_time = "2026-09-01"
        seeded.insert_blocks([older])
        block = _block("The board promise: only one surface has to be current, thinking more.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[(older, 2.5)])
        assert p.top is not None
        assert (p.top.verb, p.top.target, p.top.source) == ("file under", PMOC_TITLE, "nearest")
        assert PMOC_TITLE in p.nearest
        assert is_confident(p)  # a strong lone hit, and file under only writes a DB link
        assert not is_confident(p, margin=3.0)

    def test_nearest_writing_in_a_plain_note_extends(self, seeded, vault):
        older = _block("when the tornado spins, read this", path=NOTE_PATH, bid="old2")
        block = _block("Tornado again this morning, same doubt about the path as before.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[(older, 2.0)])
        assert p.top is not None and (p.top.verb, p.top.target) == ("extend", NOTE_TITLE)
        assert not is_confident(p)  # retrieval alone never writes into his note

    def test_nearest_daily_note_is_never_a_home(self, seeded, vault):
        older = _block("an older daily thought", path="_private/0-Fleeting-Inbox/2026-08-20.md")
        older.id = "old9"
        block = _block("Today's thought, close to one from a daily note last month.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[(older, 2.5)])
        assert p.suggestions == [] and p.nearest == ["2026-08-20"]

    def test_nearest_routed_container_counts(self, seeded, vault):
        older = _block("some earlier thought on the product", bid="old3")
        older.metadata["source_path"] = "_private/0-Fleeting-Inbox/2026-08-20.md"
        seeded.insert_blocks([older])
        seeded.insert_links(
            [Link(from_id="old3", to_id=Block.make_document_id(AMOC_PATH), kind="routed_to")]
        )
        block = _block("Another product thought, close to the one from late August.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[(older, 1.5)])
        assert p.top is not None and (p.top.verb, p.top.target) == ("file under", AMOC_TITLE)

    def test_nothing_found_means_no_suggestions(self, seeded, vault):
        block = _block("A thought with no links, no hints and nothing nearby at all.")
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert p.suggestions == [] and not is_confident(p)

    def test_capped_at_three(self, seeded, vault):
        block = _block(
            f"[[{AMOC_TITLE}]] and [[{PMOC_TITLE}]] and [[{NOTE_TITLE}]] and more.\n"
            f"aaa: extend [[{NOTE_TITLE}]]"
        )
        p = propose(block, seeded, None, {}, self._registry(seeded, vault), nearest=[])
        assert len(p.suggestions) == 3 and p.top is not None and p.top.source == "aaa"


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, system="", temperature=0.1):
        self.prompts.append(prompt)
        return self.reply


class TestJudge:
    def test_judge_reorders_and_explains(self, seeded, vault):
        block = _block(f"[[{AMOC_TITLE}]] thought, and also [[{NOTE_TITLE}]] relates.")
        llm = FakeLLM('{"memory": false, "pick": 2, "why": "it is the antidote thread"}')
        p = propose(block, seeded, None, {}, load_registry(seeded, vault), llm=llm, nearest=[])
        assert p.top is not None and p.top.target == NOTE_TITLE
        assert p.top.why == "it is the antidote thread"
        assert "CANDIDATES" in llm.prompts[0] and "OpenAugi product" in llm.prompts[0]

    def test_judge_can_call_it_a_memory(self, seeded, vault):
        block = _block(f"Kids at the pool, then [[{NOTE_TITLE}]] came up over dinner.")
        llm = FakeLLM('{"memory": true, "pick": 0, "why": "a family evening"}')
        p = propose(block, seeded, None, {}, load_registry(seeded, vault), llm=llm, nearest=[])
        assert p.memory and p.why == "a family evening" and not is_confident(p)

    def test_judge_cannot_override_an_aaa_hint(self, seeded, vault):
        block = _block(f"Kids at the pool.\naaa: extend [[{PMOC_TITLE}]]")
        llm = FakeLLM('{"memory": true, "pick": 0, "why": "family"}')
        p = propose(block, seeded, None, {}, load_registry(seeded, vault), llm=llm, nearest=[])
        assert not p.memory and p.top is not None and p.top.source == "aaa"

    def test_judge_garbage_is_ignored(self, seeded, vault):
        block = _block(f"[[{AMOC_TITLE}]] thought about the product roadmap for the fall.")
        p = propose(
            block, seeded, None, {}, load_registry(seeded, vault), llm=FakeLLM("nope"), nearest=[]
        )
        assert p.top is not None and p.top.target == AMOC_TITLE


class TestRender:
    def test_row_has_marker_suggestions_fallbacks_and_aaa(self):
        block = _block("The board promise, one surface has to be current.")
        p = Proposal(
            suggestions=[
                Suggestion("extend", PMOC_TITLE, "same thread", 0.7, "nearest"),
                Suggestion("file under", AMOC_TITLE, "product", 0.5, "nearest"),
            ]
        )
        out = render_row(block, p)
        assert "<!-- route:abc123 -->" in out
        assert '### route "The board promise' in out
        assert "*[[2026-09-03]] · 2026-09-03*" in out
        assert f"- [ ] **extend [[{PMOC_TITLE}]]** — same thread" in out
        assert f"- [ ] file under [[{AMOC_TITLE}]] — product" in out
        assert "- [ ] new note" in out and "- [ ] memory" in out and "- [ ] hold" in out
        assert out.rstrip().endswith("aaa:")

    def test_memory_verdict_renders_without_a_bold_line(self):
        block = _block("Kids, pool, dinner, bed.")
        out = render_row(block, Proposal(memory=True, why="family evening"))
        assert "*reads as a memory — family evening*" in out and "**" not in out

    def test_fixed_boxes_not_duplicated_when_suggested(self):
        out = render_row(
            _block("x"), Proposal(suggestions=[Suggestion("memory", None, "hint", 1, "aaa")])
        )
        assert out.count("memory") == 1


class TestRunRouting:
    def test_writes_rows_under_routing_with_master_box_and_ledger(
        self, seeded, vault, monkeypatch
    ):
        monkeypatch.setattr(route, "_nearest", lambda *a, **k: [])
        blocks = [
            _block(f"Board thought about the currency promise, see [[{AMOC_TITLE}]].", bid="r1"),
            _block("Kids first day at school, cold plunge after, quiet morning.", bid="r2"),
            _block("zzz: never routed, dispatch owns it and more words here", bid="r3"),
        ]
        stats = run_routing(blocks, vault, seeded, None, {})
        assert stats == {"watched": 2, "proposed": 2}
        text = augi_log.log_path(vault, "2026-09-03").read_text()
        assert text.index(augi_log.ROUTING_HEADING) < text.index(MASTER_BOX)
        assert text.count(MASTER_BOX) == 1
        assert "<!-- route:r1 -->" in text and "<!-- route:r2 -->" in text
        assert "<!-- route:r3 -->" not in text
        rows = seeded.list_records(route.ROUTING_COLLECTION, where={"kind": "row"})
        assert {r["id"] for r in rows} == {"r1", "r2"}
        r1 = next(r for r in rows if r["id"] == "r1")
        assert r1["confident"] and r1["proposed"][0]["target"] == AMOC_TITLE
        assert r1["features"]["folder"] == "_private/0-Fleeting-Inbox"
        logs = waiting_logs(seeded)
        assert len(logs) == 1 and logs[0]["day"] == "2026-09-03"
        assert logs[0]["path"] == "OpenAugi/2026/09/03/Augi Log.md"

    def test_idempotent_across_cycles(self, seeded, vault, monkeypatch):
        monkeypatch.setattr(route, "_nearest", lambda *a, **k: [])
        blocks = [_block("A thought long enough to be routed somewhere useful.", bid="r1")]
        run_routing(blocks, vault, seeded, None, {})
        assert run_routing(blocks, vault, seeded, None, {}) == {"watched": 1, "proposed": 0}
        text = augi_log.log_path(vault, "2026-09-03").read_text()
        assert text.count("<!-- route:r1 -->") == 1 and text.count(MASTER_BOX) == 1

    def test_disabled_by_config(self, seeded, vault):
        blocks = [_block("A thought long enough to be routed somewhere useful.")]
        assert run_routing(blocks, vault, seeded, None, {"routing": {"enabled": False}}) == {
            "watched": 0,
            "proposed": 0,
        }
        assert not augi_log.log_path(vault, "2026-09-03").exists()

    def test_row_and_echo_share_the_log(self, seeded, vault, monkeypatch):
        from openaugi.pipeline import echo

        monkeypatch.setattr(route, "_nearest", lambda *a, **k: [])
        path = augi_log.log_path(vault, "2026-09-03")
        augi_log.ensure_log(path, "2026-09-03")
        echo._write_sections(path, echo_md='\n<!-- echo:e1 -->\n\n### echo on "x…"\n')
        run_routing(
            [_block("A thought long enough to be routed somewhere.", bid="r1")],
            vault,
            seeded,
            None,
            {},
        )
        text = path.read_text()
        assert text.index("<!-- route:r1 -->") < text.index("<!-- echo:e1 -->")
        assert json.loads(json.dumps(augi_log.split(text)[1]))  # both sections parse
