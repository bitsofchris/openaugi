"""Reading queue — gate, push, and the highlight return leg.

No network anywhere: a FakeReader stands in for the Reader API, which is what
the ReaderAPI protocol exists for.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from openaugi.reading.harvest import STATE_COLLECTION, harvest, last_run, render_section
from openaugi.reading.note import (
    NOTE_URL_PREFIX,
    key_from_url,
    load_note,
    note_key,
    parse_note,
    to_html,
)
from openaugi.reading.push import COLLECTION, build_payload, find_flagged_notes, push_notes

NOW = datetime(2026, 9, 9, 8, 0, 0)


class FakeReader:
    """Records what was saved; replays canned documents on list."""

    def __init__(self, documents: list[dict[str, Any]] | None = None):
        self.saved: list[dict[str, Any]] = []
        self.documents = documents or []
        self.fail_on: set[str] = set()

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload["title"] in self.fail_on:
            raise RuntimeError("reader is down")
        self.saved.append(payload)
        return {"id": f"doc{len(self.saved)}", "url": "https://read.readwise.io/read/x"}

    def list_documents(
        self,
        *,
        category: str | None = None,
        updated_after: str | None = None,
        document_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        for doc in self.documents:
            if document_id is not None:
                if str(doc.get("id")) == document_id:
                    yield doc
                continue
            if category and doc.get("category") != category:
                continue
            if updated_after and str(doc.get("updated_at", "")) <= updated_after:
                continue
            yield doc


def write_note(vault: Path, rel: str, body: str, **frontmatter: Any) -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = "\n".join(
        f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in frontmatter.items()
    )
    path.write_text(f"---\n{fm}\n---\n{body}", encoding="utf-8")
    return path


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "OpenAugi" / "Research").mkdir(parents=True)
    return tmp_path


# ── the join key ───────────────────────────────────────────────────


def test_note_key_is_stable_and_path_derived():
    assert note_key("OpenAugi/Research/a.md") == note_key("OpenAugi/Research/a.md")
    assert note_key("OpenAugi/Research/a.md") != note_key("OpenAugi/Research/b.md")
    assert len(note_key("OpenAugi/Research/a.md")) == 8


def test_key_round_trips_through_the_fabricated_url():
    key = note_key("OpenAugi/Research/a.md")
    assert key_from_url(NOTE_URL_PREFIX + key) == key
    # The `augi://` spelling from the design note also parses.
    assert key_from_url(f"augi://note/{key}") == key


def test_key_from_url_ignores_real_articles():
    assert key_from_url("https://nabeelqu.substack.com/p/understanding") is None
    assert key_from_url(None) is None


# ── the gate ───────────────────────────────────────────────────────


def test_only_flagged_notes_are_candidates(vault: Path):
    write_note(vault, "OpenAugi/Research/yes.md", "# Yes\n\nprose", reading_queue=True)
    write_note(vault, "OpenAugi/Research/no.md", "# No\n\nprose")
    write_note(vault, "OpenAugi/Research/maybe.md", "# Maybe", reading_queue="maybe")

    assert [n.path.name for n in find_flagged_notes(vault)] == ["yes.md"]


def test_machinery_folders_never_ship(vault: Path):
    write_note(vault, "OpenAugi/Tasks/t.md", "# Task", reading_queue=True)
    write_note(vault, "OpenAugi/Archive/old.md", "# Old", reading_queue=True)
    assert find_flagged_notes(vault) == []


def test_broken_frontmatter_fails_closed(vault: Path):
    (vault / "OpenAugi/Research/bad.md").write_text(
        '---\nreading_queue: true\ndescription: "unclosed\n---\n# Bad\n', encoding="utf-8"
    )
    assert find_flagged_notes(vault) == []


def test_parse_note_splits_frontmatter_from_body():
    fm, body = parse_note("---\ntitle: T\n---\n# Heading\n\ntext\n")
    assert fm == {"title": "T"}
    assert body.startswith("# Heading")


# ── the payload ────────────────────────────────────────────────────


def test_payload_shape(vault: Path):
    path = write_note(
        vault,
        "OpenAugi/Research/small-models.md",
        "# How small can a parsing model be\n\nprose here",
        reading_queue=True,
        description="A study.",
    )
    payload = build_payload(load_note(path, vault))

    assert payload["url"] == NOTE_URL_PREFIX + note_key("OpenAugi/Research/small-models.md")
    assert payload["title"] == "Augi — How small can a parsing model be"
    assert payload["author"] == "augi"
    assert payload["location"] == "later"  # never jumps your own saves
    assert payload["tags"] == ["augi"]
    assert payload["summary"] == "A study."
    assert "<p>prose here</p>" in payload["html"]


def test_title_falls_back_to_heading_then_filename(vault: Path):
    heading = write_note(vault, "OpenAugi/Research/f.md", "# From heading\n", reading_queue=True)
    assert load_note(heading, vault).title == "From heading"
    bare = write_note(vault, "OpenAugi/Research/bare-name.md", "no heading\n", reading_queue=True)
    assert load_note(bare, vault).title == "bare-name"


# ── rendering ──────────────────────────────────────────────────────


def test_wikilinks_render_as_bold_not_dead_links():
    assert to_html("See [[Some Note]].") == "<p>See <strong>Some Note</strong>.</p>"
    assert "<strong>alias</strong>" in to_html("See [[Some Note|alias]].")


def test_block_constructs():
    html = to_html("## Head\n\n- one\n- two\n\n> quoted\n\n```\ncode\n```\n")
    assert "<h2>Head</h2>" in html
    assert "<ul>\n<li>one</li>\n<li>two</li>\n</ul>" in html
    assert "<blockquote>quoted</blockquote>" in html
    assert "<pre><code>code</code></pre>" in html


def test_inline_and_escaping():
    html = to_html("**bold** and *italic* and `a < b` and <script>x</script>")
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>a &lt; b</code>" in html
    assert "<script>" not in html


def test_markdown_links_survive():
    assert '<a href="https://x.com/a">text</a>' in to_html("[text](https://x.com/a)")


def test_table_renders_with_header():
    html = to_html("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<th>a</th>" in html
    assert "<td>1</td>" in html
    assert "---" not in html


# ── pushing ────────────────────────────────────────────────────────


def test_push_sends_flagged_notes_and_records_them(vault: Path, store):
    write_note(vault, "OpenAugi/Research/a.md", "# A\n\nprose", reading_queue=True)
    client = FakeReader()

    result = push_notes(vault, store, client, now=NOW)

    assert result.pushed == ["OpenAugi/Research/a.md"]
    assert len(client.saved) == 1
    record = store.list_records(COLLECTION)[0]
    assert record["id"] == note_key("OpenAugi/Research/a.md")
    assert record["reader_id"] == "doc1"
    assert record["pushed_at"].startswith("2026-09-09")


def test_cap_defers_the_rest_to_tomorrow(vault: Path, store):
    for name in ("a", "b", "c"):
        write_note(vault, f"OpenAugi/Research/{name}.md", f"# {name}\n\nprose", reading_queue=True)
    client = FakeReader()

    result = push_notes(vault, store, client, cap=2, now=NOW)

    assert len(result.pushed) == 2
    assert len(result.deferred_over_cap) == 1
    assert len(client.saved) == 2


def test_cap_counts_what_already_went_out_today(vault: Path, store):
    write_note(vault, "OpenAugi/Research/a.md", "# A\n\nprose", reading_queue=True)
    write_note(vault, "OpenAugi/Research/b.md", "# B\n\nprose", reading_queue=True)
    client = FakeReader()

    push_notes(vault, store, client, cap=1, now=NOW)
    second = push_notes(vault, store, client, cap=1, now=NOW.replace(hour=20))

    assert second.pushed == []
    assert second.deferred_over_cap == ["OpenAugi/Research/b.md"]
    # Tomorrow the cap resets.
    third = push_notes(vault, store, client, cap=1, now=NOW.replace(day=10))
    assert third.pushed == ["OpenAugi/Research/b.md"]


def test_unchanged_notes_are_not_repushed(vault: Path, store):
    write_note(vault, "OpenAugi/Research/a.md", "# A\n\nprose", reading_queue=True)
    client = FakeReader()

    push_notes(vault, store, client, now=NOW)
    again = push_notes(vault, store, client, now=NOW.replace(day=10))

    assert again.pushed == []
    assert again.skipped_unchanged == ["OpenAugi/Research/a.md"]
    assert len(client.saved) == 1


def test_edited_notes_are_repushed_to_the_same_url(vault: Path, store):
    path = write_note(vault, "OpenAugi/Research/a.md", "# A\n\nprose", reading_queue=True)
    client = FakeReader()
    push_notes(vault, store, client, now=NOW)

    path.write_text("---\nreading_queue: true\n---\n# A\n\nrevised prose", encoding="utf-8")
    again = push_notes(vault, store, client, now=NOW.replace(day=10))

    assert again.pushed == ["OpenAugi/Research/a.md"]
    assert client.saved[0]["url"] == client.saved[1]["url"]  # idempotent, updates in place


def test_dry_run_sends_and_records_nothing(vault: Path, store):
    write_note(vault, "OpenAugi/Research/a.md", "# A\n\nprose", reading_queue=True)
    client = FakeReader()

    result = push_notes(vault, store, client, dry_run=True, now=NOW)

    assert result.pushed == ["OpenAugi/Research/a.md"]
    assert client.saved == []
    assert store.list_records(COLLECTION) == []


def test_one_failure_does_not_stop_the_run(vault: Path, store):
    write_note(vault, "OpenAugi/Research/a.md", "# A\n\nprose", reading_queue=True)
    write_note(vault, "OpenAugi/Research/b.md", "# B\n\nprose", reading_queue=True)
    client = FakeReader()
    client.fail_on = {"Augi — A"}

    result = push_notes(vault, store, client, cap=5, now=NOW)

    assert result.pushed == ["OpenAugi/Research/b.md"]
    assert result.failed[0][0] == "OpenAugi/Research/a.md"
    assert len(store.list_records(COLLECTION)) == 1


# ── harvesting ─────────────────────────────────────────────────────


def _pushed(vault: Path, store, rel: str = "OpenAugi/Research/a.md") -> tuple[Path, FakeReader]:
    path = write_note(vault, rel, "# A\n\nprose", reading_queue=True)
    client = FakeReader()
    push_notes(vault, store, client, now=NOW)
    return path, client


def _highlight(hid: str, content: str, parent: str = "p1", **extra: Any) -> dict[str, Any]:
    return {
        "id": hid,
        "category": "highlight",
        "parent_id": parent,
        "content": content,
        "updated_at": "2026-09-10T12:00:00Z",
        "created_at": f"2026-09-10T12:0{hid[-1]}:00Z",
        **extra,
    }


def test_highlights_land_on_the_source_note(vault: Path, store):
    path, _ = _pushed(vault, store)
    key = note_key("OpenAugi/Research/a.md")
    reader = FakeReader(
        [
            _highlight("h1", "the marked span", notes="my inline note"),
            _highlight("h2", "a second mark"),
            {"id": "p1", "category": "article", "source_url": NOTE_URL_PREFIX + key},
        ]
    )

    result = harvest(vault, store, reader, now=datetime(2026, 9, 11, 9, 0))

    text = path.read_text()
    assert "## Read in Reader — 2026-09-11" in text
    assert "> the marked span" in text
    assert "> — note: my inline note" in text
    assert "> a second mark" in text
    assert result.highlights_appended == 2
    assert result.notes_updated == ["OpenAugi/Research/a.md"]


def test_harvest_is_idempotent(vault: Path, store):
    path, _ = _pushed(vault, store)
    key = note_key("OpenAugi/Research/a.md")
    docs = [
        _highlight("h1", "the marked span"),
        {"id": "p1", "category": "article", "source_url": NOTE_URL_PREFIX + key},
    ]

    harvest(vault, store, FakeReader(docs), since="", now=datetime(2026, 9, 11, 9, 0))
    first = path.read_text()
    second_run = harvest(vault, store, FakeReader(docs), since="", now=datetime(2026, 9, 12, 9, 0))

    assert second_run.highlights_appended == 0
    assert path.read_text() == first


def test_your_own_reading_is_left_alone(vault: Path, store):
    _pushed(vault, store)
    reader = FakeReader(
        [
            _highlight("h1", "from a real article", parent="p9"),
            {
                "id": "p9",
                "category": "article",
                "source_url": "https://nabeelqu.substack.com/p/understanding",
            },
        ]
    )

    result = harvest(vault, store, reader, now=datetime(2026, 9, 11, 9, 0))

    assert result.highlights_appended == 0
    assert result.unmatched_parents == ["p9"]


def test_harvest_window_advances(vault: Path, store):
    _pushed(vault, store)
    assert last_run(store) is None

    harvest(vault, store, FakeReader([]), now=datetime(2026, 9, 11, 9, 0))

    assert last_run(store) == "2026-09-11T09:00:00"
    assert store.list_records(STATE_COLLECTION)[0]["last_run"] == "2026-09-11T09:00:00"


def test_harvest_dry_run_writes_nothing(vault: Path, store):
    path, _ = _pushed(vault, store)
    key = note_key("OpenAugi/Research/a.md")
    before = path.read_text()
    reader = FakeReader(
        [
            _highlight("h1", "the marked span"),
            {"id": "p1", "category": "article", "source_url": NOTE_URL_PREFIX + key},
        ]
    )

    result = harvest(vault, store, reader, dry_run=True, now=datetime(2026, 9, 11, 9, 0))

    assert result.highlights_appended == 1
    assert path.read_text() == before
    assert last_run(store) is None


def test_render_section_shape():
    section = render_section(
        [{"id": "h1", "content": "one\ntwo", "notes": "why"}], datetime(2026, 9, 11)
    )
    assert section.splitlines()[:5] == [
        "## Read in Reader — 2026-09-11",
        "",
        "> one",
        "> two",
        "> — note: why",
    ]
