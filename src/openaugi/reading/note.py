"""Vault note → Reader document: the join key, the frontmatter gate, the HTML.

Three things live here, all pure functions over text so they can be tested
without a vault or a network:

1. **The join key.** `note_key()` is sha8 of the vault-relative path. It is
   fabricated into a `url` on push, comes back as `source_url` on every read,
   and is what turns a highlight back into the note that produced it. No
   mapping table: the key IS the URL.
2. **The gate.** `reading_queue: true` in frontmatter, nothing else. The rule
   for *when an agent sets that flag* is prose in the vault's `augi-agent.md`,
   deliberately not code — the whole point of gating on a flag rather than a
   folder is that the rule can be one edited sentence.
3. **The rendering.** Reader takes HTML. A markdown note is not HTML, and
   `[[wikilinks]]` are dead there, so they are rendered as bold text rather
   than as links that go nowhere.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import yaml

# The fabricated URL. Fabricated is legal (Reader only requires *a* url), and
# stable-per-note is what makes a re-push an update instead of a duplicate.
# The scheme is https rather than `augi://` because Reader is stricter about
# what it will accept than about what it will echo back.
NOTE_URL_PREFIX = "https://augi.local/note/"

# Accept the `augi://` spelling on the way back in — it appears in the design
# note and cheaply survives a hand-made `curl` test that used it.
_URL_KEY_RE = re.compile(r"^(?:https://augi\.local/note/|augi://note/)([0-9a-f]{8})$")

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)

TITLE_PREFIX = "Augi — "


@dataclass
class ReadingNote:
    """A vault note viewed as reading material."""

    path: Path
    """Absolute path on disk."""
    rel_path: str
    """Vault-relative POSIX path — the thing the key is derived from."""
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def key(self) -> str:
        return note_key(self.rel_path)

    @property
    def url(self) -> str:
        return NOTE_URL_PREFIX + self.key

    @property
    def flagged(self) -> bool:
        """Whether the note has passed the gate. Strictly `true`, not truthy —
        a stray `reading_queue: maybe` should not ship anything."""
        return self.frontmatter.get("reading_queue") is True

    @property
    def title(self) -> str:
        fm_title = self.frontmatter.get("title")
        if isinstance(fm_title, str) and fm_title.strip():
            return fm_title.strip()
        for line in self.body.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return self.path.stem

    @property
    def description(self) -> str:
        desc = self.frontmatter.get("description")
        return desc.strip() if isinstance(desc, str) else ""

    @property
    def content_hash(self) -> str:
        """Hash of the pushed body, so an unchanged note is not re-pushed."""
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:16]

    def word_count(self) -> int:
        return len(self.body.split())


def note_key(rel_path: str) -> str:
    """sha8 of the vault-relative path."""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:8]


def key_from_url(url: str | None) -> str | None:
    """Recover the note key from a Reader document's `source_url`.

    Returns None for anything we did not author — most documents in the queue
    are real articles and their `source_url` is a real URL.
    """
    if not url:
        return None
    match = _URL_KEY_RE.match(url.strip())
    return match.group(1) if match else None


def parse_note(text: str) -> tuple[dict[str, Any], str]:
    """Split a note into (frontmatter, body). Malformed YAML yields `{}`.

    A note whose frontmatter does not parse is treated as having no flag,
    which fails closed: nothing ships.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        loaded = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text[match.end() :]
    return (loaded if isinstance(loaded, dict) else {}), text[match.end() :]


def load_note(path: Path, vault: Path) -> ReadingNote:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_note(text)
    rel = path.resolve().relative_to(Path(vault).resolve()).as_posix()
    return ReadingNote(path=path, rel_path=rel, frontmatter=frontmatter, body=body)


# ── markdown → HTML ────────────────────────────────────────────────
#
# Deliberately small and dependency-free. Reader renders reading material, not
# a vault: the constructs that matter are headings, paragraphs, lists, quotes,
# code and tables. Anything whose value is its link graph is not reading-shaped
# output and should not have passed the gate.

_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")


def render_inline(text: str) -> str:
    """Inline markdown → HTML, with code spans protected from the rest."""
    spans: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        spans.append(f"<code>{escape(match.group(1))}</code>")
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE_SPAN_RE.sub(_stash, text)
    text = escape(text)
    # Wikilinks are dead in Reader — render the display text in bold.
    text = _WIKILINK_RE.sub(
        lambda m: f"<strong>{(m.group(2) or m.group(1)).strip()}</strong>", text
    )
    text = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def to_html(markdown: str) -> str:
    """Render a markdown body as the HTML Reader will display."""
    out: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None
    quote: list[str] = []
    table: list[str] = []
    in_code = False
    code: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def flush_quote() -> None:
        nonlocal quote
        if quote:
            out.append(f"<blockquote>{render_inline(' '.join(quote))}</blockquote>")
            quote = []

    def flush_table() -> None:
        nonlocal table
        if table:
            out.append(_render_table(table))
            table = []

    def flush_all() -> None:
        flush_paragraph()
        flush_list()
        flush_quote()
        flush_table()

    for raw in markdown.splitlines():
        line = raw.rstrip()

        if line.lstrip().startswith("```"):
            if in_code:
                out.append(f"<pre><code>{escape(chr(10).join(code))}</code></pre>")
                code = []
                in_code = False
            else:
                flush_all()
                in_code = True
            continue
        if in_code:
            code.append(raw)
            continue

        if not line.strip():
            flush_all()
            continue

        if heading := _HEADING_RE.match(line):
            flush_all()
            level = len(heading.group(1))
            out.append(f"<h{level}>{render_inline(heading.group(2))}</h{level}>")
            continue

        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line):
            flush_all()
            out.append("<hr />")
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            flush_list()
            flush_table()
            quote.append(line.lstrip()[1:].strip())
            continue
        flush_quote()

        if line.lstrip().startswith("|"):
            flush_paragraph()
            flush_list()
            table.append(line.strip())
            continue
        flush_table()

        for pattern, tag in ((_ULIST_RE, "ul"), (_OLIST_RE, "ol")):
            if item := pattern.match(line):
                flush_paragraph()
                if list_tag != tag:
                    flush_list()
                    out.append(f"<{tag}>")
                    list_tag = tag
                out.append(f"<li>{render_inline(item.group(1))}</li>")
                break
        else:
            flush_list()
            paragraph.append(line.strip())

    if in_code and code:
        out.append(f"<pre><code>{escape(chr(10).join(code))}</code></pre>")
    flush_all()
    return "\n".join(out)


def _render_table(rows: list[str]) -> str:
    """Pipe table → <table>. The separator row is dropped, the first row is
    the header when a separator followed it."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    has_header = len(cells) > 1 and all(re.fullmatch(r":?-{2,}:?", c) for c in cells[1] if c)
    body_start = 2 if has_header else 0
    parts = ["<table>"]
    if has_header:
        head = "".join(f"<th>{render_inline(c)}</th>" for c in cells[0])
        parts.append(f"<thead><tr>{head}</tr></thead>")
    parts.append("<tbody>")
    for row in cells[body_start:]:
        parts.append("<tr>" + "".join(f"<td>{render_inline(c)}</td>" for c in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)
