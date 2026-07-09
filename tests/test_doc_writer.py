"""VaultWriter tests — frontmatter emission, subfolder scoping, overwrite guard.

The extra_frontmatter path backs the lens-output provenance stamp
(`lens: <name>`), which is what makes View - Lenses.md reconstructible
from disk. If this drifts, the index silently loses its self-heal.
"""

from __future__ import annotations

from pathlib import Path

from openaugi.mcp.doc_writer import VaultWriter


def _read(vault: Path, rel: str) -> str:
    return (vault / rel).read_text(encoding="utf-8")


def test_standard_frontmatter(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    res = w.write_document("Note One", "a summary", "body text", subfolder="Notes")
    assert res["status"] == "created"
    text = _read(tmp_path, "OpenAugi/Notes/Note One.md")
    assert "type: document" in text
    assert "description: a summary" in text
    assert "created:" in text
    assert text.rstrip().endswith("body text")


def test_extra_frontmatter_emitted_after_standard_keys(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    w.write_document(
        "Echo Run",
        "echoes output",
        "the echo",
        subfolder="Notes",
        extra_frontmatter={"lens": "echoes"},
    )
    text = _read(tmp_path, "OpenAugi/Notes/Echo Run.md")
    lines = text.splitlines()
    assert lines[0] == "---"
    # order: type, description, created, then extras
    assert lines[1] == "type: document"
    assert lines[3].startswith("created:")
    assert "lens: echoes" in lines
    # extras land inside the frontmatter block, before the closing fence
    assert lines.index("lens: echoes") < lines.index("---", 1)


def test_extra_frontmatter_absent_by_default(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    w.write_document("Plain", "d", "b")
    text = _read(tmp_path, "OpenAugi/Notes/Plain.md")
    assert "lens:" not in text


def test_extra_frontmatter_values_flattened_to_one_line(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    w.write_document(
        "Multi",
        "d",
        "b",
        extra_frontmatter={"lens": "echoes", "scope": "line one\nline two"},
    )
    text = _read(tmp_path, "OpenAugi/Notes/Multi.md")
    fm = text.split("---")[1]
    assert "scope: line one line two" in fm  # newline collapsed, stays a scalar
    # every frontmatter line is a single key: value pair
    for line in fm.strip().splitlines():
        assert ": " in line


def test_reserved_keys_cannot_be_overridden(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    w.write_document(
        "Guarded",
        "real desc",
        "b",
        extra_frontmatter={"description": "HIJACKED", "created": "1999", "lens": "nuggets"},
    )
    fm = _read(tmp_path, "OpenAugi/Notes/Guarded.md").split("---")[1]
    assert "description: real desc" in fm
    assert "HIJACKED" not in fm
    assert fm.count("created:") == 1
    assert "1999" not in fm
    assert "lens: nuggets" in fm  # non-reserved extras still land


def test_empty_extra_frontmatter_key_skipped(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    w.write_document("Blank", "d", "b", extra_frontmatter={"": "x", "lens": "distill"})
    fm = _read(tmp_path, "OpenAugi/Notes/Blank.md").split("---")[1]
    assert "lens: distill" in fm
    assert ": x" not in fm


def test_overwrite_guard_protects_notes(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    w.write_document("Dup", "d", "first")
    res = w.write_document("Dup", "d", "second")
    assert res["status"] == "error"
    assert "already exists" in res["reason"]
    assert "first" in _read(tmp_path, "OpenAugi/Notes/Dup.md")

    ok = w.write_document("Dup", "d", "second", overwrite=True)
    assert ok["status"] == "updated"
    assert "second" in _read(tmp_path, "OpenAugi/Notes/Dup.md")


def test_subfolder_cannot_escape_root(tmp_path: Path):
    w = VaultWriter(str(tmp_path))
    res = w.write_document("Escape", "d", "b", subfolder="../../etc")
    assert res["status"] == "error"
    assert "Invalid subfolder" in res["reason"]
