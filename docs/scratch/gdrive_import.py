#!/usr/bin/env python3
"""Google Drive → vault markdown converter (M8 multi-source ingest).

A **converter, not an adapter**: reads rclone-exported Google Drive docs,
stamps honest frontmatter (dates + provenance), normalizes granularity, and
writes vault-ready markdown to a staging dir. It never touches the openaugi
pipeline or the DB — the pipeline only ever sees the resulting vault files
("files are the API"). The pipeline and file contract are described below.

Pipeline (this script does steps 2–4; step 1 is a shell `rclone copy`):
  1. rclone copy REMOTE:<folder> export/ --drive-export-formats md
     (native Google Docs → .md; uploaded .doc/.docx/.odt stay binary)
  2. For each kept doc: pandoc-convert uploads → markdown (natives already md)
  3. Stamp frontmatter: created (→ block_time), gdrive_created/modified/
     path/url provenance, and note-type tags ONLY where folder-mapped
     (reuses existing vault taxonomy — no invented tags; see the rules below)
  4. Granularity: long headerless docs get `qqq` delimiters so they don't
     ingest as one giant block

Nothing here writes to the vault. Landing + ingest are deliberate manual
steps after you review a sample of the output.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────

# Folder → note-type tag. Reuses your EXISTING vault taxonomy only; new
# values are never invented here (source/gdrive comes from your
# [vault.source_rules] config, not this script). Two rule kinds:
#   - PREFIX rules match a folder and everything nested under it.
#   - EXACT rules match one folder precisely (not its subfolders) — used to
#     type the loose top-level docs of a folder as one note-type
#     WITHOUT sweeping in Self/People or Self/Reference & Notes.
NOTE_TYPE_PREFIX_RULES: dict[str, str] = {
    "Self/Journal": "note-type/reflection",
    "Self/Physical": "note-type/reflection",
}
NOTE_TYPE_EXACT_RULES: dict[str, str] = {
    "Self": "note-type/reflection",  # loose top-level Self docs
}

# Extensions we convert. Natives arrive as .md; uploads need pandoc.
MARKDOWN_EXT = {".md"}
PANDOC_EXT = {".docx", ".doc", ".odt"}

# Granularity: docs longer than this with NO headings get qqq delimiters.
GRANULARITY_WORD_THRESHOLD = 1500
QQQ_TARGET_WORDS = 400


# ── Pure helpers (unit-tested) ─────────────────────────────────────


def note_type_for(drive_dir: str) -> str | None:
    """Map a Drive folder path to a note-type tag, or None (untyped).

    Exact rules win over prefix rules; prefix rules match nested subfolders.
    """
    d = drive_dir.strip("/")
    if d in NOTE_TYPE_EXACT_RULES:
        return NOTE_TYPE_EXACT_RULES[d]
    for prefix, tag in NOTE_TYPE_PREFIX_RULES.items():
        if d == prefix or d.startswith(prefix + "/"):
            return tag
    return None


def date_only(iso: str | None) -> str | None:
    """'2024-07-30T00:29:26.235Z' → '2024-07-30'. None-safe."""
    if not iso:
        return None
    return iso[:10] if len(iso) >= 10 else None


def drive_url(file_id: str, is_native_doc: bool) -> str:
    """Canonical link back to the source in Drive."""
    if is_native_doc:
        return f"https://docs.google.com/document/d/{file_id}"
    return f"https://drive.google.com/file/d/{file_id}"


def _yaml_escape(value: str) -> str:
    """Quote a scalar if it could confuse a YAML parser."""
    if value and (value[0] in "!&*[]{}#>|%@`\"'" or ": " in value or value.strip() != value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def build_frontmatter(
    *,
    block_date: str | None,
    gdrive_created: str | None,
    gdrive_modified: str | None,
    gdrive_path: str,
    gdrive_url_: str,
    note_type: str | None,
) -> str:
    """Assemble the YAML frontmatter block per the gdrive-import contract.

    `block_date` fills `created:` — the field the openaugi pipeline reads
    for block_time (below filename dates, above file mtime). This config chose
    modified-time stamping, so callers pass the modified date here; the true
    Drive createdTime/modifiedTime are preserved separately as provenance.
    """
    lines = ["---"]
    if block_date:
        lines.append(f"created: {block_date}")
    if gdrive_created:
        lines.append(f"gdrive_created: {gdrive_created}")
    if gdrive_modified:
        lines.append(f"gdrive_modified: {gdrive_modified}")
    lines.append(f"gdrive_path: {_yaml_escape(gdrive_path)}")
    lines.append(f"gdrive_url: {gdrive_url_}")
    if note_type:
        lines.append("tags:")
        lines.append(f"  - {note_type}")
    lines.append("---")
    return "\n".join(lines)


_ESCAPE_PUNCT = re.compile(r"\\([`*_{}\[\]()#+\-.!<>|~\"'=])")
# Formatting tags textutil/pandoc leave as raw HTML — strip the tag, keep text.
# Whitelisted so we never touch <http…> autolinks or literal <placeholder> text.
_HTML_FORMAT_TAG = re.compile(
    r"</?(?:span|sup|sub|u|b|i|strong|em|div|font|o:p)\b[^>]*>", re.IGNORECASE
)
_DATA_URI_IMG = re.compile(r"!\[[^\]]*\]\(data:[^)]*\)")  # base64 image blobs
_DATA_URI_RAW = re.compile(r"<data:[^>]*>")


def clean_export_artifacts(body: str) -> str:
    """Strip the escape + raw-HTML noise Google Docs / textutil / pandoc add.

    Removes over-escaped punctuation (`\\*`, `\\-`…), trailing-backslash hard
    breaks, standalone `\\` lines, raw formatting tags (`<span>`, `<sup>`…),
    and base64 `data:image` blobs. Preserves prose, autolinks, and any literal
    angle-bracket text the writer intended.
    """
    body = _DATA_URI_IMG.sub("", body)  # drop base64 image blobs (embedding noise)
    body = _DATA_URI_RAW.sub("", body)
    body = _HTML_FORMAT_TAG.sub("", body)  # strip formatting tags, keep inner text
    body = _ESCAPE_PUNCT.sub(r"\1", body)  # unescape over-escaped punctuation
    body = re.sub(r"[ \t]*\\+(\n)", r"\1", body)  # drop hard-break backslashes
    body = re.sub(r"(?m)^\\+[ \t]*$", "", body)  # drop standalone-backslash lines
    body = re.sub(r"[ \t]{3,}", " ", body)  # collapse runs of spaces the spans left
    body = re.sub(r"\n{3,}", "\n\n", body)  # collapse blank lines
    return body


def _has_markdown_heading(body: str) -> bool:
    """True if any line is an ATX heading (`# ` … `###### `)."""
    for ln in body.splitlines():
        s = ln.lstrip()
        hashes = len(s) - len(s.lstrip("#"))
        if 1 <= hashes <= 6 and s[hashes : hashes + 1] == " ":
            return True
    return False


def needs_granularity_split(body: str) -> bool:
    """True if a doc is long AND has no markdown headings to split on."""
    return (not _has_markdown_heading(body)) and len(body.split()) > GRANULARITY_WORD_THRESHOLD


def _paragraph_units(body: str) -> tuple[list[str], str]:
    """Break body into paragraph-like units + the separator to rejoin with.

    Prefer blank-line paragraphs; fall back to single-newline lines; finally
    to whitespace-delimited words for a wall of text with no breaks at all.
    """
    paras = [p for p in body.split("\n\n") if p.strip()]
    if len(paras) > 1:
        return paras, "\n\n"
    lines = [ln for ln in body.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines, "\n"
    return body.split(), " "


def insert_qqq(body: str, target_words: int = QQQ_TARGET_WORDS) -> str:
    """Insert `qqq` delimiter lines between groups of ~target_words.

    We never alter the text itself, only insert standalone `qqq` markers the
    splitter recognizes as block breaks. Grouping keeps paragraph integrity
    where the source has paragraph breaks.
    """
    units, sep = _paragraph_units(body)
    groups: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        current.append(unit)
        count += len(unit.split())
        if count >= target_words:
            groups.append(sep.join(current))
            current = []
            count = 0
    if current:
        groups.append(sep.join(current))
    return "\n\nqqq\n\n".join(g.strip() for g in groups if g.strip())


def stamp_document(
    body: str,
    *,
    block_date: str | None,
    gdrive_created: str | None,
    gdrive_modified: str | None,
    gdrive_path: str,
    gdrive_url_: str,
    note_type: str | None,
) -> str:
    """Full transform: clean export noise + granularity pass + frontmatter."""
    clean = clean_export_artifacts(body).strip()
    if needs_granularity_split(clean):
        clean = insert_qqq(clean)
    fm = build_frontmatter(
        block_date=block_date,
        gdrive_created=gdrive_created,
        gdrive_modified=gdrive_modified,
        gdrive_path=gdrive_path,
        gdrive_url_=gdrive_url_,
        note_type=note_type,
    )
    return f"{fm}\n\n{clean}\n"


# ── IO / orchestration ─────────────────────────────────────────────


@dataclass
class DocMeta:
    path: str  # Drive path as inventoried, e.g. "Self/Journal/Foo.docx"
    file_id: str
    bucket: str  # gdoc | docx | doc | odt
    created: str | None  # ISO btime
    modified: str | None  # ISO mtime


def load_inventory(path: Path) -> dict[tuple[str, str], DocMeta]:
    """Index inventory by (dir, stem) so exported .md files can join back.

    Native docs are inventoried under their export name (Foo.docx) but land
    as Foo.md; uploads land as Foo.md after pandoc. Both share (dir, stem).
    """
    entries = json.loads(path.read_text())
    index: dict[tuple[str, str], DocMeta] = {}
    for e in entries:
        p = Path(e["Path"])
        key = (str(p.parent), p.stem)
        meta = DocMeta(
            path=e["Path"],
            file_id=e["ID"],
            bucket=e["bucket"],
            created=e.get("created"),
            modified=e.get("modified"),
        )
        if key in index:
            print(
                f"  stem collision at {key}: {meta.path} vs {index[key].path}",
                file=sys.stderr,
            )
        index[key] = meta
    return index


def pandoc_to_markdown(src: Path, from_format: str | None = None) -> str:
    """Convert a document to GitHub-flavored markdown via pandoc.

    from_format lets callers pipe an intermediate (e.g. html from textutil).
    """
    cmd = ["pandoc", "-t", "gfm", "--wrap=none"]
    if from_format:
        cmd += ["-f", from_format, str(src)]
    else:
        cmd += [str(src)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def doc_to_markdown(src: Path) -> str:
    """Legacy .doc → markdown. pandoc can't read .doc, so go via macOS
    textutil (.doc → html) then pandoc (html → gfm)."""
    html = subprocess.run(
        ["textutil", "-convert", "html", "-stdout", str(src)],
        capture_output=True,
        check=True,
    ).stdout
    result = subprocess.run(
        ["pandoc", "-f", "html", "-t", "gfm", "--wrap=none"],
        input=html,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def read_body(src: Path) -> str:
    """Get markdown text from a staged file (native .md read; uploads converted)."""
    ext = src.suffix.lower()
    if ext in MARKDOWN_EXT:
        return src.read_text(encoding="utf-8")
    if ext == ".doc":
        return doc_to_markdown(src)
    if ext in PANDOC_EXT:
        return pandoc_to_markdown(src)
    raise ValueError(f"unsupported extension: {src.suffix}")


def convert(export_dir: Path, out_dir: Path, inventory: Path) -> dict[str, int]:
    """Walk the export dir, stamp each doc, write to out_dir. Returns stats."""
    index = load_inventory(inventory)
    stats = {"written": 0, "skipped_junk": 0, "unmatched": 0, "typed": 0}

    for src in sorted(export_dir.rglob("*")):
        if src.is_dir():
            continue
        ext = src.suffix.lower()
        if ext not in MARKDOWN_EXT and ext not in PANDOC_EXT:
            stats["skipped_junk"] += 1
            continue

        rel = src.relative_to(export_dir)
        key = (str(rel.parent), rel.stem)
        meta = index.get(key)
        if meta is None:
            print(f"  ⚠️  no metadata for {rel} — skipping", file=sys.stderr)
            stats["unmatched"] += 1
            continue

        gdrive_dir = str(Path(meta.path).parent)
        note_type = note_type_for(gdrive_dir)
        try:
            body = read_body(src)
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  pandoc failed on {rel}: {e.stderr}", file=sys.stderr)
            stats["unmatched"] += 1
            continue

        final = stamp_document(
            body,
            block_date=date_only(meta.modified),  # modified-time stamping
            gdrive_created=date_only(meta.created),
            gdrive_modified=date_only(meta.modified),
            gdrive_path=gdrive_dir,
            gdrive_url_=drive_url(meta.file_id, meta.bucket == "gdoc"),
            note_type=note_type,
        )

        out_path = out_dir / rel.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final, encoding="utf-8")
        stats["written"] += 1
        if note_type:
            stats["typed"] += 1

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stamp rclone-exported gdrive docs into vault markdown."
    )
    ap.add_argument("--export-dir", required=True, type=Path, help="rclone export root")
    ap.add_argument("--out-dir", required=True, type=Path, help="staging output for stamped .md")
    ap.add_argument("--inventory", required=True, type=Path, help="inventory-kept.json")
    args = ap.parse_args()

    stats = convert(args.export_dir, args.out_dir, args.inventory)
    print(
        f"✓ wrote {stats['written']} docs "
        f"({stats['typed']} typed note-type/reflection), "
        f"skipped {stats['skipped_junk']} non-doc files, "
        f"{stats['unmatched']} unmatched/failed"
    )


if __name__ == "__main__":
    main()
