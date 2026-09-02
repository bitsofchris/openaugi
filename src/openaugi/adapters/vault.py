"""Obsidian vault adapter — reads .md files into blocks + links.

Splitting is delegated to [splitter.py](splitter.py) — the shared deterministic
primitive. This module owns file walking, wikilink resolution, and wrapping
segments into `Block` + `Link` for the store.

Block ID = hash(source_path + content_hash) — stable across reordering.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openaugi.adapters import splitter as _splitter

# Re-export the shared splitter helpers so existing importers (including tests)
# keep working. All splitting logic lives in splitter.py — see its module
# docstring and [docs/reference/splitter.md](../../../docs/reference/splitter.md).
from openaugi.adapters.splitter import (
    _code_fence_ranges,  # noqa: F401
    _extract_augi_id,
    _extract_filename_date,
    _extract_frontmatter_created,
    _extract_links,  # noqa: F401
    _extract_tags,  # noqa: F401
    _extract_wk_date,
    _extract_zzz_instructions,  # noqa: F401
    _has_meaningful_content,  # noqa: F401
    _parse_date,
    _split_by_headings,  # noqa: F401
    _split_by_qqq,  # noqa: F401
    _strip_frontmatter,
)
from openaugi.model.block import Block
from openaugi.model.link import Link

logger = logging.getLogger(__name__)

# Re-export splitter regexes so existing importers keep working.
ANY_HEADING_PATTERN = _splitter.ANY_HEADING_PATTERN
TAG_PATTERN = _splitter.TAG_PATTERN
LINK_PATTERN = _splitter.LINK_PATTERN
FILENAME_DATE_PATTERN = _splitter.FILENAME_DATE_PATTERN
WK_DATE_PATTERN = _splitter.WK_DATE_PATTERN
FRONTMATTER_PATTERN = _splitter.FRONTMATTER_PATTERN
QQQ_PATTERN = _splitter.QQQ_PATTERN
DATAVIEW_BLOCK_PATTERN = _splitter.DATAVIEW_BLOCK_PATTERN
ZZZ_PATTERN = _splitter.ZZZ_PATTERN

DEFAULT_EXCLUDE_PATTERNS = [
    ".obsidian/**",
    ".git/**",
    ".smart-env/**",
    ".trash/**",
    "templates/**",
    "OpenAugi/Compiled/**",
    "*.excalidraw.md",
    "**/4-Tech Notes/**",
    # High-volume reference import folders — clippings, not original thought
    "**/Instapaper/**",
    "**/Readwise/**",
    "**/Snipd/**",
]


# ── Public API ─────────────────────────────────────────────────────


def parse_vault(
    vault_path: str | Path,
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
    source_rules: dict[str, str] | None = None,
    provenance_rules: dict[str, str] | None = None,
) -> tuple[list[Block], list[Link]]:
    """Parse an Obsidian vault into blocks and links.

    source_rules: {path glob → source/* tag} from [vault.source_rules] config;
    stamps ingest-origin attribution on blocks in matching folders (explicit
    source/* tags in the note text always win — text is truth).
    provenance_rules: {path glob → human|ai|reference} from
    [vault.provenance_rules]; see `resolve_provenance`.

    Returns (blocks, links) ready to insert into the store.
    """
    vault = Path(vault_path).expanduser()
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault}")
    excludes = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
    rules = _normalize_source_rules(source_rules)
    prov_rules = _normalize_provenance_rules(provenance_rules)

    all_files = list(vault.rglob("*.md"))
    if not all_files:
        _check_readable(vault)  # raises PermissionError with guidance if sandbox
    included = [f for f in all_files if _should_include(f, vault, excludes)]
    logger.info(f"Found {len(included)} files (excluded {len(all_files) - len(included)})")

    file_index = _build_file_index(included, vault)

    all_blocks: list[Block] = []
    all_links: list[Link] = []
    tag_blocks: dict[str, Block] = {}  # dedupe tag blocks globally

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_parse_file, f, vault, file_index, rules, prov_rules): f
            for f in included
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                blocks, links, file_tags = future.result()
                all_blocks.extend(blocks)
                all_links.extend(links)
                # Merge tag blocks (dedupe by tag name)
                for tag_name, tag_block in file_tags.items():
                    if tag_name not in tag_blocks:
                        tag_blocks[tag_name] = tag_block
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")

    all_blocks.extend(tag_blocks.values())
    logger.info(
        f"Parsed {len(all_blocks)} blocks and {len(all_links)} links "
        f"({len(tag_blocks)} unique tags)"
    )
    return all_blocks, all_links


def parse_vault_incremental(
    vault_path: str | Path,
    known_doc_hashes: dict[str, str],
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
    source_rules: dict[str, str] | None = None,
    provenance_rules: dict[str, str] | None = None,
) -> tuple[list[Block], list[Link], dict[str, str], list[str]]:
    """Parse vault with incremental change detection.

    Args:
        vault_path: Path to Obsidian vault.
        known_doc_hashes: {relative_path: content_hash} from previous run.
        exclude_patterns: Glob patterns to skip.
        max_workers: Thread pool size.
        source_rules: {path glob → source/* tag}; see parse_vault.
        provenance_rules: {path glob → human|ai|reference}; see parse_vault.

    Returns:
        (new_blocks, new_links, current_hashes, deleted_paths)
        - new_blocks/links: only from changed/new files
        - current_hashes: {relative_path: hash} for all current files
        - deleted_paths: relative paths no longer on disk
    """
    vault = Path(vault_path).expanduser()
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault}")
    excludes = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
    rules = _normalize_source_rules(source_rules)
    prov_rules = _normalize_provenance_rules(provenance_rules)

    all_files = list(vault.rglob("*.md"))
    if not all_files:
        _check_readable(vault)  # raises PermissionError with guidance if sandbox
    included = [f for f in all_files if _should_include(f, vault, excludes)]

    # Hash all files, determine which changed
    current_hashes: dict[str, str] = {}
    files_to_parse: list[Path] = []

    for file_path in included:
        rel_path = str(file_path.relative_to(vault))
        try:
            content_hash = _hash_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            continue
        current_hashes[rel_path] = content_hash
        if known_doc_hashes.get(rel_path) != content_hash:
            files_to_parse.append(file_path)

    deleted_paths = [p for p in known_doc_hashes if p not in current_hashes]

    logger.info(
        f"Change detection: {len(files_to_parse)} changed/new, "
        f"{len(included) - len(files_to_parse)} unchanged, "
        f"{len(deleted_paths)} deleted"
    )

    if not files_to_parse:
        return [], [], current_hashes, deleted_paths

    # Build index from ALL included files (not just changed) for wikilink resolution
    file_index = _build_file_index(included, vault)

    # Parse only changed files
    all_blocks: list[Block] = []
    all_links: list[Link] = []
    tag_blocks: dict[str, Block] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_parse_file, f, vault, file_index, rules, prov_rules): f
            for f in files_to_parse
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                blocks, links, file_tags = future.result()
                all_blocks.extend(blocks)
                all_links.extend(links)
                for tag_name, tag_block in file_tags.items():
                    if tag_name not in tag_blocks:
                        tag_blocks[tag_name] = tag_block
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")

    all_blocks.extend(tag_blocks.values())
    return all_blocks, all_links, current_hashes, deleted_paths


# ── Internal helpers ───────────────────────────────────────────────


def _check_readable(vault: Path) -> None:
    """Diagnose why rglob returned no files.

    Called only when rglob("*.md") returns empty. On macOS, rglob silently
    returns [] when sandbox permissions block access. os.listdir raises
    PermissionError, giving us a clear error to surface.
    """
    try:
        os.listdir(vault)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot read vault directory: {vault}\n"
            "On macOS, grant Full Disk Access to your terminal app in "
            "System Settings > Privacy & Security > Full Disk Access."
        ) from e


def _parse_file(
    file_path: Path,
    vault_root: Path,
    file_index: dict[str, str],
    source_rules: list[tuple[str, str]] | None = None,
    provenance_rules: list[tuple[str, str]] | None = None,
) -> tuple[list[Block], list[Link], dict[str, Block]]:
    """Parse a single .md file into blocks + links.

    Returns (blocks, links, tag_blocks_dict).
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return [], [], {}

    rel_path = str(file_path.relative_to(vault_root))
    parent_title = file_path.stem

    blocks: list[Block] = []
    links: list[Link] = []
    tag_blocks: dict[str, Block] = {}

    # Document block.
    #
    # Identity comes from the note when the note carries one. `augi_id` in the
    # frontmatter is the document's name; the path is only a fallback for
    # notes that have not been given one. This matters because every edge into
    # a container — `contains` and `routed_to` alike — is keyed on this id, so
    # deriving it from the path meant that renaming or moving a note in the
    # editor orphaned all of them, with no error and no repair: the container
    # simply looked emptier than it should.
    file_hash = _hash_content(content)
    augi_id = _extract_augi_id(content)
    doc_id = Block.make_document_id(f"augi:{augi_id}" if augi_id else rel_path)
    doc_block = Block(
        id=doc_id,
        kind="context_block:document",
        title=parent_title,
        source="vault",
        content_hash=file_hash,
        metadata={"source_path": rel_path, **({"augi_id": augi_id} if augi_id else {})},
    )
    blocks.append(doc_block)

    # Delegate splitting to the shared splitter — same rules everywhere.
    # We strip frontmatter here ourselves only to capture fm_tags; splitter
    # would drop them otherwise.
    body, fm_tags = _strip_frontmatter(content)

    title_date = _extract_filename_date(file_path)
    wk_date = _extract_wk_date(file_path)
    effective_title_date = title_date or wk_date
    fm_created = _extract_frontmatter_created(content)
    file_created = _get_file_created_time(file_path)

    segments = (
        _splitter._segments_from_single_section(body, wk_date, None)
        if wk_date
        else _splitter._segments_from_body(body)
    )

    for seg in segments:
        entry_hash = seg.raw_hash  # sha256(raw_content)[:16]
        entry_id = Block.make_id(rel_path, entry_hash)

        section_date = _parse_date(seg.section_date) if seg.section_date else None
        resolved_ts = _resolve_timestamp(
            section_date, effective_title_date, fm_created, file_created
        )

        all_tags = _apply_source_rules(rel_path, _unique_ordered(fm_tags + seg.tags), source_rules)

        # An anchored capture entry with a parsed `HH:MM —` lead gets a
        # time-of-day timestamp instead of inheriting the bare file date.
        if seg.entry_time and _parse_date(resolved_ts):
            resolved_ts = f"{resolved_ts}T{seg.entry_time}:00"

        entry_metadata: dict = {
            "source_path": rel_path,
            "section_date": seg.section_date,
            "section_heading": seg.section_heading,
            "parent_note_title": parent_title,
            "file_created_at": file_created,
            "granularity": seg.granularity,
            "provenance": resolve_provenance(rel_path, all_tags, provenance_rules),
        }
        if seg.anchor_id:
            entry_metadata["anchor_id"] = seg.anchor_id
        if seg.has_open_task:
            entry_metadata["has_open_task"] = True
        if fm_created:
            entry_metadata["frontmatter_created"] = fm_created
        if seg.zzz_instructions:
            entry_metadata["zzz_instructions"] = seg.zzz_instructions

        entry_block = Block(
            id=entry_id,
            kind="data_block",
            content=seg.clean_content,
            source="vault",
            title=parent_title,
            tags=all_tags,
            block_time=resolved_ts,
            content_hash=entry_hash,
            metadata=entry_metadata,
        )
        blocks.append(entry_block)

        links.append(Link(from_id=entry_id, to_id=doc_id, kind="contains"))

        for tag_name in all_tags:
            tag_id = Block.make_tag_id(tag_name)
            if tag_name not in tag_blocks:
                tag_blocks[tag_name] = Block(
                    id=tag_id, kind="context_block:tag", title=tag_name, source="vault"
                )
            links.append(Link(from_id=entry_id, to_id=tag_id, kind="groups"))

        for link_target in seg.links:
            target_doc_id = Block.make_document_id(_resolve_wikilink(link_target, file_index))
            links.append(
                Link(
                    from_id=entry_id,
                    to_id=target_doc_id,
                    kind="links_to",
                    metadata={"target_title": link_target},
                )
            )

    # Granularity is set per-segment by the splitter, but the splitter sees
    # each WK section independently. When the whole file yields exactly one
    # data_block (no heading or qqq split), upgrade it to "document".
    entry_blocks = [b for b in blocks if b.kind == "data_block"]
    if len(entry_blocks) == 1:
        entry_blocks[0].metadata["granularity"] = "document"

    return blocks, links, tag_blocks


def _get_file_created_time(file_path: Path) -> str | None:
    """File creation time as an ISO string; modification time when the OS
    has no creation time.

    This is the last fallback in `_resolve_timestamp`, so it only matters
    for notes with no date in the heading, filename, or frontmatter. Using
    mtime there stamped every edit of an undated MOC onto its blocks, which
    is why "what was I doing in March" surfaced notes touched in August.
    macOS and Windows expose `st_birthtime`; Linux stat() does not, and a git
    checkout resets it everywhere, so mtime stays as the floor.
    """
    try:
        stat = file_path.stat()
        ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _resolve_timestamp(
    h3_date: str | None,
    title_date: str | None,
    fm_created: str | None,
    file_created: str | None,
) -> str:
    """Apply timestamp priority: h3 > filename > frontmatter created > file_created > now.

    `created:` frontmatter sits below filename dates (daily notes stay
    authoritative) but above filesystem time, which is wrong for anything
    synced or imported — this is how converters (gdrive, chatgpt) stamp
    real historical dates onto imported docs.
    """
    if h3_date:
        return h3_date
    if title_date:
        return title_date
    if fm_created:
        return fm_created
    if file_created:
        return file_created
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_file_index(files: list[Path], vault_root: Path) -> dict[str, str]:
    """Build {stem → relative_path} index for fast wikilink resolution.

    If multiple files share a stem, the first one wins (matches Obsidian behavior
    for shortest-path resolution).
    """
    index: dict[str, str] = {}
    for f in files:
        stem = f.stem
        if stem not in index:
            index[stem] = str(f.relative_to(vault_root))
    return index


def _resolve_wikilink(link_target: str, file_index: dict[str, str]) -> str:
    """Resolve a wikilink title to a relative path using pre-built index.

    O(1) lookup instead of rglob per link.
    """
    return file_index.get(link_target, f"{link_target}.md")


# Document hashes gate file-level incremental parsing, so they are salted
# with the splitter version: a segmentation-rule change re-parses every file
# once (block-level diffing keeps unchanged segments, so only notes whose
# segmentation actually changed churn).
_DOC_HASH_SALT = f"splitter-v{_splitter.SPLITTER_VERSION}:".encode()


def _hash_file(file_path: Path) -> str:
    """Salted SHA-256 hash of file content (truncated to 16 hex chars)."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(_DOC_HASH_SALT + f.read()).hexdigest()[:16]


def _hash_content(content: str) -> str:
    """Salted SHA-256 hash of string content — must match `_hash_file`."""
    return hashlib.sha256(_DOC_HASH_SALT + content.encode("utf-8")).hexdigest()[:16]


def _should_include(file_path: Path, vault_root: Path, patterns: list[str]) -> bool:
    """Check if a file should be included (not matched by exclude patterns)."""
    try:
        relative = str(file_path.relative_to(vault_root))
    except ValueError:
        return False
    return all(not _matches_pattern(relative, pattern) for pattern in patterns)


def _matches_pattern(path: str, pattern: str) -> bool:
    """Check if path matches a glob-like exclude pattern. Same logic as v1."""
    # **/dir/** — matches a directory name anywhere in the path
    if pattern.startswith("**/") and pattern.endswith("/**"):
        middle = pattern[3:-3]
        return ("/" + middle + "/") in ("/" + path)
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return path.startswith(prefix) or path.startswith(prefix + "/")
    if pattern.startswith("**/"):
        suffix = pattern[3:]
        return path.endswith(suffix) or ("/" + suffix) in path
    if "*" not in pattern:
        return path == pattern or path.startswith(pattern + "/")
    if pattern.count("*") == 1:
        parts = pattern.split("*")
        return path.startswith(parts[0]) and path.endswith(parts[1])
    return False


def _normalize_source_rules(source_rules: dict[str, str] | None) -> list[tuple[str, str]] | None:
    """[vault.source_rules] config dict → ordered (pattern, tag) list.

    Tags may be written with or without a leading '#'; stored without.
    First matching rule wins (dict order = file order in TOML).
    """
    if not source_rules:
        return None
    return [(pattern, tag.lstrip("#")) for pattern, tag in source_rules.items()]


def _apply_source_rules(
    rel_path: str,
    tags: list[str],
    source_rules: list[tuple[str, str]] | None,
) -> list[str]:
    """Stamp a source/* tag from folder rules — unless the text already has one.

    An explicit source/* tag in the note text always wins: text is truth,
    the rule only fills the gap for bulk-imported folders.
    """
    if not source_rules or any(t.startswith("source/") for t in tags):
        return tags
    for pattern, tag in source_rules:
        if _matches_pattern(rel_path, pattern):
            return [*tags, tag]
    return tags


# ── Provenance ─────────────────────────────────────────────────────
#
# Who wrote a block: the user, a model, or someone else whose work was
# imported. A derived field, not a tag, because it is a property every block
# has exactly one value of and every read tool needs to filter on. The
# 2026-09-01 high-note analysis quoted forty AI-written reflection sessions
# back to the user as "your vault" because nothing distinguished them from his
# own writing at the query layer — docs/plans/query-provenance-and-dates.md.

PROVENANCE_HUMAN = "human"
PROVENANCE_AI = "ai"
PROVENANCE_REFERENCE = "reference"
PROVENANCE_VALUES = (PROVENANCE_HUMAN, PROVENANCE_AI, PROVENANCE_REFERENCE)

# Tags that decide provenance when no explicit provenance/* tag and no path
# rule applies. Closed list from the taxonomy's note-type and source facets.
_AI_TAGS = frozenset({"note-type/ai-summary", "note-type/ai-response", "source/ai-chat"})
_HUMAN_SOURCE_TAGS = frozenset({"source/capture"})


def _normalize_provenance_rules(
    provenance_rules: dict[str, str] | None,
) -> list[tuple[str, str]] | None:
    """[vault.provenance_rules] config dict → ordered (pattern, value) list.

    First matching rule wins (dict order = file order in TOML), so a caller
    lists the narrower folder first: `OpenAugi/Capture/** = human` before
    `OpenAugi/** = ai`. Unknown values raise — a typo here would silently
    mislabel a whole folder.
    """
    if not provenance_rules:
        return None
    out: list[tuple[str, str]] = []
    for pattern, value in provenance_rules.items():
        v = value.strip().lower()
        if v not in PROVENANCE_VALUES:
            raise ValueError(
                f"[vault.provenance_rules] {pattern!r}: {value!r} "
                f"is not one of {PROVENANCE_VALUES}"
            )
        out.append((pattern, v))
    return out


def resolve_provenance(
    rel_path: str,
    tags: list[str],
    provenance_rules: list[tuple[str, str]] | None,
) -> str:
    """Decide who wrote a block. First match wins:

    1. An explicit `provenance/<value>` tag in the note text — text is truth.
    2. The first matching path rule from [vault.provenance_rules].
    3. Tag rules: the AI note-type/source tags give `ai`; any other
       `source/*` tag except `source/capture` is imported material, `reference`.
    4. `human`.
    """
    for t in tags:
        if t.startswith("provenance/"):
            v = t.split("/", 1)[1].lower()
            if v in PROVENANCE_VALUES:
                return v
    if provenance_rules:
        for pattern, value in provenance_rules:
            if _matches_pattern(rel_path, pattern):
                return value
    tag_set = set(tags)
    if tag_set & _AI_TAGS:
        return PROVENANCE_AI
    if any(t.startswith("source/") and t not in _HUMAN_SOURCE_TAGS for t in tag_set):
        return PROVENANCE_REFERENCE
    return PROVENANCE_HUMAN


def backfill_provenance(
    store,
    provenance_rules: dict[str, str] | None,
    dry_run: bool = False,
    title_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Stamp `metadata.provenance` on every data_block already in the DB.

    Ingest only touches changed files, so adding rules does nothing for
    existing rows. Applies the same resolution as `_parse_file`. Idempotent.

    `title_patterns` is a reporting aid for the one case a rule cannot see:
    AI output the user pasted into a human folder (a "- Jung - " reflection
    in the inbox). Blocks whose title contains a pattern AND resolved to
    `human` are listed under `candidates` for the user to tag by hand. They
    are never relabelled from a title match — content is not evidence.

    Returns {"updated": {value: count}, "unchanged": n, "candidates": [(id, title)]}.
    """
    import json as _json

    rules = _normalize_provenance_rules(provenance_rules)
    patterns = [p for p in (title_patterns or []) if p]

    rows = store.conn.execute(
        """SELECT id, title, tags, metadata FROM blocks WHERE kind = 'data_block'"""
    ).fetchall()

    updates: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    unchanged = 0
    candidates: list[tuple[str, str]] = []

    for block_id, title, tags_json, metadata_json in rows:
        metadata = _json.loads(metadata_json) if metadata_json else {}
        source_path = metadata.get("source_path")
        if not source_path:
            continue
        tags = _json.loads(tags_json) if tags_json else []
        tags = tags + metadata.get("augi_tags", [])
        value = resolve_provenance(source_path, tags, rules)
        if value == PROVENANCE_HUMAN and title and any(p in title for p in patterns):
            candidates.append((block_id, title))
        if metadata.get("provenance") == value:
            unchanged += 1
            continue
        metadata["provenance"] = value
        counts[value] = counts.get(value, 0) + 1
        updates.append((_json.dumps(metadata), block_id))

    if updates and not dry_run:
        store.conn.executemany("UPDATE blocks SET metadata = ? WHERE id = ?", updates)
        store.conn.commit()
        logger.info("Backfilled provenance on %d blocks: %s", len(updates), counts)

    return {"updated": counts, "unchanged": unchanged, "candidates": candidates}


def backfill_source_tags(
    store,
    source_rules: dict[str, str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Apply source rules to data_blocks already in the DB.

    Ingest only touches changed files, so turning on [vault.source_rules]
    does nothing for existing rows — this walks every data_block, applies
    the same explicit-tag-wins logic as _parse_file, and updates tags +
    groups links in place. Idempotent. Returns {tag: blocks_updated}.
    """
    import json as _json

    from openaugi.model.link import Link as _Link

    rules = _normalize_source_rules(source_rules)
    if not rules:
        return {}

    rows = store.conn.execute(
        """SELECT id, tags, json_extract(metadata, '$.source_path')
           FROM blocks WHERE kind = 'data_block'"""
    ).fetchall()

    updates: list[tuple[str, str]] = []  # (block_id, new_tags_json)
    tag_links: list[Link] = []
    stats: dict[str, int] = {}
    needed_tag_blocks: dict[str, Block] = {}

    for block_id, tags_json, source_path in rows:
        if not source_path:
            continue
        tags = _json.loads(tags_json) if tags_json else []
        new_tags = _apply_source_rules(source_path, tags, rules)
        if new_tags == tags:
            continue
        added = new_tags[-1]
        stats[added] = stats.get(added, 0) + 1
        updates.append((_json.dumps(new_tags), block_id))
        tag_id = Block.make_tag_id(added)
        if added not in needed_tag_blocks:
            needed_tag_blocks[added] = Block(
                id=tag_id, kind="context_block:tag", title=added, source="vault"
            )
        tag_links.append(_Link(from_id=block_id, to_id=tag_id, kind="groups"))

    if dry_run or not updates:
        return stats

    store.insert_blocks(list(needed_tag_blocks.values()))
    store.conn.executemany("UPDATE blocks SET tags = ? WHERE id = ?", updates)
    store.conn.commit()
    store.insert_links(tag_links)
    logger.info("Backfilled source tags on %d blocks: %s", len(updates), stats)
    return stats


def _unique_ordered(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
