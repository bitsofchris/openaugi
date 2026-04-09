"""Obsidian vault adapter — reads .md files into blocks + links.

Port of augi-engine-v1's ObsidianEntryParser into the blocks+links data model.

What stays from v1:
- Regex patterns (H3_DATE, TAG, LINK, FILENAME_DATE, FRONTMATTER)
- Splitting logic (_split_by_h3_dates)
- Date resolution priority (h3 > filename > file mtime)
- Exclude patterns, concurrent file reading

What changes:
- Output: Block + Link instead of Entry dataclass
- Tags become Block(kind="tag") + Link(kind="tagged")
- File hashes tracked via document block content_hash
- Block ID = hash(source_path + content_hash) — stable across reordering
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openaugi.model.block import Block
from openaugi.model.link import Link

logger = logging.getLogger(__name__)

# ── Regex patterns ───────────────────────────────────────────────

ANY_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)$", re.MULTILINE)
H3_DATE_PATTERN = re.compile(r"^###\s+(\d{4}-\d{2}-\d{2})", re.MULTILINE)  # kept for reference
TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z0-9_/\-]+)")
LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
FILENAME_DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})")
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Atomic thought delimiter: a line containing only `qqq` (case-insensitive).
# Used to split sections into sub-blocks finer than H3 headers.
# See docs/plans/zzz-instructions.md.
QQQ_PATTERN = re.compile(r"^[ \t]*[qQ]{3}[ \t]*$", re.MULTILINE)
# Per-block agent instructions: lines starting with `zzz` (case-insensitive),
# optionally followed by a colon. Each match becomes one item in the
# block's `zzz_instructions` metadata list. The prefix is stripped from
# clean content before hashing. Multiple zzz lines in one block are kept
# as separate instructions. See docs/plans/zzz-instructions.md.
ZZZ_PATTERN = re.compile(r"^[ \t]*[zZ]{3}\b[:\s]*(.*?)\s*$", re.MULTILINE)

DEFAULT_EXCLUDE_PATTERNS = [
    ".obsidian/**",
    ".git/**",
    ".smart-env/**",
    ".trash/**",
    "templates/**",
    "OpenAugi/Compiled/**",
]


# ── Public API ─────────────────────────────────────────────────────


def parse_vault(
    vault_path: str | Path,
    exclude_patterns: list[str] | None = None,
    max_workers: int = 4,
) -> tuple[list[Block], list[Link]]:
    """Parse an Obsidian vault into blocks and links.

    Returns (blocks, links) ready to insert into the store.
    """
    vault = Path(vault_path)
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault}")
    excludes = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS

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
        futures = {executor.submit(_parse_file, f, vault, file_index): f for f in included}
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
) -> tuple[list[Block], list[Link], dict[str, str], list[str]]:
    """Parse vault with incremental change detection.

    Args:
        vault_path: Path to Obsidian vault.
        known_doc_hashes: {relative_path: content_hash} from previous run.
        exclude_patterns: Glob patterns to skip.
        max_workers: Thread pool size.

    Returns:
        (new_blocks, new_links, current_hashes, deleted_paths)
        - new_blocks/links: only from changed/new files
        - current_hashes: {relative_path: hash} for all current files
        - deleted_paths: relative paths no longer on disk
    """
    vault = Path(vault_path)
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault}")
    excludes = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS

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
        futures = {executor.submit(_parse_file, f, vault, file_index): f for f in files_to_parse}
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
    file_path: Path, vault_root: Path, file_index: dict[str, str]
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

    # Document block
    file_hash = _hash_content(content)
    doc_id = Block.make_document_id(rel_path)
    doc_block = Block(
        id=doc_id,
        kind="document",
        title=parent_title,
        source="vault",
        content_hash=file_hash,
        metadata={"source_path": rel_path},
    )
    blocks.append(doc_block)

    # Strip frontmatter, extract frontmatter tags
    content_body, fm_tags = _strip_frontmatter(content)

    # Split by any heading; date flows down from nearest date-heading ancestor
    sections = _split_by_headings(content_body)

    # Date from filename
    title_date = _extract_filename_date(file_path)
    file_created = _get_file_created_time(file_path)

    for section_content, section_date_str, section_heading in sections:
        # Each section is further split on `qqq` markers. When a section
        # has no qqq, _split_by_qqq returns [section_content] unchanged so
        # notes that don't use qqq behave exactly like before.
        sub_sections = _split_by_qqq(section_content)

        for sub_content in sub_sections:
            stripped = sub_content.strip()
            if not stripped:
                continue

            # Extract `zzz` agent instructions. The hash is computed on the
            # *raw* sub-section (including zzz) so that adding or editing a
            # zzz instruction triggers a new block — the heartbeat agent
            # then picks it up on the next run. The block's stored content
            # is the clean version (zzz stripped) so downstream readers see
            # only the note.
            clean_content, zzz_instructions = _extract_zzz_instructions(stripped)
            if not clean_content:
                # Sub-block contained only zzz lines — nothing meaningful to store
                continue

            # Entry block — identity keyed by raw sub-content (zzz-sensitive)
            entry_hash = Block.hash_content(stripped)
            entry_id = Block.make_id(rel_path, entry_hash)

            section_date = _parse_date(section_date_str) if section_date_str else None
            resolved_ts = _resolve_timestamp(section_date, title_date, file_created)

            # Extract tags (inline + frontmatter) from clean content
            inline_tags = _extract_tags(clean_content)
            all_tags = _unique_ordered(fm_tags + inline_tags)

            # Extract wikilinks from clean content
            entry_links = _extract_links(clean_content)

            entry_metadata: dict = {
                "source_path": rel_path,
                "section_date": section_date_str,
                "section_heading": section_heading,
                "parent_note_title": parent_title,
                "file_created_at": file_created,
            }
            if zzz_instructions:
                entry_metadata["zzz_instructions"] = zzz_instructions

            entry_block = Block(
                id=entry_id,
                kind="entry",
                content=clean_content,
                source="vault",
                title=parent_title,
                tags=all_tags,
                timestamp=resolved_ts,
                content_hash=entry_hash,
                metadata=entry_metadata,
            )
            blocks.append(entry_block)

            # Link: entry -> document (split_from)
            links.append(Link(from_id=entry_id, to_id=doc_id, kind="split_from"))

            # Tag blocks + tagged links
            for tag_name in all_tags:
                tag_id = Block.make_tag_id(tag_name)
                if tag_name not in tag_blocks:
                    tag_blocks[tag_name] = Block(
                        id=tag_id, kind="tag", title=tag_name, source="vault"
                    )
                links.append(Link(from_id=entry_id, to_id=tag_id, kind="tagged"))

            # Wikilink links
            for link_target in entry_links:
                # Resolve to document block if it exists in this vault
                target_doc_id = Block.make_document_id(_resolve_wikilink(link_target, file_index))
                links.append(
                    Link(
                        from_id=entry_id,
                        to_id=target_doc_id,
                        kind="links_to",
                        metadata={"target_title": link_target},
                    )
                )

    return blocks, links, tag_blocks


def _split_by_headings(content: str) -> list[tuple[str, str | None, str | None]]:
    """Split content by any markdown heading (# through ######).

    Date context flows down: the nearest ancestor heading whose text begins
    with YYYY-MM-DD establishes the date for all subsequent sections until
    the next date heading appears.

    Returns list of (section_content, date_str, heading_text):
    - section_content: text after the heading line (heading line not included)
    - date_str: inherited YYYY-MM-DD string, or None
    - heading_text: heading text without # prefix, or None for preamble
    """
    matches = list(ANY_HEADING_PATTERN.finditer(content))

    if not matches:
        return [(content, None, None)]

    sections: list[tuple[str, str | None, str | None]] = []
    current_date: str | None = None

    # Preamble before the first heading
    if matches[0].start() > 0:
        preamble = content[: matches[0].start()]
        if preamble.strip():
            sections.append((preamble, None, None))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()

        # Update date context if this heading begins with a date
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", heading_text)
        if date_match:
            current_date = date_match.group(1)

        # Content is everything after the heading line to the next heading
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[content_start:content_end]

        sections.append((section_content, current_date, heading_text))

    return sections


def _split_by_qqq(content: str) -> list[str]:
    """Split a section on standalone `qqq` lines (case-insensitive).

    Each segment is the text between two qqq markers (or between start/end
    and a qqq marker). Empty / whitespace-only segments are NOT dropped
    here — the caller decides, since it also needs to reject zzz-only
    segments after zzz extraction.

    If there are no qqq markers, returns [content] unchanged, so callers
    that don't use qqq markers see the same behavior as before.
    """
    matches = list(QQQ_PATTERN.finditer(content))
    if not matches:
        return [content]

    segments: list[str] = []
    cursor = 0
    for match in matches:
        segments.append(content[cursor : match.start()])
        cursor = match.end()
    segments.append(content[cursor:])
    return segments


def _strip_frontmatter(content: str) -> tuple[str, list[str]]:
    """Remove YAML frontmatter, extract tags from it."""
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return content, []

    fm_text = match.group(1)
    body = content[match.end() :]

    # Simple YAML tag extraction (avoid full yaml dependency)
    tags: list[str] = []
    in_tags = False
    for line in fm_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("tags:"):
            in_tags = True
            # Handle inline tags: [tag1, tag2]
            rest = stripped[5:].strip()
            if rest.startswith("["):
                tags.extend(
                    t.strip().strip("'\"") for t in rest.strip("[]").split(",") if t.strip()
                )
                in_tags = False
            continue
        if in_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip("'\""))
            elif stripped and not stripped.startswith("-"):
                in_tags = False

    return body, tags


def _extract_tags(text: str) -> list[str]:
    """Extract inline #tags, preserving order, deduped."""
    matches = TAG_PATTERN.findall(text)
    return _unique_ordered(matches)


def _extract_links(text: str) -> list[str]:
    """Extract [[wikilinks]], preserving order, deduped."""
    matches = LINK_PATTERN.findall(text)
    return _unique_ordered(matches)


def _extract_zzz_instructions(text: str) -> tuple[str, list[str]]:
    """Extract `zzz` agent instruction lines from block content.

    Lines starting with `zzz` (case-insensitive, optionally `zzz: ...`) are
    per-block directives for the heartbeat agent. They're stripped from the
    clean content and returned as a list of instruction strings — one per
    `zzz` line, in document order.

    Multiple zzz lines in the same block are kept as separate instructions
    so the agent can honor each independently. Bare `zzz` lines with no body
    are dropped (nothing to act on).

    Returns (clean_content, instructions) where instructions is `[]` if none.
    """
    instructions: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            instructions.append(body)
        return ""  # strip the line entirely

    stripped = ZZZ_PATTERN.sub(_capture, text)
    # Collapse any blank lines left behind by the strip
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip("\n")
    return stripped, instructions


def _extract_filename_date(file_path: Path) -> str | None:
    """Extract YYYY-MM-DD from filename."""
    match = FILENAME_DATE_PATTERN.match(file_path.stem)
    if match:
        return match.group(1)
    return None


def _get_file_created_time(file_path: Path) -> str | None:
    """Get file creation time as ISO string."""
    try:
        stat = file_path.stat()
        ts = stat.st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _resolve_timestamp(
    h3_date: str | None,
    title_date: str | None,
    file_created: str | None,
) -> str:
    """Apply timestamp priority: h3 > filename > file_created > now."""
    if h3_date:
        return h3_date
    if title_date:
        return title_date
    if file_created:
        return file_created
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(date_str: str) -> str | None:
    """Validate YYYY-MM-DD string."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except (ValueError, TypeError):
        return None


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


def _hash_file(file_path: Path) -> str:
    """SHA-256 hash of file content (truncated to 16 hex chars)."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _hash_content(content: str) -> str:
    """SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _should_include(file_path: Path, vault_root: Path, patterns: list[str]) -> bool:
    """Check if a file should be included (not matched by exclude patterns)."""
    try:
        relative = str(file_path.relative_to(vault_root))
    except ValueError:
        return False
    return all(not _matches_pattern(relative, pattern) for pattern in patterns)


def _matches_pattern(path: str, pattern: str) -> bool:
    """Check if path matches a glob-like exclude pattern. Same logic as v1."""
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


def _unique_ordered(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
