---
name: splitter
description: Deterministic block splitter — the shared primitive any agent, script, or adapter can use to chunk markdown notes the same way OpenAugi ingest does.
---

# Splitter

One function, one contract: given markdown text, return the same blocks that `openaugi ingest` would produce — without touching a database, loading a model, or calling an LLM.

This is the shared primitive. Agents, skills, scripts, `cron` jobs, and future adapters all import the same splitter so "what is a block in OpenAugi" has exactly one definition.

## When to use

- You're writing a skill or agent that needs to reason over chunks of a daily note the same way OpenAugi does.
- You want a `zk-split`-style CLI to pipe blocks into another tool.
- You're building an adapter (PDF, web, chat exports) and need segment boundaries matching the vault flow.
- You want to preview how a note will be split before running `openaugi ingest`.

If you just want blocks written to the store, use `openaugi ingest` — it calls the same splitter under the hood.

## CLI: `openaugi split`

```bash
# Full JSON output (one object, segments in an array)
openaugi split path/to/note.md

# Stream one segment per line (good for xargs / jq pipelines)
openaugi split path/to/note.md --format ndjson

# Human-readable preview
openaugi split path/to/note.md --format md
```

The JSON shape:

```json
{
  "source_path": "path/to/note.md",
  "doc_hash": "4040a48076e0d055",
  "filename_date": "2026-04-08",
  "frontmatter_tags": ["journal"],
  "segments": [
    {
      "content": "…raw sub-section, zzz lines included…",
      "clean_content": "…zzz lines stripped, ready to store or display…",
      "zzz_instructions": ["research this later"],
      "tags": ["career", "focus"],
      "links": ["Project Alpha"],
      "section_heading": "Morning",
      "section_date": "2026-04-08",
      "granularity": "section",
      "raw_hash": "a1b2c3d4e5f60718"
    }
  ]
}
```

`raw_hash` is a stable identity for the segment. If you re-split the same file unchanged, the hash is the same; if you edit a `zzz` instruction inside the block, the hash changes (the raw form includes the zzz line).

## Python API

```python
from openaugi.adapters.splitter import split_file, split_text

# In-memory
segments = split_text("# Day\nfirst thought\nqqq\nsecond thought\nzzz: research this")
for seg in segments:
    print(seg.clean_content, seg.zzz_instructions)

# From disk (frontmatter tags, filename date resolution included)
result = split_file("path/to/note.md")
print(result.filename_date, result.frontmatter_tags)
for seg in result.segments:
    print(seg.raw_hash, seg.section_heading, seg.clean_content[:60])
```

Both `Segment` and `SplitResult` are Pydantic models — call `.model_dump()` for a plain dict or `.model_dump_json()` for a JSON string.

## Splitting rules

Applied in this order:

1. **Frontmatter.** YAML frontmatter at the top of the file is stripped; `tags:` in it are captured (exposed via `SplitResult.frontmatter_tags` from `split_file` — `split_text` discards them).
2. **Headings.** Any markdown heading (`#`–`######`) starts a new section. `#` lines inside fenced code blocks (``` or `~~~`) are ignored.
3. **`qqq` markers.** Within a section, standalone `qqq` lines split further. A section without `qqq` stays whole.
4. **`zzz` instructions.** Lines starting with `zzz` (optionally `zzz: …`) are extracted into `zzz_instructions` and stripped from `clean_content`. The segment's `raw_hash` still covers them, so editing a zzz line produces a new block.
5. **Empty / structural-only segments** (just `---`, empty checkboxes, URL-only lines, dataview queries) are dropped.
6. **Dates.** A `YYYY-MM-DD` prefix on a heading sets the date for that section and every section after it until the next date-headed section. `split_file` also extracts a filename date (`2026-04-08-*.md` or `WK - 25-11-09.md`) as a fallback.
7. **Weekly-reflection notes** (2-digit-year `WK` stems) are kept as one segment so question/answer pairs don't fragment.

## Contract guarantees

- **Deterministic.** Same input string → same output, byte-for-byte. No timestamps, no randomness.
- **No side effects.** Pure functions. No filesystem writes from `split_text`; `split_file` only reads.
- **No dependencies on a store or model.** Import path: `openaugi.adapters.splitter`.
- **Stable hashes.** `raw_hash = sha256(raw_segment_content)[:16]`. You can use it as a content-addressable ID for cache keys or diffing.

If you need the full Block+Link shape (document blocks, tag blocks, `links_to` edges), call `openaugi.adapters.vault.parse_vault` — that wraps the splitter with vault-level context.

## Related

- [zzz-instructions plan](plans/zzz-instructions.md) — the `zzz:` convention
- [vault adapter](../src/openaugi/adapters/vault.py) — splitter + wikilink index + Block/Link assembly
- [splitter source](../src/openaugi/adapters/splitter.py) — the implementation
