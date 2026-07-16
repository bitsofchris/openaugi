---
name: anchor-segmentation
description: Obsidian block anchors as segment boundaries — capture daily notes ingest per entry instead of as one whole-day block.
---

# Anchor segmentation

**Status: shipped 2026-07-15** (branch `feat/anchor-segmentation`).

## The problem

The splitter cut notes on headings and `qqq` lines only. A mobile capture
daily note (`OpenAugi/Capture/YYYY-MM-DD.md` — one `# date` header,
timestamped entries each closed by a bare `^augi-<id8>` anchor line)
therefore ingested as ONE document-level block:

- one embedding for the whole day;
- zzz / tags / `#layer/bronze` attributed to the day-blob, not the entry;
- editing any entry rehashed the entire day into a new block (re-dispatching
  every zzz in it);
- the bronze down-weighting (2026-07-15) needed a workaround — a block only
  counted bronze when *every* anchored entry carried the tag.

`test_capture_daily_note_ingests_to_expected_blocks` pinned the whole-day
behavior as intentional; this change reverses that decision deliberately.

## The rule

A line consisting solely of an Obsidian block anchor (`^[A-Za-z0-9-]+` after
trim) **closes the current segment** — the anchor names the content above it,
per Obsidian's own block-reference semantics. Format-native, not a
mobile-specific delimiter: it applies to any note in the vault. Heading and
`qqq` splitting are unchanged; the anchor rule nests inside qqq sub-sections
the same way qqq nests inside sections. Inline carets and block *references*
(`[[note#^ref]]`) don't split — only a line that is solely an anchor.

## Design decisions

1. **Anchor placement.** The anchor line stays in the segment's raw
   `content` (so `raw_hash` — block identity — covers it) but is extracted
   to `Segment.anchor_id` (without the `^`) and stripped from
   `clean_content`, mirroring zzz extraction. The vault adapter carries it
   as block metadata `anchor_id`. Content stored/embedded never contains
   anchor lines.

   *Considered and deferred:* using the anchor as the durable block
   identity so an edit is an update instead of remove+add. Today identity =
   `hash(source_path + content_hash)` and the runner's block-level diff is
   hash-set-based; more importantly, remove+add is the *established
   semantic* — an edited block deliberately drops agent state (`routed_to`,
   `augi_tags`) and re-enters the review queue (see runner.py). Anchor-keyed
   identity would need a story for that state on edit, so it's future work,
   not a blocker for per-entry granularity.

2. **Entry timestamps.** Capture entries lead with `HH:MM — ` (or a bare
   `HH:MM —` line when the entry opens with a grammar token). For anchored
   segments the splitter parses this into `Segment.entry_time` ("HH:MM",
   zero-padded, validated < 24:00); the vault adapter refines a date-only
   resolved timestamp into `YYYY-MM-DDTHH:MM:00`. Entries get real
   time-of-day ordering instead of all inheriting the bare file date.
   Unanchored segments are left alone (a stray `10:30 — ` in prose
   shouldn't rewrite a note's timestamp).

3. **Trailing content after the last anchor** (human annotations,
   unanchored text) falls out of the implementation as its own anchor-less
   segment — `_split_by_anchor` appends the remainder and the existing
   empty/structural filters drop it when there's nothing there. A dangling
   anchor with no content above it names nothing and is dropped.

4. **Dispatch semantics.** With per-entry segments a zzz attaches to its
   own entry, so `source_block_id` in task files is precise and the task
   `## Context` is the entry, not the day. Dispatch fires on
   `new_data_blocks` from the runner's hash diff, so an unrelated edit
   elsewhere in the day no longer re-adds (= re-dispatches) zzz blocks —
   pinned by `TestAnchoredCaptureIncremental`. Task-file naming
   (slug + second-resolution timestamp) is unchanged.

5. **Bronze simplification.** The whole-day workaround in
   `mcp/server.py` (`_all_entries_bronze`: only treat a block as bronze if
   every anchored entry carried the tag inline) is removed — per-entry
   blocks make per-block bronze real, so bronze is by tag alone. For
   genuinely single-block notes the workaround already returned "bronze by
   tag alone", so behavior there is identical.

## Migration

Segmentation output changed, so `SPLITTER_VERSION` (new, in splitter.py) is
salted into the vault adapter's document hashes. Effect on the **first
ingest after upgrading**:

- *Every* file's doc hash misses, so every file re-parses once — expect a
  `Change detection: <all> changed/new` log line. This is parse-only churn:
  block-level diffing keeps unchanged segments, so for notes without bare
  anchor lines everything lands in `blocks_kept` and no embeddings rerun.
- Notes *with* bare anchor lines (capture daily notes) re-segment: the
  day-block is removed, N per-entry blocks are added and embedded, and any
  zzz entries in them re-dispatch once (their blocks are genuinely new).
  Expect `Block-level diff: … kept, N added, 1 removed` per capture note
  and a matching embed count. Bounded, one-time churn — not a bug.
- Any `routed_to`/`augi_tags` on old day-blocks CASCADE away with them;
  the per-entry blocks re-enter the review queue (standard edited-block
  semantics).

## Contract fixture

`tests/fixtures/contracts/capture-daily-note.md` — the fixture FILE is
unchanged (the mobile writer needs no changes; mobile re-vendors via its
`scripts/sync-contract-fixtures.sh`). Only the pinned expectations in
`tests/test_contract_fixtures.py` changed: 5 per-entry blocks, each carrying
its `anchor_id` and per-entry `block_time`; the two zzz instructions land on
their own entries.
