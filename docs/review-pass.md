---
name: review-pass
description: The write-back loop — routes new blocks into containers via augi_tags (DB-only), regenerates derived view notes in OpenAugi/Views/, and nominates structure changes on the Dashboard. Covers how augi_tags work, the capture grammar (qqq/zzz/aaa), how to run a pass, and what the human does.
---

# Review Pass

## When to use this doc

- You want to know what `augi_tags` are and where classifications live
- You're about to run (or debug) a review pass
- You forgot the capture grammar or what the Dashboard nominations mean

Design record with full rationale: [docs/plans/review-pass-v1.md](plans/review-pass-v1.md).
Agent instructions (the live prompt): `<vault>/OpenAugi/AGENT/review-pass.md`.

## The frame in one paragraph

Event sourcing applied to a vault. **Truth** is the user's own writing — append-only,
never edited by agents. **Views** are agent-generated markdown files under
`OpenAugi/Views/` — derived, regenerable caches that need no review (a wrong view
is a stale cache entry, not damage). Human review gates only **structure changes**:
new tags/areas, promoting a block cluster to a new note, merges. The agent
nominates on the Dashboard; the human commands; the agent assembles.

## augi_tags — the two tag layers

Every block has two independent tag sets:

| Layer | Where it lives | Who writes it | What it means |
|---|---|---|---|
| `block.tags` | Parsed from the user's markdown (`#area/self` etc.) | The user | Ground truth |
| `block.metadata["augi_tags"]` | SQLite only — **never written into notes** | The agent, via the `tag_block` MCP tool | Derived classification + routing |

The review pass stamps each new block's `augi_tags` with facet tags
(`area/*`, `type/*`) plus one `routed/<container-slug>` tag per container the
block belongs to (e.g. `routed/amoc-openaugi-main`). `tag_block` overwrites the
whole list — re-routing a block is one call. `search(tags=[...])` matches both
layers, so views and lenses can query either.

This is why routing costs nothing and is always correctable: the user's files
never change; only DB metadata does.

## Capture grammar

Three tokens, written anywhere in a note:

- `qqq` — block delimiter (splits a note into blocks at ingest)
- `zzz: <instruction>` — agent dispatch: spawn a task (see [task-dispatch.md](task-dispatch.md))
- `aaa: <instruction>` — routing instruction to the review pass:
  "aaa: route to OpenAugi Mobile", "aaa: find my note on X and link this"

Routing precedence: `aaa:` > explicit `[[MOC link]]`/`#area/*` in the block >
location (a block in a MOC's journal is home) > inference (most specific
container wins) > unrouted (Dashboard). Routing ≠ surfacing — every block
routes, views surface only what's salient.

## Running a pass

The pass is an agent skill, not a pipeline. In any Claude session with the
openaugi MCP server: say **"run the review pass"** (or dispatch
`zzz: run the review pass`). The agent then:

1. `get_review_state()` → the high-water mark (`meta` table keys
   `review_pass_last_run` / `review_pass_last_summary`)
2. `search(after=last_run)` → new blocks (excludes `OpenAugi/`-sourced blocks)
3. Routes each block → `tag_block(id, augi_tags)`
4. Regenerates `View - <container>.md` for touched containers via
   `write_document(..., subfolder="Views", overwrite=True)`
   (`overwrite=True` is only legal for Views)
5. Regenerates `View - Dashboard.md` — rollup, task union, gravity nominations
6. `mark_review_complete(summary)` → advances the mark

Cadence: manual, attached to the Sunday weekly plan. Schedule only after it's
boringly reliable.

## What the human does

- **One-time:** add `![[View - <AMOC name>]]` transclusions to each area MOC's
  top matter, so the MOC renders its derived head in place.
- **Weekly:** read `View - Dashboard.md`; answer the gravity nominations
  (inline note or zzz) — that's the entire review burden.
- **Anytime:** correct a bad route by telling the agent, or with an `aaa:`
  on future captures. Wrong routes are tuning signal.
