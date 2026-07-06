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

## Tags vs. routing — two mechanisms, one taxonomy

**Classification is tags; membership is links. Never conflate them.**

**Tags — one closed vocabulary, two authors.** There is exactly one taxonomy
(the user's `My Taxonomy`). What differs is who applied a tag:

| Layer | Where it lives | Who writes it |
|---|---|---|
| `block.tags` | Parsed from the user's markdown (`#area/self` etc.) | The user — ground truth |
| `block.metadata["augi_tags"]` | SQLite only — **never written into notes** | The agent, via `tag_block`, using the *same* taxonomy vocabulary |

`augi_tags` is not a second tagging system — it is "tags the agent applied,"
kept out of the user's files. The agent never invents a tag or facet, and
never re-tags a block the user already tagged; it only fills gaps.
`search(tags=[...])` matches both layers. **Untagged is a valid state** —
life-log blocks (daily memories) usually carry no tags at all; tag only what
you'd query.

**Routing — `routed_to` links.** The `route_block(block_id, container_title)`
MCP tool records "this block belongs to that container" as a link in the DB
(a block can route to many containers). Views distill a container's routed
blocks. Unrouted is also a valid state — route only what a view should
distill.

Both live in the DB only, which is why routing costs nothing and is always
correctable: the user's files never change.

## The registry is the routing map

Routing targets are the notes tagged `#note-type/amoc` / `#note-type/pmoc` /
`#note-type/moc` (active ones). Each should carry a `description` frontmatter
that says *when to route here* — skill-file style. Human-owned frontmatter on
gold notes = the map the router reads; generated frontmatter under
`OpenAugi/Views/` = agent output. If a registry note lacks a description, the
pass nominates one on the Dashboard instead of guessing.

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
`zzz: run the review pass`). Saying **"process the dashboard"** runs step 0
alone — executes your nomination answers without advancing the high-water
mark. The full pass:

0. Reads the current Dashboard for the user's inline answers to prior
   nominations and executes approved ones — before any regeneration
   overwrites them
1. `get_review_state()` → the high-water mark (`meta` table keys
   `review_pass_last_run` / `review_pass_last_summary`)
2. `search(after=last_run)` → new blocks (excludes `OpenAugi/`-sourced blocks)
3. Routes each block → `route_block(id, container_title)` for membership;
   `tag_block(id, augi_tags)` only where classification has signal
4. Regenerates `View - <container>.md` for touched containers via
   `write_document(..., subfolder="Views", overwrite=True)`
   (`overwrite=True` is only legal for Views). Regeneration is a **merge**:
   the prior view is read first as the head state; new blocks are the delta;
   stale items fall out. Untouched containers keep their old view.
   Every view ends with a `## Log` section — the container's routed blocks,
   newest first, linked to source notes — so membership is visible in
   Obsidian (DB links otherwise aren't). Every view = **recap + remote log**:
   the recap is synthesized from ALL member blocks *including the user's own
   writing in the container note* (their words are upstream input — never
   contradicted; drift gets flagged in one line), and the log lists ONLY
   blocks living in other files, so transclusion never duplicates what's
   already on the page. The container note is the single reading surface:
   user's head/pins → their in-place journal → the transcluded view. Views
   are for embedding, not visiting — add `OpenAugi/Views/` to Obsidian's
   Excluded Files. Agent-created container notes include the transclusion
   at birth.
5. Regenerates `View - Dashboard.md` — rollup, task union, gravity nominations
6. `mark_review_complete(summary)` → advances the mark

Cadence: manual, attached to the Sunday weekly plan. Schedule only after it's
boringly reliable.

## Lenses: views vs. distillations

A **lens** = an intent applied to a scope, producing a derived artifact.
Two lens families exist, with different trust levels:

| | View (review pass) | Distillation (distill lens) |
|---|---|---|
| Trigger | Scheduled / "run the review pass" | On command: "distill X" |
| Scope | Container's routed blocks since last run | Topic (agentic search + links) or user-selected context |
| Output | `OpenAugi/Views/` — regenerable cache, no review | One note in `OpenAugi/Notes|Research/`, `#human-review`, wikilinked provenance |
| Lifecycle | Overwritten every run | Created once, user reviews, graduates toward their curated notes |

Default is neither: **just-in-time distillation in chat** (retrieve + answer,
persist nothing). Persist a distillation only on reuse — the signal is
re-deriving the same synthesis repeatedly. No vault-wide batch distillation:
history is harvested incrementally, pulled by live threads (promotion flow)
or one topic at a time (distill lens). Agent instructions:
`<vault>/OpenAugi/AGENT/distill-lens.md`.

## What the human does

- **One-time:** add `![[View - <AMOC name>]]` transclusions to each area MOC's
  top matter, so the MOC renders its derived head in place.
- **Weekly:** read `View - Dashboard.md`; answer the gravity nominations
  (inline note or zzz) — that's the entire review burden.
- **Anytime:** correct a bad route by telling the agent, or with an `aaa:`
  on future captures. Wrong routes are tuning signal.
