---
name: review-pass (template)
description: >
  TEMPLATE — copied to <vault>/OpenAugi/AGENT/review-pass.md on `openaugi init`.
  The vault copy is the live version the agent reads. Edit there, not here.
  The recurring review/maintenance pass: route new blocks to containers
  (AMOCs/PMOCs), regenerate derived view notes in OpenAugi/Views/, surface
  promotion nominations on the Dashboard. Run manually or via zzz
  ("run the review pass").
---

# Review Pass

You are running the OpenAugi review pass. One loop:

> new blocks → route (tag in DB) → regenerate views → nominate structure changes → advance the high-water mark

Read [[My Taxonomy]] (OpenAugi/AGENT/My Taxonomy.md) first — it defines the
facets and the container registry.

## The frame (do not violate)

- **Truth** = the user's own writing. NEVER edit any note outside `OpenAugi/`.
  Blocks are append-only; nothing is ever deleted.
- **Views** = files you generate under `OpenAugi/Views/`. Derived, regenerable,
  disposable. Regenerate them freely — no review needed.
- **Structure changes** (new tag/area, new silver/gold note, merging notes)
  are NEVER done autonomously. You nominate on the Dashboard; the user
  commands; only then do you assemble.

## Capture grammar

- `qqq` — block delimiter (already handled at ingest).
- `zzz:` — agent dispatch. **Not yours** — the dispatch system handles these.
  Skip zzz blocks for routing commentary but still count them as activity.
- `aaa:` — a routing/parsing instruction addressed to YOU. Obey it.
  Examples: "aaa: route to OpenAugi Mobile", "aaa: find my note on
  Matryoshka embeddings and link this".

## Container registry

The registry is defined in the user's taxonomy (OpenAugi/AGENT/My Taxonomy.md):
the area notes (`#note-type/amoc`, one per `area/*` facet) plus active
projects (`#note-type/pmoc` AND `#status/active`). List each area note with
its `area/*` tag here after `openaugi init` — the AMOCs are the stable
routing targets; active PMOCs are discovered per run.

**The registry notes' `description` frontmatter is the routing map** — it
tells you *when to route here* (skill-file style: name + description). If a
registry note has no description, nominate one on the Dashboard rather than
guessing broadly.

## Routing

**Routing ≠ surfacing.** Every block routes (cheap DB tags); views surface
selectively. A "played fifa, pool with kids" life-log block routes to
`area/self` but does not appear in a view head — salience is decided at
view-generation time, not routing time.

Precedence (highest wins; a block may route to multiple containers):

1. **`aaa:` instruction** in the block — obey it.
2. **Explicit signals**: a `[[AMOC/PMOC/MOC link]]` or `#area/*` tag in the block.
3. **Location**: a block written inside a MOC's own journal is home by
   construction (route to that container; cross-links still allowed).
4. **Inference**: classify `area/*` + `type/*` + `status/*` per taxonomy;
   route to the most *specific* matching container (active PMOC beats its
   parent AMOC).
5. **Low confidence** → leave unrouted; it goes to the Dashboard's
   Unrouted/Gravity section. Never force-fit.

**Persistence — two separate mechanisms, never conflate them:**

- **Membership = links.** `route_block(block_id, container_title)` creates a
  `routed_to` edge in the DB. A block can route to multiple containers.
- **Classification = tags.** `tag_block(block_id, augi_tags)` with facets drawn
  ONLY from the user's taxonomy — it is a closed vocabulary; never invent a
  tag or a facet. If the user already tagged the block, do not re-tag; only
  fill gaps.

**Untagged and unrouted is the default, not a failure.** Life-log blocks
(daily entries, memories) usually need no tag and no route — they stay
reachable by time and semantic search. Tag only what you'd query; route only
what a view should distill. Both live in the DB only — never write into the
user's notes.

## Views

One file per touched container in `OpenAugi/Views/`, named
`View - <container title>.md` (the prefix avoids Obsidian basename
collisions and makes provenance visible). Write with
`write_document(title, description, content, subfolder="Views", overwrite=True)`.
The description should state the question the view answers
(e.g. "Where I left off and what's next in OpenAugi").

Body by container kind (convention — keep each section 3–5 high-signal
bullets, every claim linked to its source note):

- **Area (AMOC)** — rolling TLDR of the area · new-this-period highlights
  (linked) · task/idea rollup · links to active child PMOCs. No LEFT OFF.
- **Project (PMOC)** — TLDR · **LEFT OFF + next physical action** · open
  task list · new-this-period blocks.
- **Concept (MOC)** — "current understanding" summary, updated on revisit.

Footer line on every view:
`*Generated YYYY-MM-DD from N blocks since YYYY-MM-DD.*`

Always regenerate `View - Dashboard.md` (same folder):

- One line per area: what moved, what's next.
- Cross-area task rollup (union of the views' task lists).
- **Gravity section**: unrouted blocks that cluster together — nominate,
  one line each: "5 blocks over 3 weeks orbit *capture UX* — make it a note?"
  Take NO action on nominations. The user answers inline or via zzz;
  only then assemble the new note (gathering related older blocks too —
  that is the resurfacing feature).
- Anything unroutable or confusing, listed honestly.
- Permanent footer: `*How this works: docs/review-pass.md in the openaugi
  repo · design record: docs/plans/review-pass-v1.md · agent instructions:
  OpenAugi/AGENT/review-pass.md*` — the Dashboard is the discovery surface;
  this line keeps the docs findable without remembering them.

## The pass, step by step

1. `get_review_state()` → `since` = last_run. If null, this is the first
   run: backfill from a sensible recent date (e.g. two weeks back, or the
   date the user gives).
2. Pull new blocks: `search(after=since)` (browse mode, paginate via offset).
   Exclude blocks whose source path is under `OpenAugi/` — those are derived,
   not input. Use `recent`/`get_context`/`get_related` for extra context.
3. Route each block per the precedence above; persist membership with
   `route_block`, classification (when there's signal) with `tag_block`.
4. Regenerate a view for every container that received blocks
   (`overwrite=True`). Untouched containers keep their old view.
   **Regeneration is a merge, not a reset:** read the existing view first —
   it is the prior head state. Carry forward what's still true (the TLDR
   evolves; LEFT OFF advances or stands), integrate the new blocks, drop
   what's no longer salient. "New this period" covers only the current
   window. If deeper context is needed, pull the container's full membership
   via its `routed_to` links (`get_related` on the container, direction=in).
5. Regenerate `View - Dashboard.md`.
6. `mark_review_complete(summary)` — one line, e.g.
   "routed 42 blocks; regenerated 4 views + Dashboard; 2 nominations".
   Only call this after views are written successfully.

## Hard rules

- Never modify notes outside `OpenAugi/`. Never use `overwrite=True` outside
  `Views/`.
- Never invent new `area/*` or `type/*` tags — the taxonomy changes only via
  a Dashboard nomination the user approves.
- Every surfaced claim links back to its source note (block IDs in the DB,
  wikilinks in the views).
- Wrong routing is tuning signal, not damage — prefer shipping an imperfect
  pass over stalling. But never force-fit: unrouted is a valid outcome.
- If something is genuinely ambiguous, put it on the Dashboard and move on.
