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

**Two triggers:**

- **"run the review pass"** — the full loop below.
- **"process the dashboard"** — step 0 alone: read the user's inline answers
  on `View - Dashboard.md`, execute approved nominations (registry updates,
  routing, paste-lines), regenerate the Dashboard recording outcomes, and
  regenerate any views affected by new routing. Do NOT advance the
  high-water mark — no new blocks were processed.

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

**One rule: a note is a registered routing target iff it has a container
tag AND a filled `description` frontmatter.** Container tags:

- `#note-type/amoc` — areas (gold: current state of a never-ending area).
- `#note-type/pmoc` + `#status/active` — projects (gold).
- `#note-type/moc` — concept notes (silver: a facet of an area/project, an
  evolving idea; the permanent home for "I've said this before" captures).

Discover the registry per run by tag search — there is no hand-maintained
list. `OpenAugi/AGENT/My Taxonomy.md` defines the facet vocabulary the
containers map to.

**The `description` frontmatter is the routing map** — it tells you *when
to route here* (skill-file style: name + description). A note with the tag
but no description is NOT registered: don't route to it by inference
(explicit signals still work); instead nominate on the Dashboard with a
**drafted description as a paste-line** — filling the description IS
registration, so make saying yes cost one paste.

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
   route to the most *specific* matching container (a matching concept MOC
   beats an active PMOC beats its parent AMOC). Inference candidates are
   registered containers ONLY (tag + description); rules 1–3 may route to
   any note.
5. **Low confidence** → leave unrouted; it goes to the Dashboard's
   Unrouted/Gravity section. Never force-fit.

**Persistence — two separate mechanisms, never conflate them:**

- **Membership = links.** `route_block(block_id, container_title)` creates a
  `routed_to` edge in the DB. A block can route to multiple containers.
- **Classification = tags.** `tag_block(block_id, augi_tags)` with facets drawn
  ONLY from the user's taxonomy — it is a closed vocabulary; never invent a
  tag or a facet. If the user already tagged the block, do not re-tag; only
  fill gaps.

**Reference material routes as one document.** Blocks from synced external
sources (Snipd, Readwise, and similar reference imports) are one artifact:
route the parent document once (`route_block` on the document block) and let
the pieces ride along — never make per-block routing decisions over a
transcript. Reference files are synced from their source: never move, edit,
or restructure them; routing is a link only. Count reference documents
separately in the pass summary so they don't inflate the queue numbers.

**Untagged and unrouted is the default, not a failure.** Life-log blocks
(daily entries, memories) usually need no tag and no route — they stay
reachable by time and semantic search. Tag only what you'd query; route only
what a view should distill. Both live in the DB only — never write into the
user's notes.

**Edited blocks re-arrive as new — expected, not a bug (the re-derive
contract, 2026-07-09).** Block identity is a content hash, so when the user
edits a routed block, its routes drop and the edited version shows up in
your new-blocks queue. Just route it again like any new block — an `aaa:`
line in the text is the durable instruction and always wins. Do NOT treat a
familiar-looking "new" block as an error, and do NOT hand-restore old links.

## Views

One file per touched container in `OpenAugi/Views/`, named
`View - <container title>.md` (the prefix avoids Obsidian basename
collisions and makes provenance visible). Write with
`write_document(title, description, content, subfolder="Views", overwrite=True)`.
The description should state the question the view answers
(e.g. "Where I left off and what's next in OpenAugi").

**Every view has the same two parts — no per-container modes, nothing for
the user to configure:**

1. **Recap** — synthesized from ALL member blocks, *including whatever the
   user wrote in the container note itself*. The user's own head/journal are
   upstream inputs: the recap incorporates them and never contradicts them.
   If the evidence has drifted from the user's own head text, say so in one
   line ("your head says X; recent blocks suggest Y") — drift is
   information, not a correction to make silently.
2. **Log — remote blocks only.** List member blocks that physically live in
   OTHER files (dailies, inbox, mobile, random notes). NEVER list blocks
   whose source file IS the container note — they're already visible there,
   and the view is transcluded into that note; listing them would duplicate
   them on the same screen. The view *completes* the container (what arrived
   from elsewhere), it never mirrors it.

So the container note reads as one surface: the user's optional head/pins →
their in-place journal → the transcluded view (recap + remote feed).

If a container note was **renamed**, regenerate its view under the new
title and delete the stale `View - <old title>.md` — views are caches;
deleting them is always safe.

A view exists to be **embedded, never visited** — one transclusion
(`![[View - ...]]`) per container note. When YOU create a new container note
(promotion), include the transclusion line in it at birth. Suggest the user
add `OpenAugi/Views/` to Obsidian's Excluded Files so views only appear
embedded.

Recap emphasis by container kind (keep each section 3–5 high-signal
bullets, every claim linked to its source note):

- **Area (AMOC)** — rolling TLDR of the area · new-this-period highlights
  (linked) · task/idea rollup · links to active child PMOCs. No LEFT OFF.
- **Project (PMOC)** — TLDR · **LEFT OFF + next physical action** · open
  task list · new-this-period blocks.
- **Concept (MOC, silver)** — "current understanding" of the idea. The
  concept note is the permanent source of truth for that idea: on every
  regeneration, re-pull related blocks (semantic + links, not just this
  pass's arrivals) so old mentions keep merging in — resurfacing is
  repeated, never one-shot.

**Every view ends with a `## Log` section** — the container's routed blocks
(membership via `routed_to` links), newest first, one line per block:
`- YYYY-MM-DD — <one-line gist> ([[source note]])`. This materializes the
append-only log so the user can SEE membership in Obsidian (DB links are
otherwise invisible there). Small containers: full log. Big containers:
most recent ~30 with a total count line.

Footer line on every view:
`*Generated YYYY-MM-DD from N blocks since YYYY-MM-DD.*`

Always regenerate `View - Dashboard.md` (same folder):

- One line per area: what moved, what's next.
- One line per concept note (silver) that saw activity: "Positioning: +3
  this pass" — the gravity signal for where ideas are accumulating.
- Cross-area task rollup (union of the views' task lists).
- **Gravity section**: unrouted blocks that cluster together — nominate,
  one line each: "5 blocks over 3 weeks orbit *capture UX* — make it a note?"
  Take NO action on nominations. The user answers inline or via zzz.
  **When a promotion is approved (or the user says "make X canonical" via
  `aaa:`), adopt before create:**
  1. Search first — title, semantic, and tag search for an existing note
     that already is (or wants to be) the canonical home.
  2. If found: upgrade it — draft the container tag + description as a
     paste-line (that is registration; you never edit the user's note),
     then route the accumulated blocks to it.
  3. Only if nothing exists: create the concept note fresh (with the
     `![[View - ...]]` transclusion line at birth).
  4. Either way, sweep OLD blocks beyond the current window — "I've said
     this a few times" means the earlier sayings predate this pass;
     gathering them is the point (the resurfacing feature).
- **Nomination format (machine-readable — mobile review will read/write
  it):** every nomination, in the Gravity section or anywhere else on the
  Dashboard, is ONE markdown checkbox bullet ending in a stable Obsidian
  block anchor, with an empty answer slot nested under it:

  ```
  - [ ] **Promote:** 5 blocks over 3 weeks orbit *capture UX* — make it a note? ^nom-promote-capture-ux
      - answer:
  ```

  Anchor = `^nom-<verb>-<subject-slug>` — verb is the action asked
  (promote / describe / tag / merge / route), slug is kebab-case of the
  subject. Deterministic: the SAME nomination gets the SAME anchor on every
  pass, so unanswered nominations — and answers upserted by anchor from the
  phone — survive Dashboard regeneration. **Every carried-forward
  nomination shows its age** ("since YYYY-MM-DD") — a queue that only grows
  is a graveyard, and age makes that visible.

  **Decided = box checked OR answer filled** — two input surfaces, one
  signal: the checkbox is the Obsidian quick-tap, the answer slot is
  typed/mobile free text. A checked box with an empty answer is a plain
  "yes, as proposed." A filled answer (checked or not) is a specific
  instruction and takes precedence. **Unchecked + empty = still pending:**
  carry the nomination forward verbatim, anchor and checkbox included.
- Anything unroutable or confusing, listed honestly.
- Permanent footer: `*How this works: docs/reference/review-pass.md in the openaugi
  repo · design record: docs/plans/review-pass-v1.md · agent instructions:
  OpenAugi/AGENT/review-pass.md*` — the Dashboard is the discovery surface;
  this line keeps the docs findable without remembering them.

## The pass, step by step

0. **Process the user's Dashboard responses first.** Read the current
   `View - Dashboard.md` for answers to prior nominations — decided means
   checked (`- [x]`) or the `- answer:` slot is filled. Checked + empty
   answer = plain "yes, as proposed." Filled answer = specific instruction,
   takes precedence over the checkbox. Unchecked + empty = still pending —
   carry it forward verbatim, anchor and checkbox included; take no action.
   Also read any free-form inline notes or `aaa:` lines. Execute approved
   ones — update the registry, route the relevant blocks, draft paste-lines
   for anything that touches the user's notes — BEFORE regenerating
   anything, or the answers are lost to the overwrite. Record each outcome
   in the new Dashboard.
1. `get_review_state()` → `since` = last_run. If null, this is the first
   run: backfill from a sensible recent date (e.g. two weeks back, or the
   date the user gives).
2. Pull new blocks: `search(after=since, exclude_path_prefix="OpenAugi/")`
   (browse mode, paginate via offset) — the prefix filter keeps derived
   artifacts out of the queue server-side. Group reference-source blocks by
   their `source_path` and handle each reference document as one item. Use
   `recent`/`get_context`/`get_related` for extra context.
3. Decide routes for every block per the precedence above, then persist the
   whole batch with `apply_routing(decisions=[{block_id, containers,
   augi_tags}, ...])` — one call, not a route_block/tag_block loop.
4. Regenerate views for containers that received blocks (`overwrite=True`).
   Untouched containers keep their old view. **Two refresh tiers — don't
   re-derive what didn't change:**
   - **Log section: always refresh** (mechanical render of routed_to links —
     no LLM judgment, effectively free).
   - **Recap: refresh only when it would change** — salient new blocks
     arrived, drift appeared, or the user asked. A handful of life-log
     blocks routing through does NOT warrant re-synthesizing a recap; carry
     the old recap forward verbatim and only update the log. When in doubt,
     keep the old recap.
   **Regeneration is a merge, not a reset:** read the existing view first —
   it is the prior head state. Carry forward what's still true (the TLDR
   evolves; LEFT OFF advances or stands), integrate the new blocks, drop
   what's no longer salient. "New this period" covers only the current
   window. If deeper context is needed, pull the container's full membership
   via its `routed_to` links (`get_related` on the container, direction=in).
5. Regenerate `View - Dashboard.md`.
6. `write_context_pack()` — regenerates `OpenAugi/context-pack.json`, the
   sidecar the mobile app's tag/link suggestions are served from. One call,
   no arguments; the tool assembles it from the DB.
7. `mark_review_complete(summary)` — one line, e.g.
   "routed 42 blocks; regenerated 4 views + Dashboard; 2 nominations".
   Only call this after views are written successfully.

## Lens scheduling (NOT ACTIVE YET)

The lens registry (`OpenAugi/AGENT/lenses/`) declares per-lens triggers
(`on-pass`, `every <period>`). **Do not run them yet.** Scheduled lens
runs activate only after passes are boringly reliable; until then all
lenses are on-demand. When activated, this section will say: after the
Dashboard step, list the lens folder, run whatever is due, then continue.

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
