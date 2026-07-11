---
name: review-pass-v2-workstreams
description: Post-run-2 fixes merged into four workstreams — unified registry rule (tag + description), collect/surface split for silver notes, canonicalize verb (adopt-before-create), MCP tooling fixes, nomination hygiene.
---

# Review Pass V2 — Workstreams

## STATUS / LEFT OFF (update this section every session)

**As of 2026-07-11 (later):** WS3.1 SHIPPED (`source_path` in summaries +
`exclude_path_prefix` filter, all modes, tested). WS1 + WS2 prompt/doc
changes SHIPPED: unified registry rule + adopt-before-create + reference
handling reconciled across template, reference doc, and the live vault
prompt (v1 plan's registry section marked superseded). Silver semantics
clarified with Chris: view-on-touch, not on-demand — silver notes are
permanent visible sources of truth via the same transclusion pattern as
AMOCs. **NEXT:** run pass #3 — it drafts the 9 registry descriptions as
paste-lines and registers Dream Journal + mindfulness. Then WS3.2
(reference grouping in browse), WS3.3 (`apply_routing`), WS4
(weekly-reflection nomination triage).

**Inputs:** [review-pass-v1.md](review-pass-v1.md) (the frame — still
authoritative for CQRS/views/grammar), run #2 Opus feedback,
[context-block-architecture.md](context-block-architecture.md) (silver note =
canonical `context_block:document`).

---

## The diagnosis (why run 2 felt fuzzy)

1. **Three conflicting registry definitions.** Live vault prompt hard-codes a
   static list of 7; reference doc says "amoc/pmoc/moc tags"; v1 plan says
   "amoc + active pmoc." Result: real MOCs (Dream Journal, mindfulness) sit
   unregistered and their blocks dump into AMOC - Self.
2. **8 of 9 registry descriptions empty**, so inference routing (precedence
   rule 4) runs on vibes instead of a routing map.
3. **Collect and surface are coupled.** Every touched container regenerates a
   view, so routing targets are expensive and silver notes get resisted.
4. **Tooling friction**: browse mode blows token budget (79K/100 blocks), no
   `source_path` in summaries (can't apply the OpenAugi/ exclusion
   deterministically), reference material inflates the queue (182-block
   transcript = 182 routing decisions), no batch routing (~35 round-trips
   per pass).
5. **Nominations accumulate** (11 pending, some since 7/6) — good at
   surfacing decisions, bad at forcing them.

## The tier model (convention, never code)

Code and schema never learn the words bronze/silver/gold. Tiers are
capability bundles expressed through existing primitives. The loop they
serve: Chris captures → the pass puts each block in the right curated home —
either the area/project log (gold) or the specific concept note (silver).

- **Bronze** = data blocks. Chris's capture, raw thinking, reflections.
  Append-only truth. They route; they are never routing targets.
- **Reference** (not "imports") = external material — Snipd, Readwise, etc.
  Routed at **document granularity** (one artifact, one routing decision;
  blocks inherit). **Never moved or edited** — it is synced from the source.
  Never a routing target.
- **Silver = concept notes** — facets of projects and areas, evolving ideas.
  A canonical note that **collects**: container tag (`#note-type/moc`) +
  filled `description` ⇒ registered inference target. Once created, the
  silver note is the **permanent, visible source of truth** for that idea:
  Chris sees, refers to, and extends it; routing there again is the default
  ("I've said this before" → merge here). Mechanics are the AMOC pattern —
  human-owned note embeds `![[OpenAugi/Views/<concept>]]`; the view carries
  the current-understanding recap + links to all routed blocks, and each
  regeneration **re-pulls related silver/bronze blocks** (repeated
  resurfacing, not one-shot). **View-on-touch cadence:** regenerated only in
  passes that routed new blocks to it — untouched silver notes cost
  nothing. (In the data model this is just an ordinary
  `context_block:document` — no new kind.)
- **Gold** = AMOCs + active PMOCs — current state of the world, evolving,
  never-ending areas/projects. Everything silver does **plus surfaces**:
  view regenerated every pass, transclusion dashboard, links out to many
  silver and bronze blocks. Small, stable list.

WS1 and WS2 split as nouns vs verbs: WS1 defines what the tiers *are*
(registry rule, view cadence); WS2 defines the one behavior that brings a
silver note into existence (adopt-before-create + resurface).

**One registry rule replaces all three definitions:**

> A note is registered iff it has a container tag AND a filled description.
> Views regenerate **on touch** (any pass that routes new blocks to the
> container) for every tier — this was already the template's rule, and it
> extends to silver unchanged. The Dashboard regenerates every pass and is
> where gold's always-current rollup lives; in practice gold containers are
> touched nearly every pass anyway.

Corollaries:
- Filling a description IS registering. Promotion = add tag + one description
  line (doable via `aaa:`, no nomination round-trip).
- The "describe-registry-notes" and "promote-MOC" nomination *types* cease to
  exist.
- Registry remains the candidate set for inference routing only; rules 1–3
  (`aaa:`, explicit link, location) can still route to any note.

**Silver visibility (revised 2026-07-11):** silver notes are NOT invisible
accumulation — each has its transcluded view showing current state, always
up to date because it refreshes on touch. Cheapness comes from untouched
notes costing nothing, not from hiding. Gold differs only in: refresh every
pass regardless of touch, and presence in the Dashboard rollup. Dashboard
still lists per-silver-note activity ("Positioning: +3 this pass") as the
gravity signal.

---

## WS1 — Registry unification (prompt/doc changes, no code)

- [x] Reconcile the registry rule in all three places: live vault prompt
      (`OpenAugi/AGENT/review-pass.md` + repo template mirror),
      `docs/reference/` doc, and review-pass-v1 plan (marked superseded).
      Hard-coded list demoted to seed context under the rule. (2026-07-11)
- [x] View cadence: on touch for all tiers (template's existing rule,
      extended to silver); Dashboard every pass. Silver notes embed their
      view by transclusion exactly like AMOCs; agent writes stay confined
      to `OpenAugi/Views/` (adopted human notes are never edited).
- [x] Silver view regeneration re-pulls related blocks (semantic + links),
      not just this pass's routed blocks — in the MOC recap spec.
- [ ] Dashboard: per-silver-note block-count lines; add **age** to every
      pending nomination.
- [ ] Next pass: draft all 9 registry descriptions as paste-lines for Chris
      (answers `^nom-describe-registry-notes`, riding since 7/6).
- [ ] Register Dream Journal + mindfulness by tag + description (kills both
      promote nominations).

## WS2 — Canonicalize verb: adopt-before-create (prompt changes, no code)

Hard rule in the promotion step, generalizing Chris's positioning answer
("I should already have a note for this — roll these into it"):

1. **Search first** — title, semantic, and tag search for an existing note
   that already is (or wants to be) the canonical home.
2. **If found: upgrade it** — add container tag + description (that is
   registration), route the accumulated blocks to it.
3. **Only if nothing exists: create** the silver note fresh.
4. **Resurface** — sweep old blocks beyond the current window ("I've said
   this a few times" means earlier sayings predate this pass). This is the
   v1 spec's "resurfacing feature," now enforced at promotion time.

Trigger forms: approved promote-nomination, or `aaa: make X canonical`.
Uses existing tools only (`search`, `route_block`, `write_document`).
**Shipped 2026-07-11** in the template, reference doc, and live prompt
(Gravity-section promotion steps).

## WS3 — MCP tooling fixes (code, openaugi repo)

Order matters; each is a small, separately committed change with tests.

- [x] **3.1 `source_path` in `_block_summary`** + `exclude_path_prefix`
      filter — SQL-level in browse mode (LIKE-escaped), Python-filtered in
      title/keyword/semantic modes. Shipped + tested 2026-07-11.
- [x] **3.2 Reference grouping.** Browse mode collapses blocks carrying a
      `source/*` tag into `reference_documents` (one entry per source doc:
      document_id, block_count, time range, source_tags); the pass routes
      the *document* once via its document_id. Reference documents are
      never moved or edited (synced from source); routing is a link only.
      Shipped + tested 2026-07-11.
- [x] **3.3 `apply_routing(decisions=[{block_id, containers, augi_tags}])`**
      — batched write tool; per-decision errors don't block the rest.
      Prompts updated to prefer it over route_block/tag_block loops.
      Shipped + tested 2026-07-11.
- [ ] **3.4 (maybe) compact browse summaries** — trim content snippet
      500→200 chars in browse mode. Likely subsumed by 3.2; do only if the
      budget still hurts after reference grouping.

## WS4 — Process / nomination hygiene (skill changes)

- [ ] Weekly-reflection skill: add "nomination triage — drive queue to zero"
      step (accept, reject, or explicitly park each; parked ≠ pending).
- [ ] Note: WS1 deletes the two heaviest nomination types at the source.

## Sequencing

1. **Before pass #3:** WS3.1 (one-liner), WS1 registry-rule reconcile.
2. **Pass #3 itself:** drafts the 9 descriptions; Chris pastes → registry
   fully described; register Dream Journal + mindfulness.
3. **Then:** WS3.2 + WS3.3, WS2 prompt text, WS4.
4. WS3.4 only if still needed.

## Non-goals (explicitly parked, per Chris)

- No tier fields, enums, or bronze/silver/gold vocabulary in code or schema.
- No "which note types are excluded from search" mechanism yet — everything
  stays searchable; only the review queue narrows (WS3.1/3.2).
- No auto-promotion, no tuned thresholds (v1 frame unchanged).
