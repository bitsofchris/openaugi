---
name: views-as-rendered-queries
description: Design record — replace materialized view files with live-rendered queries over the DB, and unify containment with membership. Dissolves the view-cache problem, the per-container surface question, and the "my paste vs. your route" equivalence confusion. ADOPTED 2026-07-11; implementation ledger at the bottom.
---

# Views as Rendered Queries

**Status: ADOPTED, in implementation (2026-07-11 — the user: "get to work on
this entire plan"). Open questions all resolved: 1/3/5 in the
[Resolutions](#resolutions-2026-07-11-with-the-user-from-the-mobile-side-design-session)
below, 2/4 in [Decisions on the remaining open questions](#decisions-on-the-remaining-open-questions-2026-07-11-implementation-session).
Shipped state lives in the [Implementation ledger](#implementation-ledger)
at the bottom — update it as steps land.**

**Supersedes (if adopted):** the Views section of
[review-pass-v1.md](review-pass-v1.md) (materialized `View - *.md` files,
transclusion contract, overwrite semantics) and the "silver =
view-on-touch" decision in
[review-pass-v2-workstreams.md](review-pass-v2-workstreams.md).
**Does not touch:** capture grammar, routing precedence, nominations, the
trust model, the taxonomy.

**Related:** the source-reference design task in the vault
(`OpenAugi/Tasks/TASK-2026-07-11-design-source-reference-notes.md`) — both
docs answer the same underlying question: *what should the pass
materialize, and where?* This doc's answer ("nothing — render on read")
likely simplifies that task's answer too.

---

## The two incidents that motivated this (pass #3, 2026-07-11)

**1. The redundant route.** The agent routed 3 dream blocks to
`MOC - Dream Journal` — but their source file already *was* that note
(the user had pasted them there himself). The route added a DB edge pointing
a block at the note it already lives in. The user's reaction, verbatim:

> "if the block is already on the note - it should automatically have the
> same membership if you 'routed' it for me? those should be equivalent -
> it's like either I wrote the block on the note directly or you mapped it
> to there for me"

He's right, and the spec already agrees (routing precedence #3: "home by
construction"). The implementation doesn't: containment and membership are
two different things in the DB, and agents can add redundant edges on top
of containment.

**2. The unwanted view.** Registering Dream Journal as a container caused
the pass to generate `View - MOC - Dream Journal.md` — recap,
interpretation, the works — for a note the user curates entirely by hand.
He never asked for it; the spec said "registered container → view," so the
agent made one. The view was deleted; the incident stands as evidence that
**per-container surface preferences are real** and the one-size view
contract fights them.

**3. (the user, unprompted, same conversation):**

> "this friction is because we have multiple surfaces - like obsidian is my
> capture and my viewer - if we just had something render views on the
> database like datadog does this wouldn't be a problem - i would see all
> my notes and yours in the same view"

That's the design.

---

## Diagnosis: view files are a cache for a renderer that can't query

Everything awkward about the current views layer is cache-invalidation
logic:

| Current mechanism | What it actually is |
|---|---|
| `write_document(..., overwrite=True)` restricted to `Views/` | cache write with a safety perimeter |
| "regenerate views for touched containers" | cache invalidation on write |
| "Log lists remote blocks only, never contained ones" | manual dedup between cache and source visible on the same screen |
| "renamed container → delete stale view file" | cache key migration |
| "views are disposable, deleting is always safe" | the definition of a cache |
| two-tier refresh (log always, recap only when it would change) | partial cache refresh to save LLM cost |

None of this is essential to the *product* ("show me the state of this
container"). It exists because **Obsidian can only display files**, so
query results must be frozen into files to be seen at all. Datadog is the
right counter-model: nobody exports a dashboard to a file on a cron; the
dashboard is a saved query rendered at look-time.

The same root cause produces the equivalence confusion. The user's paste and
the agent's route are the same intent ("this block belongs here") but
produce different visible states — paste is visible in the note and
invisible to the DB-as-membership; route is visible in the DB and invisible
in Obsidian until a view file renders it. Two surfaces, two half-truths.

## The design

### 1. Membership = containment ∪ routing (query-level fix, ship regardless)

Every block already gets a `contains` edge from its source document at
ingest. The fix is a **rule, not a migration**:

> A block is a member of container C iff it has a `routed_to` edge to C
> **or** its `contains` parent is C (and C is a registered container).

- `route_block(block, its-own-source-doc)` → returns `already home`
  (no-op, explicit status so agents learn).
- View/lens/membership queries use the unified rule.
- Multi-membership unchanged: containment gives one home for free,
  `routed_to` adds others.
- This makes the-user-pastes-it and agent-routes-it literally the same row in
  every query result — the equivalence he asked for.

This piece is independent of everything below and should ship first
(it's small: one query change + one tool-response change).

### 2. A view is a saved query, not a file

Define a view as: **membership query + recap synthesis + render template.**

- **Membership log** — mechanical: members of C (unified rule), newest
  first, grouped contained-vs-remote if useful. Computed at render time.
  Always fresh, no invalidation logic, no dedup rules — the renderer knows
  which blocks are contained because that's just a field.
- **Recap** — the LLM synthesis (TLDR, LEFT OFF, drift notes). This is the
  expensive part, so it IS cached — but as a DB row
  (`container_id, recap_md, generated_at, membership_hash`), not a vault
  file. The review pass (or an on-open trigger) refreshes it when
  membership changed materially; the render shows `recap as of <date>`
  with staleness visible instead of silently stale.
- **Render** — happens where the user is looking (surfaces below). The
  markdown file in `OpenAugi/Views/` stops existing.

Lenses converge with this for free: a lens is already "saved question →
derived artifact." Under this model a lens whose target was `view:` or
`dashboard` becomes a saved query + cached synthesis too; only
lens outputs that are genuinely *documents* (research notes) keep writing
files. The lens registry and the view registry become one kind of thing.

### 3. Render surfaces (in order of build reality)

1. **Obsidian plugin pane** (`openaugi-obsidian-plugin` exists) — open a
   registered container note → a side pane / bottom section renders its
   view live from the DB (local MCP/HTTP, same daemon the plugin already
   talks to). The user's notes and the agent's synthesis in one screen — his
   stated ask — without the plugin writing anything into the note.
2. **Mobile app** (`private-augi-mobile`) — already reads
   `context-pack.json`; containers-with-views is the obvious next screen.
   The Dashboard (what moved, nominations with tap-to-answer) is the
   natural mobile home — the nomination anchors (`^nom-*`) were designed
   for exactly this round trip.
3. **Fallback: keep generating 2–3 markdown views** (Dashboard + the
   containers the user actually opens in Obsidian today) during transition,
   from the same saved-query definitions — a render target, not the
   source of truth. Delete when surface 1 works.

### 4. What this dissolves (not solves)

- **The equivalence confusion** — gone at the query level (§1), gone at
  the visible level (one rendered surface shows both).
- **The per-container surface question** — there is no "does this
  container get a view file" decision anymore. Every registered container
  *has* a view (it's just a query); whether anyone looks at it is the user
  opening the pane or not. Dream Journal needs zero configuration: he
  never opens its pane, nothing is generated, nothing intrudes.
  (The registry keeps a `recap: on|off` bit at most — Dream Journal
  wants membership-log-only with no LLM interpretation.)
- **Most of the review pass's write phase** — the pass reduces to:
  process dashboard answers → route new blocks → refresh stale recaps →
  nominate. No view regeneration choreography, no
  overwrite-outside-Views hazard, no stale-view-after-rename cleanup.
- **The transclusion contract** — no `![[View - ...]]` lines to maintain,
  no "embedded never visited," no Excluded Files suggestion.

## Trade-offs (the honest list)

- **Needs a running process to see views.** Markdown views work with the
  vault alone — offline, no daemon, survives the project dying. Mitigation:
  the fallback renderer (§3.3) can materialize any saved query to markdown
  on demand ("export view"), so the escape hatch is one command, and the
  vault remains complete-enough without the daemon.
- **Wikilinks out of a rendered pane** must resolve into Obsidian —
  plugin-side work (the pane renders `[[...]]` as internal links). Known
  cost, bounded.
- **Recap freshness vs. LLM cost** doesn't disappear — it moves from
  "regenerate files on a schedule" to "refresh cache rows on membership
  change," which is the same cost with better observability (staleness is
  displayed, not hidden).
- **Loses greppability of views** — today `grep Views/` finds synthesis
  text. Mitigation: recaps are DB rows; `openaugi views --dump` covers it.
- **The v2 "silver = permanent visible source of truth via transclusion"
  decision** was made with the user a day ago and this partially unwinds it.
  That decision was solving "how do silver notes stay visible" *given
  file-views*; under rendered queries the visibility comes from the pane.
  Needs an explicit re-decision with the user, not a silent override.

## Open questions for the design session

1. **Is the plugin pane acceptable as THE surface?** (If the user mostly
   reads on mobile, build order flips: mobile Dashboard first.)
2. **Recap cache policy** — refresh on pass only (predictable cost) vs.
   on-open-if-stale (fresh but bursty)?
3. **Dashboard nominations round-trip** — today answers live in the
   Dashboard *file* and survive regeneration via anchors. If the Dashboard
   is rendered, answers become DB writes from the pane/mobile (cleaner!)
   but need UI. Interim: keep the Dashboard as the ONE remaining
   materialized file until answer-UI exists.
4. **What does `write_context_pack` become?** Probably absorbed: the
   context pack is itself a rendered query the mobile app fetches.
5. Does the source-reference "cards" idea (vault task above) become just
   another saved query ("cards for argument X") instead of a filing
   scheme? (I think yes — reinforces retrieval-over-filing.)

## Suggested sequence (if adopted)

1. **Ship §1 now** (containment-as-membership + `already home`) — small,
   independent, fixes the equivalence bug regardless of the rest.
2. Define saved-query schema + recap cache table; make the review pass
   write recap rows *in addition to* files (dual-write, no behavior
   change visible).
3. Plugin pane rendering membership log + cached recap (read-only v0).
4. Cut file generation for containers the user confirms he reads via pane;
   keep Dashboard as file until answer-UI (open question 3).
5. Converge lenses onto saved queries; revisit the silver-notes decision
   with the user (trade-off #5).

*(Sequence amendment per the resolutions below: the first rendered-query
surface is the **mobile Dashboard**, not the plugin pane — step 3's surface
changes, the rest of the order holds.)*

---

## Resolutions (2026-07-11, with the user, from the mobile-side design session)

The user agreed to the following framing explicitly ("I agree with that…
the clarity around silver and those layers makes sense"). These settle
trade-off #5 and open questions 1, 3, and 5.

### The layer model (adopt as shared vocabulary)

```
TRUTH   vault: captures (bronze) + notes, incl. synthesized ones kept (silver)
INDEX   augi DB: blocks, edges, membership, embeddings — rebuildable from truth
CACHE   recaps, views, dashboards, membership logs — rendered queries, disposable
RENDER  obsidian + plugin pane, mobile app — same API, own nothing
```

The test that separates the layers: **a cache is something you could
delete with zero grief.** Views, recaps, dashboards pass that test.

### Silver notes are NOT views (resolves trade-off #5 and question 5)

A synthesized note stops being a cache the moment the user would edit it,
link to it, or build on it — then it is **truth that happens to be
machine-drafted** (the silver layer of the medallion he already thinks
in). So:

- **Rendered by default, materialized on promotion.** A stitched
  "assemble my thinking on X" starts life as a query result in a pane.
  Keeping it is an explicit human command (`zzz: save this as a note`)
  that writes it into the vault as silver truth. Agent drafts, human
  decides what becomes real — the command-bus philosophy applied to
  synthesis.
- The earlier silver-notes-via-transclusion decision wasn't wrong, it
  answered a different question: silver notes were never views. Views
  dissolve into rendered queries; promoted silver notes stay materialized
  in the vault (curated, linkable, greppable, daemon-independent).
- This also answers question 5: source-reference "cards" are a saved
  query until one is worth keeping, then it promotes. Retrieval over
  filing, with an explicit keep step.

### Build order flips to mobile (resolves question 1)

The user's demonstrated behavior is capture + triage from the phone (first
real phone session 2026-07-11: nominations triaged from the couch). The
mobile app already speaks a typed wire contract to its bridge
(`private-augi-mobile` — `/views`, `/review-queue`, `/context-pack`), and
the bridge is the natural query API: today it answers by parsing
materialized files; under this design it answers from the DB, and the
phone doesn't change. **First rendered-query surface: the mobile
Dashboard. Plugin pane second.**

### Nomination round trip stays text for now (interim of question 3 confirmed)

Decisions keep traveling as capture blocks
(`Re: "…" (Dashboard ^nom-…) — approved.` + `zzz:` dispatch) — they
inherit the mobile outbox/offline/retry machinery for free and stay in
the truth log. Revisit DB-write answers only when the rendered Dashboard
with answer-UI actually exists. Until then the Dashboard remains the one
materialized file, as §3.3 / question 3 already suggested.

---

## Addenda from the vault-side session (2026-07-11, evening)

- **The layer model's canonical home is now
  [../reference/core-principles.md](../reference/core-principles.md)**
  (capture grammar · truth/index/cache/render · trust model · promotion).
  This doc's inline copy is the design-time snapshot; if they ever
  disagree, the reference doc wins.
- **Gap found processing pass #3 answers — nominations have no
  deferred state.** the user answered two cluster-weather nominations
  "no not now" / "leave it alone for now." Under the current contract a
  filled answer = decided = closed, but the cluster-weather lens will
  happily re-nominate the same clusters on its next run. There is no
  "declined — don't re-ask until X / until the cluster changes
  materially" state. Small fix, probably a `declined_at` +
  re-nominate-only-on-material-change rule wherever nomination state
  lands (DB row once the Dashboard renders; until then the pass prompt
  must carry declined anchors forward as suppressions). Filed here
  because nomination state is this design's question 3.

---

## Decisions on the remaining open questions (2026-07-11, implementation session)

The user delegated questions 2 and 4 ("you decide"); decided as follows:

- **Q2 — recap cache policy: refresh on pass only.** The pass is the only
  LLM-writing cadence today; staleness is displayed (`recap as of <date>`
  + membership-hash mismatch), so a stale recap is visible, not a lie.
  On-open-if-stale needs a daemon-side LLM trigger that doesn't exist and
  makes cost bursty. An explicit "refresh the recap for X" in any session
  remains available. Revisit only if displayed staleness annoys in
  practice.
- **Q4 — `write_context_pack`: absorbed, with a transition.** The context
  pack becomes a rendered query the mobile bridge serves from the DB
  (same shape the app already reads). The file export stays during the
  transition — same pattern as the Dashboard file — and is deleted when
  the bridge cutover (step 3) is verified on the phone.

## Implementation ledger

Update this table as steps land (commit hashes are in this repo unless
noted).

| Step | State | Shipped as |
|---|---|---|
| Prereq: `apply_routing` = single route CRUD tool (add/remove per decision, `route_block` deleted) | **shipped 2026-07-11** | `807da60` |
| Prereq: lens index folded into Dashboard (`View - Lenses.md` deleted) | **shipped 2026-07-11** | `5f93229` |
| §1 membership = containment ∪ routing: `already_home` no-op, containment-remove is an error, `get_members` unified query tool | **shipped 2026-07-11** | `f29a5b6` |
| §2 recap cache: `recaps` table + `write_recap`/`get_view` MCP tools, pass dual-writes recap rows | **shipped 2026-07-11** | `638d637` |
| §2b `list_views` (recap row = the render list / per-container view bit) | **shipped 2026-07-11** | `40cffa0` |
| Step 3 phase 1: mobile bridge `/views` renders from daemon queries (Dashboard stays the file; fallback to file parser) | **shipped 2026-07-11** | `private-augi-mobile` `5bfe323` |
| Step 3 phase 2: Dashboard-on-phone staleness UI, `/context-pack` absorption (Q4 cutover) | pending — needs phone verification of phase 1 first | — |
| Step 4: cut view-file generation per container as the user confirms rendered-surface usage; Dashboard stays a file until answer-UI | blocked on step 3 + usage | — |
| Step 5: converge lenses onto saved queries; view-target lenses stop writing files | blocked on step 3 — **the saved-query format + engine landed 2026-07-16** ([query-layer.md](query-layer.md): `OpenAugi/AGENT/queries/*.md`, QuerySpec in frontmatter, relative-date tokens); this step converges lenses onto THAT format | — |
