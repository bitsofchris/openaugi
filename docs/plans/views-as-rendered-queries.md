---
name: views-as-rendered-queries
description: Design record — replace materialized view files with live-rendered queries over the DB, and unify containment with membership. Dissolves the view-cache problem, the per-container surface question, and the "my paste vs. your route" equivalence confusion. Discussion draft, not yet approved.
---

# Views as Rendered Queries

**Status: design draft for discussion (2026-07-11). Nothing here is built.
Chris reacted to the direction positively in conversation; this doc is the
record to argue with. — Update, later 2026-07-11: several open questions
resolved with Chris in the mobile-side design session; see
[Resolutions](#resolutions-2026-07-11-with-chris-from-the-mobile-side-design-session)
at the bottom.**

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
(Chris had pasted them there himself). The route added a DB edge pointing
a block at the note it already lives in. Chris's reaction, verbatim:

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
interpretation, the works — for a note Chris curates entirely by hand.
He never asked for it; the spec said "registered container → view," so the
agent made one. The view was deleted; the incident stands as evidence that
**per-container surface preferences are real** and the one-size view
contract fights them.

**3. (Chris, unprompted, same conversation):**

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

The same root cause produces the equivalence confusion. Chris's paste and
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
- This makes Chris-pastes-it and agent-routes-it literally the same row in
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
- **Render** — happens where Chris is looking (surfaces below). The
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
   talks to). Chris's notes and the agent's synthesis in one screen — his
   stated ask — without the plugin writing anything into the note.
2. **Mobile app** (`private-augi-mobile`) — already reads
   `context-pack.json`; containers-with-views is the obvious next screen.
   The Dashboard (what moved, nominations with tap-to-answer) is the
   natural mobile home — the nomination anchors (`^nom-*`) were designed
   for exactly this round trip.
3. **Fallback: keep generating 2–3 markdown views** (Dashboard + the
   containers Chris actually opens in Obsidian today) during transition,
   from the same saved-query definitions — a render target, not the
   source of truth. Delete when surface 1 works.

### 4. What this dissolves (not solves)

- **The equivalence confusion** — gone at the query level (§1), gone at
  the visible level (one rendered surface shows both).
- **The per-container surface question** — there is no "does this
  container get a view file" decision anymore. Every registered container
  *has* a view (it's just a query); whether anyone looks at it is Chris
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
  decision** was made with Chris a day ago and this partially unwinds it.
  That decision was solving "how do silver notes stay visible" *given
  file-views*; under rendered queries the visibility comes from the pane.
  Needs an explicit re-decision with Chris, not a silent override.

## Open questions for the design session

1. **Is the plugin pane acceptable as THE surface?** (If Chris mostly
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
4. Cut file generation for containers Chris confirms he reads via pane;
   keep Dashboard as file until answer-UI (open question 3).
5. Converge lenses onto saved queries; revisit the silver-notes decision
   with Chris (trade-off #5).

*(Sequence amendment per the resolutions below: the first rendered-query
surface is the **mobile Dashboard**, not the plugin pane — step 3's surface
changes, the rest of the order holds.)*

---

## Resolutions (2026-07-11, with Chris, from the mobile-side design session)

Chris agreed to the following framing explicitly ("I agree with that…
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

A synthesized note stops being a cache the moment Chris would edit it,
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

Chris's demonstrated behavior is capture + triage from the phone (first
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
