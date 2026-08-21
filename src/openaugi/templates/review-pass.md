---
name: review-pass (template)
description: >
  TEMPLATE — copied to <vault>/OpenAugi/AGENT/review-pass.md on `openaugi init`.
  The vault copy is the live version the agent reads. Edit there, not here.
  The recurring review/maintenance pass: route new blocks to containers
  (AMOCs/PMOCs), refresh their recap rows (write_recap — per-container view
  FILES were retired 2026-08-17), surface promotion nominations on the
  Dashboard. Run manually or via zzz ("run the review pass").
---

# Review Pass

**Two triggers:**

- **"run the review pass"** — the full loop below.
- **"process the dashboard"** — step 0 alone: read the user's inline answers
  on `View - Dashboard.md`, execute approved nominations (registry updates,
  routing, paste-lines), regenerate the Dashboard recording outcomes, and
  refresh the recap of any container affected by new routing. Do NOT advance
  the high-water mark — no new blocks were processed.

You are running the OpenAugi review pass. One loop:

> new blocks → route (tag in DB) → refresh recaps → nominate structure changes → advance the high-water mark

Read [[My Taxonomy]] (OpenAugi/AGENT/My Taxonomy.md) first — it defines the
facets and the container registry.

## The frame (do not violate)

- **Truth** = the user's own writing. NEVER edit any note outside `OpenAugi/`.
  Blocks are append-only; nothing is ever deleted.
- **Recaps** = the synthesis you write per container with `write_recap`.
  Derived, regenerable, disposable — rewrite them freely, no review needed.
  They are DB rows, not files: the mobile explorer renders them via
  `get_view`. (Per-container `View - *.md` files were retired 2026-08-17.)
- **Structure changes** (new tag/area, new silver/gold note, merging notes)
  are NEVER done autonomously. You nominate on the Dashboard; the user
  commands; only then do you assemble. **An approved nomination IS the
  command:** apply the exact edit it drafted (registration tag +
  description frontmatter, an embed/link line) directly to the target
  note — nothing beyond what was drafted — and record the edit in the
  Dashboard's Processed section. Approval-executed edits are the one
  exception to "never edit outside OpenAugi/".

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
**drafted tag + description included verbatim** — filling the description
IS registration. When the user approves, apply the drafted frontmatter to
the note yourself (approval is the command); saying yes costs one checkbox.

**Registering writes three things, not two: the tag, the `description`, and
an `augi_id`** — an opaque unique token (a UUID is fine):

```yaml
---
augi_id: 898c3199-596c-4154-8be2-b7b862ec12d7
description: …
---
```

A document's graph id derives from its `augi_id` when present and from its
file path otherwise — and every edge into a container (`contains` and
`routed_to` alike) is keyed on that id. Without one, renaming or moving the
note orphans every edge into it, silently. Never change, remove, or reuse an
`augi_id`; a registered container missing one gets it added as a repair, no
approval needed. (Adding one to an already-ingested note re-keys it — run
`scripts/migrate_container_augi_id.py` to carry the edges across.)

**Registry restraint — registration is for INFERENCE targets only.**
Register a note when captures from *elsewhere* (dailies, mobile, random
notes) should land in it automatically. A contained-and-clear note the
user writes in directly — or routes to explicitly when needed — gains
nothing from registration; nominating it just bloats the registry and
the user's yes/no queue. Unregistered ≠ unroutable: `aaa:` and explicit
links always work on any note. Precedents (both the user's calls):
Dream Journal registered as routing-target-only because dreams captured
in OTHER files must find it; Wind Turbine PMOC registration declined
2026-07-12 — "contained and clear, I route there directly on my own;
don't bloat the registry." Before nominating a registration, ask: does
anything actually arrive here from elsewhere by inference?

## Routing — you execute rules, you do not exercise judgment

**This is the line the whole pass sits on.**

> Do not guess. Most blocks do not need to be routed out of the daily note.
> A block with a link in it, a block carrying a rule, or a block asking for a
> new note — that is the whole job.

Routing is **autonomous** — you apply it without asking — and that is only
acceptable because it is never a judgment call. You are executing instructions
the user wrote. So the boundary has to be exact.

### The three rules. There is no fourth.

Route a block if and **only if** one of these matches. Record which one fired
with a record in the `routings` collection (contract below):

| `rule` | Fires on |
|---|---|
| `instruction` | an `aaa:` in the block naming a container — obey it, including to unregistered notes |
| `link` | a `[[wikilink]]` in the block pointing at a **registered** container |
| `tag` | a tag or facet on the block matching a **registered** container's registration |

**Location is not a rule, it is containment.** A block written inside a
container's own journal is already a member by construction — `apply_routing`
returns it in `already_home`. Do not route it there. Cross-links to *other*
containers still follow the rules above.

### What you do with everything else: nothing

**No semantic inference. No "this feels like it belongs in health".** If none
of the three rules fired, the block **stays in the daily note** — and this is
the expected outcome for most blocks, not a failure to classify.

Count them as `left_alone` on the pass record. **They are not a backlog, not an
unrouted queue, and not something to raise on a later pass.** A life-log block
about flag football does not need a home. The old rule 4 ("infer from
taxonomy, route to the most specific match") and rule 5 ("low confidence →
Unrouted/Gravity") are both **removed** — they produced a queue in which five
real decisions sat under nineteen items of noise.

If a block genuinely needs judgment — it looks like it wants a note of its own,
or several blocks are circling one idea — that is not routing. Use
a `proposals` record (below), which asks instead of acting.

### The record contract — collections this pass owns

openaugi has **no idea** what these mean. It stores records in named
collections (`docs/reference/records.md`); the shape below is *this prompt's*
contract, and the list of legitimate `rule` values is *this vault's* policy.
A different vault would use different collections and different rules.

```
collection "passes"     id: pass-<YYYY-MM-DD-HHMM>   ← the user's LOCAL time
  { window_from, window_to, scanned, left_alone, reviewed_at? }

collection "routings"   id: <pass_id>:<block_id>:<container>
  { pass_id, block_id, container, rule, undone_at? }

collection "proposals"  id: <kind>-<subject-slug>
  { pass_id, kind, block_ids, target, payload, why, state,
    evidence: [{id, text, rule?}] }
```

Three conventions that matter:

- **The pass id is the user's local time, to the minute — never the UTC
  date.** A pass run late in the evening otherwise names itself with
  tomorrow's date, and two passes in one day collide on a date-only id, the
  second silently upserting over the first.
- **Ids are derived from the subject, never random.** Writes upsert, so
  `promote-silver-notes` re-proposed next pass updates in place instead of
  stacking. Random ids turn the queue into a pile — that is exactly how the
  old Dashboard reached 24 open items.
- **Do not store counts.** `routed` and `proposed` are `list_records` counts.
  Storing a number you can count is how a number goes wrong.
- **`rule` must be one of `instruction`, `link`, `tag`.** openaugi will
  happily store anything; the constraint is yours, and it is the whole point
  of the routing section above. If you cannot name one of those three, do not
  route the block.

### The pass's routing loop

```
write_record("passes", "pass-2026-08-20-1225",   # local time, to the minute
             {window_from, window_to, scanned, left_alone})

for each block in the batch:
    rule = first of (instruction, link, tag) that matches — or None
    if rule is None:  continue          # stays in the daily note. Done.
    apply_routing(decisions=[{block_id, add:[container]}])
    write_record("routings", f"{pass_id}:{block_id}:{container}",
                 {pass_id, block_id, container, rule})
```

**If you find yourself wanting to write `rule: "similar"` or `"inferred"`,
that is the design telling you to leave the block alone.** openaugi will
store it — it has no opinion — which is exactly why the discipline has to be
here.

**Persistence — two separate mechanisms, never conflate them:**

- **Membership = containment ∪ routing.** A block is a member of a container
  if it physically lives there OR has a `routed_to` edge — the same fact seen
  from two sides. `apply_routing` creates and removes the edges;
  `get_members(container_title)` returns the unified list.
- **Classification = tags.** `tag_block(block_id, augi_tags)` with facets drawn
  ONLY from the user's taxonomy — a closed vocabulary; never invent a tag or a
  facet. If the user already tagged the block, do not re-tag; only fill gaps.

**Reference material routes as one document.** Blocks from synced external
sources (Snipd, Readwise, and similar) are one artifact: route the parent
document once and let the pieces ride along — never make per-block routing
decisions over a transcript. Never move, edit, or restructure reference files;
routing is a link only. Count reference documents separately so they don't
inflate the numbers.

## Proposals — the judgment half

Anything that is **not** one of the three rules goes through
`write_record("proposals", ...)` and is not done until the user accepts it.
Four kinds:

| `kind` | Means | `target` |
|---|---|---|
| `promote` | these blocks should become a new note | the proposed title |
| `adopt` | these blocks should append to an existing note | that note |
| `merge` | two notes should become one | the survivor |
| `register` | apply a drafted `description:` so a note becomes a routing target | the note |

Rules for proposing well, learned from the markdown nomination queue this
replaces — where 24 open items contained **five** actual decisions:

- **Never propose a routing.** A rule fired or it didn't.
- **Never propose engineering work.** Bugs in the tooling belong in its issue
  tracker, not in someone's knowledge review. A queue that mixes "should this
  become a note?" with "fix this SQL filter" gets abandoned, and the decisions
  that mattered are what get lost.
- **Derive the record id from the target** (`promote-silver-notes`) so
  re-proposing updates in place instead of stacking.
- **`why` is evidence, not justification.** Name the blocks or the pattern —
  *"5 blocks since January restate this"* — so the user can check you.
- **A decline is durable.** Re-propose only on genuinely new evidence, never
  because another pass ran.
- **Every proposal carries `evidence`** — the blocks themselves, as
  `[{id, text, rule?}]` with the text trimmed to a line or two — so the
  review surface can show the user's own words under the ask instead of a
  count. `rule` (only on a `register`) names which routing rule would fire
  once the registration lands.
- **A register must earn its place in the queue.** Its `evidence` is the
  blocks in *this* window the registration would catch — the reason to say
  yes. Propose it only when that list is non-empty; otherwise keep the
  drafted description in your own notes and stay quiet until a block reaches
  for it. Carrying a proposal forward is a claim that the reason still
  holds — re-check it each pass, and drop any register whose evidence is
  empty this window. (Upserting ids make carry-forward silent, which is how
  a queue of eleven asks with two real decisions builds itself.)
- **Never propose an adopt into a piece that has shipped.** A published note
  is a source the new note links to, not a container to append to. If the
  vault has no published marker, treat drafts as possibly shipped and prefer
  a linking `promote`.
- **Before proposing a `promote`, check the note does not already exist —
  outside the batch.** The batch excludes agent output (`OpenAugi/`), which
  means every note the agent has ever written is invisible to it. Two
  queries per proposed promote, exclude left off:

  ```
  search(title=<the title you are about to propose>)
  search(query=<the cluster's claim in one line>, k=5)
  ```

  If either finds a note that already collects these blocks, do not propose.

### Emergence — the second read over the left-alone blocks

Leaving a block alone is the right call for routing. It is not the end of the
pass. After the routing loop, read the left-alone blocks once more — for
content this time — and ask what is forming. Everything this produces is a
proposal; nothing is applied.

A `[[link]]` in a left-alone block points at an *unregistered* note by
definition — a registered one would have routed. Never leave one uncounted:

```
if a block asks for a note (aaa:, "make this a note"):
    propose promote            # one block is enough; an instruction is not a vote

for each cluster of left-alone blocks circling one idea:
    shared = a [[link]] the cluster's blocks have in common
    if shared and that note EXISTS:        propose register(shared)
    elif shared and that note DOES NOT:    propose promote(shared)   # the orphan link IS the title
    else:                                  propose promote(<drafted title>)
```

The middle-branch distinction is the one that gets missed: a repeated link to
a note that already exists means the user is *appending* to a thing they
have, not starting a new idea — registering it is what lets the `link` rule
fire next pass. And the `else` branch is not a licence to cluster loosely:
if you cannot quote the sentence the blocks have in common, there is no
cluster. There is NO temporal filter — an instruction makes a note today,
alone.

## Views

**Retired as files, 2026-08-17. Do NOT write `View - <container>.md`
any more.** The recap row IS the view: write it with
`write_recap(container_title, recap_md)` and stop. The mobile explorer
renders it from the DB via `get_view` (a collapsed recap card at the top of
container mode), with `stale` shown honestly — which is what the file was a
stand-in for while no renderer existed. The 15 per-container files were
deleted and their `![[View - …]]` embed lines removed from the container
notes the same day.

**Membership no longer needs a rendered log either.** The whole point of the
`## Log` section was that Obsidian can't show a `routed_to` edge; the app
can, and shows contained and routed blocks as one indistinguishable feed
(an MOC *is* a context block, routing is assignment to it). Spend the effort
on the recap, not on relisting members.

**Three files survive in `OpenAugi/Views/` and are still written:**

- `View - Dashboard.md` — **not a container recap** (it has no recap row),
  and `server/reviewQueue.ts` in the mobile bridge parses this exact path to
  build the phone's review lane. Keep writing it.
- `View - Morning Briefing.md`, `View - Open Loops.md` — lens outputs, not
  recaps. Untouched by this change.

Historical note on what the files used to contain (recap + log, per-kind
emphasis) is below; the *recap* guidance still applies verbatim — it is now
the content of the `write_recap` row.

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

If a container note was **renamed**, write its recap under the new title;
the old recap row is orphaned and harmless.

**Do not add `![[View - ...]]` transclusion lines to container notes** — the
files they point at no longer exist. A newly promoted container gets its
`description:` frontmatter and nothing else; its recap is read through the
app, not from a file.

### What goes in a recap

**One rule: a recap contains only what scrolling can't give you.** Full spec:
`docs/reference/recap-spec.md`.

This is new, and it inverts the old guidance. The recap used to carry a TLDR,
new-this-period highlights, and a member log — correct when it lived in a
file, because Obsidian cannot render a `routed_to` edge and the list was the
only way to see membership. The app renders the container's feed live, one tap
from the recap card. So what-moved and member lists are now a worse, staler
copy of what sits directly underneath them, and every line spent on them is a
line not spent on something only synthesis can produce.

Four sections. **Any may be empty; an empty section is omitted, never
padded** — a short recap is a true statement that the container is quiet.

1. **Current understanding** (3–5 sentences) — the through-line: what this
   container is *about* now, as against when it was created. The only
   always-present section, and it should change slowly. For a concept note
   (silver) this is the canonical statement of the idea, re-derived from the
   container's whole membership on every regeneration — semantic + links, not
   just this pass's arrivals, so old mentions keep merging in.
2. **Patterns across time** — what's visible only from above the window.
   *"The fourth time since March you've described the same deduplication idea,
   each time from a different angle."* **Requires an explicit span** ("since
   March", "across five months"); a pattern inside one window is not a
   pattern, it's the feed.
3. **Contradictions and open questions** — the highest-value section, because
   it needs two distant blocks held in mind at once, which scrolling never
   does. Decisions reversed without the original reasoning being addressed;
   questions asked in a block and never returned to.
4. **What's gone quiet** — absence leaves no trace in a feed. Name threads
   that were active and aren't. Observe, don't nag: some threads are finished.

**Do NOT include:** what moved this period · a `## Log` or member list ·
task rollups (the Dashboard's 14-day shelf already is that query) · counts of
new blocks · a link on every bullet.

**The link rule changed, it was not abolished.** Navigation is no longer a
reason to link — the app drills into any block. *Evidence* still is:

- **Sections 2 and 3 MUST link every claim.** They are factual assertions
  about specific blocks, and an unfalsifiable synthesis is exactly what the
  trust model exists to prevent.
- **Sections 1 and 4 need no links** — they characterize a whole, not
  particular blocks.

Fewer links, each one load-bearing.

**Aim for under 250 words.** This recap is synthesis-shaped, not list-shaped,
so it does not grow with the container. Running long almost always means
section 2 has drifted back into what-moved.

**No `## Log` section any more** (retired 2026-08-17) — it existed only
because Obsidian can't render a `routed_to` edge. The app can, so membership
is served live by `get_members` instead of being copied into a file.

Always regenerate `View - Dashboard.md` (same folder):

- One line per area: what moved, what's next.
- One line per concept note (silver) that saw activity: "Positioning: +3
  this pass" — the gravity signal for where ideas are accumulating.
- **Recent tasks (14-day shelf, rendered query).** Populate by running
  `search(has_task=True, after=<14 days ago>, exclude_path_prefix="OpenAugi/")`
  and rendering the results **verbatim** — one plain bullet per block:
  `- <first task line or gist> — [[source note]] (M/D)`. No checkboxes, no
  additions, no carrying forward: the query is the section. A task leaves
  the shelf by aging out or by its checkbox being completed in the source
  note (re-ingest drops it). Never invent a task the user didn't mark;
  recurring concerns earn renewal only by being captured again. Tasks that
  need real management belong in a PMOC's LEFT OFF, not here.
- **Gravity → a `proposals` record, not a Dashboard bullet.** Blocks that cluster
  around one idea are the *only* thing gravity produces now, and it is a
  `promote` (or `adopt`) proposal, never a routing. One per idea:
  `write_record("proposals", "promote-capture-ux", {kind: "promote",
  block_ids: [...], target: "Capture UX", state: "proposed",
  why: "5 blocks over 3 weeks restate this"})`.
  Take NO action on it — the user accepts it in the app.

  **Adopt before create, always:**
  1. Search first — title, semantic, and tag search for an existing note that
     already is (or wants to be) the canonical home.
  2. If found, propose `adopt` against it rather than `promote` a rival.
     Registration (the drafted tag + `description:`) is a separate `register`
     proposal.
  3. Only if nothing exists, propose `promote`.
  4. Either way, **sweep OLD blocks beyond the current window** — "I've said
     this a few times" means the earlier sayings predate this pass, and
     gathering them is the entire point.

  On acceptance, execute via `write_document` + `apply_routing` — the same
  path the mobile app's `POST /promote` takes. There is one promote path;
  do not hand-roll a second.
- **Nomination format — now a RENDERING of `list_records("proposals")`, not a store.**
  The proposals table is the source of truth; the Dashboard mirrors open
  proposals so they stay answerable in Obsidian on a laptop. Answers given
  there are still honoured, but the app writes straight to the table.
  Render one markdown checkbox bullet per open proposal, ending in a stable
  Obsidian block anchor matching the proposal id, with an empty answer slot:

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

  There is no "park" state and no Parked section — a nomination is
  answered (yes/no/instruction) or it pends with its age showing. An
  unanswered nomination costs nothing; a "not now" is just a slow no.
- **No honesty/appendix section.** Anything unroutable, confusing, or
  anomalous is said in ONE line inside the report prose (usually "What
  moved"), right where it's relevant — the Dashboard has no dumping
  ground. Honesty is a property of the report, not a section of it.
- Permanent footer: `*How this works: docs/reference/review-pass.md in the openaugi
  repo · design record: docs/plans/review-pass-v1.md · agent instructions:
  OpenAugi/AGENT/review-pass.md*` — the Dashboard is the discovery surface;
  this line keeps the docs findable without remembering them.

## The pass, step by step

0. **Process the user's answers first — the proposal records are canonical.**
   Answers can arrive two ways, and the record is the truth in both:

   - **Record states** (an app or API answered):
     `list_records("proposals", where={"state": "accepted"})` — each is a
     command. Apply exactly what it drafted (registration tag +
     `description` + `augi_id`, embed/link lines, the note) and nothing
     beyond it, then close it: `state: "done"`, `executed: "…"`.
   - **Dashboard checkboxes** (the user answers in the editor): for records
     still `proposed`, read `View - Dashboard.md` — checked (`- [x]`) with an
     empty `answer:` is "yes, as proposed"; a filled `answer:` is a specific
     instruction that beats the checkbox. **Write the outcome back to the
     record**, then execute as above.

   Reconcile before regenerating: a Dashboard bullet whose record is already
   answered is dropped, never re-asked — two records of one decision that
   can disagree is worse than either alone. Execute BEFORE regenerating
   views, or the answers are lost to the overwrite. A `declined` record is
   durable; never re-raise it without new evidence.

0b. **Answer the proposals the user sent back** —
   `list_records("proposals", where={"state": "discuss"})`. Each carries an
   `instruction`: what the user said to do instead, in their words. Treat it
   exactly like an `aaa:`. Do what it says — a change to the proposal means
   re-proposing with the change applied and the instruction kept on the
   record; a request for work first ("check what else relates") means doing
   that work, then re-proposing with what you found; "nothing to do" means
   `state: "done"` with `why` saying so. **Never leave one in `discuss`
   after a pass** — the review surface promises the next pass reads these.
1. `get_review_state()` → `since` = last_run. If null, this is the first
   run: backfill from a sensible recent date (e.g. two weeks back, or the
   date the user gives).
2. **Pull the new-blocks batch — TWO queries, read by every step below.**

   ```
   search(after_ingested=since, exclude_path_prefix="OpenAugi/")           # the vault
   search(after_ingested=since, include_path_prefix="OpenAugi/Capture/")   # the phone
   ```

   (browse mode, paginate each via offset; concatenate into one batch.)

   The first drops everything under `OpenAugi/` — agent-generated output.
   The second reaches back in for the one folder that isn't:
   **`OpenAugi/Capture/` is the mobile capture stream** — voice notes,
   `aaa:` instructions, and answers given from the phone. It is truth, and it
   routes like any other capture.

   Two queries rather than a list of excluded folders, deliberately: a new
   generated folder is then excluded automatically instead of leaking into
   the queue unnoticed. Skipping query 2 is a silent failure worth naming:
   the phone stream simply does not appear, so instructions captured there —
   including answers to the pass's own proposals — are never seen, and the
   pass looks healthy while ignoring them.

   `after_ingested` filters on when a block
   entered the DB; do NOT use `after=` here — it compares content dates
   (often date-only, and unchanged by edits), so it misses same-day
   captures and re-ingested edited blocks. This is the full pass scope;
   steps 3 and 4 both read from this one batch — don't re-query. Group
   reference-source blocks by their `source_path` and handle each
   reference document as one item. Use `recent`/`get_context`/`get_related`
   for extra context.

   **Partition the batch deterministically, before deciding anything:**
   - **Home blocks** — any block whose `source_path` matches a registered
     container note's own file (check against the registry from the
     Container registry section above). These are members by
     construction — `apply_routing` would return them in `already_home` if
     you tried. No routing decision needed for these.
   - **Routable blocks** — everything else. These get a routing decision
     per the precedence rules below.
3. Decide routes for every **routable** block per the precedence above,
   then persist the whole batch with `apply_routing(decisions=[{block_id,
   add, remove, augi_tags}, ...])` — one call, not a per-block loop.
   `remove` un-routes: when the user says a block was routed wrong, fix it
   with one decision carrying both `add` (right container) and `remove`
   (wrong one).
3b. **Second read over the left-alone blocks — emergence.** Everything step
   3 left in the daily notes gets read again, for content this time, per the
   Emergence section above. Resolve every `[[link]]` in a left-alone block
   (exists → `register` proposal; missing → `promote` under that title),
   cluster what remains, and write each proposal with its `evidence`.
   Re-check anything you are carrying forward and drop registers whose
   evidence is empty this window. Nothing here is applied — this step only
   writes proposal records.
4. **Compute touched containers, then refresh their recaps
   (`write_recap` — no files).**
   `touched = {containers that appear as an "add" target in step 3's
   apply_routing decisions} ∪ {registered containers whose own note
   appears as a source_path among step 2's home blocks}`.
   A container touched only by containment — nothing routed to it, but a
   new block landed straight in its own journal — still needs its recap
   checked. Home-block content is often *higher* signal than routed
   life-log (LEFT OFF edits, definition-of-done, direct season-log
   entries), not lower — don't skip a container just because nothing was
   routed there this pass. Untouched containers keep their old view.
   **Two refresh tiers — don't re-derive what didn't change:**
   - **Membership: nothing to refresh** — the app queries `get_members` live,
     so there is no rendered log to keep current.
   - **Recap: refresh only when the UNDERSTANDING would change** — a
     decision reversed, a thread going quiet, a question answered, a
     genuinely new angle on the idea, or the user asked. Blocks merely
     arriving is almost never enough: the recap no longer reports what
     moved, so what moved cannot make it stale (see the recap spec above).
     Carry the old recap forward verbatim. When in doubt, keep it —
     `get_view` reports staleness honestly, so a three-week-old recap that
     is still correct is a feature, not a debt.
   **Regeneration is a merge, not a reset:** read the existing view first —
   it is the prior head state. Carry forward what's still true (the TLDR
   evolves; LEFT OFF advances or stands), integrate the new blocks, drop
   what's no longer salient. "New this period" covers only the current
   window. If deeper context is needed, pull the container's full membership
   with `get_members(container_title)` (contained + routed, unified).
   **The recap row is the only copy now** — call
   `write_recap(container_title, recap_md)`. There is no second write and no
   file to keep in sync; the dual-write problem
   (`^nom-fix-recap-dual-write`) is dissolved rather than fixed.
5. Regenerate `View - Dashboard.md`. Open proposals may be mirrored there as
   nominations for editor answering (step 0 reconciles them back to the
   records), but the records stay canonical — never a bullet that exists
   only on the Dashboard.
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

- Never modify notes outside `OpenAugi/` — except to apply the exact edit
  an approved nomination drafted (approval is the command; apply nothing
  beyond the draft). Never use `overwrite=True` outside `Views/`, and inside
  `Views/` only `View - Dashboard.md` is still written.
- Never invent new `area/*` or `type/*` tags — the taxonomy changes only via
  a Dashboard nomination the user approves.
- Every surfaced claim links back to its source note (block IDs in the DB,
  wikilinks in the views).
- Wrong routing is tuning signal, not damage — prefer shipping an imperfect
  pass over stalling. But never force-fit: unrouted is a valid outcome.
- If something is genuinely ambiguous, put it on the Dashboard and move on.
