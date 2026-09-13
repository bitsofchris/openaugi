---
name: core-principles
description: The skeleton of OpenAugi — the four design commitments everything else hangs on: capture grammar, the layer model (truth/index/cache/render, plus the bronze/silver/gold canonicity axis), the trust model, and promotion. If a proposed feature violates one of these, the feature is wrong, not the principle.
---

# Core Principles

## When to use this doc

Before designing or building anything that touches capture, storage,
synthesis, or surfaces. These four commitments are the invariants; plans
and features churn around them. If a change would violate one, that's a
signal to redesign the change — or to make an explicit, recorded decision
to amend the principle (which should be rare).

**The intent in one line:** capture seamlessly, then let the agent help
organize, dedupe, link, and show where you left off. Every principle
below serves that sentence.

---

## 1. Capture grammar — three prefixes, nothing else

Capture must be seamless, so the grammar is minimal and does exactly two
jobs:

- **`qqq`** — block delimiter. Parser-level chunking so thoughts dumped
  in one sitting become separate, individually routable blocks. This is
  what makes "just write" possible.
- **`zzz:`** — dispatch channel. "Agent, go do this" — spawns a task.
- **`aaa:`** — instruction channel. "Agent, when you process this block,
  do it this way" — routing/parsing directives, obeyed at pass time.

**Why the grammar exists (and why "the agent can just infer it" is
wrong):** `aaa`/`zzz` are **channel separation** — they distinguish
*talking to the agent* from *content being saved*, so the agent never
has to infer which is which. Inference fails in exactly the places it's
most costly. The grammar is three prefixes; it is the cheapest component
in the system and carries the most safety. A system built around a
fallible agent *should* have explicit channels — that's not distrust as
a bug, it's distrust as an architecture requirement.

Also durable by design: an `aaa:` line lives in the text, so it survives
edits and re-ingest (unlike DB state) — it's the durable form of routing
intent under the re-derive contract.

## 2. The layer model — truth / index / cache / render

```
TRUTH   vault: captures + notes, incl. synthesized ones kept on command
INDEX   augi DB: blocks, edges, membership, embeddings — rebuildable from truth
CACHE   recaps, views, dashboards, membership logs — rendered queries, disposable
RENDER  obsidian + plugin pane, mobile app — same API, own nothing
```

The test that separates the layers: **a cache is something you could
delete with zero grief.** Views, recaps, dashboards pass. A note the
user would edit, link to, or build on does not — that's truth, whoever
drafted it.

Consequences:

- Truth is append-only and belongs to the user. Agents never edit notes
  outside `OpenAugi/` (and even there, only their own artifacts).
- The index is a disposable projection. Wrong routing is tuning signal,
  not damage; edited blocks re-enter the queue and get re-decided (the
  re-derive contract).
- Caches regenerate freely, no review needed. Anything that needs review
  before overwrite is, by definition, not a cache — check which layer
  it's really in.
- Render surfaces own nothing: same query API whether the pixels are in
  an Obsidian pane or the mobile app
  (see [../plans/views-as-rendered-queries.md](../plans/views-as-rendered-queries.md)).

### The other axis: bronze / silver / gold — canonicity

`truth / index / cache / render` says **where a thing lives**. It does not say
**how canonical it is**. That's a second, independent axis, and conflating them
has already caused one naming collision (see below).

| Layer | What it is | In the contextgraph | Marker |
|---|---|---|---|
| **Bronze** | Raw capture. Almost everything. | data block | **none — it is the default** |
| **Silver** | Nuggets aggregated across the vault into something canonical | context block | `#note-type/moc` + a filled `description` |
| **Gold** | The big never-ending notes you append to; the map | container | `#note-type/amoc` / `#note-type/pmoc` + `description` |

Most blocks stay bronze and need no tag. Promotion to silver means a few
nuggets from across the vault were aggregated into something canonical, and
that is the layer where `description` metadata appears and becomes browsable.
The map is the high-level overview of the gold notes.

Three things follow, and they're the whole reason this axis is worth naming:

- **Bronze is the absence of a marker, never a tag.** Tagging the default
  would mean tagging 32,000 blocks to say nothing. If a block carries no
  container tag and no description, it is bronze.
- **The `description` is what promotion actually buys.** It is not decoration:
  a filled `description` is what makes a note a routing target. So silver is
  the layer at which a note starts *attracting* blocks instead of merely
  holding them — which is also what makes it browsable.
- **Promotion is bronze → silver, and it is principle 4.** Rendered by
  default, materialized on command. The two principles are the same rule seen
  from the storage axis and the canonicity axis.

**Retired 2026-08-18: `#layer/bronze`.** It used to mean *"the user demoted
this block; it carries no signal"* — a **salience** flag, not a layer, and
under "bronze is the default" the two readings can't both hold. It affected 5
blocks of 32,045. Hiding deprecated or ephemeral material is a real need and
gets its own mechanism later; it is not a layer.

## 3. The trust model — nominate → command → assemble

Structure changes (new containers, new tags, merges, anything that
shapes the map) are **never autonomous**. The agent nominates, the user
commands, the agent assembles. Corollaries:

- The taxonomy is a closed vocabulary; agents never invent facets.
- The registry (containers + their `description` routing map) is the
  agent's map of the user's world — maintained through nominations,
  readable as skill-style name+description entries.
- Agent output that enters truth carries `#human-review` until vetted.
- From the user's side "the process" is exactly two things: **act on all
  `aaa:`/`zzz:` instructions, and process all new blocks — which the
  user then reviews.** Everything else is implementation.

## 4. Promotion — rendered by default, materialized on command

Synthesis starts life as a query result (a rendered view, a stitched
answer, a lens output in a pane). It becomes a vault note **only on an
explicit human command** ("save this as a note"). Then it is silver:
truth that happens to be machine-drafted — curated, linkable, greppable,
daemon-independent.

- Silver notes are canonical groupings/synthesis of ideas the user is
  working through and can't find or forgot they had. They are NOT views
  and never regenerate silently.
- AMOCs/PMOCs are the active-work containers; silver notes hang off
  them.
- This one rule structurally prevents the agent from flooding the vault:
  drafts are free, keeping is a decision.

## Linked docs

- [user-guide.md](user-guide.md) — the day-to-day manual built on these principles
- [review-pass.md](review-pass.md) — the write-back loop (grammar + routing in practice)
- [data-model.md](data-model.md) — the index layer in detail
- [../plans/views-as-rendered-queries.md](../plans/views-as-rendered-queries.md) — the cache/render split, design in flight
- [../vision/values.md](../vision/values.md) — the project values these principles implement
