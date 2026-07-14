---
name: agentic-kb-field-guide
description: The portable ruleset for standing up an agentic knowledge base anywhere — what actually building OpenAugi hardened or simplified from the "agent + janitor + flat folder" starting advice, with the prior-art map showing each rule is a database lesson (log/views, CQRS, view selection, event time). System-agnostic; transplantable to a work/team knowledge base without any OpenAugi code.
---

# Agentic KB Field Guide

## When to use this doc

You're standing up an agentic knowledge base somewhere OpenAugi isn't —
a team drive at work, a repo wiki, a different toolchain — and want the
rules without the implementation. Or you're sanity-checking a new
OpenAugi feature against what the build actually taught. The
repo-specific mechanics live in [core-principles.md](core-principles.md)
and [review-pass.md](review-pass.md); this doc is the subset that
transplants.

Companion to the public post "The Map of Agentic Knowledge Bases and How
to Get Started" — the post is the starting advice; this is what running
the system for months did to that advice.

## The starting advice that survived intact

- **Agent + janitor + flat folder.** Still the right week-one system.
  Don't design the taxonomy first; stratify with rough tags, let
  clusters reveal the real structure.
- **Route, don't synthesize, by default.** The base exists to compound
  the author's thinking, not to replace it with generated pages.
- **Climb a rung only when the current one creaks.** Every layer we
  added answered a problem the previous layer created — none was
  built speculatively.

## Rules the build hardened

Each entry: the loose starting rule → the hard version, and what forced it.

### 1. "Files are the source of truth" → four layers, one test

Truth (the human's writing) / index (rebuildable projection: blocks,
links, embeddings) / cache (rendered views, dashboards, recaps) /
render (UI surfaces that own nothing). The test that assigns a layer:
**a cache is something you could delete with zero grief.** Anything a
human would edit, link to, or build on is truth — whoever drafted it.
Two-layer thinking ("files + derived stuff") kept misfiling things;
the grief test settles every argument in one question.

Corollaries: truth is append-only and agents never edit it outside
their own output folder; anything that needs review before overwrite
is by definition not a cache.

### 2. "Talk to the agent in your notes" → explicit channels, never inference

A tiny capture grammar (here: `qqq` block delimiter, `zzz:` dispatch,
`aaa:` processing instruction) separates *talking to the agent* from
*content being saved*. Don't let the agent infer which is which —
inference fails exactly where it's most costly. A system built around
a fallible agent should have explicit channels: that's distrust as an
architecture requirement, not a bug. Three prefixes is the cheapest
component in the system and carries the most safety.

Bonus durability: an instruction that lives in the text survives edits
and re-ingest, unlike anything stored in a database (see rule 6).

### 3. "The janitor keeps it tidy" → the janitor never changes structure

Cleaning (applying existing tags, linking, filing into existing
containers) is autonomous. Structure changes — new tags, new
containers, merges, promotions — are **never** autonomous: the agent
nominates, the human commands, the agent assembles. The taxonomy is a
closed vocabulary the agent cannot extend, and the agent never re-tags
what the human already tagged. This one split is what makes the system
trustable enough to run unattended.

### 4. "Synthesize on demand" → rendered by default, kept on command

Synthesis starts life as a query result — a view, a stitched answer in
chat — and becomes a real note **only on an explicit human command**.
Drafts are free; keeping is a decision. This single rule structurally
prevents the agent from flooding the base with generated pages, which
is the failure mode that kills the route-over-synthesize principle in
practice. Persist a synthesis only when you catch yourself re-deriving
it — reuse is the signal.

### 5. "Build a routing index" → registry restraint

The routing map (containers the agent files into by inference) must
stay small. **Register a note only when captures from *elsewhere*
should land in it automatically.** A self-contained note the human
writes in directly gains nothing from registration — nominating it
just bloats the map and the human's yes/no queue. Unregistered ≠
unroutable: explicit instructions and links always work on any note.
Before nominating a registration, ask: does anything actually arrive
here from elsewhere by inference?

Related: **adopt before create** — when a cluster earns a home, first
look for an existing note that already is the canonical home and
upgrade it; mint a new note only when nothing exists.

### 6. "Derived state is rebuildable" → the re-derive contract

Push it all the way: filing *decisions* are also derived state. When
content is edited, its old routing drops and it re-enters the queue to
be **re-decided, not fuzzily preserved** (we built a similarity
matcher to migrate decisions across edits, then ripped it out).
Durable human intent belongs in the text itself, where it survives
re-ingest; the database holds only re-derivable state. Wrong filing is
tuning signal, not damage.

### 7. "Process what's new" → queue on ingest time, not content time

Operational, and it will bite anyone who builds a janitor: content
dates are unreliable (date-only stamps sort before same-day
timestamps; edits don't change them), so a "new since last run" queue
keyed on content time silently drops items. Key the queue on **when
the item entered the system**. Found when a pass returned zero blocks
while ten sat waiting.

### 8. "Keep the views fresh" → refresh by tier

Not every derived artifact should refresh on every pass, and the
principled line is cost of recomputation, not importance.
**Self-maintainable** artifacts — computable from the delta alone
(counts, date ranges, tag histograms, mechanical renders) — refresh
every pass for free. **Re-query** artifacts — anything needing LLM
synthesis over the full scope — refresh only when a would-it-change
check says their inputs actually shifted. Synthesis on a timer burns
tokens rewriting unchanged prose; delta-cheap stats on a lazy schedule
go stale for no reason.

## Rules the build simplified

- **The human's process is two verbs.** Answer the nominations; keep
  capturing. Everything else — routing, views, recaps, registry — is
  implementation the human never has to think about. If the human's
  job description grows past one sentence, the design is leaking.
- **One entry point.** Every surface (nominations, activity, parked
  items, lens index) folds into a single dashboard. Separate index
  files get minted, then never visited; fold them in.
- **Empty is a valid state.** Untagged and unfiled are fine — tag only
  what you'd query, file only what a view should distill. Routing ≠
  surfacing: everything can route cheaply, views surface only what's
  salient. Chasing 100% classification is make-work.
- **Nominations need a third answer.** Yes / no / **parked** — "no
  strong feelings, not now." Parked items sit on a visible shelf and
  fall away automatically after ~2 weeks untouched; a real cluster can
  re-nominate later. Without this, the review queue fills with
  decisions the human is avoiding, and the weekly review stops
  happening.

## The prior-art map — every rule is a database lesson

These rules aren't preferences. An agentic knowledge base recreates
the conditions databases were built for — multiple fallible writers
sharing one state — so running one re-derives the same invariants
data systems converged on decades ago (Kreps' *The Log*, Kleppmann's
*DDIA* Part III, CQRS/event sourcing). Each hardening above has a
name:

| Rule | The database idea it rediscovered |
|---|---|
| Four layers + the grief test (1) | The log vs. derived data; a cache is anything rebuildable by replay |
| Explicit channels (2) | Commands vs. data in event sourcing — commands are never inferred from state |
| Janitor never changes structure (3) | DML vs. DDL: content writes are autonomous, schema changes are gated — CQRS ownership split |
| Rendered by default, kept on command (4) | Materialized-view selection: materializing is an economic decision (benefit × reuse ÷ cost, the 1996 data-cube papers) |
| Registry restraint (5) | You can't materialize the whole cube lattice; an aggregate navigator needs a small routing table |
| Re-derive contract (6) | Projections rebuild by replay; the named anti-pattern is a projection writing back into the event store |
| Ingest time, not content time (7) | Event time vs. processing time — the classic stream-processing bug, silent-drop failure mode included |
| Refresh by tier (8) | Incremental view maintenance: self-maintainable views vs. views that require re-query |
| Staleness surfaced, not hidden | Eventual consistency, disclosed instead of masked |
| Parked with auto fall-away | Backpressure with TTL on the human review queue |

Two things the databases never had, which is where this stops being
rediscovery: the derived metadata here is *probabilistic and
expensive* (LLM-generated, embeddings), which flips the economics of
what to materialize — and the log is a *person*, so the system's
job is to compound the human writer, not just serve queries. Route
over synthesize and promotion-on-command exist for that reason; no
database textbook has them.

## Meta-rules — about the rules themselves

- **The rules live in the files.** A rule that exists only in chat
  history or an agent's session memory violates the system's own
  premise. When the agent learns a durable rule from a human decision,
  codify it into the prompt/config file the agent actually reads —
  that commit is the fix, not the memory.
- **One behavior spec.** The step-by-step procedure lives in exactly
  one file (the prompt the agent runs from); reference docs carry
  concepts only. If procedure shows up in two places, delete one.
- **The config file is load-bearing.** One file telling the agent how
  this base works turns a generic model into a librarian who knows
  these shelves. Every serious system in the field has one; treat it
  as the most important file you own.

## Linked docs

- [core-principles.md](core-principles.md) — the four invariants, OpenAugi-specific and in full
- [review-pass.md](review-pass.md) — the write-back loop these rules run inside
- [../plans/views-as-rendered-queries.md](../plans/views-as-rendered-queries.md) — design record behind rules 1 and 6
