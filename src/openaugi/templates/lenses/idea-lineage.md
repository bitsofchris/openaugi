---
name: idea-lineage
description: >-
  "How did my thinking on X evolve?" — the full biography of one idea: earliest mention → major revisions → current strongest form → dead branches. Durable artifact per topic.
scope: >-
  a named topic, required at apply time ("apply lens idea-lineage to dopamine"). Evidence = `openaugi lineage` pre-compute over ALL history, then targeted retrieval into the strongest eras.
trigger: on-demand
target: >-
  note — OpenAugi/Notes/YYYY-MM-DD - Lineage - <topic>.md (durable, #human-review)
---
# Idea Lineage

## Intent

The biography of one idea, in Chris's own artifact shape (his 2026-06-06
"Persistent Memory Artifact System" note): *show how an idea evolved
instead of re-discovering it.* Not a summary of a topic — the story of
how HIS thinking on it moved.

Sibling lens: **echoes** answers "have I thought this before?" for
*current* thinking — small, disposable, recognition. Idea-lineage is the
opposite duty cycle: one *named* topic, full history, durable artifact.
If the trigger is "I just wrote something," use echoes; if it's "trace
X for me," this lens.

## Process

1. **Pre-compute** (repo: openaugi — see [[Repos]]):
   `openaugi lineage "<topic>" --json --write`
   The JSON gives era-bucketed evidence: first/last mention, quarterly
   activity, dormant gaps, per-era strongest blocks, each flagged
   `third_party` when it carries a `source/*` tag. `--write` drops
   `OpenAugi/lineage/<slug>.json` — the same payload the mobile app will
   render as a timeline; always pass it.
2. **Deepen the pivotal eras.** The pre-compute finds WHEN; you find
   WHAT changed. Pull full content (`get_blocks`) for the strongest
   blocks of the first era, each era where the framing shifted, and the
   latest era. Follow 1-hop wikilinks where a block points at the idea's
   ancestors.
3. **Write ONE note** — the Persistent Memory Artifact shape, verbatim
   quotes throughout:

```
# Lineage — <topic>

**Earliest known mention:** YYYY-MM-DD — "<verbatim>" ([[source]])

**Major revisions:** (one line per real shift, dated, quoted)
- YYYY-MM-DD — reframed as … ([[source]])

**Repeated language:** phrases he returns to verbatim across eras

**Current strongest form:** YYYY-MM-DD — "<his best articulation, quoted>" ([[source]])

**Dead branches:** directions tried and abandoned; dormant gaps from the
pre-compute ("quiet 2025-Q2→Q3, returned changed")

**Influences:** third-party blocks (books, podcasts, AI chats) that fed
the idea — cited as influences, NEVER counted as his voice

**Next concrete test:** ONLY if he named one in his notes — quote it.
Never invent a next step.

#human-review · *Lineage run YYYY-MM-DD over N blocks (YYYY-MM-DD → YYYY-MM-DD). Sidecar: OpenAugi/lineage/<slug>.json*
```

## Hard rules

- His words carry the artifact — quote verbatim, link every quote.
- `third_party` blocks are influences, never his voice; keep the
  sections separate.
- Revisions are shifts in HIS framing, not activity spikes. A busy
  quarter with no new framing is not a revision.
- No judgment, no advice, no invented next steps. Dead branches are
  reported as history, not failure.
- One note per run per topic. Re-running the same topic overwrites its
  lineage note (it's a derived artifact) — but never any other note.
