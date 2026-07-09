---
name: lens-framework
description: NEXT UP (not started). Lenses as data — scope + trigger + prompt(context/persona + intent) specs that generalize views, distillations, and future proactive dashboards. Chris's definition, captured 2026-07-06.
---

# Lens Framework — lenses as data

**Status: MVP SHIPPED 2026-07-07 as files + prose** — see
[docs/reference/lenses.md](../reference/lenses.md) for the live system. Lens specs are
markdown files in `<vault>/OpenAugi/AGENT/lenses/` (spec shape below,
lightly adapted); the "engine" is the generic apply-lens section in
`augi-agent.md`; the lens list rides the context pack so mobile can render
apply-chips. distill + nuggets migrated as the first two specs. Still
deliberately deferred: a code engine (deterministic scope resolution,
run-state tracking) — build trigger unchanged: when lens prose visibly
outgrows what the agent can follow reliably. Scheduling stays dormant
until master-plan M4 passes.

## Chris's definition (2026-07-06, verbatim intent)

A **lens** =

1. **Scope** — what data it reads: an openaugi retrieval recipe
   (search/get_context queries, container membership via `routed_to`,
   clusters, time window, `source/*` filters).
2. **Trigger / frequency** — when it runs: on-pass · on-demand ·
   after-N-new-blocks · every-N-days · on-event (future).
3. **Prompt** — two parts:
   - **Context / persona**: framing knowledge the lens carries — *"Here is
     James Hollis and his teachings; take on his role"* — a reference corpus
     or voice, not just instructions.
   - **Intent**: the question or transformation applied to the scope.

Plus a **render target**: where the derived artifact lands (a View file, a
Dashboard section, a distillation note, an HTML page, a push notification).

## The reframe this enables

The existing system is already lens outputs — this framework just names it:

- The **view kinds** (AMOC/PMOC/MOC recap emphases — rolling TLDR vs LEFT OFF
  vs current-understanding) are **three default lenses** shipped with the
  review pass. Different frontmatter/recap per note type = different lens
  spec per container kind.
- The **Dashboard** is a lens (scope: all containers + unrouted; intent:
  what moved / what needs my call).
- The **distill lens** is a lens with trigger=on-demand, target=Notes/.
- Future **proactive lenses** (habit trends, tornado detection, cluster
  weather, "you thought this in March") are new specs, not new machinery.

## Spec shape (when built)

One markdown file per lens under `OpenAugi/AGENT/lenses/`, skill-file style:

```yaml
---
name: <lens name>
description: <when this lens runs and what it answers>
scope: <retrieval recipe — queries, containers, sources, window>
trigger: on-pass | on-demand | after-n-blocks: N | every: <period>
persona: <optional — reference corpus/voice the lens speaks from>
intent: <the question/transformation>
target: view:<container> | dashboard:<section> | note:<folder> | html
---
```

The review pass becomes the scheduler: it reads specs, decides which lenses
are due (refresh tiers: mechanical always, synthesis on salience), runs them,
renders to targets. Persona corpora can be NotebookLM notebooks or vault
reference notes (`source/notebook` scoping already defined in My Taxonomy).

## Why not now

Two lenses, one user, policies don't diverge yet — a spec engine today is
framework-before-two-real-cases. The definition is captured here so nothing
is lost; the skill prose is the interim implementation.

## Related

- docs/plans/review-pass-v1.md (the running system these specs would configure)
- Vault: `2026-05-06 - Lens - the missing primitive in the Contextgraph`
- Kleppmann DDIA Part III (derived data / materialized view refresh — the
  theory Chris re-derived); Kreps, "The Log"
