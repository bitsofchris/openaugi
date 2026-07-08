---
name: lenses
description: The lens system — saved questions applied to your data. One markdown file per lens in <vault>/OpenAugi/AGENT/lenses/ (scope + trigger + intent + target); applied from any surface via the trigger contract ("apply lens X to SCOPE"); listed in the context pack so mobile can render apply-chips. The engine is prose (augi-agent.md), not code.
---

# Lenses

## When to use this doc

- You want to add, edit, or apply a lens
- You're wiring a new surface (mobile, plugin) to lenses
- You forgot the spec format or scope grammar

Design record: [docs/plans/lens-framework.md](plans/lens-framework.md).
Live mechanics (the prompt the agent follows): the **Lenses** section of
`<vault>/OpenAugi/AGENT/augi-agent.md`.

## The idea in one paragraph

A **lens** = a saved question applied to your data: *scope + intent →
derived artifact*. The durable value of OpenAugi is not routing or views —
those are plumbing and delivery — it is the growing library of questions
you can re-ask forever ("what nuggets are buried in my notes?", "distill
my thinking on X"). So lenses are **data, not code**: one markdown file
per lens, in the vault, editable like any note. Adding a lens = writing a
file. The "engine" is the agent following the generic apply-lens
instructions in augi-agent.md; a code engine is deliberately deferred
until lens specs visibly outgrow prose.

## The spec — one file per lens (the contract)

**The authoritative, annotated contract is
`src/openaugi/templates/lens-template.md`** (vault copy:
`OpenAugi/AGENT/lens-template.md` — copy it to start a new lens). Same
pattern as the task-file contract: one file defines the format, writers
and readers agree on it, and `tests/test_lens_contract.py` breaks if a
shipped lens or the reader drifts. In brief —
`<vault>/OpenAugi/AGENT/lenses/<name>.md`:

```yaml
---
name: nuggets                # kebab-case, matches filename. REQUIRED (all five are)
description: >-
  What this lens answers, one line (surfaces display this).
scope: >-
  Default retrieval recipe, plain prose (overridable at apply time).
trigger: on-demand           # on-pass / every 7d (no colon!) — dormant until scheduling activates
target: >-
  dashboard                  # dashboard | note — <path> | view — overwrite <View - X.md>
---
<intent — the prompt body. Optional persona/reference links.>
```

`description`/`scope`/`target` are folded scalars (`>-`) — bare values
with quotes or `: ` are invalid YAML. Validate: `openaugi lenses --check`.

Shipped lenses: `lenses/distill.md` (topic → one curated note with
provenance), `lenses/nuggets.md` (working notes → 3–7 promotion
nominations), `lenses/cluster-weather.md` (concept-cluster growth/death →
Dashboard nominations; backed by the deterministic pre-compute in
[docs/clustering.md](clustering.md) — `openaugi cluster` +
`openaugi cluster-weather`), `lenses/idea-lineage.md` (one topic → its
full biography in the Persistent Memory Artifact shape; backed by
`openaugi lineage "<topic>" --json --write`, whose
`OpenAugi/lineage/<slug>.json` sidecar doubles as the mobile timeline
payload), plus the vault-side JARVIS starter set (morning-briefing,
open-loops, echoes, …). The old `distill-lens.md` / `nugget-lens.md`
paths are pointer stubs.

## Applying a lens — from any surface

Every surface converges on the trigger contract (a task file), so this is
one instruction shape everywhere: **"apply lens NAME"** or **"apply lens
NAME to SCOPE"** (lens names also work naturally: "distill X", "find the
nuggets").

- **Chat:** say it in any Claude session with the openaugi MCP.
- **Any note:** `zzz: apply lens nuggets to this week` — dispatch handles it.
- **Mobile:** the context pack carries `lenses: [{name, description}]`;
  the app renders them as chips — tapping one appends
  `zzz: apply lens <name>` to the block text, which dispatches after sync
  + ingest. Block-scoped lens application with zero mobile-specific server
  work.
- **Plugin (later):** "Apply lens to selection" generalizes the M3b
  "Distill selection" command — selection becomes the scope.

**Scope grammar** (loose text, LLM-interpreted; explicit scope overrides
the spec default): `this block` · `[[Note]]` · `container: <title>` ·
`since: 14d` · `query: <terms>` · or handed/selected context (never
expanded uninvited).

**Targets follow the trust model:** `dashboard` output uses the standard
nomination grammar (checkbox + `^nom-*` anchor + answer slot); `note`
output is one `#human-review` note with provenance; only `view:*` targets
regenerate silently.

## Creating a lens — from any surface

**"new lens NAME: INTENT"** (chat, zzz, mobile capture). The agent writes
the spec file directly — lenses live in agent-space, so no nomination
gate — tagged `#human-review`, with a one-line Dashboard note. You tune a
lens by editing its file; you delete a lens by deleting its file.

**Robustness (productionized 2026-07-07):** broken frontmatter cannot
lose a lens. Invalid YAML (e.g. a description starting with a `"quoted
phrase"` or containing `: `) is **salvaged, not skipped** — the lens
still ships to the context pack with a best-effort name/description, and
a warning lands in the log. Validate any time with **`openaugi lenses`**
(table of every lens + status) or `openaugi lenses --check` (non-zero
exit on broken specs — CI-able). The safe authoring style is folded
scalars: `description: >-` with the text indented on the next line.

## Scheduling (dormant)

Specs declare `trigger: on-pass` / `every <period>` now, but the review
pass does NOT run them until the routing-quality gate (master plan M4)
passes. Then the pass becomes the scheduler: list the folder, run what's
due. No cron, no daemon — the pass is already the heartbeat.

## Wiring notes (for surfaces)

- The machine-readable lens list is in `context-pack.json` (`lenses`
  field), built by `src/openaugi/pipeline/context_pack.py` from the lens
  folder's frontmatter. **Transport-agnostic:** the mobile bridge serves
  the file today; a future HTTP endpoint serves the same builder's output.
- Templates for new users: `src/openaugi/templates/lenses/*.md`, copied
  by `openaugi init` (vault copies are the live versions).
