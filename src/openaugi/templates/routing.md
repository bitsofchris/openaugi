---
kind: engine
type: agent-routing
status: status/needs-review
created: 2026-06-06
---

#area/openaugi #area/meta

# OpenAugi Agent Routing

Simple rule: **search first, then either append, create, or do nothing.**

This file exists so agent output becomes durable memory instead of one-off summaries.

## Write Rules

1. Search with OpenAugi before writing.
2. If the output continues an existing long-running thread, append to a mirror note.
3. If the output is a standalone artifact, create it in the right `OpenAugi/` folder.
4. If the idea already exists, do not create a duplicate. Link the existing note.
5. Every agent-created or agent-edited note needs a `- [ ] seen` checkbox as
   the first line of its body.

## Mirror Notes

**Superseded for container heads (2026-07-06):** the review pass
([[review-pass]], OpenAugi/AGENT/review-pass.md) now maintains derived views
in `OpenAugi/Views/` that the AMOCs transclude — prefer that over creating
new mirrors. Mirrors remain valid only for appending one-off agent output to
a long-running thread that has no view yet.

Mirror notes are agent-owned shadows of user-owned AMOCs, PMOCs, and MOCs.

Use the prefix `MIRROR -` so they are visually distinct in Obsidian.

Examples:

| Source | Mirror |
|---|---|
| [[AMOC - OpenAugi Main]] | `OpenAugi/Threads/MIRROR - AMOC - OpenAugi Main.md` |
| [[PMOC - Audacity to take Action - Season 2 - Q2 2026]] | `OpenAugi/Threads/MIRROR - PMOC - Audacity to take Action - Season 2 - Q2 2026.md` |
| [[MOC - What's Next Journal - The One Thing]] | `OpenAugi/Threads/MIRROR - MOC - What's Next Journal - The One Thing.md` |
| [[MOC - Advice on Finding Your Niche Special Knowledge]] | `OpenAugi/Threads/MIRROR - MOC - Advice on Finding Your Niche Special Knowledge.md` |
| [[Weakness - Not resting enough]] | `OpenAugi/Threads/MIRROR - Weakness - Not resting enough.md` |

Append format:

```markdown
### YYYY-MM-DD (augi)

- [ ] seen

Source: [[source note]]

...
```

Do not write into the source note unless the human explicitly asks.

## Folder Routing

| Need | Destination |
|---|---|
| Continue an active thread | `OpenAugi/Threads/MIRROR - <source>.md` |
| Product / architecture / durable reference | `OpenAugi/Docs/` |
| Research synthesis | `OpenAugi/Research/` |
| Plan | `OpenAugi/Plans/` |
| Atomic concept | `OpenAugi/Notes/` |
| Concrete task | `OpenAugi/Tasks/` |

## PAUGI Artifacts

For PAUGI / personal observability work, prefer one of these artifact shapes:

- **Evidence map:** claim, support, counterevidence, confidence, next observation.
- **Idea lineage:** earliest mention, revisions, current form, next test.
- **Recurring-problem trail:** trigger, behavior, cost, working intervention.
- **Energy trail:** what creates pull, what state preceded it, what artifact resulted.
- **Season state:** active commitments, parked pulls, risk, next action.
- **Question ledger:** recurring unresolved questions with latest evidence.

## Contextgraph Rule

Use the "stratify then cluster" method:

1. **Stratify:** route by area, source, type, status, thread, or explicit link.
2. **Cluster:** within that scoped pile, use semantic similarity, backlinks, time, and repeated language.
3. **Promote:** when a cluster matters, create a context block / PAUGI artifact.
4. **Link:** connect the artifact back to raw evidence and parent threads.

Tags are rough sorting. Links and clusters reveal structure. Artifacts preserve what was learned.
