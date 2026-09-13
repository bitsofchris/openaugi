---
kind: engine
name: decision-audit
description: >-
  "What am I actually deciding, and what does my own evidence say?" — find live decision language ('should I', 'deciding between', 'torn on') and audit ONE decision against my own prior thinking. Superforecasting applied to myself.
scope: >-
  at apply time — a specific decision named by me, or (default) my own writing from the last 14 days scanned for open decision language. Evidence gathering runs against the whole vault.
trigger: on-demand
target: >-
  note — OpenAugi/Notes/YYYY-MM-DD - Decision Audit - <slug>.md (opens with `- [ ] seen`)
---

# Decision Audit

## Intent

The trigger design: *decision language detected in recent notes ('should I',
'deciding between') → decision audit.* This lens is that audit, on demand:
take ONE live decision and lay out what THEIR OWN accumulated evidence says —
options, past positions, what they've tried before, and what would change
their mind.
A mirror with structure, not advice.

## Process

1. **Find the decision.** If named in the apply instruction, use it.
   Otherwise scan recent blocks for open decision language ("should I",
   "deciding between", "torn", "not sure whether", "vs"). Multiple found
   → audit the one with the most recent activity; list the others in one
   footer line. None found → say so, write nothing.
2. **Gather THEIR evidence** (whole vault): past positions on this exact
   choice (echoes!), prior similar decisions and how they went, stated
   values/constraints that bear on it ([[My Taxonomy]] areas, season
   goals, operating-system notes).
3. **Write ONE audit note:**

```
# Decision Audit — <the decision, one line>

**The decision:** as they framed it, quoted + linked.
**Options on the table:** A / B (/ C) — from their words, not invented.
**Your own evidence:**
- For A: past blocks, quoted + dated + linked
- For B: same
- Prior similar call: <what they chose then, what happened> ([[source]])
**Where you already lean:** if their recent language shows a lean, name it
with the quote that shows it. If genuinely split, say so.
**What would change your mind:** 1–2 falsifiable observations per option
— the superforecaster move: pre-register the evidence, then watch for it.
**Resolution check:** the date/event when this decision becomes gradeable.

- [ ] seen

*Audited YYYY-MM-DD from N blocks.*
```

## Hard rules

- ONE decision per run. Depth beats coverage.
- Evidence is THEIR words only — quoted, dated, linked. No external
  research, no pros/cons the vault doesn't support.
- NEVER recommend an option. The lens structures; they decide. The
  strongest allowed move is naming their own visible lean.
- If a past audit of the same decision exists, read it first and note
  what changed — decision drift is signal.
