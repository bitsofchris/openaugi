---
kind: engine
name: open-loops
description: >-
  "What did I say I'd do and never close?" — promises, unanswered questions, and 'I should…' statements aging in the capture stream.
scope: >-
  my own writing, last 30 days on first run; afterwards, new blocks since the last run PLUS every loop still open on the current view (loops carry forward until closed or dismissed). Exclude OpenAugi/-sourced blocks and third-party source/* material.
trigger: on-demand   # on-pass once scheduling activates (after M4 gate)
target: >-
  view — overwrite OpenAugi/Views/View - Open Loops.md each run
---

# Open Loops

## Intent

Surface the commitments and questions aging silently in the stream —
the things the user said they'd do, asked and never answered, or flagged and
never followed. This is trust-builder #1: the system remembers what they'd
otherwise drop. Mirror, not nag: report the loop and its age; never
scold, never editorialize about being behind.

## What counts as a loop

- **A commitment to a person** — "I'll send X to Y", "told Z I'd review…"
  (highest weight; people-facing loops age worst)
- **A question they asked and never answered** — genuine open questions in
  their own writing, not rhetorical ones
- **A stated intention with no follow-through block** — "I should…",
  "need to…", "next step is…", unchecked `- [ ]` items in raw notes
- **A promised follow-up** — "revisit this after…", "check back when…"

NOT loops: idle musings, completed items (see closure), anything already
tracked as a pending Dashboard nomination (link, don't duplicate), zzz
blocks (the dispatch system owns those).

## Closure evidence

Before listing a loop, search for a later block that closes it (the
thing shipped, the question answered, an explicit "done/dropped"). When
in doubt, keep it open but say why it looks possibly-closed.

## Output — the view

Overwrite `View - Open Loops.md`. Checkbox grammar, one loop per line,
stable anchor (same slug across runs so checked state survives):

```
# Open Loops

- [ ] **To Sarah:** send the draft — 12d old ([[2026-06-25]]) ^loop-send-draft-sarah
- [ ] **Question:** is the qqq splitter worth keeping? — 8d ([[2026-06-29]]) ^loop-qqq-splitter
...

*N open · oldest 21d · Generated YYYY-MM-DD from M blocks. Checking a box = closed or let go — next run removes it.*
```

Order: people-facing commitments first, then by age. Cap at ~15 lines —
if more, keep the heaviest and add one line: "…and K more, run 'apply
lens open-loops to everything' for the full list."

**A checked box means closed-or-dismissed** — next run drops that loop
without comment. Unchecked loops carry forward verbatim, age updated.

## Hard rules

- Their voice only; loops come from their own words, never inferred from
  third-party material.
- Every loop links to the source note where they said it.
- Never write into their notes; the view is the only output.
- Age honestly, dismiss silently — no guilt-framing, ever.
