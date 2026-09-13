---
kind: engine
name: nuggets
description: Find stand-alone insights buried in recent working notes and nominate the best few for promotion (bronze→silver). Post seeds included.
scope: my own writing since the last nugget run (fallback ~2 weeks) — dailies, working notes, MOC journals. Exclude OpenAugi/-sourced blocks and third-party source/* material.
trigger: on-demand
target: dashboard
---

# Nuggets

## Intent

Find the handful of blocks that are INDIVIDUALLY worth keeping — then ask,
never act.

A nugget is ONE block (occasionally 2–3 adjacent) that stands alone:

- a design principle or aphorism in the user's own words
- a reusable framing/insight that would survive out of context
- a post seed — something a reader would highlight
- a decision rationale worth finding again in a year

A nugget is NOT a theme. "Five blocks orbit capture-UX" is **gravity**
(the review pass's job — many blocks, one cluster). A nugget is the
opposite shape: one block, self-contained. If you find a theme, leave it
for gravity. If you find a quote-worthy sentence, that's yours.

**Bar: ruthless.** Nominate 3–7 per run, never more. A mediocre nugget
nomination costs user trust; an unnominated block stays findable by
search. When unsure, skip.

## Process

1. **Scope the window.** Check the Dashboard's Nuggets section for the
   last run's date (footer line); scan blocks since then via
   `search(after=...)` / `recent`, applying the scope above.
2. **Harvest candidates.** Read for stand-alone value, not topical
   relevance. The test: would this sentence be worth reading with NO
   surrounding context? Check the block isn't already promoted (search
   for an existing note covering it — if one exists, skip or suggest a
   link instead).
3. **Nominate on `View - Dashboard.md`** under a `## Nuggets` section,
   using the standard nomination grammar (checkbox + stable anchor +
   answer slot):

   ```
   - [ ] **Nugget:** "capture is a database write" — post seed? promote to a note? ([[2026-07-03]]) ^nom-promote-capture-is-a-db-write
       - answer:
   ```

   Quote the nugget verbatim (trimmed), link the source note, suggest a
   disposition (note / post seed / link into [[existing note]]). End the
   section with `*Nugget lens last run: YYYY-MM-DD over N blocks.*`
4. **On an answered nomination** (checked box = yes as proposed; filled
   answer = specific instruction): assemble — usually ONE small note via
   `write_document` to `OpenAugi/Notes/` opening with `- [ ] seen`, the
   verbatim nugget, a line of context, and a wikilink back to the source
   note. Post seeds get appended to the content-pipeline section instead.

## Hard rules

- Nominate-only. No notes, no tags, no routing without an answered
  nomination.
- User's voice only — never nominate third-party material as a nugget.
- Never edit notes outside `OpenAugi/`. Sources are never marked or moved.
- Preserve unanswered nominations verbatim (anchor included) when the
  Dashboard regenerates.
