---
name: nugget-lens (template)
description: >
  TEMPLATE — copied to <vault>/OpenAugi/AGENT/nugget-lens.md on `openaugi init`.
  The vault copy is the live version the agent reads. Edit there, not here.
  Recurring lens: scan recent working notes for NUGGETS — individually
  valuable insights buried in daily notes and working files — and nominate
  the best few on the Dashboard for promotion (bronze→silver). Use when the
  user says "run the nugget lens", "find the nuggets", "what's worth
  promoting from my notes", or a task file carries that instruction.
  Nominate-only: never creates notes without an answered nomination.
---

# Nugget Lens

A lens = scope + trigger + prompt(intent) + target. This one, spelled out
(prose is the implementation; the spec engine waits until lenses diverge):

- **Scope:** data blocks since the last nugget run (fallback: ~2 weeks) in
  the user's own writing — dailies, working notes, MOC journals. Exclude
  `OpenAugi/`-sourced blocks (derived) and third-party `source/*` material
  (readwise, notebooks — nuggets are the USER's ideas only).
- **Trigger:** on-demand ("run the nugget lens", zzz, or a task file).
- **Intent:** find the handful of blocks that are individually worth
  keeping — then ask, never act.
- **Target:** Dashboard nominations (the one answer surface).

## What a nugget is (and is not)

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
   `search(after=...)` / `recent`, excluding `OpenAugi/`-sourced blocks.
2. **Harvest candidates.** Read for stand-alone value, not topical
   relevance. The test: would this sentence be worth reading with NO
   surrounding context? Check the block isn't already promoted (search
   for an existing note covering it — if one exists, skip or suggest a
   link instead).
3. **Nominate on `View - Dashboard.md`** under a `## Nuggets` section,
   using the standard nomination grammar (stable anchor + answer slot):

   ```
   - **Nugget:** "capture is a database write" — post seed? promote to a note? ([[2026-07-03]]) ^nom-promote-capture-is-a-db-write
       - answer:
   ```

   Quote the nugget verbatim (trimmed), link the source note, suggest a
   disposition (note / post seed / link into [[existing note]]). End the
   section with `*Nugget lens last run: YYYY-MM-DD over N blocks.*`
4. **On an answered nomination** (this run's step 1, or when processing
   the dashboard): assemble per the answer — usually ONE small note via
   `write_document` to `OpenAugi/Notes/` with `#human-review`, the verbatim
   nugget, a line of context, and a wikilink back to the source note.
   Post seeds get appended to the content-pipeline section instead.

## Hard rules

- Nominate-only. No notes, no tags, no routing without an answered
  nomination.
- User's voice only — never nominate third-party material as a nugget.
- Never edit notes outside `OpenAugi/`. Sources are never marked or moved.
- Preserve unanswered nominations verbatim (anchor included) when the
  Dashboard regenerates — same rule as all nominations.
