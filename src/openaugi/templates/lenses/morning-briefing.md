---
kind: engine
name: morning-briefing
description: >-
  "What matters today?" — yesterday distilled, open loops due, pending nominations, one resurfaced thought. One screen to start the day.
scope: >-
  [[Slowly Changing Context]] (read first), plus my own writing from the last ~48h, plus unanswered Dashboard nominations, plus open items from View - Open Loops if it exists. Exclude OpenAugi/-sourced blocks (the Dashboard and Open Loops view are the only derived inputs).
trigger: on-demand   # every: 1d once scheduling activates (after M4 gate)
target: >-
  view — overwrite OpenAugi/Views/View - Morning Briefing.md each run (a briefing is a cache, never an archive)
---

# Morning Briefing

## Intent

One screen that answers "what matters today?" in under a minute of
reading. The reader is the user, groggy, first thing: short, concrete, linked.
This is a MIRROR of what they already wrote and owe — never generic
advice, never coach-talk, no moralizing.

## Anchor

Read [[Slowly Changing Context]] first. It holds the slowly changing
commitment the user re-affirms every Sunday. Frame every "where you left
off" and "next smallest step" against its **Weekly focus** and **This
season** sections. Two checks, one line each:

- **Drift from commitment:** if yesterday's writing is mostly about a
  thread on the Season's parked list, say so once, with the source —
  evidence, not a verdict.
- **What changed since last Sunday:** one line from the note's change
  log. Flag once if the note is more than 7 days past `last_reviewed`.

## Process

1. Pull the last ~48h of their own blocks (`recent` / `search(after=...)`).
2. Read the current `View - Dashboard.md` for unanswered nominations and
   `View - Open Loops.md` (if present) for aging loops.
3. Pick ONE resurfaced thought: semantically related past block (~6–18
   months old) that speaks to what they're currently working on
   (`get_context` on yesterday's dominant theme). Skip if nothing truly
   fits — a forced callback is worse than none.
4. Overwrite `View - Morning Briefing.md`:

```
# Morning Briefing — YYYY-MM-DD (Weekday)

*Since last Sunday: <one line from [[Slowly Changing Context]] change log> · Focus this week: <the one line from Weekly focus that today's writing touches>*

**Yesterday:** 2–4 bullets, highest-signal only, each linked to its
source note. What moved, what was decided, what was captured that
matters.

**Waiting on you:** pending Dashboard nominations (count + the one-line
gist of each, linked) · open loops that are aging (top 2–3 by age/weight).
Omit the section if empty — an empty section is noise.

**From your past self:** the one resurfaced block, quoted verbatim,
dated, linked. One line on why it surfaced.

*Generated YYYY-MM-DD HH:MM from N blocks.*
```

## Hard rules

- 15 lines max in the body. Anything longer buries the point of a briefing.
- Every claim links to its source note. No unlinked assertions.
- Mirror, not coach: report what they wrote and owe; never prescribe
  habits, never add motivational framing.
- Nominate nothing here — the briefing reads state, it doesn't create
  decisions. Nominations stay on the Dashboard.
