---
kind: engine
name: echoes
description: >-
  "Have I thought this before?" — match current thinking against older notes and show the lineage: where an idea first appeared, how it grew, stalled, or returned in a new form.
scope: >-
  at apply time — a block, a note, or today's captures (default: my own writing from the last 48h). Matches run against the WHOLE vault history, all years. Exclude OpenAugi/-sourced blocks as inputs; they may be cited as evidence.
trigger: on-demand
target: >-
  note — OpenAugi/Notes/YYYY-MM-DD - Echoes - <topic>.md (one per run; small, disposable, opens with `- [ ] seen`)
---

# Echoes

## Intent

The most powerful surface may be the one that shows idea lineage over time:
how a current thought echoes older notes, grows, stalls, or returns in a new
form. This lens is that surface, on demand.

Given current thinking (the scope), find its **echoes**: past blocks —
months or years old — where they already circled this idea. Then say what
the lineage looks like. The goal is the *"you wrote nearly this in
March"* moment: recognition, not summary.

## Process

1. Extract the 1–3 core ideas from the scope (not topics — claims,
   framings, questions).
2. For each, `get_context` / semantic search across ALL history — bias
   OLD: an echo from 2024 is worth ten from last week. Follow wikilinks
   of strong hits one hop (`get_related`).
3. Keep only true echoes: same idea, prior form. Topical overlap is not
   an echo. 2–5 echoes max; zero is a valid, reportable result ("this
   appears genuinely new").
4. Write ONE small note:

```
# Echoes — <current idea, compressed>

**Now:** <one line, linked to the triggering block/note>

**Then:**
- **2025-03-10** — "<verbatim quote, trimmed>" ([[source note]]) — first appearance
- **2025-11-02** — "<quote>" ([[source]]) — grew: added X
- **2026-04-19** — "<quote>" ([[source]]) — stalled here / returned as Y

**Lineage read:** 2–3 sentences: is this idea growing, looping without
progress, or back with something new? If it keeps returning unresolved,
say so plainly — that's the signal.

- [ ] seen

*Echoes run YYYY-MM-DD over full history.*
```

## Hard rules

- Quote the past verbatim — the punch is their own words, not paraphrase.
- Never force an echo. "No prior trace" is a finding.
- Looping-without-progress is reported as observation, never as
  judgment ("this is the 4th unresolved return" — full stop, no advice).
- One note per run; never edit their notes; sources never marked.
