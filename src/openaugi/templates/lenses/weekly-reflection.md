---
kind: engine
name: weekly-reflection
description: >-
  The Sunday pass, drafted. What the week was about, everything surfaced that should not slip, the patterns they cannot see from inside the week, then two decision sections they answer with checkboxes: the three slots for next week, and where everything outside the slots goes. The only place priorities move.
scope: >-
  Every block from the last 7 days via the OpenAugi MCP (`recent` or `search` with a date window): daily notes, ideas, tasks, reflections. [[Slowly Changing Context]] §3 and §4. The Board and the Backlog (`_private/0-Current Focus/Kanban.md`, `Backlog.md`). The 2–3 most recent prior reflections in `_private/5-Journals/Weekly Reflections/` (`WK - YY-MM-DD.md`), for longitudinal signal only. Pending nominations on `OpenAugi/Views/View - Dashboard.md`.
trigger: on-demand   # Sunday, by them; the /weekly-reflection skill is a pointer to this file
target: >-
  note — OpenAugi/Drafts/WK-YY-MM-DD-Reflection.md (date = end of the week reflected on). Not to chat. They read it, take their own notes, and answer in their own WK note (`# My Thoughts` above, this draft under `# AI Thoughts`).
---

# Weekly Reflection

## Intent

The bigger pass. The daily board is a terse pointer; this is where the week
gets read, priorities get ruled, and everything captured outside the slots
gets a home. It is read top to bottom, so the recap comes first and every
decision it asks is a checkbox. Honest, not encouraging. No padding, no
cheerful wrap-up. The value is what is hard to see from inside the week.

Who the user is, for the synthesis, and their commitments and this season's
lanes are in [[Slowly Changing Context]]; read it before writing a word.

## Process

1. **Gather.** All blocks from the last 7 days. The prior 2–3 reflections,
   for pattern detection across weeks, never for re-summarizing. §3 and §4 of
   the commitment note. Both boards. The Dashboard's pending nominations.
2. **Synthesize** (sections below, in this order).
3. **Priority ruling**, then **Triage outside the slots**, as checkboxes.
4. Write the file. First body line `- [ ] seen`, then `#area/weekly-reflection`.
   Confirm the path in chat and nothing else.

Concept extraction from the week's notes (grouping ideas into new notes) is
`parse-notes`, run separately.

## Sections

**TLDR** — two or three sentences. What the week was actually about, the
underlying thing, not a list.

**What happened / what you were thinking about** — factual recap grouped by
lane (OpenAugi, research, content, self, work). Memory jog, not a re-read.

**Surfaced this week** — an extraction pass over every block; the safety
net, so when in doubt include it. One line per item, duplicates merged with
a count, labeled:

- `[idea]` — project, content idea, workstream, feature
- `[research]` — question worth answering
- `[lesson]` — insight or thing learned
- `[todo]` — something they said should happen
- `[recurring]` — appeared in more than one block; stronger signal

**Patterns and connections** — what keeps recurring, where two threads
rhyme without being connected, what has been deferred more than one week.
Name tensions and contradictions when the data shows them.

**Bottlenecks and blind spots** — where energy is stuck; what gets the most
words and the least progress; whether the activity mix matches the bigger
bets or is drift.

**Am I working on the right things?** — given the longer arc, is this
week's pattern high-leverage. Direct.

**Questions to sit with** — one or two they would not ask themselves.

## Priority ruling — next week's three slots

§4 of [[Slowly Changing Context]] is three named slots, P0–P2, plus Self
and Work lines. For each: quote the current line, then affirm / change /
done, with dated evidence in their words. A proposed change is written as the
full replacement line so accepting costs one tick. The other sections (North
Star, Season, Parked) get one line each only if the week contradicts them;
otherwise "no contradicting writing" once. Never edit the note; they tick,
then applies or asks a session to.

```
**Priority ruling — week of YYYY-MM-DD** ([[Slowly Changing Context]] §4)

- **P0 · <lane>** — current: "<line>"
    - [ ] affirm
    - [ ] change to: "<proposed line>" — evidence: [[2026-09-11]] "..."
    - [ ] done, retire
- **P1 · <lane>** — current: "<line>"
    - [ ] affirm · [ ] change to: "..." · [ ] done
- **P2 · <lane>** — current: "<line>"
    - [ ] affirm · [ ] change to: "..." · [ ] done
- **Self / Work** — one line each, same boxes.
- **Season / Parked / North Star** — no contradicting writing this week. (Or one line per contradiction, dated.)
```

## Triage outside the slots

Everything in *Surfaced this week* that is not a slot gets one line, one
destination, one checkbox. They tick; the next session executes. An unrouted
item is the failure mode. Destinations:

- `→ Backlog · <column>` — an idea or thread with an end state; a card on
  the Backlog in its AMOC column with a source date. If it deserves its own
  note first, a tier-two dated note per `OpenAugi/AGENT/pmoc.md`, linked
  from the card.
- `→ Board · Household / Other` — errands, family, money logistics.
- `→ [[<PMOC>]]` — belongs to an active project; a dated entry there, not
  a card.
- `→ dispatch` — a task worth running; state the one-line ask.
- `→ drop` — noise.

Also here, from `OpenAugi/AGENT/kanban.md`: one proposed destination per
card in the Board's Triage column; one line naming the cards for any AMOC
column over its cap of three, with no nomination; a pull from the Backlog
only if a column has room and their writing touched a Backlog card this week.
No push suggestions. They move every card; this section only proposes.

Dashboard nominations go in this section as their own sub-list (accept /
reject / park) with each one's age, so there is one triage surface. The
queue leaves the reflection at zero pending; decisions are written into the
Dashboard's `- answer:` slots so the review pass executes them.

```
**Triage outside the slots**

- [ ] `→ dispatch` <ask in one line>. Source: [[2026-09-09]]
- [ ] `→ [[PMOC - …]]` <what and why>. Source: [[2026-09-10]]
- [ ] `→ Backlog · Content` <idea>. Source: [[2026-09-09]]

Triage column: - [ ] `→ Board · OpenAugi` <card> · - [ ] `→ Backlog · OpenAugi`
Over cap: OpenAugi has 4 — <cards>. Which one moves?

Nominations (from [[View - Dashboard]]):
- [ ] accept · [ ] reject · [ ] park — <nomination>, riding since <date>
```

## Rules

- File order is fixed: the synthesis sections, then Priority ruling, then
  Triage outside the slots. Decisions are checkboxes, never prose proposals.
- Prior reflections are for patterns across weeks, not for summarizing again.
- Never edit [[Slowly Changing Context]], the Board, or the Backlog from this
  lens. Propose; they move.
