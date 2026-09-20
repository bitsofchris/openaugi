---
kind: engine
name: weekly-reflection
description: >-
  The Sunday pass. Drafted at 06:30: what the week was about, a recap by lane, what kept coming up, what their writing contradicts, what they learned and are avoiding, candidates for next week, questions. Then the Sunday session: they open a chat and say `run the Sunday pass`; the agent walks the decisions one question at a time — Weekly focus lines, card moves, janitor fixes — and applies each as it is ruled. The only place priorities move.
scope: >-
  Every block from the last 7 days via the OpenAugi MCP (`recent` or `search` with a date window): daily notes, ideas, tasks, reflections — and every `Aaa:` instruction inside them. [[Slowly Changing Context]] §3 and §4. The Board and the Backlog (`_private/0-Current Focus/Kanban.md`, `Backlog.md`). The 2–3 prior reflections in `OpenAugi/Research/Weekly Reflection - *.md`, for patterns across weeks only. For Propose, additionally: their own `# My Thoughts` in `_private/0-Current Focus/WK - YY-MM-DD.md` and this Sunday's entry in `OpenAugi/Notes/System Janitor.md`.
trigger: every 7d   # Sunday 06:30 local (10:30 UTC; the scheduler has no clock, so this drifts an hour at DST — re-seed the ledger row then). Anchored 2026-09-20. On demand: the /weekly-reflection skill points here.
target: >-
  note — OpenAugi/Research/Weekly Reflection - YYYY-MM-DD.md (date = the Sunday it runs, the end of the week reflected on). Not to chat. The Sunday session appends its receipt to the same note; nothing else on Sunday gets its own file.
---

# Weekly Reflection

## Run

at: 06:30
on: Sun

The window is the seven days ending today. Read [[Slowly Changing Context]]
first, then both boards, then the prior 2–3 reflections. Follow **Draft**
below exactly: the scheduled brief carries no format of its own, and neither
may a board proposal — **the currency board never proposes this lens; it is
scheduled**, after the board (06:00) and the janitor (06:15).

dedupe: OpenAugi/Research/Weekly Reflection - {date}.md

Today's reflection existing is proof the run already landed. A rerun is their
call: delete the note, or say `apply lens weekly-reflection`.

## The Sunday, start to finish

1. **06:00** the board · **06:15** the janitor · **06:30** this draft. All
   scheduled; nothing to start.
2. They read the draft. Their own thoughts go in their WK note
   (`_private/0-Current Focus/WK - YY-MM-DD.md`, `# My Thoughts`; the draft
   linked under `# AI Thoughts`). That note is theirs; no lens writes to it.
3. They open a chat session and say **`run the Sunday pass`**. The session
   follows *The Sunday session* below: one question per turn, each ruling
   applied the moment it is made — Weekly focus, cards, janitor fixes.
4. **Done** is one fact: `last_reviewed` in [[Slowly Changing Context]] equals
   this Sunday. Monday's board says one line if it does not.

No boxes on the note, no second surface, no Claude-level skill: the process
is this file. The janitor's fixes, the boards and the context note are all
decided in that one session.

## Intent

The daily board is a terse pointer; this is where the week gets read. Honest,
not encouraging. No padding, no cheerful wrap-up. The value is what is hard to
see from inside the week, said in their own dated words. The draft is a read,
under ~700 words; the decisions come later and are checkboxes, never prose
proposals.

## Draft (the 06:30 run)

Gather the window, the boards, §3 and §4, the prior reflections. **Pull the
week's `Aaa:` lines as their own list** — grep the week's notes for a line
beginning `aaa:` (any case) — and check each against disk (a task in
`OpenAugi/Tasks/`, a note, a routed block). Then write, in this order, first
body line `- [ ] seen`, then `#area/weekly-reflection`:

**TLDR** — two or three sentences on the underlying thing the week was
about, then one line per lane (fundamentals, OpenAugi, content, self, work):
what actually happened, memory jog only. Under 200 words together.

**What kept coming up** — the themes, each with one dated quote. Merge
duplicates with a count. Lessons and research questions live here, in a
line, not in a list of their own.

**What your writing contradicts** — at most three items: where what they
wrote this week disagrees with the season, the slots, or itself. Dated
quotes. No energy, no drift, no scorekeeping.

**What you learned, what you are avoiding, what to do differently** — three
short paragraphs, or three bullets each. Direct. Given the longer arc in
[[Slowly Changing Context]], is this week's pattern high-leverage?

**Candidates for next week** — at most five, one line each with the date:
only things they captured that could become a slot or a card. This is what
Propose turns into slot and Backlog boxes. Below it, if any, **Instructions
you typed that nothing ran** — each `Aaa:` with nothing on disk, quoted
verbatim, marked `[unanswered]`; Propose routes every one.

**Questions to sit with** — one or two they would not ask themselves.

Then the closing line, verbatim:

```
*Next: open a session and say `run the Sunday pass`.*
```

Confirm the path in the task's `## Results` and nothing else. No habits
section (the read is off), no bottlenecks list, no labeled surfaced list, no
checkboxes.

## The Sunday session (interactive, in chat)

Trigger: they say `run the Sunday pass` (or `apply lens weekly-reflection,
Sunday session`) in a chat session on a Sunday. Read first, silently: this
Sunday's draft, their `# My Thoughts` in the WK note, [[Slowly Changing
Context]] §3–§4, the Board and the Backlog with `OpenAugi/AGENT/kanban.md`,
and the newest dated entry in `OpenAugi/Notes/System Janitor.md`. Then ask,
**one question per turn**, offering the choices as a question card where the
client has one (numbered options otherwise), in this order. Their thoughts
outrank the draft: where they disagree, offer what they wrote.

1. **Weekly focus** — one question per line, P0 · P1 · P2 · Self · Work.
   Quote the current line; offer *affirm*, one drafted full replacement
   (smallest next step first, from their words, dated), and *retire*. Write
   the ruled line into §4 before asking the next. The heading becomes
   `## 4. Weekly focus — week of <next Monday>` on the first change.
2. **The other sections** — only where the week contradicts them, at most
   three, one question each: *edit as drafted* or *leave*. Explain what the
   section is for in two lines when they ask; never assume they remember.
3. **Cards** — one question each, in this order: every card carrying an
   `aaa:` on either board (do what the words ask); every card in the Board's
   Triage column (one destination); any column over its cap (name the cards,
   ask which moves, nominate nothing); at most three Backlog adds from the
   draft's *Candidates for next week* (source date on each); every
   `[unanswered]` instruction (dispatch or drop). Perform each ruling before
   the next question, per `kanban.md`. An add to a Board column is fine when
   they approve it in the session. A card's `aaa:` gets ` (done <date>)`
   appended; their words are never deleted.
4. **Janitor** — one multi-select question listing this Sunday's unfixed
   items. Run the chosen ones as `system-janitor.md` says and rewrite each to
   `✓ fixed <date>`. Leave the rest.
5. **Close** — set `last_reviewed: <this Sunday>` and append one change-log
   line to [[Slowly Changing Context]] (`- **<date>** — Sunday pass from
   [[Weekly Reflection - <date>]]: <one clause per changed slot>`). Append
   `## Applied` to the draft: at most five lines — what changed in §4, cards
   moved and added, fixes run, anything ruled that could not be done. End in
   chat with one line and a *what's left* list, ideally empty.

Rules for the session: never more than one question in a turn, never a
list of decisions to answer at once. Nothing is written before its answer;
an answer of "leave" or "skip" means untouched. "Discuss" means a short
conversation, then the same question again. Never propose PMOC entries, a
pull from the Backlog, or a demotion unless they ask. If they stop midway,
the receipt names what was ruled and what was not, and a later `run the
Sunday pass` resumes there — a §4 heading already carrying next Monday's
date and a card `aaa:` already marked done are skipped.

## Rules

- The draft is a read: the sections above in that order, no checkboxes, no
  decisions. The decisions are the Sunday session, and it writes only what
  they ruled.
- Every `Aaa:` in the week that nothing ran appears in the draft as
  `[unanswered]` and comes up in the session as its own question. Never
  silently fold one into a nearby idea line.
- **The board is a menu, never evidence.** Untaken moves, unticked offers,
  repeated boards and missed board runs are never cited, counted or named.
  The only things the reflection may weigh are their own writing and what
  shipped. A tick they made that nothing executed is a system fault for the
  janitor, not a pattern about them.
- Prior reflections are for patterns across weeks, not for summarizing again.
- The draft never edits [[Slowly Changing Context]], the Board, or the
  Backlog. Only the session writes, and only what was ruled.
- Concept extraction from the week's notes (grouping ideas into new notes) is
  `parse-notes`, run separately.
