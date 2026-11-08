---
kind: engine
name: weekly-reflection
description: >-
  The Sunday pass. Drafted at 06:30: what the week was about, a recap by lane, what kept coming up, what their writing contradicts, what they learned and are avoiding, an outside view (right what, and the bottleneck), the week's focus-block mix (Learn / Do / Write), candidates for next week, questions. Then the Sunday session: they open a chat and say `run the Sunday pass`; the agent goes through every typed instruction of the week, updates the context note, agrees the week's bottleneck with them, and only then asks about priorities, cards and janitor fixes — one question at a time, each applied as it is ruled. The only place priorities move.
scope: >-
  Every block from the last 7 days via the OpenAugi MCP (`recent` or `search` with a date window): daily notes, ideas, tasks, reflections — and every `Aaa:` instruction inside them. [[Slowly Changing Context]] §3 and §4. The Board and the Backlog (`_private/0-Current Focus/Kanban.md`, `Backlog.md`). The 2–3 prior reflections in `OpenAugi/Research/Weekly Reflection - *.md`, for patterns across weeks only. For the Sunday session, additionally: the week's daily notes and dated side notes in full, their own `# My Thoughts` in `_private/0-Current Focus/WK - YY-MM-DD.md` and this Sunday's entry in `OpenAugi/Notes/System Janitor.md`.
trigger: every 7d   # Sunday 06:30 local; the `## Run` anchor below is the schedule.
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
   applied the moment it is made — the week's instructions, the context
   note, a review of the week and what goes into it, then this week's
   cards, other card moves, a use review, janitor fixes.
4. **Done** is one fact: `last_reviewed` in [[Slowly Changing Context]] equals
   this Sunday. Monday's board says one line if it does not.

No boxes on the note, no second surface, no Claude-level skill: the process
is this file. The janitor's fixes, the boards and the context note are all
decided in that one session.

## Intent

The daily board holds each day to the plan and routes what was captured; this is where the week gets read. Honest,
not encouraging. No padding, no cheerful wrap-up. The value is what is hard to
see from inside the week, said in their own dated words. The draft is a read,
under ~850 words; the decisions come later and are checkboxes, never prose
proposals.

## Draft (the 06:30 run)

Gather the window, the boards, §3 and §4, the prior reflections. **List the
week's `Aaa:` lines with the script, never by reading:**
`scripts/instruction_list.py --start <Monday> --end <Sunday>` in the openaugi
repo. It finds every `aaa:` (any case) in a daily note or dated note of the
week, and under a dated `###` heading of the week in any journal note. Check
each against disk (a task in `OpenAugi/Tasks/`, a note, a routed block, a
board line that did it). The count in the draft must equal the script's
count. Then write, in this order, first
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

**The outside view: right what, and the bottleneck** — step outside the
week and look at it whole, against their goals (the North Star, §3, and
where they say they are going). The method: the right problem first
(Hamming), then the one constraint that caps the system (Goldratt). Three
short paragraphs, your own judgment; it may name something they have not.

*Last week's bottleneck.* Read the prior reflection's **Agreed** block.
Two lines: did they run the experiment, and did the constraint move,
judged from their writing and what shipped? If it did not move,
ask why before naming a new one; it may still be the bottleneck.

*Right what.* At a high level, is the one thing they are working on the
most important problem they could be working on? Or a nearby, easier one?
Say it plainly either way.

*The bottleneck.* What is slowing them down or keeping them from getting
where they want to go, that they may not see? It can be role-specific
(work, research, content) or about their life and self overall. Name the
one constraint that matters most right now, ask yourself why it is there
until you reach something they can change, and give the first step.

**Focus blocks** — the week's `Focus block` lines from the daily notes'
`# Habits`, weekend included: a table with one row per day — day, the
category (Learn / Do / Write), the thing, and the override reason when the
category was not §5's default for that day (blank line: `—`). Under it one
line: how the blocks split across the categories against the default, and
what the override reasons have in common, quoted. This is the one count
the draft carries; it describes the mix, never grades it. The categories:
[[Slowly Changing Context]] §5.

**Tornado** — the week's flare-ups, from the daily notes, the WK note's
thoughts and the capture notes: the moments they relitigated the path
(content vs career vs trading vs product, "what's it all for", "should I
pivot"). A table, one row per flare-up — date, the trigger as they wrote
it (a comparison, a scoreboard, a tired afternoon, a free Sunday),
*indulged* (they kept going: more planning, a new vision note, a new
project) or *caught* (they named it and went to the next step or rest),
and what broke it. No rows: `None logged.` Under it one line: what the
triggers have in common, quoted. Describes, never grades. The practice
it checks is in their Tornado Antidote note; the agreed rule (11/01) is
that the path question has one slot, the Sunday session, and everywhere
else the move is name it, log it, next step.

**Self** — what is working and what is not for their health,
energy, performance, happiness and well-being, and for the inner work (the
patterns that keep repeating, what they are growing into). Only from their
own writing this week — daily notes, the WK note's thoughts, and their self
and health area notes if touched. Two short lists, *Working* and *Not working*,
two or three bullets each, one dated quote per bullet; then one line: what
they may not be seeing. Family, mood and the hard things count when they wrote
them — say them plainly, without advice lists, counts or scores.

**Candidates for next week** — at most five, one line each with the date:
only things they captured that could become a slot or a card. This is what
the session turns into Backlog adds. Below it, **Every instruction you typed
this week** — a table with one row per `Aaa:` the script listed, oldest
first: date, the instruction quoted verbatim, and its state, one of `done`
(name the artifact), `partly` (say what is missing) or `[unanswered]`. Never
leave a row out because it was handled; the session goes through all of
them.

**Questions to sit with** — one or two they would not ask themselves.

Then the closing line, verbatim:

```
*Next: open a session and say `run the Sunday pass`.*
```

Confirm the path in the task's `## Results` and nothing else. No habit
scoreboard, streaks or rates (the Focus blocks and Tornado tables are the only counts), one bottleneck, never a list, no
labeled surfaced list, no checkboxes.

## The Sunday session (interactive, in chat)

Trigger: they say `run the Sunday pass` (or `apply lens weekly-reflection,
Sunday session`) in a chat session on a Sunday. Read first, silently: this
Sunday's draft, **the week's daily notes and dated side notes in full** (the
draft's quotes are not enough to discuss the week), their `# My Thoughts` in
the WK note, [[Slowly Changing Context]] §3–§4, the Board and the Backlog
with `OpenAugi/AGENT/kanban.md`, and the newest dated entry in
`OpenAugi/Notes/System Janitor.md`. Rerun `scripts/instruction_list.py` for
the week so the list is the script's, not the draft's. Then go in this
order, **one question per turn**, offering the choices as a question card
where the client has one (numbered options otherwise). Their thoughts
outrank the draft: where they disagree, offer what they wrote. Priorities
come third, not first: they are decided after the instructions and the
context note have been gone through. Every proposed wording (a context
edit, a lens or template line, a card clause) is shown in chat, before and
not inside the question: question-card previews do not reach them.

1. **Instructions** — open with a short read of the week and the full table
   of the week's `Aaa:` lines with each one's state. Then one question per
   line that is `[unanswered]` or `partly`, oldest first: the instruction
   quoted, what sits above it, and two to four concrete things to do with it
   (where it would land), one of them *drop*, one of them *discuss*. Do what
   they rule before the next question. Then every card carrying an `aaa:` on
   either board (do what the words ask; append ` (done <date>)` to the
   card's `aaa:`), and any unanswered `zzz:`. An `Aaa:` in a daily note is
   never edited: its ruling is recorded in the receipt.
2. **The context note** — only where the week contradicts
   [[Slowly Changing Context]], at most three sections plus the §4 **Work**
   line, one question each: *edit as drafted* (show the wording), *leave*,
   or *discuss*. Explain what the section is for in two lines when they
   ask; never assume they remember. Any note written as "input for Sunday"
   during the week brings its rulings here.
3. **This week's cards** — first one turn, no card: show the draft's
   outside view in full and ask if it is right and what would unblock
   it; talk until it is, and let it shape the card order. It ends with
   one agreed bottleneck and one experiment against it. Write them under
   the draft's outside view as **Agreed** (the bottleneck and the
   experiment), at most three lines, in their words. Then one question on
   **Into the week**, the §4 block in [[Slowly Changing Context]] that the
   board quotes every morning (at most three lines, each dated the Sunday
   it was set): show the block as it would stand, with the *Experiment*
   line already rewritten to the agreed experiment, each other line marked
   *keep*, *rewrite* (show the wording) or *drop*, and at most one new line
   drafted from what this session ruled (a practice or an affirmation in
   their words). A line set four or more Sundays ago is shown as *move to
   §7* (it is a habit now; §7 is not quoted daily) or *drop*, never
   *keep*. Write the block on their answer. Then one question on
   the focus blocks: show the draft's Focus blocks table and §5's default
   schedule, and ask *keep* or a shift drafted from the mix and the
   override reasons (which weekday moves to which category); on a shift,
   rewrite the §5 line on their yes. Then, only when the draft's Tornado
   table has rows, one question: show it and ask whether the triggers
   point at anything the week should change, or nothing (the usual
   answer). Record the answer under the table in one line. Then one
   question: the top card of each Board column,
   listed in §4 lane order, with *confirm* or a reorder drafted from their
   thoughts and from the outside view (which card rises, which drops, per
   column).
   Perform the reorder on the Board before the next question.
4. **Cards** — one question each, in this order: any column over its cap
   (name the cards, ask which moves, nominate nothing); at most three
   Backlog adds from the draft's *Candidates for next week* that step 1 did
   not already card (source date on each). Perform each ruling before the
   next question, per `kanban.md`. An add to a Board column is fine when
   they approve it in the session.
5. **Use review** — one question, before the janitor: at most three parts
   of the system (a board section, a reflection section, an *Into the
   week* line, a session step, a kind of AI output they said they don't
   read) with one dated quote each from their own
   writing this week about using, skipping or not needing it, and *cut*,
   *change* (show the wording) or *keep* for each. Their writing and their
   answer only: board ticks, untaken moves and missed runs are never the
   evidence. Nothing quoted: ask once whether anything went unused, and
   take *nothing* as the answer. A cut or change is performed in the
   session as a rule change: every AGENT file that states it, the OS note,
   the deck, and `sync_templates.py --write` in openaugi.
6. **Janitor** — one multi-select question listing this Sunday's unfixed
   items that are plain fixes, then one question for each item that needs a
   choice. Run the chosen ones as `system-janitor.md` says and rewrite each
   to `✓ fixed <date>`. Leave the rest.
7. **Close** — set `last_reviewed: <this Sunday>` and append one change-log
   line to [[Slowly Changing Context]] (`- **<date>** — Sunday pass from
   [[Weekly Reflection - <date>]]: <the top cards this week, one clause>`).
   Append `## Applied` to the draft: the top cards, the *Into the week* block as written, the focus-block default (kept or shifted), what the use review cut or changed, each instruction and how
   it was ruled (one line each), context edits, cards moved and added, fixes
   run, anything ruled that could not be done. Then write the week's
   **structured row** (below). End in chat with one line and a *what's
   left* list, ideally empty.

Rules for the session: never more than one question in a turn, never a
list of decisions to answer at once. Nothing is written before its answer;
an answer of "leave" or "skip" means untouched. "Discuss" means a short
conversation, then the same question again. Never propose PMOC entries, a
pull from the Backlog, or a demotion unless they ask. If they stop midway,
the receipt names what was ruled and what was not, and a later `run the
Sunday pass` resumes there — an instruction already in the receipt is
skipped, and `last_reviewed` already equal to this Sunday means only the
receipt is missing.

## The structured row and the synthesis

Each week leaves one structured record in a fixed schema, so patterns can be
read across weeks without rereading prose. At Close, build one file holding
their WK note followed by this Sunday's reflection (with its `## Applied`),
named like the WK note, and run the extract on it; then run the merge. The
extract skips a week it already has.

The **synthesis** is on demand only: they say `run the synthesis` (or
`apply lens weekly-reflection, synthesis`). Read the merged rows, analyze
them the way the first run did (priorities by season, energy / health /
mood, what keeps returning, claims checked against the raw notes), and
append a dated `### YYYY-MM-DD` entry at the top of the journal in
[[Weekly Reflection Synthesis]] — what changed since the last entry, not
the whole history again. Never scheduled, never run by the draft or the
session unless they ask. The draft may read the newest entry for patterns
across weeks.

## Rules

- The draft is a read: the sections above in that order, no checkboxes, no
  decisions. The decisions are the Sunday session, and it writes only what
  they ruled.
- Every `Aaa:` the script lists for the week appears in the draft's table
  with its state, and every one not `done` comes up in the session as its
  own question. Never silently fold one into a nearby idea line, and never
  find them by reading instead of the script.
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
