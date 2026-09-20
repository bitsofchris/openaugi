---
kind: engine
name: currency-board
description: >-
  The terse daily board — this week's slots, where each active thing left off (from its PMOC and the coding sessions), one move each, at most three judgment items, at most two proposals. Read at re-entry, answered with checkboxes.
scope: >-
  [[Slowly Changing Context]] first, every run. Then the `#status/active` PMOCs (newest dated entry each), the coding sessions since the last board (Claude Code ~/.claude/projects, Codex ~/.codex/sessions via scripts/session_harvest.py in the openaugi repo), their own writing since the last board (daily notes, PMOC/AMOC entries; default 72h) including every `Aaa:` instruction in it, [[Quiz Schedule - Foundations]] (the quiz review rows), and OpenAugi/Board/.board-state.json. Never OpenAugi-generated prose except the board state and the previous board.
trigger: every 1d
target: >-
  note — OpenAugi/Board/YYYY-MM-DD - Board.md (a new dated note per run; yesterday's ticked boxes stay readable). Then overwrite OpenAugi/Views/View - Board.md with a link and embed of today's board.
---

# Currency Board

## Intent

Occupy the first moment of re-entry with settled memory. The board answers
"where did I leave off on each active thing, and what is the one next move,"
then gets out of the way. Under 60 lines. The weekly reflection is the bigger
pass and the only place priorities move; this board never re-sets a slot.

Voice: memory, not coaching. Report what they wrote and what it implies for a
next action, quoting their own commitments. Never argue a path, never present a
menu of directions, never comment on how they spent their time.

The board is the only surface that promises currency. If a claim here is
stale, the board is broken.

## Run

at: 06:00

What a scheduled run needs that a person asking for one would have said out
loud. Read `OpenAugi/Board/.board-state.json` before building anything — it is
step 1 of the Process below and it is the one file a cold run cannot infer.

dedupe: OpenAugi/Board/{date} - Board.md

Today's board existing is proof the day's run already landed, whatever the
schedule records say. A rerun is their call: delete the board.

## Anchor

Read [[Slowly Changing Context]] first. §4 is three named slots, P0–P2, set
by them on Sunday, plus Self and Work lines. Every left-off and every move is
framed against §4 and §3; a move that serves neither is not a move for this
week. If §4 looks superseded by their writing, one italic line under `This
week` says so; the board does not act on it. §4 is rewritten by the Sunday
pass's Apply step (`lenses/weekly-reflection.md`), never by this board.

## Process

1. **Read `.board-state.json`, and only read it.** It holds each item's
   state (`open` / `done` / `not-doing` / `someday`), `reason`, and
   `appearances`, and each proposal's state. Never re-propose `done`,
   `not-doing`, or `declined`. `someday` resurfaces only at the weekly
   reflection. Honor `reason` text literally. `board_janitor.py` is the
   file's only writer.
2. **Gather the window.** Their writing since `last_run` (fall back to 72h) by
   ingest time: daily notes, PMOC and AMOC entries. The previous board's
   `aaa:` lines and ticks.
3. **Collect the `Aaa:` instructions in the window.** An `Aaa:` line in their own
   writing is an instruction addressed to augi (capture grammar,
   `OpenAugi/AGENT/review-pass.md`) — the typed equivalent of them telling an
   agent to do something. Every one in the window that has no artifact on disk
   yet becomes a **proposal** in `Augi could run these`, with its brief written
   from the instruction and the block it sits under, and the block quoted on the
   `↳` line. Its key is `aaa-<kebab of the instruction>`, so a declined or
   dispatched one never returns. This is the board's only source of proposals it
   did not derive from a lane. More than two in a window: carry the two whose
   blocks are closest to the §4 slots and leave the rest to the weekly
   reflection, which routes all of them.
   **The two `aaa` are different things.** `aaa:` under a board item is their
   feedback about *that item*. `Aaa:` in a daily note or a MOC entry is an
   instruction to *do something*. Never read one as the other.
4. **Build the left-off for each active thing.** Active things are the
   `#status/active` PMOCs on [[Dashboard]] plus the §4 slots. For each, read
   two sources together: the PMOC's newest dated `###` entry, quoted, and the
   coding sessions since the last board that touched it, matched by repo and
   title (a session's repo or vault name maps to the PMOC it serves).
   From a session take only the end state: the last thing they asked, what
   finished, what is uncommitted or waiting on them. Name the session on the
   `↳` line so they can resume it. A session is a source for the left-off,
   never a section, a thread, or a status line.
5. **One move per active thing.** Derived from the left-off, never invented.
   A move names a physical action, carries an activity chip (`focus` /
   `quick` / `read` / `ship` / `build` / `decide`) and a time estimate, and
   carries its context on a `↳` line: what to open to start, the note, path,
   or session. A quiet lane gets its left-off line and no move. When the
   source is vague, quote the vagueness and make identifying the object the
   move; never manufacture precision.
6. **Quiz reviews due.** Read the table in [[Quiz Schedule - Foundations]].
   A row whose `due` is today or earlier and whose `done` box is unticked is
   due. When one is, the Fundamentals lane's move *is* that review:
   `quick · 10 min · Quiz review N of unit NN`. The `↳` line gives the quiz
   file path and how to start: Claude Code in `~/repos/deep-learning`,
   Prompt B from [[Lecture Learning Prompt]]. It replaces the move step 5
   would have derived for that lane; the left-off line stays. Nothing due,
   this step writes nothing.
   Only rows in the note count; never invent a due date.
7. **Judgment, cap three, section omitted when empty.** Decisions that block
   something and that only they can make: blocked builds, tasks at
   `needs-input`, a registration or ingest check. Ranked by age and by
   whether their recent writing mentions it. Each states in one line what
   ticking means.
8. **Proposals, cap two, section omitted when empty.** Work augi would do:
   first the `Aaa:` instructions from step 3, which take the slots ahead of
   anything derived, then work derived from the priorities already on the board. The `↳` brief is the
   prompt the agent receives verbatim on `do`: three sentences, what, over
   what scope, producing what artifact. Multi-day work restates its whole
   context every day it appears. Never re-offer a `declined` or `dispatched`
   proposal unless its output is on disk and the next step is different.
   Propose nothing rather than pad.
9. **Habits: parse yesterday, then show it back.** First apply
   [[habit-parse]] for yesterday (it writes `OpenAugi/Habit Log/<yesterday>.md`
   from the daily note's `# Habits` section, or nothing if the section is
   absent or the file already exists). Then, under `This week`, one line
   naming the active habits from the Self column of [[Kanban]] and yesterday's
   tick count (`2/3 yesterday`, or `no log yesterday`), followed by an embed
   of the log file — `![[OpenAugi/Habit Log/<yesterday>]]` — so they can check
   the parse and fix a row in place. Never interpret, never streak-count,
   never comment on a miss. The journal prose stays theirs.
   When the newest `OpenAugi/Notes/YYYY-MM-DD - Habit Read.md` carries
   unanswered questions (a `<!-- habit:... -->` block with neither box
   ticked), repeat those blocks verbatim under the same habits line, under
   the heading `*From the habit read (<its date>):*`, until they tick one or
   [[habit-read]] drops it. Carry the boxes; add nothing to them.
10. **Write the board** in the format below, then overwrite
   `OpenAugi/Views/View - Board.md` with a link and embed of it.

Other lenses feed sections, never add them: `open-loops` over the window for
left-off lines; `decision-audit` for ranking judgment items by real decision
language; `openaugi-state` and `content-pipeline` read from their last View.
Not run on a daily board: `chat-harvest`, `nuggets`, `echoes`,
`idea-lineage`, `cluster-weather`, `emerging`, `morning-briefing`,
`substack-batch`, `ping-read`. Those belong to their own triggers and the
Sunday pass.

## The checkbox contract

Every move and judgment item carries three boxes:

```
- [ ] done
- [ ] not doing
- [ ] someday
aaa: <optional one line, free text, read by the next board>
```

- **done** — it happened; the item retires.
- **not doing** — retires permanently for this thread state; the `aaa:`
  reason is stored and honored.
- **someday** — parks it; resurfaces only at the weekly reflection.
- **untouched** — still open; the board carries it and its age increments.
  On its third board the item gets one plain clause ("third board, do it or
  say not doing") unless the window shows them working it or its `aaa:` says
  to keep it. Staleness is neglect, not duration.

Proposals carry two boxes:

```
- [ ] do
- [ ] no
aaa: <optional one line, same channel>
```

- **do** — the janitor writes a pending task file into `OpenAugi/Tasks/`;
  the task watcher launches an agent on the brief as written. The line
  becomes `✓ do → \`<task file>\``.
- **no** — retires the offer permanently; the reason is honored.
- **untouched** — a suggestion, not a debt; re-offer only if still right.

Checkboxes are the buttons. `aaa:` under an item is feedback about that
item and attaches to its key. The `Notes to augi` block at the bottom is
feedback about the board itself; the janitor logs it and the next build
treats it as an instruction. The board never writes into that block.

## What persists

Ground truth is two append-only things: the dated board notes in
`OpenAugi/Board/` and `OpenAugi/Capture/feedback-log.ndjson`. The state file
is a projection rebuilt from them; delete it and it rebuilds exactly. Bare
`done` / `not-doing` decisions age out after 90 days. Anything `someday`,
and anything carrying a `reason`, is durable.

## Format

Plain markdown, headings not callouts, so every line can be tapped into
on a phone. Empty sections are omitted. Bookkeeping lives in a collapsed
`<details>` block near the bottom, never at the top. Lanes render in §4
priority order.

````markdown
---
type: document
cssclasses:
  - board
description: Currency board for <date> — this week's slots, where each active thing left off, one move each.
created: <date>
---

# Board — YYYY-MM-DD (Weekday)

## This week

- **P0 · <lane>** — <§4 line, quoted>
- **P1 · <lane>** — <§4 line, quoted>
- **P2 · <lane>** — <§4 line, quoted>
*<optional: one rule of theirs that applies today>*

## <Lane name> · <three-word state>

*Left off: [[PMOC]] <date> — "<their newest dated entry, quoted>." Session `<repo · title>` ended with <finished / uncommitted / waiting on them>.*

- **<The one concrete next move>** `chip · time`
    ↳ <what to open to start: [[note]] · path · session>
    - [ ] done
    - [ ] not doing
    - [ ] someday
    <!-- item:<stable-kebab-key> -->

## Needs your judgment (N of 3)

*Decisions that block something and that only you can make. `aaa:` reshapes an item instead of answering it.*

- **<The specific yes/no>** — <one line of what ticking means>
    ↳ <what to open>
    - [ ] done
    - [ ] not doing
    - [ ] someday
    <!-- item:<key> -->

## Augi could run these

*`do` dispatches an agent on the brief as written; `no` means never offer it again.*

- **<The task, phrased as you would want to receive it cold>** `chip · est`
    ↳ <three sentences: what, over what scope, producing what artifact. Day two of a multi-day piece restates the whole context.>
    - [ ] do
    - [ ] no
    <!-- propose:<stable-kebab-key> -->

<details>
<summary>Sources and bookkeeping</summary>

*Built HH:MM from N blocks · window <start> → <end> · previous board [[YYYY-MM-DD - Board]]*
*Your `aaa:` lines yesterday: "<quote>" → <what the board did>. (one clause each)*
*Since last Sunday: <one line from the [[Slowly Changing Context]] change log>*
*Sources: <the notes read this run>*

</details>

## Notes to augi

*Anything that isn't a checkbox. Type below the marker. For feedback about one item, use its `aaa:` line.*

<!-- board-note -->
````

The markers are load-bearing: `<!-- item:<key> -->` under every three-box
item, `<!-- propose:<key> -->` under every proposal, `<!-- board-note -->`
once at the bottom. The janitor finds items by those comments and the lane
by the nearest heading above.

## Hard rules

- Item keys are stable across runs; a returning item keeps its key, a new
  item gets a fresh kebab key from its substance. Never reuse a key for
  different content.
- Never write `.board-state.json`.
- The first thing under the title is `## This week`.
- Every left-off line links its source note and names its session when
  there is one. Sessions are never items.
- Every move has a `↳` line naming what to open. `↳` lines cap at 60 words;
  proposal briefs at 120. Overflow means the item is too heavy to answer
  cold and belongs as a scoped task.
- Never invent precision the source does not have.
- Never re-offer a `declined` proposal or re-propose a retired item.
- Never propose running the weekly reflection, the janitor, or any other
  `every` lens — those are scheduled and fire on their own. On a Sunday the
  board says under `## This week` that today's reflection lands at ~06:30
  (`OpenAugi/Research/Weekly Reflection - <today>.md`), nothing more.
- **Monday only:** if `last_reviewed` in [[Slowly Changing Context]] is older
  than yesterday, the first italic line under `## This week` is *Sunday pass
  not applied — [[Weekly Reflection - <date>]] is waiting on its ticks.* One
  line, no proposal, no other change. That is the whole "is Sunday done"
  check.
- Never drop an `Aaa:` instruction from the window without either
  proposing it or leaving it for the weekly reflection. A typed
  instruction that no surface picked up is the one failure this board
  cannot have.
- The only states are `done` / `not-doing` / `someday` / `open`, each from a
  ticked box.
- Never write a "not yet seen" list, a drift section, a flags section, a
  since-yesterday narrative, or standing reminders. Unreviewed output lives
  on [[Dashboard]]; errands live on the Board's Household / Other column;
  drift and harvest belong to the Sunday pass.
- Never leave a receipt in the `Notes to augi` block.
- Mirror, not coach. Nominate nothing new here; structure changes stay on
  the Dashboard.
