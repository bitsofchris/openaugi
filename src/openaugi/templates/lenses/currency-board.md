---
kind: engine
name: currency-board
description: >-
  The slim daily board — holds yesterday against the season and week you decided on Sunday so you execute instead of re-deciding, routes every `Aaa:` you typed, names the coding sessions that ended waiting on you, maps tasks it finds in your writing onto Kanban cards, and (experiment) spots what you can't see from inside the day and suggests one Short from what you learned. Kanban is the left-off; this board is not. Answered with checkboxes; untouched items are not carried.
scope: >-
  [[Slowly Changing Context]] first, every run. Then the `#status/active` PMOCs (newest dated entry each), the coding sessions since the last board (Claude Code ~/.claude/projects, Codex ~/.codex/sessions via scripts/session_harvest.py in the openaugi repo), their own writing since the last board (daily notes, PMOC/AMOC entries; default 72h) including every `Aaa:` instruction in it, [[Quiz Schedule - Foundations]] (the quiz review rows), and OpenAugi/Board/.board-state.json. For the Spotted section only: OpenAugi search over older notes, to link yesterday to what already exists. Never OpenAugi-generated prose except the board state and the previous board.
trigger: every 1d
target: >-
  note — OpenAugi/Board/YYYY-MM-DD - Board.md (a new dated note per run; yesterday's ticked boxes stay readable). Then overwrite OpenAugi/Views/View - Board.md with a link and embed of today's board.
---

# Currency Board

## Intent

[[Kanban]] already answers "where did I leave off and what is next"; the
board does not repeat it. The board does what Kanban cannot: it holds the
day to the plan decided calmly on Sunday (season and week in [[Slowly
Changing Context]]), so the weekday job stays capture and execute, never
decide in the moment ([[Decide the Plan Calm, Execute in the Moment]]); it routes every
`Aaa:` they typed, names a session that ended waiting on them, turns a task
it finds in their writing into an edit to a card (never a new checkbox to
come back to), and spots what they can't see from inside the day. Under 40
lines. The weekly reflection is the bigger pass and the only place
priorities move; this board never re-sets a slot.

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

**Nothing is carried.** Every section is built from the window alone. An
item left untouched is not carried to the next board; the one exception is
an `Aaa:` line, which is never dropped (Hard rules). On track is written
every day; on a day with nothing else, the board is On track alone. Sessions are read incrementally and never written to the vault:
`scripts/session_cards.py --dry-run --days 2` lists them, and only
transcripts changed since the previous board are opened.

## Anchor

Read [[Slowly Changing Context]] first. §4 says: **this week is the top card
in each column of the Board** (`_private/0-Current Focus/Kanban.md`), plus a
Work line. §3 is the season: its primary, what is supporting, and what is
parked. Read the Board's current order every run. The board does not render
the lanes; §3 and §4 feed On track (step 3b) and are the filter for Spotted
and for card suggestions. It never edits the Board except by a `yes` on a card
suggestion that showed the exact change.

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
3. **Collect the `Aaa:` instructions in the window, all of them.** An `Aaa:`
   line in their own writing is an instruction addressed to augi (capture
   grammar, `OpenAugi/AGENT/review-pass.md`) — the typed equivalent of them
   telling an agent to do something. List them with
   `scripts/instruction_list.py` in the openaugi repo (`--start` / `--end` set
   to the window's days); never find them by reading. There is no cap. Each
   one with no artifact on disk yet goes one of three ways, and every one
   gets a line under `## Your instructions`:
   - **Small and clear: do it on this run.** The instruction names the thing
     and the thing is one of: a note under `OpenAugi/`, a Backlog card with
     its source (per `kanban.md`, related links into the note), or a dated
     journal block on an existing MOC / AMOC / PMOC that links or records
     something. Do it, then report `✓ did: "<quote>" → [[artifact]]`. They
     comment with an `aaa:` under the line if it is wrong.
   - **Real work an agent can do alone** (a build, research, a rule edit):
     a **proposal** in `Augi could run these`, brief written from the
     instruction and the block it sits under, the block quoted on the `↳`
     line, key `aaa-<kebab of the instruction>` so a declined or dispatched
     one never returns. Report `→ proposed: "<quote>"`.
   - **Needs them** (a judgment, a priority, a Board card, an edit to their
     own file, a question only a conversation answers, or anything unclear):
     it waits for the Sunday session. Report `→ Sunday: "<quote>"`. Do not
     turn it into a judgment item.
   When unsure between the first and the third, it is the third.
   **The two `aaa` are different things.** `aaa:` under a board item is their
   feedback about *that item*. `Aaa:` in a daily note or a MOC entry is an
   instruction to *do something*. Never read one as the other.
3b. **On track.** Always written, first section, three lines at most, no
   boxes. Sources: §3 and §4 of [[Slowly Changing Context]], yesterday's
   writing, and yesterday's sessions by repo.
   - *Decided:* the season's primary and the §4 Work line, quoted, as one
     line. This is the plan already chosen, restated so the day starts
     from it; never a new choice.
   - *Yesterday:* where yesterday went, against that plan, from what they
     wrote and which repos had sessions: primary, supporting, or parked.
     Name a parked item touched (§3 "Parked") or a supporting lane well
     past its share (OpenAugi ~10%) once, quoting §3. Facts and their
     words only; no advice, no score, no streak.
   - *Captured, held for Sunday:* new ideas, priorities or plan changes
     they wrote yesterday that are outside the plan, one clause each,
     marked "for Sunday". They are not items and carry no boxes: the board
     never asks for a decision on the plan during the week. They go into
     the Sunday draft by the normal `Aaa:` / harvest path.
   If yesterday matched the plan, *Yesterday* says so in one clause and
   *Captured* is omitted.
4. **Sessions waiting on you.** From the transcripts changed since the last
   board, take only the end state. List a session only if it ended with
   something uncommitted, a question to them, or a finished artifact they
   have not opened. One line each: what waits, and `claude --resume <id>`
   (or the tmux name). Scheduled lens runs and task runs that finished
   clean are not listed. Nothing waiting, the section is omitted.
5. *(retired: no per-lane moves. Kanban is the next move.)*
6. **Quiz reviews due.** A row in [[Quiz Schedule - Foundations]] whose
   `due` is today or earlier with an unticked `done` box gets one line under
   Sessions waiting on you: the quiz file path and Prompt B from
   [[Lecture Learning Prompt]]. Shown on its due day and the day after,
   then not carried. Never invent a due date.
7. **Card suggestions, cap three, section omitted when empty.** A task they
   wrote in the window that is not yet on a card, or a task at
   `needs-input`, becomes a question: "this belongs on <existing card>" or
   "this is a new Backlog card". The item shows the exact edit (the sub-item
   line under that card on [[Kanban]], or the Backlog card with its source)
   and carries `yes` / `no`. `yes` is approval for that edit only
   ("Where you write" 2(b)); the next run makes it. Untouched is not
   carried. Never a done / not doing / someday item.
7b. **Spotted (experiment), cap three, section omitted when empty.** What
   they cannot see from inside the day, from yesterday's writing only, one
   line each with a `↳` to the notes and `yes` / `no`:
   - *Connects.* A block from yesterday that matches an older note
     (OpenAugi search, `echoes` intent). `yes` appends a link block to the
     older note's MOC / PMOC Journal by 2(a), or writes the link into an
     `OpenAugi/` note.
   - *Emerging.* A theme in two or more days of the window that is on no
     card. `yes` adds it as a Backlog card with its source.
   - *Remember.* Something they said worked or they learned, worth keeping.
     `yes` writes it into the note it belongs to by 2(a), or an `OpenAugi/`
     note.
   Untouched is not carried; `no` on a spot is not re-raised.
8. **Proposals — the delegation queue. `Aaa:` only, section omitted when
   empty.** Only `Aaa:` proposals from step 3. No per-card proposals: an
   agent step for a card comes from an `Aaa:` or the Sunday pass. The board
   never creates a card, note, or PMOC except by a `yes` on a card
   suggestion. The `↳` brief is the prompt the agent receives verbatim
   on `do`: three sentences, what, over what scope, producing what artifact.
   Then two labeled lines: `for: [[card note]]`, the note the receipt lands
   in, and `verify: <how they know it is done without reading the transcript>`
   (a test passing, a file at a path, a note with links). A card whose next
   step needs them shows in step 7 with the question named, and gets no
   proposal.
   Multi-day work restates its whole context every day it appears. Never
   re-offer a `declined` or `dispatched` proposal unless its output is on disk
   and the next step is different. Propose nothing rather than pad.
9. **Write the board** in the format below, then overwrite
   `OpenAugi/Views/View - Board.md` with a link and embed of it.

Other lenses feed sections, never add them: `open-loops` over the window for
card suggestions; `openaugi-state` and `content-pipeline` read from their last View.
Not run on a daily board: `chat-harvest`, `nuggets`, `idea-lineage`,
`cluster-weather`, `morning-briefing`, `ping-read`. `echoes` and `emerging`
feed Spotted (step 7b) over yesterday only. Those belong to their own triggers and the Sunday pass.
[[shorts-queue]] is not run here either: this board only does its intake
(step 6, "Shorts"); storyboards are on demand.

## The checkbox contract

No item on this board carries three boxes any more; every item is a
question with `yes` / `no`. The three-box form below stays only so old
boards still parse:

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
- **untouched** — not carried. Nothing on this board ages; what matters
  is on [[Kanban]] or comes back through their writing.

A judgment item that is a question carries boxes named by its answers
instead:

```
- [ ] yes: <what yes does, a few words>
- [ ] no: <what no does, a few words>
aaa: <optional one line, same channel>
```

- **yes** — recorded as `done`; the line becomes `✓ yes: …`.
- **no** — recorded as `not-doing`; the `aaa:` reason is honored.
- When the question is an edit to a file outside `OpenAugi/`, the item shows
  the exact change, and a yes — the box, or a yes in its `aaa:` — is
  approval for that change only ("Where you write" 2(b) in
  `augi-agent.md`).

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
`<details>` block near the bottom, never at the top.

````markdown
---
type: document
cssclasses:
  - board
description: Currency board for <date> — this week's slots, where each active thing left off, one move each.
created: <date>
---

# Board — YYYY-MM-DD (Weekday)

## On track

- **Decided:** <season primary, quoted> · <§4 Work line, quoted>
- **Yesterday:** <primary / supporting / parked, from their writing and session repos, in their words>
- **Captured, held for Sunday:** <idea>, <idea> — not today's to decide

## Your instructions

*Every `Aaa:` you typed since the last board. Small ones are already done; `aaa:` under a line to correct it. The rest wait for Sunday.*

- ✓ did: "<quote>" → [[<artifact>]]
- → proposed: "<quote>" (under *Augi could run these*)
- → Sunday: "<quote>"

## Sessions waiting on you

- <what waits: uncommitted / a question / an unopened artifact> — `claude --resume <id>`
- <quiz review N of unit NN, due <date>> — <quiz file path> · Prompt B

## On your cards (N of 3)

- **<task> → <existing card | new Backlog card>** — <the exact edit>
    ↳ <the block it came from>
    - [ ] yes: make that edit
    - [ ] no
    <!-- item:card-<kebab> -->

## Spotted (experiment)

- **<weekly goal | connects | emerging | remember>:** <one line, quoting them>
    ↳ [[<yesterday's note>]] · [[<the older note or card>]]
    - [ ] yes: <what yes does>
    - [ ] no
    <!-- item:spot-<kebab> -->

## Short from today (experiment)

- **<the question a viewer would search>** — from [[<source note>]]: "<their line>" `build`
    ↳ <the brief: question, three beats, storyboard stub in the video repo; session left open>
    for: [[<source note>]]
    verify: <the storyboard stub at its path>
    - [ ] do
    - [ ] no
    <!-- propose:short-<kebab> -->

## Augi could run these

*Your `Aaa:` proposals; only what an agent can do alone. `do` dispatches it on the brief as written; the receipt lands in the `for:` note. `no` means never offer it again.*

- **<The task, phrased as you would want to receive it cold>** `chip`
    ↳ <three sentences: what, over what scope, producing what artifact. Day two of a multi-day piece restates the whole context.>
    for: [[<the card note the receipt lands in>]]
    verify: <how they know it is done without reading the transcript>
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
- The first thing under the title is `## On track`. It never carries a
  box and never asks for a decision about the plan; plan changes wait for
  Sunday.
- Sessions are lines, never items, and never written to the vault.
- Every item has a `↳` line naming what to open. `↳` lines cap at 60 words;
  proposal briefs at 120. Overflow means the item is too heavy to answer
  cold and belongs as a scoped task.
- Never invent precision the source does not have.
- Never re-offer a `declined` proposal or re-propose a retired item.
- Never propose running the weekly reflection, the janitor, or any other
  `every` lens — those are scheduled and fire on their own. On a Sunday the
  board's first line under `## On track` says that today's reflection lands at ~06:30
  (`OpenAugi/Research/Weekly Reflection - <today>.md`), nothing more.
- **Monday only:** if `last_reviewed` in [[Slowly Changing Context]] is older
  than yesterday, the first italic line under `## On track` is *Sunday pass
  not applied — [[Weekly Reflection - <date>]] is waiting on its ticks.* One
  line, no proposal, no other change. That is the whole "is Sunday done"
  check.
- Never drop an `Aaa:` instruction from the window. Each one is done,
  proposed, or left for Sunday, and each has its line under `## Your
  instructions`. A typed instruction that no surface picked up is the one
  failure this board cannot have.
- The only states are `done` / `not-doing` / `someday` / `open`, each from a
  ticked box (`yes` is `done`, `no` is `not-doing`).
- Never write a "not yet seen" list, a drift section, a flags section, a
  since-yesterday narrative, or standing reminders. Unreviewed output lives
  on [[Dashboard]]; errands live on the Board's Household / Other column;
  drift and harvest belong to the Sunday pass.
- Never leave a receipt in the `Notes to augi` block.
- Mirror, not coach. Spotted reports what they wrote and what it connects
  to, and On track reports where the day went against the plan; neither
  advises. Structure changes stay on the Dashboard.
