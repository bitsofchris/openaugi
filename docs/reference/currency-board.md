---
name: currency-board
description: The one surface that promises currency. A scheduled, unprompted board — where each thread left off, the next concrete moves, at most three things needing human judgment, and what drifted. Answered with checkboxes; the janitor turns those answers into state the next board honors.
---

# The Currency Board

Every other surface in OpenAugi is append-only truth: daily notes, blocks,
Augi Logs, recaps. None of them promise to be *current*. The board is the
single exception, and that exception is the whole design — **the vault never
has to be current; only the board does.** If a claim on the board is stale,
the board is broken.

## When to use

Read it at re-entry: start of day, sitting down to a thread, coming back from
an interruption. Answer it with checkboxes when something is done, will not
happen, or belongs to someday. Never operate it beyond that — the moment the
board needs tending, it has failed the way every previous review surface failed
(the `#human-review` backlog, nomination queues, Human-Todo lists).

## The problem it solves

Not "brief me in the morning." Reconstruction cost was never the pain —
re-entry *vacuum* was. What rushes into that vacuum is the tornado ("is this
even the right path?") or the inbox (other people's priorities). The board's
job is to occupy the first moment of re-entry with settled memory so neither
can. That is why it restates rather than asks: a menu of options at re-entry
is the failure mode, not the feature.

Voice is fixed at **memory + flags**. It reports what was written and what it
implies for a next action, quotes commitments back, and names a pattern once.
It never argues a path, coaches, or offers life direction.

## The three jobs

1. **Where each thread left off** — one memory line per lane (linked to its
   source), then 1–2 concrete next moves *derived from that line*, each with
   an activity chip (`focus` / `quick` / `read` / `ship` / `build` / `decide`)
   and a time estimate. A quiet lane gets its line and zero moves. Empty is fine.
2. **Needs your judgment — capped at three.** The most valuable diffs waiting on
   a human, ranked by container gravity, age, and whether recent writing mentions
   them. Three, not four. The other two hundred stay where they are; the cap is
   the feature.
3. **What drifted** — threads whose "active" status the user's own recent writing
   contradicts: an `#status/active` PMOC gone quiet, a commitment restated but
   untouched, two commitments that contradict each other within days. Evidence
   with dates, never verdicts. A previous flag is **withdrawn explicitly** when
   new writing answers it — showing the reversal is how the board proves it is
   current.

Plus standing reminders, at most three one-line flags, 2–3 tasks augi offers to
run, and the chat harvest (below).

Claude sessions themselves were **dropped from the board 2026-09-08** — no
status lines, no "resume this" roll-up. Only their *content* still gets a pass.

## The checkbox contract

Every move, judgment item and drift option carries the same three boxes, with
a stable key in an HTML comment:

```markdown
- **Pull the three insurance numbers from the benefits portal** `quick · 5m`
    - [ ] done
    - [ ] not doing
    - [ ] someday
    aaa: waiting for open enrollment in October
    <!-- item:insurance-numbers -->
```

| Answer | Effect |
|---|---|
| `done` | Item retires. |
| `not doing` | Retires permanently for this thread state; the `aaa:` reason is stored and honored literally by future boards. |
| `someday` | Parked; resurfaces only at weekly reflection. |
| *(untouched)* | Still open. `appearances` increments; on the **third** board the item gets one plain staleness line, never a repeated nag. |

Checkboxes are the buttons — the grammar chosen because it is the one review
gesture that has ever survived in this vault (the Augi Log's promote / good
match / bad match).

**Two channels for anything that isn't a checkbox**, because feedback about an
*item* and feedback about the *board* need different homes:

| Channel | For | Where it goes |
|---|---|---|
| `aaa: <why>` under an item | That item — why not, what's actually needed, a correction to carry forward | Stored on the item's key; future boards honor it literally |
| The `Notes to augi` callout | The board itself — wrong, missing, too vague, noise | Logged as `currency-board-note`, read as an instruction by the next build. **The janitor never edits this section** — his text is his. The log is the read marker: a line already logged for that board is not logged again, and a line added later is logged on its own. (Until 2026-09-10 the janitor overwrote his first line with `✓ noted <day>` and blanked the rest, which also stopped everything he wrote afterwards from ever being logged.) |

`aaa:` is already the review pass's instruction grammar, so neither needs a new
parser or a trip outside Obsidian.

## Every move carries its context

A move the reader cannot *start* is a failed move, however well phrased. Each
one gets a `↳` line naming where it came from and what to open — the source
note, the artifact, the session, the file path, the specific blocks.

The sharper rule is about honesty. When the source is vague, the board must
**name the unknown rather than invent precision**. The first board shipped an
item that read "Draft the work doc, send it to yourself" — crisp, actionable,
and false: the underlying note said only *"Neeed to I guess write the thing
first,"* a phrase with no antecedent anywhere in the vault. The board had
manufactured confidence out of ambiguity, which is the most expensive failure
available to it: a stale claim can be corrected, but a confident claim about
something the vault never named teaches the reader not to trust any of it.
Quote the vague phrase, say what could not be resolved, list the candidates,
and make *identifying the object* the first action.

## How it works

```
watcher drain tick → `trigger: every 1d` → OpenAugi/Tasks/TASK-<date>-currency-board.md
              → task_watcher picks it up → agent applies the currency-board lens
              → OpenAugi/Board/<date> - Board.md  (+ View - Board.md embed)

user ticks boxes → watcher sees the change → board_janitor.sync_board()
              → feedback-log.ndjson (append) + line rewritten to "✓ done"
              → rebuild_state() projects .board-state.json from board notes + log
              → next board READS state and never re-proposes what was answered

user ticks `do` on a proposal → board_janitor writes OpenAugi/Tasks/board-<date>-<key>.md
              → task_watcher hydrates it → an agent runs the brief as written
```

- **The lens** (`OpenAugi/AGENT/lenses/currency-board.md` in the vault) holds
  the intent, process and hard rules. Target family is `note` — a new dated
  note per run, so history is free and yesterday's answers stay readable.
  `View - Board.md` is overwritten each run as the stable pointer.
- **No new daemon.** Scheduling is a launchd job that writes a task file; the
  existing `task_watcher` runs it. Lens `every <period>` triggers stay dormant
  until scheduled-lens activation ships, and this deliberately does not wait
  for that gate.
- **`pipeline/board_janitor.py`** is the write-back half, a sibling of
  `echo_janitor.py` and wired into the same watcher cycle. An answered item is
  rewritten into a confirmation, so it is never processed again.
- **Plain markdown, not callouts** (2026-09-04). The board was built from
  `> [!board-lane]` callouts until it met the interaction it exists for: inside
  a callout every line carries a `> ` prefix, so there is nowhere to tap and
  type an `aaa:` line without fighting the editor — on a phone, effectively
  nowhere at all. Lanes are now `##` headings and the janitor reads the lane
  from the nearest heading above an item (`_HEADING_RE`). The callout shape is
  still parsed (`_LANE_RE`), so every board already on disk still answers.
- **Proposals are the outbound half of the channel.** Under `## Augi could run
  these` sit two-button offers — `do` / `no` — for work *augi* would do. `do`
  writes a pending task file into `OpenAugi/Tasks/` and the existing task
  watcher launches an agent on it; the proposal's `↳` brief is handed over
  verbatim, so what he read is what the agent gets, with nothing regenerated
  in between. `no` retires the offer permanently, and `declined` proposals are
  projected into state alongside items — the next board must never re-offer
  one. Dispatch is idempotent on the task filename: a proposal cannot launch
  the same agent twice.
- **"How does today compare to yesterday?"** was unanswerable from the note
  alone, so every board opens with a `## Since yesterday` section linking the
  previous board. `previous_board_summary(vault, before=<today>)` computes the
  split — what he answered, what is carried — from the previous note plus
  state.
- **State is a projection, not a source.** `OpenAugi/Board/.board-state.json`
  is rebuilt in full by `rebuild_state()` from two append-only sources: the
  dated board notes (who was proposed, when — `appearances` and `last_seen`
  are *counted* from these, never incremented) and
  `OpenAugi/Capture/feedback-log.ndjson` (every tick, with its reason). Delete
  the file and it rebuilds exactly.

  It was mutated in place until 2026-09-03, by this module *and* by board-build
  agent sessions. With two writers and no owner the counters drifted to 3 on a
  two-day-old board, falsely tripping the "third board" staleness flag on
  thirteen of sixteen items. **`board_janitor` is now the only writer, and the
  lens says the file is read-only to the board build.**
- **Pruning is bounded but lossless.** A bare `done` / `not-doing` older than
  `PRUNE_DAYS` (90) is dropped — the board reads a 72h window, so nothing that
  old is reachable from fresh writing. Anything `someday`, and anything
  carrying a `reason`, is durable and never pruned.
- **Why not SQLite,** given `store/sqlite.py` exists: that store holds what is
  *derived* from the vault (blocks, links, FTS, vectors) and is rebuilt by
  re-ingesting. Decisions exist nowhere else — they are primary data, and they
  stay greppable, diffable and repairable in git. The 2026-09-03 corruption was
  diagnosed with `git show HEAD:.board-state.json`; a binary blob would not
  have offered that.

## Setup and operation

Installed on the user's machine 2026-09-02. To reproduce elsewhere:

The board's cadence is a **lens field, not code**: `trigger: every 1d` in
`<vault>/OpenAugi/AGENT/lenses/currency-board.md`, plus a `## Run` section
naming the state to read first and the dedupe key. The watcher's drain tick
reads it and writes the day's task file — see
[lenses.md](lenses.md) "Scheduling".

```toml
# 1. the schedule — ~/.openaugi/config.toml, then restart the watcher
[tasks]
schedule_lenses = true
```

```markdown
# 2. the cadence — in the lens file, not in this repo
trigger: every 1d

## Run

Read `OpenAugi/Board/.board-state.json` before building.

dedupe: OpenAugi/Board/{date} - Board.md
```

```bash
# 3. build one now, without waiting for the tick — write the task file by hand
#    (or ask for it: `zzz: apply lens currency-board`)

# 4. rendering — install and enable the snippet once
cp src/openaugi/templates/board.css "<vault>/.obsidian/snippets/board.css"
#    Obsidian → Settings → Appearance → CSS snippets → enable "board"
```

**Migrated off launchd 2026-09-20.** The original install scheduled the board
with `~/Library/LaunchAgents/com.openaugi.board.plist` calling
`scripts/write-board-task.sh` at 06:00. Both are gone, as are
`com.openaugi.substack.plist` and `scripts/write-substack-task.sh`; the cadence
is the `trigger:` field on each lens file and nothing else. The order mattered
and is worth keeping written down, because each step depended on the one
before it: (1) the lens files get their triggers, (2) the config key goes on
and the watcher restarts, (3) one lens is watched firing from the tick, (4)
only then are the plists and scripts deleted. Doing 4 before 3 leaves no
schedule at all.

**What the trade costs.** `launchctl` fired at 06:00 whether or not anything
else was up. The drain tick only fires while `com.openaugi.up` is running, so a
stopped watcher means no board — and a stopped watcher is currently invisible.

`.obsidian/` is gitignored in the vault, so the snippet's versioned copy lives
at `src/openaugi/templates/board.css` in this repo — edit there, copy across.

- **Logs:** `~/.openaugi/logs/up.err` for the schedule (`Scheduled lens
  currency-board → ...`); the agent run itself lands in the task file's
  `## Results` and in its tmux session.
- **Rebuild today's board:** delete `OpenAugi/Board/<date> - Board.md`; the
  `dedupe:` line is what was suppressing the rerun, so the next tick builds it. State is keyed by item, not by file, so answers already
  recorded survive the rebuild and the new board still won't re-propose them.
- **Turn it off:** set `trigger: on-demand` in the lens file (or
  `schedule_lenses = false` to stop every scheduled lens at once). Nothing else
  in the system depends on the board existing.
- **Inspect or repair state:** `OpenAugi/Board/.board-state.json` is plain JSON,
  but it is generated — hand-edits are discarded on the next tick. To repair it,
  delete it and replay:

  ```python
  from pathlib import Path
  from openaugi.pipeline.board_janitor import rebuild_state
  rebuild_state(Path("<vault>"))
  ```

  `open_items(vault)` and `retired_items(vault)` are the read helpers. An
  unreadable state file logs an error and starts fresh rather than crashing the
  build, and an item whose `state` is outside the vocabulary is dropped on load
  with a warning.

## Hard rules

- **Never re-propose what state says is retired.** One violation costs the
  board's trust permanently — this is the exact failure that killed every
  previous surface, and the reason the janitor exists at all.
- Item keys are stable across runs; a returning item keeps its key so age
  survives. Never reuse a key for different content.
- Every left-off line links its source note, and every move carries a `↳`
  context line naming what to open. No unlinked claims, no unstartable moves.
- Never invent precision the source doesn't have — name the gap instead.
- Three judgment items maximum; two or three proposals, and none rather than
  padding.
- Never re-offer a proposal he declined — same rule, same reason.
- A proposal's brief is a prompt, not a summary: it is executed verbatim.
- Mirror, not coach. Drift states evidence; the human rules.
- Omit empty sections — an empty section is noise.

## Rendering

The board note carries `cssclasses: [board]`. The vault snippet
`.obsidian/snippets/board.css` styles the lane headings (and, for the older
boards, the callout types) and — the part that matters — renders the answer
boxes inline, so an item costs three lines of markdown but one line of
attention. A plugin `ItemView` that renders
the same markdown with real buttons and a lane/activity group-by toggle is the
natural next step; the markdown stays the truth either way.

## Chat harvest — `## Worth keeping` (added 2026-09-08)

A lot of thinking now happens in chat windows and dies there. Step 10 of the
board build applies the **`chat-harvest`** lens
(`<vault>/OpenAugi/AGENT/lenses/chat-harvest.md`) over *yesterday* and merges
one section into the board:

```
transcripts ──▶ scripts/session_harvest.py ──▶ chat-harvest lens ──▶ ## Worth keeping
(~/.claude/projects,   extraction, no judgment    judgment, routing,   one two-box
 ~/.codex/sessions)                               the drafted note      proposal
```

- **`scripts/session_harvest.py`** is the extractor and knows nothing about
  worth. It slices one local calendar day out of the transcript stores, keeps
  only human turns inside the window, drops harness-injected turns, subagent
  sidechains, trivial acknowledgements ("ok", "do it") and sessions augi
  dispatched to itself, and attaches the *longest* assistant message before the
  next human turn as context. Markdown by default, `--json` for machines.
  `python3 scripts/session_harvest.py --day 2026-09-07`.
- **The lens** does the judging: anchored on the user's own prompts (their
  questions are the record of what they were working out), the coding layer excluded,
  **at most one** candidate note per day, under 150 words, routed per
  `OpenAugi/AGENT/routing.md` to a new note or an append, with real links.
  Zero candidates is the common and correct answer.
- **The offer** uses the existing proposal grammar: `do` / `no` boxes and a
  `<!-- propose:keep-<slug> -->` marker, so the janitor and task watcher carry
  it with no new code. The full note text lives in the `↳` brief, which is
  handed to the agent verbatim — ticking `do` saves exactly what he read.

The lens never writes the note itself. The tick is what saves.

## Relationship to other passes

The board **surfaces** what other passes produce; it does not create decisions.
Structure changes (new containers, registrations, merges) stay nominations on
`View - Dashboard.md` — the board just picks the top three of them to show.
It replaces the July `morning-briefing` lens, which was mirror-only and had no
answer channel.
