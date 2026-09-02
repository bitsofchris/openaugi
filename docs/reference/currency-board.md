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

Plus a session roll-up (recent Claude sessions, one left-off line each) and at
most three one-line flags.

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
| The `Notes to augi` callout | The board itself — wrong, missing, too vague, noise | Logged as `currency-board-note`, marked `✓ noted`, read as an instruction by the next build |

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
launchd 06:00 → scripts/write-board-task.sh → OpenAugi/Tasks/TASK-<date>-currency-board.md
              → task_watcher picks it up → agent applies the currency-board lens
              → OpenAugi/Board/<date> - Board.md  (+ View - Board.md embed)

user ticks boxes → watcher sees the change → board_janitor.sync_board()
              → .board-state.json + feedback-log.ndjson + line rewritten to "✓ done"
              → next board reads state and never re-proposes what was answered
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
  `echo_janitor.py` and wired into the same watcher cycle. It is idempotent
  twice over: appearances increment once per board date however often it runs,
  and an answered item is rewritten into a confirmation so it is never
  processed again.
- **State** lives in `OpenAugi/Board/.board-state.json`; signals append to the
  shared `OpenAugi/Capture/feedback-log.ndjson` stream, so board answers
  accumulate next to echo feedback as training data for what the board should
  stop proposing.

## Setup and operation

Installed on Chris's machine 2026-09-02. To reproduce elsewhere:

```bash
# 1. the schedule — writes one task file a day at 06:00
cp ~/Library/LaunchAgents/com.openaugi.board.plist ~/Library/LaunchAgents/   # edit paths first
launchctl load ~/Library/LaunchAgents/com.openaugi.board.plist
launchctl list | grep com.openaugi.board                                     # "- 0 com.openaugi.board" = loaded, idle

# 2. build one now, without waiting for 06:00
scripts/write-board-task.sh        # honors $OPENAUGI_VAULT; no-ops if today's board or task exists

# 3. rendering — install and enable the snippet once
cp src/openaugi/templates/board.css "<vault>/.obsidian/snippets/board.css"
#    Obsidian → Settings → Appearance → CSS snippets → enable "board"
```

`.obsidian/` is gitignored in the vault, so the snippet's versioned copy lives
at `src/openaugi/templates/board.css` in this repo — edit there, copy across.

- **Logs:** `/tmp/openaugi-board.log` and `/tmp/openaugi-board.err` for the
  schedule; the agent run itself lands in the task file's `## Results` and in
  its tmux session.
- **Rebuild today's board:** delete `OpenAugi/Board/<date> - Board.md` and run
  the script again. State is keyed by item, not by file, so answers already
  recorded survive the rebuild and the new board still won't re-propose them.
- **Turn it off:** `launchctl unload ~/Library/LaunchAgents/com.openaugi.board.plist`.
  Nothing else in the system depends on the board existing.
- **Inspect or repair state:** `OpenAugi/Board/.board-state.json` is plain JSON,
  hand-editable. `board_janitor.open_items(vault)` and `retired_items(vault)`
  are the read helpers; an unreadable state file logs an error and starts fresh
  rather than crashing the build.

## Hard rules

- **Never re-propose what state says is retired.** One violation costs the
  board's trust permanently — this is the exact failure that killed every
  previous surface, and the reason the janitor exists at all.
- Item keys are stable across runs; a returning item keeps its key so age
  survives. Never reuse a key for different content.
- Every left-off line links its source note, and every move carries a `↳`
  context line naming what to open. No unlinked claims, no unstartable moves.
- Never invent precision the source doesn't have — name the gap instead.
- Three judgment items maximum.
- Mirror, not coach. Drift states evidence; the human rules.
- Omit empty sections — an empty section is noise.

## Rendering

The board note carries `cssclasses: [board]`. The vault snippet
`.obsidian/snippets/board.css` styles the four callout types and — the part
that matters — renders the three answer boxes inline, so an item costs three
lines of markdown but one line of attention. A plugin `ItemView` that renders
the same markdown with real buttons and a lane/activity group-by toggle is the
natural next step; the markdown stays the truth either way.

## Relationship to other passes

The board **surfaces** what other passes produce; it does not create decisions.
Structure changes (new containers, registrations, merges) stay nominations on
`View - Dashboard.md` — the board just picks the top three of them to show.
It replaces the July `morning-briefing` lens, which was mirror-only and had no
answer channel.
