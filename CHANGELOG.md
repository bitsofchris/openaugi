# Changelog

## Unreleased

**Privacy guard.** `scripts/check_private_vocab.py` runs as a pre-commit
hook at commit, commit-msg and push time. It refuses any file or commit
message containing a word from `<vault>/OpenAugi/AGENT/private-vocabulary.txt`
— the list lives in the vault so the guard itself can never be the leak — and
any notebook with cell outputs, which is how vault text once reached the
history. Matches are printed masked. `docs/reference/privacy-guard.md`.

**`ping_stats.py` is generic.** The check-in counter no longer carries its
author's field names and value lists as module constants. It parses any
`- [HH:MM] <kind>: key=value …` line, takes the scheduled and on-demand kinds,
the target key, the free-text key and the "absent" words as flags
(`--scheduled`, `--ondemand`, `--target`, `--free`, `--absent`), and discovers
keys and values from the data; the vault lens that invokes it names the
vocabulary. `docs/reference/pings.md` is rewritten around the generic grammar,
and AGENTS.md carries the rule: mechanism in the repo, vocabulary in the vault.

**The engine / personal line, made executable.** Every file in the vault's
`OpenAugi/AGENT/` folder now declares `kind: engine` or `kind: personal`.
Engine files are the operating system anyone who installs OpenAugi runs; each
has a template twin written by the new `scripts/sync_templates.py` (personal
regions — `%% personal %%` … `%% /personal %%`, the user's own rulings — are
stripped on the way out), and `openaugi init` copies every template that
declares `kind: engine` instead of a hand-kept list. Personal files never ship.
Twenty-three engine files ship now, up from fourteen: `kanban.md`, `pmoc.md`,
`routing.md`, `snapshot-agent.md` and nine more lenses, and the shipped
`augi-agent.md` and `review-pass.md` are current with the vault again (the old
templates predated the checkbox review signal). `src/openaugi/agent_files.py`
carries the kind, the region stripping and the template walk;
`tests/test_agent_files.py` and `tests/test_sync_templates.py` cover it, and
`test_lens_contract.py` now requires `kind: engine` on every shipped lens.
`scripts/session_cards.py` takes its vault from the openaugi config instead of
a literal path.

**One write-back module for every janitor.** `pipeline/writeback.py` now owns
what the board, the echo log and the routing rows each used to define for
themselves: the `feedback-log.ndjson` path (spelled out in four modules), the
append and the tolerant ndjson read, the UTC timestamp, and builders for the
shared `- [x] label` / `aaa:` grammar. The vocabulary stays per-surface —
`done / not doing / someday` is the board's, the routing verbs are routing's —
so the builders take strictness as arguments instead of imposing one regex on
surfaces that genuinely differ (a board line is inside a callout; a routing row
is not). Behavior is unchanged; the existing janitor tests are the proof.
Alongside it, the prose the engine generates and the docstrings it ships no
longer name or gender their user, and `tests/test_impersonal_engine.py` keeps
it that way — the first commit of the decision brief *Where a Personal
Surface's Code Lives*, whose criterion is that the engine knows nothing about
whose vault it is.
**The review signal is a checkbox now, not a tag.** Every agent-written note
used to carry `#human-review`, and accepting it meant opening the note and
deleting the tag — enough friction that the queue grew to 263 notes. Notes now
open with `- [ ] seen`, and the review surfaces
(`OpenAugi/Inbox - Agent Review.md`, the Dashboard's `## Review queue`) run a
Dataview `TASK` query, so ticking the box **from the queue page** writes the
`[x]` back into the source note. One click, no file opened. `augi_log`,
`echo_janitor` and `routing_janitor` write the box instead of the tag, and the
agent templates say never to write the tag again. Existing tagged notes keep
their tag: a second, legacy `FROM #human-review` query runs alongside the new
one until that backlog drains.

**Two board bugs, both about respecting what he wrote.**

*One instruction, one task.* A block's id is the hash of its whole raw text, so
appending a sentence to the paragraph a `zzz` line sits in deletes the block
and inserts a new one — with the instruction byte-for-byte unchanged. Dispatch
read that as a brand-new instruction and fired again: on 2026-09-09 one
research `zzz` launched three agents over five hours. Dispatch now **carries
the ledger row forward** when a successor's zzz text is identical to a
predecessor that already dispatched — it inherits the row and the task file, so
nothing new is written, the running session is left alone, and the chain stays
intact for the next edit. A *changed* instruction is still a real edit and
still supersedes (ARCHITECTURE.md § ZZZ Dispatch).

*His note is not a receipt slot.* The board janitor used to overwrite the first
line of the `Notes to augi` section with `✓ noted <day>` and blank the rest —
destroying what he wrote, and worse, the receipt then made every later note on
that board look already-processed, so nothing he added afterwards was ever
logged. The section is now **read-only**: the janitor logs it and never edits
it, and the append-only feedback log is the read marker — a line already logged
for that board is not logged again, a line added later is logged on its own,
and legacy `✓ noted` receipts already on disk are skipped rather than treated
as a terminator (docs/reference/currency-board.md).

**One reading queue, and it is Readwise Reader.** `openaugi reading push` ships
notes carrying `reading_queue: true` in their frontmatter to Reader as
documents authored by augi — rendered to HTML, `location: later` so they never
jump your own saves, at most two a day. The `url` is fabricated and stable
(`https://augi.local/note/<sha8-of-vault-path>`), which is the whole trick: it
makes a re-push an in-place update rather than a duplicate, and it comes back
as `source_url` on every read, so `openaugi reading harvest` can take a
highlight, walk to its parent document, and append it to the note that produced
it under `## Read in Reader — <date>` — your marks on augi's text, next to the
original, one artifact per idea instead of a detached mirror in a reference
folder. `openaugi reading status` shows what is flagged, what has been pushed
and what came back. Both commands are manual and both are safe to re-run;
nothing is scheduled and no agent sets the flag automatically yet, because the
open question is not technical (docs/reference/reading-queue.md).

**The board harvests yesterday's chats.** Thinking that happens in a chat
window used to die there. Step 10 of the currency-board build now applies a new
**`chat-harvest`** lens over yesterday's Claude Code / Codex transcripts and
merges one `## Worth keeping` section into the board: at most one candidate
note, under 150 words, anchored on the user's own prompts rather than the model's
answers, with the coding layer excluded and the destination (new note, or an
append to a named note) already decided. `scripts/session_harvest.py` does the
extraction and no judging — one local calendar day, human turns only, harness
noise and subagent sidechains and trivial acknowledgements and augi's own
dispatched sessions dropped, the longest reply per turn attached as context
(`--day`, `--days`, `--json`). The offer reuses the existing two-box proposal
grammar, so the janitor and task watcher carry it with no new code: the full
note text lives in the `↳` brief and ticking `do` saves exactly what he read.
Zero candidates is the common answer, and the section is omitted when there
are none. Sessions themselves stay off the board — content only, never a
status line (docs/reference/currency-board.md § Chat harvest).

**The Augi Log is the routing surface.** Every new human daily-note block
gets one row under `## Routing`: up to three proposed homes (`extend [[X]]`,
`link [[X]]`, `file under [[X]]`, `new note`), each with a one-clause why,
plus the fixed `memory` and `hold` boxes and an `aaa:` line for anything the
boxes can't say. Proposals come from the block's own `aaa:` hint, its
wikilinks, and where the nearest older writing lives, with an optional
temperature-0 judge. Nothing is applied until the day's master box
(`- [ ] process this log`) is ticked; then ticked rows apply as chosen,
untouched rows take the bold suggestion when augi is confident (his hint or
link; retrieval only for DB-only verbs), otherwise `memory`, which writes
nothing. `extend` inserts the block into the target newest-first under a
dated heading, wrapped so `undo` removes exactly that. Every resolution lands
in `feedback-log.ndjson` with the proposal, the choice and the block's
features, and `route.load_priors` turns that history into a bounded nudge on
later proposals — only ever reordering augi's own guesses, never his links or
hints. `openaugi routing stats` shows the tallies and the logs still waiting.
This is the answer to "blocks getting lost" that the review pass and the view
notes never became (docs/reference/augi-log-routing.md).

**Board state is a projection, not a mutable file.**
`OpenAugi/Board/.board-state.json` had two writers — `board_janitor.py` and the
board-build agent session the lens invited to read it. On 2026-09-03 the
counters drifted: `appearances` jumped 1 → 3 on thirteen of sixteen items while
`last_seen` stayed put, which is arithmetically impossible for a two-day-old
board with two board notes, and it falsely tripped the "third board — do it or
say not doing" staleness flag on nearly every item. A state named `withdrawn`,
outside the janitor's vocabulary, showed which writer did it.

`rebuild_state()` now regenerates the whole file from two append-only sources
that already existed: the dated board notes (`appearances` and `last_seen` are
*counted* from these) and `feedback-log.ndjson` (every tick, with its reason).
`sync_board` appends and rebuilds; it never edits state in place. Delete the
file and it replays exactly — the corruption above would have been a non-event.
Unknown states are dropped on load with a warning, bare `done`/`not-doing`
retirements older than 90 days are pruned (the window is 72h, so nothing that
old is reachable), and anything `someday` or carrying a `reason` is durable.
The lens now states the file is read-only to the board build, and that a
withdrawn drift flag is prose rather than a state. Nine new tests, including a
regression for the exact counter corruption.

**Blocks know who wrote them.** Every data block now carries
`metadata.provenance`: `human`, `ai`, or `reference`, resolved at ingest from an
explicit `provenance/*` tag, then `[vault.provenance_rules]` path globs, then the
AI and `source/*` tag rules. `search` and `get_context` take `provenance=[...]`
in every mode; semantic retrieval drops `[retrieval] exclude_provenance`
(default `["reference"]`) unless the caller names a provenance, so a synced
podcast no longer returns as twenty near-identical hits. `openaugi
backfill-provenance` stamps existing rows. This came out of an analysis run
that quoted forty model-written reflections back to the user as his own
writing because nothing at the query layer could tell them apart
(docs/plans/query-provenance-and-dates.md).

**`get_context` takes filters.** `after`, `before`, `tags`,
`exclude_path_prefix`, `include_path_prefix`, and `provenance`, applied to the
candidate pool before rerank. "What was I thinking about X in March" is now one
call instead of a browse plus a grep.

**Undated notes take the file's creation time, not its last edit.** The last
date fallback read `st_mtime`, which stamped every later edit of an undated MOC
onto its blocks. `st_birthtime` is preferred where the OS has it.

**The currency board — the one surface that promises to be current.**
Everything else in OpenAugi is append-only truth that never claims to be
up to date; the board is the deliberate exception, which is what makes the
promise keepable. Built unprompted at 06:00 into
`OpenAugi/Board/<date> - Board.md`: where each thread left off, 1–2 concrete
next moves per lane, at most three items needing human judgment, and what
drifted — threads whose "active" status the user's own recent writing
contradicts.

The half that makes it survive is the answer channel. Every item carries
three checkboxes (`done` / `not doing` / `someday`) and an optional
`aaa: <why>` comment line. `pipeline/board_janitor.py` — a sibling of
`echo_janitor.py`, wired into the same watcher cycle — turns those ticks
into `OpenAugi/Board/.board-state.json`, appends signals to the shared
`feedback-log.ndjson` stream, and rewrites answered lines into
confirmations. The next board reads that state and **never re-proposes a
retired item**, honoring a `not doing` reason literally. Untouched items
are carried with an `appearances` counter, so an item on its third board
earns one plain staleness line instead of a repeated nag.

Scheduling adds no daemon: a launchd job runs `scripts/write-board-task.sh`,
which writes a task file the existing `task_watcher` picks up — so this does
not wait on the dormant `every <period>` lens-trigger gate. The intent lives
in a vault lens (`OpenAugi/AGENT/lenses/currency-board.md`), not in code.
Supersedes the mirror-only `morning-briefing` lens. See
[docs/reference/currency-board.md](docs/reference/currency-board.md).

**One `zzz` instruction now dispatches one task.** Editing a `zzz` line used to
fire it again: block identity is the hash of the raw text including that line,
so finishing a half-typed instruction is a delete plus an insert, and the
post-ingest hook saw a brand-new block with a brand-new instruction. Writing
one sentence in two passes launched two agents on it.

Dispatch is now queued through a `zzz_queue` ledger. A zzz block becomes a task
only after it has sat unchanged for `tasks.zzz_settle_seconds` (default 120), so
drafts abandoned inside that window never become tasks at all. Past the window
the draft has already launched, so `run_layer0` now reports the entries it
deleted and a rewritten instruction *supersedes* its predecessor — the old task
file is marked `status: superseded` and its tmux session killed, leaving the
final wording as the only one running. Dispatch is also idempotent across
restarts now: a block id dispatched once never dispatches again.

`SQLiteStore.delete_record` was added so the ledger prunes settled rows.

## 0.2.1 — 2026-08-21

**Dependency pins so the package installs.** `mcp>=1.0` was an open range and
`mcp` 2.0.0 shipped, which moves `mcp.server.fastmcp` and drops `readOnlyHint`
from `ToolAnnotations` — so a fresh install resolved to a version that does not
import. Now `mcp>=1.0,<2`; lift that bound with a migration, not by widening it.

`httpx` is also declared explicitly. It is imported directly by
`auth/cloudflare.py` and had been arriving transitively via mcp 1.x; under
2.x that transitive dep became `httpx2` and the import broke. Depend on what
you import.

0.2.0 was tagged but never published — the release workflow gates on the type
check that this failure tripped.

## 0.2.0 — 2026-08-20

The release where OpenAugi stopped being a retrieval library with a write
hook bolted on, and became a loop: capture → route by rule → propose the rest
→ a human answers.

173 commits since `v0.1.0`. The themes, not the commits.

### One query engine

Every deterministic read semantic moved out of the MCP layer into `query/`:
a serializable `QuerySpec` plus an engine, with **three thin adapters over
it** — MCP, a new HTTP `/api/*` on the same daemon, and the CLI. The wire
format is pinned by a 40-case golden harness, so an adapter can't drift from
the engine without a test saying so.

**Saved queries** are markdown files with a `QuerySpec` in frontmatter, with
`-14d` / `today` / `$review-mark` resolved at run time.

### The review pass executes rules; it proposes judgments

The pass used to infer where a block belonged — classify by taxonomy, route
to the most specific match, and put anything low-confidence in a queue. That
queue reached 24 open items containing five actual decisions, and stopped
being read.

Routing now fires on **three deterministic rules only** — an `aaa:`
instruction naming a container, a `[[wikilink]]` to a registered one, a tag
matching a registration. Anything else stays where it was captured. That is
the expected outcome for most blocks and is not a backlog.

Everything requiring judgment — a new note, a merge, a registration — becomes
a **proposal** that is not acted on until a human answers it.

### Records: three generic tools instead of eight bespoke ones

Workflow state (what a run did, what awaits approval) lives in a **collection
store**: `write_record`, `list_records`, `update_record`. A collection is a
name the caller picks; OpenAugi validates nothing and has no schema for it.

This replaced eight named tools — 30% of the MCP surface — added for one
workflow, one of which hardcoded a list of legitimate routing rules. **Schemas
belong in the caller's prompt, policy in the caller's config, only mechanism
in a tool.** `docs/reference/records.md` carries the test for whether a new
tool belongs at all.

MCP surface: 14 tools → 22, despite that removal.

### Recaps are rows, not files

A container's synthesis is a `write_recap` row read through `get_view`, with
staleness reported honestly. Per-container `View - *.md` files are retired —
they were a stand-in from before anything could render a recap.

`docs/reference/recap-spec.md` defines what one contains: only what scrolling
can't give you — cross-window patterns, contradictions, unanswered questions,
what's gone quiet. Not what-moved, not member lists.

### A document is named by `augi_id`, not by its path

Container identity no longer depends on where a file sits, so moving or
renaming a note stops orphaning its membership edges.

### `#layer/bronze` retired

It meant "the user demoted this block" — a salience flag wearing a layer's
name, which collided with a layer model where bronze is the untagged default.
Removed along with its retrieval down-weighting and `[layers]` config.

### Ops — read this if you run the daemons

**`openaugi up` no longer serves MCP.** It is the background service: ingest,
watch, dispatch. `openaugi serve` is the interface. They have opposite
cardinality and this is why they are now firmly separate commands:

- **`up` is singleton**, enforced by an advisory lock. Two watchers over one
  vault both dispatch the same `zzz:`, so one instruction becomes two agents
  racing over one database.
- **`serve` is per-client.** Every stdio MCP client spawns its own; it takes
  no lock and any number can run.

**If a client (Claude Desktop, Codex, an editor) is configured to launch
`openaugi up`, point it at `openaugi serve` instead.** Pass `--serve` to `up`
only for single-terminal use with exactly one client.

Also: the shipped `review-pass.md` template is generic again — it had picked
up references to one user by name.

### Other

- `include_path_prefix` — the mirror of exclude, so a pass can reach one
  folder inside an otherwise-excluded tree.
- `anchor_id`, `ingested_at`, and `routed_to` projected on every block
  summary, so callers resolve anchors and see membership without a second call.
- Task dispatch finds its repo map under `OpenAugi/AGENT/` as well as the
  legacy location; it had been silently resolving zero repos.
- Daily-note template header preamble no longer becomes a block.
