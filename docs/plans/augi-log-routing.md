---
name: augi-log-routing
description: The Augi Log becomes the capture/routing surface. Every new human daily-note block gets one routing row (new note / extend / link / file under / memory / hold) with a proposed target; Chris answers with checkboxes and aaa: hints, unanswered rows auto-apply after N days, and every answer is logged so later proposals are biased by his history. Replaces the dead review pass and the failed view-note / mobile render attempts.
---

# Augi Log routing

**Status: DECIDED 2026-09-03, implementation starting.** Branch `feat/augi-log-routing`. Decisions are recorded inline as **Decided:** lines.

## Where this came from

Chris, daily note 2026-09-02:

> The augi log - it's the routing decisions as I go. With optional proactive
> lenses or suggestions but first it's the capture / routing surface.
> Basically I hit checkbox on routing suggestions ... You can just suggest
> where to stick things or what to merge with based on my hints and I confirm
> it there. If I don't respond in X day or something then just do it
> automatically. Here's how you learn me, how I route and keep things useful.

> Is this block a new note? Extend an existing one? Or link to something? Or
> part of project/ area? Or just a memory I save? Or a working thought
> building to something (bronze?)?

The pain behind it, raised 08-31, 09-01, 09-02 and in the PMOC on 09-01:
"dumping a lot of blocks without clear labels into the daily notes ... don't
really have the review pass / routing here, or a good way to render it."
Two prior answers died: the review pass (a batch agent pass, last run
2026-08-21, needed a render surface that never worked) and View notes / the
mobile app as that surface. The 09-01 note names the rule that survived
instead: "use checkboxes as the buttons to communicate with agents in
Obsidian, easy."

## What changes, in one paragraph

The Augi Log stops being an echo-only debug log and becomes the place
routing happens. For every new **human** daily-note block the watcher
appends a routing row: the block snippet, one proposed verb and target with
a one-clause why, up to two alternates, and the fixed fallbacks *memory* and
*hold*. Chris ticks boxes, or writes an `aaa:` line, or does nothing. When he
ticks the day's master box (`process this log`) a janitor applies the whole
log: ticked rows as chosen (DB routing link, and for *extend* an append into
the target note under a dated heading), untouched rows to the top suggestion
when confident, else *memory*. A log he never ticks is never applied. Every answer, correction and auto-apply lands in
`feedback-log.ndjson` with the block's features, and the proposer reads that
history to bias the next proposal. The morning board reports how many rows
are waiting.

## What stays the same

- Truth is append-only. The daily note is never edited, never trimmed. A
  block is never deleted or moved by this feature.
- Routing membership is the existing `routed_to` link (`apply_routing`).
  Views, `get_members`, `get_view` keep working unchanged.
- Echo keeps its section, its boxes and its janitor. Echo is the "you
  thought this before" lane; routing is the "where does this live" lane.
- The board's three-box grammar (done / not doing / someday + `aaa:`) is the
  model for the routing row: checkboxes are buttons, `aaa:` is the comment
  channel, the janitor rewrites answered lines to `✓ …` so nothing is
  processed twice.
- `aaa:` stays unparsed at ingest. The proposer reads it from the block
  text; nothing else changes about the grammar.

## The six design questions

Each was discussed with Chris on 2026-09-03; the decision is recorded under
each question.

### 1. Verbs and checkbox rows

Vocabulary (Chris's list, one word each):

| Verb | Means | Apply writes |
|---|---|---|
| `new note` | this block starts a note of its own | `OpenAugi/Notes/<slug>.md` (the echo janitor's promote shape: context header + dated log), `routed_to` link from the block to it |
| `extend [[X]]` | this belongs in an existing note's running log | append the block under `### YYYY-MM-DD` in X + `routed_to` link (see Q3) |
| `link [[X]]` | related, but lives where it is | `routed_to` link only |
| `file under [[MOC]]` | belongs to an area / project container | `routed_to` link to the container (what the review pass did) |
| `memory` | life-log, stays in the daily note | ledger only, `augi_tags: ["memory"]` so retrieval can filter it |
| `hold` | working thought, not ready | ledger state `held`; re-proposed after 14 days or when a related block lands |

Row shape, recommended (option B: one box per suggestion, capped):

```
<!-- route:<block_id> -->
### route "First day of school. Morning cold plunge…"
*[[2026-09-03]] · 2026-09-03*
- [ ] **extend [[PMOC - Audacity to take Action - Season 2 - Q2 2026]]** — same thread as your 09-01 entry there
- [ ] file under [[AMOC - OpenAugi Main]]
- [ ] new note
- [ ] memory
- [ ] hold
aaa:
```

Rules: at most three proposed lines (top suggestion bold, alternates
plain), `new note` shown only when proposed, `memory` and `hold` always
present. An `aaa:` line overrides everything ("aaa: link [[X]]",
"aaa: memory"), so anything the boxes cannot say still has a channel.

**Decided 2026-09-03: option B.** Chris's framing: the row is augi's
suggestion, the boxes are his answer, and `aaa:` is how he speaks to augi
directly when the boxes cannot say it ("I correct you with aaa:").

### 2. When rows apply: a master box, not a timer

**Decided 2026-09-03.** Chris asked for a top-level checkbox that gates
processing, and chose it over the per-row timer for v1.

- Each day's Augi Log carries one master box at the top of the Routing
  section: `- [ ] process this log`.
- Ticking individual row boxes only records intent. Nothing is applied until
  the master box is ticked.
- Ticking the master box applies the whole log at once: ticked rows as
  chosen; untouched rows resolve to the top suggestion when it is
  *confident*, otherwise to **memory**, which writes nothing to any note.
  Confident means: the block carries an `aaa:` hint naming the target, or
  an explicit wikilink to it (any verb); or the top candidate came from
  retrieval, its z-margin over the runner-up clears `confident_margin`,
  **and the verb only writes a DB link** (`file under`, `link`). Retrieval
  alone never writes into one of his notes (`extend`) or mints one
  (`new note`) — those need his tick or his hint. Implemented in
  `route.is_confident` (step 3). The section is then rewritten to `✓ …`
  confirmations, each with a `- [ ] undo` box the janitor honors.
- A log that is never ticked is left alone. Nothing runs. The board reports
  the count of unprocessed logs so the pile stays visible (see Q6).

Why: one tick is one decision per day, made by him, instead of a background
clock routing while he is away. Checkboxes are the buttons. The known cost
is that unprocessed logs can accumulate (the review-queue failure mode);
the board count is the first mitigation, and a timer can be added later as
an opt-in (`[routing] auto_apply_days`, off by default) with the master box
still the way to process early.

Config: `[routing] enabled = true`, `[routing] confident_margin`.

### 3. What "apply" writes for extend

This is the one place the feature touches a human-authored note, so it needs
Chris's explicit yes. Three options:

- **(a) DB link only.** Safest, and it is exactly what the review pass did.
  It failed because nothing rendered the result where he reads.
- **(b) Append the block text into the target** under a `### YYYY-MM-DD`
  heading (the way he writes in PMOC / AMOC notes: `PMOC - Audacity…` has
  `### 2026-09-01`), with a trailing source line `— from [[2026-09-03]]` and
  an idempotence marker `<!-- augi:routed <block_id> -->`. Append-only,
  never deletes, never rewrites his text, and the words are his so
  provenance stays human. Downside: two copies; a later edit in the daily
  note does not propagate.
- **(c) Append an Obsidian block embed** `![[2026-09-03#^<id>]]`. One copy,
  but it requires writing a `^id` anchor into the daily note, which edits a
  human note, and embeds render poorly on mobile.

**Decided 2026-09-03: (b), with placement rules.** Chris: "typically look
for the Journal H1 or the other H3s - I like the most recent entry to be on
top." So the append is really an *insert*:

1. Find the note's `# Journal` H1 if it has one; otherwise the first run of
   `### YYYY-MM-DD` headings anywhere in the note. Verified 2026-09-03 across
   AMOC Health / Finances / OpenAugi Main, the Q2 PMOC and the Dream Journal:
   all newest-first; a heading may carry a suffix (`### 2026-01-14 - Archived`),
   so the matcher keys on the leading date only.
2. If a `### <today>` heading already exists in that run, insert the block
   at the end of that day's section (his own words for the day stay first).
3. Otherwise insert a new `### <today>` heading **above the most recent
   dated heading** (newest first), directly under `# Journal` when that is
   the anchor.
4. If the note has neither a Journal H1 nor dated H3s, append a `# Journal`
   H1 and the dated heading at the end of the file, and say so in the
   confirmation line.

Each inserted block ends with `— from [[<daily note>]]` and is wrapped in
`<!-- augi:routed <block_id> -->` … `<!-- /augi:routed -->` markers so undo
removes exactly what was inserted and nothing else. Gated by
`[routing] extend_writes_note = true`; off falls back to (a).

### 4. Learning

Every resolution appends one record to `OpenAugi/Capture/feedback-log.ndjson`:

```json
{"ts": "...", "source": "routing", "block_id": "...", "signal": "accepted|corrected|memory|hold|auto|undo",
 "proposed": {"verb": "extend", "target": "PMOC - …"}, "chosen": {"verb": "file under", "target": "AMOC - …"},
 "features": {"folder": "_private/0-Fleeting-Inbox", "tags": ["area/openaugi"], "nearest": ["PMOC - …", "AMOC - …"], "had_aaa": true, "day_of_week": 3}}
```

The proposer reads the log once per cycle and computes two deterministic
priors: per-target acceptance rate (accepted / proposed) and per-verb rate
per folder. These add a bounded bonus to the candidate score. No model, no
training; the bias is legible and can be dumped with a CLI
(`openaugi routing stats`). Phase 2, after the row and janitor ship.

**Decided 2026-09-03: as above.** Log everything from the first commit;
priors land in step 7. `aaa:` lines are per-row instructions only; standing
rules are not parsed from them in v1.

### 5. Placement in the Augi Log

Recommendation: a **`## Routing` section at the top**, above the echoes,
with the existing `## Quiet` and heartbeat staying last. File shape becomes
`[header][## Routing][## Echoes][## Quiet][heartbeat]`. Reasons: routing is
now the primary job and should be the first thing on screen; both janitors
key on section boundaries, and interleaving per block would make the echo
regexes fragile. The header description changes from "ephemeral, delete
freely" to say that unanswered routing rows still resolve on their own, so
deleting the file is still safe.

The section writer becomes a small shared module (`pipeline/augi_log.py`)
that echo and routing both use, replacing `_write_sections` in `echo.py`.

**Decided 2026-09-03: Routing on top.** The master box sits directly under
the `## Routing` heading; the existing echo rows move under a `## Echoes`
heading (the echo janitor's regex boundaries gain that heading).

### 6. The morning board

**Decided 2026-09-03.** The routing ledger is a `routing_queue` collection
in the `records` table (the `zzz_queue` pattern: idempotent across restarts,
prunable), and each log's master-box state is recorded there when the log
is written and when it is processed. The board lens gains one rule: if any
Augi Log has an unticked master box, the *Needs your judgment* callout
carries a single line, "N logs waiting to be processed", linking the oldest
one, and it takes one of the three judgment slots. The board never lists
rows; the log is the surface. The lens reads the count with
`list_records("routing_queue", where={"status": "waiting"})`.

## Proposer (deterministic first, one cheap judge call)

Eligibility: `provenance == human`, path under `_private/0-Fleeting-Inbox/`,
≥40 chars of prose, no `zzz:`. Same gate as echo, extracted to a shared
helper.

Candidates and ranking, first match sets the top suggestion:

1. `aaa:` hint naming a note (fuzzy title match against the vault) and,
   optionally, a verb word.
2. An explicit `[[wikilink]]` in the block to a registered container → `file under`;
   to any other note → `link`.
3. Nearest older blocks from `engine.context` (the echo retrieval, same
   filters) → their containers via `routed_to` / `contains` → `file under`;
   a strongly matching non-container note → `extend`.
4. Nothing above the floor → `memory` proposed as the top line.

Then one temperature-0 judge call (same LLM as echo) with the candidates,
answering: life-log or working thought, and which verb+target, with a
one-clause why. `memory` for life-log is the judge's call; the deterministic
step only orders the candidates. Registered containers come from the
review-pass rule (container tag + filled `description`), discovered by tag
search per cycle and cached.

## Implementation sequence (one concern per commit, tests with each)

1. `docs(plan)`: this file with decisions recorded.
2. `refactor(echo)`: extract `pipeline/augi_log.py` (section-aware writer +
   header) and `is_capture_eligible`; echo behavior unchanged, existing
   tests green.
3. `feat(routing)`: `pipeline/route.py`: eligibility, candidates, judge,
   row rendering, `routing_queue` ledger. Wired in `watcher.py` after echo.
   Tests with a fake store and a fake LLM. **Shipped 2026-09-03.** Registry
   discovery: document blocks carry no tags or description, so the DB only
   prefilters (titles containing "MOC", outside `OpenAugi/`) and the file
   decides (container tag + filled description), cached by mtime. A daily
   note is never a home, keyed on its date-shaped title rather than its
   folder, because his concept notes also live in `0-Fleeting-Inbox`.
4. `feat(routing)`: `pipeline/routing_janitor.py`: the master box, ticked
   boxes and `aaa:` overrides, apply (file under / link / extend / new note
   / memory / hold), extend-insert newest-first with markers, `✓` rewrites,
   `undo`, feedback records. Tests. **Shipped 2026-09-03**, folding step 5
   in: the master box is the apply trigger, so it could not ship separately.
   The link helper stayed local to the janitor rather than being lifted out
   of the `apply_routing` MCP tool; sharing it is a follow-up refactor.
5. (folded into 4)
6. `feat(board)`: lens rule + template change for the waiting-rows line;
   vault copy of `currency-board.md` updated.
7. `feat(routing)`: history priors from the feedback log; `openaugi routing
   stats`. Tests.
8. `docs`: `docs/reference/augi-log-routing.md` (skill format), links from
   ARCHITECTURE.md, review-pass docs marked superseded, changelog entry.

Every commit: `scripts/check.sh` first.

## Open items to settle with Chris

- All six questions decided 2026-09-03; see the **Decided** lines above.
- Changelog: `CHANGELOG.md` exists (added with the board); entries go under
  Unreleased.
- v1 scope assumption, not yet confirmed: routing covers the daily-note
  folder only, not `OpenAugi/Capture/**`.
- Whether routing should also cover `OpenAugi/Capture/**` (human provenance
  by rule) or only the daily-note folder for v1. Recommendation: daily notes
  only.
- Restarting the three `openaugi serve` processes after the merge for the
  provenance and `get_context` changes. Routing itself needs no MCP change.

## Risks

- **Noise.** Echo proved most blocks should produce nothing. Routing rows
  are one per block by design, so a heavy capture day means many rows. The
  `memory` default and the 3-day auto-resolve keep the queue from becoming
  the `#human-review` backlog that killed earlier surfaces. If rows become
  noise, the next lever is "propose only when not memory".
- **Duplicate text** from extend (Q3b). Mitigated by the source line and
  marker; accepted as the price of a render surface that exists.
- **Regex coupling.** Three janitors now read the same file. The shared
  section writer and a fixture Augi Log covering all sections are the guard.
