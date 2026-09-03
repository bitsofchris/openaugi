---
name: augi-log-routing
description: The Augi Log as the capture/routing surface. Every new human daily-note block gets one routing row (extend / link / file under / new note / memory / hold) with a proposed home; the user answers with checkboxes and aaa: lines, ticks the day's master box, and the janitor applies the log. Every answer is logged so later proposals can learn.
---

# Augi Log routing — "where does this block live?"

## When to use this doc

- You want to know what the rows under `## Routing` in an Augi Log mean and
  what ticking each box does
- You are about to change the proposer, the janitor, or the row format
- You want to know what the feature writes, and where, before enabling extend

Design record with the decisions and their rationale:
[docs/plans/augi-log-routing.md](../plans/augi-log-routing.md).

## The problem it solves

Capture is easy; the daily note fills with blocks that belong elsewhere — a
season log, an area MOC, a concept note — and nothing puts them there. The
review pass tried to do it as a batch agent job and died for lack of a
surface to answer on. The user's own framing (2026-09-02): the Augi Log
should be "the routing decisions as I go … I hit checkbox on routing
suggestions … If I don't respond … just do it automatically. Here's how you
learn me."

## How it works

```
new human block (ingest) → eligible? → propose home(s) → append row under ## Routing
                                                             ↓ (he ticks the master box)
                                                        janitor applies the whole log
```

### 1. Eligibility

`route.is_routing_eligible`: the shared capture gate (daily-note folder,
≥40 chars of prose, not a `zzz:` instruction — `augi_log.is_capture_block`)
plus `provenance == human`. AI and reference blocks are never routed.

### 2. Proposals — deterministic first

Evidence, strongest first, all on one scale so a hint always beats retrieval:

| Evidence | Verb | Confident? |
|---|---|---|
| `aaa:` line naming a `[[note]]` or a registered container title, with an optional verb word (extend / merge / link / route to / new note / memory / hold) | as hinted | yes |
| an explicit `[[wikilink]]` to a registered container | `file under` | yes |
| an explicit `[[wikilink]]` to any other note | `link` | yes |
| nearest older writing (echo's retrieval) lives in a container, or is routed to one | `file under` | only if the z-margin over the runner-up clears `confident_margin` |
| nearest older writing lives in a plain human note | `extend` | never |

A daily note is never a home (keyed on its date-shaped title, since concept
notes also live in `0-Fleeting-Inbox`). Up to three suggestions survive.

**Registered containers** follow the review-pass rule: a container tag
(`note-type/amoc`, `note-type/moc`, or `note-type/pmoc` + `status/active`)
AND a filled `description` frontmatter. Document blocks in the DB carry
neither, so the DB only prefilters (titles containing "MOC", outside
`OpenAugi/`) and the file decides; results are cached by mtime.

**The judge** (optional, the `[models.llm]` echo model at temperature 0)
sees the block and the candidates with their descriptions, says whether the
block is life-log or a working thought, may reorder the candidates, and
writes the one-clause why. It cannot override an `aaa:` hint. If the judge
fails or is unset, the deterministic order stands.

### 3. The row

```
<!-- route:<block_id> -->

### route "First day of school. Morning cold plunge…"
*[[2026-09-03]] · 2026-09-03*

- [ ] **extend [[PMOC - Audacity to take Action - Season 2 - Q2 2026]]** — same thread as your 09-01 entry
- [ ] file under [[AMOC - OpenAugi Main]] — your nearest older writing on this lives there (2026-08-31)
- [ ] new note
- [ ] memory
- [ ] hold
aaa:
```

The bold line is the top suggestion. `memory` and `hold` are always offered.
`aaa:` is the direct channel: anything typed there overrides the boxes
(`aaa: extend [[X]]`, `aaa: just a memory`, `aaa: link [[Y]]`).

### 4. The master box

At the top of `## Routing`:

```
- [ ] process this log
```

Ticking individual rows only records intent. **Nothing is applied until the
master box is ticked.** Then the janitor applies the whole log: ticked rows
as chosen; untouched rows take the bold suggestion when the ledger marked it
confident, otherwise `memory`. A log that is never ticked is left alone;
the morning board reports how many are waiting.

### 5. What each verb writes

| Verb | Writes |
|---|---|
| `file under [[X]]`, `link [[X]]` | a `routed_to` link in the DB — the review-pass primitive; no file changes |
| `extend [[X]]` | the block's text (minus its `aaa:` lines) inserted into X **newest-first** under `### <day>`, plus the link. Anchor: the `# Journal` H1 if X has one, else the run of dated `###` headings; if `### <day>` already exists the text goes at the end of that day's section; a note with neither gets a `# Journal` appended. The insert is wrapped in `<!-- augi:routed <id> -->` … `<!-- /augi:routed -->` and ends with `— from [[<daily note>]]`. |
| `new note` | `OpenAugi/Notes/<slug>.md` (`#human-review`, dated section, same wrapper), plus the link |
| `memory`, `hold` | ledger state only. `memory` is the safe default: it writes nothing anywhere. |

Extend is the only verb that touches one of the user's own notes. It is
append-only and reversible, and `[routing] extend_writes_note = false` turns
it into a plain link.

### 6. Confirmations and undo

Each applied row is rewritten to `- ✓ <what happened> (you | your aaa: | auto)`
plus `- [ ] undo` where something was written. Ticking undo removes the
link and, for extend, exactly the wrapped text (and the dated heading if it
is left empty). A new note is left in place and unlinked — the janitor never
deletes a file. The master box becomes `- ✓ processed <time>`.

### 7. What is recorded

- **Ledger** — `routing_queue` collection in the records table. One record
  per row (`kind: row`: status `proposed → applied | memory | held | failed |
  gone | undone`, the proposals, confidence, features) and one per log
  (`kind: log`: `waiting → processed`). `route.waiting_logs(store)` is what
  the board reads.
- **Feedback** — `OpenAugi/Capture/feedback-log.ndjson`, `source: routing`,
  one line per resolution: `signal` (`accepted | corrected | memory | hold |
  auto | undo`), `by` (`you | aaa | auto`), the top proposal, the choice, and
  the block's features (folder, tags, nearest notes, whether it had an
  `aaa:`). This is the history later proposals are biased by.

### 8. Priors — how it learns

`route.load_priors` reads every `source: routing` line of the feedback log
and tallies, per target and per (folder, verb), how often that candidate was
on the table and how often it was chosen (an `undo` reverses the choice it
undoes). `Priors.bonus` turns that into a nudge: `(rate − 0.5)`, weighted by
how many decisions back it (full trust at 5), capped at ±0.1. The nudge is
applied **only to retrieval-sourced suggestions**, so it can reorder what
augi guessed but never lift a guess above his own link or `aaa:` hint. No
model, no training; `openaugi routing stats` prints the tallies and the logs
still waiting.

## Configuration

| Key | Default | Effect |
|---|---|---|
| `[routing] enabled` | `true` | Turn the rows off entirely |
| `[routing] extend_writes_note` | `true` | `false` makes extend a plain link |
| `[routing] confident_margin` | `1.0` | z-margin a retrieval-only `file under` needs to auto-apply |
| `[models.llm]` | unset | The judge; without it proposals are purely deterministic |

CLI: `openaugi routing stats [--path VAULT] [--db DB]`.

## Known limits

- One row per block by design, so a heavy capture day means many rows. The
  master box and the `memory` default keep it from becoming a queue that
  must be tended; if rows become noise, the next lever is "propose only when
  not memory".
- Extend copies text; a later edit in the daily note does not propagate.
- The registry prefilter needs "MOC" in the note title. A registered
  container named otherwise is still reachable by `aaa:` and wikilinks.

## Files

- `pipeline/route.py` — eligibility, registry, proposals, judge, row, ledger
- `pipeline/routing_janitor.py` — master box, answers, apply, undo, feedback
- `pipeline/augi_log.py` — the shared log file and its sections
- `pipeline/watcher.py` — the post-ingest hooks
- `tests/test_route.py`, `tests/test_routing_janitor.py`
