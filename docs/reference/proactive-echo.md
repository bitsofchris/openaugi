---
name: proactive-echo
description: The one thing that runs unasked. New daily-note blocks are matched against the user's own prior writing; when the match would genuinely help, an echo is appended to a dated Augi Log with promote/feedback checkboxes.
---

# Proactive Echo — "you thought this before"

Everything else in OpenAugi is pull: you type `zzz:`, you ask for a lens, you
run the review pass. Echo is the exception. As you write in a daily note, it
looks for older thinking of your own that bears on what you are writing and
appends it to `OpenAugi/YYYY/MM/DD/Augi Log.md`.

It is deliberately the *cheap* pass — one retrieval, one judgment call per
block — not an agent loop. See "Echo vs. a `zzz:` task" below.

## The loop

```
new block (ingest) → eligible? → retrieve → stratify/rank → judge → append
                          ↓ no                        ↓ silent
                        skip                       count as quiet
```

No new daemon: this is a post-ingest hook inside the existing watcher cycle
(`pipeline/watcher.py`, alongside `dispatch.py`), so it inherits the file
watching, the debounce, and the launchd restart policy from `openaugi up`.

### 1. Eligibility (deterministic, no cost)

A block must be in the daily-capture folder, carry ≥40 characters of real
prose (wikilinks and markdown stripped), and not be a `zzz:` instruction —
dispatch owns those. This is what keeps link-only stub blocks out; in the
validation replay a block whose entire content was one wikilink was the
corpus's top "match."

### 2. Retrieval — hybrid, one shot

`query.engine.context()`: FTS keyword prong + semantic vector prong, deduped,
MMR-reranked for diversity, then one hop of link expansion. Candidates that
are the block itself, same-day-or-newer, or OpenAugi-derived artifacts are
dropped. Link-expanded neighbours inherit their parent's score at a 0.9
discount, so graph proximity competes on the same scale instead of scoring 0.

### 3. Ranking — stratify, then cluster

See [echo-ranking](#ranking-details) below. Global similarity cannot rank a
single-author vault; this step is what makes the pool meaningful.

### 4. Judgment — the actual gate

An LLM sees the new block and the ranked candidates (annotated with
recurrence: *"you have returned to this note 3×, 2026-06-15…2026-07-14"*) and
answers a single question: would any of these help right now? It runs at
**temperature 0** — a block that echoes on one pass must not go silent on the
next.

Silence is the default and the common case. `[models.llm]` unset disables echo
entirely.

### 5. Output — an ephemeral, read-optional log

Each echo is appended under an `<!-- echo:<block_id> -->` marker (idempotent —
a block is never echoed twice), with three checkboxes: promote → new note,
good match, bad match. The day's last line is a heartbeat
(`watched 12 · spoke 3 · quiet 9`) so silence is legible rather than
indistinguishable from breakage.

**The log is a cache, not truth.** Delete it with zero grief; nothing depends
on it and no queue accumulates. This is deliberate: every "come back and tend
it" surface in this project's history has died, while every fire-and-forget
one has survived.

### 6. The janitor closes the loop

Ticking a box *is* the command (the review-pass precedent: an accepted
proposal is the command). On the next watcher cycle `echo_janitor.py`
promotes to a note (context header + dated append-only log) or records
feedback to `OpenAugi/Capture/feedback-log.ndjson` — the same stream the
mobile app's "you said this before" feature wrote — then rewrites the line to
`✓ …` so it never re-processes.

## Ranking details

Implemented in `pipeline/echo_rank.py`. The problem it solves: in a
single-author vault, cosine similarity mostly measures *"this is the same
person writing"* — voice, vocabulary, recurring nouns — not *"this is the same
thought."* A 7-day validation replay found top-scores clustered at 0.53–0.68
whether the block was an architecture note or a journal entry about a kid's
bike. **No global threshold can separate that.**

The fix is the vault's own Contextgraph Rule — stratify, then cluster:

1. **Stratify** by the taxonomy's `source` and `note-type` facets, read from
   tags when present and **inferred from the folder when not**. The inference
   is load-bearing: only ~26% of blocks carry any tag and ~2.4% carry routing,
   so a tag-only stratifier would gut recall. External sources (podcast,
   readwise, webclip, ai-chat, notebookLM, gdrive) never echo — you have not
   "already thought" someone else's article.
2. **Score relative.** Rank by z-score against the retrieved pool's own
   distribution: 0.63 in a pool of 0.60±0.02 is noise, 0.63 in a pool of
   0.52±0.03 is a spike. A pool too small or too flat for a z-score to mean
   anything defers to judgment instead.
3. **Cluster** by source note, so recurrence and date span are available to
   the judge — a thread returned to five times is worth more than any single
   match.

**Scores must stay comparable for this to work.** Two scoring bugs were found
by building this feature and are worth not reintroducing: semantic scores were
computed as `1 − L2` on unit vectors (clamping everything below cosine 0.5 to
zero), and FTS hits were pinned at a constant 1.0 (making every keyword hit an
unbeatable outlier and inflating the pool's variance). Both now resolve to
true cosine.

## Echo vs. a `zzz:` task

Same retrieval substrate, opposite orchestration. Keep them distinct:

| | Proactive echo | `zzz:` research task |
|---|---|---|
| Queries | one (the block text) | many, agent-chosen and refined |
| Tools | a single retrieval call | search, get_related, traverse, read, loop |
| LLM turns | 1 | dozens |
| Cost / latency | ~a cent, seconds | dollars, minutes |
| Failure mode | stays silent | writes a document |

Echo runs on *every* eligible block, all day. That budget is why it is one
cheap shot: an agent loop per paragraph would cost real money and minutes, and
would destroy the property that makes it tolerable — that you do not notice
the machinery.

## Configuration

| Key | Effect |
|---|---|
| `[models.llm]` | The judge. **Unset it to disable echo entirely.** |
| `[salience] resurface` | Noise floor only, *not* the relevance gate — judgment is. |
| `--debounce` on `openaugi up` | How long after you stop typing echo may speak. |

## Known limits

- **Mixed blocks slip through.** A daily entry containing both life-logging
  and a line about work can echo on the work half. Structural rules to prevent
  it (e.g. "daily-journal never echoes to daily-journal") would also kill good
  echoes — one of the best in the replay was exactly that shape. The good/bad
  ticks are the intended tuning mechanism.
- **Thresholds are not yet fit from feedback.** The accumulated
  `feedback-log.ndjson` signal is what a per-stratum fit would use; that work
  is deliberately deferred until there is enough of it.
- **Pattern tier is detector-only.** Generic recurrence counting is available
  to the judge as context but does not itself trigger an echo.

## Files

- `pipeline/echo.py` — eligibility, retrieval, judgment, log writing, heartbeat
- `pipeline/echo_rank.py` — stratify / relative-score / cluster
- `pipeline/echo_janitor.py` — checkbox handling, promotion, feedback
- `pipeline/watcher.py` — the post-ingest hook that calls all three
- `tests/test_echo.py`, `tests/test_echo_rank.py` — 29 tests
