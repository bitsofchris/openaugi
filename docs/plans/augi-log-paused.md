---
name: augi-log-paused
description: >-
  The Augi Log (proactive echo + routing rows) is PAUSED as of 2026-09-04.
  This is the single resume note: why it was turned off, the exact settings it
  was running at, the four ideas for how to make the matches useful, and a
  copy-paste prompt to restart the work.
---

# Augi Log — paused 2026-09-04, and how to resume it

**Status: PAUSED.** Both writers into `OpenAugi/YYYY/MM/DD/Augi Log.md` are
off. Nothing is deleted, nothing is broken, the existing logs are still on
disk. This note is the one place that says how to turn it back on and what to
change first.

Related docs — read them, this note does not repeat them:

- [docs/reference/proactive-echo.md](../reference/proactive-echo.md) — how echo works
- [docs/reference/augi-log-routing.md](../reference/augi-log-routing.md) — how routing rows work
- [docs/plans/augi-log-routing.md](augi-log-routing.md) — the routing design record and its decisions
- Vault: `OpenAugi/2026/08/29 - Design - Proactive Echo (Watcher + Daily Augi Log).md` — the original design, §10 already argues the fix

## Why it was paused

The user, 2026-09-04:

> I think we shut down the augi log? It's not working well at all. The blocks
> need their intent or area clarified more — the matches are incorrect or too
> coarse to be useful. I want block style unrest, maybe I mix the blocks too
> much. Need to do better capture like the high level taxonomy and routing
> first? Then can try a pass at similar notes? Or set the thresholds much
> higher to find duplicates or related I forgot about.

What the logs actually showed over its four live days:

| Day | routing rows | echoes | quiet | boxes ticked | `aaa:` answers |
|---|---|---|---|---|---|
| 09-01 | 0 | 3 | 14 | 0 | 1 |
| 09-02 | 0 | 15 | 13 | 0 | 0 |
| 09-03 | 13 | 5 | 13 | 0 | 0 |
| 09-04 | 13 | 6 | 7 | 0 | 1 |

**No log was ever processed** — the master box was never ticked, so the
routing janitor never applied anything and no `extend` ever wrote into a
note. All 26 routing rows are still sitting unapplied, which is the designed
safe state.

Feedback in `OpenAugi/Capture/feedback-log.ndjson`: 3 `liked`, 4 `disliked`,
1 `correct-silence` — too little to fit thresholds from, and net negative.

The canonical failure, from 09-04's own log: a block reading *"Thunderstorm
last night woke us all up. Morning plunge again today…"* was proposed as
**file under [[Tornado Antidote - Read When Tornado Spins]]**, and the user
answered in the `aaa:` line — *"no not at all, this was a simple memory last
night like journal? Nothing to do with tornado."* Storm-word similarity, zero
intent understanding.

That one row is also a clean specimen of the other two problems:

- **It is a mixed block.** The block is not the thunderstorm sentence — it is
  the whole `Journal` section (`granularity: document`), thunderstorm *and*
  a paragraph about the RSU vest, insurance, health and family. Two unrelated
  intents, one embedding, one routing decision.
- **It appeared twice.** Editing the block later in the day changed its
  content hash, so it re-ingested under a new id (`ba65f97af3c002f9` →
  `fe930b976f782413`) and got a second row; the first id is no longer in the
  DB. The `<!-- route:<id> -->` idempotence marker is keyed on the id, so it
  cannot see that it has already asked this question.

This is exactly the failure mode the design predicted and did not fully fix:
in a single-author vault, global similarity mostly measures *"this is the user
writing"*, not *"this is the same thought"* — so everything was pushed onto
the LLM judge, and the judge alone is not enough.

## Where things were set at pause

All in `~/.openaugi/config.toml` (backup of the pre-pause file:
`~/.openaugi/config.toml.bak-2026-09-04`).

| Setting | Value while live | State now |
|---|---|---|
| `[models.llm]` | `openai` / `gpt-4o-mini` (the echo judge **and** the routing judge) | **commented out** → echo disabled entirely |
| `[routing] enabled` | absent, i.e. default `true` | **`false`** → no routing rows |
| `[routing] extend_writes_note` | never set → default `true` | untouched (never exercised — no log was processed) |
| `[routing] confident_margin` | never set → default `1.0` | untouched |
| `[salience] resurface` | `0.50` | unchanged — noise floor only, never the relevance gate |
| `[salience] push` | `0.62` | unchanged, no consumer |
| `--debounce` on `openaugi up` | `45` (in `~/Library/LaunchAgents/com.openaugi.up.plist`) | unchanged |

**What still runs:** `openaugi up` itself — ingest, `zzz:` task dispatch, the
currency board, the echo/routing janitors. The janitors are harmless with the
writers off: they only act on boxes ticked in logs that already exist, so the
26 pending rows can still be answered and applied by hand if wanted.

**To turn it back on:** uncomment `[models.llm]`, set `[routing] enabled =
true`, then `launchctl kickstart -k gui/$(id -u)/com.openaugi.up`.

## The four ideas to try before turning it back on

These are the user's, in his order. Idea 1 is the one the design doc (§10) and
`AGENT/routing.md`'s Contextgraph Rule both already point at, and it is the
prerequisite for the rest.

### 1. Clarify block intent/area at capture, before matching

The system currently *infers* what a block is (`echo_rank.py` stratifies by
the `source` and `note-type` facets, inferred from the folder when tags are
missing — only ~26% of blocks carry any tag, ~2.4% carry routing). Inference
at the daily-note level is too coarse: every block in `0-Fleeting-Inbox`
looks the same to the stratifier, so a journal line and an architecture note
compete in one pool.

The idea: **capture-time taxonomy first.** Get intent (memory / working
thought / idea / task / reference) and area (the `area/*` facet) onto the
block *when it is written* — by a light classifier pass over the block, by a
capture grammar the user types, or by the mobile capture client's tag-assist —
and only then let anything match. See `AGENT/My Taxonomy.md` for the facets
and `docs/plans/capture-tag-stream-loop.md` for the adjacent idea.

Consequence: the intent gate becomes **structural, not a prompt instruction**.
A block tagged `memory` never enters the echo or routing pipeline at all,
which alone would have killed the thunderstorm row.

### 2. "Block style unrest" — stop mixing blocks

The user's own suspicion: *"maybe I mix the blocks too much."* A single daily
block containing family logging *and* a line about work is genuinely
un-routable — the design already lists this as the known unfixed limit
("mixed blocks slip through"), and the thunderstorm block is one
(storm + cold plunge + something else).

**The mechanic, stated plainly:** a daily-note block is the text between
`Qqq` markers within a section. On 09-04 the first block ran from the
`# Journal` heading to the first `Qqq` — thunderstorm + plunge in one
paragraph, RSU vest + insurance + family in the next. Two thoughts, one
block, because there was no `Qqq` between them.

So the two directions are:

- **Capture discipline** — a `Qqq` per thought. A habit change, not a code
  change, and free to try immediately: it costs nothing to type more `Qqq`
  while the log is off, and it makes the corpus better for whatever comes next.
- **Split at ingest** — segment a block into single-intent units before
  classifying (paragraph split, or `docs/plans/anchor-segmentation.md`), so
  capture stays sloppy and the machine does the work.

Worth deciding which before rebuilding the matcher, since it changes what a
"block" even is downstream. They are not exclusive — discipline now, ingest
splitting later.

### 3. Then, and only then, a pass at similar notes

Once blocks are single-intent and typed, retry matching — but *within* the
stratum, ranked against **clusters** of prior thinking rather than individual
blocks (design §10 step 2; `openaugi cluster` already exists). "This belongs
to a thread you've returned to 6 times" is a useful thing to say; "here is one
block with similar words" is not.

### 4. Or: raise the thresholds much higher, and change the job

The alternative framing, and the cheapest experiment: **stop trying to be
interesting and only speak on near-certainty.** Raise the bar until the only
things that fire are near-duplicates and things the user has genuinely forgotten
he wrote. One or two a week instead of six a day. That is a different product
— a duplicate/forgotten-thread detector, not a conversational echo — and it
does not need the taxonomy work first, which is what makes it worth trying as
a standalone probe.

Note the trap, from the replay: **a raw score threshold cannot do this.** Real
scores cluster 0.53–0.68 regardless of block type, so "much higher" has to
mean a much higher **z-score against the block's own stratum** (or a stricter
recurrence requirement), not a bigger `[salience] resurface`.

### Also fix, whenever it resumes

- **Re-ingest duplicates rows.** An edited block gets a new content hash and
  therefore a new id, so it is proposed again under a second
  `<!-- route:<id> -->` marker (09-04's thunderstorm block, twice). Same for
  echo. Idempotence needs a second key that survives an edit — block position
  / source anchor, or a text-similarity check against rows already in the day's
  log.
- **Fit thresholds from feedback.** Still the deferred step 4 of design §10 —
  but it needs feedback volume the four-day run never produced. Don't count on
  it until the system is speaking usefully enough to be answered.

## Resume prompt

Copy-paste this to pick the work back up:

```
The Augi Log (proactive echo + routing rows) has been paused since 2026-09-04.
Read docs/plans/augi-log-paused.md in the openaugi repo first — it has the
full state, the settings it was running at, and four ideas for fixing it.

The problem: matches were incorrect or too coarse. Blocks need their intent
and area clarified before anything tries to match them, and I probably mix
too much into one block. Four ideas, in my order of preference:

1. Capture-time taxonomy and routing FIRST — get intent (memory / working
   thought / idea / task) and area onto a block when it is written, so the
   intent gate is structural instead of a prompt instruction. A block tagged
   `memory` should never enter the pipeline at all.
2. Decide the "block style" question — do I split mixed blocks at capture
   (habit) or segment them at ingest (code)? This changes what a block is
   downstream, so decide it before rebuilding the matcher.
3. Only then retry similar-note matching, ranked against clusters of prior
   thinking rather than individual blocks.
4. Alternative cheap probe: raise the bar until it only speaks on
   near-duplicates and genuinely forgotten threads — one or two a week, not
   six a day. Remember a raw score threshold can't do this; it has to be a
   z-score within the block's own stratum.

Start by telling me which of these you'd do first and why, and what the
smallest replay experiment is that would tell us if it works — do not turn
anything back on until we've agreed on that. Validate with a replay over
recent daily notes (the 2026-08-29 design doc §5 has the replay protocol)
before re-enabling the live watcher.
```
