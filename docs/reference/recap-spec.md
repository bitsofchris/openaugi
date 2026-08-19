---
name: recap-spec
description: What a container recap should contain now that the feed is live — only what scrolling can't give you. Cross-month patterns, contradictions, unanswered questions, and what's gone quiet. Not what-moved, not member lists, not a link on every line.
---

# Recap spec

**When to use:** writing or reviewing the recap half of the review pass, or
changing how `write_recap` output is rendered. The prompt that implements this
is `src/openaugi/templates/review-pass.md` (vault copy:
`OpenAugi/AGENT/review-pass.md`).

## The one rule

> **A recap contains only what scrolling can't give you.**

Everything else follows from that, and it is a *new* rule — it only became
true in August 2026.

## Why the old spec was right and is now wrong

The old recap carried a TLDR, new-this-period highlights, a task rollup, and a
`## Log` section listing every routed block with a link. That was correct when
the recap lived in `View - <container>.md` and was read in Obsidian, because
**Obsidian cannot render a `routed_to` edge**. The member list wasn't
redundancy; it was the only way to see membership at all. The links weren't
decoration; they were the only navigation.

Two things changed:

1. **Per-container view files were retired (2026-08-17).** The recap is a DB
   row, read through `get_view`.
2. **The mobile explorer renders the feed live.** Container mode shows unified
   membership — contained and routed, one indistinguishable feed — one tap
   from the recap card, sorted newest first, with the full text of every
   block.

So the member list, the what-moved list, and the new-this-period highlights
are now **a worse copy of the thing sitting directly underneath them**. Worse
because it's partial, and worse because it goes stale the moment a block
arrives while the feed never does.

Duplicating the feed also has a cost beyond redundancy: it fills the recap's
budget with the cheapest possible content, which is exactly why recaps have
felt like filler. Every line spent restating what moved is a line not spent on
something only synthesis can produce.

## What a recap contains

Four sections. Any may be empty, and an empty section is **omitted, never
padded** — a recap that says nothing is a true and useful statement that this
container is quiet.

### 1. Current understanding (3–5 sentences)

The through-line. What this container is *about* right now, as opposed to when
it was created. This is the only section that is always present, and it should
change slowly — if it churns every pass, it's describing activity rather than
understanding.

For a concept note (silver), this is the canonical statement of the idea, and
it is re-derived from the container's whole membership on every regeneration,
not just this pass's arrivals.

### 2. Patterns across time

Things visible only from above the window. The feed shows you a month; this
shows you a year.

- *"This is the fourth time since March you've described the same
  deduplication idea, each time from a different angle."*
- *"Capture-UX thinking clusters in the two weeks after each phone build,
  then stops."*

**Requires a span.** A pattern inside one window isn't a pattern — the feed
already shows it. State the span explicitly: "since March", "across five
months". Without a span this section degenerates back into what-moved.

### 3. Contradictions and open questions

The highest-value section and the hardest, because it's the one no amount of
scrolling produces: it requires holding two distant blocks in mind at once.

- **Contradictions** — *"In June you decided views should be files; in August
  you retired them. The June reasoning about Obsidian navigation was never
  addressed, it was overtaken."*
- **Unanswered questions** — a question asked in a block and never returned
  to. These are invisible in a feed because nothing marks them; they just
  scroll past.

**Both kinds of claim MUST carry their sources.** See the link rule below.

### 4. What's gone quiet

Absence is literally unrenderable in a feed — a block that stopped arriving
leaves no trace. Name the threads that were active and aren't.

- *"Nothing on the earnings-wiki since 2026-07-30, after six weeks of near-daily
  blocks."*

Do NOT editorialize about it. "Gone quiet" is an observation, not a nag; some
threads are finished and some are abandoned on purpose.

## What a recap does NOT contain

| Dropped | Because |
|---|---|
| "What moved this period" | The feed **is** what moved, sorted newest first. |
| A `## Log` / member list | `get_members` serves this live; a copied list is stale on arrival. |
| Task rollups | The Dashboard's 14-day shelf is a rendered query over the same data, and it's already the place tasks live. |
| A link on every bullet | See below — links become evidence rather than decoration. |
| Counts of new blocks | A number that is only ever a proxy for "go look at the feed". |

## The link rule — changed, not abolished

The old rule was *"every surfaced claim links back to its source note"*. The
proposal that prompted this spec was to drop source links entirely, on the
grounds that the app can now drill into any block.

**Partly adopted, and here's the part that isn't.** Navigation is no longer a
reason to link — but *evidence* still is, and the two were never the same
thing. A recap that asserts "you contradicted yourself in June" without saying
which blocks is unfalsifiable, and an unfalsifiable claim in a derived surface
is precisely the failure mode the whole trust model exists to prevent.

So:

- **Sections 2 and 3 (patterns, contradictions) MUST link every claim.** These
  are factual assertions about specific blocks. Without sources they can't be
  checked, and a synthesis that can't be checked shouldn't be trusted.
- **Sections 1 and 4 (understanding, gone quiet) need no links.** They are
  characterizations of a whole, not claims about particular blocks.

The net effect is fewer links, each one load-bearing.

## Length

**Aim for under 250 words.** The old recap had no ceiling because it was
list-shaped and lists grow with the container. This one is synthesis-shaped and
does not: a container with 400 blocks does not have four times as many
cross-month patterns as one with 100.

If a recap is running long, the usual cause is that section 2 has drifted back
into what-moved.

## Regeneration cadence

A consequence worth stating: **this recap goes stale far more slowly.** What
moved changes daily; the through-line and the contradictions change monthly.

So the existing "refresh only when it would change" tier gets stricter — a
handful of blocks routing through a container is now almost never a reason to
regenerate. Regenerate when the *understanding* would change: a decision
reversed, a thread going quiet, a question answered, or a genuinely new angle
on the idea.

Staleness is already reported honestly through `get_view`, so a recap that is
three weeks old and still correct is a feature, not a debt.

---
*2026-08-19. Written against the retirement of per-container view files
(2026-08-17) and the mobile explorer's live container feed. Supersedes the
"recap emphasis by container kind" guidance in review-pass.md.*
