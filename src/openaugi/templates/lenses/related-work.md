---
kind: engine
name: related-work
description: >-
  The proactive researcher, on demand — frame the problem I am working on from the active project note, find the papers most closely related to it, and return one line per paper on what it does, why it matters for this problem, and what to read first, with checkboxes to read, park, or add to a reading unit.
scope: >-
  at apply time — the active project note named in the instruction (default: the PMOC or task note of the card I am working on): its newest dated entry, any open questions in it, and optionally a code diff or a failed-experiment receipt pasted in the instruction. Papers already in the vault (any note with an arXiv id or DOI in its body or frontmatter) are the dedupe set. Never OpenAugi-generated prose as an input, except prior related-work notes for the same project, which supply positive and negative seeds.
trigger: on-demand
target: >-
  note — OpenAugi/Research/YYYY-MM-DD - Related Work - <project>.md (one per run; opens with `- [ ] seen`; the series for a project compounds, each run reads the previous)
---

# Related Work

## Intent

As I work a problem, one action dispatches an agent that frames the problem
from my own context and returns the papers most closely related to it, with a
reason each. Every published paper-finder starts from a typed question; this
one starts from the project note, which is the step they all lack.

Propose, never act: the note carries checkboxes; ticking them is my move.
Papers I have already read or parked are named as links, never re-offered.

## Process

1. **Frame.** From the project note's newest entry and open questions, write
   three sentences: what we are trying to predict or measure, what method we
   are using, what is failing or unknown. This is the reranking target and
   the note's header. If the entry is too vague to frame, say so and stop:
   one line in the note, no papers.
2. **Queries.** Five: two method-centric, two problem-centric, one from an
   adjacent field.
3. **Pull.** OpenAlex search and the arXiv API for each query; Semantic
   Scholar search for abstracts. Cap sixty candidates.
4. **Seed and chase.** Top three by similarity to the framing become seeds:
   Semantic Scholar recommendations for each, plus one hop of references and
   citations. Papers ticked `read` in earlier runs for this project are
   positive ids; papers ticked `park` are negative ids.
5. **Rerank** each abstract against the framing on one question: does the
   method section answer the open question? Topical overlap does not count.
   Keep eight to twelve.
6. **Dedupe** against the vault. A paper already in a note is listed once
   under *Already in the vault* as a link, with no boxes.
7. **Write** the note in the format below.

## Format

```markdown
- [ ] seen
#area/research #ai-generated

# Related work — <project> — YYYY-MM-DD

**Framing.** <the three sentences>

## Papers

- **<Title>** (<year>, <venue or arXiv id>)
    what: <one line>
    why here: <one line against the framing>
    read first: <section or figure>
    - [ ] read
    - [ ] park
    - [ ] add to reading unit <name>

## Already in the vault
- [[<note>]] — <one line on why it is related>

## What to try
- <idea> — from <paper>
- <idea> — from <paper>
- <idea> — from <paper>

<details><summary>Run</summary>
Queries: … · Seeds: … · Positive/negative ids: … · Candidates pulled: N · Kept: N
</details>
```

## Hard rules

- One run per project per day. Each run is a new dated note; never overwrite.
- Every paper line carries a reason against the framing. A paper without a reason is not listed.
- Verify every title, year and id against the source before writing it. Model output is untrusted input.
- The framing is written before any search, and never rewritten after the results are in.

## Later, not now

Proactive triggers, once the on-demand run has proved useful on a real work
problem: a new unchecked question appearing in a project note's entry; a
failed-experiment receipt landing; a `read` box ticked (re-run with it as a
positive seed).
