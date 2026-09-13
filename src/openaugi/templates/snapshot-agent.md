---
kind: engine
type: document
description: Instructions for generating a human-reviewable OpenAugi snapshot dashboard by area using recent notes, backlinks, area anchors, and deeper pattern searches.
created: 2026-06-21T11:27:22
---

# snapshot-agent

# Snapshot Agent

- [ ] seen

#area/openaugi #area/meta

Use this when the user asks for a snapshot, proactive lens, dashboard, what is emerging, what is trying to emerge, recent overview, or area-by-area OpenAugi view.

The goal is a human-reviewable dashboard, not a full research report. Bias toward small useful next actions and visible evidence.

## Output

Write one markdown note under `OpenAugi/Notes/` or `OpenAugi/Docs/` titled like `YYYY-MM-DD - OpenAugi Snapshot`.

Use this structure:

```markdown
# OpenAugi Snapshot - YYYY-MM-DD

- [ ] seen

#area/openaugi #area/meta

## OpenAugi

### What is active

### What is trying to emerge

### What is recurring

### Next useful action

### Evidence

## Content

### What is active

### What is trying to emerge

### What is recurring

### Next useful action

### Evidence

## Self

### What is active

### What is trying to emerge

### What is recurring

### Next useful action

### Evidence

## Uncategorized / Cross-Lane

### What does not fit cleanly

### Why it may matter

### Next useful action

### Evidence
```

Keep each section concise. Prefer 3-5 high-signal bullets per subsection over exhaustive summaries.

## Retrieval Rule

Run each area as an independent pass. Pull all recent blocks in past 7 days. Then from that context look at each area seed. Within each area seed feel free to use OpenAugi to retrieve more related blocks if connecting recent blocks or needing more context for a specific question for the snapshot.

Do not use blocks from the OpenAugi folder as input, those are ai generated. You can link or reference them but they are secondary sources already derived from my human data.
## Area Seeds

### OpenAugi

Primary anchors:

- [[PMOC - Audacity to take Action - Season 2 - Q2 2026]]
- [[AMOC - OpenAugi Main]] when useful
- [[routing]] for agent output routing

Recent evidence:

- Daily notes
- Blocks tagged `#area/openaugi`
- Recent backlinks to the anchors

### Content

Primary anchors:

- Recent daily-note-linked writing/content notes

Recent evidence:

- Blocks tagged `#area/content`
- Writing-process notes created or linked recently
- Daily notes mentioning publishing, writing, audience, podcast, shorts, newsletter, public second brain, or content strategy

### Self

Primary anchors:

- Recent daily notes
- [[MOC - Self - Jung]]
- [[AMOC - Self - Weaknesses - Jung - Growth]]

Recent evidence:

- Daily-notes and links
- Blocks tagged `#area/self` or ’notetype/reflection’
- Backlinks to the Jung or AMOC notes
- Recent reflections about energy, envy, comparison, family, body, action, avoidance, confidence, or recurring personal patterns

## Lens Questions

Ask these for each area:

- What is active right now?
- What is trying to emerge but is not yet explicit?
- What pattern has appeared before?
- What old note or anchor changes how this recent evidence should be read?
- What is the smallest useful next action?

For recurring patterns, search deeper across the vault for older echoes. Do not overfit from one recent note.

## Personal Pattern Handling

If evidence resembles a documented personal pattern, name it lightly and cite why. Examples include tornado pattern, action-over-reflection, comparison/envy, or startup/content pull. Do not diagnose. Treat patterns as hypotheses for the user's review.
