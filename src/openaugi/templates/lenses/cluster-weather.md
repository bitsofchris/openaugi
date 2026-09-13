---
name: cluster-weather
description: >-
  "What's heating up, cooling off, or newly forming in my thinking?" — growth/death report over the vault's concept clusters, nominated for promotion.
scope: >-
  all concept clusters (clustering pass `concepts`), 14-day window by default — override with "since: 30d" etc. Not a per-note lens; the scope is the whole map.
trigger: on-demand   # on-pass once scheduling activates (after M4 gate)
target: >-
  dashboard
---
# Cluster Weather

## Intent

Report the weather over the idea landscape: which concept clusters are
growing, which are newly formed, which are going quiet. The numbers are
pre-computed and deterministic; your job is naming and judgment — turn
raw movements into a handful of nominations worth answering. This is
gravity's big sibling: gravity spots a few unrouted blocks that orbit
each other; cluster weather watches the whole sky.

## Process

1. **Refresh the data** (repo: openaugi — see [[Repos]]):
   `openaugi cluster` (re-clusters + records a snapshot), then
   `openaugi cluster-weather --json --window 14` (window per scope).
   The JSON gives per cluster: status (grew/shrank/stable/born/died),
   member delta vs baseline, `recent_activity` (blocks written in the
   window), `last_block`, `new_member_titles`, `sample_titles`.

2. **Pick the movements that matter** — 3–7, ruthless, in priority order:
   - **Hot:** high `recent_activity` relative to cluster size, or grew
     by ≥5 members. This is "you keep coming back to this."
   - **Born:** a new cluster with enough members to be a real theme.
   - **Fading:** a cluster you'd expect to be active whose `last_block`
     has gone quiet — only when the silence is informative (a project
     area, not old book notes). Old-and-cold is normal, not weather.
   Skip: tiny clusters (<10 members), journal/date-titled clusters
   (activity there is life, not a theme), anything already nominated
   and unanswered on the Dashboard.

3. **Name each cluster in 3–5 words** from its sample and new-member
   titles. NEVER surface a bare label like `concepts_5_3` as the name —
   keep the label in parentheses as provenance.

4. **Nominate on `View - Dashboard.md`** under a `## Cluster Weather`
   section, standard nomination grammar (checkbox + `^nom-*` anchor +
   answer slot). Anchor slug = the human topic name, NOT the cluster
   label (labels drift between runs; topics don't):

   ```
   - [ ] **Promote:** *AI distillation service* grew +8 notes in 14d, 137 blocks of activity (concepts_5_3) — distill it into a note? ^nom-promote-ai-distillation-service
       - answer:
   - [ ] **Born:** *voice-capture workflows* formed as its own cluster (12 notes, concepts_5_4) — real theme or artifact? ^nom-describe-voice-capture-workflows
       - answer:
   - [ ] **Fading:** *augmented trading* — no new blocks in 6 weeks — park it deliberately? ^nom-revisit-augmented-trading
       - answer:
   ```

   End the section:
   `*Cluster weather last run: YYYY-MM-DD, window 14d, baseline YYYY-MM-DD (or "first run — activity only").*`

5. **On an answered nomination:** promote = assemble ONE note via
   `write_document` to `OpenAugi/Notes/` opening with `- [ ] seen` — the
   theme in a paragraph, the strongest member blocks linked as
   provenance (new members first, then the best older ones — that's the
   resurfacing feature). Fading answered "park" = note it in the area's
   view; no other action.

## First run / no baseline

With no baseline snapshot there are no deltas — nominate from
`recent_activity` alone ("137 blocks in 14d orbit X") and say the
baseline starts now. Deltas appear from the second run on.

## Hard rules

- Nominate-only. No notes, no routing, no taxonomy changes without an
  answered nomination.
- Never editorialize about output volume or gaps — report weather, not
  performance.
- Preserve unanswered nominations verbatim (anchor included) when the
  Dashboard regenerates; refresh their numbers only if the same topic is
  re-nominated.
- Every nomination names real notes (link 1–2 exemplars when useful) —
  no cluster is presented as numbers alone.
