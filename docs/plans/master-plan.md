---
name: master-plan
description: The long-running sequence — what to build and use next, in order, each milestone gated on the previous proving out in daily use. Start every session here.
---

# Master Plan — the sequence

## STATUS / LEFT OFF (update every session)

**2026-07-06:** M1 shipped and run (review pass v1 + distill lens + `openaugi
review` CLI). Chris is in M2: living the loop during hackathon week, running
passes on demand. Next gate: the DoD call after 2+ passes.

## The value (why any of this)

Capture is cheap; curation is expensive; Chris was the only curator — so
every visibility system died. The system separates truth (his append-only
writing) from derived views (agent-maintained), gating human review to
structure changes only. Success metric, always: **time-to-context — "where
did I leave off / what do I know about X" answered without archaeology.**

## Sequence (each gated on the one before)

### M1 — Review pass v1 ✅ (2026-07-06)
Route → views (recap + remote log, two refresh tiers) → Dashboard
nominations → high-water mark. Distill lens. `openaugi review` CLI.
Details: [review-pass-v1.md](review-pass-v1.md).

### M2 — Live the loop (NOW, gate: the DoD)
Run passes on demand (`openaugi review`, phrase, or zzz). Answer
nominations. **Gate:** after 2+ passes, the Dashboard answers "where did I
leave off per area" with zero archaeology and Chris trusts the recaps.
Kill condition: two mushy runs → fix routing before building anything.

### M3 — Trigger surfaces (spec below)
Obsidian plugin commands. **Gate to start:** M2 passed.

### M4 — Routing quality
Real `type/*` facets; `aaa:` honored in the wild; salience tuning from
Chris's corrections; `description` frontmatter on all registry notes;
rename handling exercised. **Gate:** two weeks of passes with <handful of
manual corrections each.

### M5 — Lenses ([lens-framework.md](lens-framework.md))
Lens specs as data when a third lens diverges. First candidates: nugget
nominations from working notes (bronze→silver extraction), cluster weather
(growth/death feeding gravity), habit/tornado trend lens. Scheduled runs
land here too (only after passes are boringly reliable).

### M6 — Rich render surface
HTML dashboard / JARVIS view: merged chronological logs, timelines,
cluster maps (knowledge-timeline + cluster-viewer precedent). Read-only
over the same DB.

### M7 — Mobile capture (private-augi-mobile M1.5+)
Real server replaces the mock contract server: `POST /capture` → block
ingest; `GET /context-pack` → taxonomy + registry + recent concepts (built
in M1–M4, mobile just reads it). Tap-to-route = capture-time `aaa:`.

### M8 — Data lake + curator
Multi-source ingest (gdrive = Chris's voice; readwise/notebooks =
attributed third-party), `source/*` firewall, dedup/identity. Curator /
self-improving taxonomy (needs accumulated accept/reject signal).

## M3 spec — trigger surfaces (the task file is the API)

Everything converges on one contract: write a pending task file to
`OpenAugi/Tasks/` (see `templates/task-template.md`); the task watcher
launches the session. Already true for: zzz grammar, `openaugi review` CLI.

**Obsidian plugin commands (in the plugin repo, not here):**

1. **"Augi: Run review pass"** — write task file with instruction
   "run the review pass".
2. **"Augi: Process dashboard"** — instruction "process the dashboard".
3. **"Augi: Distill selection"** — take the current selection (or active
   note) as the Context section of the task file, instruction "distill this
   per OpenAugi/AGENT/distill-lens.md". This is the plugin-as-scope-selector
   idea: user picks context, lens does intent.
4. (later) **"Augi: aaa this block"** — prompt for a container, append an
   `aaa: route to X` line at cursor.

No new server API needed for any of these — file writes only. Buttons =
command palette entries; optionally a ribbon icon for #1.

## Content pipeline (extract as you go, per the operating system)

- Post draft exists: "I accidentally rebuilt Kafka for my second brain"
  (vault: OpenAugi/Notes/). The auto-KB canonical piece is also ready
  (research done 6/24).
- Future posts fall out of milestones: M5 lenses ("saved questions for your
  life"), M7 mobile ("capture is a database write").
