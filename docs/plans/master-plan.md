---
name: master-plan
description: The long-running sequence — what to build and use next, in order, each milestone gated on the previous proving out in daily use. Start every session here.
---

# Master Plan — the sequence

## STATUS / LEFT OFF (update every session)

**2026-07-07 (eod):** M2 passed (Chris's call). M3 rescoped to "converge
every surface on the file contracts" (spec below) and **built the same
day** across all three repos: M3a shipped here (context-pack writer +
nomination anchors, live-verified against the real vault), M3b committed
in `openaugi-obsidian-plugin` (521399e — task-file commands, legacy Task
Dispatch deprecated), M3c vault mode built in the parallel
`private-augi-mobile` thread. **M3 loose ends (human verification, not
build):** (1) Chris verifies the plugin commands in Obsidian, then cuts a
plugin release; (2) phone-loop verify for vault mode; (3) the next review
pass is the first to exercise nomination anchors + `write_context_pack()`
— watch it. **M4 (routing quality) is now IN PROGRESS** — usage-gated, two
weeks of passes; see "M4 posture" note at the end of this doc. M5/M6
exploration can run in parallel (read-only / prompt-level, can't
destabilize the loop).

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

### M2 — Live the loop ✅ (2026-07-07)
Run passes on demand (`openaugi review`, phrase, or zzz). Answer
nominations. Gate passed: the Dashboard answers "where did I leave off per
area" with zero archaeology and Chris trusts the recaps.

### M3 — Converge surfaces on the file contracts ✅ (2026-07-07, spec below)
Rescoped 2026-07-07 from "Obsidian plugin commands" to the full interface:
mobile + plugin + CLI all land on the same three vault-file contracts.
Built same-day in all three repos; remaining loose ends are human
verification (plugin release, phone loop, first anchored pass) — tracked
in STATUS above.

### M4 — Routing quality (IN PROGRESS since 2026-07-07 — usage-gated)
Real `type/*` facets; `aaa:` honored in the wild; salience tuning from
Chris's corrections; `description` frontmatter on all registry notes;
rename handling exercised. **Gate:** two weeks of passes with <handful of
manual corrections each.

### M5 — Lenses ([lens-framework.md](lens-framework.md)) — lens #3 shipped as prose
Lens specs as data when lenses visibly diverge — never before. **Nugget
lens shipped 2026-07-07 as a prose skill file**
(`<vault>/OpenAugi/AGENT/nugget-lens.md`, + template): scans recent
working notes for individually valuable insights, nominates 3–7 on the
Dashboard (`## Nuggets`, standard anchor + answer-slot grammar),
nominate-only. Trigger: "run the nugget lens" / zzz / task file. **Next
lens: cluster weather** (growth/death feeding gravity — needs the
clustering pipeline, pairs with an M6 cluster map). Then habit/tornado
trends (needs accumulated passes). The spec engine gets built the moment
lens prose starts duplicating — extracted from real cases, not designed.
Scheduled runs land here too (only after passes are boringly reliable).

### M6 — Rich render surface (IN PROGRESS 2026-07-07 — lifestream v1 shipped)
**Decided: static self-contained HTML from the DB — the file contract
again, no server.** `openaugi render` → `OpenAugi/render/lifestream.html`
(data inlined as JSON, client-side filters, syncs to phone via the
vault). First screen shipped: **lifestream** — merged chronological block
stream + commit-graph heat strip, filterable by area/day/search
(`src/openaugi/render/lifestream.py`; 5.3k blocks live-verified).
Iterate in scratch → promote; next screens: cluster map (pairs with
cluster-weather lens), container timelines. A live server only if
staleness ever bites.

### M7 — Mobile review flow (was: mobile capture — capture moved into M3)
Mobile capture landed early: the Node bridge writing vault markdown *is*
the backend (M3c); a real HTTP endpoint on openaugi only when mobile needs
pipeline-only data (semantic suggestions) or the app ships publicly. M7 is
now the native review UX: phone pulls nominations (anchored on the
Dashboard per M3a), renders a review queue, upserts answers by anchor;
"process the dashboard" executes them. Plus reachability (Tailscale).

### M8 — Data lake + curator
Multi-source ingest (gdrive = Chris's voice; readwise/notebooks =
attributed third-party), `source/*` firewall, dedup/identity. Curator /
self-improving taxonomy (needs accumulated accept/reject signal).

## M3 spec — converge surfaces on the file contracts (decided 2026-07-07)

**The vault filesystem is the API.** Mobile, the Obsidian plugin, the CLI,
and zzz all reduce to three file contracts — no HTTP into openaugi, no
coupling between repos beyond file formats.

**1. Capture contract (write side).** Markdown blocks in the vault.
Desktop: Chris types in daily notes (already true). Mobile: the Node
bridge (formerly "mock server" — it's the real mobile→vault bridge now)
writes into **mobile's own daily file** (`Capture/<YYYY-MM-DD>.md`), one
paragraph per block with a trailing Obsidian-native `^<block-id>` anchor,
**upsert-by-anchor** for idempotent re-sync. Own file avoids write
contention with the daily note open in Obsidian. Block identity note:
ingest IDs blocks by content hash, so an edit = a new block — fine,
because "text is truth" (mobile M6): tags and `aaa:` live in the block
text and re-ingest carries them.

**2. Trigger contract.** A `status: pending` task file in `OpenAugi/Tasks/`
(see `templates/task-template.md`); the task watcher is the single
execution path. Already true for: zzz grammar, `openaugi review` CLI.
Mobile needs **no trigger surface** — `zzz:` in a captured block
dispatches after ingest. Plugin commands write the task file directly via
the Obsidian vault API (decision: option (b); no shell-out, no HTTP).

**3. Read contract.** The review pass emits derived artifacts:
human views (Dashboard etc.) **plus machine-readable sidecars** —
`context-pack.json` (taxonomy, registry concepts, note titles, recents,
generated from the DB; the bridge serves it to the phone; staleness of
hours is fine for tag suggestions). Foundation for mobile review (later):
each Dashboard nomination gets a **stable anchor + structured answer
slot**, so a phone can upsert answers by anchor — same writer mechanics
as capture — and "process the dashboard" stays the single executor
regardless of where answers came from. (Obsidian Mobile on the synced
vault is the zero-code fallback review surface.)

**Work items:**

- **M3a (this repo) ✅ 2026-07-07:** context-pack writer
  (`pipeline/context_pack.py`, MCP `write_context_pack`, CLI
  `openaugi context-pack`); nomination anchors + answer-slot shape in
  `<vault>/OpenAugi/AGENT/review-pass.md` (mirrored to the repo template).
- **M3b (plugin repo):** commands as task-file writers — "Run review pass",
  "Process dashboard", "Distill selection" (selection/active note becomes
  the Context section; plugin-as-scope-selector), (later) "aaa this block".
  **Deprecate the plugin's legacy Task Dispatch** (it launches tmux itself,
  a parallel execution path that will drift from the watcher: different
  session names, duplicate repo-path settings). Community users without the
  Python watcher are the migration concern — deprecate over a release or two.
- **M3c (mobile repo, in progress in a parallel thread):** vault mode per
  `private-augi-mobile` ROADMAP "NOW" — daily-file writer with anchors as
  above, real context pack served from the sidecar file.

Connectivity: home wifi now (mobile's offline queue covers gaps) →
Tailscale when away-from-home matters (env-var change) → cloud endpoint
only if shipping as an app.

## M4 posture (2026-07-07)

M4 is a **usage milestone, not a build milestone** — the gate is two weeks
of passes with a <handful of manual corrections each. The build work in
this repo is small and reactive:

- **`type/*` facets** — arrive via Dashboard nominations when the signal
  appears (taxonomy changes only via nomination; never pre-build them).
- **Registry `description` frontmatter** — the pass already nominates
  missing ones; answering the nominations is the work.
- **Salience/routing tuning** — prompt edits to the vault
  `review-pass.md` in response to corrections. Corrections ARE the
  instrumentation: a wrong route fixed with `aaa:` or told to the agent is
  the signal; no new tooling needed yet.
- **Rename handling** — exercise deliberately once: rename a container,
  confirm the next pass regenerates the view under the new title and
  deletes the stale one.

Safe to build in parallel without violating the gate: the content
pipeline (Kafka post), and M6 render-surface prototypes — read-only over
the same DB, so they can't destabilize the loop.

## Content pipeline (extract as you go, per the operating system)

- Post draft exists: "I accidentally rebuilt Kafka for my second brain"
  (vault: OpenAugi/Notes/). The auto-KB canonical piece is also ready
  (research done 6/24).
- Future posts fall out of milestones: M5 lenses ("saved questions for your
  life"), M7 mobile ("capture is a database write").
