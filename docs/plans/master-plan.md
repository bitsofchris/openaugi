---
name: master-plan
description: The long-running sequence — what to build and use next, in order, each milestone gated on the previous proving out in daily use. Start every session here.
---

# Master Plan — the sequence

## STATUS / LEFT OFF (update every session)

**2026-07-09 (later): route durability re-decided — similarity matcher
RIPPED OUT, re-derive contract in (Chris's call).** The CQRS discussion
landed the right model: the vault is the current-state store for content,
the DB is a projection, and routing is decision-state that gets
RE-DECIDED, not fuzzily preserved — an edited block drops its routes by
design and re-enters the next pass's queue (`aaa:` in text = the durable
form of human intent; the vault-side decision-log idea was dropped as
redundant). Contract pinned in docs/reference/data-model.md ("what
routed_to really is: auto-filing" + re-derive contract), vault
review-pass.md (edited blocks re-arrive as new — route them again, don't
hand-restore) and augi-agent.md lens engine (lenses reading routed
context check `get_review_state` and disclose staleness); both mirrored
to templates. Tests: test_route_rederive.py replaces the migration
tests. 457 green.

**2026-07-09 (gdrive thread): M8 gdrive import opened (History RAG).**
Decided with Chris: Google Drive docs enter via a **converter** (script →
vault markdown → exit; the Readwise "files are the API" pattern, NOT an
adapter), selective by manifest, provenance in frontmatter only, **no
tags written by tooling ever** (Chris's hard rule: taxonomy changes only
with his approval — converter emits zero tags; `source/gdrive` comes from
his own source_rules config line). Shipped the
one core change: **`created:` frontmatter now feeds `block_time`**
(priority: heading date > filename date > frontmatter created > file
mtime > now; `_extract_frontmatter_created` in splitter.py, resolver in
vault.py, 11 tests, check.sh green) — the generic hook every converter
(gdrive, ChatGPT-history) uses to stamp real historical dates; makes old
docs date-filterable via MCP `search(after=/before=)` + visible to
lineage eras. Plan + file contract: [gdrive-import.md](gdrive-import.md).
rclone OAuth done; scope decided: **Google Docs only** (by MIME type).
Agent runbook written: [gdrive-explore-runbook.md](gdrive-explore-runbook.md)
(Phase 1 inventory → keep/skip report → GATE 1 Chris; Phase 2 converter +
sample GATE 2 + ingest + lineage verification). **Next:** run Phase 1.
Not committed yet.

**2026-07-09: both found-in-the-wild defects FIXED; idea-lineage lens
proven on a real topic.** (1) Agent state now survives block edits —
`run_layer0` matches removed→added blocks by content similarity and
migrates `routed_to` + `augi_tags` before deleting (the trading-MOC
route loss can't recur; real deletions still drop state). (2)
Cluster-weather snapshots carry 256-dim centroids; cross-run matching
accepts centroid cosine ≥ .9, killing the label-drift born/died churn —
first centroid-bearing snapshot recorded, deltas clean from next run.
Lineage enhancements (drift/branches/genealogy) PARKED with revive
conditions (Chris's call — value unproven; see future-work.md). Lens
shakedown: "apply lens idea-lineage to advice on finding your niche" ran
end-to-end → artifact at `OpenAugi/Notes/2026-07-09 - Lineage - advice
on finding your niche.md` (146 blocks, 2023-11→2026-07, the idea
inverts: "pick a category" → "don't niche first"); one CLI bug found +
fixed (--json stdout contamination). Also: `source/podcast` added
(Chris's call) — 2,480 Snipd/podcast blocks attributed; total source
firewall coverage now ~3.2k blocks.

**2026-07-08: salience gating centralized here (cross-repo with mobile).**
`get_context` gained an additive optional `purpose` parameter (other MCP
clients unaffected) that applies a min-score gate from config `[salience]`
(`resurface = 0.06` live-calibrated, `push = 0.15` reserved for the future
push surface) — the resurfacing threshold that lived in mobile's
`server/resurface.ts` now belongs to the intelligence layer. Scoring math
untouched (gate filters only); live probe reproduced the 2026-07-07
calibration (substantive capture → 1 hit @ 0.0604, "picked up the dry
cleaning" → all 0.0, silent). Mobile's bridge passes
`purpose: 'resurface'` through the RelatedSource seam;
`RESURFACE_MIN_SCORE` env survives one release as a deprecated local
override (effective gate = max of the two). See mobile's
`docs/plans/task-salience-gating.md` (moved to archive there when done)
+ its ROADMAP STATUS. Docs: docs/reference/MCP_SERVER.md §salience.

**2026-07-08: full-system shakedown on the prod vault — pass #2 ran
agent-run end-to-end.** Everything built this week was exercised for real:
review pass #2 (17 blocks routed, first pass with nomination anchors +
`write_context_pack`; 5 anchored nominations pending on the Dashboard),
first views for AMOC Health + Fiction, morning-briefing + open-loops lens
views created, cluster snapshot #2 + first real cluster-weather deltas,
two new lineage sidecars (trading, die-before-you-die), context pack
regenerated (10 lenses listed). **Two bugs found in the wild (filed in
future-work.md "Found in the wild"):** (1) `routed_to` links silently die
when a routed block is edited (content-hash identity + CASCADE) — the
trading MOC had lost all 3 routes in 48h, restored by hand; this bleeds
M4 routing quality until fixed. (2) cluster-weather cross-run matching is
churn-heavy (born/died label-drift pairs dominate the real grew/shrank
signal) — consider centroid-similarity matching. **Next:** Chris reads
the Dashboard/briefing/loops and answers nominations ("process the
dashboard" executes them); Sunday 7/12 pass is the M2-style DoD test for
the anchored flow; fix the route-durability bug before M4 accumulates
corrections.**

**2026-07-08: cross-repo contract fixtures landed (tests+fixtures only, no
behavior change).** The three text/file contracts that couple this repo to
OpenAugi Mobile — Dashboard **nomination grammar**, **context-pack.json**
shape, and mobile's **capture daily-note anchors** — now have golden fixtures
in `tests/fixtures/contracts/` (shared source of truth; mobile vendors copies).
`tests/test_contract_fixtures.py` pins openaugi's side; regenerate the
context-pack sample via `scripts/gen_contract_fixtures.py` (real builder over
`tests/contract_corpus.py`). Refresh ritual: README "Contract fixtures".
**Cross-repo heads-up:** private-augi-mobile got the vendored copies +
`scripts/sync-contract-fixtures.sh` + its parser tests repointed at the
fixtures the same day (task-contract-fixtures.md; item 5 in that repo's
engineering queue). Both suites green; deliberately breaking the `^nom-` anchor
prefix was demonstrated to fail a test in BOTH repos. Documented latent
behavior worth noting: a mobile daily note (one `# YYYY-MM-DD` header,
blank-line-separated entries) ingests as a SINGLE content-hash block — the
splitter cuts on headings/`qqq` only — so editing any entry rehashes the whole
note; consistent with "text is truth," pinned not fixed.

**2026-07-07 (last session of the day): M8 opened — source firewall LIVE
+ idea-lineage lens.** Two builds, both grounded in Chris's own notes
(History RAG PMOC, "Persistent Memory Artifact System" 6/6): (1)
**Source attribution:** `[vault.source_rules]` config (folder glob →
`source/*` tag, explicit text tags win), applied at ingest +
`openaugi backfill-source-tags` for existing rows — Chris's DB now has
684 blocks attributed (533 readwise / 128 webclip / 23 ai-chat); rules
live in his config for Readwise/Instapaper/Articles/Reddit/AI
Conversations. No Readwise API adapter, ever — Readwise→Obsidian plugin
+ folder rules replace phase3's API-adapter plan (files are the API).
(2) **Idea lineage:** `openaugi lineage "<topic>" [--json --write]`
(pipeline/lineage.py) — semantic evidence over all history, quarter
eras, dormant gaps, third_party flags; `--write` emits
`OpenAugi/lineage/<slug>.json`, the mobile timeline payload (heads-up
filed in mobile's ROADMAP parking lot; mobile also shipped M13 read tab
+ M14 lens chips today in a parallel thread). Lens:
`lenses/idea-lineage.md` (vault + template, `lenses --check` ok) —
distinct from echoes (current thinking → recognition) as topic → full
biography. Live demo: `OpenAugi/lineage/dopamine.json` (100 blocks,
2024-03 → 2026-05). **Next in M8:** ChatGPT-history converter when
Chris re-exports (queued in future-work.md); curator waits for
nomination-answer signal.

**2026-07-07 (late night): milestone reset + surface decision (Chris's
call).** M4 and M5 are CLOSED — no more gating on them; M5 shakedowns are
parked (the try-it checklist lives in the user guide §0, docs/scratch/
2026-07-06-session/user-guide.html — Chris runs it when he can, nothing
blocks on it). **Surface decision: no web app, ever, for now** — the two
surfaces are the mobile app (the one UI we own) and Obsidian as the
desktop app (vault files are the API; the plugin stays a thin task-file
writer). M6 (HTML render) goes from parked to closed as a product
surface; `render/` survives only as internal infra. **Discovery while
resequencing:** the mobile repo is ahead of this plan — its M10 (mobile
review: triage nominations from the phone, answers written back as
capture-block commands), M11 (share-to-LLM bundles), M12 (resurfacing)
are BUILT, pending phone verification — so this repo's M7 is mostly
already delivered from the mobile side; what remains here is
reachability (Tailscale) and verification. **Next build: M8 (data lake
+ curator), starting with the source firewall + first third-party
adapter — see M8 section (resequenced 2026-07-07).** Note: the ChatGPT
export at ~/Downloads/chatgpt-history no longer exists; Chris re-exports
when M8's ChatGPT adapter comes up.

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
weeks of passes; see "M4 posture" note at the end of this doc.

**2026-07-07 (night): cluster-weather lens shipped — first lens with real
machinery.** Assessment first, on the live DB: coarse doc-level k-means
(dims=96, k=10) reproduces the April life-area quality; the config's
block-level HDBSCAN fine passes were confirmed dead (100% noise on cluster
5's 4,072 blocks) — `~/.openaugi/config.toml` rewritten: `concepts` is now
doc-level k-means (k=8, dims=1536, nameable sub-clusters verified),
`cross_domain` commented out. Built on top: per-run
`context_block:cluster_run` snapshots (auto on every committed
`openaugi cluster`), cross-run diffing by member overlap (k-means labels
drift; Jaccard ≥.5 or containment ≥.7), `openaugi cluster-weather [--json]`
report (grew/shrank/born/died + recent_activity from block timestamps —
first run works activity-only), and a temporal fix so doc-level clusters
carry real block_timestamps. Lens spec: vault `lenses/cluster-weather.md`
(mirrored to repo templates), nominations in gravity grammar. Live DB now
has 10 life areas + 80 concept clusters + snapshot #1; Dashboard has a
try-it line. Next weather run gets real deltas. See docs/reference/clustering.md
("Cluster weather"). Remaining M5: habit/tornado lens (needs accumulated
passes).

**2026-07-07 (later): lens system MVP shipped; lifestream parked.**
Chris reframed M5: the lens is the product primitive ("saved questions"),
and the MVP had to be end-to-end before refining individual lenses. Built:
lens registry (`OpenAugi/AGENT/lenses/`, distill + nuggets migrated),
prose engine in `augi-agent.md` (apply/create from any surface), `lenses`
field in the context pack for mobile chips, dormant scheduler note in
review-pass. See M5 below + [docs/reference/lenses.md](../reference/lenses.md). Lifestream
(M6 first screen) got Chris's verdict — no value add — and is parked.
Nomination format meanwhile evolved to checkboxes (other thread).

**Mobile ↔ backend contract (how the repos work together):** the wire
types are pinned in mobile's `shared/contract.ts`; openaugi owns payload
ASSEMBLY (`pipeline/context_pack.py` builds the dict); TRANSPORT is
mobile's choice — today the bridge serves `OpenAugi/context-pack.json`,
and if mobile moves to an HTTP endpoint, that endpoint serves the same
builder's output (nothing here changes). New fields are additive (`lenses`
added 2026-07-07 — mobile ignores it until it renders lens chips).
Cross-repo changes get a heads-up line in each repo's plan STATUS.

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

### M5 — Lens system (MVP SHIPPED 2026-07-07 — [docs/reference/lenses.md](../reference/lenses.md))
**Reframed 2026-07-07 (Chris): the lens is the product primitive** —
saved questions applied to your life data; blocks/routing/surfaces are
substrate and delivery. MVP shipped end-to-end as files + prose:
- **Registry:** one markdown file per lens in `OpenAugi/AGENT/lenses/`
  (name/description/scope/trigger/target + intent body). distill +
  nuggets migrated in; old skill-file paths are pointer stubs.
- **Engine:** the generic apply-lens + create-lens sections in
  `augi-agent.md` (prose, not code). Apply from chat / zzz / task file;
  scope grammar is loose text; explicit scope overrides the spec default.
- **Create from anywhere:** "new lens NAME: INTENT" → agent writes the
  spec file directly (agent-space), `#human-review`, Dashboard note.
- **Mobile:** context pack now carries `lenses: [{name, description}]`
  → app renders apply-chips (tap → `zzz: apply lens X` in block text).
- **Scheduling dormant** until M4 passes; specs already declare triggers.
Lenses shipped 2026-07-07 (vault-side, `OpenAugi/AGENT/lenses/`):
distill · nuggets · **morning-briefing** (daily "what matters today" →
`View - Morning Briefing.md`) · **open-loops** (unclosed commitments →
`View - Open Loops.md`, checkbox = closed). Starter library + JARVIS
rationale: user guide §3b; vision + release-video treatment:
`docs/scratch/2026-07-06-session/vision-jarvis-in-your-pocket.md`
(demo video target: end of week 2026-07-11).
**cluster-weather** shipped 2026-07-07 (night) — the first lens with
deterministic machinery behind it: cluster snapshots + growth/death diffs
(`openaugi cluster-weather`; see docs/reference/clustering.md "Cluster weather").
Remaining M5 work: habit/tornado (needs accumulated passes), spec engine
only if prose visibly fails.

### M6 — Rich render surface (lifestream PARKED 2026-07-07 — verdict: no value add)
Lifestream v1 shipped and Chris's verdict was it doesn't add anything
(plus a heat-strip UTC/local filter bug — not fixed, not worth it).
Gate discipline: built cheap, looked, parked. The `render/` package +
`openaugi render` CLI stay as infrastructure. Next candidate only when a
lens WANTS a visual: cluster map alongside cluster weather. Static
self-contained HTML remains the decided shape if/when revived.

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
