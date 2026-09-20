---
name: lenses
description: The lens system — saved questions applied to your data. One markdown file per lens in <vault>/OpenAugi/AGENT/lenses/ (scope + trigger + intent + target); applied from any surface via the trigger contract ("apply lens X to SCOPE"); listed in the context pack so mobile can render apply-chips. The engine is prose (augi-agent.md), not code.
---

# Lenses

## When to use this doc

- You want to add, edit, or apply a lens
- You're wiring a new surface (mobile, plugin) to lenses
- You forgot the spec format or scope grammar

Design record: [docs/plans/lens-framework.md](../plans/lens-framework.md).
Live mechanics (the prompt the agent follows): the **Lenses** section of
`<vault>/OpenAugi/AGENT/augi-agent.md`.

## The idea in one paragraph

A **lens** = a saved question applied to your data: *scope + intent →
derived artifact*. The durable value of OpenAugi is not routing or views —
those are plumbing and delivery — it is the growing library of questions
you can re-ask forever ("what nuggets are buried in my notes?", "distill
my thinking on X"). So lenses are **data, not code**: one markdown file
per lens, in the vault, editable like any note. Adding a lens = writing a
file. The "engine" is the agent following the generic apply-lens
instructions in augi-agent.md; a code engine is deliberately deferred
until lens specs visibly outgrow prose.

## The spec — one file per lens (the contract)

**The authoritative, annotated contract is
`src/openaugi/templates/lens-template.md`** (vault copy:
`OpenAugi/AGENT/lens-template.md` — copy it to start a new lens). Same
pattern as the task-file contract: one file defines the format, writers
and readers agree on it, and `tests/test_lens_contract.py` breaks if a
shipped lens or the reader drifts. In brief —
`<vault>/OpenAugi/AGENT/lenses/<name>.md`:

```yaml
---
name: nuggets                # kebab-case, matches filename. REQUIRED (all five are)
kind: engine                 # engine ships as a template | personal never does (AGENTS.md)
description: >-
  What this lens answers, one line (surfaces display this).
scope: >-
  Default retrieval recipe, plain prose (overridable at apply time).
trigger: on-demand           # on-pass / every 7d (no colon!) — `every` is live, see Scheduling
target: >-
  dashboard                  # dashboard | note — <path> | view — overwrite <View - X.md>
---
<intent — the prompt body. Optional persona/reference links.>
```

`description`/`scope`/`target` are folded scalars (`>-`) — bare values
with quotes or `: ` are invalid YAML. Validate: `openaugi lenses --check`.

Shipped lenses: `lenses/distill.md` (topic → one curated note with
provenance), `lenses/nuggets.md` (working notes → 3–7 promotion
nominations), `lenses/cluster-weather.md` (concept-cluster growth/death →
Dashboard nominations; backed by the deterministic pre-compute in
[docs/clustering.md](clustering.md) — `openaugi cluster` +
`openaugi cluster-weather`), `lenses/idea-lineage.md` (one topic → its
full biography in the Persistent Memory Artifact shape; backed by
`openaugi lineage "<topic>" --json --write`, whose
`OpenAugi/lineage/<slug>.json` sidecar doubles as the mobile timeline
payload), and the rest of the engine set — `currency-board`,
`weekly-reflection`, `open-loops`, `chat-harvest`, `echoes`, `emerging`,
`decision-audit`, `create-note-from-block`, `morning-briefing` — every lens
whose vault copy declares `kind: engine`, kept in sync by
`scripts/sync_templates.py`. The old `distill-lens.md` / `nugget-lens.md`
paths are pointer stubs.

## The two axes every lens sits on (formalized 2026-07-08)

Beyond trigger/scope, every lens has two properties that determine how
you run it and what happens to its output. Both are declared in the
existing frontmatter — no new keys — but writers must be deliberate
about them:

**Axis 1 — input: does the lens need a topic?** Stated in `scope`.

- **Batch** — runs over its default scope with no argument; "apply lens
  NAME" is a complete instruction. morning-briefing · open-loops ·
  nuggets · content-pipeline · emerging · cluster-weather ·
  decision-audit (picks its own decision from the stream).
- **Targeted** — meaningless without a subject; the apply instruction
  must carry one ("distill X", "lineage of X"). distill · echoes ·
  idea-lineage. A targeted lens's `scope` MUST say so explicitly
  ("topic given at apply time") so surfaces know a bare chip-tap isn't
  enough (mobile will need a text prompt for these, not just a chip).

**Axis 2 — output mode: what happens to the artifact?** Determined by
the `target` family. Three modes, three persistence behaviors:

| target      | persistence | you see | history |
|-------------|-------------|---------|---------|
| `view — overwrite <View - X.md>` | **cache** — same file overwritten every run | the most recent run only | none, by design (a briefing is a cache, never an archive) |
| `note — <path pattern>` | **artifact** — new dated file per run | every run, dated | accumulates in `OpenAugi/Notes/`; durable (lineage) or disposable (echoes) per lens |
| `dashboard` | **merge** — nominations upserted into `View - Dashboard.md` by `^nom-*` anchor | pending nominations on the shared Dashboard | answered nominations recorded in the next pass's "processed" section; no standalone artifact |

Consequences writers must respect: view-target lenses may use
`overwrite=True` (the only place it's legal); note-target lenses never
overwrite — a re-run on the same topic makes a new dated note;
dashboard-target lenses produce NO file of their own, so their only
trace is the nomination block (same-anchor rule keeps re-runs
idempotent). When creating a lens, pick the mode from the question's
shape: recurring status question → view; durable answer to a one-off
question → note; "should we change structure?" → dashboard.

## Applying a lens — from any surface

Every surface converges on the trigger contract (a task file), so this is
one instruction shape everywhere: **"apply lens NAME"** or **"apply lens
NAME to SCOPE"** (lens names also work naturally: "distill X", "find the
nuggets").

- **Chat:** say it in any Claude session with the openaugi MCP.
- **Any note:** `zzz: apply lens nuggets to this week` — dispatch handles it.
- **Mobile:** the context pack carries `lenses: [{name, description}]`;
  the app renders them as chips — tapping one appends
  `zzz: apply lens <name>` to the block text, which dispatches after sync
  + ingest. Block-scoped lens application with zero mobile-specific server
  work.
- **Plugin (later):** "Apply lens to selection" generalizes the M3b
  "Distill selection" command — selection becomes the scope.

**Scope grammar** (loose text, LLM-interpreted; explicit scope overrides
the spec default): `this block` · `[[Note]]` · `container: <title>` ·
`since: 14d` · `query: <terms>` · or handed/selected context (never
expanded uninvited).

**Targets follow the trust model:** `dashboard` output uses the standard
nomination grammar (checkbox + `^nom-*` anchor + answer slot); `note`
output is one `#human-review` note with provenance; only `view:*` targets
regenerate silently.

## Creating a lens — from any surface

**"new lens NAME: INTENT"** (chat, zzz, mobile capture). The agent writes
the spec file directly — lenses live in agent-space, so no nomination
gate — tagged `#human-review`, with a one-line Dashboard note. You tune a
lens by editing its file; you delete a lens by deleting its file.

**Robustness (productionized 2026-07-07):** broken frontmatter cannot
lose a lens. Invalid YAML (e.g. a description starting with a `"quoted
phrase"` or containing `: `) is **salvaged, not skipped** — the lens
still ships to the context pack with a best-effort name/description, and
a warning lands in the log. Validate any time with **`openaugi lenses`**
(table of every lens + status) or `openaugi lenses --check` (non-zero
exit on broken specs — CI-able). The safe authoring style is folded
scalars: `description: >-` with the text indented on the next line.

## The lens index (`## Lenses` on the Dashboard)

Because the three output modes scatter their artifacts (view caches in
`Views/`, dated notes in `Notes/`, nominations merged into the
Dashboard), there is one place that answers **"what has each lens
produced, and how do I run it?"** — the `## Lenses` section of
`View - Dashboard.md`, one row per lens. Columns: **lens · mode ·
last run · latest output · waiting on you · run it**. The last column is
a launcher — the exact copy-paste phrase, with a `<topic>` placeholder
on targeted lenses so the index doubles as the menu.

**Why on the Dashboard, not its own file.** It lived in
`View - Lenses.md` until 2026-07-11; folded into the Dashboard because
the Dashboard is the single entry point after a pass — a separate index
file was one more surface to remember. "Is there new lens output?" is
already answered by `#human-review`; the unmet need is
*latest-run-per-lens, grouped, in order* — a cache of current state that
now renders as a Dashboard section (upserts edit the section in place,
never the rest of the Dashboard).

**Two mechanisms keep it current, and they reinforce each other:**

1. **Row upsert on apply.** The apply-lens engine (augi-agent.md, step 5)
   updates the running lens's row after writing its artifact — date,
   output link, waiting-on-you note. Works for all three modes, including
   dashboard lenses that have no file of their own.
2. **Full regen from provenance stamps.** Every note/view a lens writes
   carries `lens: <name>` in frontmatter (via `write_document`'s
   `extra_frontmatter`). The review pass rebuilds the section from those
   stamps across `Notes/` + `Views/` when it regenerates the Dashboard —
   a self-heal if upserts drift, and the reason the index is
   reconstructible from disk alone.

Invariant: the index lists exactly the lenses in
`OpenAugi/AGENT/lenses/` — adding/removing a lens file adds/removes its
row on the next regeneration.

## Scheduling — the trigger field, live

`trigger: every <period>` fires. The watcher's drain tick
(`pipeline/watcher.py:_drain_tick`, already running on a timer) lists the
registry, asks `pipeline/schedule.py` what is due, and writes a pending
task file per due lens into `OpenAugi/Tasks/`. The task watcher hydrates
and launches it exactly as it launches a `zzz` dispatch — so a scheduled
run and a typed one are the same mechanism, and `zzz` is just the version
where you are the trigger. No cron, no daemon, no per-surface shell script.

**It is off until you turn it on.** In `~/.openaugi/config.toml`:

```toml
[tasks]
schedule_lenses = true
```

With the gate closed nothing is read and nothing fires — a vault that has
never heard of scheduling behaves exactly as before.

**Periods:** `every 30m`, `every 12h`, `every 1d`, `every 7d`, `every 2w`
(`s m h d w`). `on-demand` means "never auto-fires" and is the default;
`on-pass` is reserved for the review pass and does not fire on the tick. A
trigger that will not parse is **skipped and logged, never guessed at** —
`trigger:` is load-bearing now, and a typo must not become an agent
session. So is a lens whose spec already fails the contract.

### `## Run` — what a scheduled run cannot ask you for

An on-demand apply gets two things from the conversation that a 06:00 run
has no one to ask. Put them in an optional `## Run` section in the lens
body:

```markdown
## Run

Read `OpenAugi/Board/.board-state.json` before building: never re-propose
an item whose state is `done`, `not-doing` or `someday`.

dedupe: OpenAugi/Board/{date} - Board.md
```

- **The prose** is copied verbatim into the task file — state to read
  first, anything the run must know before it starts.
- **`dedupe: <path>`** names the output that proves the run already
  happened. `{date}` expands to the run's date. If that file exists, the
  lens is not due, whatever the records say.

Both are optional; most lenses need neither.

### How "already ran" is decided

Three guards, because the question has three failure modes:

| Guard | Answers | Survives |
|---|---|---|
| `lens_schedule` record (last run per lens) | is the cadence up? | a restart |
| `OpenAugi/Tasks/TASK-{date}-{lens}.md` exists | is it already queued? | a lost database |
| the `dedupe:` output exists | did the work already land? | a re-import, a rebuilt vault |

**The ledger records the slot, not the tick.** The drain tick rides the
debounce, so a run whose boundary lands while the vault is busy fires late.
What gets stamped as `last_run` is the boundary it belongs to — the latest
`previous + n·period` not after now — never the moment the tick happened to
run. A fire forty minutes late therefore does not move the next day's fire
time, and a machine asleep for three days produces **one** catch-up run and
lands back on its own grid, not three runs and a new anchor. A lens that has
never run has no grid yet; its first run is the origin.

### The trade this makes

`launchctl` fired whether or not anything else was running. The drain tick
does not: **if the watcher is stopped, no scheduled lens runs.** What makes
that survivable is that a stopped watcher is no longer invisible: the tick
writes a heartbeat view every five minutes, the Dashboard renders a red line
from it when it goes stale, and `openaugi doctor` prints every scheduled
lens's last run and next due and exits non-zero when the tick is stale. See
[heartbeat.md](heartbeat.md).

## Wiring notes (for surfaces)

- The machine-readable lens list is in `context-pack.json` (`lenses`
  field), built by `src/openaugi/pipeline/context_pack.py` from the lens
  folder's frontmatter. **Transport-agnostic:** the mobile bridge serves
  the file today; a future HTTP endpoint serves the same builder's output.
- Templates for new users: `src/openaugi/templates/lenses/*.md`, copied
  by `openaugi init` (vault copies are the live versions).
