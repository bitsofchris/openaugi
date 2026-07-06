---
name: review-pass-v1
description: The v1 review/maintenance pass — route new blocks to containers, regenerate derived view notes, surface promotion nominations. The first write-back loop.
---

# Review Pass V1 — route blocks, materialize views

**Decided:** 2026-07-06 (session with Claude). Supersedes the Streams subsystem as the
write-back mechanism. Companion vault docs: `OpenAugi/Docs/OpenAugi - The Picture.md`,
`OpenAugi/Docs/OpenAugi - Design Brief - The Note-Thread Context Graph.md`,
`OpenAugi/Research/Build Brief - OpenAugi V1 Maintenance Pass.md`.

## The frame (locked decisions)

Event sourcing / CQRS applied to the vault:

- **Truth** = Chris's own writing (blocks ingested from his notes). Append-only.
  The agent NEVER edits human-authored notes. Blocks are never deleted; things
  demote by ceasing to materialize, not by removal.
- **Views** = agent-generated markdown files under `OpenAugi/Views/`. Derived,
  regenerable, disposable caches. **No review required** — a wrong view is a stale
  cache entry, and wrongness is tuning signal, not damage.
- **Review gate exists only where structure changes:** new tag/area, promoting a
  block cluster to a new silver/gold note, combining notes, taxonomy edits.
  Agent nominates → human commands → agent assembles. No auto-promotion,
  no tuned thresholds.
- **Obsidian stays** as the editor-of-truth and the first renderer. AMOCs embed
  their view via transclusion (`![[OpenAugi/Views/<slug>]]`) — the MOC becomes a
  dashboard over agent-derived data without the agent touching the MOC.
- **Blocks-as-files: parked.** Physical block-level capture arrives with the
  mobile app's capture path, not by retroactively cutting up existing notes.

## Containers (the registry)

The registry is the taxonomy, not a new file: `#note-type/amoc` plus
(`#note-type/pmoc` AND `#status/active`). Starting set — the five AMOCs from
`OpenAugi/AGENT/My Taxonomy.md`:

- AMOC - OpenAugi Main
- AMOC - AI Research Engineer
- AMOC - Bits of Chris - Content Creator
- AMOC - Self - Weaknesses - Jung - Growth
- AMOC - Meta - Productivity Process

plus active PMOCs (e.g. PMOC - Audacity to take Action - Season 2).

## View note contract

Minimal — exactly what `write_document` already auto-generates (skill-file
discipline: description is the contract, body is freeform prompt convention,
never parsed by code):

```yaml
---
description: <one line — what this view shows / the question it answers>
created: <timestamp>
---
```

- No `name` (it's the filename), no `kind` (input to the generation prompt,
  read from the container's `#note-type/*` tag, not restated in output),
  no `derived` flag (**location is provenance** — everything under
  `OpenAugi/Views/` is agent-written by the guardrails).
- Source lineage is a body footer line ("*Generated <date> from N blocks
  since <date>*"), for debugging, not schema.
- Views are regenerated in place: `write_document(..., overwrite=True)`
  (collision-suffix behavior is for notes, not views).

Body by kind (convention, from the Design Brief):
- **area** — rolling TLDR, new-this-period blocks (quoted/linked to source),
  task/idea rollup, links to active child PMOCs.
- **project** — TLDR, **LEFT OFF + next action**, task list, new blocks.
- **concept** — "current understanding" summary, updated on revisit.

Plus `OpenAugi/Views/Dashboard.md`: cross-area rollup (the 6/24 "aggregate
tasks/projects by area" ask), and a **gravity section** — unrouted blocks that
cluster together, nominated for promotion with one line each. Promotion happens
only on Chris's command (inline note or zzz), then the agent assembles the new
note (gathering related old blocks — this is the resurfacing feature).

## Capture grammar & routing logic

Three capture tokens (resist adding more):

- `qqq` — block delimiter (existing)
- `zzz:` — agent dispatch: go do a task (existing)
- `aaa:` — routing/parsing instruction to the review pass (new, prompt-level
  convention, no code): e.g. "aaa: route to OpenAugi Mobile",
  "aaa: find my note on X and link this".

**Routing ≠ surfacing.** Every block routes (cheap: facet tags via `tag_block`);
views surface selectively (salience is the view prompt's job). Life-log blocks
route to `area/self` but don't clutter the head.

Routing precedence (highest wins; multi-routing allowed):

1. `aaa:` instruction — obey it.
2. Explicit `[[MOC link]]` / `#area/*` tag in the block.
3. Location — a block written inside a MOC's journal is home by construction.
4. Inference — classify `area/*` per My Taxonomy; most *specific* container
   wins (active PMOC beats parent AMOC).
5. Low confidence → Dashboard unrouted/gravity section. Never force-fit.

Persistence (amended 2026-07-06 after run #1): **membership = `routed_to`
links** via the `route_block` MCP tool (tags are a closed taxonomy — a
`routed/*` facet was wrong and was migrated to links); **classification =
`augi_tags`** drawn only from My Taxonomy, applied only where there's signal
and never duplicating user tags. Untagged/unrouted is the default for
life-log blocks. Registry notes' `description` frontmatter is the routing
map (skill-file style: when to route here).

**Stable contracts (future-proofing):** (1) append-only truth, (2) block IDs +
tags as routing substrate (mobile becomes just another block writer),
(3) view frontmatter contract, (4) the grammar. Everything else is tunable
prompt-ware.

## The pass (algorithm)

1. Read high-water mark (last-pass timestamp; stored in DB metadata or a state
   file). First run: backfill `--since 2026-06-23`.
2. Ingest/refresh, then pull blocks created since the mark
   (`source/capture` focus; skip previously generated OpenAugi/ artifacts).
3. For each block: classify facets per My Taxonomy (persist via `tag_block`),
   match to containers (explicit links > tags > semantic similarity).
4. Regenerate the view note for every container that received blocks.
5. Regenerate Dashboard (rollup + gravity nominations + anything unroutable).
6. Update high-water mark. Log a one-line run summary.

Trigger: **manual** (CLI command or zzz task) for now. Attach to the Sunday
weekly plan ritual. Schedule only after it's boringly reliable.

## Implementation shape

Agent-first (per repo convention): the pass is an **agent skill**
(`OpenAugi/AGENT/review-pass.md` in the vault, seeded from
`src/openaugi/templates/`) using existing MCP tools (`recent`, `search`,
`get_related`, `tag_block`, `write_document`). Code changes kept minimal:

- [ ] high-water-mark state (small store helper or state file) + date-scoped block pull
- [ ] `write_document` accepts `Views` subfolder (verify path handling)
- [ ] review-pass skill file (the real work: routing + view prompts)
- [ ] Chris adds `![[OpenAugi/Views/<slug>]]` transclusion to each AMOC (manual, once)

Cleanup (separate commit): archive StreamManager + `make_stream`/`update_stream`/
`get_stream_context`/`list_streams` and `write_snip`/`write_thread` with a docs
note (purpose superseded by views / done manually). Keep `write_document`.

## Definition of done (per the operating system)

- **Outcome:** the pass runs over blocks since 6/23 and materializes 5–6 view
  notes + Dashboard, transcluded into the AMOCs.
- **Done when:** run twice (mid-week + Sunday); the Sunday Dashboard answers
  "where did I leave off / what's next per area" without re-reading notes.
- **Kill condition:** if after two runs the views are mush Chris doesn't trust,
  stop and fix routing (block quality / container scope) before adding anything.

## Later (explicitly parked)

- Proactive lenses as additional view types (habit trends, pattern detection,
  "you thought this in March" in-the-moment surfacing)
- HTML/JARVIS dashboard + Augi-the-Otter proactive surface (build on the
  knowledge-timeline/cluster-viewer precedent)
- Scheduled runs; Obsidian plugin review UI (only if inline review proves clunky)
- Blocks-as-files capture (arrives with mobile); mobile M1.5+ contract server
- Curator / self-improving taxonomy (needs accumulated accept/reject signal)
