# Changelog

## Unreleased

**Blocks know who wrote them.** Every data block now carries
`metadata.provenance`: `human`, `ai`, or `reference`, resolved at ingest from an
explicit `provenance/*` tag, then `[vault.provenance_rules]` path globs, then the
AI and `source/*` tag rules. `search` and `get_context` take `provenance=[...]`
in every mode; semantic retrieval drops `[retrieval] exclude_provenance`
(default `["reference"]`) unless the caller names a provenance, so a synced
podcast no longer returns as twenty near-identical hits. `openaugi
backfill-provenance` stamps existing rows. This came out of an analysis run
that quoted forty model-written reflections back to the user as his own
writing because nothing at the query layer could tell them apart
(docs/plans/query-provenance-and-dates.md).

**`get_context` takes filters.** `after`, `before`, `tags`,
`exclude_path_prefix`, `include_path_prefix`, and `provenance`, applied to the
candidate pool before rerank. "What was I thinking about X in March" is now one
call instead of a browse plus a grep.

**Undated notes take the file's creation time, not its last edit.** The last
date fallback read `st_mtime`, which stamped every later edit of an undated MOC
onto its blocks. `st_birthtime` is preferred where the OS has it.

## 0.2.1 — 2026-08-21

**Dependency pins so the package installs.** `mcp>=1.0` was an open range and
`mcp` 2.0.0 shipped, which moves `mcp.server.fastmcp` and drops `readOnlyHint`
from `ToolAnnotations` — so a fresh install resolved to a version that does not
import. Now `mcp>=1.0,<2`; lift that bound with a migration, not by widening it.

`httpx` is also declared explicitly. It is imported directly by
`auth/cloudflare.py` and had been arriving transitively via mcp 1.x; under
2.x that transitive dep became `httpx2` and the import broke. Depend on what
you import.

0.2.0 was tagged but never published — the release workflow gates on the type
check that this failure tripped.

## 0.2.0 — 2026-08-20

The release where OpenAugi stopped being a retrieval library with a write
hook bolted on, and became a loop: capture → route by rule → propose the rest
→ a human answers.

173 commits since `v0.1.0`. The themes, not the commits.

### One query engine

Every deterministic read semantic moved out of the MCP layer into `query/`:
a serializable `QuerySpec` plus an engine, with **three thin adapters over
it** — MCP, a new HTTP `/api/*` on the same daemon, and the CLI. The wire
format is pinned by a 40-case golden harness, so an adapter can't drift from
the engine without a test saying so.

**Saved queries** are markdown files with a `QuerySpec` in frontmatter, with
`-14d` / `today` / `$review-mark` resolved at run time.

### The review pass executes rules; it proposes judgments

The pass used to infer where a block belonged — classify by taxonomy, route
to the most specific match, and put anything low-confidence in a queue. That
queue reached 24 open items containing five actual decisions, and stopped
being read.

Routing now fires on **three deterministic rules only** — an `aaa:`
instruction naming a container, a `[[wikilink]]` to a registered one, a tag
matching a registration. Anything else stays where it was captured. That is
the expected outcome for most blocks and is not a backlog.

Everything requiring judgment — a new note, a merge, a registration — becomes
a **proposal** that is not acted on until a human answers it.

### Records: three generic tools instead of eight bespoke ones

Workflow state (what a run did, what awaits approval) lives in a **collection
store**: `write_record`, `list_records`, `update_record`. A collection is a
name the caller picks; OpenAugi validates nothing and has no schema for it.

This replaced eight named tools — 30% of the MCP surface — added for one
workflow, one of which hardcoded a list of legitimate routing rules. **Schemas
belong in the caller's prompt, policy in the caller's config, only mechanism
in a tool.** `docs/reference/records.md` carries the test for whether a new
tool belongs at all.

MCP surface: 14 tools → 22, despite that removal.

### Recaps are rows, not files

A container's synthesis is a `write_recap` row read through `get_view`, with
staleness reported honestly. Per-container `View - *.md` files are retired —
they were a stand-in from before anything could render a recap.

`docs/reference/recap-spec.md` defines what one contains: only what scrolling
can't give you — cross-window patterns, contradictions, unanswered questions,
what's gone quiet. Not what-moved, not member lists.

### A document is named by `augi_id`, not by its path

Container identity no longer depends on where a file sits, so moving or
renaming a note stops orphaning its membership edges.

### `#layer/bronze` retired

It meant "the user demoted this block" — a salience flag wearing a layer's
name, which collided with a layer model where bronze is the untagged default.
Removed along with its retrieval down-weighting and `[layers]` config.

### Ops — read this if you run the daemons

**`openaugi up` no longer serves MCP.** It is the background service: ingest,
watch, dispatch. `openaugi serve` is the interface. They have opposite
cardinality and this is why they are now firmly separate commands:

- **`up` is singleton**, enforced by an advisory lock. Two watchers over one
  vault both dispatch the same `zzz:`, so one instruction becomes two agents
  racing over one database.
- **`serve` is per-client.** Every stdio MCP client spawns its own; it takes
  no lock and any number can run.

**If a client (Claude Desktop, Codex, an editor) is configured to launch
`openaugi up`, point it at `openaugi serve` instead.** Pass `--serve` to `up`
only for single-terminal use with exactly one client.

Also: the shipped `review-pass.md` template is generic again — it had picked
up references to one user by name.

### Other

- `include_path_prefix` — the mirror of exclude, so a pass can reach one
  folder inside an otherwise-excluded tree.
- `anchor_id`, `ingested_at`, and `routed_to` projected on every block
  summary, so callers resolve anchors and see membership without a second call.
- Task dispatch finds its repo map under `OpenAugi/AGENT/` as well as the
  legacy location; it had been silently resolving zero repos.
- Daily-note template header preamble no longer becomes a block.
