---
name: cluster-next-steps-prompt
description: Resume prompt after autoresearch clustering is complete — picks up at agent summarization and MCP tooling.
---

# Resume: Clustering Next Steps

Clustering autoresearch is done. Params are locked in `~/.openaugi/config.toml`. Results are in `docs/plans/cluster-results.md`.

Read these docs before doing anything:
- `docs/plans/hierarchical-embeddings.md` — full design and implementation sequence
- `docs/plans/context-block-architecture.md` — Phase 2b (hub summaries) still outstanding
- `docs/clustering.md` — feature doc, SQL queries, data model
- `docs/plans/cluster-results.md` — what the clusters actually look like

## What's done

- Embeddings: title-prepend + text-embedding-3-large (3072 dims), all 20k data_blocks
- `openaugi cluster` CLI command — runs HDBSCAN DAG, writes `context_block:cluster` blocks
- `context_block:cluster` blocks in DB with temporal metadata, centroid, groups/contains links
- `cluster_assignments.{pass_id}` on every data_block metadata

## What's next (in order)

1. **Agent summarization** — `openaugi cluster-summarize` command: spawns a Claude Code agent that reads representative blocks per cluster and writes LLM summaries into `context_block:cluster.content`. Requires a new MCP tool `update_context_block(block_id, content, title)` (write-only to context blocks, not data blocks).

2. **`get_context_map` MCP tool** — returns all context blocks at a given level for agent orientation. Filter by `pass_id` param.

3. **Phase 2b hub summaries** — see `docs/plans/context-block-architecture.md`. Separate from clustering, graph-topology based, infrastructure already exists in `hubs` CLI command.
