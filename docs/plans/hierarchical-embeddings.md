---
name: hierarchical-embeddings
description: Hierarchical cluster context blocks — re-embed with title-prepend + large model, then agglomerative/HDBSCAN clustering to form a multi-level context layer for self-understanding and agent routing.
---

# Hierarchical Embeddings & Cluster Context Blocks

Extension of [context-block-architecture.md](context-block-architecture.md) Phase 2a.
The goal is two-sided: **self-understanding** (explore your vault as a landscape), then **agent routing** (promote useful cluster levels to navigable context blocks).

---

## What We're Building

A hierarchy of `context_block:cluster` nodes generated offline from embedding space.
Each level is a different granularity cut over the same embeddings:

```
context_block:cluster (coarse, ~10)   ← should map to area/* taxonomy
  --contains--> context_block:cluster (medium, ~40)   ← topic-level
    --contains--> context_block:cluster (fine, ~150+)  ← specific idea clusters
      --groups--> data_block × N
```

Each cluster block has:
- `content` — LLM-generated summary ("what is this cluster about?")
- `title` — short human-readable label ("OpenAugi / Agent Design")
- `metadata.clusters` — method, dims, cut params, member_count, centroid
- `metadata.temporal` — monthly activity histogram, return event count, date range

The hypothesis: **coarse clusters naturally surface the `area/*` taxonomy** — not because we hardcoded it, but because the embedding space independently organizes your vault the same way your intuition did. Testable claim, meaningful if true.

---

## Embedding Change (Done)

**Already shipped:**
- `pipeline/embed.py` now prepends `block.title` to content before embedding:
  `"{note_title}\n\n{chunk_content}"` per OpenAI embedding best practices
- `store/sqlite.py` has `reset_embeddings(kind)` to null embeddings for a full re-embed
- `openaugi re-embed` CLI command to reset + re-embed with current model config

**To kick off:**
```bash
# In ~/.openaugi/config.toml:
# [models.embedding]
# provider = "openai"
# model = "text-embedding-3-large"

openaugi re-embed
```

This switches from 1536 → 3072 dims. `ensure_vec_table` auto-drops and recreates `vec_blocks`.
Store full 3072 dims — truncate locally at cluster time (free), embed once (expensive).

---

## Clustering Approach

### Algorithm choice

**Primary: agglomerative (scipy linkage + fcluster)**
- Run once, get the full dendrogram
- Cut at any height — no re-running for different granularity levels
- O(n²) memory — fine for 20k blocks (~3.2GB at float32, may need batching or sklearn's memory-efficient variant)

**Alternative: HDBSCAN**
- Handles noise/outliers naturally (blocks that don't belong to any cluster)
- Doesn't give a clean hierarchy, but produces more semantically coherent clusters
- Worth running both and comparing

**Not k-means** — requires choosing k, no hierarchy, less coherent clusters for text.

### Dimensionality for different levels

Local numpy truncation of stored 3072-dim vectors:
```python
vec_3072 = np.frombuffer(blob, dtype=np.float32)  # full stored embedding
vec_coarse = vec_3072[:64]    # ~10 clusters, should surface area/* zones
vec_medium = vec_3072[:256]   # ~40 clusters, topic-level
vec_fine   = vec_3072[:512]   # ~150 clusters, specific idea clusters
```

Tag each data block with cluster assignments in metadata:
```json
{
  "cluster_assignments": {
    "agglo_d64_h1.2": 3,
    "agglo_d256_h1.0": 17,
    "agglo_d512_h0.8": 94
  }
}
```

---

## Data Model Fit

No schema changes needed. The two-table schema handles everything.

**New block kind:** `context_block:cluster`

**Link kinds used:**
| Link | Meaning |
|------|---------|
| `context_block:cluster --contains--> context_block:cluster` | coarse → medium → fine hierarchy |
| `context_block:cluster --groups--> data_block` | cluster → member data blocks |

The `contains` and `groups` link kinds already exist from Phase 1.
Traversal via existing `traverse` MCP tool works immediately.

**Cluster block metadata shape:**
```json
{
  "method": "agglo",
  "dims": 64,
  "cut_height": 1.2,
  "level": "coarse",
  "member_count": 847,
  "centroid": null,
  "temporal": {
    "first_block": "2022-01-15",
    "last_block": "2026-03-28",
    "monthly_counts": {"2022-01": 12, "2022-02": 8},
    "return_events": 34
  }
}
```

**Cluster block content (LLM-generated):**
"This cluster captures thinking about OpenAugi's agentic architecture — heartbeat design, context engineering, MCP tool design, and the evolution from a retrieval tool toward a personal intelligence layer. Active from 2023 onward with increasing intensity."

---

## Implementation Sequence

### Step 1 — Exploration script (not pipeline)
A standalone script (not CLI, not a production pipeline step) that:
1. Reads all `data_block` embeddings from the DB as numpy arrays
2. Truncates to target dims (64, 256, 512)
3. Runs agglomerative clustering with scipy
4. Prints cluster sizes, representative blocks, and a dendrogram or silhouette score
5. Writes cluster assignments to block metadata (no context blocks yet)

Goal: understand whether the cluster structure is meaningful before committing to the full pipeline.

### Step 2 — Temporal analysis
Once assignments are in metadata, compute per cluster:
- Monthly activity histogram from `block_time`
- Return events (gaps > 30 days in activity)
- Active period ranges
Render as text or simple ASCII plot.

### Step 3 — LLM cluster summaries
For each cluster, take top N representative blocks (closest to centroid), prompt an LLM:
"Here are N journal/note chunks from a semantic cluster. Write a 2-3 sentence summary of what theme or idea this cluster represents, and note any evolution over time."
Write result as `content` on a `context_block:cluster` block.

### Step 4 — Persist to DB as context blocks
Write cluster blocks + `contains` hierarchy links + `groups` membership links to the DB.
Now agents can use `get_context_map` to orient before searching.

### Step 5 — MCP tools (from context-block-architecture plan)
- `get_context_map` — all context blocks at a given level
- `get_cluster` — one cluster + member block IDs
Existing `traverse` and `get_related` already work on cluster blocks.

### Step 6 — Viewer (future)
Resurrect or build a simple visualizer:
- List/tree view of cluster hierarchy
- Temporal activity plots per cluster
- Toggle between granularity levels

---

## Configurable Pipeline (Future)

Once the exploration phase validates the approach, promote to a CLI command:

```bash
openaugi cluster          # run agglomerative clustering, write context_block:cluster blocks
openaugi cluster --level coarse|medium|fine
openaugi cluster --method agglo|hdbscan
openaugi cluster --refresh   # regenerate only clusters whose membership changed
```

Cluster runs are tagged by method + params so multiple views can coexist in the DB.
A `cluster_run_id` in metadata lets you compare runs or retire old ones.

---

## Open Questions

- Agglomerative at 20k: scipy `linkage` uses ~3.2GB RAM at float32 × 3072 dims (actually it's the pairwise distance matrix that's O(n²) — for n=20k that's 1.6B entries = way too large). Need sklearn's `AgglomerativeClustering` with `memory_efficient=True` or truncate to lower dims first (64 dims for initial exploration is much more tractable: 20k × 64 × 4 bytes = 5MB, pairwise is 1.6GB still but manageable).
- Ward linkage vs. average/complete linkage — Ward tends to produce more equal-sized clusters (better for our use case).
- HDBSCAN min_cluster_size tuning — start with ~50 blocks minimum to avoid noise proliferation.
- Should cluster centroid be stored as a blob in metadata for future similarity queries?
- Refresh strategy: full re-cluster vs. incremental (add new blocks to nearest cluster without full refit).

---

## Related

- [context-block-architecture.md](context-block-architecture.md) — Phase 2a is the cluster context block spec
- [hierarchical-embeddings.md](../../hierarchical-embeddings.md) — original brainstorm doc (workspace root)
