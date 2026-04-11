---
name: clustering
description: Config-driven HDBSCAN clustering pipeline that generates context_block:cluster nodes — a hierarchical map of your vault for self-understanding and agent routing. Documents the config format, data model, SQL queries for each view, and how to tune params.
---

# Clustering

OpenAugi can cluster your vault's data blocks into a hierarchy of `context_block:cluster` nodes — a map of your knowledge at different levels of granularity.

Two uses:
1. **Self-understanding** — see what life areas, topics, and ideas emerge from your notes; track how they evolve over time
2. **Agent routing** — agents orient on the cluster map before searching, enabling Map → Assess → Drill → Retrieve instead of brute-force semantic search

Clustering is offline and optional — it doesn't run on ingest. Run it after embedding is complete.

---

## Quick start

```bash
openaugi cluster              # run all configured passes, write to DB
openaugi cluster --dry-run    # compute and print stats, write nothing
openaugi cluster --pass life_areas  # run one pass only
```

---

## Config

Clustering passes are defined in `~/.openaugi/config.toml` as a list of `[[clustering.passes]]` entries. Each pass has a unique `id` — this becomes the label on every cluster block it creates and the key in each data_block's `cluster_assignments` metadata.

```toml
# Lens 1A — Coarse life areas
# Document-level mean-pooling so long docs (podcasts, book notes) contribute
# one topical vector instead of dominating HDBSCAN density.
[[clustering.passes]]
id = "life_areas"
description = "Coarse life area clusters"
type = "kmeans"
n_clusters = 10
dims = 96
scope = "all"
input_level = "document"
embedding_col = "content_only_embedding"
store_centroid = true

# Lens 1B — Concept/idea clusters within each life area
# Block-level so individual moments of engagement with an idea are preserved.
# block_timestamps in metadata → plot each block as an event on a timeline.
[[clustering.passes]]
id = "concepts"
description = "Recurring concept/idea clusters within each life area"
type = "hdbscan"
min_cluster_size = 15
min_samples = 5
dims = 1536
scope = "within"
parent_pass = "life_areas"
input_level = "block"
embedding_col = "content_only_embedding"
store_centroid = true

# Lens 2 — Unconstrained cross-domain
# No coarse scoping — blocks from different life areas can cluster together.
# bridge_detection surfaces noise points near 2+ centroids (cross-domain ideas).
[[clustering.passes]]
id = "cross_domain"
description = "Unconstrained cross-area connections and bridge ideas"
type = "hdbscan"
min_cluster_size = 20
min_samples = 5
dims = 1536
scope = "all"
input_level = "block"
embedding_col = "content_only_embedding"
store_centroid = true
bridge_detection = true
```

### Pass fields

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique. Becomes the `pass_id` label on all cluster blocks and data_block `cluster_assignments` key |
| `dims` | yes | Truncate stored embeddings to this dim before clustering. Lower = coarser separation. |
| `scope` | yes | `all` = all data_blocks; `within` = run per-cluster on parent's members |
| `min_cluster_size` | yes | HDBSCAN minimum group size. Bigger → fewer, larger clusters |
| `description` | no | Human label |
| `parent_pass` | if scope=within | Pass id whose clusters define the subsets |
| `min_samples` | no | HDBSCAN conservativeness. Bigger → more noise. Defaults to `min_cluster_size` |
| `store_centroid` | no | Store centroid blob in cluster block metadata (default true) |
| `bridge_detection` | no | Find noise points near 2+ centroids (cross-domain bridges, scope=all only) |
| `embedding_col` | no | `"embedding"` (title-prepended, default) or `"content_only_embedding"` (pure content signal) |

### Why HDBSCAN

HDBSCAN doesn't require choosing k (number of clusters). It finds density-based clusters and labels outliers as noise (-1). This matters for personal notes: not every block belongs to a coherent cluster, and forcing k-means assignment produces garbage singletons. Noise points are also information — the `bridge_detection` flag surfaces noise points that sit between two cluster centroids as cross-domain bridge candidates.

### Why matryoshka truncation

`text-embedding-3-large` stores 3072-dim vectors. The first dimensions capture the broadest semantic structure; later dimensions add fine-grained detail. Truncating to 64 dims before coarse clustering gives HDBSCAN a lower-dimensional space where life areas separate cleanly. Running at full 3072 dims within each coarse cluster finds the specific recurring ideas inside that area.

---

## Block granularity

Every `data_block` carries a `metadata.granularity` field set at ingest time by the vault adapter. It records *why* a block was created — information the vault adapter has at parse time but that downstream consumers would otherwise have to re-derive.

| Value | Meaning | Clustering use |
|---|---|---|
| `"document"` | No heading or qqq split — the block IS the whole file (atomic note) | Cluster individually; block == document |
| `"section"` | Split by a heading or `qqq` marker — one intentional structural unit | Cluster individually; each section is a real idea |
| `"document_chunk"` | Length/sentence-based mechanical chunk (future adapters: PDF, web) | Aggregate to document-level mean before clustering |

The vault adapter never produces `"document_chunk"` — every split it makes is structural (heading or qqq). The value is reserved for future adapters where chunking is a mechanical necessity rather than an authorial choice.

**Why this matters for clustering:** `"document"` and `"section"` blocks should cluster individually — each represents one idea. `"document_chunk"` blocks should be mean-pooled per source document before clustering, so a 144-chunk podcast transcript contributes one vector (its topic) rather than 144 vectors (its internal structure).

### Backfill for existing data

If you have a DB ingested before this field was added, run:

```sql
-- Single-chunk documents → "document"
UPDATE blocks
SET metadata = json_set(metadata, '$.granularity', 'document')
WHERE kind = 'data_block'
  AND (
    SELECT COUNT(*) FROM blocks b2
    WHERE b2.kind = 'data_block'
      AND json_extract(b2.metadata, '$.source_path')
          = json_extract(blocks.metadata, '$.source_path')
  ) = 1;

-- Multi-chunk documents → "section"
UPDATE blocks
SET metadata = json_set(metadata, '$.granularity', 'section')
WHERE kind = 'data_block'
  AND (
    SELECT COUNT(*) FROM blocks b2
    WHERE b2.kind = 'data_block'
      AND json_extract(b2.metadata, '$.source_path')
          = json_extract(blocks.metadata, '$.source_path')
  ) > 1;
```

---

## Data model

No schema changes from the base two-table model. Clustering adds new block kinds and reuses existing link kinds.

### Cluster blocks

`kind = "context_block:cluster"`

| Field | Value |
|---|---|
| `id` | `hash("cluster:{pass_id}:{label}")` for scope=all, `hash("cluster:{pass_id}:{parent_label}_{label}")` for scope=within |
| `title` | `{pass_id}_{label}` or `{pass_id}_{parent_label}_{label}` |
| `content` | `null` until agent summarization (future) |
| `source` | `"pipeline:cluster"` |

Metadata on each cluster block:
```json
{
  "pass_id": "life_areas",
  "parent_pass_id": null,
  "parent_cluster_label": null,
  "cluster_label": 3,
  "dims": 64,
  "min_cluster_size": 50,
  "min_samples": 10,
  "member_count": 847,
  "noise_count": 12,
  "centroid": "<base64 float32 blob>",
  "temporal": {
    "first_block": "2022-01-15",
    "last_block": "2026-03-28",
    "block_timestamps": ["2022-01-15", "2022-01-28", "2022-03-04", "..."]
  }
}
```

### Bridge blocks

`kind = "context_block:bridge"` — written only when `bridge_detection = true`.

Metadata: `source_block_id`, `pass_id`, `near_clusters` (list of cluster block IDs + cosine similarity).

### Links

| Link | Meaning |
|---|---|
| `context_block:cluster --groups--> data_block` | cluster membership |
| `context_block:cluster --contains--> context_block:cluster` | coarse → fine hierarchy (scope=within passes) |

### Data block metadata update

After each pass, every non-noise data_block gets `cluster_assignments.{pass_id}` added to its metadata:

```json
{
  "cluster_assignments": {
    "life_areas": "3",
    "life_areas_fine": "3_7",
    "cross_domain": "12"
  }
}
```

Value is `"{label}"` for scope=all, `"{parent_label}_{label}"` for scope=within. Noise blocks get no entry.

---

## Querying clusters

### View one clustering level

```sql
-- Cluster summary for one pass
SELECT cb.title,
       json_extract(cb.metadata, '$.member_count') AS members,
       json_extract(cb.metadata, '$.temporal.first_block') AS first,
       json_extract(cb.metadata, '$.temporal.last_block') AS last
FROM blocks cb
WHERE cb.kind = 'context_block:cluster'
  AND json_extract(cb.metadata, '$.pass_id') = 'life_areas'
ORDER BY CAST(json_extract(cb.metadata, '$.member_count') AS INTEGER) DESC;
```

### Sample block content per cluster

```sql
SELECT cb.title AS cluster,
       db.title AS note_title,
       substr(db.content, 1, 180) AS snippet
FROM blocks cb
JOIN links l ON l.from_id = cb.id AND l.kind = 'groups'
JOIN blocks db ON db.id = l.to_id
WHERE cb.kind = 'context_block:cluster'
  AND json_extract(cb.metadata, '$.pass_id') = 'life_areas'
ORDER BY cb.id, RANDOM()
LIMIT 300;
```

### Temporal activity for a concept cluster (for timeline plots)

```sql
-- Individual block timestamps — join to get one row per engagement event.
-- Plot as scatter/rug/KDE at any time resolution.
SELECT db.block_time,
       db.title,
       db.id
FROM blocks cb
JOIN links l ON l.from_id = cb.id AND l.kind = 'groups'
JOIN blocks db ON db.id = l.to_id
WHERE cb.id = '<concept_cluster_block_id>'
ORDER BY db.block_time;
```

Or read from cluster metadata if you already have it loaded:
```python
timestamps = json.loads(cluster_block.metadata)["temporal"]["block_timestamps"]
# ["2022-01-15", "2022-03-04", "2022-11-18", ...]
```

### All data_blocks with cluster label (for 2D rendering)

```sql
SELECT b.id,
       b.title,
       b.block_time,
       b.embedding,
       json_extract(b.metadata, '$.cluster_assignments.life_areas') AS cluster
FROM blocks b
WHERE b.kind = 'data_block'
  AND json_extract(b.metadata, '$.cluster_assignments.life_areas') IS NOT NULL;
```

This is the primary query for a 2D cluster explorer — pull all data_blocks for a pass, position by embedding (reduce with UMAP/PCA), color by `cluster`.

### Cross-pass hierarchy

```sql
-- Coarse clusters and their fine sub-clusters
SELECT parent.title AS coarse_cluster,
       child.title AS fine_cluster,
       json_extract(child.metadata, '$.member_count') AS members
FROM blocks parent
JOIN links l ON l.from_id = parent.id AND l.kind = 'contains'
JOIN blocks child ON child.id = l.to_id
WHERE parent.kind = 'context_block:cluster'
  AND json_extract(parent.metadata, '$.pass_id') = 'life_areas'
ORDER BY parent.title, members DESC;
```

### Bridge blocks

```sql
SELECT b.title,
       substr(db.content, 1, 200) AS content,
       b.metadata AS bridge_metadata
FROM blocks b
JOIN blocks db ON db.id = json_extract(b.metadata, '$.source_block_id')
WHERE b.kind = 'context_block:bridge'
LIMIT 20;
```

---

## Tuning params

Clustering is idempotent — re-running a pass replaces old cluster blocks for that pass. Tune freely.

**`life_areas` (k-means, document-level, dims=96):**
- 8–15 clusters; start at 10 and adjust `n_clusters`
- Each cluster nameable in 3–5 words from 8–10 sample docs
- Should surface rough life areas: building/projects, finance/trading, personal reflection, research/learning, family, health, content
- No single cluster > 40% of docs

**`concepts` (HDBSCAN within each life area, block-level, dims=1536):**
- 3–12 sub-clusters per parent life area
- Sub-clusters are recurring ideas: "within OpenAugi, I can see agent design, data model, marketing"
- `block_timestamps` in metadata: plot each block as a dot on a timeline to see when you engaged with the idea and how often
- Noise < 30% within each parent (fine clustering is noisier; noise blocks are ones that don't recur)

**`cross_domain` (HDBSCAN, all blocks, dims=1536):**
- More clusters than `life_areas` (15–60 expected)
- Interesting signal: clusters where `note_title` values come from different life areas
- Bridge blocks = noise points near 2+ centroids = ideas that connect different domains

**Param adjustments:**
- `life_areas` too merged → increase `n_clusters`; too fragmented → decrease it
- `concepts` too few sub-clusters → decrease `min_cluster_size`; too many → increase it
- Too much noise in HDBSCAN → decrease `min_samples`
- Concept clusters still within-document → lower `dims` to reduce over-specificity

---

## Implementation

`src/openaugi/pipeline/cluster.py` — all clustering logic.

Key functions:
- `parse_cluster_passes(config)` — parse and validate config
- `run_cluster_dag(store, passes, dry_run)` — entry point; topological sort, execute passes in order
- `load_embeddings(store)` — read embedding blobs into numpy arrays
- `truncate_and_normalize(vecs, dims)` — matryoshka truncation + L2 normalize

Passes with `parent_pass` always execute after their parent (topological sort). Multiple passes with `scope=all` and no parent can run in any order.

See [docs/plans/hierarchical-embeddings.md](plans/hierarchical-embeddings.md) for the full design rationale.
