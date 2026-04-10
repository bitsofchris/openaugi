---
name: hierarchical-embeddings
description: Hierarchical cluster context blocks — re-embed with title-prepend + large model, then config-driven HDBSCAN clustering DAG to form a multi-level context layer for self-understanding and agent routing.
---

# Hierarchical Embeddings & Cluster Context Blocks

Extension of [context-block-architecture.md](context-block-architecture.md) Phase 2a.
The goal is two-sided: **self-understanding** (explore your vault as a landscape), then **agent routing** (promote useful cluster levels to navigable context blocks).

---

## Embedding Change (Done)

- `pipeline/embed.py` prepends `block.title` to content before embedding: `"{note_title}\n\n{chunk_content}"`
- `store/sqlite.py` has `reset_embeddings(kind)` to null embeddings for a full re-embed
- `openaugi re-embed` CLI command — resets + re-embeds with current model config
- Model: `text-embedding-3-large` (3072 dims), stored full-dim, truncated locally at cluster time

---

## Clustering Strategy

Two passes over the same embeddings, answering different questions.

### Pass 1 — Coarse → Fine (Structure & Temporal Analysis)

**Question:** What are my life areas, and how do ideas within them evolve over time?

- **Step A — Coarse:** HDBSCAN on dim-64 truncated embeddings → ~8-15 life area clusters. Matryoshka truncation: lower dims = broader semantic separation.
- **Step B — Fine:** HDBSCAN on full-dim (3072) embeddings scoped to each coarse cluster's members → specific recurring ideas within each area.
- **Step C — Temporal:** For each fine cluster, activity histogram over time from `block_time`. LLM narrates evolution of interesting ones (future step).

Hierarchy is enforced — cross-area connections suppressed. Depth within areas, not breadth across.

### Pass 2 — Unconstrained (Surprising Connections)

**Question:** What connects across life areas that I haven't consciously noticed?

- **Step A:** Full-dim HDBSCAN across all blocks, no scoping → cross-domain clusters.
- **Step B — Bridge detection:** Noise points (label=-1) near 2+ cluster centroids from different life areas = ideas that straddle domains.
- **Step C — Cross-cluster similarity:** Cosine similarity between fine sub-cluster centroids across coarse boundaries → "this idea in trading rhymes with this idea in personal growth."

### Two Lenses, One Dataset

| | Pass 1 (Coarse → Fine) | Pass 2 (Unconstrained) |
|---|---|---|
| Question | Life areas + within-area evolution | Cross-area connections |
| Hierarchy | Matryoshka truncation (dim-64 → 3072) | HDBSCAN condensed tree |
| Temporal analysis | Primary | Secondary |
| Bridge detection | No | Yes — noise points between clusters |

---

## Config Design

Clustering is configured as a list of named passes in `~/.openaugi/config.toml`.
Each pass has a unique `id` — this becomes the label on every `context_block:cluster` it creates.
`parent_pass` references create the DAG; execution order is topologically sorted.

```toml
[[clustering.passes]]
id = "life_areas"
description = "Coarse life area clusters — should surface area/* taxonomy"
type = "hdbscan"
dims = 64                 # truncate stored 3072-dim vectors to this
scope = "all"             # run on all data_blocks
min_cluster_size = 50
min_samples = 10
store_centroid = true
bridge_detection = false

[[clustering.passes]]
id = "life_areas_fine"
description = "Fine topic clusters within each life area"
type = "hdbscan"
dims = 3072               # full-dim within each coarse cluster
scope = "within"          # run on members of each parent cluster separately
parent_pass = "life_areas"
min_cluster_size = 20
min_samples = 5
store_centroid = true
bridge_detection = false

[[clustering.passes]]
id = "cross_domain"
description = "Unconstrained full-dim clustering — cross-area connections"
type = "hdbscan"
dims = 3072
scope = "all"
min_cluster_size = 30
min_samples = 10
store_centroid = true
bridge_detection = true   # surface noise points near 2+ centroids from different life_areas clusters
```

**Pass fields:**

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique. Used as `pass_id` in all cluster block metadata. |
| `description` | no | Human label, stored in cluster block summary placeholder |
| `type` | yes | `hdbscan` only for now |
| `dims` | yes | Truncate stored embeddings to this dim before clustering |
| `scope` | yes | `all` = all data_blocks; `within` = per-cluster subset from parent |
| `parent_pass` | if scope=within | Pass id whose clusters define the scope |
| `min_cluster_size` | yes | HDBSCAN param — min blocks to form a cluster |
| `min_samples` | no | HDBSCAN param — defaults to `min_cluster_size` |
| `store_centroid` | no | Store centroid blob in cluster block metadata (default true) |
| `bridge_detection` | no | Identify noise points near 2+ clusters (default false) |

---

## Data Model

No schema changes. Two-table schema handles everything.

**Block kind:** `context_block:cluster`

**Link kinds:**
| Link | Meaning |
|---|---|
| `context_block:cluster --contains--> context_block:cluster` | coarse → fine hierarchy (scope=within passes) |
| `context_block:cluster --groups--> data_block` | cluster membership |

**Cluster block shape:**
- `id` — `hash("cluster:{pass_id}:{cluster_label}")` — deterministic, stable across re-runs with same params
- `kind` — `context_block:cluster`
- `title` — `"{pass_id}_{cluster_label}"` (e.g. `life_areas_3`, `life_areas_fine_3_7`)
- `content` — `None` until agent summarization pass (future)
- `source` — `"pipeline:cluster"`
- `metadata`:

```json
{
  "pass_id": "life_areas_fine",
  "parent_pass_id": "life_areas",
  "parent_cluster_label": 3,
  "cluster_label": 7,
  "dims": 3072,
  "min_cluster_size": 20,
  "min_samples": 5,
  "member_count": 94,
  "noise_count": 12,
  "centroid": "<base64 blob or null>",
  "temporal": {
    "first_block": "2022-03-10",
    "last_block": "2026-01-22",
    "monthly_counts": {"2022-03": 3, "2022-04": 1},
    "return_events": 8
  }
}
```

**Data block metadata update:** Each data_block gets `cluster_assignments` added to its existing metadata:
```json
{
  "cluster_assignments": {
    "life_areas": 3,
    "life_areas_fine": "3_7",
    "cross_domain": 12
  }
}
```

**Bridge blocks:** Noise points with `bridge_detection=true` get a `context_block:bridge` written (separate kind) with metadata listing the nearest cluster block IDs and their cosine distances. No `groups` link — a bridge connects, doesn't belong.

---

## Module Design

### `src/openaugi/pipeline/cluster.py`

New module. All clustering logic lives here.

```
load_embeddings(store, kind="data_block") -> dict[str, np.ndarray]
    Read embedding blobs from blocks table, deserialize to float32 arrays.
    Returns {block_id: vec}.

truncate_vecs(vecs, dims) -> dict[str, np.ndarray]
    Slice each vector to [:dims] and L2-normalize (important after truncation).

run_hdbscan(vecs, min_cluster_size, min_samples) -> np.ndarray
    Run HDBSCAN. Returns label array aligned to input order.
    Label -1 = noise.

compute_centroids(vecs, labels) -> dict[int, np.ndarray]
    Mean of member vectors per cluster label. Excludes noise.

compute_temporal(block_ids, store) -> dict
    Pull block_time for each block_id, compute monthly_counts,
    first/last date, return_events (gaps > 30d between sorted dates).

detect_bridges(noise_ids, noise_vecs, centroids, threshold) -> list[BridgeCandidate]
    For each noise point, find top-2 nearest centroids.
    If cosine distance < threshold and centroids are from different parent clusters → bridge.

write_pass_results(store, pass_cfg, block_ids, labels, centroids, temporal) -> int
    Idempotent. Delete existing cluster blocks for this pass_id, then:
    - Insert context_block:cluster for each cluster label
    - Insert groups links (cluster → data_block)
    - Insert contains links (parent_cluster → child_cluster) for scope=within passes
    - Update data_block metadata with cluster_assignments[pass_id]
    Returns count of cluster blocks written.

run_pass(store, pass_cfg, all_vecs, parent_results=None) -> PassResult
    Orchestrates one pass end-to-end.
    For scope=within: loops over parent clusters, runs HDBSCAN per subset,
    collects sub-cluster results.

run_cluster_dag(store, passes) -> None
    Topological sort passes by parent_pass dependency.
    Execute in order, threading PassResult from parent to child.
    Print stats after each pass (see below).
```

### Stats output (printed after each pass)

```
Pass: life_areas (dims=64, min_cluster_size=50)
  Clusters:  11
  Noise:     843 blocks (4.2%)
  Sizes:     min=52  median=187  max=2341
  Date range: 2022-01-01 → 2026-04-08
  Top clusters by size:
    life_areas_0  (2341 blocks, 2022-2026)
    life_areas_3  ( 847 blocks, 2023-2026)
    life_areas_7  ( 412 blocks, 2022-2024)
    ...
```

### Config parsing

Add to `config.py`:
```python
def get_cluster_passes(config: dict) -> list[dict]:
    """Return clustering passes from config, validated."""
    return config.get("clustering", {}).get("passes", [])
```

Validation: each pass has required fields, `parent_pass` references an existing `id`, no cycles.

---

## CLI Command

```python
@app.command()
def cluster(
    db: str | None = typer.Option(None, "--db"),
    pass_id: str | None = typer.Option(None, "--pass", help="Run only this pass id"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print stats, no DB writes"),
    stats: bool = typer.Option(False, "--stats", help="Print current cluster stats from DB, no recompute"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run HDBSCAN clustering DAG, write context_block:cluster blocks to DB."""
```

`--dry-run` runs all clustering computation but skips `write_pass_results`.
`--stats` queries existing cluster blocks from DB and prints a summary without recomputing.

---

## Implementation Sequence

### Step 1 — Config parsing + validation
- Add `[[clustering.passes]]` parsing to `config.py`
- Validate: required fields present, `parent_pass` references exist, no cycles
- Tests: valid config parses correctly, invalid configs raise clear errors

### Step 2 — `pipeline/cluster.py` core
- `load_embeddings` — reads from DB, deserializes
- `truncate_vecs` + normalize
- `run_hdbscan` — thin wrapper, returns labels
- `compute_centroids`
- `compute_temporal`
- Tests: unit test each function with small synthetic inputs

### Step 3 — `write_pass_results`
- Idempotent: delete-then-insert pattern keyed on `pass_id`
- Insert cluster blocks, groups links, contains links
- Update data_block metadata (merge into existing metadata JSON, don't overwrite)
- Tests: write a pass, verify blocks + links in DB; re-run same pass, verify idempotent

### Step 4 — `run_pass` + `run_cluster_dag`
- DAG topological sort
- `scope=within` loop (per parent cluster)
- `detect_bridges` if configured
- Tests: two-pass DAG (coarse + fine) with synthetic embeddings produces expected hierarchy

### Step 5 — CLI command + stats output
- Wire `openaugi cluster` with all flags
- Stats printer for both post-run and `--stats` mode
- Manual test against `~/.openaugi/*.db`

---

## Parameter Tuning

HDBSCAN params (`min_cluster_size`, `min_samples`) are tuned via the config. A separate agent (autoresearch style) will iterate on params by:
1. Running `openaugi cluster --dry-run` to get stats
2. Evaluating cluster quality (size distribution, noise %, coherence of sample titles)
3. Adjusting `min_cluster_size` / `min_samples` in config
4. Repeating until clusters are meaningful

Good signals:
- Coarse pass: 8-15 clusters, noise < 10%, no single cluster > 50% of blocks
- Fine pass: 5-20 sub-clusters per coarse cluster, noise < 20%
- Sizes roughly log-normally distributed (not one giant cluster + many tiny ones)

---

## Future: Agent Summarization

Not in this phase. Once cluster blocks exist in the DB:
- `openaugi cluster-summarize` spawns a Claude Code agent with MCP tool access
- Agent iterates cluster by cluster, reads representative member blocks, writes LLM summary into `context_block:cluster.content`
- Requires a new MCP tool `update_context_block(block_id, content, title)` — write-only to context blocks

---

## Knowledge Explorer (experiments/knowledge-explorer/) — Done

A WebGL scatter plot for exploring cluster embeddings visually.
Built with React + deck.gl (ScatterplotLayer). Works offline against
`public/fixture.json`; wires to the FastAPI backend once clustering is ready.

**What was built:**
- `frontend/` — React/Vite app (no dependencies added beyond deck.gl + react)
  - `ScatterPlot.tsx` — deck.gl scatter plot; noise points gray, cluster points colored
    by stable hash keyed on `{pass_id}:{label}` so colors are cross-pass consistent
  - `DetailPane.tsx` — click a point to see content, source note snippet,
    cluster membership across all passes, temporal info, LLM summary placeholder
  - `FilterBar.tsx` — pass selector (dynamic from `PassInfo[]`), color mode
    (cluster / date / source), search, source filter chip
  - `TimelinePlayer.tsx` — scrubber + playback for temporal animation; shows
    items up to current date when active
  - `ZoomControls.tsx` — zoom in/out/fit buttons
  - `api.ts` — fetches `/api/data`, falls back to `/fixture.json` silently
  - `types.ts` — `Block`, `ClusterInfo`, `PassInfo`, `ExplorerData` matching
    openaugi's actual SQLite schema
- `public/fixture.json` — 300-block synthetic dataset, 2 passes (`life_areas` dims=64,
  `life_areas_fine` dims=512), ~5% noise, matching real `cluster_assignments` format
- `backend/server.py` — FastAPI server: reads `blocks` table for embeddings
  and cluster assignments, runs UMAP 2D projection, returns `ExplorerData` JSON
- `start.sh` — boots backend + Vite dev server together
- `README.md` — quickstart + data contract

**To use against a real DB (once clustering is done):**
```bash
cd experiments/knowledge-explorer
pip install fastapi uvicorn umap-learn numpy
npm install   # once inside frontend/
./start.sh --db ~/.openaugi/openaugi.db
# or: cd frontend && npm run dev  (fixture.json only)
```

**Status:** Frontend works in fixture mode. Backend written but untested against a
real clustered DB — waiting on clustering tuning to complete.

---

## Open Questions

- **Bridge detection threshold** — cosine distance cutoff for "near 2+ centroids." Needs tuning from actual cluster spread. Start at 0.3, adjust.
- **Centroid storage format** — base64 string in JSON metadata vs. separate blob column. Base64 in metadata is simplest (no schema change), small cost for 3072-dim centroids (~12KB each).
- **Refresh strategy** — re-run full pass vs. assign new blocks to nearest centroid. Full refit is correct; approximate is fast. Decide after first successful run.
- **Noise point treatment** — HDBSCAN noise includes both genuine bridge candidates and low-quality fragments (very short chunks, formatting debris). Filter by content length before bridge detection.

---

## Related

- [context-block-architecture.md](context-block-architecture.md) — Phase 2a framing; Phase 2b (hub summaries) is separate and outstanding
- [hierarchical-embeddings.md](../../hierarchical-embeddings.md) — original brainstorm doc (workspace root)
