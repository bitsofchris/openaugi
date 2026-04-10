---
name: knowledge-explorer
description: Interactive WebGL scatter plot for exploring openaugi block embeddings — cluster coloring, timeline playback, and cluster membership detail.
---

# Knowledge Explorer

WebGL-powered visualization of openaugi's embedding space. Adapted from the private knowledge-explorer experiment.

## What it shows

- **Scatter plot** — UMAP 2D projection of `data_block` embeddings, colored by cluster, date, or source note
- **Level selector** — switch between coarse (~10 clusters) / medium (~40) / fine (~150) cuts
- **Detail pane** — click a point to see block content, cluster membership at all levels, and source note
- **Timeline player** — scrub through time to watch the vault grow; heatmap shows monthly activity
- **Noise points** — HDBSCAN outliers shown as small gray points

## Quickstart (frontend only, no DB needed)

```bash
cd frontend
npm install
npm run dev
# Opens http://localhost:5173 using public/fixture.json
```

## With real data (after clustering is done)

```bash
./start.sh [--db ~/.openaugi/openaugi.db]
```

Requires the clustering step to have run and written `cluster_assignments` to block metadata.

## Data contract (`/api/data` or `public/fixture.json`)

```json
{
  "generated_at": "2026-04-10",
  "block_count": 1234,
  "levels": ["coarse", "medium", "fine"],
  "blocks": [
    {
      "id": "block_abc",
      "content": "...",
      "source_path": "journal/2024-03-15.md",
      "source_content": "...",
      "x": 0.42,
      "y": -1.3,
      "date": "2024-03",
      "cluster_coarse": "C_agentic",
      "cluster_medium": "M_3",
      "cluster_fine": "F_7",
      "is_noise": false
    }
  ],
  "clusters": {
    "C_agentic": {
      "id": "C_agentic",
      "level": "coarse",
      "summary": "Thinking about agentic systems...",
      "member_count": 847
    }
  }
}
```

`cluster_*` fields are `null` for noise points. `summary` is `null` until Step 3 (LLM summaries) runs.

## Backend cluster key mapping

The backend reads `cluster_assignments` from block metadata (written by the clustering script):

```json
{
  "cluster_assignments": {
    "hdbscan_d64":  "C_agentic",
    "hdbscan_d256": "M_3",
    "hdbscan_d512": "F_7"
  }
}
```

## Related

- [hierarchical-embeddings.md](../../docs/plans/hierarchical-embeddings.md) — the clustering plan this visualizes
