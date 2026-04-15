---
name: embedding-research
description: Matryoshka embedding probe experiment — finds the optimal truncation dimension for idea-level semantic retrieval across personal journal blocks.
---

# Matryoshka Embedding Probe Experiment

## Research Question

**At what level of representational granularity do Matryoshka embeddings capture "idea-level" similarity in unstructured personal notes — and can we find that level empirically?**

The corpus is embedded with `text-embedding-3-large` at 3072 dimensions. Because these are Matryoshka embeddings, any prefix is a valid embedding: 32, 64, 128, 256, 512, 1024, 1536, or 3072 dims. The goal is to find the sweet spot where the embedding discriminates *ideas* (fine-grained), not just *topics* (coarse-grained) — without the noise from excess dimensions.

Sub-questions:
- At what truncation dimension does broad topic separation emerge?
- At what dimension does fine-grained idea-level distinction emerge?
- Is there a sweet spot where we get idea-level discrimination without noise from excess dimensions?
- How do cluster quality metrics behave across truncation levels?

## Downstream Application

Once the sweet spot dimension is found:
1. **Retrieval** — truncate incoming blocks to the sweet spot dim, compare via cosine similarity
2. **Time-series** — cluster assignment per block × timestamp → bursty vs. steady idea patterns
3. **Fast classifier** — logistic regression at sweet spot dim for incremental block classification

---

## Phases

### Phase 1: Labeling (`01_sample_and_label.py`)

Sample 200 random `data_block` entries from the SQLite DB. Use an LLM in two passes:

1. **Taxonomy pass** — scan ~30 sampled blocks, propose coarse (5–8 categories) and fine (30–50 themes) taxonomies that emerge from the data.
2. **Labeling pass** — label all 200 blocks against the established taxonomy in batches.

Output: `labeled.json` — **manually review and correct before running Phase 2.**

Format:
```json
[{
  "block_id": "abc123",
  "text": "...",
  "source_path": "journal/2024-03-15.md",
  "date": "2024-03",
  "embedding_3072": [0.012, -0.034, ...],
  "coarse_label": "productivity systems",
  "fine_label": "over-planning failure mode"
}]
```

### Phase 2: Linear Probing (`02_linear_probe.py`)

For each truncation level in `[32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]`:
- Train logistic regression (5-fold CV) to predict coarse label
- Train logistic regression (5-fold CV) to predict fine label

Output: accuracy table + dual-axis plot (log x-scale).

**Key signal:** where the coarse curve plateaus but the fine curve is still climbing = the idea-level sweet spot.

### Phase 3: Cluster Quality Metrics (`03_cluster_quality.py`)

For each `(dims, k)` combination — dims as above, k in `[3, 5, 8, 10, 15, 20, 30, 50]`:
- Run k-means, compute silhouette score, Davies-Bouldin index, Calinski-Harabasz index, inertia
- Run k-means 10× with different seeds → ARI stability score

Output: heatmaps (metric × dims × k) + stability vs. dims plots.

**Key signal:** dimension where ARI plateaus = real cluster structure emerging vs. noise.

### Phase 4: Visualization (`04_visualize.py`)

- Dual-axis probe accuracy plot (Phases 2 output)
- Cluster quality heatmaps (Phase 3 output)
- UMAP projections at 32, sweet-spot, and 3072 dims colored by coarse and fine labels
- Cosine similarity heatmap of 200 blocks sorted by sweet-spot cluster assignment

---

## Running

```bash
cd experiments/embedding_research

# Phase 1 — generates labeled.json for manual review
python 01_sample_and_label.py

# After reviewing/correcting labeled.json:
python 02_linear_probe.py
python 03_cluster_quality.py
python 04_visualize.py   # combines outputs from 2 + 3
```

All scripts default to `~/.openaugi/openaugi.db`. Override with `--db /path/to/other.db`.

Outputs are written to `output/`.

## Dependencies

All already installed via the dev environment:
- `scikit-learn` — logistic regression, cluster quality metrics, cross-validation
- `umap-learn` — UMAP projections for Phase 4
- `matplotlib` / `seaborn` — plots and heatmaps
- `anthropic` — LLM labeling in Phase 1 (uses `ANTHROPIC_API_KEY`)
- `numpy` — truncation, normalization
