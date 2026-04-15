> **Deprecated** — moved to [`experiments/embedding_research/README.md`](../../experiments/embedding_research/README.md). Implementation scripts live there. This doc is kept as a reference artifact.

# Matryoshka Embedding Probe Experiment

## Project Context

I have a personal second brain — a collection of notes from a pen-only journal, sliced into atomic idea "blocks" (roughly one or two ideas per chunk). The entire corpus has been embedded using OpenAI's text-embedding-3-large at 3072 dimensions. These embeddings support Matryoshka truncation, meaning I can slice them to 32, 64, 128, 256, 512, 1024, 1536, or 3072 dimensions and each prefix is a valid embedding.

I also have a 2D visualizer that projects embeddings via UMAP/t-SNE, and I plan to run k-means clustering at various truncation levels.

---

## The Task

When a new block enters the system (e.g., from today's journal), I want to surface semantically related blocks from my history — even if they use different words, were written weeks or months apart, or approach the idea from a different angle. This is **semantic near-duplicate detection across time**, but softer than traditional near-duplicate. I'm looking for ideas that *rhyme*, not ideas that match.

---

## Research Question

**At what level of representational granularity do Matryoshka embeddings capture "idea-level" similarity in unstructured personal notes — and can we find that level empirically?**

Sub-questions:
- At what truncation dimension does broad topic separation emerge?
- At what dimension does fine-grained idea-level distinction emerge?
- Is there a sweet spot where we get idea-level discrimination without noise from excess dimensions?
- How do cluster quality metrics behave across truncation levels?

---

## Experimental Design

### Phase 1: Labeling

**Input:** ~200 randomly sampled blocks from the corpus (with their text content).

**Process:**
1. Use an LLM agent to label each block at **two granularities**:
   - **Coarse labels** (5–8 categories): Broad life/thinking domains. Examples might include "productivity systems," "relationships," "health," "creative projects," "learning/education," "career/work," "philosophy/worldview." These should emerge from the data — let the agent scan a sample first and propose the taxonomy, then apply it.
   - **Fine labels** (30–50 categories): Specific recurring ideas or themes. Examples might include "tension between structure and spontaneity," "morning routines and energy," "creative resistance," "note-taking workflows," "compounding knowledge." Again, let the agent propose these from the data.
2. **Manually review and correct all 200 labels.** This is the gold standard. Let the LLM do the first pass, then edit/correct. Pay attention to cases where the LLM grouped things in ways you wouldn't have — those disagreements are informative.

**Output:** A labeled dataset of ~200 blocks, each with a coarse label and a fine label. Store as JSON or CSV:

```json
{
  "block_id": "abc123",
  "text": "...",
  "embedding_3072": [0.012, -0.034, ...],
  "coarse_label": "productivity systems",
  "fine_label": "over-planning failure mode"
}
```

---

### Phase 2: Linear Probing

**Goal:** Measure how much semantic structure the embedding captures at each truncation level by training a simple linear classifier (logistic regression) and measuring accuracy.

**Truncation levels to test:** 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072

**Procedure (repeat for each truncation level):**

1. Truncate all 200 embeddings to the target dimension (just slice the first N values).
2. **Coarse probe:** Train a logistic regression classifier to predict the coarse label from the truncated embedding. Use **5-fold cross-validation**. Record mean accuracy and standard deviation.
3. **Fine probe:** Train a logistic regression classifier to predict the fine label from the truncated embedding. Use **5-fold cross-validation**. Record mean accuracy and standard deviation.

**Implementation notes:**
- Use `sklearn.linear_model.LogisticRegression` with default settings (or `max_iter=1000` if convergence warnings appear).
- Use `sklearn.model_selection.cross_val_score` with `cv=5`.
- Normalize/standardize embeddings before training (important since we're comparing across different dimensionalities).

**Output:** A table and plot:

| Dimensions | Coarse Accuracy (mean ± std) | Fine Accuracy (mean ± std) |
|------------|------------------------------|----------------------------|
| 32         | 0.XX ± 0.XX                  | 0.XX ± 0.XX                |
| 64         | ...                          | ...                        |
| ...        | ...                          | ...                        |

Plot both curves on the same graph (x = truncation dimension, y = accuracy). Use log scale on x-axis since the dimensions are roughly geometric.

---

### Phase 3: Cluster Quality Metrics

**Goal:** Independently measure how well-defined the clusters are at each truncation level, without relying on labels.

**Procedure (repeat for each truncation level and a range of k values):**

For k-means with k in [3, 5, 8, 10, 15, 20, 30, 50]:

1. Truncate embeddings to target dimension.
2. Run k-means.
3. Compute:
   - **Silhouette Score** — how similar each point is to its own cluster vs. nearest neighbor cluster. Range [-1, 1], higher is better.
   - **Davies-Bouldin Index** — ratio of within-cluster scatter to between-cluster separation. Lower is better.
   - **Calinski-Harabasz Index** — ratio of between-cluster to within-cluster variance. Higher is better.
   - **Inertia** — within-cluster sum of squares (the elbow method value).
4. Also compute **stability** via Adjusted Rand Index: run k-means 10 times with different random seeds and compute pairwise ARI between all runs. Report mean ARI as a stability score.

**Output:** Heatmaps or line plots showing each metric as a function of (truncation dimension, k).

---

### Phase 4: Interpretation & Visualization

**Key things to look for:**

1. **The gap between curves.** Where the coarse probe has plateaued but the fine probe is still climbing — that's the region where the embedding is learning to distinguish ideas *within* the same broad topic. This is the sweet spot for the retrieval use case.

2. **Phase transitions.** Any sharp jumps in probe accuracy or cluster quality suggest a dimensionality where meaningful semantic structure "clicks into place."

3. **Cluster stability plateau.** The dimension where k-means starts producing consistent clusters (high ARI across runs) tells you where there's real structure vs. noise.

4. **Elbow comparison.** Compare inertia elbow plots at different truncation levels. If the elbow sharpens as dimensionality increases, structure is emerging.

**Visualizations to generate:**
- Dual-axis probe accuracy plot (coarse + fine vs. dimensions)
- Cluster quality metric heatmaps (metric × dimensions × k)
- Stability (ARI) vs. dimensions for a few key values of k
- 2D UMAP projections at a few key truncation levels (e.g., 32, the sweet spot dimension, and 3072) colored by coarse and fine labels
- Cosine similarity heatmap of all 200 blocks sorted by cluster assignment at the sweet spot dimension

---

## Downstream Application

Once the sweet spot dimension is identified:

1. **Retrieval system:** New blocks get embedded, truncated to sweet spot dimension, and compared against existing block embeddings via cosine similarity or nearest-centroid lookup.
2. **Time-series analysis:** Assign each block a timestamp and cluster ID. Analyze inter-arrival times between blocks in the same cluster to find bursty vs. steady idea patterns.
3. **Incremental classification:** The trained logistic regression at the sweet spot dimension can serve as a fast classifier for incoming blocks — assign them to known clusters or flag them as potentially novel.

---

## File & Data Requirements

- **Embeddings file:** All blocks with their full 3072-dim embeddings (the truncation happens in code by slicing).
- **Block metadata:** Block ID, source document, timestamp, any existing tags/titles, the raw text.
- **Labels file (created in Phase 1):** The 200 labeled blocks with coarse and fine labels.

---

## Success Criteria

The experiment is successful if we can:
1. Identify a truncation dimension that maximizes fine-grained probe accuracy without significant gains beyond it.
2. Show that cluster quality metrics corroborate the probe results.
3. Demonstrate that retrieval at the sweet spot dimension surfaces meaningfully related blocks that a human would agree are "idea rhymes."
