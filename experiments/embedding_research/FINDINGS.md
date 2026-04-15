---
name: embedding-research findings
description: Running results log for the Matryoshka embedding probe experiment — updated after each phase.
---

# Embedding Research Findings

Corpus: 200 randomly sampled `data_block` entries from `~/.openaugi/openaugi.db` (seed=42).
Embeddings: `text-embedding-3-large` at 3072d (Matryoshka — any prefix is valid).

---

## Phase 1 — Taxonomy & Labels

**Taxonomy discovered by Claude from 30 sampled blocks:**

Coarse (7 categories):
| Label | Count |
|---|---|
| Content creation & personal brand | 48 |
| Building OpenAugi & technical craft | 43 |
| Identity, psychology & self-development | 38 |
| Career & financial direction | 24 |
| Knowledge systems & second brain | 17 |
| Productivity, attention & work rhythms | 16 |
| Relationships, family & community | 14 |

Fine: 40 themes (collapsed to 39 after labeling — see `output/taxonomy.json`).

**Note:** Fine taxonomy has some singleton classes (1 example). These should be merged before re-running Phase 2. Recommend collapsing to ~20 themes.

---

## Phase 2 — Linear Probe

**Method:** Logistic regression (saga solver, C=1.0, max_iter=200), 3-fold stratified CV.
Each truncation level: slice embedding to `dims`, L2-normalize, train/eval.

**Results:**

| Dims | Coarse acc | Fine acc |
|------|-----------|---------|
| 32 | 47.0% | 20.0% |
| 64 | 55.0% | 22.0% |
| 96 | 56.5% | 23.5% |
| 128 | 58.1% | 25.5% |
| 192 | 60.0% | 25.5% |
| 256 | 63.0% | 25.0% |
| 384 | 63.0% | 24.0% |
| 512 | 62.5% | 24.0% |
| 768 | **64.0%** | 23.5% |
| 1024 | 62.5% | 24.0% |
| 1536 | 63.5% | 25.0% |
| 2048 | 63.5% | 25.0% |
| 3072 | 63.5% | 24.5% |

**Key findings:**

- **Coarse accuracy** peaks at 768d (64%) and flatlines — broad topic separation saturates there.
- **Fine accuracy** peaks at 128–192d (25.5%) and slightly degrades at higher dims — extra dimensions add noise, not signal, for idea-level discrimination.
- **Sweet spot for idea-level retrieval: 128–256d.** Fine discrimination peaks in this range.
- Fine accuracy ceiling is low (25.5%) partly because 39 labels across 200 blocks leaves singleton classes — CV can't learn them. Merging rare fine labels will give a cleaner signal.

**Plot:** `output/probe_plot.png`

---

## Phase 3 — Cluster Quality Metrics

**Method:** k-means (n_init=3, max_iter=100), 5 runs per (dims, k) for ARI stability.
Grid: dims ∈ [32, 64, 128, 256, 512, 768, 1024, 1536, 3072], k ∈ [3, 5, 7, 10, 15, 20].
Metrics: silhouette (cosine), Davies-Bouldin, Calinski-Harabasz, pairwise ARI across runs.

**Best silhouette per dimension (always at k=3):**

| Dims | Silhouette | ARI stability |
|------|-----------|--------------|
| 32 | 0.103 | 0.515 |
| 64 | 0.086 | **0.666** |
| 128 | 0.086 | 0.484 |
| 256 | 0.076 | 0.515 |
| 512 | 0.082 | **0.680** |
| 768 | 0.075 | 0.467 |
| 1024 | 0.070 | 0.442 |
| 1536 | 0.067 | 0.395 |
| 3072 | 0.065 | 0.367 |

**Key findings:**

- **Silhouette scores are uniformly low** (max 0.103) — this corpus does not form tight, well-separated clusters at any dimension. Ideas bleed into each other. This is expected for a personal journal: entries are interconnected, not categorically distinct.
- **ARI stability peaks at 64d and 512d** (0.666, 0.680 at k=3) — these are the dimensions where k-means finds the same 3-cluster solution most consistently across random restarts. That's meaningful structure even if the clusters aren't tight.
- **Quality degrades monotonically beyond 512–768d** — more dimensions hurt cluster consistency. Aligns with Phase 2: noise enters above ~512d.
- **k=3 dominates throughout** — the data has ~3 naturally stable groupings at every dimension, roughly matching the 3 biggest coarse categories (content/building/identity).
- **Davies-Bouldin rises with dims** — clusters become more diffuse in high-dimensional space, as expected from the curse of dimensionality.

**Convergence with Phase 2:** Both probes point to **64–256d as the sweet spot**:
- Fine-label probe accuracy peaks at 128–192d
- ARI stability peaks at 64d and 512d
- Silhouette best at 32–64d

**Plots:** `output/cluster_heatmaps.png`, `output/cluster_stability.png`

---

## Phase 4 — Visualization

**Plots generated:**
- `output/viz_probe_and_stability.png` — side-by-side: probe accuracy (Phase 2) + ARI stability (Phase 3)
- `output/viz_umap_triptych.png` — UMAP at 64d / 256d / 3072d coloured by coarse label
- `output/viz_cosine_heatmap.png` — cosine similarity matrix at 256d, sorted by coarse label

---

## Conclusion

**Recommended sweet spot: 64–256d**

All three phases converge on the same range:

| Signal | Sweet spot |
|---|---|
| Fine-label probe accuracy peak | 128–192d |
| Coarse-label probe plateau starts | 256–512d |
| ARI stability peaks (k=3) | 64d, 512d |
| Silhouette peak | 32–64d |

**What this means for OpenAugi:**
- Truncate to **256d** for retrieval — captures idea-level similarity, cheap to compute, well below the noise floor
- For broad topic clustering (coarse), **512d** gives slightly more stable assignments
- Full 3072d adds no measurable signal for either task and actively degrades cluster stability
- The corpus has ~3 naturally stable groupings regardless of dimension — it is not a neatly segmented collection; ideas connect across domains

---

## Phase 5 — Dense Hub Probe (cleaner ground truth)

**Motivation:** Phase 2 used 200 random blocks with LLM-assigned labels — too sparse (39 labels, singletons). This uses vault links as ground truth: every block that `links_to` a hub IS a member of that hub's topic cluster, deliberately linked by the author.

**8 hubs selected** for topical distinctness and block count:

| Hub | Blocks |
|---|---|
| jung_self | 83 |
| parenting | 81 |
| niche_knowledge | 79 |
| priorities_vision | 68 |
| physical_health | 52 |
| embedding_research | 49 |
| victory_feedback | 44 |
| overthinking | 33 |

**489 blocks total, 8 classes, 3-fold CV.**

**Results:**

| Dims | Accuracy | Within-hub cosine | Between-hub cosine | Gap |
|------|---------|-------------------|--------------------|-----|
| 32 | 65.8% | 0.635 | 0.495 | **0.139** |
| 64 | 71.2% | 0.618 | 0.497 | 0.122 |
| 96 | 70.8% | 0.667 | 0.563 | 0.104 |
| 128 | 71.2% | 0.646 | 0.535 | 0.111 |
| 192 | 72.6% | 0.631 | 0.515 | 0.117 |
| 256 | 72.4% | 0.612 | 0.488 | 0.123 |
| 384 | 73.2% | 0.604 | 0.484 | 0.120 |
| 512 | 73.8% | 0.597 | 0.475 | 0.122 |
| 768 | **74.6%** | 0.585 | 0.464 | 0.121 |
| 1024 | 74.6% | 0.581 | 0.459 | 0.121 |
| 1536 | 74.4% | 0.577 | 0.458 | 0.119 |
| 3072 | 74.6% | 0.558 | 0.438 | 0.120 |

**Key findings — this changes the picture:**

- **Probe accuracy keeps climbing all the way to 768d** and plateaus there. More dimensions genuinely help a classifier distinguish hubs — because hubs are broad topic areas, not fine ideas. This matches Phase 2's coarse plateau at ~768d.

- **The similarity gap (within − between) peaks at 32d (0.139) and shrinks with more dims.** This is the more important signal. At 32d, same-hub blocks are proportionally most distinguishable from cross-hub blocks. As you add dims, the absolute cosine values fall (more dimensions = smaller dot products), but the gap narrows too — the "same topic" signal is actually clearest at very low dims.

- **The two metrics are measuring different things:**
  - Accuracy measures: can a classifier learn to separate 8 broad topics? → needs ~512–768d
  - Similarity gap measures: do same-idea blocks naturally score higher similarity? → peaks at 32–64d

- **Absolute cosine values fall with dims:** at 32d, within=0.635; at 3072d, within=0.558. The embeddings become sparser and harder to distinguish at higher dims despite the classifier learning to separate them — this is the curse of dimensionality showing up.

**Revised interpretation:** These 8 hubs are broad topic areas (parenting, physical health, Jung, etc.) — they're *more like coarse labels* than fine idea clusters. The classifier needs ~512–768d to separate them because they're genuinely different domains. But if you want same-idea retrieval at the atomic block level, the gap signal suggests **64–128d** is where same-topic blocks are most relatively similar.

**What's still missing:** A test using *fine* idea clusters — e.g., 5–10 blocks all about the same specific concept (morning walks, a specific project decision, one book's ideas). That would tell you the retrieval dim for actual idea-level search.

**Plots:** `output/dense_probe_accuracy.png`, `output/dense_similarity_gap.png`

---

## Conclusion (updated)

| Task | Recommended dims | Evidence |
|---|---|---|
| Coarse topic separation (8 broad hubs) | 512–768d | Phase 5 probe accuracy plateau |
| Broad-topic retrieval | 256d | Phase 2+3 convergence, Phase 5 gap still strong |
| Idea-level retrieval (hypothesis) | 64–128d | Phase 5 similarity gap peaks at 32–64d; Phase 2 fine acc peaks at 128–192d |

Full 3072d adds no measurable benefit for any task and increases noise.

---

## Phase 6 — Within-Hub Cluster Stability

**Method:** For each hub independently, run k-means within just that hub's blocks across dims. ARI stability = how consistent the clusters are across 6 random restarts.

**Peak dims per hub:**

| Hub | Peak dim | ARI | Note |
|---|---|---|---|
| priorities_vision | 192d | **1.000** | Perfect stability — very tight structure |
| victory_feedback | 192d | 0.827 | Strong |
| parenting | 64d | 0.705 | Peaks early, falls off |
| physical_health | 768d | 0.729 | Broad topic, needs more dims |
| jung_self | 1536d | 0.612 | Abstract/interconnected, needs high dims |
| overthinking | 384d | 0.621 | Mid-range |
| niche_knowledge | 512d | 0.515 | Moderate |
| embedding_research | 1536d | 0.492 | Research notes, scattered |

**Median peak dim: 448d**

**Key finding:** No single sweet-spot dimension. Different topic areas peak at radically different dims:
- **Concrete, time-stamped topics** (priorities, wins, parenting) peak early at 64–192d — the ideas are specific and the structure is tight
- **Abstract, interconnected topics** (Jung, overthinking, research) need 512–1536d — the concepts bleed into each other and need high-dimensional signal to distinguish

k=3 dominates everywhere — within a hub, there are ~3 stable sub-clusters regardless of topic.

**Implication:** A fixed retrieval dimension is probably wrong. The right approach is topic-adaptive — short dims for concrete hubs, longer dims for abstract ones.

---

## Phase 7 — Block Length Stratification

**Method:** Split all 15,741 data_blocks into length tertiles:
- short: ≤263 chars (~1–2 sentences)
- medium: ≤597 chars (~1 paragraph)
- long: >597 chars (~multi-paragraph)

300 blocks sampled per tertile. k-means (k=5,8,12) across dims.

**Mean ARI across all dims and k values:**

| Tertile | Mean ARI |
|---|---|
| short | 0.314 |
| medium | 0.382 |
| **long** | **0.475** |

**This is the opposite of the hypothesis.** Long blocks cluster *better* than short ones at every dimension. The chunking-is-the-confound hypothesis is wrong for this corpus.

**Why:** Long blocks in this vault are synthesized reflections — they develop one idea in depth and stay on-topic. Short blocks (≤263 chars) tend to be fragments, raw voice notes, or isolated observations that lack enough context for the embedding to place them precisely. The embedding model performs better when given more textual context, not less.

**Implication for OpenAugi:** Do not pre-filter to short blocks for clustering. Medium-to-long blocks (~200–600 chars) are actually the most embeddable. Short fragments may benefit from expansion or context injection before embedding.

---

## Conclusion (final)

| Question | Answer | Evidence |
|---|---|---|
| Best single retrieval dim? | **256–512d** — good trade-off for most hubs | Ph2+3+5 convergence |
| Fine idea sweet spot (within topic)? | **Depends on topic** — 64–192d for concrete, 512–1536d for abstract | Ph6 per-hub peaks |
| Does chunk length hurt clustering? | **No — long blocks cluster better** | Ph7: long ARI=0.475 > short ARI=0.314 |
| Is chunking a confound? | **For short fragments yes** — ≤263 chars too sparse for precise embedding | Ph7 |
| How many natural sub-clusters per hub? | **~3** consistently | Ph3, Ph6 |
| Does full 3072d help? | **No** — adds noise for most tasks | All phases |

---

## Action Items

- [ ] Merge singleton fine labels in `output/labeled.json` (aim for ~20 themes, min 5 examples each)
- [ ] Re-run Phase 2 with merged fine labels for a cleaner fine-accuracy curve

- [ ] Merge singleton fine labels in `output/labeled.json` (aim for ~20 themes, min 5 examples each)
- [ ] Re-run Phase 2 with merged fine labels for a cleaner fine-accuracy curve
- [ ] Complete Phase 3 cluster quality metrics
- [ ] Confirm sweet spot dimension empirically from Phase 3 stability plots
