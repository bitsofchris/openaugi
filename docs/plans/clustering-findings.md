---
name: clustering-findings
description: What we tried, what failed, what worked, and why — coarse and fine clustering experiments on a 4,226-doc Obsidian vault. Includes block vs document-level embedding analysis and final config recommendation.
---

# Clustering Findings

Experiments run April 2026 on a vault of ~20,015 `data_block`s across 4,226 source documents.
Embedding model: `text-embedding-3-large` (3072 dims, Matryoshka).

---

## What we tried

### 1. HDBSCAN at block level (coarse pass)

Config variations: dims 64–256, min_cluster_size 20–100.

**Result: 85–87% noise at all settings.**

Root cause: the vault has many long documents (podcast transcripts, book notes) with 50–150 sections each. HDBSCAN is density-based — it found within-document density, not semantic life areas. A 144-chunk transcript creates a tight density cluster of near-identical vectors that dominates the neighborhood graph. Everything else became noise.

Increasing dims made it worse (O(n²) distance computation; at dims=256 HDBSCAN ran for 1h25m and was killed). Decreasing dims produced 2 mega-clusters.

**Conclusion: HDBSCAN at block level does not work for coarse clustering on heterogeneous vaults.**

---

### 2. K-means at block level (coarse pass)

Config variations: dims 64–512, k 5–20.

**Result: Better, but distorted by long documents.**

K-means assigns every point, so no noise problem. At dims=96, k=10 results were recognizable life-area clusters. But the cluster sizes were driven by block counts, not document counts — a 150-block podcast got 150 votes vs a 1-block atomic note getting 1 vote.

**Conclusion: Works, but unfair weighting. Document-level is cleaner.**

---

### 3. Document-level mean-pooling + k-means (coarse pass) ✓ WINNER

Each source document's blocks are mean-pooled into one vector before clustering.
4,226 vectors instead of 20,015.

**Why this is correct:**
- A podcast transcript contributes one topical vector (its overall subject) not 150 near-duplicate vectors
- Atomic notes (1 block = 1 doc) are weighted equally to long notes
- This is what you actually want: "which life area does this document live in?"

**Params explored:** dims 64–512 × k 5–20.

**Winner: dims=96, k=10**

| Cluster | Docs | % | Name |
|---|---|---|---|
| 0 | 509 | 12% | Content creation / public second brain |
| 1 | 253 | 6% | Personal life / family / social |
| 2 | 243 | 6% | Active work log (trading, startup decisions) |
| 3 | 267 | 6% | CS/infra fundamentals |
| 4 | 519 | 12% | Weekly/monthly check-in journals |
| 5 | 565 | 13% | OpenAugi dev + productivity system |
| 6 | 255 | 6% | ML/deep learning learning notes |
| 7 | 580 | 14% | Writing craft / life philosophy |
| 8 | 410 | 10% | AI + augmented intelligence |
| 9 | 624 | 15% | Career direction / self-reflection |

All 10 nameable. No degenerate clusters. Even distribution (243–624 docs).

**Why dims=96:**
- Matryoshka: the first ~100 dims of `text-embedding-3-large` encode the broadest semantic signal — life areas separate cleanly here
- dims=64: too compressed, creates mega-clusters (1 cluster = 29% of all docs)
- dims=128+: begins splitting "journal check-ins" from "topic-titled journal entries" into two separate clusters — same life area, different writing style. This is a false split.
- dims=256+: one giant catch-all cluster (18-32% of docs); the fine-grained signal overwhelms the coarse structure

---

### 4. HDBSCAN at block level (fine pass — within a coarse cluster)

Tested within coarse cluster 7 ("writing/life philosophy", 1517 blocks, 402 unique docs).

Config: dims=1536 and 3072, min_cluster_size=15 and 25.

**Result: 60–66% noise. Clusters are individual long documents.**

Every found cluster was just one long document's sections:
- Cluster 0: all sections of "The Out of Sync Child" (book notes)
- Cluster 1: all sections of "Marriage Patterns Analysis"
- Cluster 7: all sections of "Writing Books with James Clear and Mark Manson"

Cluster 7 has 11 documents with 20+ blocks (avg 43 blocks each) — those 11 docs account for ~31% of blocks. They dominate HDBSCAN density at any reasonable min_cluster_size.

The "noise" (63%) was everything else: single-block notes, journal entries, standalone ideas — exactly what you want to cluster.

**Conclusion: HDBSCAN at block level doesn't work for fine clustering either, for the same root cause as coarse.**

---

### 5. K-means at block level (fine pass — within a coarse cluster)

Tested within cluster 7: dims=1536 and 3072, k=5, 8, 10.

**Result: mixed — some good clusters, some are just one long document.**

At dims=1536, k=10, the good clusters:
- Writing craft (439 blocks): Julian Shapiro, Why I Write, writing lead with specific — **multiple documents, recurring theme** ✓
- Productivity/philosophy books (241 blocks): Seth Godin, 30-day experiments, Algorithms to Live By — **multiple docs** ✓
- Parenting/family (208 blocks): Ryan's school talk, neurodivergence, wife stories — **multiple docs** ✓

The bad clusters:
- "A conversation on focus..." (123 blocks): one 121-block podcast transcript — **one document repeated** ✗
- "Writing Books with James Clear..." (137 blocks): one 56-block transcript — **one document, just happens to be big** ✗

**The good clusters ARE finding recurring ideas across documents and time.** Writing craft notes written in 2025 cluster with book notes from 2024 and journal entries from early 2026. The algorithm found "you keep returning to this theme" — which is exactly what we want.

**But long documents still contaminate.** The 121-block podcast gets 8× the weight of a 15-block note, so it becomes its own cluster.

---

## Are block-level embeddings useful at all?

**Not for clustering.** At any granularity (coarse or fine), long documents dominate density estimation (HDBSCAN) or centroid computation (k-means), drowning out the actual semantic structure.

**Yes for retrieval.** When an agent or search query needs the most relevant *passage* back, block-level embeddings are exactly right — you want to return the specific section of a note that matches, not the whole document. This is the sqlite-vec ANN search path, unchanged.

**The right level for each task:**
| Task | Right level | Why |
|---|---|---|
| Coarse life-area clustering | document (mean-pooled) | One topical vote per document |
| Fine recurring-idea clustering | document (mean-pooled) | Same reason |
| Semantic retrieval | block | Return the relevant passage |
| Bridge detection (cross-domain) | document (mean-pooled) | Find docs that straddle two life areas |

---

## What to try next for fine clustering

**Use document-level mean-pooling within each coarse cluster, then k-means.**

For cluster 7 (580 docs after mean-pooling), dims=1536, k=8 should give:
- Writing craft / process
- Storytelling theory
- Life philosophy / personal growth books
- Family / parenting
- Marriage / relationships
- Management / leadership
- Career / life's work
- Misc / noise bucket

**Why dims=1536 not 3072 for fine:**
- Within a coarse cluster, documents share broad semantic area; you need mid-range dims to find sub-topic structure
- 3072 adds fine-grained detail that may over-separate (same topic, different vocabulary)
- 1536 = half the Matryoshka space; good balance for within-cluster sub-topic separation

**Why k-means not HDBSCAN for fine:**
- After mean-pooling, clusters are 250–580 documents — manageable, evenly distributed
- Every document should belong to a recurring theme (or a catch-all misc cluster)
- HDBSCAN with 58–66% noise would make the fine layer useless for agent routing

---

## Config recommendation

```toml
[[clustering.passes]]
id = "life_areas"
description = "Coarse life area clusters — one vector per document, k-means"
type = "kmeans"
n_clusters = 10
dims = 96
scope = "all"
input_level = "document"
store_centroid = true

[[clustering.passes]]
id = "life_areas_fine"
description = "Recurring ideas within each life area — document-level k-means"
type = "kmeans"
n_clusters = 8
dims = 1536
scope = "within"
parent_pass = "life_areas"
input_level = "document"
store_centroid = true
```

The cross-domain pass (finding ideas that bridge life areas) is also a document-level mean-pool + k-means job, not HDBSCAN. Open question: whether k=30-50 at dims=512-1024 would surface cross-area connections cleanly.
