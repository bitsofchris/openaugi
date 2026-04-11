---
name: embedding-strategy
description: What we know about title-prepended vs content-only embeddings, what the evidence shows, and open questions to investigate. Includes experiments to run and hypotheses about where each strategy helps or hurts.
---

# Embedding Strategy: Title-Prepended vs Content-Only

## Current state

Every `data_block` is embedded as `"{title}\n\n{content}"` — the parent note title
is prepended to the block content before calling the embedding API. This is the
`_build_embed_text()` function in `src/openaugi/pipeline/embed.py`.

This is stored in the `embedding` BLOB column and used for:
- Semantic search (sqlite-vec ANN)
- All clustering passes

---

## What we found

### Title-prepending hurts clustering

During fine clustering experiments (April 2026), we found that HDBSCAN at block
level within a coarse cluster produced 60–66% noise, and every identified cluster
was just all sections of the same long document repeated:

```
Cluster 0: The Out of Sync Child × 32
Cluster 1: Marriage Patterns Analysis × 43
Cluster 7: Writing Books with James Clear and Mark Manson × 56
```

The root cause: every block from "A conversation on focus and finding your life's work"
starts with that title string in its embedding. At fine dims (1536–3072), the
title signal dominates over content signal within a single document. HDBSCAN
finds title-cohesion, not idea-cohesion.

**Hypothesis:** content-only embeddings at block level would let HDBSCAN and
k-means find recurring ideas across documents and time, rather than finding
"same title" clusters. A block about "compounding returns" from a 2024 book note
would cluster with a 2025 journal entry also about compounding — even if their
parent titles are completely different.

**Workaround used:** switched to document-level mean-pooling before clustering.
This sidesteps the problem by averaging out the title signal across all sections,
but loses within-document granularity and the ability to find specific idea
recurrence at block level.

---

## What we don't know (open questions)

### 1. Does title-prepending help or hurt retrieval?

**The argument for title-prepending:**
- Short/ambiguous chunks need context. "got stopped out" means nothing without
  knowing it's from a trading journal. The title anchors the embedding in the
  right semantic neighborhood.
- OpenAI recommends this pattern for RAG — include context with each chunk.
- For retrieval, a query like "trading risk management" should match chunks
  from trading notes, not ambiguous short sentences from anywhere.

**The argument against:**
- For atomic notes (1 block = 1 document), the title IS often the content — prepending
  it doubles the signal and may skew the vector toward the title's semantic space.
- For long podcast transcripts, a 3-hour conversation covers many topics — the
  single title doesn't accurately represent any individual chunk.
- Title-prepending reduces the Matryoshka coarse-level diversity signal: two blocks
  from very different content but the same note title cluster together.

**Experiment to run:**
- Evaluate retrieval precision@5 with and without title-prepending
- Ground truth: manually label 50 queries with expected source documents
- Compare which embedding strategy returns the right documents

### 2. Does it matter at which dims?

Title is typically 5–15 tokens. At dims=96 (coarse clustering), the title might
dominate the entire vector. At dims=3072 (fine clustering), its proportion of
total signal is smaller. The relationship between title weight and dims truncation
in Matryoshka space is not well understood.

**Experiment to run:**
- Take 20 long documents (50+ blocks each)
- Embed all blocks with and without title
- PCA/UMAP to 2D, compare cluster tightness within-document
- Hypothesis: title-prepended blocks are much tighter within a document at all dims;
  content-only blocks spread out into semantic sub-topics

### 3. Is there a middle path?

Options between "full title" and "nothing":

1. **Lightweight context:** just the document date or top-level heading, not the full title
2. **Section heading only:** for QQQ/heading-split blocks, use the H2/H3 they're under
   rather than the parent note title — closer to the actual content's context
3. **Hybrid columns:** keep `embedding` (title-prepended, good for retrieval) and
   `content_only_embedding` (content-only, good for clustering) — use each where appropriate
4. **Prefix only for short blocks:** prepend title only when `len(content) < 200 chars`,
   otherwise trust the content to carry its own context

### 4. Does granularity matter?

- `granularity = "document"` (single-block atomic note): prepending the title is
  essentially a no-op — the title is often a summary of the content anyway
- `granularity = "section"` (heading/QQQ split): the section may be much more
  specific than the parent title — title-prepending may dilute the section's signal
- `granularity = "document_chunk"` (future: PDF/web): long mechanically-chunked docs
  where title is irrelevant to any individual chunk — title-prepending is clearly wrong

---

## Current approach (what's implemented)

**Two embedding columns:**

| Column | Text embedded | Use for |
|---|---|---|
| `embedding` | `"{title}\n\n{content}"` | Semantic retrieval (ANN search) |
| `content_only_embedding` | `"{content}"` | Clustering experiments |

Clustering passes can select which column via `input_level` or a future
`embedding_col` config option. The `explore_fine_cluster` and
`explore_kmeans_grid` tools accept an `--embedding-col` flag.

**Why keep both:**
- `embedding` (title-prepended) is already computed for all 20k blocks and
  powers retrieval — don't change what's working
- `content_only_embedding` is cheap to compute ($0.08) and lets us run controlled
  experiments on clustering without touching retrieval

---

## Someday / maybe experiments

These are worth doing but not blocking current work:

1. **Retrieval precision comparison:** does title-prepending improve or hurt search?
   Build a small labeled eval set (50 queries × expected docs) and compare.

2. **HDBSCAN on content-only block embeddings:** after computing `content_only_embedding`,
   re-run the fine cluster explore on cluster 7. Does HDBSCAN now find cross-document
   recurring ideas instead of within-document density?

3. **Section heading as context:** for heading-split blocks, embed as
   `"{section_heading}\n\n{content}"` instead of the parent note title.
   The section heading is more specific and accurate context.

4. **Matryoshka dims vs title weight analysis:** PCA on title-prepended vs
   content-only at dims 64, 96, 128, 256, 512, 1536, 3072. Where does the
   title signal fade relative to content?

5. **Fine clustering at block level with content-only:** the original goal —
   can we cluster blocks (not documents) and find recurring ideas if we remove
   the title anchor?
