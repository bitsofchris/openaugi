# OpenAugi Clustering Strategy

## Embedding Setup
- OpenAI `text-embedding-3-large`, store full-dim per chunk
- Prepend parent note title to each chunk before embedding
- Truncate to dim-64 at query time for coarse pass

---

## Pass 1: Coarse → Fine (Structure & Temporal Analysis)

**Purpose:** Understand life areas, find recurring ideas within them, track their evolution over time.

**Step A — Coarse clustering:**
HDBSCAN on dim-64 truncated embeddings → ~8-15 life area clusters (career, trading, personal growth, OpenAugi, family, etc.)

**Step B — Fine clustering within each coarse cluster:**
HDBSCAN on full-dim embeddings, scoped to each coarse cluster → specific recurring ideas within each area

**Step C — Temporal analysis:**
For each fine sub-cluster, plot activity over time. See what persists, what flares and dies, what oscillates. LLM agent narrates evolution of the most interesting ones.

---

## Pass 2: Unconstrained (Surprising Connections)

**Purpose:** Find ideas that rhyme across life areas, and bridge notes that connect different domains.

**Step A — Full-dim HDBSCAN, no coarse scoping:**
Chunks from different life areas can land in the same cluster → these are the cross-domain connections you wouldn't explicitly link.

**Step B — Bridge detection:**
HDBSCAN noise points that sit between multiple clusters = ideas that connect different areas. Surface points close to 2+ cluster centroids but not in any cluster.

**Step C — Cross-cluster similarity:**
Compare fine sub-cluster centroids across coarse boundaries. "This idea in personal growth is surprisingly similar to this idea in trading."

---

## Two Lenses, One Dataset

| | Coarse → Fine | Unconstrained |
|---|---|---|
| Question | What are my life areas and how do ideas evolve within them? | What connects across areas that I haven't noticed? |
| Hierarchy | Matryoshka truncation creates it | HDBSCAN condensed tree |
| Temporal | Primary use case | Secondary |
| Surprises | No (boundaries enforced) | Yes (boundaries removed) |
