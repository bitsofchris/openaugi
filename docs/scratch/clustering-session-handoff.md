---
name: clustering-session-handoff
description: Session context dump — where clustering work stands, what was tried, what's next. Written April 2026.
---

# Clustering Session Handoff

## Where I am

Building a hierarchical clustering system for the vault (4,226 docs / 20,015 blocks).
The coarse pass is working well. Fine clustering is unresolved — blocked on an
embedding strategy question that's now being tested.

---

## What's working

### Coarse clustering: DONE ✓

Config (`~/.openaugi/config.toml`):
```toml
[[clustering.passes]]
id = "life_areas"
type = "kmeans"
n_clusters = 10
dims = 96
scope = "all"
input_level = "document"
store_centroid = true
```

10 named life-area clusters, even distribution (243–624 docs), all nameable:

| # | Name | Docs |
|---|---|---|
| 0 | Content creation / public second brain | 509 |
| 1 | Personal life / family / social | 253 |
| 2 | Active work log (trading, startup decisions) | 243 |
| 3 | CS/infra fundamentals | 267 |
| 4 | Weekly/monthly check-in journals | 519 |
| 5 | OpenAugi dev + productivity system | 565 |
| 6 | ML/deep learning learning notes | 255 |
| 7 | Writing craft / life philosophy | 580 |
| 8 | AI + augmented intelligence | 410 |
| 9 | Career direction / self-reflection | 624 |

Key insight: **document-level mean-pooling** was the fix. HDBSCAN and block-level
k-means both failed because long docs (podcast transcripts, book notes with 50-150
sections) dominated density. One mean-pooled vector per document solved it.
dims=96 is the sweet spot — coarser loses life-area separation, finer splits the
journal cluster by writing style instead of topic.

---

## What's unresolved

### Fine clustering: BLOCKED on embedding experiment

Fine pass config in `config.toml` is **stale/wrong** — still has HDBSCAN params,
needs updating once the embedding experiment resolves.

The problem: when we ran `cluster-explore-within --cluster 7`, HDBSCAN produced
60–66% noise and every cluster was just one long document repeated:
- "The Out of Sync Child" × 32 blocks
- "Marriage Patterns Analysis" × 43 blocks

Root cause identified: **title-prepending**. Every block is currently embedded as
`"{title}\n\n{content}"`. At fine dims (1536–3072), the title signal dominates,
so all blocks from the same document cluster together regardless of their content.

### The experiment queued up

1. **Run `openaugi embed-content-only`** — embeds all 20k blocks as content-only
   (no title prefix) into a new `content_only_embedding` column. ~$0.08, a few minutes.

2. **Re-run fine cluster explore on cluster 7 with content-only embeddings:**
   ```bash
   openaugi cluster-explore-within --cluster 7 \
     --embedding-col content_only_embedding \
     --dims "1536,3072" \
     --hdbscan-sizes "15,25" \
     --k "5,8,10"
   ```

3. **If HDBSCAN works**: clusters should be cross-document recurring ideas
   (e.g. "writing craft" blocks from multiple different notes grouped together),
   not within-document sections. That's the signal we want.

4. **If HDBSCAN still blows up**: try document-level mean-pooling with
   content-only embeddings at k-means, dims=1536, k=8 for fine pass.

---

## What's been built

### New CLI commands

```bash
openaugi embed-content-only          # embed 20k blocks without title prefix
openaugi cluster-explore             # grid explore coarse clustering (dims × k)
openaugi cluster-explore-within      # explore fine clustering within one cluster
  --cluster 7                        # required: which coarse cluster
  --embedding-col content_only_embedding  # compare title vs content-only
  --dims 1536,3072
  --hdbscan-sizes 15,25
  --k 5,8,10
```

### New/updated files

| File | What changed |
|---|---|
| `src/openaugi/pipeline/cluster.py` | faiss k-means (replaced sklearn), `explore_kmeans_grid`, `explore_fine_cluster`, `embedding_col` param throughout, `load_embeddings` accepts col param |
| `src/openaugi/pipeline/embed.py` | `run_embed_content_only()`, refactored to `_run_embed_batch()` shared loop |
| `src/openaugi/store/sqlite.py` | `content_only_embedding` column via auto-migration, `get_blocks_needing_content_only_embeddings`, `update_content_only_embeddings` |
| `src/openaugi/cli/main.py` | `embed-content-only`, `cluster-explore`, `cluster-explore-within` commands |
| `docs/plans/clustering-findings.md` | Full write-up: what failed, why, what worked |
| `docs/plans/embedding-strategy.md` | Title-prepend vs content-only analysis, open questions, someday experiments |
| `docs/clustering.md` | Block granularity section, backfill SQL |
| `src/openaugi/adapters/vault.py` | `granularity` field on data_blocks ("document" vs "section") |

### Schema changes

`content_only_embedding BLOB` column added to `blocks` table via auto-migration
in `SQLiteStore._apply_migrations()`. Safe to re-run, uses `PRAGMA table_info`.

---

## Key decisions made (and why)

**Document-level mean-pooling for coarse clustering**
Not a workaround — this is semantically correct. "Which life area does this
document belong to?" is a document-level question. Block-level gives unfair weight
to long documents.

**faiss k-means instead of sklearn MiniBatchKMeans**
Faster, same results. Already installed as a transitive dep.

**Two embedding columns, not one**
`embedding` (title-prepended) powers retrieval and shouldn't change — it's
already working and re-embedding 20k blocks to change search behavior is risky.
`content_only_embedding` is the experiment column. Use each where it makes sense.

**HDBSCAN still in the plan for fine clustering**
Not giving up on it — the hypothesis is title-prepending was causing the 65% noise.
Content-only embeddings may fix it. If HDBSCAN works at block level with content-only
embeddings, it's actually better than k-means for fine clustering because noise is
signal (not every block needs to be a recurring theme).

---

## Config needs updating after experiment

Once the fine clustering approach is decided, update `~/.openaugi/config.toml`:

**If document-level k-means wins:**
```toml
[[clustering.passes]]
id = "life_areas_fine"
type = "kmeans"
n_clusters = 8
dims = 1536
scope = "within"
parent_pass = "life_areas"
input_level = "document"
store_centroid = true
```

**If block-level HDBSCAN with content-only wins:**
```toml
[[clustering.passes]]
id = "life_areas_fine"
type = "hdbscan"
min_cluster_size = 15
dims = 1536
scope = "within"
parent_pass = "life_areas"
input_level = "block"
embedding_col = "content_only_embedding"   # not yet wired into run_cluster_dag
store_centroid = true
```

Note: `embedding_col` is not yet wired into `run_cluster_dag` / `_compute_pass` —
only the exploration commands support it. Would need to add that before running
a committed fine pass.

---

## Immediate next step

```bash
.venv/bin/openaugi embed-content-only
```

Then re-run the fine explore on cluster 7 with `--embedding-col content_only_embedding`
and compare cluster quality to the title-prepended version.
