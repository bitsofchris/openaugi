---
name: cluster-autoresearch-prompt
description: Agent prompt for iterative HDBSCAN param tuning — run, sample clusters, evaluate quality, adjust, repeat until coarse/fine/cross-domain passes all look good.
---

# Clustering Autoresearch Loop

## Context

You are working in `/Users/chris/repos/openaugi`. This is the OpenAugi codebase — a personal knowledge engine over an Obsidian vault. 20,015 data_blocks are embedded with text-embedding-3-large (3072 dims, title-prepended). The DB is at `~/.openaugi/openaugi.db`.

We have a new `openaugi cluster` CLI command that runs HDBSCAN clustering passes defined in `~/.openaugi/config.toml`. Each pass produces `context_block:cluster` nodes in the DB linked to their member data_blocks.

Your job: iterate on clustering params until the clusters are qualitatively good. This is a Karpathy-style autoresearch loop — run, eval, adjust, repeat.

The venv is at `.venv/`. All commands: `.venv/bin/openaugi <cmd>` or `.venv/bin/python`.

---

## Starting config

First, add this to `~/.openaugi/config.toml` (append to the file):

```toml
[[clustering.passes]]
id = "life_areas"
description = "Coarse life area clusters"
dims = 64
scope = "all"
min_cluster_size = 50
min_samples = 10
store_centroid = true

[[clustering.passes]]
id = "life_areas_fine"
description = "Fine topic clusters within each life area"
dims = 3072
scope = "within"
parent_pass = "life_areas"
min_cluster_size = 20
min_samples = 5
store_centroid = true

[[clustering.passes]]
id = "cross_domain"
description = "Unconstrained cross-area connections"
dims = 3072
scope = "all"
min_cluster_size = 30
min_samples = 10
store_centroid = true
bridge_detection = true
```

---

## The loop

Work on one pass at a time. Start with `life_areas`. Only move to the next pass once the current one is good.

### Step 1 — Run clustering

```bash
.venv/bin/openaugi cluster --pass life_areas
```

This writes cluster blocks to the DB (idempotent — safe to re-run).

### Step 2 — Sample each cluster

Run this sqlite3 query to see what's in each cluster. Read the output carefully.

```bash
sqlite3 ~/.openaugi/openaugi.db "
SELECT
  cb.title AS cluster,
  json_extract(cb.metadata, '$.member_count') AS members,
  json_extract(cb.metadata, '$.temporal.first_block') AS first,
  json_extract(cb.metadata, '$.temporal.last_block') AS last,
  db.title AS note_title,
  substr(db.content, 1, 180) AS snippet
FROM blocks cb
JOIN links l ON l.from_id = cb.id AND l.kind = 'groups'
JOIN blocks db ON db.id = l.to_id
WHERE cb.kind = 'context_block:cluster'
  AND json_extract(cb.metadata, '$.pass_id') = 'life_areas'
ORDER BY cb.id, RANDOM()
LIMIT 300;
"
```

Also get the cluster-level summary:

```bash
sqlite3 ~/.openaugi/openaugi.db "
SELECT
  cb.title,
  json_extract(cb.metadata, '$.member_count') AS members,
  json_extract(cb.metadata, '$.noise_count') AS noise,
  json_extract(cb.metadata, '$.temporal.first_block') AS first,
  json_extract(cb.metadata, '$.temporal.last_block') AS last
FROM blocks cb
WHERE cb.kind = 'context_block:cluster'
  AND json_extract(cb.metadata, '$.pass_id') = 'life_areas'
ORDER BY CAST(json_extract(cb.metadata, '$.member_count') AS INTEGER) DESC;
"
```

### Step 3 — Evaluate

Apply this rubric to the `life_areas` pass:

**Hard requirements (must fix if violated):**
- Cluster count: 6–18. Fewer = over-merged (decrease min_cluster_size). More = over-fragmented (increase min_cluster_size).
- Noise: < 20% of 20,015 blocks. High noise = decrease min_samples.
- No single cluster > 40% of blocks. A mega-cluster means the coarse level isn't separating.
- No cluster < 30 blocks (statistical accidents, not real areas).

**Qualitative signal (good clusters):**
- You can name each cluster in 3–5 words from reading 8–10 sample blocks. If you can't, it's too mixed.
- Clusters feel like distinct life areas — OpenAugi/building, trading/finance, personal reflection/self, family, health, research/learning, content creation, career/work are the expected rough shapes.
- A block from "trading journal" and a block about "NVDA options" should be in the same cluster, not split.
- Blocks about "OpenAugi architecture" and "Claude API" should be together, not scattered.

**For `life_areas_fine`:**
- Each parent cluster should produce 3–12 sub-clusters. Fewer means fine clustering isn't finding sub-topics. More means noise.
- Sub-clusters should be nameable — "within OpenAugi, I can see a cluster about heartbeat/agent design, a cluster about data model, a cluster about marketing."
- Noise within each parent is OK up to 30% — fine-grained clustering on dense text is noisier.

**For `cross_domain`:**
- Expect 15–60 clusters, more than life_areas.
- The interesting signal: find clusters where `note_title` values come from different life areas. Those are the cross-domain connections.
- Bridge blocks (context_block:bridge kind) are the most interesting — blocks that HDBSCAN couldn't assign anywhere. Sample a few:

```bash
sqlite3 ~/.openaugi/openaugi.db "
SELECT b.title, substr(db.content, 1, 200) AS content
FROM blocks b
JOIN blocks db ON db.id = json_extract(b.metadata, '$.source_block_id')
WHERE b.kind = 'context_block:bridge'
LIMIT 20;
"
```

### Step 4 — Adjust and re-run

Edit `~/.openaugi/config.toml` to adjust params for the failing pass, then re-run. The command is idempotent — it deletes old cluster blocks for that pass before writing new ones.

**Tuning levers:**
- `min_cluster_size`: primary control. Bigger → fewer, larger clusters. Smaller → more, smaller clusters. Adjust in steps of ~10.
- `min_samples`: controls how conservative assignment is. Bigger → more noise (stricter). Smaller → less noise (more permissive). Start at 5 if noise is too high.
- `dims` for life_areas: 64 is the starting point. Try 32 (coarser) or 128 (finer) if the life-area separation isn't working.

You can run one pass at a time: `--pass life_areas`. No need to re-run all passes while tuning a single one.

### Step 5 — Move to next pass when current is good

Once `life_areas` looks good:
- Run `life_areas_fine`
- Evaluate sub-clusters within each life area
- Then run `cross_domain`
- Evaluate cross-domain clusters and bridges

---

## Stopping criteria

You're done when all three passes satisfy:
1. `life_areas`: 6–18 clusters, < 20% noise, each cluster nameable, no mega-cluster
2. `life_areas_fine`: 3–12 sub-clusters per parent, sub-clusters nameable
3. `cross_domain`: 15–60 clusters, at least a few visibly cross-domain, bridge blocks non-trivial

When done, write a short summary of the final params chosen for each pass and what the resulting clusters look like at a high level (how many, what themes emerged, anything surprising). Save this summary to `docs/plans/cluster-results.md`. It will be read back into future sessions to understand the knowledge landscape.

---

## What NOT to do

- Do not modify any source code in `src/openaugi/` — params only, via config.toml
- Do not run `openaugi re-embed` — embeddings are already done
- Do not run `openaugi ingest` or `openaugi up`
- Do not overthink it. Run, look, adjust. 5–8 iterations is normal. 15+ means something is wrong structurally.

---

## If something breaks

Logs: `~/.openaugi/logs/openaugi.log`

Check the cluster command output — it prints stats for each pass (cluster count, noise %, size distribution) before writing to DB. If HDBSCAN hangs (unlikely at 20k blocks with dims=64 but possible at full dims), kill and reduce min_cluster_size.

The DB path is `~/.openaugi/openaugi.db`. If you need to inspect raw blocks:

```bash
sqlite3 ~/.openaugi/openaugi.db "
SELECT id, kind, title, substr(content, 1, 100)
FROM blocks
WHERE kind = 'data_block'
LIMIT 20;
"
```
