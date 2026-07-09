---
name: knowledge-timeline
description: How the concept cluster timeline visualization was built — data pipeline, HTML explorer, and how to regenerate it.
---

# Knowledge Timeline

A standalone HTML visualization showing recurring concept clusters over time. Designed to answer: *which ideas do I keep returning to, which have I abandoned, and what am I obsessing over right now?*

Output: `knowledge-explorer/cluster_timeline.html` — self-contained, no server needed.

---

## What it shows

Each row is a **concept cluster** — a group of notes the embedding model determined share the same recurring idea, scoped within a life area.

| Column | Meaning |
|---|---|
| Sparkline | Activity density over time, binned weekly. Blue bars = how many blocks fell in that period. |
| Count | Total blocks in the cluster |
| Returns | Times you re-engaged after a 60+ day gap — the "revisited idea" signal |
| Recent/mo | Blocks per month in the last 90 days |
| Last seen | Green = active (<14 days), red = dormant (>6 months) |

Click any row → expands to show every note in the cluster with date, document title (dim), and content snippet (prominent).

**Sort options:** Returns (default), Count, Recent activity, Dormant first, Longest span.

---

## Prerequisites

These must already be done before regenerating:

1. **Embeddings**: all `data_block` rows have `content_only_embedding` populated
2. **Clustering**: `life_areas` (k-means, document-level) and `concepts` (HDBSCAN within each life area, block-level) passes committed to DB

Verify:
```bash
sqlite3 ~/.openaugi/openaugi.db "
SELECT COUNT(*) FROM blocks WHERE content_only_embedding IS NOT NULL AND kind = 'data_block';
"
# should be ~20k

sqlite3 ~/.openaugi/openaugi.db "
SELECT COUNT(*) FROM blocks
WHERE kind = 'context_block:cluster'
  AND json_extract(metadata, '$.pass_id') = 'concepts';
"
# should be ~90
```

---

## How to regenerate

### Step 1 — Export cluster metadata

```bash
sqlite3 -json ~/.openaugi/openaugi.db "
SELECT
  cb.id,
  cb.title,
  json_extract(cb.metadata, '$.pass_id') AS pass_id,
  json_extract(cb.metadata, '$.parent_cluster_label') AS parent_label,
  json_extract(cb.metadata, '$.cluster_label') AS cluster_label,
  json_extract(cb.metadata, '$.member_count') AS member_count,
  json_extract(cb.metadata, '$.temporal.first_block') AS first_block,
  json_extract(cb.metadata, '$.temporal.last_block') AS last_block,
  json_extract(cb.metadata, '$.temporal.block_timestamps') AS timestamps
FROM blocks cb
WHERE cb.kind = 'context_block:cluster'
  AND json_extract(cb.metadata, '$.pass_id') = 'concepts'
ORDER BY CAST(json_extract(cb.metadata, '$.member_count') AS INTEGER) DESC
" > /tmp/concepts_data.json
```

### Step 2 — Export block content

```bash
sqlite3 -json ~/.openaugi/openaugi.db "
SELECT
  l.from_id AS cluster_id,
  db.title,
  db.block_time,
  substr(db.content, 1, 400) AS snippet
FROM links l
JOIN blocks db ON db.id = l.to_id
JOIN blocks cb ON cb.id = l.from_id
WHERE l.kind = 'groups'
  AND cb.kind = 'context_block:cluster'
  AND json_extract(cb.metadata, '$.pass_id') = 'concepts'
ORDER BY l.from_id, db.block_time
" > /tmp/concept_blocks.json
```

### Step 3 — Process and compute metrics

```python
import json
from datetime import date, timedelta

LIFE_AREA_NAMES = {
    0: 'AI / Augmented Intelligence',
    1: 'Daily Life / Personal Journal',
    2: 'Comedy / Content Writing',
    3: 'Engineering / Health / Misc',
    4: 'Creator / Podcast / Trading',
    5: 'Learning / Growth / Reading',
    6: 'OpenAugi / Data Eng Work',
    7: 'Inner Life / Psychology / Jung',
    8: 'Weekly Reflections / Big Picture',
    9: 'ML Research / Trading Tech',
}

with open('/tmp/concepts_data.json') as f:
    raw = json.load(f)

with open('/tmp/concept_blocks.json') as f:
    block_rows = json.load(f)

# Group blocks by cluster
blocks_by_cluster = {}
for row in block_rows:
    cid = row['cluster_id']
    blocks_by_cluster.setdefault(cid, []).append({
        't': row['title'] or '',
        'd': row['block_time'][:10] if row['block_time'] else '',
        's': (row['snippet'] or '').strip(),
    })

# Global date range
all_dates = []
for row in raw:
    ts = json.loads(row['timestamps']) if isinstance(row['timestamps'], str) else row['timestamps']
    all_dates.extend(ts)
all_dates.sort()
global_min, global_max = all_dates[0], all_dates[-1]

clusters = []
for row in raw:
    ts = json.loads(row['timestamps']) if isinstance(row['timestamps'], str) else row['timestamps']
    ts = sorted(ts)
    if len(ts) < 2:
        continue

    returns = sum(
        1 for i in range(1, len(ts))
        if (date.fromisoformat(ts[i]) - date.fromisoformat(ts[i-1])).days > 60
    )

    cutoff = date.fromisoformat(global_max) - timedelta(days=90)
    recent = sum(1 for t in ts if date.fromisoformat(t) >= cutoff)
    lifetime_months = max((date.fromisoformat(ts[-1]) - date.fromisoformat(ts[0])).days / 30, 1)

    parent = int(row['parent_label']) if row['parent_label'] is not None else None

    clusters.append({
        'id': row['id'],
        'title': row['title'],
        'parent_label': parent,
        'area_name': LIFE_AREA_NAMES.get(parent, f'Area {parent}'),
        'member_count': int(row['member_count']),
        'timestamps': ts,
        'first': ts[0],
        'last': ts[-1],
        'returns': returns,
        'recent': recent,
        'velocity_hist': round(len(ts) / lifetime_months, 2),
        'velocity_recent': round(recent / 3, 2),
        'span_days': (date.fromisoformat(ts[-1]) - date.fromisoformat(ts[0])).days,
        'days_since': (date.fromisoformat(global_max) - date.fromisoformat(ts[-1])).days,
        'blocks': blocks_by_cluster.get(row['id'], []),
    })

clusters.sort(key=lambda x: (-x['returns'], -x['member_count']))

out = {'global_min': global_min, 'global_max': global_max, 'clusters': clusters}
with open('/tmp/concepts_processed.json', 'w') as f:
    json.dump(out, f)
```

### Step 4 — Inject into HTML template and open

```python
import json, os

with open('/tmp/concepts_processed.json') as f:
    data = json.dumps(json.load(f))

# HTML template is in knowledge-explorer/cluster_timeline_template.html (if saved)
# or rebuild from scratch — see the source in knowledge-explorer/cluster_timeline.html

template = open('knowledge-explorer/cluster_timeline.html').read()
# If regenerating, replace the embedded data blob between the DATA = ... assignment
# Easier: keep a template with CLUSTER_DATA_PLACEHOLDER and do:
html = template.replace('CLUSTER_DATA_PLACEHOLDER', data)

os.makedirs('knowledge-explorer', exist_ok=True)
with open('knowledge-explorer/cluster_timeline.html', 'w') as f:
    f.write(html)

os.system('open knowledge-explorer/cluster_timeline.html')
```

---

## Key design decisions

**Why `content_only_embedding` not `embedding`?**
The default embedding prepends the document title to each block before embedding. At fine clustering dims, this causes HDBSCAN to group blocks by document identity rather than idea similarity — every block from "Weekly Reflection 2025-03-08" clusters together regardless of content. Content-only embeddings let the idea itself drive clustering.

**Why block-level for concepts, document-level for life areas?**
Life areas are a document-level question ("what is this note about?"). Concepts are a block-level question ("what specific idea appears repeatedly across many notes?"). Mean-pooling at document level for coarse pass avoids long documents (50-150 section podcast transcripts) dominating HDBSCAN density.

**Why individual `block_timestamps` not monthly counts?**
Pre-bucketed counts lock you into one time resolution and throw away the original signal. Raw timestamps let the viz bucket at any granularity and let you see exact engagement patterns — a burst of 8 entries in one week looks very different from 8 entries spread over a month.

**Why 60-day gap for "return"?**
Two months is long enough to represent genuine discontinuity rather than normal note-taking rhythm. A week gap isn't a return — it's just how you write. 30 days was too sensitive for monthly journalers; 60 days captures genuine re-engagement.

---

## What the life area labels mean

These were inferred from sampling doc titles after clustering — update if vault changes substantially:

| # | Label | Character |
|---|---|---|
| 0 | AI / Augmented Intelligence | OpenAugi, augmented engineer, AI tools |
| 1 | Daily Life / Personal Journal | Daily entries, family, personal events |
| 2 | Comedy / Content Writing | Comedy bits, social commentary, short-form |
| 3 | Engineering / Health / Misc | Code snippets, health/fitness, misc reference |
| 4 | Creator / Podcast / Trading | Content strategy, podcast, trading ideas |
| 5 | Learning / Growth / Reading | Books, frameworks, career growth |
| 6 | OpenAugi / Data Eng Work | Work tasks, Datadog, OpenAugi dev |
| 7 | Inner Life / Psychology / Jung | Self-reflection, Jung, relationships |
| 8 | Weekly Reflections / Big Picture | Weekly notes, energy/drain, direction |
| 9 | ML Research / Trading Tech | Transformers, time series, quant |

---

## Next steps

- **Agent summarization** — run `openaugi cluster-summarize` to write 1-line LLM descriptions into each cluster's `content` field, replacing the opaque `concepts_8_26` titles with readable names
- **Cross-domain view** — same viz but for `cross_domain` pass (172 clusters, no life-area scoping) to surface surprising connections
- **Bridge block explorer** — separate view for the 10,133 bridge blocks: ideas that sit between multiple clusters
