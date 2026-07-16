---
description: Blocks ingested since the last review pass, excluding OpenAugi/ derived artifacts.
query:
  after_ingested: "$review-mark"
  exclude_path_prefix: "OpenAugi/"
---

The review-pass queue. `$review-mark` resolves to the high-water mark set
by mark_review_complete; before the first pass it resolves to the epoch,
so the first run is a full backfill. Filtering on ingest time (not
block_time) is deliberate — it catches same-day date-only blocks and
re-ingested edits that `after:` would silently miss.
