---
name: future-work
description: Features deferred from M0 — task system, additional MCP tools, lenses, sync engine, and other post-launch work
---

# Future Work — Deferred from M0

*Updated: 2026-03-24*

Features and systems that exist in augi-engine-v1 or were designed in earlier plans but are not part of the M0 port. Tracked here so nothing is lost.

---

## Task System (from v1, private workflow)

v1 has a full task scanning and thread management system. This is a personal workflow tool, not core engine functionality. Stays in private repo for now.

| v1 Component | What It Does |
|---|---|
| `scan_tasks` | Find `#to_*` tags and `- [ ]` checkboxes in recent entries |
| `write_thread` | Create structured task thread notes in vault |
| `update_task` | Append results, update status in thread frontmatter |
| `write_document` | Create standalone docs in vault |
| `thread_creator.py` | Task scanning logic, dedup via context_hash, template rendering |
| `thread_updater.py` | Append to `## Results` section, update YAML status |
| `doc_writer.py` | Create docs, link from task threads |

**MCP tools not ported:** `scan_tasks`, `write_thread`, `update_task`, `write_document`

**Future:** Could become a Lens or a separate `openaugi-tasks` package.

---

## MCP Tools Deferred

| v1 Tool | Why Deferred | When |
|---|---|---|
| `search_hubs` | Folded into `search` + `traverse` in v5 tool design | M0 (covered by new tools) |
| `get_summary` | Hub summaries deferred to M2 | M2 |
| `reload_index` | Automatic in v5 (detect DB changes) | M0 (implicit) |

---

## Lenses (from architecture-v5)

Scheduled enrichment passes that read the store, apply reasoning through a framework, and write new blocks back. The opinionated layer.

| Lens | Schedule | What It Produces |
|------|----------|-----------------|
| WeeklyReview | Weekly | What shipped, stuck, surfaced, should drop |
| OrphanDetect | Nightly | Blocks with zero links |
| DeadStreamDetect | Weekly | Tags with no new blocks in 4+ weeks |
| RecurrenceAlert | Weekly | Tags reappearing after inactivity |
| HubSummaryRefresh | Nightly | Updated summaries for changed hubs |
| TornadoDetect | Weekly | WIP overload detection |
| BurkemanReflect | Weekly | 4000 Weeks framework analysis |
| SelfRules | On demand | Check plans against personal operating rules |

**Protocol defined in v5.** Implementation starts M3.

---

## Layer 2 Processing (M2)

| Feature | What It Does |
|---|---|
| Entity extraction | LLM extracts entities from entries → extraction blocks + links |
| Hub summaries | LLM summarizes top-N hubs → summary blocks |
| Clustering | HDBSCAN on embeddings → cluster blocks + member_of links |
| Block summaries | 16-word self-description per entry block |
| Ontology | YAML config constraining entity types for extraction |

---

## Sync Engine (M3)

| Feature | What It Does |
|---|---|
| File watcher | Watch vault for changes, auto-ingest |
| Daemon mode | `openaugi daemon` runs pipeline on schedule |
| API source scheduling | Periodic sync from Readwise, etc. |
| Config YAML | Source definitions, schedules, credentials |

---

## Temporal Intelligence (M3)

| Feature | What It Does |
|---|---|
| Hub velocity | Rising vs declining topics (entry count in windows) |
| Recurrence detection | Topics that go quiet then return |
| Dead stream detection | Threads you dropped |
| Centroid drift | Embedding distance between hub snapshots over time |
| Hub snapshots table | Weekly centroid embeddings for drift analysis |

---

## Additional Adapters (M1+)

| Adapter | Source | Priority |
|---|---|---|
| ChatGPT | JSON export → document blocks → turn-level entries | M1 |
| Claude | JSON export → same pattern | M1 |
| Readwise | API → highlight-level entries | M1-M2 |
| Google Drive | Download → markdown conversion | M2+ |
| Notion | Export → markdown | Community |
| Pocket/Instapaper | API → article entries | Community |

---

## Other Deferred Items

- **Adapter protocol / registry / entry points** — extract from concrete implementations when second source arrives (M1)
- **Config wizard / onboarding CLI** — interactive first-time setup
- **Observation lifecycle** — living blocks that track recurring patterns, supersede older versions
- **Context blocks** — curated collections assembled for a purpose
- **Deployment modes** — Docker, hosted/managed, REST API
- **Obsidian plugin** — direct integration instead of MCP
- **Mobile capture** — quick-capture via CLI or API
