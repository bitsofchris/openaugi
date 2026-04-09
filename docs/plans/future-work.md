---
name: future-work
description: Features deferred from M0 — task system, additional MCP tools, lenses, sync engine, and other post-launch work
---

# Future Work — Deferred from M0

*Updated: 2026-04-08*

Features and systems that exist in augi-engine-v1 or were designed in earlier plans but are not part of the M0 port. Tracked here so nothing is lost. For the active roadmap (Phases 2–7), see [m2-feature-roadmap.md](m2-feature-roadmap.md).

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
| `get_summary` | Free-SQL hub summaries shipped in Phase 2 via `get_index` + context blocks. LLM hub narratives pending Phase 4. | Phase 4 |
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

## Layer 2 Processing (Phase 4)

Free-SQL hub summaries shipped as Phase 2 context blocks. LLM-powered pieces remain:

| Feature | What It Does |
|---|---|
| Entity extraction | LLM extracts entities from entries → extraction blocks + links |
| Hub narratives | LLM summarizes top-N hubs → `context_type=hub_narrative` blocks (themes, tensions, decisions) |
| Clustering | HDBSCAN on embeddings → cluster blocks + member_of links |
| Block summaries | 16-word self-description per entry block |
| Ontology | YAML config constraining entity types for extraction |

---

## Sync Engine

File watcher + daemon mode shipped in M1.5 (`openaugi watch`, `openaugi up`, launchd service). Remaining:

| Feature | What It Does |
|---|---|
| API source scheduling | Periodic sync from Readwise, etc. (Phase 3 adapters) |
| Config YAML | Per-source schedules and credentials beyond the current `config.toml` (Phase 3) |

---

## Temporal Intelligence (Phase 5)

| Feature | What It Does |
|---|---|
| Hub velocity | Rising vs declining topics (entry count in windows) |
| Recurrence detection | Topics that go quiet then return |
| Dead stream detection | Threads you dropped |
| Centroid drift | Embedding distance between hub snapshots over time |
| Hub snapshots table | Weekly centroid embeddings for drift analysis |

---

## Additional Adapters

ChatGPT, Readwise, Research output, and a LlamaIndex bridge are planned for Phase 3 — see [phase3-adapters.md](phase3-adapters.md). Beyond that:

| Adapter | Source | Priority |
|---|---|---|
| Claude export | JSON export → same pattern as ChatGPT | Post-Phase 3 |
| Google Drive | Download → markdown conversion | Post-Phase 3 |
| Notion | Export → markdown | Community |
| Pocket/Instapaper | API → article entries | Community |

---

## Other Deferred Items

- **Adapter protocol / registry / entry points** — to be defined in Phase 3 alongside the first non-vault adapters
- **Config wizard / onboarding CLI** — `openaugi init` shipped in M1.5; full multi-source wizard is Phase 7
- **Observation lifecycle** — living blocks that track recurring patterns, supersede older versions
- **Deployment modes** — Docker, hosted/managed, REST API
- **Obsidian plugin** — direct integration instead of MCP
- **Mobile capture** — quick-capture via CLI or API
