---
name: graph-integrity
description: The invariants the stored graph must keep against the vault, the checker that enforces them (`openaugi check-graph`, `check_graph_integrity`), and the testing rules that keep drift visible — the after-ingest check, the link oracle, and incremental-equals-full.
---

# Graph integrity

## When to use

- Changing anything that writes blocks or links: the vault adapter, the
  splitter, the runner, identity rules, a backfill, a routing writer.
- Suspecting the graph answers with less than the vault holds (a note "has no
  backlinks", a container looks emptier than it is).
- Writing an ingest test: you get the integrity check for free; know what it
  asserts.

## The class of bug

Silent drift: the stored graph stops saying what the vault says, and nothing
raises. Known shapes:

- an edge keyed by one identity rule while its endpoint is keyed by another
  (wikilinks resolved by path into notes keyed on `augi_id`);
- an incremental ingest ending in a different state than a fresh full ingest
  (only changed notes are re-parsed, so whatever depends on an unchanged
  note is never revisited);
- the runner's valid-endpoint filter dropping edges;
- leftover records (two documents for one source_path, entries of a deleted
  note).

## How it works

### The invariants — `src/openaugi/pipeline/integrity.py`

`check_graph_integrity(store, vault_root=None)` returns findings; it is
read-only. Store-only checks:

| Check | Invariant |
|---|---|
| `duplicate_document` | one document block per source_path |
| `dangling_edge` | every `contains` / `links_to` / `routed_to` / `groups` edge has both endpoints |
| `orphan_data_block` / `multiple_contains` | every vault data_block has exactly one `contains` |
| `contains_wrong_document` | ... and it points at the document of its own source_path |
| `edge_target_not_document` | `links_to` / `routed_to` target a document |

With a vault, it re-parses the vault with the ingest parser and compares only
notes whose file hash equals the stored one (a note edited a second ago is
not drift):

| Check | Invariant |
|---|---|
| `document_without_file` | every stored document still has its file |
| `document_identity` | every document's id is what the identity rule gives it (`document_id_for`) |
| `links_to_missing` / `links_to_extra` | the stored `links_to` edges of each unchanged note equal a fresh parse of it |
| `unresolved_wikilink` (info) | wikilinks whose target is not a note; no edge is stored for them |

`openaugi check-graph` runs it against a DB (`--no-vault` for store-only,
`--json` for a scheduled job). It opens the DB read-only and exits 1 on any
error-level finding.

### The runtime guard — `pipeline/runner.py`

The runner still drops edges whose endpoint is missing, but it now says so:
a wikilink to a note that is not in the vault is an INFO count; any other
dropped edge is a WARNING with examples, pointing at `check-graph`.

### The testing rules — `tests/`

1. **Every ingest test checks integrity.** `tests/conftest.py` wraps the
   runner's parse step for any test that uses the `store` fixture, and after
   the test runs `assert_graph_integrity(store, last_vault)`
   (`tests/graph_integrity.py`). A test that builds a drifted graph on
   purpose opts out with `@pytest.mark.graph_drift_expected(reason=...)`.
2. **The oracle.** `tests/test_graph_integrity.py::oracle_links` resolves
   wikilinks from the markdown with a deliberately simple parser written from
   how Obsidian behaves, not from the adapter. Each wikilink form is a
   parametrized case against it; unresolved targets are pinned, never
   silently equal.
3. **Incremental equals full.** `test_incremental_equals_full` runs a table
   of lifecycle steps (add, edit, add/remove `augi_id`, rename, move, delete,
   link-before-target, duplicate titles, link forms), each followed by an
   incremental ingest and the integrity check, then compares
   `graph_snapshot` (paths, content hashes, document ids, edges by path)
   against one fresh ingest of the final vault. A new lifecycle step or
   link form is one row.

Known-open drift is `xfail(strict=True)` with its reason: it stays visible,
and fixing it turns the row into an XPASS failure, so the marker must be
removed in the same change. Never weaken a row to make it pass.
