"""Golden corpus for the query layer — deterministic DB + case matrix.

The read-tool refactor (docs/plans/query-layer.md step 1) needs a contract:
the MCP wire format must stay BYTE-stable while the query semantics move
from mcp/server.py into query/. This module is that contract's fixture:

- `build_store(db_path)` inserts a fully deterministic dataset (fixed ids,
  block times, ingested_at stamps, embeddings, recaps, review state) that
  exercises every rule the engine owns: mode dispatch, tag/time/kind/source
  filters, after_ingested normalization, has_task + bronze exclusion,
  exclude_path_prefix, reference-document grouping, pagination edges,
  membership (contained/routed/both), recap staleness, salience gating.
- `CASES` maps case name → (tool name, kwargs) across the whole read surface.
- `run_all()` executes every case against the MCP tool functions and returns
  {case: raw_json_string}.

`scripts/gen_query_golden.py` dumps run_all() to
tests/fixtures/golden/query_golden.json; test_query_golden.py asserts
byte-equality against that file. Regenerate ONLY when the wire format is
meant to change, and say so in the commit.

Embeddings use `FakeEmbedder` — deterministic hash-derived vectors, no
network — so semantic mode and get_context are pinned too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from openaugi.model.block import Block
from openaugi.model.link import Link
from openaugi.store.sqlite import SQLiteStore

DIM = 8

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden" / "query_golden.json"


class FakeEmbedder:
    """Deterministic embedding model — sha256-derived unit vectors."""

    def embed_query(self, text: str) -> list[float]:
        return _vec(text)


def _vec(seed: str) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    raw = np.frombuffer(digest[: DIM * 4], dtype=np.uint32).astype(np.float64)
    v = (raw / np.iinfo(np.uint32).max) - 0.5
    v = v / np.linalg.norm(v)
    return [float(x) for x in v]


def _blob(seed: str) -> bytes:
    return np.array(_vec(seed), dtype=np.float32).tobytes()


def _data_block(
    id: str,
    content: str,
    *,
    day: str,
    path: str,
    tags: list[str] | None = None,
    augi_tags: list[str] | None = None,
    ingested: str | None = None,
    has_open_task: bool = False,
    block_time: str | None = None,
) -> Block:
    metadata: dict = {"source_path": path}
    if augi_tags:
        metadata["augi_tags"] = augi_tags
    if has_open_task:
        metadata["has_open_task"] = True
    return Block(
        id=id,
        kind="data_block",
        content=content,
        source="vault",
        title=Path(path).stem,
        tags=tags or [],
        block_time=block_time or day,
        metadata=metadata,
        ingested_at=ingested or f"{day}T10:00:00.000Z",
    )


def _doc(id: str, title: str, path: str) -> Block:
    return Block(
        id=id,
        kind="context_block:document",
        title=title,
        source="vault",
        metadata={"source_path": path},
        ingested_at="2026-05-01T00:00:00.000Z",
    )


CONTAINER_ID = "cont-alpha-0001"
CONTAINER2_ID = "cont-beta-0002"


def build_store(db_path: Path | str) -> None:
    """Insert the deterministic golden dataset. Content is synthetic."""
    store = SQLiteStore(db_path)

    docs = [
        _doc("doc-daily-0601", "2026-06-01", "Daily/2026-06-01.md"),
        _doc(CONTAINER_ID, "MOC - Alpha Project", "Containers/MOC - Alpha Project.md"),
        _doc(CONTAINER2_ID, "MOC - Beta Stream", "Containers/MOC - Beta Stream.md"),
        _doc(
            "doc-ref-0001", "Zebra Protocol Article", "_sources/Readwise/Zebra Protocol Article.md"
        ),
    ]
    tag_block = Block(
        id=Block.make_tag_id("idea"), kind="context_block:tag", title="idea", source="vault"
    )

    data = [
        _data_block(
            "b1-quantum-idea",
            "quantum garden notes about the alpha project #idea",
            day="2026-06-01",
            path="Daily/2026-06-01.md",
            tags=["idea"],
        ),
        _data_block(
            "b2-open-task",
            "quantum beta systems checklist\n- [ ] wire the flux capacitor",
            day="2026-06-02",
            path="Daily/2026-06-02.md",
            has_open_task=True,
        ),
        _data_block(
            "b3-bronze-task",
            "quantum bronze scaffolding thought #layer/bronze\n- [ ] never surfaces as task",
            day="2026-06-03",
            path="Daily/2026-06-03.md",
            tags=["layer/bronze"],
            has_open_task=True,
        ),
        _data_block(
            "b4-tagged-task",
            "follow up on the zebra rollout plan",
            day="2026-06-04",
            path="Daily/2026-06-04.md",
            augi_tags=["type/task", "area/work"],
        ),
        _data_block(
            "b5-derived",
            "quantum derived artifact living under OpenAugi",
            day="2026-06-05",
            path="OpenAugi/Views/View - Alpha.md",
        ),
        _data_block(
            "b6-old-ingest",
            "quantum note ingested long ago but dated recently",
            day="2026-06-06",
            path="Daily/2026-06-06.md",
            ingested="2026-01-01T00:00:00.000Z",
        ),
        _data_block(
            "b7-ref-one",
            "zebra protocol highlight one",
            day="2026-06-07",
            path="_sources/Readwise/Zebra Protocol Article.md",
            tags=["source/readwise"],
        ),
        _data_block(
            "b8-ref-two",
            "zebra protocol highlight two",
            day="2026-06-08",
            path="_sources/Readwise/Zebra Protocol Article.md",
            tags=["source/readwise"],
        ),
        _data_block(
            "b9-contained",
            "pasted directly into the alpha container",
            day="2026-06-09",
            path="Containers/MOC - Alpha Project.md",
        ),
        _data_block(
            "b10-both",
            "lives in alpha and was also routed there",
            day="2026-06-10",
            path="Containers/MOC - Alpha Project.md",
        ),
    ]

    store.insert_blocks(docs + [tag_block] + data)

    links = [
        Link(from_id="b1-quantum-idea", to_id="doc-daily-0601", kind="contains"),
        Link(from_id="b1-quantum-idea", to_id=tag_block.id, kind="groups"),
        Link(from_id="b7-ref-one", to_id="doc-ref-0001", kind="contains"),
        Link(from_id="b8-ref-two", to_id="doc-ref-0001", kind="contains"),
        Link(from_id="b9-contained", to_id=CONTAINER_ID, kind="contains"),
        Link(from_id="b10-both", to_id=CONTAINER_ID, kind="contains"),
        Link(from_id="b10-both", to_id=CONTAINER_ID, kind="routed_to"),
        Link(from_id="b1-quantum-idea", to_id=CONTAINER_ID, kind="routed_to"),
        Link(from_id="doc-daily-0601", to_id=CONTAINER_ID, kind="links_to"),
    ]
    store.insert_links(links)

    # Embeddings — every data block, deterministic vectors.
    store.ensure_vec_table(DIM)
    store.update_embeddings({b.id: _blob(b.id) for b in data})

    # Recaps: alpha fresh (hash captured after membership settled), beta stale.
    store.upsert_recap(
        CONTAINER_ID,
        "## Alpha recap\n\nSynthetic recap body.",
        "2026-06-10T00:00:00+00:00",
        store.membership_hash(CONTAINER_ID),
    )
    store.upsert_recap(
        CONTAINER2_ID,
        "## Beta recap\n\nStale on purpose.",
        "2026-06-01T00:00:00+00:00",
        "deadbeefdeadbeef",
    )

    store.set_review_state("2026-06-05T00:00:00+00:00", "golden fixture pass")
    store.close()


# ── Case matrix ─────────────────────────────────────────────────────

CASES: dict[str, tuple[str, dict]] = {
    # search — mode dispatch
    "search_error_no_params": ("search", {}),
    "search_title": ("search", {"title": "Alpha"}),
    "search_keyword": ("search", {"keyword": "quantum"}),
    "search_keyword_exclude_prefix": (
        "search",
        {"keyword": "quantum", "exclude_path_prefix": "OpenAugi/"},
    ),
    "search_keyword_after_ingested": (
        "search",
        {"keyword": "quantum", "after_ingested": "2026-05-01T00:00:00Z"},
    ),
    "search_title_has_task": ("search", {"title": "2026-06-02", "has_task": True}),
    "search_semantic": ("search", {"query": "quantum garden", "k": 5}),
    "search_semantic_filters": (
        "search",
        {"query": "quantum garden", "k": 5, "tags": ["idea"], "after": "2026-05-31"},
    ),
    "search_semantic_before": (
        "search",
        {"query": "quantum garden", "k": 5, "before": "2026-06-04"},
    ),
    # search — browse mode
    "browse_all": ("search", {"after": "2026-01-01"}),
    "browse_tags": ("search", {"tags": ["idea"], "after": "2026-01-01"}),
    "browse_window": ("search", {"after": "2026-06-02", "before": "2026-06-05"}),
    "browse_after_ingested": ("search", {"after_ingested": "2026-05-01T00:00:00Z"}),
    "browse_exclude_prefix": (
        "search",
        {"after": "2026-01-01", "exclude_path_prefix": "OpenAugi/"},
    ),
    "browse_has_task": ("search", {"has_task": True, "after": "2026-01-01"}),
    "browse_pagination_p1": ("search", {"after": "2026-01-01", "k": 3}),
    "browse_pagination_p2": ("search", {"after": "2026-01-01", "k": 3, "offset": 3}),
    "browse_kind_docs": (
        "search",
        {"kind": "context_block:document", "after_ingested": "2026-01-01"},
    ),
    # block fetch
    "get_block_found": ("get_block", {"block_id": "b1-quantum-idea"}),
    "get_block_missing": ("get_block", {"block_id": "nope-0000"}),
    "get_blocks_mixed": (
        "get_blocks",
        {"block_ids": ["b2-open-task", "missing-0001", "b1-quantum-idea"]},
    ),
    "get_blocks_too_many": ("get_blocks", {"block_ids": [f"x{i}" for i in range(51)]}),
    # graph
    "related_both": ("get_related", {"block_id": "b1-quantum-idea"}),
    "related_out_kind": (
        "get_related",
        {"block_id": "b1-quantum-idea", "direction": "out", "kind": "routed_to"},
    ),
    "related_in": ("get_related", {"block_id": CONTAINER_ID, "direction": "in"}),
    "traverse_two_hops": ("traverse", {"start_id": "b1-quantum-idea", "max_hops": 2}),
    "traverse_kind_limited": (
        "traverse",
        {"start_id": "b1-quantum-idea", "max_hops": 2, "link_kinds": ["routed_to"], "limit": 5},
    ),
    # recent
    "recent_default": ("recent", {"k": 5}),
    "recent_tagged": ("recent", {"k": 5, "tags": ["idea"]}),
    # context
    "context_plain": ("get_context", {"query": "quantum garden", "k": 4}),
    "context_no_expand": ("get_context", {"query": "quantum garden", "k": 4, "expand": False}),
    "context_purpose": (
        "get_context",
        {"query": "quantum garden", "k": 4, "purpose": "resurface"},
    ),
    # containers / views
    "members_alpha": ("get_members", {"container_title": "MOC - Alpha Project"}),
    "members_paged": (
        "get_members",
        {"container_title": "MOC - Alpha Project", "limit": 1, "offset": 1},
    ),
    "members_missing": ("get_members", {"container_title": "MOC - Nope"}),
    "view_alpha_fresh": ("get_view", {"container_title": "MOC - Alpha Project"}),
    "view_beta_stale": ("get_view", {"container_title": "MOC - Beta Stream"}),
    "view_missing": ("get_view", {"container_title": "MOC - Nope"}),
    "list_views": ("list_views", {}),
    "review_state": ("get_review_state", {}),
}


def run_all(db_path: Path | str) -> dict[str, str]:
    """Execute every case against the MCP tool functions; return raw outputs.

    Pins config to DEFAULT_CONFIG (no host config.toml leakage) and installs
    the FakeEmbedder so semantic cases are deterministic and offline.
    """
    import os

    import openaugi.mcp.server as srv
    from openaugi.config import DEFAULT_CONFIG

    old_db = os.environ.get("OPENAUGI_DB")
    old_load_config = srv.load_config
    os.environ["OPENAUGI_DB"] = str(db_path)
    srv._store = None
    srv._embedding_model = FakeEmbedder()
    srv.load_config = lambda: DEFAULT_CONFIG  # type: ignore[assignment]
    try:
        out: dict[str, str] = {}
        for name, (tool, kwargs) in CASES.items():
            fn = getattr(srv, tool)
            out[name] = fn(**kwargs)
        return out
    finally:
        srv.load_config = old_load_config
        srv._store = None
        srv._embedding_model = None
        if old_db is None:
            os.environ.pop("OPENAUGI_DB", None)
        else:
            os.environ["OPENAUGI_DB"] = old_db
