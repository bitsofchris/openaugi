"""The generic records store — a collection store for agent workflow state.

The point of these tests is what they DON'T assert. There is no "proposal",
no "routing", no "pass" here, because openaugi does not know those words. A
collection is a name the caller chose; the shape of its data lives in the
caller's prompt; the policy about what is legitimate lives in the caller's
config. See docs/reference/records.md.
"""

import json

import pytest

from openaugi.store.sqlite import SQLiteStore

NOW = "2026-08-20T10:00:00Z"
LATER = "2026-08-20T11:00:00Z"


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "records.db")
    yield s
    s.close()


class TestWriteAndRead:
    def test_round_trips_arbitrary_data(self, store: SQLiteStore):
        # openaugi has no schema for this and should never grow one.
        store.write_record("anything", "r1", {"a": 1, "nested": {"b": [1, 2]}}, NOW)
        (rec,) = store.list_records("anything")
        assert rec["a"] == 1
        assert rec["nested"] == {"b": [1, 2]}
        assert rec["id"] == "r1"

    def test_collections_are_isolated(self, store: SQLiteStore):
        store.write_record("alpha", "r1", {"x": 1}, NOW)
        store.write_record("beta", "r1", {"x": 2}, NOW)
        # Same id in two collections is two records, not a collision.
        assert store.list_records("alpha")[0]["x"] == 1
        assert store.list_records("beta")[0]["x"] == 2

    def test_rewriting_an_id_replaces_rather_than_stacks(self, store: SQLiteStore):
        store.write_record("q", "same-subject", {"n": 1}, NOW)
        store.write_record("q", "same-subject", {"n": 2}, LATER)
        records = store.list_records("q")
        # Stable ids derived from the subject are how a queue avoids re-asking
        # the same question every run.
        assert len(records) == 1
        assert records[0]["n"] == 2
        assert records[0]["created_at"] == NOW  # created_at survives a rewrite
        assert records[0]["updated_at"] == LATER

    def test_an_unknown_collection_is_empty_not_an_error(self, store: SQLiteStore):
        assert store.list_records("never-written") == []


class TestFiltering:
    def test_filters_on_a_top_level_field(self, store: SQLiteStore):
        store.write_record("q", "a", {"state": "open"}, NOW)
        store.write_record("q", "b", {"state": "closed"}, NOW)
        assert [r["id"] for r in store.list_records("q", where={"state": "open"})] == ["a"]

    def test_filters_combine_as_and(self, store: SQLiteStore):
        store.write_record("q", "a", {"state": "open", "run": "1"}, NOW)
        store.write_record("q", "b", {"state": "open", "run": "2"}, NOW)
        got = store.list_records("q", where={"state": "open", "run": "2"})
        assert [r["id"] for r in got] == ["b"]

    def test_a_field_that_does_not_exist_matches_nothing(self, store: SQLiteStore):
        store.write_record("q", "a", {"state": "open"}, NOW)
        assert store.list_records("q", where={"missing": "x"}) == []


class TestOrdering:
    def test_oldest_first_by_default(self, store: SQLiteStore):
        store.write_record("q", "old", {}, "2026-08-01T00:00:00Z")
        store.write_record("q", "new", {}, "2026-08-20T00:00:00Z")
        # For a human queue, oldest first: a decision that has waited three
        # runs should not sit under one raised this morning.
        assert [r["id"] for r in store.list_records("q")] == ["old", "new"]

    def test_desc_reverses_it(self, store: SQLiteStore):
        store.write_record("q", "old", {}, "2026-08-01T00:00:00Z")
        store.write_record("q", "new", {}, "2026-08-20T00:00:00Z")
        assert [r["id"] for r in store.list_records("q", desc=True)] == ["new", "old"]

    def test_an_unknown_order_falls_back_rather_than_injecting(self, store: SQLiteStore):
        store.write_record("q", "a", {}, NOW)
        # `order` reaches an ORDER BY clause, so anything unrecognised must be
        # discarded, never interpolated.
        assert store.list_records("q", order="id; DROP TABLE records") == store.list_records("q")
        assert store.list_records("q") != []

    def test_limit_caps_the_result(self, store: SQLiteStore):
        for i in range(5):
            store.write_record("q", str(i), {}, NOW)
        assert len(store.list_records("q", limit=2)) == 2


class TestUpdate:
    def test_merges_rather_than_replaces(self, store: SQLiteStore):
        store.write_record("q", "a", {"state": "open", "keep": "me"}, NOW)
        assert store.update_record("q", "a", {"state": "closed"}, LATER) is True
        (rec,) = store.list_records("q")
        assert rec["state"] == "closed"
        assert rec["keep"] == "me"

    def test_reports_a_missing_record_instead_of_creating_one(self, store: SQLiteStore):
        # Updating something that vanished usually means a stale client;
        # silently creating it hides that.
        assert store.update_record("q", "ghost", {"x": 1}, LATER) is False
        assert store.list_records("q") == []


class TestMcpSurface:
    def test_the_tools_carry_no_vocabulary_of_their_own(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        # A workflow names its own collection; openaugi validates nothing
        # about it, which is the whole boundary being tested.
        assert json.loads(srv.write_record("whatever", "x", {"rule": "made-up"}))["status"] == "ok"
        got = json.loads(srv.list_records("whatever"))
        assert got["records"][0]["rule"] == "made-up"

    def test_write_record_requires_a_collection_and_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        assert json.loads(srv.write_record("", "x", {}))["status"] == "error"
        assert json.loads(srv.write_record("c", "  ", {}))["status"] == "error"

    def test_update_record_errors_on_a_missing_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        out = json.loads(srv.update_record("c", "nope", {"a": 1}))
        assert out["status"] == "error"
        assert "no such record" in out["reason"]

    def test_no_bespoke_workflow_tools_remain(self):
        import openaugi.mcp.server as srv

        # These encoded one user's review workflow — including a hardcoded
        # list of legitimate routing rules — into a general library. If one
        # comes back, the boundary has been crossed again.
        for gone in (
            "record_pass",
            "record_routing",
            "write_proposal",
            "list_proposals",
            "list_passes",
            "list_routings",
            "answer_proposal",
            "undo_routing",
        ):
            assert not hasattr(srv, gone), f"{gone} is workflow policy, not a general tool"
