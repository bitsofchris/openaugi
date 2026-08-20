"""Review-pass accountability — the pass log and the proposals queue.

These tables exist to make one line legible: **the agent executes Chris's
rules, it does not exercise judgment.** So the tests that matter are the ones
policing the boundary — routing must name a deterministic rule, and a
proposal must not silently become an action.

See docs/plans/changeset-review.md in the private-augi-mobile repo.
"""

import json

import pytest

from openaugi.store.sqlite import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "review.db")
    yield s
    s.close()


class TestPassLog:
    def test_a_pass_counts_what_it_left_alone(self, store: SQLiteStore):
        # Left-alone is the expected majority, not a backlog: most blocks
        # belong in the daily note and are not debt.
        store.record_pass("p1", "2026-08-20T10:00:00Z", scanned=68, left_alone=35)
        (run,) = store.list_passes()
        assert run["scanned"] == 68
        assert run["left_alone"] == 35
        assert run["routed"] == 0

    def test_recording_a_pass_twice_updates_rather_than_forks(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-20T10:00:00Z", scanned=10)
        store.record_pass("p1", "2026-08-20T10:05:00Z", scanned=68)
        passes = store.list_passes()
        assert len(passes) == 1
        assert passes[0]["scanned"] == 68

    def test_passes_come_back_newest_first(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-18T10:00:00Z")
        store.record_pass("p2", "2026-08-20T10:00:00Z")
        assert [p["id"] for p in store.list_passes()] == ["p2", "p1"]


class TestRoutingAudit:
    def test_every_routing_carries_the_rule_that_fired(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-20T10:00:00Z")
        store.record_routing("p1", "b1", "AMOC - Health", "tag", "2026-08-20T10:00:01Z")
        (r,) = store.list_routings("p1")
        # Routing is autonomous BECAUSE it is only ever obedience. Without the
        # rule recorded, that claim is unfalsifiable.
        assert r["rule"] == "tag"
        assert r["container"] == "AMOC - Health"

    def test_routing_increments_the_pass_count(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-20T10:00:00Z")
        for i in range(3):
            store.record_routing("p1", f"b{i}", "AMOC - Health", "link", "2026-08-20T10:00:01Z")
        assert store.list_passes()[0]["routed"] == 3

    def test_an_undone_routing_is_tombstoned_not_deleted(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-20T10:00:00Z")
        store.record_routing("p1", "b1", "AMOC - Health", "tag", "2026-08-20T10:00:01Z")
        (r,) = store.list_routings("p1")

        assert store.mark_routing_undone(r["id"], "2026-08-20T11:00:00Z") is True
        # Hidden by default — an undo is a correction, not history to re-read.
        assert store.list_routings("p1") == []
        # But the log stays honest about what the pass actually did.
        (undone,) = store.list_routings("p1", include_undone=True)
        assert undone["undone_at"] == "2026-08-20T11:00:00Z"

    def test_undoing_twice_is_a_no_op(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-20T10:00:00Z")
        store.record_routing("p1", "b1", "X", "tag", "2026-08-20T10:00:01Z")
        (r,) = store.list_routings("p1")
        assert store.mark_routing_undone(r["id"], "2026-08-20T11:00:00Z") is True
        assert store.mark_routing_undone(r["id"], "2026-08-20T12:00:00Z") is False


class TestProposals:
    def test_a_proposal_is_not_an_action(self, store: SQLiteStore):
        store.write_proposal(
            "promote-silver",
            "promote",
            "2026-08-20T10:00:00Z",
            block_ids=["b1", "b2"],
            target="Silver notes",
            why="5 blocks orbit this",
        )
        (p,) = store.list_proposals()
        # Nothing is applied by proposing. State is the whole point.
        assert p["state"] == "proposed"
        assert p["answered_at"] is None
        assert p["block_ids"] == ["b1", "b2"]

    def test_reproposing_the_same_idea_updates_in_place(self, store: SQLiteStore):
        store.write_proposal(
            "promote-silver",
            "promote",
            "2026-08-20T10:00:00Z",
            block_ids=["b1"],
            target="Silver notes",
        )
        store.write_proposal(
            "promote-silver",
            "promote",
            "2026-08-21T10:00:00Z",
            block_ids=["b1", "b2", "b3"],
            target="Silver notes",
        )
        proposals = store.list_proposals()
        # Stacking duplicates is how the Dashboard reached 24 open items.
        assert len(proposals) == 1
        assert proposals[0]["block_ids"] == ["b1", "b2", "b3"]

    def test_proposals_come_back_oldest_first(self, store: SQLiteStore):
        store.write_proposal("a", "promote", "2026-08-01T10:00:00Z")
        store.write_proposal("b", "promote", "2026-08-20T10:00:00Z")
        # A decision waiting three passes shouldn't sit under this morning's.
        assert [p["id"] for p in store.list_proposals()] == ["a", "b"]

    def test_accepting_can_edit_in_the_same_call(self, store: SQLiteStore):
        store.write_proposal(
            "promote-silver",
            "promote",
            "2026-08-20T10:00:00Z",
            target="Silver notes",
            block_ids=["b1", "b2"],
        )
        assert store.answer_proposal(
            "promote-silver",
            "accepted",
            "2026-08-20T11:00:00Z",
            target="Silver notes on the fly",
            block_ids=["b1"],
        )
        (p,) = store.list_proposals(state="accepted")
        # Chris reviews by changing the thing, not by rejecting and waiting.
        assert p["target"] == "Silver notes on the fly"
        assert p["block_ids"] == ["b1"]

    def test_a_decline_is_durable_and_leaves_the_queue(self, store: SQLiteStore):
        store.write_proposal("promote-x", "promote", "2026-08-20T10:00:00Z")
        store.answer_proposal("promote-x", "declined", "2026-08-20T11:00:00Z")
        assert store.list_proposals(state="proposed") == []
        # Recorded, so it isn't re-proposed next week just because a pass ran.
        assert [p["id"] for p in store.list_proposals(state="declined")] == ["promote-x"]

    def test_answering_a_missing_proposal_reports_failure(self, store: SQLiteStore):
        assert store.answer_proposal("nope", "accepted", "2026-08-20T11:00:00Z") is False

    def test_proposals_count_onto_their_pass(self, store: SQLiteStore):
        store.record_pass("p1", "2026-08-20T10:00:00Z")
        store.write_proposal("a", "promote", "2026-08-20T10:00:01Z", pass_id="p1")
        store.write_proposal("b", "adopt", "2026-08-20T10:00:02Z", pass_id="p1")
        assert store.list_passes()[0]["proposed"] == 2


class TestMcpBoundary:
    """The tools are where the rule boundary is enforced, so it is tested there."""

    def test_record_routing_refuses_a_rule_that_is_not_a_rule(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        srv.record_pass("p1")
        # "similar" is the whole thing this design exists to refuse: semantic
        # resemblance is not a rule, and a block that merely feels related
        # stays in the daily note.
        out = json.loads(srv.record_routing("p1", "b1", "AMOC - Health", "similar"))
        assert out["status"] == "error"
        assert "instruction" in out["reason"]
        assert json.loads(srv.list_routings("p1"))["count"] == 0

    def test_record_routing_accepts_the_three_rules(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        srv.record_pass("p1")
        for i, rule in enumerate(("instruction", "link", "tag")):
            assert json.loads(srv.record_routing("p1", f"b{i}", "X", rule))["status"] == "ok"
        assert json.loads(srv.list_routings("p1"))["count"] == 3

    def test_write_proposal_refuses_a_routing_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        # Routings are decided by rules, never proposed. A "maybe this goes
        # here" queue is exactly what this replaces.
        out = json.loads(srv.write_proposal("p", "route", target="X"))
        assert out["status"] == "error"

    def test_answer_proposal_refuses_an_unknown_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAUGI_DB", str(tmp_path / "t.db"))
        import openaugi.mcp.server as srv

        srv._store = None
        srv.write_proposal("p", "promote", target="X")
        assert json.loads(srv.answer_proposal("p", "maybe"))["status"] == "error"
