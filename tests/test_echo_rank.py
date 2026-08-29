"""Stratify-then-cluster ranking: provenance strata, relative scoring, threads."""

from openaugi.model.block import Block
from openaugi.pipeline import echo_rank
from openaugi.pipeline.echo_rank import CAPTURE, rank


def _b(
    bid: str,
    title: str = "note",
    path: str = "_private/0-Fleeting-Inbox/2026-01-01.md",
    tags: list[str] | None = None,
    time: str = "2026-01-01",
) -> Block:
    return Block(
        id=bid,
        kind="data_block",
        content="x" * 80,
        title=title,
        block_time=time,
        tags=tags or [],
        metadata={"source_path": path},
    )


class TestSourceFacet:
    def test_capture_is_the_default_when_untagged(self):
        """Taxonomy: source/capture is assumed when no source tag is present."""
        assert echo_rank.source_facet(_b("1")) == CAPTURE
        assert echo_rank.source_facet(_b("2", path="_private/5-Journals/wk.md")) == CAPTURE

    def test_external_source_tags_recognized(self):
        """Imported content is someone else's words — never an echo."""
        assert echo_rank.source_facet(_b("3", tags=["source/podcast"])) == "source/podcast"
        assert echo_rank.source_facet(_b("4", tags=["#source/readwise"])) == "source/readwise"
        assert not echo_rank.is_own_writing(_b("5", tags=["source/ai-chat"]))

    def test_folder_infers_source_when_untagged(self):
        """74% of blocks carry no tags — the folder has to carry the facet."""
        assert not echo_rank.is_own_writing(_b("6", path="_private/2-Reference/Snipd/ep.md"))

    def test_non_source_tags_do_not_change_the_facet(self):
        assert echo_rank.is_own_writing(_b("7", tags=["note-type/moc", "area/openaugi"]))


class TestNoteType:
    def test_tag_wins(self):
        assert echo_rank.note_type(_b("1", tags=["note-type/pmoc"])) == "note-type/pmoc"

    def test_folder_infers_when_untagged(self):
        assert echo_rank.note_type(_b("2")) == "note-type/daily-journal"
        assert (
            echo_rank.note_type(_b("3", path="_private/3-MOCs and Projects/x.md"))
            == "note-type/moc"
        )
        assert (
            echo_rank.note_type(_b("4", path="_private/5-Journals/w.md")) == "note-type/reflection"
        )

    def test_unknown_when_neither(self):
        assert echo_rank.note_type(_b("5", path="somewhere/else.md")) == "note-type/unknown"


class TestRelativeScoring:
    def test_external_source_blocks_dropped(self):
        """A higher-scoring podcast transcript must still lose to own writing."""
        scored = [(_b("a"), 0.7), (_b("b", tags=["source/podcast"]), 0.9)]
        out = rank(scored)
        assert out.dropped_external == 1
        assert [b.id for b in out.blocks] == ["a"]

    def test_near_flat_pool_is_thinned_to_the_top(self):
        """The observed real-world case: everything ~0.60, no real separation.

        A z-cut still thins it to the top of the ramp instead of handing eight
        indistinguishable candidates to the judge.
        """
        scored = [(_b(str(i)), 0.60 + i * 0.001) for i in range(8)]
        out = rank(scored)
        assert len(out.candidates) < len(scored)
        assert out.candidates[0][0].id == "7"  # best first
        assert out.dropped_noise == len(scored) - len(out.candidates)

    def test_perfectly_flat_pool_defers_to_judgment(self):
        """Zero variance means nothing to discriminate on — keep all, let the
        judge decide rather than picking arbitrarily."""
        scored = [(_b(str(i)), 0.61) for i in range(8)]
        out = rank(scored)
        assert len(out.candidates) == 8
        assert out.dropped_noise == 0

    def test_genuine_spike_survives_and_ranks_first(self):
        scored = [(_b(str(i)), 0.52) for i in range(7)] + [(_b("spike"), 0.85)]
        out = rank(scored)
        assert out.blocks[0].id == "spike"
        assert out.candidates[0][2] > 2.0  # z-score well above the pool

    def test_small_pool_skips_z_scoring(self):
        """With too few candidates a z-score is meaningless — defer to judgment."""
        scored = [(_b("a"), 0.9), (_b("b"), 0.4)]
        out = rank(scored)
        assert len(out.candidates) == 2
        assert out.dropped_noise == 0

    def test_empty_input(self):
        out = rank([])
        assert out.candidates == [] and out.threads == []


class TestClustering:
    def test_threads_group_by_title_and_count_recurrence(self):
        scored = [
            (_b("a", title="PMOC - Audacity", time="2026-07-14"), 0.80),
            (_b("b", title="PMOC - Audacity", time="2026-06-15"), 0.78),
            (_b("c", title="Other note", time="2026-05-01"), 0.50),
            (_b("d", title="Third", time="2026-04-01"), 0.50),
            (_b("e", title="Fourth", time="2026-03-01"), 0.50),
            (_b("f", title="Fifth", time="2026-02-01"), 0.50),
        ]
        out = rank(scored)
        recurring = out.recurring
        assert len(recurring) == 1
        thread = recurring[0]
        assert thread.title == "PMOC - Audacity"
        assert thread.recurrence == 2
        assert thread.span == ("2026-06-15", "2026-07-14")
        assert thread.note_type == "note-type/daily-journal"

    def test_single_hit_notes_are_not_threads(self):
        out = rank([(_b(str(i), title=f"n{i}"), 0.5 + i * 0.1) for i in range(5)])
        assert out.recurring == []
