"""Proactive echo — eligibility filters, log writing, and janitor actions."""

import json
from pathlib import Path

from openaugi.model.block import Block
from openaugi.pipeline import echo, echo_janitor


def _block(content: str, path: str = "_private/0-Fleeting-Inbox/2026-08-29.md", **meta) -> Block:
    return Block(
        id=meta.pop("id", "abc123"),
        kind="data_block",
        content=content,
        title="2026-08-29",
        block_time="2026-08-29",
        metadata={"source_path": path, **meta},
    )


class TestEligibility:
    def test_daily_note_prose_is_eligible(self):
        assert echo.is_echo_eligible(_block("Thinking about the medallion architecture again."))

    def test_non_daily_paths_excluded(self):
        assert not echo.is_echo_eligible(
            _block(
                "Real prose that is plenty long to pass the filter.", path="OpenAugi/Notes/x.md"
            )
        )

    def test_link_only_block_excluded(self):
        """A bare wikilink scored 1.000 against another bare link in the replay."""
        assert not echo.is_echo_eligible(_block("[[Class Reunion]]"))

    def test_short_block_excluded(self):
        assert not echo.is_echo_eligible(_block("Meditate on the good books"))

    def test_zzz_instruction_excluded(self):
        """Dispatch owns zzz blocks; echo must not double-handle them."""
        long = "zzz: go research all of my high notes and build the analysis for me"
        assert not echo.is_echo_eligible(_block(long))
        assert not echo.is_echo_eligible(
            _block("A perfectly ordinary long thought here.", zzz_instructions=["do a thing"])
        )


class TestLogWriting:
    def test_render_includes_marker_and_boxes(self):
        block = _block("Following up on the autowiki structure and medallion levels.")
        cand = Block(
            id="old1",
            kind="data_block",
            content="bronze silver gold medallion",
            title="PMOC - Audacity",
            block_time="2026-07-14",
            metadata={},
        )
        out = echo._render(
            block,
            [{"title": "PMOC - Audacity", "why": "you named this in July"}],
            {"PMOC - Audacity": cand},
        )
        assert "<!-- echo:abc123 -->" in out
        assert "[[PMOC - Audacity]] (2026-07-14) — you named this in July" in out
        assert "- [ ] promote → new note" in out
        assert "- [ ] good match" in out

    def test_ensure_log_creates_dated_file(self, tmp_path: Path):
        path = echo._log_path(tmp_path, "2026-08-29")
        assert path == tmp_path / "OpenAugi/2026/08/29/Augi Log.md"
        echo._ensure_log(path, "2026-08-29")
        assert "# Augi Log — 2026-08-29" in path.read_text()

    def test_no_llm_configured_is_a_no_op(self, tmp_path: Path):
        stats = echo.run_echo([_block("x" * 100)], tmp_path, None, None, {})
        assert stats == {"watched": 0, "spoke": 0, "quiet": 0}


class TestJanitor:
    def _log_with(self, tmp_path: Path, checked: str) -> Path:
        path = tmp_path / "OpenAugi/2026/08/29/Augi Log.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Augi Log — 2026-08-29\n\n"
            "<!-- echo:abc123 -->\n"
            '### echo on "the medallion architecture…"\n'
            "- [[PMOC - Audacity]] (2026-07-14) — you named this in July\n\n"
            f"- [{'x' if checked == 'promote' else ' '}] promote → new note\n"
            f"- [{'x' if checked == 'good' else ' '}] good match\n"
            f"- [{'x' if checked == 'bad' else ' '}] bad match\n",
            encoding="utf-8",
        )
        return path

    def test_good_match_writes_feedback_and_confirms(self, tmp_path: Path):
        path = self._log_with(tmp_path, "good")
        assert echo_janitor.process_log(path, tmp_path) == 1
        record = json.loads((tmp_path / echo_janitor.FEEDBACK_LOG).read_text().strip())
        assert record["signal"] == "liked"
        assert record["source"] == "proactive-echo"
        assert record["block_id"] == "abc123"
        text = path.read_text()
        assert "✓ feedback recorded" in text
        assert "- [x]" not in text  # consumed
        assert "- [ ]" not in text  # remaining boxes cleared once answered

    def test_bad_match_records_dislike(self, tmp_path: Path):
        path = self._log_with(tmp_path, "bad")
        echo_janitor.process_log(path, tmp_path)
        assert (
            json.loads((tmp_path / echo_janitor.FEEDBACK_LOG).read_text())["signal"] == "disliked"
        )

    def test_promote_writes_note_with_context_and_log(self, tmp_path: Path):
        path = self._log_with(tmp_path, "promote")
        assert echo_janitor.process_log(path, tmp_path) == 1
        notes = list((tmp_path / "OpenAugi/Notes").glob("*.md"))
        assert len(notes) == 1
        body = notes[0].read_text()
        assert "## Context" in body and "## Log" in body
        assert "[[PMOC - Audacity]]" in body
        assert "#human-review" in body
        assert "✓ promoted →" in path.read_text()

    def test_janitor_is_idempotent(self, tmp_path: Path):
        path = self._log_with(tmp_path, "good")
        assert echo_janitor.process_log(path, tmp_path) == 1
        assert echo_janitor.process_log(path, tmp_path) == 0  # nothing left to do
        assert len((tmp_path / echo_janitor.FEEDBACK_LOG).read_text().strip().splitlines()) == 1

    def test_untouched_log_does_nothing(self, tmp_path: Path):
        path = self._log_with(tmp_path, "none")
        assert echo_janitor.process_log(path, tmp_path) == 0
        assert not (tmp_path / echo_janitor.FEEDBACK_LOG).exists()

    def test_process_changed_only_targets_augi_logs(self, tmp_path: Path):
        path = self._log_with(tmp_path, "good")
        other = tmp_path / "notes.md"
        other.write_text("- [x] good match\n")
        assert echo_janitor.process_changed({str(path), str(other)}, tmp_path) == 1
