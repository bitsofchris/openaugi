"""The shared Augi Log file — section order, legacy logs, heartbeat, eligibility."""

from pathlib import Path

from openaugi.model.block import Block
from openaugi.pipeline import augi_log
from openaugi.pipeline.augi_log import (
    ECHO_HEADING,
    QUIET_HEADING,
    ROUTING_HEADING,
    append_to_section,
    assemble,
    ensure_log,
    is_capture_block,
    log_path,
    split,
    write_heartbeat,
)


def _block(content: str, path: str = "_private/0-Fleeting-Inbox/2026-09-03.md", **meta) -> Block:
    return Block(
        id="b1", kind="data_block", content=content, metadata={"source_path": path, **meta}
    )


def _log(tmp_path: Path) -> Path:
    path = log_path(tmp_path, "2026-09-03")
    ensure_log(path, "2026-09-03")
    return path


class TestEligibility:
    def test_daily_prose_passes(self):
        assert is_capture_block(_block("A real thought with enough words to count as prose."))

    def test_other_folders_fail(self):
        assert not is_capture_block(
            _block("A real thought with enough words.", path="OpenAugi/x.md")
        )

    def test_link_only_and_short_fail(self):
        assert not is_capture_block(_block("[[Some Note]]"))
        assert not is_capture_block(_block("too short"))

    def test_zzz_blocks_belong_to_dispatch(self):
        assert not is_capture_block(_block("zzz: research this for me please, thoroughly"))
        assert not is_capture_block(
            _block("A perfectly ordinary long thought here.", zzz_instructions=["do it"])
        )


class TestSections:
    def test_sections_land_in_canonical_order_whatever_the_write_order(self, tmp_path):
        path = _log(tmp_path)
        append_to_section(path, QUIET_HEADING, "<!-- quiet:a -->\nq\n", intro="*debug*")
        append_to_section(path, ECHO_HEADING, "<!-- echo:b -->\ne\n")
        append_to_section(path, ROUTING_HEADING, "<!-- route:c -->\nr\n", intro="- [ ] process")
        text = path.read_text()
        assert text.index(ROUTING_HEADING) < text.index(ECHO_HEADING) < text.index(QUIET_HEADING)
        assert text.index("- [ ] process") < text.index("<!-- route:c -->")
        assert text.count(QUIET_HEADING) == 1 and "*debug*" in text

    def test_intro_written_once_and_rows_accumulate(self, tmp_path):
        path = _log(tmp_path)
        for i in range(3):
            append_to_section(path, QUIET_HEADING, f"<!-- quiet:{i} -->\n", intro="*debug*")
        text = path.read_text()
        assert text.count("*debug*") == 1
        assert all(f"<!-- quiet:{i} -->" in text for i in range(3))

    def test_heartbeat_stays_last(self, tmp_path):
        path = _log(tmp_path)
        write_heartbeat(path, {"watched": 1, "spoke": 0, "quiet": 1})
        append_to_section(path, ECHO_HEADING, "<!-- echo:b -->\n")
        append_to_section(path, ROUTING_HEADING, "<!-- route:c -->\n")
        text = path.read_text()
        assert text.rstrip().endswith("*watched 1 · spoke 0 · quiet 1*")
        assert text.count("<!-- heartbeat") == 1

    def test_heartbeat_accumulates(self, tmp_path):
        path = _log(tmp_path)
        write_heartbeat(path, {"watched": 2, "spoke": 1, "quiet": 1})
        write_heartbeat(path, {"watched": 3, "spoke": 0, "quiet": 3})
        assert "<!-- heartbeat 5 1 4 -->" in path.read_text()

    def test_unknown_heading_is_refused(self, tmp_path):
        path = _log(tmp_path)
        try:
            append_to_section(path, "## Nope", "x")
        except ValueError:
            return
        raise AssertionError("unknown section accepted")

    def test_split_and_assemble_round_trip(self, tmp_path):
        path = _log(tmp_path)
        append_to_section(path, ROUTING_HEADING, "r\n")
        append_to_section(path, QUIET_HEADING, "q\n", intro="*debug*")
        write_heartbeat(path, {"watched": 1, "spoke": 1, "quiet": 0})
        text = path.read_text()
        assert assemble(*split(text)) == text


class TestLegacyLogs:
    """Logs from before the sections existed: echoes sit right under the header."""

    LEGACY = (
        "# Augi Log — 2026-09-02\n\n- [ ] seen\n\n"
        '<!-- echo:old -->\n\n### echo on "something…"\n- [[Note]] — why\n\n'
        "- [ ] promote → new note\n- [ ] good match\n- [ ] bad match\n\n"
        "## Quiet — closest match, not surfaced\n\n*debug*\n\n"
        '<!-- quiet:q1 -->\n\n**"x…"**\n- [ ] should have surfaced\n- [ ] correctly quiet\n\n'
        "<!-- heartbeat 2 1 1 -->\n*watched 2 · spoke 1 · quiet 1*\n"
    )

    def test_old_rows_stay_and_new_rows_go_under_headings(self, tmp_path):
        path = tmp_path / "Augi Log.md"
        path.write_text(self.LEGACY, encoding="utf-8")
        append_to_section(path, ECHO_HEADING, '<!-- echo:new -->\n\n### echo on "y…"\n')
        append_to_section(path, QUIET_HEADING, "<!-- quiet:q2 -->\n")
        text = path.read_text()
        assert (
            text.index("<!-- echo:old -->")
            < text.index(ECHO_HEADING)
            < text.index("<!-- echo:new -->")
        )
        assert (
            text.index("<!-- quiet:q1 -->")
            < text.index("<!-- quiet:q2 -->")
            < text.index("<!-- heartbeat")
        )
        assert text.count(QUIET_HEADING) == 1
        assert "<!-- heartbeat 2 1 1 -->" in text

    def test_janitor_echo_body_stops_at_any_heading(self, tmp_path):
        from openaugi.pipeline import echo_janitor

        path = tmp_path / "Augi Log.md"
        path.write_text(
            '# Augi Log\n\n<!-- echo:a1 -->\n\n### echo on "s…"\n- [[Note]] — why\n\n'
            "- [x] good match\n- [ ] bad match\n\n"
            '## Echoes\n\n<!-- echo:b2 -->\n\n### echo on "t…"\n- [[Other]] — why\n\n'
            "- [ ] good match\n- [ ] bad match\n",
            encoding="utf-8",
        )
        assert echo_janitor.process_log(path, tmp_path) == 1
        text = path.read_text()
        assert "✓ feedback recorded (good match)" in text
        # the untouched echo under the heading kept its boxes
        assert text.count("- [ ] bad match") == 1 and "<!-- echo:b2 -->" in text


def test_echo_module_reexports_the_shared_names():
    from openaugi.pipeline import echo

    assert echo.QUIET_HEADING == augi_log.QUIET_HEADING
    assert echo.DAILY_PREFIX == augi_log.DAILY_PREFIX
