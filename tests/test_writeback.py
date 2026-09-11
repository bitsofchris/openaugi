"""Write-back — the shared feedback log and the tick grammar it is read with."""

import inspect
import json
import re
from pathlib import Path

import pytest

from openaugi.pipeline import board_janitor, echo_janitor, route, routing_janitor, writeback
from openaugi.pipeline.writeback import (
    FEEDBACK_LOG,
    aaa_re,
    append_feedback,
    box_re,
    now,
    read_feedback,
    ticked,
)


class TestLog:
    def test_append_creates_the_folder_and_one_line_per_record(self, tmp_path):
        append_feedback(tmp_path, {"source": "currency-board", "signal": "done"})
        append_feedback(tmp_path, {"source": "routing", "signal": "undo"})

        lines = (tmp_path / FEEDBACK_LOG).read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["signal"] for line in lines] == ["done", "undo"]

    def test_read_is_empty_when_the_log_does_not_exist(self, tmp_path):
        assert list(read_feedback(tmp_path)) == []

    def test_read_skips_blank_truncated_and_non_object_lines(self, tmp_path):
        path = tmp_path / FEEDBACK_LOG
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"source": "routing", "signal": "liked"}\n'
            "\n"
            '{"source": "routing", "signal": tru\n'  # truncated mid-write
            "[1, 2, 3]\n"  # valid JSON, but not a record
            "   \n"
            '{"source": "routing", "signal": "undo"}\n',
            encoding="utf-8",
        )
        assert [r["signal"] for r in read_feedback(tmp_path)] == ["liked", "undo"]

    def test_read_filters_by_source(self, tmp_path):
        for source in ("routing", "currency-board", "routing"):
            append_feedback(tmp_path, {"source": source})
        assert len(list(read_feedback(tmp_path, source="routing"))) == 2
        assert list(read_feedback(tmp_path, source="nobody")) == []

    def test_now_is_utc_iso(self):
        stamp = now()
        assert stamp.endswith("+00:00")
        assert stamp[4] == "-" and stamp[10] == "T"


class TestBoxGrammar:
    def test_matches_a_plain_box_and_reports_the_label(self):
        pattern = box_re("done", "not doing", "someday")
        match = pattern.match("- [x] not doing")
        assert match and match.group("label") == "not doing"
        assert ticked(match)

    def test_an_untouched_box_is_not_ticked(self):
        match = box_re("done").match("- [ ] done")
        assert match and not ticked(match)

    def test_either_case_of_x_counts(self):
        assert ticked(box_re("done").match("- [X] done"))

    def test_a_label_outside_the_vocabulary_does_not_match(self):
        assert box_re("done", "someday").match("- [x] not doing") is None

    def test_callout_prefix_is_captured_only_when_asked_for(self):
        line = ">   - [x] done"
        assert box_re("done").match(line) is None
        match = box_re("done", callout=True).match(line)
        assert match and match.group("pre") == ">   "

    def test_pre_is_empty_rather_than_absent_without_callout(self):
        assert box_re("done").match("- [x] done").group("pre") == ""

    def test_no_labels_accepts_any_label(self):
        match = box_re().match("- [x] extend [[Some Note]]")
        assert match and match.group("label") == "extend [[Some Note]]"

    def test_suffix_annotation_is_dropped_from_the_label(self):
        pattern = box_re(suffix=True)
        assert pattern.match("- [ ] link [[X]] — because you linked it").group("label") == (
            "link [[X]]"
        )
        assert pattern.match("- [ ] memory").group("label") == "memory"

    def test_multiline_finds_every_box_in_a_body(self):
        body = "some text\n- [ ] memory\n- [x] new note\nmore text\n"
        found = [(m.group("label"), ticked(m)) for m in box_re(multiline=True).finditer(body)]
        assert found == [("memory", False), ("new note", True)]


class TestAaaGrammar:
    def test_reads_the_comment_and_is_case_insensitive(self):
        assert aaa_re().match("aaa: not until October").group("reason") == "not until October"
        assert aaa_re().match("AAA: later").group("reason") == "later"

    def test_an_empty_placeholder_matches_with_an_empty_reason(self):
        assert aaa_re().match("aaa:").group("reason") == ""

    def test_require_text_rejects_the_empty_placeholder(self):
        assert aaa_re(require_text=True).match("aaa:") is None
        assert aaa_re(require_text=True).match("aaa: why").group("reason") == "why"

    def test_indent_and_callout_are_opt_in(self):
        assert aaa_re().match("  aaa: why") is None
        assert aaa_re(indent=True).match("  aaa: why").group("reason") == "why"
        assert aaa_re(indent=True).match("> aaa: why") is None
        assert aaa_re(callout=True).match(">  aaa: why").group("reason") == "why"

    def test_spaced_colon_is_opt_in(self):
        assert aaa_re().match("aaa : why") is None
        assert aaa_re(spaced_colon=True).match("aaa : why").group("reason") == "why"

    def test_multiline_searches_a_body(self):
        body = "- [ ] hold\naaa: wait for October\n"
        assert aaa_re(multiline=True).search(body).group("reason") == "wait for October"


class TestTheJanitorsShareOneLog:
    """The point of the module: one path, one shape, four callers."""

    def test_the_log_path_is_defined_in_exactly_one_place(self):
        """Four modules spelled this literal out before. Prose may mention it."""
        package = Path(writeback.__file__).parent.parent
        assignment = re.compile(r"\s*[A-Z_]+\s*=\s*[\"']OpenAugi/Capture/feedback-log\.ndjson")
        defined_in = [
            str(path.relative_to(package))
            for path in sorted(package.rglob("*.py"))
            if any(
                assignment.match(line) for line in path.read_text(encoding="utf-8").splitlines()
            )
        ]
        assert defined_in == ["pipeline/writeback.py"]

    @pytest.mark.parametrize("module", [board_janitor, echo_janitor, route, routing_janitor])
    def test_every_janitor_writes_through_the_shared_helpers(self, module):
        source = inspect.getsource(module)
        assert "def _append_feedback" not in source
        assert "def _now" not in source

    def test_a_board_tick_and_a_routing_tick_land_in_the_same_file(self, tmp_path):
        append_feedback(tmp_path, {"ts": now(), "source": "currency-board", "signal": "done"})
        append_feedback(tmp_path, {"ts": now(), "source": "routing", "signal": "liked"})

        rows = list(read_feedback(tmp_path))
        assert len(rows) == 2
        assert {row["source"] for row in rows} == {"currency-board", "routing"}
