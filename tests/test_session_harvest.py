"""Tests for scripts/session_harvest.py (loaded by file path; scripts/ is not a package)."""

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "session_harvest", Path(__file__).parent.parent / "scripts" / "session_harvest.py"
)
sh = importlib.util.module_from_spec(_SPEC)
sys.modules["session_harvest"] = sh  # dataclasses need the module importable by name
_SPEC.loader.exec_module(sh)

# The window every test uses unless it says otherwise: one local calendar day.
DAY = datetime(2026, 9, 7)
NEXT_DAY = datetime(2026, 9, 8)


def _write_jsonl(path: Path, objs) -> None:
    path.write_text("\n".join(json.dumps(o) for o in objs), encoding="utf-8")


def _ts(hour: int, day: int = 7) -> str:
    """A naive local timestamp — _local() leaves naive stamps alone, so no tz math."""
    return f"2026-09-{day:02d}T{hour:02d}:30:00"


def _user(text: str, ts: str, **extra) -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": ts,
        "origin": {"kind": "human"},
        "cwd": "/Users/testuser/repos/demo",
        **extra,
    }


def _assistant(text: str, ts: str, **extra) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "timestamp": ts,
        **extra,
    }


def _claude_file(tmp_path: Path, objs, name: str = "abc12345-session") -> Path:
    proj = tmp_path / "projects" / "-Users-someone-repos-demo"
    proj.mkdir(parents=True, exist_ok=True)
    path = proj / f"{name}.jsonl"
    _write_jsonl(path, objs)
    return path


# --- window resolution -----------------------------------------------------


def test_day_pins_one_local_calendar_day():
    start, end = sh.resolve_window("2026-09-07", days=1)
    assert (start, end) == (DAY, NEXT_DAY)


def test_days_window_ends_at_today_midnight_so_today_is_excluded():
    now = datetime(2026, 9, 8, 6, 0)
    start, end = sh.resolve_window(None, days=1, now=now)
    assert (start, end) == (DAY, NEXT_DAY)
    start, _ = sh.resolve_window(None, days=3, now=now)
    assert start == datetime(2026, 9, 5)


def test_utc_timestamps_are_converted_to_local():
    """Transcripts stamp UTC with a trailing Z; windows are local."""
    utc = sh._local("2026-09-07T12:00:00Z")
    naive = sh._local("2026-09-07T12:00:00")
    assert utc is not None and utc.tzinfo is None
    assert naive == datetime(2026, 9, 7, 12, 0)
    assert sh._local("not-a-time") is None
    assert sh._local("") is None


# --- parsing ---------------------------------------------------------------


def test_only_turns_inside_the_window_are_kept_but_all_are_counted(tmp_path):
    path = _claude_file(
        tmp_path,
        [
            _user("before the window", _ts(10, day=6)),
            _user("inside one", _ts(9)),
            _user("inside two", _ts(11)),
            _user("after the window", _ts(10, day=8)),
        ],
    )
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert [t.text for t in s.turns] == ["inside one", "inside two"]
    assert s.total_human_turns == 4
    assert s.opening_turn == "before the window"  # opening is the session's, not the window's


def test_session_with_no_turns_in_window_is_dropped(tmp_path):
    path = _claude_file(tmp_path, [_user("last week", _ts(10, day=1))])
    assert sh.parse_claude_slice(path, DAY, NEXT_DAY) is None


def test_harness_noise_and_non_human_origin_are_excluded(tmp_path):
    path = _claude_file(
        tmp_path,
        [
            _user("<system-reminder>not me</system-reminder>", _ts(8)),
            _user("Caveat: injected", _ts(8)),
            _user("   ", _ts(8)),
            _user("a hook wrote this", _ts(9), origin={"kind": "hook"}),
            _user("real question", _ts(10)),
        ],
    )
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert [t.text for t in s.turns] == ["real question"]


def test_sidechain_subagent_turns_are_excluded(tmp_path):
    path = _claude_file(
        tmp_path,
        [
            _user("mine", _ts(9)),
            _user("subagent prompt", _ts(10), isSidechain=True),
        ],
    )
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert [t.text for t in s.turns] == ["mine"]


def test_reply_is_the_longest_assistant_message_before_the_next_turn(tmp_path):
    path = _claude_file(
        tmp_path,
        [
            _user("what is a harness?", _ts(9)),
            _assistant("Let me check.", _ts(9)),
            _assistant("A harness is context, state, tools and prompts.", _ts(9)),
            _assistant("Done.", _ts(9)),
            _user("thanks", _ts(10)),
            _assistant("You bet.", _ts(10)),
        ],
    )
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert s.turns[0].reply == "A harness is context, state, tools and prompts."
    assert s.turns[1].reply == "You bet."


def test_replies_do_not_attach_to_turns_outside_the_window(tmp_path):
    path = _claude_file(
        tmp_path,
        [
            _user("inside", _ts(9)),
            _assistant("in-window reply", _ts(9)),
            _user("outside", _ts(10, day=8)),
            _assistant("out-of-window reply", _ts(10, day=8)),
        ],
    )
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert [t.reply for t in s.turns] == ["in-window reply"]


def test_ai_title_and_cwd_are_captured(tmp_path):
    path = _claude_file(
        tmp_path,
        [{"type": "ai-title", "aiTitle": "Harness matters"}, _user("go", _ts(9))],
    )
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert s.title == "Harness matters"
    assert s.project_name == "demo"
    assert s.resume_command.endswith("claude --resume abc12345-session")


def test_malformed_lines_are_skipped(tmp_path):
    path = _claude_file(tmp_path, [_user("real", _ts(9))])
    path.write_text("{not json\n" + path.read_text() + "\nnull\n", encoding="utf-8")
    s = sh.parse_claude_slice(path, DAY, NEXT_DAY)
    assert [t.text for t in s.turns] == ["real"]


def test_codex_slice_parses_meta_and_both_user_shapes(tmp_path):
    path = tmp_path / "rollout-2026-09-07.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "session_meta",
                "payload": {"session_id": "cdx-1", "cwd": "/Users/testuser/repos/demo"},
                "timestamp": _ts(8),
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "first shape"},
                "timestamp": _ts(9),
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "second shape"},
                "timestamp": _ts(10),
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "an answer"},
                "timestamp": _ts(10),
            },
        ],
    )
    s = sh.parse_codex_slice(path, DAY, NEXT_DAY)
    assert [t.text for t in s.turns] == ["first shape", "second shape"]
    assert s.turns[1].reply == "an answer"
    assert s.session_id == "cdx-1"
    assert s.resume_command == "codex resume cdx-1"


# --- selection -------------------------------------------------------------


def _slice_with(texts, opening=None) -> "sh.SessionSlice":
    s = sh.SessionSlice(tool="claude", session_id="s", source_path="p")
    s.opening_turn = opening if opening is not None else (texts[0] if texts else "")
    s.turns = [sh.Turn(timestamp="2026-09-07 09:00", text=t) for t in texts]
    s.total_human_turns = len(texts)
    return s


@pytest.mark.parametrize("trivial", ["y", "ok", "Do it", "next", "thanks", "??", "yeah"])
def test_trivial_turns_do_not_count_as_substance(trivial):
    s = _slice_with([trivial, "a real question about what I'm learning"])
    assert [t.text for t in s.substantive_turns] == ["a real question about what I'm learning"]


def test_agent_dispatched_sessions_are_excluded_by_default():
    agent = _slice_with(["one", "two"], opening="Read your skill file first:\n  /path")
    mine = _slice_with(["one", "two"])
    assert agent.is_agent_dispatched and not mine.is_agent_dispatched
    assert sh.select([agent, mine], min_turns=1, include_agent_sessions=False) == [mine]
    assert len(sh.select([agent, mine], min_turns=1, include_agent_sessions=True)) == 2


def test_min_turns_drops_thin_sessions():
    thin = _slice_with(["just one thing"])
    thick = _slice_with(["one", "two"])
    assert sh.select([thin, thick], min_turns=2, include_agent_sessions=False) == [thick]
    assert len(sh.select([thin, thick], min_turns=1, include_agent_sessions=False)) == 2


# --- rendering -------------------------------------------------------------


def test_markdown_flattens_turns_to_one_bullet_without_losing_words():
    s = _slice_with(["line one\n\nline two\nline three"])
    out = sh.render_markdown([s], DAY, NEXT_DAY, turn_chars=500, reply_chars=0)
    body = [ln for ln in out.splitlines() if ln.startswith("- **")]
    assert body == ["- **2026-09-07 09:00** — line one / line two line three"]


def test_markdown_trims_long_turns_and_replies():
    s = _slice_with(["x" * 200])
    s.turns[0].reply = "y" * 200
    out = sh.render_markdown([s], DAY, NEXT_DAY, turn_chars=20, reply_chars=10)
    assert "x" * 19 + "…" in out
    assert "y" * 9 + "…" in out


def test_reply_chars_zero_drops_replies_entirely():
    s = _slice_with(["a question"])
    s.turns[0].reply = "an answer"
    out = sh.render_markdown([s], DAY, NEXT_DAY, turn_chars=500, reply_chars=0)
    assert "an answer" not in out


def test_empty_window_renders_a_plain_statement():
    out = sh.render_markdown([], DAY, NEXT_DAY, turn_chars=500, reply_chars=100)
    assert "*No sessions in this window.*" in out
    assert "0 session(s), 0 substantive human turn(s)" in out


def test_json_render_is_valid_and_carries_provenance():
    s = _slice_with(["a question"])
    s.project = "/Users/testuser/repos/demo"
    payload = json.loads(sh.render_json([s], DAY, NEXT_DAY, turn_chars=500, reply_chars=100))
    assert payload["window"]["start"] == DAY.isoformat()
    (session,) = payload["sessions"]
    assert session["project_name"] == "demo"
    assert session["turns"][0]["text"] == "a question"
    assert session["resume"].startswith('cd "/Users/testuser/repos/demo"')


def test_collect_skips_files_untouched_since_the_window_opened(tmp_path):
    stale = _claude_file(tmp_path, [_user("old but in window", _ts(9))], name="stale")
    os.utime(stale, (0, 0))  # last written long before the window
    _claude_file(tmp_path, [_user("in window", _ts(9))], name="fresh")
    found = sh.collect(DAY, NEXT_DAY, tmp_path / "projects", tmp_path / "nope")
    assert [s.session_id for s in found] == ["fresh"]
