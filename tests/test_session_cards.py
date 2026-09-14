"""Tests for scripts/session_cards.py (loaded by file path; scripts/ is not a package)."""

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "session_cards", Path(__file__).parent.parent / "scripts" / "session_cards.py"
)
sc = importlib.util.module_from_spec(_SPEC)
sys.modules["session_cards"] = sc  # dataclasses need the module importable by name
_SPEC.loader.exec_module(sc)


def _write_jsonl(path, objs):
    path.write_text("\n".join(json.dumps(o) for o in objs), encoding="utf-8")


def _claude_user(text, ts, uuid="u1", **extra):
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": ts,
        "uuid": uuid,
        "origin": {"kind": "human"},
        "cwd": "/Users/someone/repos/demo",
        "gitBranch": "main",
        **extra,
    }


def test_parse_claude_session(tmp_path):
    f = tmp_path / "abc12345-0000-0000-0000-000000000000.jsonl"
    _write_jsonl(
        f,
        [
            {"type": "ai-title", "aiTitle": "Fix the widget"},
            _claude_user("Please fix the widget", "2026-08-28T10:00:00Z"),
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "On it"}]},
                "timestamp": "2026-08-28T10:00:05Z",
            },
            # noise turns that must not count as human
            _claude_user("<command-name>/clear</command-name>", "2026-08-28T10:01:00Z"),
            _claude_user({"tool_use_id": "t"}, "2026-08-28T10:01:30Z"),
            _claude_user("sidechain text", "2026-08-28T10:02:00Z", isSidechain=True),
            _claude_user("Thanks, now add a test", "2026-08-28T10:03:00Z"),
        ],
    )
    s = sc.parse_claude_session(f)
    assert s is not None
    assert s.title == "Fix the widget"
    assert s.human_turns == 2
    assert s.agent_turns == 1
    assert s.first_human == "Please fix the widget"
    assert s.last_human == "Thanks, now add a test"
    assert s.project == "/Users/someone/repos/demo"
    assert s.started == "2026-08-28T10:00:00Z"
    assert s.last_active == "2026-08-28T10:03:00Z"
    assert "claude --resume abc12345" in s.resume_command


def test_parse_claude_skips_humanless_session(tmp_path):
    f = tmp_path / "empty.jsonl"
    _write_jsonl(f, [{"type": "ai-title", "aiTitle": "warmup"}, {"not": "json-message"}])
    assert sc.parse_claude_session(f) is None


def test_parse_codex_session(tmp_path):
    f = tmp_path / "rollout-2026-08-28-xyz.jsonl"
    _write_jsonl(
        f,
        [
            {
                "type": "session_meta",
                "timestamp": "2026-08-28T09:00:00Z",
                "payload": {"session_id": "cdx-1", "cwd": "/Users/someone/repos/demo"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-28T09:00:10Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello codex"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-08-28T09:00:20Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            },
        ],
    )
    s = sc.parse_codex_session(f)
    assert s is not None
    assert s.session_id == "cdx-1"
    assert s.human_turns == 1
    assert s.agent_turns == 1
    assert s.first_human == "hello codex"
    assert s.resume_command == "codex resume cdx-1"


def test_card_render_and_filename():
    s = sc.Session(
        tool="claude",
        session_id="abc12345-0000",
        source_path="/tmp/x.jsonl",
        project="/Users/someone/repos/demo",
        title="Fix the widget",
        started="2026-08-28T10:00:00Z",
        last_active="2026-08-28T10:03:00Z",
        human_turns=2,
        agent_turns=1,
        first_human="Please fix the widget",
        last_human="Thanks, now add a test",
    )
    card = sc.render_card(s)
    assert "session_id: abc12345-0000" in card
    assert "## Resume" in card
    assert 'cd "/Users/someone/repos/demo" && claude --resume abc12345-0000' in card
    assert sc.card_filename(s) == "2026-08-28 - claude - Fix-the-widget - abc12345.md"
    index = sc.render_index([s])
    assert "[[2026-08-28 - claude - Fix-the-widget - abc12345]]" in index
