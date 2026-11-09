"""`openaugi kb`: every MCP tool as a CLI command, with the same answers.

The CLI-only install has no server, so this surface is the whole agent API.
Parity is pinned two ways: the command list is the MCP registry (a new tool
shows up here or this file fails), and a call through the CLI returns the
same JSON as the tool function.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import openaugi.mcp.server as srv
from openaugi.cli.kb import commands
from openaugi.cli.main import app
from openaugi.pipeline.runner import run_layer0
from openaugi.store.sqlite import SQLiteStore

runner = CliRunner()


@pytest.fixture
def db(tmp_path: Path, vault_path: Path) -> Path:
    path = tmp_path / "kb.db"
    store = SQLiteStore(path)
    run_layer0(vault_path, store)
    store.close()
    return path


@pytest.fixture(autouse=True)
def _point_at_test_db(db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_vault = tmp_path / "vault"
    out_vault.mkdir()
    monkeypatch.setenv("OPENAUGI_DB", str(db))
    monkeypatch.setenv("OPENAUGI_VAULT_PATH", str(out_vault))
    srv._store = None
    srv._embedding_model = None


def kb(*args: str, input: str | None = None):
    return runner.invoke(app, ["kb", *args], input=input)


def ok(*args: str, input: str | None = None) -> dict:
    result = kb(*args, input=input)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


# ── parity with the MCP registry ───────────────────────────────────────────


def test_every_mcp_tool_and_resource_is_a_command():
    tools = {t.name.replace("_", "-") for t in asyncio.run(srv.mcp.list_tools())}
    templates = asyncio.run(srv.mcp.list_resource_templates())
    assert tools | {"note"} == set(commands())
    assert [t.name for t in templates] == ["get_note_resource"]


def test_every_tool_argument_is_an_option():
    for tool in asyncio.run(srv.mcp.list_tools()):
        cmd = commands()[tool.name.replace("_", "-")]
        assert {p.name for p in cmd.params} == set(tool.inputSchema.get("properties", {})), (
            tool.name
        )


def test_cli_answer_is_the_tool_answer():
    via_cli = kb("search", "--keyword", "career", "--k", "5")
    assert via_cli.exit_code == 0, via_cli.output
    assert via_cli.output.strip() == srv.search(keyword="career", k=5).strip()


def test_guide_names_only_real_commands_and_options():
    """`openaugi kb --help` is what an agent reads first; it must not lie."""
    from openaugi.cli.kb import GUIDE

    cmds = commands()
    named = 0
    for line in GUIDE.splitlines():
        for m in re.finditer(r"openaugi kb ([a-z-]+)([^,;(]*)", line):
            cmd = cmds[m.group(1)]
            opts = {o for p in cmd.params for o in (*p.opts, *p.secondary_opts)}
            for opt in re.findall(r"(?<!\S)--[a-z-]+", m.group(2)):
                assert opt in opts, f"{m.group(1)} has no {opt}"
            for other in re.findall(r"/ ([a-z-]+)", m.group(2)):
                assert other in cmds, other
            named += 1
    assert named >= 10
    search_opts = {o for p in cmds["search"].params for o in p.opts}
    for opt in ("--provenance", "--include-path-prefix", "--exclude-path-prefix", "--offset"):
        assert opt in search_opts


def test_underscored_names_work():
    assert ok("get_review_state") == ok("get-review-state")


# ── reads ──────────────────────────────────────────────────────────────────


def _first_id() -> str:
    return ok("search", "--keyword", "career", "--k", "1")["results"][0]["id"]


def test_graph_reads_chain_from_search():
    block_id = _first_id()
    assert ok("get-block", "--block-id", block_id)["id"] == block_id
    assert ok("get-related", "--block-id", block_id)["block_id"] == block_id
    assert ok("traverse", "--start-id", block_id, "--max-hops", "1")["start_id"] == block_id


def test_list_option_repeats():
    block_id = _first_id()
    out = ok("get-blocks", "--block-ids", block_id, "--block-ids", "missing-id")
    assert [b["id"] for b in out["blocks"]] == [block_id]
    assert out["missing"] == ["missing-id"]


def test_boolean_flag_pair():
    expanded = ok("get-context", "--query", "career direction")
    flat = ok("get-context", "--query", "career direction", "--no-expand")
    assert expanded["direct_results"]
    assert flat["expanded"] == []


def test_resource_template_as_command():
    result = kb("note", "--title", "Project Alpha")
    assert result.exit_code == 0, result.output
    assert "Project Alpha" in result.output


# ── writes ─────────────────────────────────────────────────────────────────


def test_records_round_trip_with_json_and_stdin():
    ok("write-record", "--collection", "c", "--record-id", "r1", "--data", '{"state": "open"}')
    ok("update-record", "--collection", "c", "--record-id", "r1", "--patch", "-",
       input='{"state": "done"}')  # fmt: skip
    listed = ok("list-records", "--collection", "c", "--where", '{"state": "done"}')
    assert [r["id"] for r in listed["records"]] == ["r1"]


def test_write_document_content_from_stdin(tmp_path: Path):
    out = ok("write-document", "--title", "Memo", "--description", "a memo", "--content", "-",
             input="# Memo\n\nLong body.\n")  # fmt: skip
    assert out["status"] == "created"
    written = tmp_path / "vault" / "OpenAugi" / "Notes" / "Memo.md"
    assert "Long body." in written.read_text()


def test_vault_option_overrides_env(tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    result = runner.invoke(
        app,
        ["kb", "--vault", str(other), "write-document",
         "--title", "Here", "--description", "d", "--content", "x"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (other / "OpenAugi" / "Notes" / "Here.md").exists()


# ── errors ─────────────────────────────────────────────────────────────────


def test_error_envelope_exits_1():
    result = kb("get-block", "--block-id", "nope")
    assert result.exit_code == 1
    assert "Block not found" in json.loads(result.output)["error"]


def test_write_error_status_exits_1(tmp_path: Path):
    args = ("write-document", "--title", "Twice", "--description", "d", "--content", "x")
    ok(*args)
    result = kb(*args)
    assert result.exit_code == 1
    assert json.loads(result.output)["status"] == "error"


def test_bad_json_is_a_usage_error():
    result = kb("write-record", "--collection", "c", "--record-id", "r", "--data", "{nope")
    assert result.exit_code == 2
    assert "not valid JSON" in result.output


def test_missing_required_option_is_a_usage_error():
    result = kb("get-block")
    assert result.exit_code == 2
    assert "--block-id" in result.output


def test_two_stdin_arguments_are_refused():
    result = kb("write-recap", "--container-title", "-", "--recap-md", "-", input="x")
    assert result.exit_code == 2
    assert "only one argument" in result.output


def test_top_level_help_does_not_import_the_server():
    """`openaugi --help` must not pay for the MCP import; `kb` loads it on demand."""
    import subprocess
    import sys

    code = (
        "import sys; from openaugi.cli.main import app; "
        "print('openaugi.mcp.server' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
