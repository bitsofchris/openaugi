"""`openaugi kb` — every MCP tool as a CLI command: same arguments, same JSON.

The CLI-only install (no `serve`, no `up`) gives an agent the whole knowledge
base through short-lived shell calls. Nothing here re-implements a tool: the
commands are generated from the MCP server's own tool registry and each call
goes through `FastMCP.call_tool`, so argument validation, defaults and output
are exactly what an MCP client gets. A tool added to `mcp/server.py` appears
here with no CLI change; `tests/test_kb_cli.py` pins that.

Argument mapping (from each tool's JSON schema):

    string / integer / number  → --name VALUE
    boolean                    → --name / --no-name
    array of strings           → --name A --name B   (repeat the option)
    object / array of objects  → --name '<json>'
    any string or JSON value   → --name -            (read it from stdin)

Underscored MCP names work too: `openaugi kb get_context` = `kb get-context`.
Exit status is 1 when the tool answers with an error envelope, 2 on a usage
error. stdout carries only the tool's JSON; logs stay on stderr at WARNING.

Commands are built with Typer's public API (`typer.main.get_command`), never
with `click` directly: recent Typer vendors its own click, and a command built
on the other one would turn every usage error into a traceback.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import re
import sys
from collections.abc import Callable
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

STDIN = "-"


def _server():
    """Import the MCP server module on first use — `openaugi --help` stays fast."""
    from openaugi.mcp import server

    return server


def _cli_name(name: str) -> str:
    return name.replace("_", "-")


def _unwrap(prop: dict) -> dict:
    """`anyOf: [X, null]` (an Optional) → X. Anything else passes through."""
    variants = [v for v in prop.get("anyOf", []) if v.get("type") != "null"]
    if len(variants) == 1:
        merged = {k: v for k, v in prop.items() if k != "anyOf"}
        return {**variants[0], **merged}
    return prop


class _Param:
    """One tool argument: its Typer option and how its value converts back."""

    def __init__(self, name: str, prop: dict, required: bool):
        self.name = name
        self.schema = _unwrap(prop)
        self.required = required
        self.default = prop.get("default")
        self.flag = "--" + _cli_name(name)

    @property
    def kind(self) -> str:
        t = self.schema.get("type")
        if t == "object" or (
            t == "array" and self.schema.get("items", {}).get("type") == "object"
        ):
            return "json"
        if t == "array":
            return "list"
        return t if t in ("boolean", "integer", "number") else "string"

    def parameter(self) -> inspect.Parameter:
        kind = self.kind
        annotation: Any = {
            "boolean": bool,
            "integer": int,
            "number": float,
            "list": list[str],
        }.get(kind, str)
        negative = f"--no-{_cli_name(self.name)}"
        decls = [f"{self.flag}/{negative}"] if kind == "boolean" else [self.flag]
        help_text = {
            "json": "JSON value, or - to read it from stdin.",
            "list": "Repeat the option for each value.",
        }.get(kind)
        default: Any = inspect.Parameter.empty  # required: Typer enforces it
        if not self.required:
            default = self.default
            if default is None or kind == "list":
                annotation = annotation | None
                default = None
        # Annotated form: the option holds only declarations; the default
        # lives on the parameter.
        option = typer.Option(*decls, help=help_text, metavar="JSON" if kind == "json" else None)
        return inspect.Parameter(
            self.name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Annotated[annotation, option],
            default=default,
        )

    def convert(self, value: Any, read_stdin: Callable[[str], str]) -> Any:
        if value == STDIN and self.kind in ("string", "json"):
            value = read_stdin(self.flag)
        if self.kind == "list":
            return list(value)
        if self.kind == "json" and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(f"not valid JSON ({e.msg})", param_hint=self.flag) from e
        return value


def _stdin_reader() -> Callable[[str], str]:
    """stdin can be read once per call; a second `-` is a usage error."""
    used = False

    def read(flag: str) -> str:
        nonlocal used
        if used:
            raise typer.BadParameter(
                "only one argument per call can be read from stdin (-)", param_hint=flag
            )
        used = True
        return sys.stdin.read()

    return read


def _emit(payload: Any) -> None:
    """Print the tool's answer as-is; exit 1 when it is an error envelope."""
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    typer.echo(text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return
    # Two error shapes exist: read tools answer {"error": ...}, write tools
    # {"status": "error", "reason": ...}.
    if isinstance(parsed, dict) and ("error" in parsed or parsed.get("status") == "error"):
        raise typer.Exit(1)


def _call_tool(name: str, arguments: dict) -> Any:
    from mcp.server.fastmcp.exceptions import ToolError

    try:
        result = asyncio.run(_server().mcp.call_tool(name, arguments))
    except ToolError as e:
        return {"error": str(e)}
    # Tools that return str come back as (content, {"result": str}).
    if isinstance(result, tuple):
        structured = result[1]
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        result = result[0]
    if isinstance(result, dict):
        return result
    return "".join(getattr(block, "text", "") for block in result)


def _make_command(
    name: str, description: str | None, params: list[_Param], run: Callable[[dict], Any]
):
    by_name = {p.name: p for p in params}

    def callback(**values):
        read_stdin = _stdin_reader()
        arguments = {}
        for key, value in values.items():
            p = by_name[key]
            if value is None or (p.kind == "list" and not value):
                continue
            arguments[key] = p.convert(value, read_stdin)
        _emit(run(arguments))

    callback.__signature__ = inspect.Signature([p.parameter() for p in params])  # type: ignore[attr-defined]
    callback.__doc__ = description

    # Plain help: tool docstrings hold literal [brackets] that rich markup would eat.
    single = typer.Typer(rich_markup_mode=None, add_completion=False)
    first_line = (description or "").strip().split("\n", 1)[0]
    single.command(name=_cli_name(name), short_help=first_line)(callback)
    command = typer.main.get_command(single)
    command.name = _cli_name(name)
    return command


def _tool_command(tool):
    schema = tool.inputSchema or {}
    required = set(schema.get("required", []))
    params = [_Param(n, prop, n in required) for n, prop in schema.get("properties", {}).items()]
    return _make_command(
        tool.name, tool.description, params, lambda args: _call_tool(tool.name, args)
    )


def _template_command(template):
    """A resource template (vault://note/{title}) as a command named after its subject."""
    fields = re.findall(r"{(\w+)}", template.uriTemplate)
    params = [_Param(f, {"type": "string"}, True) for f in fields]
    fn = getattr(_server(), template.name)
    name = template.name.removeprefix("get_").removesuffix("_resource")
    return _make_command(name, template.description, params, lambda args: fn(**args))


@functools.cache
def commands() -> dict[str, Any]:
    """Every MCP tool and resource template, keyed by CLI name."""
    mcp = _server().mcp
    built = {_cli_name(t.name): _tool_command(t) for t in asyncio.run(mcp.list_tools())}
    for template in asyncio.run(mcp.list_resource_templates()):
        cmd = _template_command(template)
        built[cmd.name or template.name] = cmd
    return built


class KbGroup(TyperGroup):
    """Commands resolved from the MCP registry on demand."""

    def list_commands(self, ctx) -> list[str]:
        return sorted({*super().list_commands(ctx), *commands()})

    def get_command(self, ctx, cmd_name: str):
        return super().get_command(ctx, cmd_name) or commands().get(_cli_name(cmd_name))


# What an agent reads first (`openaugi kb --help`): the MCP server's
# INSTRUCTIONS in command form. Every `openaugi kb <cmd> --opt` named here is
# checked against the generated commands by tests/test_kb_cli.py.
GUIDE = """\
The user's notes vault, indexed as blocks (note sections) with full-text \
search, a link graph and, if configured, embeddings. Every MCP tool of \
`openaugi serve` is a command here with the same arguments and the same JSON \
answer, and no server: each call opens the database, answers, and exits. For \
any question about what exists, related or earlier ideas, or what the user \
wrote in a date window, use these before grep/find.

\b
How to search:
- Research, the default: openaugi kb get-context --query "..."
  (search + dedup + link expansion in one call)
- Exact terms: openaugi kb search --keyword "..."
- Note titles: openaugi kb search --title "..."
- By meaning: openaugi kb search --query "..."
- By date: openaugi kb search --after 2022-03-01 --before 2022-03-31
  (paginate with --offset); newest blocks: openaugi kb recent
- By links: openaugi kb get-related --block-id ID (one hop),
  openaugi kb traverse --start-id ID (multi-hop)
- Only what the user wrote: add --provenance human
- Scope by folder: --include-path-prefix / --exclude-path-prefix

\b
Results are summaries. Read full content:
  openaugi kb get-block --block-id ID
  openaugi kb get-blocks --block-ids ID1 --block-ids ID2

\b
Writing (lands only under the vault's OpenAugi/ folder):
  openaugi kb write-document --title T --description D --content - < note.md
  openaugi kb write-record / list-records / update-record keep workflow state

\b
Arguments: lists repeat the option; JSON values are strings ('{"k": "v"}');
- reads one value from stdin. Exit 0 = answer, 1 = the tool returned an
error envelope, 2 = bad arguments. `openaugi kb <command> --help` shows a
command's arguments and full description; underscores work too (get_context).
"""

kb_app = typer.Typer(
    cls=KbGroup,
    help=GUIDE,
    short_help="Every MCP tool as a CLI command for agents — JSON out, no server.",
    no_args_is_help=True,
    rich_markup_mode=None,
)


@kb_app.callback()
def kb_main(
    db: str | None = typer.Option(None, "--db", help="Database path (default ~/.openaugi)"),
    vault: str | None = typer.Option(None, "--vault", help="Vault path (default: config)"),
):
    # FastMCP logs at INFO on import; an agent reading this output wants the
    # JSON and real problems only.
    for name in ("mcp", "httpx", "httpcore", "openai", "openaugi"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if db:
        os.environ["OPENAUGI_DB"] = db
    if vault:
        os.environ["OPENAUGI_VAULT_PATH"] = vault
