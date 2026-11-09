"""`openaugi init` — the guided setup.

Three questions, each with a safe default:

1. 📁 Vault — where the notes are.
2. 🔎 Search — keyword-only (built in, nothing leaves the machine), local
   embeddings, or OpenAI embeddings. Only the first needs nothing installed.
3. 🪶 How it runs — Light (the default): the CLI only. Agents call
   `openaugi kb …` when they need the notes; no server, no watcher, nothing
   in the background. Full: `openaugi up` keeps the index fresh and launches
   an agent for each `zzz:` line, and Claude Code connects over MCP.

Then it writes `~/.openaugi/config.toml` (backing up any existing one first),
saves an API key to `~/.openaugi/.env` if one was given, copies the agent files
into `<vault>/OpenAugi/AGENT/`, and prints the next commands for the choices
made. Every question also has a flag, so the whole thing can run unattended.

The pure pieces (config text, env merge, note count) are separate functions so
they can be tested without a terminal.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import typer
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

from openaugi.models import NO_EMBEDDINGS, PROVIDER_PACKAGES, provider_installed

console = Console()


# ── Choices ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SearchMode:
    key: str  # what --search takes
    label: str
    blurb: str
    provider: str
    model: str | None = None


SEARCH_MODES = (
    SearchMode(
        "none",
        "🔤 Keyword + links",
        "Built in. Nothing to install, nothing leaves this machine.",
        NO_EMBEDDINGS,
    ),
    SearchMode(
        "local",
        "💻 Local semantic",
        "Search by meaning, on-device. Needs openaugi\\[local] (PyTorch, ~2 GB).",
        "sentence-transformers",
        "all-MiniLM-L6-v2",
    ),
    SearchMode(
        "openai",
        "☁️  OpenAI semantic",
        "Best results. Sends note text to OpenAI. Needs openaugi\\[openai] + API key.",
        "openai",
        "text-embedding-3-small",
    ),
)

RUN_MODES = {
    "light": (
        "🪶 Light",
        "CLI only. Agents run `openaugi kb …` when they need your notes. "
        "No server, no watcher, nothing in the background.",
    ),
    "full": (
        "⚡ Full",
        "`openaugi up` keeps the index fresh and launches an agent for each zzz: line. "
        "Claude Code connects over MCP.",
    ),
}
# `--tasks manual|auto` shipped in 0.3.0; it still works and means the same thing.
LEGACY_TASK_MODES = {"manual": "light", "auto": "full"}

# What an agent needs to know in a Light install, pasted into its CLAUDE.md.
AGENT_SNIPPET = """\
## My notes (OpenAugi)
My notes vault is indexed by OpenAugi. Before grep/find, use its CLI:
run `openaugi kb --help` once for the tools, start research with
`openaugi kb get-context --query "..."`. Output is JSON."""


# ── Pure helpers ───────────────────────────────────────────────────────────


def build_config_toml(
    vault_path: str,
    search: SearchMode,
    auto_dispatch: bool,
    today: str | None = None,
) -> str:
    """The config.toml `init` writes. Strings go through json.dumps, which
    produces valid TOML basic strings (quotes and backslashes escaped)."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# OpenAugi config — written by `openaugi init` on {today}.",
        "# Re-run `openaugi init` any time; it saves a backup of this file first.",
        "",
        "[vault]",
        f"default_path = {json.dumps(vault_path)}",
        "",
        "[models.embedding]",
        '# "none" = keyword, title, tag and link search only.',
        '# "sentence-transformers" (local) or "openai" add search by meaning.',
        f"provider = {json.dumps(search.provider)}",
    ]
    if search.model:
        lines.append(f"model = {json.dumps(search.model)}")
    lines += [
        "",
        "[tasks]",
        "# false = Light: nothing launches agents; zzz: tasks wait for `openaugi tasks run`.",
        "# true  = Full: `openaugi up` launches a Claude Code agent for each one by itself.",
        f"auto_dispatch = {'true' if auto_dispatch else 'false'}",
        "",
    ]
    return "\n".join(lines)


def merge_env(existing: str, updates: dict[str, str]) -> str:
    """`.env` text with `updates` set, every other line kept as it was."""
    out: list[str] = []
    seen: set[str] = set()
    for line in existing.splitlines():
        key = line.partition("=")[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    return "\n".join(out) + "\n"


def count_notes(vault: Path, limit: int = 100_000) -> int:
    """Markdown files in the vault, skipping dot-folders (.obsidian, .trash…)."""
    n = 0
    for _root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        n += sum(f.endswith(".md") for f in files)
        if n >= limit:
            break
    return n


def backup_path(config_path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d-%H%M%S")
    return config_path.with_name(f"{config_path.name}.bak-{stamp}")


def tilde(path: Path | str) -> str:
    """`path` with the home directory written as ~, for display."""
    p, home = str(path), str(Path.home())
    return "~" + p[len(home) :] if p == home or p.startswith(home + os.sep) else p


def openaugi_executable() -> str:
    """Absolute path of the running `openaugi`, for `claude mcp add`."""
    argv0 = Path(sys.argv[0])
    if argv0.name == "openaugi" and argv0.exists():
        return str(argv0.resolve())
    return shutil.which("openaugi") or "openaugi"


# ── Interactive steps ──────────────────────────────────────────────────────


def _step(n: int, icon: str, title: str) -> None:
    console.print(f"\n[bold]{icon} Step {n} of 3 · {title}[/bold]")


def _options_table(rows: list[tuple[str, str, str, str]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold cyan")
    table.add_column(no_wrap=True, style="bold")
    table.add_column(style="dim")
    table.add_column(no_wrap=True)
    for row in rows:
        table.add_row(*row)
    return table


def _choose(n_options: int, default: int) -> int:
    choices = [str(i) for i in range(1, n_options + 1)]
    answer = typer.prompt(
        "  Choose", default=str(default), type=click.Choice(choices), show_choices=False
    )
    return int(answer)


def ask_vault(default: str, given: str | None) -> Path:
    _step(1, "📁", "Your vault")
    while True:
        raw = given or typer.prompt("  Where is your Obsidian vault?", default=tilde(default))
        vault = Path(raw.strip().strip("'\"")).expanduser()
        if vault.is_dir():
            notes = count_notes(vault)
            kind = "Obsidian vault" if (vault / ".obsidian").is_dir() else "Folder"
            console.print(f"  ✅ {kind} · [bold]{notes:,}[/bold] notes")
            return vault.resolve()
        console.print(f"  [yellow]⚠️  {tilde(vault)} doesn't exist — try again.[/yellow]")
        if given:
            raise typer.Exit(1)


def ask_search(current: str | None, given: str | None) -> SearchMode:
    _step(2, "🔎", "Search")
    if given:
        mode = next(m for m in SEARCH_MODES if m.key == given)
        console.print(f"  {mode.label}")
    else:
        rows = []
        for i, m in enumerate(SEARCH_MODES, 1):
            if m.provider == NO_EMBEDDINGS:
                status = "[green]✓ ready[/green]"
            elif provider_installed(m.provider):
                status = "[green]✓ installed[/green]"
            else:
                status = "[dim]✗ not installed[/dim]"
            rows.append((str(i), m.label, m.blurb, status))
        console.print(Padding(_options_table(rows), (0, 0, 0, 2)))
        default = next((i for i, m in enumerate(SEARCH_MODES, 1) if m.provider == current), 1)
        mode = SEARCH_MODES[_choose(len(SEARCH_MODES), default) - 1]

    if mode.provider != NO_EMBEDDINGS and not provider_installed(mode.provider):
        extra = PROVIDER_PACKAGES[mode.provider][1]
        console.print(
            f"  [yellow]⚠️  Not installed yet.[/yellow] Before indexing, run:\n"
            f'     [cyan]pip install "openaugi\\[{extra}]"[/cyan]   '
            f"[dim](from a checkout: uv sync --extra {extra})[/dim]"
        )
    return mode


def ask_openai_key(interactive: bool) -> str | None:
    """The key to save, or None. An env var already set is offered, not re-asked."""
    existing = os.environ.get("OPENAI_API_KEY", "")
    if existing:
        masked = f"{existing[:7]}…"
        console.print(f"  🔑 OPENAI_API_KEY found in your environment [dim]({masked})[/dim]")
        if not interactive or typer.confirm("  Save it to ~/.openaugi/.env?", default=True):
            return existing
        return None
    if not interactive:
        console.print("  [yellow]⚠️  No OPENAI_API_KEY set.[/yellow] Add it to ~/.openaugi/.env.")
        return None
    key = typer.prompt(
        "  🔑 OpenAI API key (Enter to skip)", default="", show_default=False, hide_input=True
    )
    return key.strip() or None


def _missing_agent_tools() -> list[str]:
    """Which of tmux / claude the task launcher can't find (same lookup it uses)."""
    from openaugi.agents.task_watcher import detect_claude, detect_tmux

    missing = []
    for name, detect in (("tmux", detect_tmux), ("claude", detect_claude)):
        try:
            detect()
        except FileNotFoundError:
            missing.append(name)
    return missing


def ask_mode(current: bool | None, given: str | None) -> bool:
    """Light or Full. Returns True for Full (which is `[tasks] auto_dispatch`)."""
    _step(3, "🪶", "How it runs")
    console.print(
        "  Every install has every tool. The choice is whether anything runs in the background."
    )
    given = LEGACY_TASK_MODES.get(given or "", given)
    if given:
        label, _ = RUN_MODES[given]
        console.print(f"  {label}")
        auto = given == "full"
    else:
        modes = enumerate(RUN_MODES.values(), 1)
        rows = [(str(i), label, blurb, "") for i, (label, blurb) in modes]
        console.print(Padding(_options_table(rows), (0, 0, 0, 2)))
        auto = _choose(2, 2 if current else 1) == 2

    if auto:
        missing = _missing_agent_tools()
        if missing:
            console.print(
                f"  [yellow]⚠️  Not found on PATH: {', '.join(missing)}.[/yellow] "
                "Agents can't launch until it is installed."
            )
    return auto


# ── Writing ────────────────────────────────────────────────────────────────


def write_config(config_path: Path, text: str, assume_yes: bool) -> tuple[bool, Path | None]:
    """Write config.toml, backing up an existing one. Returns (written, backup)."""
    backup = None
    if config_path.exists():
        if config_path.read_text() == text:
            return True, None
        console.print(f"\n  [yellow]⚠️  {tilde(config_path)} already exists.[/yellow]")
        console.print(
            "  [dim]Writing replaces it (other settings in it are dropped); "
            "a dated backup is kept next to it.[/dim]"
        )
        if not assume_yes and not typer.confirm("  Replace it?", default=False):
            return False, None
        backup = backup_path(config_path)
        shutil.copy2(config_path, backup)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text)
    return True, backup


def write_env(env_path: Path, updates: dict[str, str]) -> None:
    existing = env_path.read_text() if env_path.exists() else ""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(merge_env(existing, updates))
    env_path.chmod(0o600)  # owner-only read/write


def copy_agent_files(vault: Path) -> tuple[int, int, Path]:
    """Copy engine templates into <vault>/OpenAugi/AGENT/. Returns (copied, kept, dir).

    Existing files are never overwritten — they may hold the user's edits.
    """
    from openaugi.agent_files import engine_templates

    agent_dir = vault / "OpenAugi" / "AGENT"
    copied = kept = 0
    for filename, text in engine_templates():
        dest = agent_dir / filename
        if dest.exists():
            kept += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        copied += 1
    return copied, kept, agent_dir


# ── Summary ────────────────────────────────────────────────────────────────


def next_steps(search: SearchMode, auto: bool) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    if search.provider != NO_EMBEDDINGS and not provider_installed(search.provider):
        extra = PROVIDER_PACKAGES[search.provider][1]
        steps.append((f'pip install "openaugi[{extra}]"', "install the search extra you picked"))
    if auto:
        steps += [
            ("openaugi up", "index the vault, keep it fresh, launch zzz: tasks (leave running)"),
            ("openaugi tasks", "see what launched, and anything still waiting"),
        ]
    else:
        steps += [
            ("openaugi ingest", "index your vault — run it again whenever you want it fresh"),
            ('openaugi kb get-context --query "budget"', "what an agent sees: JSON, then exit"),
        ]
    return steps


def print_summary(
    vault: Path,
    search: SearchMode,
    auto: bool,
    config_path: Path,
    config_written: bool,
    backup: Path | None,
    copied: int,
    kept: int,
    agent_dir: Path,
    key_saved: Path | None,
) -> None:
    console.print("\n[bold]💾 Saving[/bold]")
    if config_written:
        note = f"  [dim](previous one saved as {backup.name})[/dim]" if backup else ""
        console.print(f"  ✅ Config        [cyan]{tilde(config_path)}[/cyan]{note}")
    else:
        console.print(f"  ⏭️  Config        kept your existing [cyan]{tilde(config_path)}[/cyan]")
    if key_saved:
        where = tilde(key_saved)
        console.print(f"  ✅ API key       [cyan]{where}[/cyan] [dim](owner-only)[/dim]")
    kept_note = f" [dim]({kept} already there, left as is)[/dim]" if kept else ""
    where = tilde(agent_dir)
    console.print(f"  ✅ Agent files   {copied} copied to [cyan]{where}[/cyan]{kept_note}")

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("Vault", tilde(vault))
    summary.add_row("Search", search.label)
    summary.add_row("Runs", RUN_MODES["full" if auto else "light"][0])
    console.print()
    console.print(Panel(summary, title="🎉 You're set up", title_align="left", expand=False))

    console.print("\n[bold]Next[/bold]")
    steps = next_steps(search, auto)
    width = max(len(cmd) for cmd, _ in steps)
    for i, (cmd, why) in enumerate(steps, 1):
        console.print(f"  {i}. [cyan]{escape(cmd.ljust(width))}[/cyan]  [dim]{why}[/dim]")

    if auto:
        console.print("\n[bold]🔌 Connect Claude Code[/bold]")
        console.print(
            f"  [cyan]claude mcp add --scope user openaugi -- {openaugi_executable()} serve[/cyan]"
        )
        console.print(
            "  [dim]Runs only while Claude Code is open. Agent writes stay inside "
            "<vault>/OpenAugi/.[/dim]\n"
        )
        return
    console.print(
        "\n[bold]🤝 Tell your agent[/bold] [dim]— paste into ~/.claude/CLAUDE.md "
        "(or your agent's instructions file)[/dim]\n"
    )
    console.print(Padding(AGENT_SNIPPET, (0, 0, 0, 2)), markup=False, highlight=False)
    console.print(
        "\n  [dim]Agent writes stay inside <vault>/OpenAugi/. "
        "Want a server or background service later? Re-run init and pick ⚡ Full.[/dim]\n"
    )


# ── Entry point ────────────────────────────────────────────────────────────


def run_setup(
    vault: str | None = None,
    search: str | None = None,
    mode: str | None = None,
    yes: bool = False,
) -> None:
    """The whole `openaugi init` flow. Flags answer their question; `yes`
    accepts the default for any question left and replaces an existing config."""
    from openaugi.config import _load_toml

    config_dir = Path.home() / ".openaugi"
    config_path = config_dir / "config.toml"
    env_path = config_dir / ".env"

    current: dict[str, Any] = {}
    if config_path.exists():
        try:
            current = _load_toml(config_path)
        except Exception:
            current = {}
    current_vault = current.get("vault", {}).get("default_path")
    current_provider = current.get("models", {}).get("embedding", {}).get("provider")
    current_auto = current.get("tasks", {}).get("auto_dispatch")

    if yes:
        vault = vault or current_vault
        search = search or next(
            (m.key for m in SEARCH_MODES if m.provider == current_provider), "none"
        )
        mode = mode or ("full" if current_auto else "light")
        if not vault:
            console.print("[red]--yes needs --vault the first time.[/red]")
            raise typer.Exit(1)

    console.print()
    console.print(
        Panel(
            "Your notes, searchable by your AI agents.\n"
            "[dim]Three questions. Everything is stored on this machine.[/dim]",
            title="🧠 OpenAugi setup",
            title_align="left",
            expand=False,
        )
    )

    vault_path = ask_vault(current_vault or str(Path.home() / "Documents" / "vault"), vault)
    search_mode = ask_search(current_provider, search)
    key = ask_openai_key(interactive=not yes) if search_mode.provider == "openai" else None
    auto = ask_mode(current_auto, mode)

    text = build_config_toml(str(vault_path), search_mode, auto)
    written, backup = write_config(config_path, text, assume_yes=yes)
    key_saved = None
    if key:
        write_env(env_path, {"OPENAI_API_KEY": key})
        key_saved = env_path
    copied, kept, agent_dir = copy_agent_files(vault_path)

    print_summary(
        vault_path,
        search_mode,
        auto,
        config_path,
        written,
        backup,
        copied,
        kept,
        agent_dir,
        key_saved,
    )
    if not written:
        console.print("  [dim]To apply these choices by hand, put this in your config:[/dim]\n")
        console.print(text, markup=False, highlight=False)
