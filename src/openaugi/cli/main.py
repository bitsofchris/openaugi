"""OpenAugi CLI — ingest, serve, search, hubs, status.

Entry point for the `openaugi` command.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from openaugi._version import __version__

app = typer.Typer(
    name="openaugi",
    help="Self-hostable personal intelligence engine.",
    invoke_without_command=True,
)
console = Console()


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )


def _default_db() -> Path:
    return Path.home() / ".openaugi" / "openaugi.db"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
):
    if version:
        console.print(f"openaugi {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def init():
    """Set up OpenAugi — choose embedding model, configure API keys, set vault path."""
    config_dir = Path.home() / ".openaugi"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    env_path = config_dir / ".env"

    console.print("\n[bold]OpenAugi Setup[/bold]\n")

    # 1. Embedding model
    console.print("[bold]Embedding model[/bold]")
    console.print("  1. OpenAI (text-embedding-3-small) — best quality, requires API key")
    console.print("  2. Local (sentence-transformers) — free, no API key, runs on CPU")
    console.print("  3. None — skip embeddings, keyword search only")

    choice = typer.prompt("Choose", default="1")

    embedding_provider = None
    embedding_model = None
    needs_openai_key = False

    if choice == "1":
        embedding_provider = "openai"
        embedding_model = "text-embedding-3-small"
        needs_openai_key = True
    elif choice == "2":
        embedding_provider = "sentence-transformers"
        embedding_model = "all-MiniLM-L6-v2"
    # else: no embeddings

    # 2. API key (if needed)
    env_lines: list[str] = []
    if needs_openai_key:
        # Check if already set
        import os

        existing = os.environ.get("OPENAI_API_KEY", "")
        if existing:
            masked = f"{existing[:8]}..."
            console.print(f"  OPENAI_API_KEY already set in environment [dim]({masked})[/dim]")
            save_key = typer.confirm("Save to ~/.openaugi/.env for persistence?", default=True)
            if save_key:
                env_lines.append(f"OPENAI_API_KEY={existing}")
        else:
            key = typer.prompt("  Enter your OpenAI API key", hide_input=True)
            if key:
                env_lines.append(f"OPENAI_API_KEY={key}")
                console.print("  [green]Key saved to ~/.openaugi/.env[/green]")

    # 3. Default vault path
    vault_path = typer.prompt(
        "\nDefault vault path (Obsidian vault)",
        default=str(Path.home() / "Documents" / "vault"),
    )
    vault_path = vault_path.strip().strip("'\"")

    # Write config.toml
    toml_lines = []
    if embedding_provider:
        toml_lines.extend(
            [
                "[models.embedding]",
                f'provider = "{embedding_provider}"',
                f'model = "{embedding_model}"',
                "",
            ]
        )
    toml_lines.extend(
        [
            "[vault]",
            f'default_path = "{vault_path}"',
        ]
    )

    config_path.write_text("\n".join(toml_lines) + "\n")
    console.print(f"\n  Config written to [cyan]{config_path}[/cyan]")

    # Write .env (append, don't overwrite existing vars)
    if env_lines:
        existing_env = env_path.read_text() if env_path.exists() else ""
        new_vars = {}
        for line in env_lines:
            key, _, value = line.partition("=")
            new_vars[key] = value

        # Parse existing
        existing_vars: dict[str, str] = {}
        for line in existing_env.splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                existing_vars[key.strip()] = value.strip()

        # Merge (new overrides existing)
        existing_vars.update(new_vars)
        env_content = "\n".join(f"{k}={v}" for k, v in existing_vars.items()) + "\n"
        env_path.write_text(env_content)
        env_path.chmod(0o600)  # owner-only read/write
        console.print(f"  Keys written to [cyan]{env_path}[/cyan] (chmod 600)")

    console.print("\n[bold green]Setup complete![/bold green]\n")
    console.print("Next steps:")
    console.print("  openaugi ingest          # uses default vault path from config")
    console.print("  openaugi serve           # start MCP server for Claude")


@app.command()
def ingest(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run Layer 0 + Layer 1 pipeline: ingest vault → embed → store."""
    _setup_logging(verbose)

    from openaugi.config import load_config
    from openaugi.pipeline.runner import run_layer0
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()

    # Resolve vault path: CLI arg > config > error
    vault_path = path or config.get("vault", {}).get("default_path")
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())
    store = SQLiteStore(db_path)

    try:
        exclude = config.get("vault", {}).get("exclude_patterns")
        workers = config.get("vault", {}).get("max_workers", 4)

        console.print(f"[bold]Ingesting vault:[/bold] {vault_path}")
        console.print(f"[bold]Database:[/bold] {db_path}")

        result = run_layer0(vault_path, store, exclude_patterns=exclude, max_workers=workers)

        stats = result["stats"]
        console.print(
            f"\n[green]Done.[/green] {stats['total_blocks']} blocks, {stats['total_links']} links"
        )

        # Layer 1: embedding (optional — requires sentence-transformers or openai)
        try:
            from openaugi.models import get_embedding_model
            from openaugi.pipeline.embed import run_embed

            model = get_embedding_model(config.get("models", {}).get("embedding"))
            count = run_embed(store, model)
            if count:
                console.print(f"[green]Embedded {count} blocks[/green]")
            else:
                console.print("[dim]All blocks already embedded[/dim]")
        except ImportError as e:
            console.print(
                f"[yellow]Skipping embeddings:[/yellow] {e}\n"
                "Install with: pip install openaugi[local]"
            )
    finally:
        store.close()


@app.command()
def serve(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="Transport: stdio or streamable-http"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP host (streamable-http only)"),
    port: int = typer.Option(8787, "--port", "-p", help="HTTP port (streamable-http only)"),
):
    """Start MCP server.

    Default: stdio transport for Claude Desktop/Code.
    Use --transport streamable-http for remote access (Claude mobile, Tailscale, etc.).
    """
    import os

    if db:
        os.environ["OPENAUGI_DB"] = db

    from openaugi.mcp.server import run_server

    resolved = "streamable-http" if transport == "http" else transport
    run_server(transport=resolved, host=host, port=port)  # type: ignore[arg-type]


@app.command()
def watch(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    debounce: float = typer.Option(
        30.0, "--debounce", "-d", help="Seconds to wait after last change"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Watch vault for changes and run incremental ingest.

    Runs as a long-lived process alongside 'openaugi serve'.
    Debounces rapid saves before triggering Layer 0 + optional Layer 1 (embedding).
    """
    _setup_logging(verbose)

    from openaugi.config import load_config
    from openaugi.pipeline.watcher import watch_vault

    config = load_config()

    vault_path = path or config.get("vault", {}).get("default_path")
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())

    console.print(f"[bold]Watching vault:[/bold] {vault_path}")
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Debounce:[/bold] {debounce}s")
    console.print()

    watch_vault(vault_path, db_path, config, debounce_seconds=debounce)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(10, "--k", "-k", help="Number of results"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    keyword: bool = typer.Option(False, "--keyword", help="Use FTS instead of semantic"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Search the knowledge base from terminal."""
    _setup_logging(verbose)

    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())
    store = SQLiteStore(db_path, read_only=True)

    try:
        if keyword:
            results = store.search_fts(query, limit=k)
            for b in results:
                _print_block(b)
        else:
            # Semantic search via sqlite-vec
            from openaugi.config import load_config
            from openaugi.models import get_embedding_model

            config = load_config()
            model = get_embedding_model(config.get("models", {}).get("embedding"))
            query_vec = model.embed_query(query)
            hits = store.semantic_search(query_vec, k=k)

            if not hits:
                console.print(
                    "[yellow]No semantic search results. Run 'openaugi ingest' first.[/yellow]"
                )
            for block_id, distance in hits:
                block = store.get_block(block_id)
                if block:
                    _print_block(block, score=round(1.0 - distance, 4))
    finally:
        store.close()


@app.command()
def hubs(
    k: int = typer.Option(20, "--k", "-k", help="Number of hubs"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Show top hubs by link count."""
    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())
    store = SQLiteStore(db_path, read_only=True)

    try:
        scores = store.get_hub_scores(limit=k)
        if not scores:
            console.print("[dim]No hubs found. Run 'openaugi ingest' first.[/dim]")
            return

        table = Table(title="Top Hubs")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Title", style="bold")
        table.add_column("In", width=4)
        table.add_column("Out", width=4)
        table.add_column("Entries", width=8)

        for hub in scores:
            table.add_row(
                f"{hub['hub_score']:.2f}",
                hub["title"] or "(untitled)",
                str(hub["in_links"]),
                str(hub["out_links"]),
                str(hub["entry_count"]),
            )
        console.print(table)
    finally:
        store.close()


@app.command()
def status(
    db: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Show store stats: block counts, embedding coverage, link counts."""
    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())

    if not Path(db_path).exists():
        console.print(f"[red]Database not found:[/red] {db_path}")
        console.print("Run 'openaugi ingest --path /your/vault' first.")
        raise typer.Exit(1)

    store = SQLiteStore(db_path, read_only=True)
    try:
        stats = store.get_stats()

        console.print(f"\n[bold]OpenAugi Status[/bold]  ({db_path})\n")
        console.print(f"Total blocks: [cyan]{stats['total_blocks']}[/cyan]")
        console.print(f"Total links:  [cyan]{stats['total_links']}[/cyan]")
        console.print(f"Embedded:     [cyan]{stats['embedded_blocks']}[/cyan]")

        if stats["blocks_by_kind"]:
            console.print("\n[bold]Blocks by kind:[/bold]")
            for kind, count in sorted(stats["blocks_by_kind"].items(), key=lambda x: -x[1]):
                console.print(f"  {kind}: {count}")

        if stats["links_by_kind"]:
            console.print("\n[bold]Links by kind:[/bold]")
            for kind, count in sorted(stats["links_by_kind"].items(), key=lambda x: -x[1]):
                console.print(f"  {kind}: {count}")
    finally:
        store.close()


# ── Service management ─────────────────────────────────────────────

service_app = typer.Typer(help="Manage OpenAugi as a launchd service (macOS).")
app.add_typer(service_app, name="service")

_PLIST_NAME = "com.openaugi.server"
_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_NAME}.plist"
_LOG_DIR = Path.home() / ".openaugi" / "logs"


def _find_openaugi_bin() -> str:
    """Find the openaugi executable path."""
    import shutil
    import sys

    # Prefer the bin in the same venv as the running Python
    venv_bin = Path(sys.executable).parent / "openaugi"
    if venv_bin.exists():
        return str(venv_bin)

    # Fall back to PATH lookup
    found = shutil.which("openaugi")
    if found:
        return found

    raise FileNotFoundError(
        "Cannot find 'openaugi' executable. Install with: pipx install openaugi"
    )


def _generate_plist(bin_path: str, port: int = 8787, allowed_hosts: str = "") -> str:
    """Generate a launchd plist for the OpenAugi HTTP server."""
    env_block = """    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>"""
    if allowed_hosts:
        env_block += f"""
        <key>OPENAUGI_ALLOWED_HOSTS</key>
        <string>{allowed_hosts}</string>"""
    env_block += """
    </dict>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{bin_path}</string>
        <string>serve</string>
        <string>--transport</string>
        <string>streamable-http</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>{port}</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{_LOG_DIR / "server.log"}</string>

    <key>StandardErrorPath</key>
    <string>{_LOG_DIR / "server.err"}</string>

{env_block}
</dict>
</plist>
"""


@service_app.command("install")
def service_install(
    port: int = typer.Option(8787, "--port", "-p", help="HTTP port for the server"),
    allowed_hosts: str = typer.Option(
        "", "--allowed-hosts", help="Comma-separated tunnel hostnames (e.g. mcp.example.com)"
    ),
):
    """Install OpenAugi as a launchd service (starts on boot, restarts on crash)."""
    import subprocess

    try:
        bin_path = _find_openaugi_bin()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    # Unload existing if present
    if _PLIST_PATH.exists():
        subprocess.run(
            ["launchctl", "unload", str(_PLIST_PATH)],
            capture_output=True,
        )

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    plist_content = _generate_plist(bin_path, port=port, allowed_hosts=allowed_hosts)
    _PLIST_PATH.write_text(plist_content)

    result = subprocess.run(
        ["launchctl", "load", str(_PLIST_PATH)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]Failed to load service:[/red] {result.stderr}")
        raise typer.Exit(1)

    console.print("[green]Service installed and started.[/green]")
    console.print(f"  Plist: [cyan]{_PLIST_PATH}[/cyan]")
    console.print(f"  Logs:  [cyan]{_LOG_DIR}[/cyan]")
    console.print(f"  URL:   [cyan]http://127.0.0.1:{port}/mcp[/cyan]")
    console.print("\nThe server will start on boot and restart on crash.")


@service_app.command("uninstall")
def service_uninstall():
    """Stop and remove the OpenAugi launchd service."""
    import subprocess

    if not _PLIST_PATH.exists():
        console.print("[yellow]Service not installed.[/yellow]")
        raise typer.Exit(0)

    subprocess.run(
        ["launchctl", "unload", str(_PLIST_PATH)],
        capture_output=True,
    )
    _PLIST_PATH.unlink(missing_ok=True)
    console.print("[green]Service stopped and removed.[/green]")


@service_app.command("stop")
def service_stop():
    """Kill switch: stop MCP server AND any running cloudflared tunnels."""
    import subprocess

    # Stop MCP service
    if _PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(_PLIST_PATH)], capture_output=True)
        console.print("[green]MCP service stopped.[/green]")
    else:
        console.print("[dim]MCP service not installed.[/dim]")

    # Kill any cloudflared tunnel processes
    result = subprocess.run(
        ["pkill", "-f", "cloudflared tunnel"],
        capture_output=True,
    )
    if result.returncode == 0:
        console.print("[green]Cloudflared tunnel killed.[/green]")
    else:
        console.print("[dim]No cloudflared tunnel running.[/dim]")

    # Verify nothing is listening
    port_check = subprocess.run(
        ["lsof", "-i", ":8787"],
        capture_output=True,
        text=True,
    )
    if port_check.stdout.strip():
        console.print("[yellow]Warning: something still listening on :8787[/yellow]")
    else:
        console.print("[green]Port 8787 is clear. Nothing is exposed.[/green]")


@service_app.command("start")
def service_start():
    """Re-start the MCP service (if installed). Does NOT start the tunnel."""
    import subprocess

    if not _PLIST_PATH.exists():
        console.print("[yellow]Service not installed. Run: openaugi service install[/yellow]")
        raise typer.Exit(1)

    subprocess.run(["launchctl", "load", str(_PLIST_PATH)], capture_output=True)
    console.print("[green]MCP service started (localhost only).[/green]")
    console.print("To expose via tunnel, run: cloudflared tunnel run openaugi")


@service_app.command("status")
def service_status():
    """Check if the MCP service and tunnel are running."""
    import subprocess

    # MCP service
    if not _PLIST_PATH.exists():
        console.print("[yellow]MCP service: not installed[/yellow]")
    else:
        result = subprocess.run(
            ["launchctl", "list", _PLIST_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]MCP service: running[/green]")
        else:
            console.print("[yellow]MCP service: installed but not running[/yellow]")

    # Port check
    port_check = subprocess.run(
        ["lsof", "-i", ":8787"],
        capture_output=True,
        text=True,
    )
    if port_check.stdout.strip():
        console.print("[green]Port 8787: listening (localhost)[/green]")
    else:
        console.print("[dim]Port 8787: not listening[/dim]")

    # Tunnel check
    tunnel_check = subprocess.run(
        ["pgrep", "-f", "cloudflared tunnel"],
        capture_output=True,
        text=True,
    )
    if tunnel_check.stdout.strip():
        console.print("[yellow]Cloudflared tunnel: running (publicly reachable)[/yellow]")
    else:
        console.print("[dim]Cloudflared tunnel: not running[/dim]")


def _print_block(block, score: float | None = None):
    """Print a block to the console."""
    score_str = f" [cyan]({score:.3f})[/cyan]" if score is not None else ""
    title = block.title or block.id[:12]
    console.print(f"\n[bold]{title}[/bold]{score_str}  [{block.kind}]")
    if block.tags:
        console.print(f"  tags: {', '.join(f'#{t}' for t in block.tags)}")
    if block.content:
        preview = block.content[:200].replace("\n", " ")
        console.print(f"  {preview}")


if __name__ == "__main__":
    app()
