"""OpenAugi CLI — ingest, serve, search, hubs, status.

Entry point for the `openaugi` command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

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

IgnoreSourceOption = Annotated[
    list[str] | None,
    typer.Option(
        "--ignore-source",
        help=(
            "fnmatch pattern on source_path to exclude "
            "(repeatable, e.g. --ignore-source 'journals/HW/*')"
        ),
    ),
]

IgnoreHeadingOption = Annotated[
    list[str] | None,
    typer.Option(
        "--ignore-heading",
        help=(
            "Section heading text to exclude (case-insensitive, repeatable, "
            "e.g. --ignore-heading HW --ignore-heading Private)"
        ),
    ),
]


def _setup_logging(verbose: bool):
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s  %(message)s", datefmt="%H:%M:%S"
    )

    # Console: INFO (or DEBUG with --verbose), always to stderr
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # File: DEBUG for our code, WARNING for third-party noise.
    # Rotated at 5 MB, keep 3 backups (~20 MB max).
    log_dir = Path.home() / ".openaugi" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "openaugi.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)s  %(message)s")
    )
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    for name in ("httpcore", "httpx", "openai", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


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


@app.command(name="re-embed")
def re_embed(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    kind: str = typer.Option(
        "data_block", "--kind", help="Block kind to re-embed (default: data_block)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Reset embeddings and re-embed from scratch with the current model config.

    Use after switching embedding models or enabling title-prepend.
    Nulls out all embeddings for the given block kind, then re-embeds
    using the model in your config (~/.openaugi/config.toml).

    Example — switch to text-embedding-3-large:
        # In ~/.openaugi/config.toml:
        # [models.embedding]
        # provider = "openai"
        # model = "text-embedding-3-large"

        openaugi re-embed
    """
    _setup_logging(verbose)

    from openaugi.config import load_config
    from openaugi.models import get_embedding_model
    from openaugi.pipeline.embed import run_embed
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    db_path = db or str(_default_db())
    store = SQLiteStore(db_path)

    try:
        console.print(f"[bold]Database:[/bold] {db_path}")
        console.print(f"[bold]Kind:[/bold] {kind}")

        count = store.reset_embeddings(kind=kind)
        console.print(f"[yellow]Reset {count} embeddings → will re-embed[/yellow]")

        model = get_embedding_model(config.get("models", {}).get("embedding"))
        console.print(f"[bold]Model:[/bold] {model.name} ({model.dimensions} dims)")

        embedded = run_embed(store, model)
        console.print(f"[green]Done. Embedded {embedded} blocks.[/green]")
    finally:
        store.close()


@app.command()
@app.command()
def serve(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="Transport: stdio or streamable-http"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP host (streamable-http only)"),
    port: int = typer.Option(8787, "--port", "-p", help="HTTP port (streamable-http only)"),
    auth: str | None = typer.Option(
        None, "--auth", help="Auth provider for remote access (e.g. cloudflare)"
    ),
):
    """Start MCP server.

    Default: stdio transport for Claude Desktop/Code.
    Use --transport streamable-http for remote access (Claude mobile, Tailscale, etc.).
    Use --auth cloudflare to enable OAuth via Cloudflare Access.
    """
    import os

    _setup_logging(verbose=False)

    if db:
        os.environ["OPENAUGI_DB"] = db

    from openaugi.mcp.server import run_server

    resolved = "streamable-http" if transport == "http" else transport
    run_server(transport=resolved, host=host, port=port, auth_provider=auth)  # type: ignore[arg-type]


@app.command()
def up(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="Transport: stdio or streamable-http"
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP host (streamable-http only)"),
    port: int = typer.Option(8787, "--port", help="HTTP port (streamable-http only)"),
    auth: str | None = typer.Option(
        None, "--auth", help="Auth provider for remote access (e.g. cloudflare)"
    ),
    debounce: float = typer.Option(30.0, "--debounce", "-d", help="Watcher debounce seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ingest, watch, and serve — the one command to run OpenAugi.

    1. Runs incremental ingest (fast if already up-to-date)
    2. Starts file watcher as a background thread
    3. Starts MCP server in the foreground
    """
    import os

    _setup_logging(verbose)

    # Status output must go to stderr — stdout is the MCP stdio protocol channel.
    err = Console(stderr=True)

    from openaugi.config import load_config
    from openaugi.pipeline.runner import run_layer0
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()

    vault_path = path or config.get("vault", {}).get("default_path")
    if not vault_path:
        err.print("[red]No vault path specified.[/red]")
        err.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())

    err.print(f"[bold]Vault:[/bold] {vault_path}")
    err.print(f"[bold]Database:[/bold] {db_path}")

    # Step 1: Incremental ingest
    err.print("\n[bold]Syncing vault...[/bold]")
    store = SQLiteStore(db_path)
    try:
        exclude = config.get("vault", {}).get("exclude_patterns")
        workers = config.get("vault", {}).get("max_workers", 4)
        result = run_layer0(vault_path, store, exclude_patterns=exclude, max_workers=workers)
        stats = result["stats"]
        err.print(
            f"  {stats['total_blocks']} blocks, {stats['total_links']} links "
            f"({result['blocks_added']} new, {result['blocks_removed']} removed)"
        )

        # Embedding — graceful fallback
        try:
            from openaugi.models import get_embedding_model
            from openaugi.pipeline.embed import run_embed

            model = get_embedding_model(config.get("models", {}).get("embedding"))
            count = run_embed(store, model)
            if count:
                err.print(f"  Embedded {count} blocks")
        except Exception as e:
            err.print(f"  [dim]Embeddings skipped: {e}[/dim]")

    finally:
        store.close()

    # Step 2: File watcher
    from openaugi.pipeline.watcher import start_watcher_thread

    start_watcher_thread(vault_path, db_path, config, debounce_seconds=debounce)
    err.print(f"\n[bold]Watcher:[/bold] debounce={debounce}s")
    err.print(f"[bold]MCP:[/bold] {transport}")
    err.print()

    # Step 3: MCP server (blocks main thread)
    if db:
        os.environ["OPENAUGI_DB"] = db

    from openaugi.mcp.server import run_server

    resolved = "streamable-http" if transport == "http" else transport
    run_server(transport=resolved, host=host, port=port, auth_provider=auth)  # type: ignore[arg-type]


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
def heartbeat(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    max_blocks: int = typer.Option(
        50, "--max-blocks", "-n", help="Max new blocks to process in one run"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Build the prompt but do not launch the agent"
    ),
    ingest: bool = typer.Option(
        False,
        "--ingest",
        help="Run incremental ingest before processing (use when `up` is not running)",
    ),
    ignore_source: IgnoreSourceOption = None,
    ignore_heading: IgnoreHeadingOption = None,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Heartbeat — find new blocks and hand them to a Claude Code agent.

    The Python side is a dumb script:
      1. Finds entry blocks added since ~/.openaugi/last_heartbeat
      2. Builds a prompt listing the blocks + any `zzz:` instructions
      3. Spawns `claude -p <prompt>` with openaugi MCP tools allowed
      4. Updates the timestamp on success

    Assumes `openaugi up` is already running and keeping the DB current.
    Pass --ingest to run a one-off incremental ingest first if `up` is not
    running.

    The reasoning lives in the skill file at
    <vault>/OpenAugi/heartbeat-skill.md. Edit the skill file, not the code,
    when the agent does the wrong thing.

    See docs/plans/heartbeat.md for the full design.
    """
    _setup_logging(verbose)

    from openaugi.config import load_config
    from openaugi.pipeline.heartbeat import run_heartbeat
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()

    vault_path = path or config.get("vault", {}).get("default_path")
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())
    store = SQLiteStore(db_path)

    try:
        # Step 1: incremental ingest — opt-in when `up` is not running.
        if ingest:
            from openaugi.pipeline.runner import run_layer0

            console.print(f"[bold]Syncing vault:[/bold] {vault_path}")
            exclude = config.get("vault", {}).get("exclude_patterns")
            workers = config.get("vault", {}).get("max_workers", 4)
            result = run_layer0(vault_path, store, exclude_patterns=exclude, max_workers=workers)
            stats = result["stats"]
            console.print(
                f"  {stats['total_blocks']} blocks, {stats['total_links']} links "
                f"({result['blocks_added']} new, {result['blocks_removed']} removed)"
            )

            # Embedding — graceful fallback (agent search still works via FTS without it).
            try:
                from openaugi.models import get_embedding_model
                from openaugi.pipeline.embed import run_embed

                model = get_embedding_model(config.get("models", {}).get("embedding"))
                count = run_embed(store, model)
                if count:
                    console.print(f"  Embedded {count} blocks")
            except Exception as e:
                console.print(f"  [dim]Embeddings skipped: {e}[/dim]")

        # Step 2–4: find blocks, build prompt, spawn agent.
        ignore = list(ignore_source or []) or config.get("heartbeat", {}).get("ignore_sources", [])
        ignore_h = list(ignore_heading or []) or config.get("heartbeat", {}).get(
            "ignore_headings", []
        )
        try:
            result = run_heartbeat(
                store=store,
                vault_path=vault_path,
                max_blocks=max_blocks,
                dry_run=dry_run,
                ignore_sources=ignore or None,
                ignore_headings=ignore_h or None,
            )
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        since = result["since"] or "(first run)"
        console.print(f"\n[bold]Heartbeat window:[/bold] {since} → {result['now']}")
        console.print(f"[bold]Skill file:[/bold] {result['skill_file']}")
        console.print(f"[bold]Heartbeat log:[/bold] {result['heartbeat_log']}")
        console.print(f"[bold]New blocks:[/bold] {result['block_count']}")

        if result["batch_capped"]:
            console.print(
                f"[yellow]Batch capped at {max_blocks}.[/yellow] "
                "Run again to process the next batch, or raise --max-blocks."
            )

        if result["block_count"] == 0:
            console.print("[dim]Nothing to process. Timestamp advanced.[/dim]")
            return

        if dry_run:
            console.print("\n[bold]--- Prompt (dry run) ---[/bold]")
            console.print(result["prompt"])
            return

        if result["return_code"] == 0:
            console.print(
                f"\n[green]Heartbeat complete.[/green] Review the log at {result['heartbeat_log']}"
            )
        else:
            console.print(
                f"\n[yellow]Agent exited with {result['return_code']}.[/yellow] "
                "Timestamp was NOT advanced — the next run will retry these blocks."
            )
            raise typer.Exit(result["return_code"] or 1)
    finally:
        store.close()


@app.command("agent")
def agent_cmd(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    interval: int = typer.Option(5, "--interval", "-i", help="Heartbeat interval in minutes"),
    max_blocks: int = typer.Option(
        50, "--max-blocks", "-n", help="Max blocks per heartbeat batch"
    ),
    ingest: bool = typer.Option(
        False,
        "--ingest",
        help="Run incremental ingest on each heartbeat cycle (use when `up` is not running)",
    ),
    lookback: int = typer.Option(
        24,
        "--lookback",
        help="Hours of history to process on first run (no prior state). 0 = all.",
    ),
    ignore_source: IgnoreSourceOption = None,
    ignore_heading: IgnoreHeadingOption = None,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Heartbeat loop + task dispatch — the agent processing companion to `up`.

    While `up` handles vault sync and the MCP server, `agent` handles processing:

      1. Runs heartbeat every --interval minutes (default 5)
      2. Watches OpenAugi/Tasks/ and dispatches pending tasks as Claude Code
         agents in named tmux sessions

    Assumes `openaugi up` is running and keeping the DB current.
    Pass --ingest if `up` is not running.
    """
    import threading
    import time

    _setup_logging(verbose)

    err = Console(stderr=True)

    from openaugi.config import load_config
    from openaugi.pipeline.heartbeat import run_heartbeat
    from openaugi.pipeline.task_watcher import watch_tasks
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()

    vault_path = path or config.get("vault", {}).get("default_path")
    if not vault_path:
        err.print("[red]No vault path specified.[/red]")
        err.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())
    ignore = list(ignore_source or []) or config.get("heartbeat", {}).get("ignore_sources", [])
    ignore_h = list(ignore_heading or []) or config.get("heartbeat", {}).get("ignore_headings", [])

    err.print(f"[bold]Vault:[/bold] {vault_path}")
    err.print(f"[bold]Heartbeat:[/bold] every {interval}m")
    if ignore:
        err.print(f"[bold]Ignoring sources:[/bold] {', '.join(ignore)}")
    if ignore_h:
        err.print(f"[bold]Ignoring headings:[/bold] {', '.join(ignore_h)}")

    # Start task-dispatch in a background daemon thread
    dispatch_thread = threading.Thread(
        target=watch_tasks,
        kwargs=dict(
            vault_path=vault_path,
            tasks_folder="OpenAugi/Tasks",
            repos_note="OpenAugi/Repos.md",
            poll_interval=5.0,
            settle=30.0,
        ),
        daemon=True,
        name="task-dispatch",
    )
    dispatch_thread.start()
    err.print("[bold]Task dispatch:[/bold] watching OpenAugi/Tasks/\n")

    # On first run (no prior state), apply lookback window to avoid processing full vault.
    from openaugi.pipeline.heartbeat import get_last_heartbeat, set_last_heartbeat

    if get_last_heartbeat() is None and lookback > 0:
        from datetime import UTC, datetime, timedelta

        default_since = (datetime.now(UTC) - timedelta(hours=lookback)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        set_last_heartbeat(default_since)
        err.print(f"[dim]First run — lookback {lookback}h, starting from {default_since}[/dim]")

    # Heartbeat loop — runs in the main thread
    while True:
        store = SQLiteStore(db_path)
        try:
            if ingest:
                from openaugi.pipeline.runner import run_layer0

                exclude = config.get("vault", {}).get("exclude_patterns")
                workers = config.get("vault", {}).get("max_workers", 4)
                res = run_layer0(vault_path, store, exclude_patterns=exclude, max_workers=workers)
                stats = res["stats"]
                err.print(
                    f"  Synced: {stats['total_blocks']} blocks "
                    f"({res['blocks_added']} new, {res['blocks_removed']} removed)"
                )
                try:
                    from openaugi.models import get_embedding_model
                    from openaugi.pipeline.embed import run_embed

                    model = get_embedding_model(config.get("models", {}).get("embedding"))
                    count = run_embed(store, model)
                    if count:
                        err.print(f"  Embedded {count} blocks")
                except Exception as e:
                    err.print(f"  [dim]Embeddings skipped: {e}[/dim]")

            try:
                result = run_heartbeat(
                    store=store,
                    vault_path=vault_path,
                    max_blocks=max_blocks,
                    ignore_sources=ignore or None,
                    ignore_headings=ignore_h or None,
                )
            except FileNotFoundError as e:
                err.print(f"[red]{e}[/red]")
                raise typer.Exit(1) from None

            if result["block_count"] == 0:
                err.print(f"[dim]Heartbeat: nothing new. Next run in {interval}m.[/dim]")
            elif result["return_code"] == 0:
                err.print(
                    f"[green]Heartbeat complete[/green] ({result['block_count']} blocks). "
                    f"Next run in {interval}m."
                )
            else:
                err.print(
                    f"[yellow]Heartbeat agent exited {result['return_code']}.[/yellow] "
                    f"Retrying same window next run in {interval}m."
                )
        finally:
            store.close()

        time.sleep(interval * 60)


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


# ── Task dispatch ──────────────────────────────────────────────────


@app.command("task-dispatch")
def task_dispatch(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    tasks_folder: str = typer.Option(
        "OpenAugi/Tasks",
        "--tasks-folder",
        help="Relative folder in the vault where task files live",
    ),
    repos_note: str = typer.Option(
        "OpenAugi/Repos.md",
        "--repos-note",
        help="Path (relative to vault) of the note mapping repo names → absolute paths",
    ),
    interval: float = typer.Option(5.0, "--interval", help="Poll interval in seconds"),
    settle: float = typer.Option(
        30.0,
        "--settle",
        help="Seconds a file must be unchanged before processing (debounces writes)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Watch OpenAugi/Tasks/ and dispatch pending task files to tmux.

    When a task file lands with `status: pending` in frontmatter, the
    watcher hydrates it (assigns a task_id, timestamps, tmux session name),
    resolves the working directory from `working_dir` / `repo` frontmatter
    (via `OpenAugi/Repos.md` by default), builds a prompt, and launches
    `claude` in a detached named tmux session. Attach any time with
    `tmux attach -t <task_id>`.

    Raw notes are never modified — the watcher only touches files in the
    configured Tasks folder. See src/openaugi/templates/heartbeat-skill.md
    for the task file format the heartbeat agent writes.
    """
    _setup_logging(verbose)

    from openaugi.config import load_config
    from openaugi.pipeline.task_watcher import watch_tasks

    config = load_config()
    vault_path = path or config.get("vault", {}).get("default_path")
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    try:
        watch_tasks(
            vault_path=vault_path,
            tasks_folder=tasks_folder,
            repos_note=repos_note,
            poll_interval=interval,
            settle=settle,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


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
