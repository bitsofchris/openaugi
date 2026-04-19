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

    # Copy agent templates to vault (only if files don't already exist)
    import importlib.resources

    vault = Path(vault_path)
    agent_dir = vault / "OpenAugi" / "AGENT"
    agent_dir.mkdir(parents=True, exist_ok=True)

    templates = importlib.resources.files("openaugi") / "templates"
    agent_templates = {
        "augi-agent.md": "Base agent skill — how the agent handles tasks",
        "research-agent.md": "Research sub-agent — NotebookLM, source ingestion",
    }

    copied = 0
    for filename, desc in agent_templates.items():
        dest = agent_dir / filename
        if dest.exists():
            console.print(f"  [dim]Skipping {filename} (already exists)[/dim]")
            continue
        source = templates / filename
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        # Strip the TEMPLATE marker from the vault copy's frontmatter
        text = dest.read_text(encoding="utf-8")
        text = text.replace(" (template)", "")
        text = text.replace(
            "  TEMPLATE — copied to <vault>/OpenAugi/AGENT/",
            "  Live agent skill file at OpenAugi/AGENT/",
        )
        text = text.replace(
            "  The vault copy is the live version the agent reads. Edit there, not here.\n"
            "  This file is the factory default for new users.\n",
            "  Edit this file to change agent behavior.\n",
        )
        text = text.replace(
            "  The vault copy is the live version the agent reads. Edit there, not here.\n",
            "  Edit this file to change agent behavior.\n",
        )
        dest.write_text(text, encoding="utf-8")
        console.print(f"  [green]Copied {filename}[/green] — {desc}")
        copied += 1

    if copied:
        console.print(f"\n  Agent skills at [cyan]{agent_dir}[/cyan]")
        console.print("  Edit these in Obsidian to customize agent behavior.")

    console.print("\n[bold green]Setup complete![/bold green]\n")
    console.print("Next steps:")
    console.print("  openaugi up              # sync vault + start everything")
    console.print("  openaugi serve           # MCP server only (for Claude Desktop)")


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
def split(
    file: str = typer.Argument(..., help="Path to a markdown file to split"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json | md | ndjson"),
):
    """Split a markdown file into blocks — deterministic, no DB, no LLM.

    Prints segments to stdout using the same rules as `openaugi ingest`. Useful
    from agents, skills, scripts, or cron jobs that need "give me the blocks
    OpenAugi would see in this file" without touching a store.

    Formats:
      json    — single JSON object: {source_path, doc_hash, filename_date,
                frontmatter_tags, segments: [...]}
      ndjson  — one JSON line per segment (stream-friendly)
      md      — human-readable markdown, one section per segment
    """
    import json as _json
    import sys

    from openaugi.adapters.splitter import split_file

    result = split_file(file)

    if format == "json":
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
        return
    if format == "ndjson":
        for seg in result.segments:
            sys.stdout.write(_json.dumps(seg.model_dump(), ensure_ascii=False) + "\n")
        return
    if format == "md":
        sys.stdout.write(f"# {result.source_path}\n")
        sys.stdout.write(
            f"_{len(result.segments)} segments · doc_hash={result.doc_hash}"
            + (f" · date={result.filename_date}" if result.filename_date else "")
            + "_\n\n"
        )
        for i, seg in enumerate(result.segments, 1):
            head = f"Block {i}"
            if seg.section_heading:
                head += f" — {seg.section_heading}"
            if seg.section_date:
                head += f" ({seg.section_date})"
            sys.stdout.write(f"## {head}\n\n{seg.clean_content}\n\n")
            if seg.zzz_instructions:
                sys.stdout.write(f"**zzz:** {' | '.join(seg.zzz_instructions)}\n\n")
            meta = []
            if seg.tags:
                meta.append("tags: " + ", ".join(f"#{t}" for t in seg.tags))
            if seg.links:
                meta.append("links: " + ", ".join(f"[[{ln}]]" for ln in seg.links))
            if meta:
                sys.stdout.write("_" + " · ".join(meta) + "_\n\n")
        return

    console.print(f"[red]Unknown format: {format}[/red] (use json | ndjson | md)")
    raise typer.Exit(1)


@app.command(name="re-embed")
def re_embed(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    kind: str = typer.Option(
        "data_block", "--kind", help="Block kind to re-embed (default: data_block)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Reset embeddings and re-embed from scratch with the current model config.

    Use after switching embedding models or changing the embedding strategy.
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
def cluster(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    pass_id: str | None = typer.Option(None, "--pass", help="Run only this pass id"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print stats, no DB writes"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run HDBSCAN clustering DAG, write context_block:cluster blocks to DB.

    Reads [[clustering.passes]] from ~/.openaugi/config.toml.
    Each pass produces context_block:cluster blocks linked to member data_blocks.
    Idempotent — re-running replaces existing cluster blocks for each pass.

    Example config.toml:

    \b
        [[clustering.passes]]
        id = "life_areas"
        dims = 64
        scope = "all"
        min_cluster_size = 50
        description = "Coarse life area clusters"

        [[clustering.passes]]
        id = "life_areas_fine"
        dims = 3072
        scope = "within"
        parent_pass = "life_areas"
        min_cluster_size = 20
        description = "Fine topic clusters within each life area"
    """
    _setup_logging(verbose)

    from openaugi.config import load_config
    from openaugi.pipeline.cluster import parse_cluster_passes, run_cluster_dag
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    db_path = db or str(_default_db())

    try:
        passes = parse_cluster_passes(config)
    except (ValueError, KeyError) as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1) from e

    if not passes:
        console.print("[yellow]No clustering passes configured.[/yellow]")
        console.print("Add [[clustering.passes]] to ~/.openaugi/config.toml")
        raise typer.Exit(0)

    if pass_id:
        passes = [p for p in passes if p.id == pass_id]
        if not passes:
            console.print(f"[red]No pass with id '{pass_id}' found in config.[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Passes:[/bold] {', '.join(p.id for p in passes)}")
    if dry_run:
        console.print("[yellow]Dry run — no DB writes[/yellow]")

    store = SQLiteStore(db_path)
    try:
        run_cluster_dag(store, passes, dry_run=dry_run)
        console.print("\n[green]Done.[/green]")
    finally:
        store.close()


@app.command(name="cluster-explore")
def cluster_explore(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    dims: str = typer.Option(
        "64,96,128,256,512", "--dims", help="Comma-separated dim sizes to try"
    ),
    k: str = typer.Option("5,8,10,12,15", "--k", help="Comma-separated cluster counts to try"),
    samples: int = typer.Option(8, "--samples", help="Sample titles per cluster"),
    input_level: str = typer.Option("document", "--input", help="'document' or 'block'"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Explore k-means cluster combinations without writing to DB.

    Tries every (dims, k) combination and prints cluster summaries with sample
    document titles so you can assess quality before committing to config.toml.

    Example:

    \b
        openaugi cluster-explore --dims 64,96,128 --k 8,10,12,15
    """
    _setup_logging(verbose)

    from openaugi.pipeline.cluster import explore_kmeans_grid
    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())
    dims_list = [int(x.strip()) for x in dims.split(",")]
    k_list = [int(x.strip()) for x in k.split(",")]

    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Dims:[/bold] {dims_list}")
    console.print(f"[bold]K values:[/bold] {k_list}")
    console.print(f"[bold]Input level:[/bold] {input_level}")

    store = SQLiteStore(db_path)
    try:
        explore_kmeans_grid(store, dims_list, k_list, n_samples=samples, input_level=input_level)
    finally:
        store.close()


@app.command(name="cluster-explore-within")
def cluster_explore_within(
    parent_pass: str = typer.Option("life_areas", "--pass", help="Coarse pass id"),
    cluster: str = typer.Option(..., "--cluster", help="Cluster label to explore (e.g. '7')"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    dims: str = typer.Option("1536,3072", "--dims", help="Comma-separated dims to try"),
    hdbscan_sizes: str = typer.Option(
        "10,20,30", "--hdbscan-sizes", help="min_cluster_size values for HDBSCAN"
    ),
    k: str = typer.Option("5,8,10", "--k", help="k values for k-means fallback"),
    samples: int = typer.Option(8, "--samples", help="Sample titles per cluster"),
    embedding_col: str = typer.Option(
        "embedding", "--embedding-col", help="'embedding' or 'content_only_embedding'"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Explore fine clustering within one coarse cluster — HDBSCAN and k-means at multiple dims.

    Tries HDBSCAN (noise = signal; recurring ideas emerge as density) and k-means
    (every block assigned) at each dims setting. Writes nothing to DB.

    Use --embedding-col content_only_embedding after running embed-content-only
    to compare clustering with and without title-prepending.

    Example:

    \b
        openaugi cluster-explore-within --cluster 7
        openaugi cluster-explore-within --cluster 7 --embedding-col content_only_embedding
    """
    _setup_logging(verbose)

    from openaugi.pipeline.cluster import explore_fine_cluster
    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())
    dims_list = [int(x.strip()) for x in dims.split(",")]
    min_sizes = [int(x.strip()) for x in hdbscan_sizes.split(",")]
    k_list = [int(x.strip()) for x in k.split(",")]

    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Pass:[/bold] {parent_pass}  cluster={cluster}")
    console.print(f"[bold]Dims:[/bold] {dims_list}  HDBSCAN min_sizes={min_sizes}  k={k_list}")
    console.print(f"[bold]Embedding:[/bold] {embedding_col}")

    store = SQLiteStore(db_path)
    try:
        explore_fine_cluster(
            store,
            parent_pass_id=parent_pass,
            cluster_label=cluster,
            dims_list=dims_list,
            hdbscan_min_sizes=min_sizes,
            k_list=k_list,
            n_samples=samples,
            embedding_col=embedding_col,
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
    no_agent: bool = typer.Option(
        False, "--no-agent", help="Disable task dispatch (watcher + MCP only, no agent sessions)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ingest, watch, dispatch, and serve — the one command to run OpenAugi.

    1. Runs incremental ingest (fast if already up-to-date)
    2. Starts file watcher as a background thread (zzz → task files)
    3. Starts task dispatch as a background thread (task files → tmux agents)
    4. Starts MCP server in the foreground
    """
    import os
    import threading

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

        # Dispatch any zzz instructions from initial ingest
        new_blocks = result.get("new_data_blocks", [])
        if new_blocks:
            from openaugi.pipeline.dispatch import dispatch_zzz_blocks

            dispatched = dispatch_zzz_blocks(new_blocks, vault_path)
            if dispatched:
                err.print(f"  Dispatched {len(dispatched)} zzz task(s)")

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

    # Step 2: File watcher (includes zzz dispatch post-ingest)
    from openaugi.pipeline.watcher import start_watcher_thread

    start_watcher_thread(vault_path, db_path, config, debounce_seconds=debounce)
    err.print(f"\n[bold]Watcher:[/bold] debounce={debounce}s")

    # Step 3: Task dispatch (picks up pending task files → tmux agents)
    if not no_agent:
        from openaugi.agents.task_watcher import watch_tasks

        dispatch_thread = threading.Thread(
            target=watch_tasks,
            kwargs=dict(vault_path=vault_path),
            daemon=True,
            name="task-dispatch",
        )
        dispatch_thread.start()
        err.print("[bold]Agent:[/bold] watching OpenAugi/Tasks/")

    err.print(f"[bold]MCP:[/bold] {transport}")
    err.print()

    # Step 4: MCP server (blocks main thread)
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
    configured Tasks folder. Task files are written by the zzz dispatch
    hook (pipeline/dispatch.py) or manually by the user.
    """
    _setup_logging(verbose)

    from openaugi.agents.task_watcher import watch_tasks
    from openaugi.config import load_config

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


def _explore_prewarm(db_path: str, backend_dir: Path) -> None:
    """Pre-generate and cache UMAP projections for all dims options."""
    import subprocess
    import sys

    DIMS_OPTIONS = [32, 64, 96, 128, 256, 512, 1024, 1280, 1536, 2048, 2560, 3072]

    # Install backend deps if needed
    try:
        import numpy  # noqa: F401
        import umap  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        console.print("[yellow]Installing backend deps…[/yellow]")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "umap-learn",
                "numpy",
                "hdbscan",
                "scikit-learn",
            ]
        )

    # Import server helpers directly — no server needed for cache-only work
    sys.path.insert(0, str(backend_dir))
    try:
        from server import (  # pyright: ignore[reportMissingImports]
            _normalize_coords,
            _truncate_normalize,
            compute_umap,
            load_data_blocks,
            open_db,
        )
    except ImportError as e:
        console.print(f"[red]Could not import server: {e}[/red]")
        raise typer.Exit(1) from None

    import numpy as np

    console.print(f"[bold]Pre-warming UMAP cache[/bold] for {len(DIMS_OPTIONS)} dims values")
    console.print(f"  db: {db_path}")
    console.print("  cache: ~/.openaugi/umap_cache/\n")

    conn = open_db(db_path)
    try:
        blocks = load_data_blocks(conn)
    finally:
        conn.close()

    console.print(f"  Loaded [cyan]{len(blocks):,}[/cyan] blocks\n")

    full_matrix = np.stack(
        [np.frombuffer(b["embedding"], dtype=np.float32).copy() for b in blocks]
    )

    import time

    for dims in DIMS_OPTIONS:
        actual = min(dims, full_matrix.shape[1])
        console.print(f"  [bold]dims={dims}[/bold]  ({actual} actual) …", end="")
        t0 = time.time()
        matrix = _truncate_normalize(full_matrix, dims)
        _normalize_coords(compute_umap(matrix))
        elapsed = time.time() - t0
        console.print(f"  [green]done[/green] ({elapsed:.1f}s)")

    console.print(
        "\n[green]All UMAP projections cached.[/green] Run 'openaugi explore' to launch."
    )


@app.command()
def explore(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    backend_port: int = typer.Option(8000, "--backend-port", help="API server port"),
    frontend_port: int = typer.Option(5173, "--frontend-port", help="Vite dev server port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
    prewarm: bool = typer.Option(
        False, "--prewarm", help="Pre-generate UMAP caches for all dims, then exit"
    ),
):
    """Launch the Knowledge Explorer — WebGL cluster visualization.

    Starts the FastAPI backend (UMAP projection) and Vite frontend, then opens
    the browser. Installs Python and npm deps automatically on first run.

    Use --prewarm to pre-generate UMAP projections for all embedding dims so
    exploration is instant later (runs without launching the browser).

    Example:

    \b
        openaugi explore
        openaugi explore --db /path/to/custom.db
        openaugi explore --prewarm
    """
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    _setup_logging(verbose=False)

    repo_root = Path(__file__).resolve().parents[3]
    experiment_dir = repo_root / "experiments" / "knowledge-explorer"
    backend_dir = experiment_dir / "backend"
    frontend_dir = experiment_dir / "frontend"

    if not experiment_dir.exists():
        console.print(f"[red]Knowledge Explorer not found at {experiment_dir}[/red]")
        raise typer.Exit(1)

    db_path = db or str(_default_db())
    if not Path(db_path).exists():
        console.print(f"[red]Database not found:[/red] {db_path}")
        console.print("Run 'openaugi ingest --path /your/vault' first.")
        raise typer.Exit(1)

    # ── Prewarm: generate UMAP caches for all dims, no browser ──────────────────
    if prewarm:
        _explore_prewarm(db_path, backend_dir)
        return

    # ── Backend deps ────────────────────────────────────────────────────────────
    try:
        import fastapi  # noqa: F401  # pyright: ignore[reportMissingImports]
        import umap  # noqa: F401  # pyright: ignore[reportMissingImports]
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[yellow]Installing backend deps…[/yellow]")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "fastapi",
                "uvicorn",
                "umap-learn",
                "numpy",
                "hdbscan",
                "scikit-learn",
            ]
        )

    # ── Frontend deps ───────────────────────────────────────────────────────────
    if not (frontend_dir / "node_modules").exists():
        console.print("[yellow]Installing frontend deps (npm install)…[/yellow]")
        subprocess.check_call(["npm", "install"], cwd=frontend_dir)

    # ── Launch backend ──────────────────────────────────────────────────────────
    console.print(f"[bold]Starting backend[/bold] on :{backend_port}  (db: {db_path})")
    backend_proc = subprocess.Popen(
        [sys.executable, "server.py", "--db", db_path, "--port", str(backend_port)],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait up to 8s for backend to bind, streaming output so errors are visible
    import threading

    backend_ready = threading.Event()
    backend_lines: list[str] = []

    def _stream_backend() -> None:
        assert backend_proc.stdout
        for line in backend_proc.stdout:
            line = line.rstrip()
            backend_lines.append(line)
            console.print(f"  [dim]{line}[/dim]")
            if "Application startup complete" in line:
                backend_ready.set()

    t = threading.Thread(target=_stream_backend, daemon=True)
    t.start()

    if not backend_ready.wait(timeout=10):
        rc = backend_proc.poll()
        if rc is not None:
            console.print(
                f"[red]Backend exited immediately (code {rc}). "
                f"Is port {backend_port} already in use?[/red]"
            )
            raise typer.Exit(1)
        # Still starting — continue anyway
    console.print("  [green]Backend ready[/green]")

    # ── Launch frontend ─────────────────────────────────────────────────────────
    env = os.environ.copy()
    env["VITE_API_PORT"] = str(backend_port)
    console.print(f"[bold]Starting frontend[/bold] on :{frontend_port}")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(frontend_port), "--host"],
        cwd=frontend_dir,
        env=env,
    )

    url = f"http://localhost:{frontend_port}"

    if open_browser:
        time.sleep(2)
        subprocess.Popen(["open", url])

    console.print(f"\n  [bold green]Knowledge Explorer:[/bold green] {url}")
    console.print(f"  [dim]Backend API:        http://localhost:{backend_port}/api[/dim]")
    console.print("  [dim](Ctrl-C to stop)[/dim]\n")

    try:
        # Keep streaming backend output; wait for either process to exit
        t.join(timeout=None)  # streams until backend stdout closes (on exit)
        backend_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        console.print("[dim]Stopped.[/dim]")


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
