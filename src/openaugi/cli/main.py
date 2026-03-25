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
        toml_lines.extend([
            "[models.embedding]",
            f'provider = "{embedding_provider}"',
            f'model = "{embedding_model}"',
            "",
        ])
    toml_lines.extend([
        "[vault]",
        f'default_path = "{vault_path}"',
    ])

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
        console.print(f"\n[green]Done.[/green] "
                       f"{stats['total_blocks']} blocks, {stats['total_links']} links")

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
):
    """Start MCP server (stdio transport)."""
    import os

    if db:
        os.environ["OPENAUGI_DB"] = db

    from openaugi.mcp.server import run_server

    run_server()


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
            for kind, count in sorted(
                stats["blocks_by_kind"].items(), key=lambda x: -x[1]
            ):
                console.print(f"  {kind}: {count}")

        if stats["links_by_kind"]:
            console.print("\n[bold]Links by kind:[/bold]")
            for kind, count in sorted(
                stats["links_by_kind"].items(), key=lambda x: -x[1]
            ):
                console.print(f"  {kind}: {count}")
    finally:
        store.close()


@app.command(name="migrate-vec")
def migrate_vec(
    db: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Migrate existing embeddings into the sqlite-vec vector table.

    Run this once on existing databases after upgrading to sqlite-vec.
    No re-embedding is needed — copies existing blobs from blocks table.
    """
    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())

    if not Path(db_path).exists():
        console.print(f"[red]Database not found:[/red] {db_path}")
        raise typer.Exit(1)

    store = SQLiteStore(db_path)
    try:
        # Infer dimension from first embedding blob
        import numpy as np

        blocks = store.get_blocks_with_embeddings()
        if not blocks:
            console.print(
                "[yellow]No embeddings found in database. Run 'openaugi ingest' first.[/yellow]"
            )
            return

        dim = len(np.frombuffer(blocks[0].embedding, dtype=np.float32))  # type: ignore[arg-type]
        console.print(f"Detected embedding dimension: [cyan]{dim}[/cyan]")
        console.print(f"Migrating [cyan]{len(blocks)}[/cyan] embeddings to vec_blocks...")

        count = store.populate_vec_from_blocks(dim)
        console.print(f"[green]✓[/green] Migrated {count} embeddings")
    finally:
        store.close()


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
