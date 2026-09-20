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
    vault_path = str(Path(vault_path.strip().strip("'\"")).expanduser())

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

    # Copy the engine templates to the vault (only files that don't already exist).
    # Which files ship is decided by the templates themselves: every markdown
    # file under templates/ that declares `kind: engine` — see agent_files.py.
    from openaugi.agent_files import engine_templates, template_description

    vault = Path(vault_path)
    agent_dir = vault / "OpenAugi" / "AGENT"
    agent_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for filename, text in engine_templates():
        dest = agent_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            console.print(f"  [dim]Skipping {filename} (already exists)[/dim]")
            continue
        dest.write_text(text, encoding="utf-8")
        console.print(f"  [green]Copied {filename}[/green] — {template_description(text)}")
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

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.runner import run_layer0
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()

    # Resolve vault path: CLI arg > config > error
    vault_path = resolve_vault_path(path, config)
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())
    store = SQLiteStore(db_path)

    try:
        exclude = config.get("vault", {}).get("exclude_patterns")
        workers = config.get("vault", {}).get("max_workers", 4)
        source_rules = config.get("vault", {}).get("source_rules")
        provenance_rules = config.get("vault", {}).get("provenance_rules")

        console.print(f"[bold]Ingesting vault:[/bold] {vault_path}")
        console.print(f"[bold]Database:[/bold] {db_path}")

        result = run_layer0(
            vault_path,
            store,
            exclude_patterns=exclude,
            max_workers=workers,
            source_rules=source_rules,
            provenance_rules=provenance_rules,
        )

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


routing_app = typer.Typer(help="Augi Log routing — what waits, and what your history says.")
app.add_typer(routing_app, name="routing")


@routing_app.command(name="stats")
def routing_stats(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Show the routing priors (per target, per folder+verb) and the logs still waiting."""
    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.route import load_priors, waiting_logs
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    vault_path = resolve_vault_path(path, config)
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        raise typer.Exit(1)

    priors = load_priors(Path(vault_path))
    console.print(f"[bold]{priors.decisions} routing decisions[/bold]")
    for signal, n in sorted(priors.signals.items(), key=lambda kv: -kv[1]):
        console.print(f"  {signal:<10} {n}")
    if priors.targets:
        console.print("\n[bold]Targets[/bold]  chosen / seen")
        for title, (chosen, seen) in sorted(priors.targets.items(), key=lambda kv: -kv[1][1]):
            console.print(f"  {chosen:>3} / {seen:<3}  {title}")
    if priors.verbs:
        console.print("\n[bold]Verbs by folder[/bold]  chosen / seen")
        for (folder, verb), (chosen, seen) in sorted(priors.verbs.items()):
            console.print(f"  {chosen:>3} / {seen:<3}  {verb:<10} {folder}")

    store = SQLiteStore(db or str(_default_db()))
    try:
        waiting = waiting_logs(store)
    finally:
        store.close()
    console.print(f"\n[bold]{len(waiting)} log(s) waiting[/bold] for 'process this log'")
    for log in waiting:
        console.print(f"  {log.get('day')}  {log.get('path')}")


@app.command(name="context-pack")
def context_pack(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Regenerate OpenAugi/context-pack.json (mobile capture-assist sidecar)."""
    _setup_logging(verbose)

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.context_pack import write_context_pack
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    vault_path = resolve_vault_path(path, config)
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    store = SQLiteStore(db or str(_default_db()))
    try:
        out = write_context_pack(store, vault_path)
        console.print(f"[green]Wrote[/green] {out}")
    finally:
        store.close()


@app.command()
def lenses(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    check: bool = typer.Option(
        False, "--check", help="Exit non-zero if any lens has broken frontmatter"
    ),
):
    """List the lens registry (OpenAugi/AGENT/lenses/) and flag broken specs."""
    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.context_pack import LENSES_DIR, read_lens_specs

    config = load_config()
    vault_path = resolve_vault_path(path, config)
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    specs = read_lens_specs(Path(vault_path))
    if not specs:
        console.print(f"[yellow]No lenses found[/yellow] in {Path(vault_path) / LENSES_DIR}")
        raise typer.Exit(1 if check else 0)

    table = Table(title=f"Lens registry — {len(specs)} lens(es)")
    table.add_column("name", style="bold")
    table.add_column("trigger")
    table.add_column("description", max_width=60)
    table.add_column("status")
    broken = 0
    for lens in specs:
        if "error" in lens:
            broken += 1
            status = f"[red]INVALID YAML[/red] {lens['error'][:40]}"
        elif not lens["description"]:
            status = "[yellow]no description[/yellow] (mobile chip will be blank)"
        else:
            status = "[green]ok[/green]"
        table.add_row(lens["name"], lens.get("trigger", ""), lens["description"], status)
    console.print(table)

    if broken:
        console.print(
            f"\n[red]{broken} lens(es) have invalid frontmatter[/red] — they still ship "
            "to the context pack (salvaged), but fix them: use folded scalars "
            "(`description: >-`) for values with quotes or colons."
        )
        if check:
            raise typer.Exit(1)


@app.command()
def render(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    days: int = typer.Option(180, "--days", help="Window of recent days to include"),
    out: str | None = typer.Option(
        None, "--out", help="Output HTML path (default: vault OpenAugi/render/)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Render the lifestream — static HTML stream + heat strip from the DB."""
    _setup_logging(verbose)

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.render.lifestream import render_lifestream
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    vault_path = resolve_vault_path(path, config)
    if not vault_path and not out:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path / --out, or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    store = SQLiteStore(db or str(_default_db()), read_only=True)
    try:
        target = render_lifestream(store, vault_path or ".", days=days, out=out)
        console.print(f"[green]Wrote[/green] {target}")
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


@app.command()
def lineage(
    query: str = typer.Argument(..., help="The idea/topic to trace, e.g. 'dopamine'"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    k: int = typer.Option(150, "--k", help="Semantic candidates to consider"),
    max_distance: float = typer.Option(
        1.2, "--max-distance", help="Similarity cutoff (L2 on unit vectors; lower = stricter)"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit full report as JSON"),
    write: bool = typer.Option(
        False, "--write", help="Also write <vault>/OpenAugi/lineage/<slug>.json (mobile payload)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Time-ordered semantic evidence for one idea — feeds the idea-lineage lens.

    Semantic search across all history, bucketed into quarters: first mention,
    activity per era, dormant gaps, last mention. Deterministic evidence; the
    narrative (revisions, dead branches, strongest form) is the lens's job.
    """
    _setup_logging(verbose)

    import json as json_mod

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.models import get_embedding_model
    from openaugi.pipeline.lineage import (
        compute_lineage,
        render_lineage_markdown,
        write_lineage_sidecar,
    )
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    db_path = db or str(_default_db())
    store = SQLiteStore(db_path)
    try:
        model = get_embedding_model(config.get("models", {}).get("embedding"))
        report = compute_lineage(store, model, query, k=k, max_distance=max_distance)
        if json_out:
            print(json_mod.dumps(report, indent=2))
        else:
            print(render_lineage_markdown(report))
        if write:
            vault_path = resolve_vault_path(config=config)
            err = Console(stderr=True)  # keep stdout clean for --json consumers
            if not vault_path:
                err.print("[red]--write needs [vault] default_path in config.toml[/red]")
                raise typer.Exit(1)
            out = write_lineage_sidecar(report, vault_path)
            err.print(f"[green]Sidecar written:[/green] {out}")
    finally:
        store.close()


@app.command(name="backfill-source-tags")
def backfill_source_tags_cmd(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report matches, write nothing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Apply [vault.source_rules] to data_blocks already in the DB.

    Ingest only touches changed files, so enabling source rules does nothing
    for existing rows — run this once after adding rules to config.toml.
    Explicit source/* tags in note text always win. Idempotent.
    """
    _setup_logging(verbose)

    from openaugi.adapters.vault import backfill_source_tags
    from openaugi.config import load_config
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    rules = config.get("vault", {}).get("source_rules")
    if not rules:
        console.print("[yellow]No [vault.source_rules] in ~/.openaugi/config.toml[/yellow]")
        raise typer.Exit(0)

    db_path = db or str(_default_db())
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Rules:[/bold] {rules}")
    if dry_run:
        console.print("[yellow]Dry run — no DB writes[/yellow]")

    store = SQLiteStore(db_path)
    try:
        stats = backfill_source_tags(store, rules, dry_run=dry_run)
        if not stats:
            console.print("[green]Nothing to do — all matching blocks already attributed.[/green]")
        else:
            for tag, count in sorted(stats.items()):
                console.print(f"  {tag}: {count} blocks")
            console.print(
                f"[green]{'Would update' if dry_run else 'Updated'} "
                f"{sum(stats.values())} blocks.[/green]"
            )
    finally:
        store.close()


@app.command(name="backfill-provenance")
def backfill_provenance_cmd(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report matches, write nothing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Stamp metadata.provenance (human | ai | reference) on data_blocks already in the DB.

    Applies [vault.provenance_rules], explicit provenance/* tags, and the
    AI/source tag rules — the same resolution ingest uses. Idempotent.
    Blocks whose title matches [vault.provenance_title_patterns] but resolved
    to `human` are listed as candidates for a hand-added provenance/ai tag;
    a title is never treated as evidence on its own.
    """
    _setup_logging(verbose)

    from openaugi.adapters.vault import backfill_provenance
    from openaugi.config import load_config
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()
    vault_cfg = config.get("vault", {})
    rules = vault_cfg.get("provenance_rules")
    patterns = vault_cfg.get("provenance_title_patterns") or []

    db_path = db or str(_default_db())
    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print(f"[bold]Rules:[/bold] {rules or '(none — tag rules and default only)'}")
    if dry_run:
        console.print("[yellow]Dry run — no DB writes[/yellow]")

    store = SQLiteStore(db_path)
    try:
        out = backfill_provenance(store, rules, dry_run=dry_run, title_patterns=patterns)
        updated = out["updated"]
        if not updated:
            console.print(
                f"[green]Nothing to do — {out['unchanged']} blocks already stamped.[/green]"
            )
        else:
            for value, count in sorted(updated.items()):
                console.print(f"  {value}: {count} blocks")
            console.print(
                f"[green]{'Would update' if dry_run else 'Updated'} "
                f"{sum(updated.values())} blocks ({out['unchanged']} unchanged).[/green]"
            )
        if out["candidates"]:
            console.print(
                f"\n[yellow]{len(out['candidates'])} human-labelled blocks match a title "
                "pattern — add a provenance/ai tag to the note if a model wrote it:[/yellow]"
            )
            seen: set[str] = set()
            for _, title in out["candidates"]:
                if title not in seen:
                    seen.add(title)
                    console.print(f"  {title}")
    finally:
        store.close()


@app.command(name="cluster-weather")
def cluster_weather(
    db: str | None = typer.Option(None, "--db", help="Database path"),
    window: int = typer.Option(14, "--window", help="Activity/delta window in days"),
    top: int = typer.Option(10, "--top", help="Clusters shown per pass (markdown output)"),
    json_out: bool = typer.Option(False, "--json", help="Emit full report as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Deterministic cluster growth/death report for the cluster-weather lens.

    Diffs the latest cluster_run snapshot against the most recent one older
    than the window (clusters matched across runs by member overlap), plus
    per-cluster recent activity from block timestamps. Run 'openaugi cluster'
    first to refresh clusters and record a snapshot.
    """
    _setup_logging(verbose)

    import json as json_mod

    from openaugi.pipeline.cluster_weather import compute_weather, render_weather_markdown
    from openaugi.store.sqlite import SQLiteStore

    db_path = db or str(_default_db())
    store = SQLiteStore(db_path)
    try:
        report = compute_weather(store, window_days=window)
        if json_out:
            print(json_mod.dumps(report, indent=2))
        else:
            print(render_weather_markdown(report, top=top))
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
def review(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    dashboard_only: bool = typer.Option(
        False,
        "--dashboard-only",
        help="Process Dashboard nomination answers only (no new-block routing)",
    ),
):
    """Trigger a review pass by writing a task file (the task watcher runs it).

    The task file is the API: this command, the zzz grammar, and (future)
    Obsidian plugin buttons all converge on the same OpenAugi/Tasks/ contract.
    Requires `openaugi up` (or the task watcher) running to pick it up.
    """
    from datetime import datetime

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.dispatch import DEFAULT_TASKS_FOLDER

    config = load_config()
    vault_path = resolve_vault_path(path, config)
    if not vault_path:
        console.print("[red]No vault path specified.[/red]")
        console.print("Use --path or run 'openaugi init' to set a default.")
        raise typer.Exit(1)

    instruction = "process the dashboard" if dashboard_only else "run the review pass"
    slug = instruction.replace(" ", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tasks_dir = Path(vault_path) / DEFAULT_TASKS_FOLDER
    tasks_dir.mkdir(parents=True, exist_ok=True)
    filepath = tasks_dir / f"{slug}-{timestamp}.md"

    task = f"""---
status: pending
source_block_id: cli
source_note: "[[openaugi review]]"
---

# {instruction}

## Context

Triggered via `openaugi review` CLI at {timestamp}.

## User instruction

> {instruction}

## Task

Read OpenAugi/AGENT/review-pass.md and execute: {instruction}.

## Human Todo

## Results
"""
    filepath.write_text(task, encoding="utf-8")
    console.print(f"[green]Task file written:[/green] {filepath}")
    console.print("The task watcher will pick it up (requires 'openaugi up' running).")


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
    serve_mcp: bool = typer.Option(
        False,
        "--serve",
        help=(
            "Also serve MCP in the foreground. Single-terminal use only — "
            "normally point clients at `openaugi serve`"
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """The background service: ingest, watch, dispatch. ONE per vault.

    1. Runs incremental ingest (fast if already up-to-date)
    2. Starts file watcher as a background thread (zzz → task files)
    3. Starts task dispatch as a background thread (task files → tmux agents)

    **This does not serve MCP.** Use `openaugi serve` for that, and point every
    MCP client at it.

    The two have opposite cardinality, which is why they are separate commands:

    - A watcher/dispatcher must be **singleton**. Two of them see the same
      `zzz:` land and both dispatch it, so one instruction becomes two agents
      racing over one database.
    - A stdio MCP server is inherently **per-client**. Claude Desktop, Codex,
      and any other client each spawn their own; that is how stdio transport
      works.

    Conflating them meant a second client's launch either raced the first or,
    once the singleton lock existed, exited immediately and handed that client
    a dead server. `--serve` exists for the single-terminal case where you want
    both and there is only one client.
    """
    import os
    import threading

    _setup_logging(verbose)

    # Status output must go to stderr — stdout is the MCP stdio protocol channel.
    err = Console(stderr=True)

    # ONE INSTANCE ONLY, and this has to come before anything else touches the
    # vault or the database. Two `up` processes both watch and both dispatch,
    # so one `zzz:` becomes two agents racing over the same DB. Exit 0 rather
    # than 1: under launchd's KeepAlive a non-zero exit is a crash to retry,
    # and retrying forever against a healthy instance is worse than stopping.
    from openaugi.singleton import AlreadyRunning, acquire

    try:
        acquire("up")
    except AlreadyRunning as exc:
        err.print(f"[yellow]openaugi up: {exc}[/yellow]")
        err.print("Stop it first, or use `openaugi serve` for a second read-only server.")
        raise typer.Exit(0) from None

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.runner import run_layer0
    from openaugi.store.sqlite import SQLiteStore

    config = load_config()

    vault_path = resolve_vault_path(path, config)
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

    # Stamp the code this process started from. A daemon that runs for weeks
    # keeps executing whatever it imported at startup, so a fix committed
    # since then is not running and nothing else would say so.
    from datetime import datetime

    from openaugi.service_version import head_sha, record_service_start

    record_service_start(store, datetime.now().isoformat(timespec="seconds"), os.getpid())
    running = head_sha()
    err.print(f"[bold]Code:[/bold] {running[:12] if running else 'not a git checkout'}")

    try:
        exclude = config.get("vault", {}).get("exclude_patterns")
        workers = config.get("vault", {}).get("max_workers", 4)
        source_rules = config.get("vault", {}).get("source_rules")
        provenance_rules = config.get("vault", {}).get("provenance_rules")
        result = run_layer0(
            vault_path,
            store,
            exclude_patterns=exclude,
            max_workers=workers,
            source_rules=source_rules,
            provenance_rules=provenance_rules,
        )
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

    if db:
        os.environ["OPENAUGI_DB"] = db

    # Step 4: block the main thread. Either serving MCP, or just parking so the
    # daemon threads keep running.
    if not serve_mcp:
        # The default: this is a service, not an interface. Park the main
        # thread so the watcher and dispatcher keep running.
        #
        # Serving here by default was the old behaviour and it was wrong twice
        # over. Under a supervisor, stdin is /dev/null — the stdio server reads
        # EOF and returns, taking the watcher down with it, after logging a
        # clean startup. And under an MCP client, every client spawns its own,
        # so the singleton watcher could not coexist with them.
        err.print("[bold]MCP:[/bold] not served — point clients at `openaugi serve`")
        err.print()
        import threading as _threading

        _threading.Event().wait()  # park forever; daemon threads do the work
        return

    err.print(f"[bold]MCP:[/bold] {transport}")
    err.print()

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

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.pipeline.watcher import watch_vault

    config = load_config()

    vault_path = resolve_vault_path(path, config)
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
    query: str | None = typer.Argument(None, help="Semantic search query"),
    k: int = typer.Option(10, "--k", "-k", help="Number of results"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    keyword: bool = typer.Option(False, "--keyword", help="Use FTS instead of semantic"),
    title: str | None = typer.Option(None, "--title", help="Search note titles only"),
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Filter by tag (repeatable)")
    ] = None,
    after: str | None = typer.Option(None, "--after", help="block_time >= (YYYY-MM-DD)"),
    before: str | None = typer.Option(None, "--before", help="block_time <= (YYYY-MM-DD)"),
    after_ingested: str | None = typer.Option(
        None, "--after-ingested", help="Ingested since (ISO timestamp)"
    ),
    kind: str | None = typer.Option(None, "--kind", help="Block kind (default data_block)"),
    exclude_path_prefix: str | None = typer.Option(
        None, "--exclude-path-prefix", help="Drop blocks whose source_path starts with this"
    ),
    include_path_prefix: str | None = typer.Option(
        None, "--include-path-prefix", help="Keep only blocks whose source_path starts with this"
    ),
    task: bool = typer.Option(False, "--task", help="Only user-marked tasks"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Search the knowledge base from the terminal.

    Same engine and rules as the MCP search tool: positional QUERY is
    semantic, --keyword flips it to FTS, --title searches titles, and
    filters alone browse. Every filter the agents get works here too.
    """
    _setup_logging(verbose)

    from openaugi.query import QuerySpec
    from openaugi.store.sqlite import SQLiteStore

    spec = QuerySpec(
        query=None if keyword else query,
        keyword=query if keyword else None,
        title=title,
        tags=list(tag) if tag else None,
        after=after,
        before=before,
        after_ingested=after_ingested,
        kind=kind,
        exclude_path_prefix=exclude_path_prefix,
        include_path_prefix=include_path_prefix,
        has_task=task or None,
        k=k,
    )
    if spec.is_empty():
        console.print("[red]Provide a query, --title, or at least one filter.[/red]")
        raise typer.Exit(1)

    store = SQLiteStore(db or str(_default_db()), read_only=True)
    try:
        _run_and_print_spec(store, spec)
    finally:
        store.close()


def _run_and_print_spec(store, spec) -> None:
    """Execute a QuerySpec and render results for the terminal."""
    from openaugi.config import load_config
    from openaugi.models import get_embedding_model
    from openaugi.query import engine

    model = None
    if spec.mode == "semantic":
        config = load_config()
        model = get_embedding_model(config.get("models", {}).get("embedding"))

    result = engine.run(store, spec, embedding_model=model)

    if not result.blocks:
        console.print("[yellow]No results. (New vault? Run 'openaugi ingest' first.)[/yellow]")
    for b in result.blocks:
        _print_block(b, score=result.scores.get(b.id))
    if result.mode == "browse" and result.reference_documents:
        console.print(
            f"\n[dim]{result.reference_block_count} reference block(s) collapsed into "
            f"{len(result.reference_documents)} source document(s).[/dim]"
        )
    if result.has_more:
        console.print("[dim]More results available — raise -k or use --after/--before.[/dim]")


@app.command("query")
def query_cmd(
    name: str | None = typer.Argument(None, help="Saved query name (omit to list)"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    vault: str | None = typer.Option(None, "--vault", help="Vault path (default: config)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a saved query from <vault>/OpenAugi/AGENT/queries/ (list with no args)."""
    _setup_logging(verbose)

    from openaugi.config import load_config, resolve_vault_path
    from openaugi.query import saved
    from openaugi.store.sqlite import SQLiteStore

    vault_path = resolve_vault_path(vault, load_config())
    if not vault_path:
        console.print("[red]No vault path configured. Use --vault or run 'openaugi init'.[/red]")
        raise typer.Exit(1)

    if name is None:
        queries = saved.list_saved(vault_path)
        if not queries:
            console.print(f"[yellow]No saved queries in {saved.queries_dir(vault_path)}[/yellow]")
            return
        for q in queries:
            console.print(f"[bold]{q.name}[/bold] — {q.description}")
        return

    try:
        sq = saved.load_saved(vault_path, name)
    except saved.SavedQueryNotFound:
        console.print(f"[red]Saved query not found: {name}[/red] (run 'openaugi query' to list)")
        raise typer.Exit(1) from None
    except saved.SavedQueryError as e:
        console.print(f"[red]Saved query '{name}' is invalid:[/red] {e}")
        raise typer.Exit(1) from None

    store = SQLiteStore(db or str(_default_db()), read_only=True)
    try:
        resolved = saved.resolve_spec(sq.spec, store=store)
        console.print(f"[dim]{sq.description}[/dim]")
        _run_and_print_spec(store, resolved)
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

        # Loudest line on the page when it fires: everything below can be
        # healthy while the daemon runs code from a week ago.
        from openaugi.service_version import version_drift

        drift = version_drift(store)
        if drift:
            behind = drift["commits_behind"]
            plural = "" if behind == 1 else "s"
            trailer = f" ({behind} commit{plural} behind)" if behind else ""
            console.print(
                f"[red]⚠ `openaugi up` is running stale code{trailer}.[/red]\n"
                f"  running: [yellow]{drift['running_sha'][:12]}[/yellow] "
                f"(started {drift['started_at']})\n"
                f"  HEAD:    [green]{drift['head_sha'][:12]}[/green]\n"
                f"  Restart it: [bold]launchctl kickstart -k gui/$(id -u)/com.openaugi.up[/bold]\n"
            )

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


@app.command()
def doctor(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Is `openaugi up` alive and ticking? Exits 1 when the heartbeat is stale.

    Watcher pid, running commit vs HEAD, age of the last drain tick, and every
    scheduled lens's last run and next due — the terminal twin of the
    Dashboard's heartbeat block, for when Obsidian is not open. Read-only.
    """
    from openaugi.config import load_config, resolve_vault_path
    from openaugi.doctor import diagnose, render
    from openaugi.store.sqlite import SQLiteStore

    vault_path = resolve_vault_path(path, load_config())
    if not vault_path:
        console.print("[red]No vault path specified.[/red] Use --path or set it in config.")
        raise typer.Exit(1)

    db_path = db or str(_default_db())
    if not Path(db_path).exists():
        console.print(f"[red]Database not found:[/red] {db_path}")
        raise typer.Exit(1)

    store = SQLiteStore(db_path, read_only=True)
    try:
        report = diagnose(Path(vault_path), store)
    finally:
        store.close()

    console.print(render(report), markup=False, highlight=False)
    if not report["healthy"]:
        raise typer.Exit(1)


# ── Reading queue ──────────────────────────────────────────────────

reading_app = typer.Typer(
    help="Reading queue — flagged notes out to Readwise Reader, highlights back."
)
app.add_typer(reading_app, name="reading")


def _reading_vault(path: str | None) -> str:
    from openaugi.config import load_config, resolve_vault_path

    vault_path = resolve_vault_path(path, load_config())
    if not vault_path:
        console.print("[red]No vault path specified.[/red] Use --path or set it in config.")
        raise typer.Exit(1)
    return vault_path


@reading_app.command(name="push")
def reading_push(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    cap: int = typer.Option(2, "--cap", help="Maximum documents to push in one run"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would ship; send nothing, record nothing"
    ),
    show_html: bool = typer.Option(
        False, "--show-html", help="With --dry-run, print the rendered HTML of each note"
    ),
):
    """Push notes flagged `reading_queue: true` into Readwise Reader.

    Nothing ships unless a note carries the flag, and never more than --cap in
    one run. Re-running is safe: an unchanged note is skipped, an edited one
    updates its Reader document in place.
    """
    from openaugi.reading.note import to_html
    from openaugi.reading.push import build_payload, find_flagged_notes, push_notes
    from openaugi.reading.reader_api import MissingTokenError, ReaderClient
    from openaugi.store.sqlite import SQLiteStore

    vault_path = _reading_vault(path)

    if dry_run and show_html:
        for note in find_flagged_notes(vault_path):
            console.print(f"\n[bold]{note.rel_path}[/bold]  ({note.word_count()} words)")
            console.print(f"[dim]{build_payload(note)['url']}[/dim]")
            console.print(to_html(note.body))

    client = None
    if not dry_run:
        try:
            client = ReaderClient()
        except MissingTokenError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    store = SQLiteStore(db or str(_default_db()))
    try:
        result = push_notes(vault_path, store, client, cap=cap, dry_run=dry_run)
    finally:
        store.close()
        if client:
            client.close()

    label = "[yellow]would push[/yellow]" if dry_run else "[green]pushed[/green]"
    for rel in result.pushed:
        console.print(f"  {label} {rel}")
    for rel in result.deferred_over_cap:
        console.print(f"  [dim]over cap, waits for tomorrow[/dim] {rel}")
    for rel in result.skipped_unchanged:
        console.print(f"  [dim]unchanged[/dim] {rel}")
    for rel, err in result.failed:
        console.print(f"  [red]failed[/red] {rel}: {err}")
    console.print(f"\n{result.summary}")


@reading_app.command(name="harvest")
def reading_harvest(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
    since: str | None = typer.Option(
        None, "--since", help="ISO timestamp; defaults to the last harvest"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be appended; write nothing"
    ),
):
    """Append new Reader highlights to the augi notes that produced them.

    Only documents augi pushed are touched — everything else in the Reader
    queue is your own reading and is left alone.
    """
    from openaugi.reading.harvest import harvest
    from openaugi.reading.reader_api import MissingTokenError, ReaderClient
    from openaugi.store.sqlite import SQLiteStore

    vault_path = _reading_vault(path)

    try:
        client = ReaderClient()
    except MissingTokenError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    store = SQLiteStore(db or str(_default_db()))
    try:
        result = harvest(vault_path, store, client, since=since, dry_run=dry_run)
    finally:
        store.close()
        client.close()

    verb = "would append to" if dry_run else "appended to"
    for rel in result.notes_updated:
        console.print(f"  [green]{verb}[/green] {rel}")
    console.print(f"\n{result.summary}")


@reading_app.command(name="status")
def reading_status(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    db: str | None = typer.Option(None, "--db", help="Database path"),
):
    """What is flagged, what has been pushed, and what came back."""
    from openaugi.reading.harvest import last_run
    from openaugi.reading.push import COLLECTION, find_flagged_notes
    from openaugi.store.sqlite import SQLiteStore

    vault_path = _reading_vault(path)
    flagged = find_flagged_notes(vault_path)

    store = SQLiteStore(db or str(_default_db()), read_only=True)
    try:
        ledger = store.list_records(COLLECTION, limit=10_000)
        harvested_at = last_run(store)
    finally:
        store.close()

    console.print(f"\n[bold]Reading queue[/bold]  ({vault_path})\n")
    console.print(f"Flagged notes: [cyan]{len(flagged)}[/cyan]")
    console.print(f"Pushed:        [cyan]{len(ledger)}[/cyan]")
    console.print(f"Last harvest:  [cyan]{harvested_at or 'never'}[/cyan]")

    if ledger:
        table = Table(show_header=True, header_style="bold")
        table.add_column("pushed")
        table.add_column("highlights", justify="right")
        table.add_column("note")
        for row in sorted(ledger, key=lambda r: str(r.get("pushed_at")), reverse=True):
            table.add_row(
                str(row.get("pushed_at", ""))[:10],
                str(len(row.get("harvested_ids") or [])),
                str(row.get("path", "")),
            )
        console.print(table)


# ── Task dispatch ──────────────────────────────────────────────────


@app.command("task-dispatch")
def task_dispatch(
    path: str | None = typer.Option(None, "--path", "-p", help="Path to Obsidian vault"),
    tasks_folder: str = typer.Option(
        "OpenAugi/Tasks",
        "--tasks-folder",
        help="Relative folder in the vault where task files live",
    ),
    repos_note: str | None = typer.Option(
        None,
        "--repos-note",
        help=(
            "Path (relative to vault) of the note mapping repo names → absolute paths. "
            "Default: OpenAugi/AGENT/Repos.md, falling back to OpenAugi/Repos.md"
        ),
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
    from openaugi.config import load_config, resolve_vault_path

    config = load_config()
    vault_path = resolve_vault_path(path, config)
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
