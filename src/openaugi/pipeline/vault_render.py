"""Vault rendering — write compiled content as .md files to the vault.

Extracts context blocks to OpenAugi/Compiled/ for human-readable browsing.
Not currently wired up; preserved for future use.
"""

from __future__ import annotations

from pathlib import Path

from openaugi.model.block import Block


def render_blocks_to_vault(context_blocks: list[Block], vault_path: str | Path) -> int:
    """Render context blocks as .md files to OpenAugi/Compiled/ in vault.

    Returns the number of files written.
    """
    compiled_dir = Path(vault_path) / "OpenAugi" / "Compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    _clean_compiled_dir(compiled_dir)

    rendered = 0
    for block in context_blocks:
        ct = block.metadata.get("context_type", "unknown")

        if ct == "index":
            path = compiled_dir / "INDEX.md"
        elif ct == "recent_activity":
            path = compiled_dir / "RECENT.md"
        elif ct == "graph_health":
            path = compiled_dir / "HEALTH.md"
        elif ct == "hub_summary":
            name = _safe_filename(block.metadata.get("scope", "unknown").replace("tag:", ""))
            path = compiled_dir / "hubs" / f"{name}.md"
        elif ct == "concept":
            name = _safe_filename(block.metadata.get("scope", "unknown").replace("tag:", ""))
            path = compiled_dir / "concepts" / f"{name}.md"
        else:
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block.content or "", encoding="utf-8")
        rendered += 1

    return rendered


def _clean_compiled_dir(compiled_dir: Path) -> None:
    """Remove all .md files from the compiled directory."""
    if not compiled_dir.exists():
        return
    for md_file in compiled_dir.rglob("*.md"):
        md_file.unlink()
    for subdir in sorted(compiled_dir.rglob("*"), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()


def _safe_filename(name: str) -> str:
    """Convert a tag/hub name to a safe filename."""
    safe = name.replace("/", "-").replace("\\", "-")
    safe = "".join(c for c in safe if c.isalnum() or c in "-_ ")
    return safe.strip() or "unnamed"
