#!/usr/bin/env python3
"""sync_templates.py — the vault's engine files become the shipped templates.

The vault copy of every agent file is the source of truth (AGENTS.md). This
script is the one way templates get refreshed: every file under
``<vault>/OpenAugi/AGENT/`` that declares ``kind: engine`` is transformed
(personal regions stripped, the ``- [ ] seen`` tick dropped) and written to
``src/openaugi/templates/`` at the same relative path. ``kind: personal`` files
are skipped; a file with no ``kind:`` is an error, because the boundary is only
real if every file is on one side of it.

    python3 scripts/sync_templates.py --check      # report drift, exit 1 on any
    python3 scripts/sync_templates.py --write      # refresh the templates

The vault path comes from the openaugi config unless ``--vault`` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openaugi.agent_files import (
    ENGINE,
    NON_AGENT_TEMPLATES,
    PERSONAL,
    PersonalRegionError,
    iter_templates,
    read_kind,
    to_template,
)
from openaugi.config import load_config, resolve_vault_path

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "src" / "openaugi" / "templates"
AGENT_DIR = Path("OpenAugi") / "AGENT"


def plan(vault: Path, templates: Path = TEMPLATES) -> dict[str, list[str]]:
    """What a sync would do, keyed by outcome. Pure — reads, never writes."""
    agent_dir = vault / AGENT_DIR
    out: dict[str, list[str]] = {
        "unchanged": [],
        "drift": [],
        "new": [],
        "personal": [],
        "no_kind": [],
        "bad_region": [],
        "orphan_template": [],
    }
    seen_engine: set[str] = set()
    for f in sorted(agent_dir.rglob("*.md")):
        rel = f.relative_to(agent_dir).as_posix()
        text = f.read_text(encoding="utf-8")
        kind = read_kind(text)
        if kind == PERSONAL:
            out["personal"].append(rel)
            continue
        if kind != ENGINE:
            out["no_kind"].append(rel)
            continue
        seen_engine.add(rel)
        try:
            shipped = to_template(text)
        except PersonalRegionError as e:
            out["bad_region"].append(f"{rel}: {e}")
            continue
        dest = templates / rel
        if not dest.exists():
            out["new"].append(rel)
        elif dest.read_text(encoding="utf-8") != shipped:
            out["drift"].append(rel)
        else:
            out["unchanged"].append(rel)
    for rel, text in iter_templates(templates):
        if rel in seen_engine or Path(rel).name in NON_AGENT_TEMPLATES:
            continue
        if read_kind(text) == ENGINE and rel not in seen_engine:
            out["orphan_template"].append(rel)
    return out


def write(vault: Path, templates: Path = TEMPLATES) -> list[str]:
    """Refresh every engine template from its vault twin; return what was written."""
    agent_dir = vault / AGENT_DIR
    written: list[str] = []
    for f in sorted(agent_dir.rglob("*.md")):
        text = f.read_text(encoding="utf-8")
        if read_kind(text) != ENGINE:
            continue
        rel = f.relative_to(agent_dir).as_posix()
        dest = templates / rel
        shipped = to_template(text)
        if dest.exists() and dest.read_text(encoding="utf-8") == shipped:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(shipped, encoding="utf-8")
        written.append(rel)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vault", default=None, help="vault root (default: openaugi config)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report drift; exit 1 if any")
    mode.add_argument("--write", action="store_true", help="refresh templates from the vault")
    args = ap.parse_args(argv)

    raw = args.vault or resolve_vault_path(None, load_config())
    if not raw:
        ap.error("no vault: pass --vault or set [vault] default_path in the openaugi config")
    vault = Path(raw).expanduser()
    if not (vault / AGENT_DIR).is_dir():
        ap.error(f"no {AGENT_DIR} under {vault}")

    report = plan(vault)
    problems = report["no_kind"] or report["bad_region"]
    for key in ("no_kind", "bad_region", "orphan_template"):
        for item in report[key]:
            print(f"{key:16} {item}")
    if problems:
        print("\nevery AGENT file needs `kind: engine` or `kind: personal`; fix the above first")
        return 1

    if args.check:
        for key in ("new", "drift"):
            for item in report[key]:
                print(f"{key:16} {item}")
        n = len(report["new"]) + len(report["drift"])
        print(
            f"\n{len(report['unchanged'])} in sync, {n} to refresh, "
            f"{len(report['personal'])} personal (never shipped)"
        )
        return 1 if n else 0

    for rel in write(vault):
        print(f"wrote            {rel}")
    print(f"\n{len(report['personal'])} personal files skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
