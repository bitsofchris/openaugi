#!/usr/bin/env python3
"""check_private_vocab.py — refuse to commit private vocabulary or notebook output.

This repo is public; the vault is not. Two things have leaked before, and
this hook stops both at commit time:

1. **Private vocabulary.** The user's own field names, value words, note
   titles — anything that only means something in their vault. The list of
   words lives *outside* the repo, in the vault (`OpenAugi/AGENT/
   private-vocabulary.txt`), so the guard itself can never be the leak.
   One entry per line; blank lines and `#` comments are ignored; entries
   match case-insensitively as plain substrings, or as a regex when the
   line starts with `re:`; a line starting with `skip:` is a path glob the
   check leaves alone (LICENSE, NOTICE — where the public author identity
   belongs). Offending lines are printed with the match masked, so the
   hook's own output is safe to paste anywhere.

2. **Notebook outputs.** A `.ipynb` with cell outputs is how real vault
   text once reached the history. Any notebook with a non-empty `outputs`
   list is refused; clear outputs before committing.

The word list is resolved from `--denylist`, then `$OPENAUGI_PRIVATE_VOCAB`,
then `<vault>/OpenAugi/AGENT/private-vocabulary.txt` with the vault taken
from the openaugi config. With no list found the vocabulary check is
skipped with a notice (other installs, CI); the notebook check always runs.

Usage (pre-commit passes the file names; see .pre-commit-config.yaml):
    python3 scripts/check_private_vocab.py FILE [FILE ...]
    python3 scripts/check_private_vocab.py --all          # every tracked file
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DENYLIST_REL = "OpenAugi/AGENT/private-vocabulary.txt"
ENV_VAR = "OPENAUGI_PRIVATE_VOCAB"
MASK = "▮▮▮"
SKIP_PARTS = ("docs/scratch", ".git", ".venv", "node_modules", "__pycache__")


def resolve_denylist(explicit: str | None) -> Path | None:
    """The word list: the flag, the env var, or the vault file via the openaugi config."""
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    try:
        from openaugi.config import load_config, resolve_vault_path
    except ImportError:
        return None
    vault = resolve_vault_path(None, load_config())
    return Path(vault) / DENYLIST_REL if vault else None


def load_rules(path: Path) -> tuple[list[re.Pattern[str]], list[str]]:
    """The list as (word patterns, path globs to skip)."""
    rules, skips = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("skip:"):
            skips.append(line[5:].strip())
            continue
        pattern = line[3:].strip() if line.startswith("re:") else re.escape(line)
        rules.append(re.compile(pattern, re.IGNORECASE))
    return rules, skips


def notebook_has_outputs(text: str) -> bool:
    try:
        nb = json.loads(text)
    except ValueError:
        return False
    return any(cell.get("outputs") for cell in nb.get("cells", []) if isinstance(cell, dict))


def is_binary(path: Path) -> bool:
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)


def scan_file(path: Path, rules: list[re.Pattern[str]], label: str | None = None) -> list[str]:
    """Offending lines as `label:n: <line with matches masked>`; [] when clean."""
    label = label or str(path)
    if not path.is_file() or is_binary(path):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    offenders = []
    if path.suffix == ".ipynb" and notebook_has_outputs(text):
        offenders.append(f"{label}: notebook has cell outputs — clear them before committing")
    for number, line in enumerate(text.splitlines(), 1):
        masked, hits = line, 0
        for rule in rules:
            masked, n = rule.subn(MASK, masked)
            hits += n
        if hits:
            offenders.append(f"{label}:{number}: {masked.strip()}")
    return offenders


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout
    return [root / p for p in out.split("\0") if p]


def skipped(path: Path, globs: list[str] = ()) -> bool:
    posix = path.as_posix()
    if any(part in posix for part in SKIP_PARTS):
        return True
    rel = posix.removeprefix(Path.cwd().as_posix() + "/")
    return any(fnmatch.fnmatch(rel, g) for g in globs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*", help="files to scan (pre-commit passes these)")
    ap.add_argument("--all", action="store_true", help="scan every tracked file instead")
    ap.add_argument("--denylist", help=f"word list (default: ${ENV_VAR}, then the vault file)")
    args = ap.parse_args(argv)

    root = Path.cwd()
    files = tracked_files(root) if args.all else [Path(f) for f in args.files]

    denylist = resolve_denylist(args.denylist)
    rules: list[re.Pattern[str]] = []
    skips: list[str] = []
    if denylist and denylist.exists():
        rules, skips = load_rules(denylist)
    else:
        where = denylist or f"${ENV_VAR} / <vault>/{DENYLIST_REL}"
        print(
            f"check_private_vocab: no word list at {where}; vocabulary check skipped",
            file=sys.stderr,
        )

    files = [f for f in files if not skipped(f, skips)]
    offenders = [line for f in files for line in scan_file(f, rules)]
    if offenders:
        print("Private content refused. Vocabulary lives in the vault, not the repo:")
        print("  " + "\n  ".join(offenders))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
