---
name: privacy-guard
description: The commit-time guard that keeps private content out of this public repo. `scripts/check_private_vocab.py` runs as a pre-commit hook at commit, commit-msg and push time; it refuses any file or commit message containing a word from a list that lives in the vault (never in the repo), and any notebook with cell outputs. The list's location and the reason it is outside the repo.
---

# Privacy guard

**Name:** privacy-guard · `scripts/check_private_vocab.py` · pre-commit hook

**Description:** This repo is public and operates on a private vault. The
line is *mechanism in the repo, vocabulary in the vault*: the engine never
carries the user's field names, value words, note titles, schedules or
note paths. The guard makes that line executable without itself becoming
the leak: the word list is a vault file, so the repo holds only the
mechanism that reads it.

**When to use:** it runs on its own on every commit, commit message and
push once the hooks are installed. Run it by hand before a large merge or
after editing the word list:

```bash
python3 scripts/check_private_vocab.py --all
```

## How it works

**The word list** is `<vault>/OpenAugi/AGENT/private-vocabulary.txt`, found
through the openaugi config's `[vault] default_path` (or `--denylist PATH`,
or `$OPENAUGI_PRIVATE_VOCAB`). One entry per line, matched
case-insensitively as a plain substring; a line starting with `re:` is a
regex; blank lines and `#` comments are ignored. It is the user's file: add
a word the moment you coin one for a personal schema.

**The checks.** For every file pre-commit hands it (the staged files at
commit time, the message file at commit-msg time, the changed files at
push time):

1. any line matching a word from the list fails the commit. The offending
   line is printed with the match masked (`▮▮▮`), so the hook's own output
   never repeats the word;
2. any `.ipynb` whose cells have non-empty `outputs` fails the commit.
   Notebook outputs are how real vault text once reached the history.
   Clear outputs before committing.

Binary files and `docs/scratch/` are skipped. When no word list can be
found (a fresh clone, CI) the vocabulary check is skipped with a one-line
notice on stderr and the notebook check still runs.

**Install.** `.pre-commit-config.yaml` declares the hook for the
`pre-commit`, `commit-msg` and `pre-push` stages. `pre-commit install`
covers the first; the other two need their hook types once per clone:

```bash
.venv/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

`scripts/check.sh` runs every hook over all files, so the guard is part of
the pre-push habit as well.

**What it does not do.** A word list catches vocabulary that has been
named. It cannot recognise a paragraph of journal text it has never seen.
The rest of the line is judgment, and AGENTS.md ("Where things live")
spells it out: anything that is *output from* the vault stays out of the
repo.

Tests: `tests/test_check_private_vocab.py` (a throwaway word list;
masking; notebook outputs; missing list; `--all`).
