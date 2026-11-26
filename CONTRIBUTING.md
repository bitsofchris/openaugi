# Contributing to OpenAugi

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/openaugi/openaugi.git
cd openaugi
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Development

```bash
# Run tests
.venv/bin/python -m pytest tests/ -v

# Lint
.venv/bin/ruff check src tests

# Type check
.venv/bin/pyright src
```

## Testing ingest and the graph

Anything that writes blocks or links keeps the graph equal to what the vault
says. Two rules (details: [docs/reference/graph-integrity.md](docs/reference/graph-integrity.md)):

- **Integrity is checked for you.** Every test that ingests into the `store`
  fixture ends with `assert_graph_integrity` (`tests/graph_integrity.py`). If
  your test builds a drifted graph on purpose, mark it
  `@pytest.mark.graph_drift_expected(reason=...)`.
- **Incremental equals full.** A new lifecycle case or wikilink form is a new
  row in `tests/test_graph_integrity.py` (`SCENARIOS`, `LINK_FORMS`), not a
  one-off test. Known-open drift is `xfail(strict=True)` with its reason;
  remove the marker in the change that fixes it.

## Pull Requests

- One focused change per PR
- Include tests for new functionality
- Run `pytest` and `ruff check` before submitting
- Keep PRs small — easier to review, faster to merge

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the system is structured.

## Docs

Docs live in `docs/`; `docs/scratch/` is gitignored personal scratch space
and should never be committed to. Keep example content in docs generic
(category/tag names are fine) — don't include real personal notes, journal
content, or other output from anyone's actual "second brain" data.

## Code Style

- Python 3.12+
- Ruff for linting and formatting
- Pyright for type checking
- Pydantic for data models
- Keep it simple — no premature abstraction
