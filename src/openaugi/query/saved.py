"""Saved queries — named QuerySpecs as markdown files in the vault.

The lens principle applied to retrieval (docs/plans/query-layer.md §2):
a saved query is a named, described QuerySpec living at
`<vault>/OpenAugi/AGENT/queries/<name>.md`, beside AGENT/lenses/. The
vault filesystem is the API — users edit these in Obsidian, agents and
UIs execute them by name, and views-as-rendered-queries step 5 converges
lenses onto this format.

File format (frontmatter holds the machine part, body is prose):

    ---
    description: Open tasks from the last two weeks.
    query:
      has_task: true
      after: "-14d"
    ---
    Optional notes about when to use this query.

Relative-date tokens keep a saved query meaningful over time; they are
resolved at RUN time, never stored resolved:

- `today`                → today's date (after/before/after_ingested)
- `-<N>d` (e.g. "-14d")  → N days before today
- `$review-mark`         → the review-pass high-water mark (after_ingested
  only). If no pass has run yet, resolves to the epoch — first run is a
  full backfill, matching the review-pass docstring.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ValidationError

from openaugi.query.spec import QuerySpec

if TYPE_CHECKING:
    from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

QUERIES_SUBDIR = Path("OpenAugi") / "AGENT" / "queries"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_RELATIVE_DAYS_RE = re.compile(r"^-(\d+)d$")
_TOKEN_FIELDS = ("after", "before", "after_ingested")
REVIEW_MARK_TOKEN = "$review-mark"
_EPOCH = "1970-01-01T00:00:00Z"


class SavedQueryError(ValueError):
    """A saved-query file exists but can't be parsed into a QuerySpec."""


class SavedQueryNotFound(LookupError):
    """No saved query with that name."""


class SavedQuery(BaseModel):
    name: str
    description: str = ""
    spec: QuerySpec  # tokens unresolved — resolve_spec() at run time


def queries_dir(vault_path: str | Path) -> Path:
    return Path(vault_path) / QUERIES_SUBDIR


def list_saved(vault_path: str | Path) -> list[SavedQuery]:
    """Every parseable saved query, sorted by name.

    Unparseable files are skipped with a warning — the vault is
    user-edited, so the reader stays lenient (same posture as every
    other file contract).
    """
    folder = queries_dir(vault_path)
    if not folder.is_dir():
        return []
    out: list[SavedQuery] = []
    for path in sorted(folder.glob("*.md")):
        try:
            out.append(_parse_file(path))
        except SavedQueryError as e:
            logger.warning("Skipping saved query %s: %s", path.name, e)
    return out


def load_saved(vault_path: str | Path, name: str) -> SavedQuery:
    """Load one saved query by name (filename stem)."""
    path = queries_dir(vault_path) / f"{name}.md"
    if not path.is_file():
        raise SavedQueryNotFound(name)
    return _parse_file(path)


def resolve_spec(
    spec: QuerySpec,
    store: SQLiteStore | None = None,
    today: date | None = None,
) -> QuerySpec:
    """Resolve relative-date tokens into concrete values.

    Pure given (spec, today) except `$review-mark`, which reads the
    review-pass state from the store. Returns a new QuerySpec; the input
    (and the file it came from) keeps its tokens.
    """
    today = today or date.today()
    updates: dict[str, str] = {}
    for field in _TOKEN_FIELDS:
        value = getattr(spec, field)
        if value is None:
            continue
        if value == REVIEW_MARK_TOKEN:
            if field != "after_ingested":
                raise SavedQueryError(f"{REVIEW_MARK_TOKEN} is only valid for after_ingested")
            if store is None:
                raise SavedQueryError(f"{REVIEW_MARK_TOKEN} needs a store to resolve")
            mark = store.get_review_state().get("last_run")
            # No pass yet → epoch: the first run is a full backfill.
            updates[field] = mark or _EPOCH
            continue
        if value == "today":
            updates[field] = today.isoformat()
            continue
        match = _RELATIVE_DAYS_RE.match(value)
        if match:
            updates[field] = (today - timedelta(days=int(match.group(1)))).isoformat()
    return spec.model_copy(update=updates) if updates else spec


def _parse_file(path: Path) -> SavedQuery:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SavedQueryError("missing frontmatter")
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        raise SavedQueryError(f"bad YAML: {e}") from e
    if not isinstance(data, dict):
        raise SavedQueryError("frontmatter must be a mapping")

    raw_spec = data.get("query")
    if not isinstance(raw_spec, dict):
        raise SavedQueryError("frontmatter needs a `query:` mapping (the QuerySpec)")
    try:
        spec = QuerySpec.model_validate({k: _coerce_scalar(v) for k, v in raw_spec.items()})
    except ValidationError as e:
        raise SavedQueryError(f"invalid QuerySpec: {e}") from e

    return SavedQuery(
        name=path.stem,
        description=str(data.get("description", "")),
        spec=spec,
    )


def _coerce_scalar(value):
    """YAML eagerly parses bare dates — QuerySpec wants ISO strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
