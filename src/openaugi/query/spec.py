"""QuerySpec — the serializable query object the engine executes.

Captures exactly the `search` tool surface. Mode is derived, never stored:
title → title search, keyword → FTS, query → semantic, none of those →
browse. The same object round-trips to JSON/YAML and is the saved-query
file format, so an ad-hoc query and a saved one are indistinguishable to
the engine.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Mode = Literal["title", "keyword", "semantic", "browse"]


class QuerySpec(BaseModel):
    """One query against the knowledge base.

    Field semantics match the MCP `search` tool docstring: `after`/`before`
    compare block_time (content date); `after_ingested` compares ingest
    time (the review-queue axis); `has_task` keeps user-marked tasks and
    always excludes layer/bronze; `exclude_path_prefix` drops blocks whose
    source_path starts with the prefix.
    """

    query: str | None = None
    keyword: str | None = None
    title: str | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    after_ingested: str | None = None
    kind: str | None = None
    source: str | None = None
    exclude_path_prefix: str | None = None
    has_task: bool | None = None
    k: int = 100
    offset: int = 0

    @property
    def mode(self) -> Mode:
        """Dispatch precedence: title > keyword > semantic > browse."""
        if self.title:
            return "title"
        if self.keyword:
            return "keyword"
        if self.query:
            return "semantic"
        return "browse"

    def is_empty(self) -> bool:
        """True when nothing was asked: no text mode and no filter.

        `has_task=False` counts as not provided (matches the historical
        `any([...])` truthiness check in the MCP tool).
        """
        return not (
            self.query
            or self.keyword
            or self.title
            or any(
                [
                    self.tags,
                    self.after,
                    self.before,
                    self.after_ingested,
                    self.kind,
                    self.source,
                    self.has_task,
                ]
            )
        )
