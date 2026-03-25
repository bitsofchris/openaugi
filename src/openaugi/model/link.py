"""Link — edges between blocks in the knowledge graph.

Structure lives in the links, not in the schema.
Link kinds: split_from, tagged, links_to, member_of, summarizes, ...

See docs/plans/m0.md for the canonical schema.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Link(BaseModel):
    """An edge between two blocks.

    Composite primary key: (from_id, to_id, kind).
    """

    from_id: str
    to_id: str
    kind: str  # split_from, tagged, links_to, member_of, summarizes, ...
    weight: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow)

    model_config = {"frozen": False}

    def metadata_json(self) -> str:
        """Serialize metadata to JSON string for storage."""
        return json.dumps(self.metadata) if self.metadata else "{}"
