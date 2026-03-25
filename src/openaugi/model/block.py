"""Block — the universal node in the knowledge graph.

Everything is a block: documents, entries, tags, clusters, summaries.
Structure lives in the links, not in the schema.

See docs/plans/m0.md for the canonical schema.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Block(BaseModel):
    """A node in the knowledge graph.

    Kinds: document, entry, tag, cluster, summary, extraction.
    Kind-specific data goes in metadata (JSON).
    """

    id: str
    kind: str  # document, entry, tag, cluster, summary, ...
    content: str | None = None
    summary: str | None = None
    embedding: bytes | None = Field(default=None, exclude=True, repr=False)

    source: str | None = None  # adapter name: "vault", "chatgpt", "pipeline"
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    timestamp: str | None = None  # ISO-8601
    occurred_at: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None
    created_at: str = Field(default_factory=_utcnow)

    model_config = {"frozen": False}

    @staticmethod
    def make_id(source_path: str, content_hash: str) -> str:
        """Deterministic block ID: hash(source_path + content_hash).

        Stable across runs as long as content is identical,
        regardless of section position within the file.
        """
        combined = f"{source_path}:{content_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def hash_content(content: str) -> str:
        """SHA-256 content hash (truncated to 16 hex chars)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def make_tag_id(tag_name: str) -> str:
        """Deterministic ID for a tag block."""
        return hashlib.sha256(f"tag:{tag_name}".encode()).hexdigest()[:16]

    @staticmethod
    def make_document_id(source_path: str) -> str:
        """Deterministic ID for a document block."""
        return hashlib.sha256(f"doc:{source_path}".encode()).hexdigest()[:16]

    @model_validator(mode="after")
    def _compute_content_hash(self) -> Block:
        """Auto-compute content_hash if content is set and hash is missing."""
        if self.content and not self.content_hash:
            self.content_hash = self.hash_content(self.content)
        return self

    def metadata_json(self) -> str:
        """Serialize metadata to JSON string for storage."""
        return json.dumps(self.metadata) if self.metadata else "{}"

    def tags_json(self) -> str:
        """Serialize tags list to JSON string for storage."""
        return json.dumps(self.tags) if self.tags else "[]"
