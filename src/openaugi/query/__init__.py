"""The query layer — one engine, thin adapters.

All deterministic read semantics live here (see
docs/plans/query-layer.md). MCP, HTTP, and the CLI are presentation
adapters over `engine`; `QuerySpec` is the serializable query object and
the saved-query file format.

The boundary rule: this package never imports `mcp.*` or any web
framework. If a piece of code needs those, it's presentation and belongs
in an adapter.
"""

from openaugi.query.spec import QuerySpec

__all__ = ["QuerySpec"]
