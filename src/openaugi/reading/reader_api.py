"""Thin Readwise Reader API client — save, list, get. Nothing else.

Scope is deliberately narrow: the two calls the round trip needs
(`POST /save/` to push, `GET /list/` to harvest) plus paging. The token comes
from the `READWISE_TOKEN` environment variable and from nowhere else — a copy
also sits in the Obsidian plugin's `data.json`, and reading it from there would
make our code a second place a credential lives.

API reference: https://readwise.io/reader_api
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from typing import Any, Protocol

logger = logging.getLogger(__name__)

BASE_URL = "https://readwise.io/api/v3"
TOKEN_ENV = "READWISE_TOKEN"


class MissingTokenError(RuntimeError):
    """Raised when READWISE_TOKEN is not in the environment."""


class ReaderAPI(Protocol):
    """What push and harvest need. Tests supply a fake; nothing mocks httpx."""

    def save(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def list_documents(
        self,
        *,
        category: str | None = None,
        updated_after: str | None = None,
        document_id: str | None = None,
    ) -> Iterator[dict[str, Any]]: ...


class ReaderClient:
    """Live client. One httpx client per instance; close it or use `with`."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
    ):
        resolved = token or os.environ.get(TOKEN_ENV, "").strip()
        if not resolved:
            raise MissingTokenError(
                f"{TOKEN_ENV} is not set. Export it (it is already in ~/.zshrc) or pass --token."
            )
        import httpx

        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Token {resolved}"},
        )

    def __enter__(self) -> ReaderClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`POST /save/`. 201 = created, 200 = the same `url` updated in place."""
        response = self._request("POST", "/save/", json=payload)
        return response.json()

    def list_documents(
        self,
        *,
        category: str | None = None,
        updated_after: str | None = None,
        document_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """`GET /list/`, following `nextPageCursor` until exhausted."""
        params: dict[str, str] = {}
        if category:
            params["category"] = category
        if updated_after:
            params["updatedAfter"] = updated_after
        if document_id:
            params["id"] = document_id

        cursor: str | None = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params["pageCursor"] = cursor
            data = self._request("GET", "/list/", params=page_params).json()
            yield from data.get("results", [])
            cursor = data.get("nextPageCursor")
            if not cursor:
                return

    def _request(self, method: str, path: str, **kwargs: Any):
        """One retry on 429, honouring Retry-After. The documented limit is
        50/min for /save/ and 20/min for /list/, and a daily push of two notes
        is nowhere near it — this exists so a burst backs off instead of
        failing the run."""
        url = f"{self._base_url}{path}"
        for attempt in (1, 2):
            response = self._client.request(method, url, **kwargs)
            if response.status_code == 429 and attempt == 1:
                wait = float(response.headers.get("Retry-After", "5"))
                logger.warning("Reader rate limit hit, waiting %.0fs", wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable")
