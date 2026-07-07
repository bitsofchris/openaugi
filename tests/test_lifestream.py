"""Tests for the lifestream renderer (M6 first screen)."""

import json
import re
from pathlib import Path

import pytest

from openaugi.model.block import Block
from openaugi.render.lifestream import build_lifestream_data, render_lifestream
from openaugi.store.sqlite import SQLiteStore


def _blk(content: str, path: str, when: str, tags: list[str] | None = None, augi=None) -> Block:
    meta = {"source_path": path}
    if augi:
        meta["augi_tags"] = augi
    return Block(
        id=Block.make_id(path, Block.hash_content(content)),
        kind="data_block",
        content=content,
        source="vault",
        block_time=when,
        tags=tags or [],
        metadata=meta,
    )


@pytest.fixture
def populated(store: SQLiteStore) -> SQLiteStore:
    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    store.insert_blocks(
        [
            _blk("thought about capture", "Daily/a.md", today, tags=["area/openaugi"]),
            _blk("workout log", "Daily/b.md", today, augi=["#area/self", "type/log"]),
            _blk("derived thing", "OpenAugi/Views/x.md", today),
            _blk("ancient thought", "Daily/old.md", "2019-01-01"),
            _blk("x" * 500, "Daily/long.md", today),
        ]
    )
    return store


class TestBuildLifestreamData:
    def test_shape_and_exclusions(self, populated: SQLiteStore):
        data = build_lifestream_data(populated, days=30)
        srcs = [b["src"] for b in data["blocks"]]
        assert "Daily/a.md" in srcs
        assert "OpenAugi/Views/x.md" not in srcs  # derived excluded
        assert "Daily/old.md" not in srcs  # outside window
        assert data["days"] == 30

    def test_area_from_user_and_augi_tags(self, populated: SQLiteStore):
        data = build_lifestream_data(populated, days=30)
        by_src = {b["src"]: b for b in data["blocks"]}
        assert by_src["Daily/a.md"]["a"] == "openaugi"
        assert by_src["Daily/b.md"]["a"] == "self"  # from augi_tags, # stripped
        assert "type/log" in by_src["Daily/b.md"]["tags"]

    def test_snippet_truncated(self, populated: SQLiteStore):
        data = build_lifestream_data(populated, days=30)
        long = next(b for b in data["blocks"] if b["src"] == "Daily/long.md")
        assert len(long["s"]) <= 240
        assert long["s"].endswith("…")


class TestRenderLifestream:
    def test_writes_selfcontained_html(self, populated: SQLiteStore, tmp_path: Path):
        out = render_lifestream(populated, tmp_path, days=30)
        assert out == tmp_path / "OpenAugi" / "render" / "lifestream.html"
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        # inlined data parses back as JSON
        m = re.search(r"const DATA = (\{.*?\});\n", html, re.DOTALL)
        assert m, "inlined DATA payload not found"
        data = json.loads(m.group(1).replace("<\\/", "</"))
        assert any(b["src"] == "Daily/a.md" for b in data["blocks"])
        assert "</script>" not in json.dumps(data)  # payload can't break the script tag

    def test_explicit_out_path(self, populated: SQLiteStore, tmp_path: Path):
        target = tmp_path / "custom.html"
        out = render_lifestream(populated, tmp_path, days=30, out=target)
        assert out == target
        assert target.exists()
