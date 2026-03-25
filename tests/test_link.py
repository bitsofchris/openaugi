"""Tests for Link model."""

from openaugi.model.link import Link


class TestLinkModel:
    def test_create_link(self):
        lnk = Link(from_id="a", to_id="b", kind="split_from")
        assert lnk.from_id == "a"
        assert lnk.to_id == "b"
        assert lnk.kind == "split_from"
        assert lnk.weight is None

    def test_link_with_weight(self):
        lnk = Link(from_id="a", to_id="b", kind="tagged", weight=1.0)
        assert lnk.weight == 1.0

    def test_metadata_json_empty(self):
        lnk = Link(from_id="a", to_id="b", kind="tagged")
        assert lnk.metadata_json() == "{}"

    def test_metadata_json_with_data(self):
        lnk = Link(from_id="a", to_id="b", kind="links_to", metadata={"alias": "foo"})
        assert '"alias"' in lnk.metadata_json()
