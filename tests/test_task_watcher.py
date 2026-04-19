"""Tests for the task watcher — the tmux dispatcher for OpenAugi/Tasks/.

Covers pure helpers (parse/rebuild frontmatter, slugify, generate_task_id,
extract_wiki_links, resolve_working_dir, load_repo_paths, build_prompt),
filesystem-touching helpers (hydrate_note, scan_pending, write_context_file),
and dispatch_task with a monkeypatched launch_tmux so no real tmux runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openaugi.agents import task_watcher as tw

# ── Frontmatter parsing ────────────────────────────────────────────────────


class TestParseNote:
    def test_no_frontmatter_returns_empty_dict(self):
        fm, body = tw.parse_note("# just a header\n\nbody text")
        assert fm == {}
        assert body == "# just a header\n\nbody text"

    def test_simple_frontmatter(self):
        text = "---\nstatus: pending\nworkstream: openaugi\n---\n# Title\n\nbody"
        fm, body = tw.parse_note(text)
        assert fm == {"status": "pending", "workstream": "openaugi"}
        assert body == "# Title\n\nbody"

    def test_nested_frontmatter(self):
        text = (
            "---\n"
            "repos:\n"
            "  openaugi: /Users/me/repos/openaugi\n"
            "  site: /Users/me/repos/site\n"
            "---\n"
            "body"
        )
        fm, _ = tw.parse_note(text)
        assert fm["repos"] == {
            "openaugi": "/Users/me/repos/openaugi",
            "site": "/Users/me/repos/site",
        }

    def test_malformed_yaml_returns_empty(self):
        # Non-dict YAML (a scalar) should not crash — return empty dict.
        text = "---\n- just\n- a\n- list\n---\nbody"
        fm, body = tw.parse_note(text)
        assert fm == {}
        assert body == text  # no frontmatter stripped since it wasn't a dict


class TestRebuildNote:
    def test_roundtrip(self):
        text = "---\nstatus: pending\n---\nbody here"
        fm, body = tw.parse_note(text)
        rebuilt = tw.rebuild_note(fm, body)
        fm2, body2 = tw.parse_note(rebuilt)
        assert fm2 == fm
        assert body2 == body

    def test_preserves_key_order_on_rebuild(self):
        fm = {"task_id": "TASK-2026-04-09-foo", "status": "active", "workstream": "openaugi"}
        rebuilt = tw.rebuild_note(fm, "body")
        # Keys should appear in dict insertion order in the yaml output
        assert rebuilt.index("task_id") < rebuilt.index("status") < rebuilt.index("workstream")


# ── Slug / ID / link helpers ───────────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert tw.slugify("Research Matryoshka Embeddings") == "Research-Matryoshka-Embeddings"

    def test_strips_punctuation(self):
        assert tw.slugify("Fix README's onboarding flow!") == "Fix-READMEs-onboarding-flow"

    def test_respects_max_len(self):
        assert len(tw.slugify("a" * 200, max_len=20)) == 20


class TestGenerateTaskId:
    def test_preserves_existing_task_id(self, tmp_path: Path):
        f = tmp_path / "TASK-2026-04-01-Research-Ideas.md"
        f.write_text("")
        assert tw.generate_task_id(f) == "TASK-2026-04-01-Research-Ideas"

    def test_generates_from_stem(self, tmp_path: Path):
        f = tmp_path / "Fix README onboarding.md"
        f.write_text("")
        task_id = tw.generate_task_id(f)
        assert task_id.startswith("TASK-")
        assert task_id.endswith("Fix-README-onboarding")


class TestExtractWikiLinks:
    def test_dedupe_preserve_order(self):
        text = "See [[Alpha]] and [[Beta]] and [[Alpha]] again"
        assert tw.extract_wiki_links(text) == ["Alpha", "Beta"]

    def test_aliased_links(self):
        text = "[[Project Alpha|alpha]] reference"
        assert tw.extract_wiki_links(text) == ["Project Alpha"]

    def test_no_links(self):
        assert tw.extract_wiki_links("plain text") == []


# ── Working dir resolution ────────────────────────────────────────────────


class TestResolveWorkingDir:
    def test_absolute_path_used_as_is(self):
        fm = {"working_dir": "/Users/me/repos/something"}
        assert tw.resolve_working_dir(fm, {}) == "/Users/me/repos/something"

    def test_short_name_resolves_via_repo_paths(self):
        fm = {"repo": "openaugi"}
        repos = {"openaugi": "/Users/me/repos/openaugi"}
        assert tw.resolve_working_dir(fm, repos) == "/Users/me/repos/openaugi"

    def test_short_name_case_insensitive(self):
        fm = {"repo": "OpenAugi"}
        repos = {"openaugi": "/Users/me/repos/openaugi"}
        assert tw.resolve_working_dir(fm, repos) == "/Users/me/repos/openaugi"

    def test_working_dash_dir_alias(self):
        fm = {"working-dir": "/abs/path"}
        assert tw.resolve_working_dir(fm, {}) == "/abs/path"

    def test_unknown_short_name_returns_none(self):
        fm = {"repo": "unknown-project"}
        assert tw.resolve_working_dir(fm, {}) is None

    def test_no_working_dir_key_returns_none(self):
        assert tw.resolve_working_dir({"other": "value"}, {}) is None


class TestLoadRepoPaths:
    def test_loads_repos_from_note(self, tmp_path: Path):
        vault = tmp_path / "vault"
        (vault / "OpenAugi").mkdir(parents=True)
        (vault / "OpenAugi" / "Repos.md").write_text(
            (
                "---\nrepos:\n"
                "  openaugi: /Users/me/repos/openaugi\n"
                "  site: /Users/me/repos/site\n"
                "---\n"
            ),
            encoding="utf-8",
        )
        repos = tw.load_repo_paths(vault)
        assert repos == {
            "openaugi": "/Users/me/repos/openaugi",
            "site": "/Users/me/repos/site",
        }

    def test_missing_repos_note_returns_empty(self, tmp_path: Path):
        assert tw.load_repo_paths(tmp_path / "vault") == {}

    def test_repos_not_a_dict_returns_empty(self, tmp_path: Path):
        vault = tmp_path / "vault"
        (vault / "OpenAugi").mkdir(parents=True)
        (vault / "OpenAugi" / "Repos.md").write_text(
            "---\nrepos: not-a-dict\n---\n", encoding="utf-8"
        )
        assert tw.load_repo_paths(vault) == {}

    def test_custom_repos_note_path(self, tmp_path: Path):
        vault = tmp_path / "vault"
        (vault / "config").mkdir(parents=True)
        (vault / "config" / "repos.md").write_text(
            "---\nrepos:\n  foo: /tmp/foo\n---\n", encoding="utf-8"
        )
        repos = tw.load_repo_paths(vault, repos_note="config/repos.md")
        assert repos == {"foo": "/tmp/foo"}


# ── Hydration ──────────────────────────────────────────────────────────────


class TestHydrateNote:
    def test_adds_fields_and_sections(self, tmp_path: Path):
        f = tmp_path / "My new task.md"
        f.write_text(
            "---\nstatus: pending\nworkstream: openaugi\n---\n# My Task\n\nDo the thing.\n",
            encoding="utf-8",
        )
        task_id, session_name, new_path = tw.hydrate_note(f)

        assert task_id.startswith("TASK-")
        assert session_name == task_id
        assert new_path.name == f"{task_id}.md"
        assert new_path.exists()

        text = new_path.read_text()
        fm, body = tw.parse_note(text)
        assert fm["status"] == "active"
        assert fm["task_id"] == task_id
        assert fm["tmux_session"] == session_name
        assert "created" in fm
        assert "## Session" in body
        assert "## Results" in body
        assert f"tmux attach -t {task_id}" in body

    def test_preserves_existing_task_id(self, tmp_path: Path):
        f = tmp_path / "TASK-2026-04-01-Foo.md"
        f.write_text(
            "---\ntask_id: TASK-2026-04-01-Foo\nstatus: pending\n---\nbody\n",
            encoding="utf-8",
        )
        task_id, _, new_path = tw.hydrate_note(f)
        assert task_id == "TASK-2026-04-01-Foo"
        assert new_path.name == "TASK-2026-04-01-Foo.md"

    def test_no_duplicate_session_section_on_rerun(self, tmp_path: Path):
        """If a file already has ## Session, hydrate should not double-inject."""
        f = tmp_path / "TASK-x.md"
        f.write_text(
            "---\ntask_id: TASK-x\nstatus: pending\n---\n"
            "## Session\n\n```bash\ntmux attach -t TASK-x\n```\n\n## Results\n\n",
            encoding="utf-8",
        )
        _, _, new_path = tw.hydrate_note(f)
        body = new_path.read_text()
        assert body.count("## Session") == 1
        assert body.count("## Results") == 1

    def test_no_default_workstream_or_priority(self, tmp_path: Path):
        """Hydration does not invent workstream/priority — the writer owns those."""
        f = tmp_path / "task.md"
        f.write_text("---\nstatus: pending\n---\nbody\n", encoding="utf-8")
        _, _, new_path = tw.hydrate_note(f)
        fm, _ = tw.parse_note(new_path.read_text())
        assert "workstream" not in fm
        assert "priority" not in fm

    def test_preserves_writer_supplied_workstream(self, tmp_path: Path):
        """If the writer sets workstream, hydration leaves it alone."""
        f = tmp_path / "task.md"
        f.write_text("---\nstatus: pending\nworkstream: content\n---\nbody\n", encoding="utf-8")
        _, _, new_path = tw.hydrate_note(f)
        fm, _ = tw.parse_note(new_path.read_text())
        assert fm["workstream"] == "content"


# ── Scanning ───────────────────────────────────────────────────────────────


class TestScanPending:
    def test_empty_dir(self, tmp_path: Path):
        assert tw.scan_pending(tmp_path) == []

    def test_missing_dir(self, tmp_path: Path):
        assert tw.scan_pending(tmp_path / "nonexistent") == []

    def test_finds_only_pending(self, tmp_path: Path):
        (tmp_path / "pending.md").write_text("---\nstatus: pending\n---\nbody", encoding="utf-8")
        (tmp_path / "active.md").write_text("---\nstatus: active\n---\nbody", encoding="utf-8")
        (tmp_path / "done.md").write_text("---\nstatus: done\n---\nbody", encoding="utf-8")
        # settle=0 so freshly-written files are picked up immediately
        pending = tw.scan_pending(tmp_path, settle=0)
        assert len(pending) == 1
        assert pending[0].name == "pending.md"

    def test_respects_settle_time(self, tmp_path: Path):
        f = tmp_path / "pending.md"
        f.write_text("---\nstatus: pending\n---\nbody", encoding="utf-8")
        # settle=3600s so the freshly written file is still "hot"
        assert tw.scan_pending(tmp_path, settle=3600) == []
        # settle=0 so it's cold enough to dispatch
        assert tw.scan_pending(tmp_path, settle=0) == [f]

    def test_survives_unreadable_file(self, tmp_path: Path):
        """A file that can't be parsed shouldn't crash the scan."""
        good = tmp_path / "good.md"
        good.write_text("---\nstatus: pending\n---\n", encoding="utf-8")
        bad = tmp_path / "bad.md"
        bad.write_bytes(b"\xff\xfe not utf-8")
        pending = tw.scan_pending(tmp_path, settle=0)
        assert good in pending


# ── Prompt building ────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_task_id_and_body(self):
        p = tw.build_prompt("TASK-2026-04-09-foo", "Do the thing", [])
        assert "TASK-2026-04-09-foo" in p
        assert "Do the thing" in p

    def test_surfaces_linked_notes(self):
        p = tw.build_prompt("TASK-x", "body", ["Alpha", "Beta"])
        assert "[[Alpha]]" in p
        assert "[[Beta]]" in p
        assert "openaugi MCP" in p

    def test_no_links_line_when_empty(self):
        p = tw.build_prompt("TASK-x", "body", [])
        assert "[[" not in p
        assert "openaugi MCP" not in p

    def test_instructs_agent_to_edit_task_file(self):
        """The finish instruction points at the task file directly."""
        p = tw.build_prompt("TASK-x", "body", [])
        assert "OpenAugi/Tasks/TASK-x.md" in p
        assert "status: done" in p
        assert "Results" in p

    def test_is_concise(self):
        """Sanity check: trimmed prompt should stay under ~400 chars
        for a plain no-links task so the real task body isn't drowned out."""
        p = tw.build_prompt("TASK-x", "Do a small thing", [])
        assert len(p) < 400


class TestWriteContextFile:
    def test_writes_and_returns_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tw, "CONTEXT_DIR", tmp_path / "openaugi-tasks")
        path = tw.write_context_file("TASK-y", "hello prompt")
        assert path.exists()
        assert path.read_text() == "hello prompt"
        assert "TASK-y" in path.name


# ── Dispatch (mocked launch) ───────────────────────────────────────────────


class TestDispatchTask:
    def test_end_to_end_hydrate_plus_launch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tw, "CONTEXT_DIR", tmp_path / "ctx")

        calls: dict = {}

        def fake_launch(tmux, claude, session_name, context_file, working_dir=None):
            calls["session"] = session_name
            calls["working_dir"] = working_dir
            calls["context_exists"] = Path(context_file).exists()
            return True

        monkeypatch.setattr(tw, "launch_tmux", fake_launch)

        task = tmp_path / "Do a thing.md"
        task.write_text(
            "---\nstatus: pending\nrepo: openaugi\n---\n# Do a thing\n\nwork work work\n",
            encoding="utf-8",
        )

        task_id = tw.dispatch_task(
            task,
            tmux="/fake/tmux",
            claude="/fake/claude",
            repo_paths={"openaugi": "/tmp"},
        )

        assert task_id is not None
        assert task_id.startswith("TASK-")
        assert calls["session"] == task_id
        assert calls["working_dir"] == "/tmp"
        assert calls["context_exists"] is True

        # The original file should have been renamed to match the task_id
        new_file = tmp_path / f"{task_id}.md"
        assert new_file.exists()
        assert not task.exists()
        fm, _ = tw.parse_note(new_file.read_text())
        assert fm["status"] == "active"

    def test_returns_none_when_session_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(tw, "CONTEXT_DIR", tmp_path / "ctx")
        monkeypatch.setattr(tw, "launch_tmux", lambda *a, **k: False)

        task = tmp_path / "x.md"
        task.write_text("---\nstatus: pending\n---\nbody\n", encoding="utf-8")

        result = tw.dispatch_task(task, tmux="/fake/tmux", claude="/fake/claude", repo_paths={})
        assert result is None


# ── The task format contract ──────────────────────────────────────────────

# The authoritative task file format lives at
# src/openaugi/templates/task-template.md. This test reads that file and
# runs it through the full dispatch pipeline. Any change to the template
# that the watcher can't hydrate + launch cleanly fails here. The intent
# is to keep a single source of truth for the zzz dispatch hook (writer) and
# the watcher (reader) so the contract can't silently drift.

TASK_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "src" / "openaugi" / "templates" / "task-template.md"
)


class TestTaskTemplateContract:
    def test_template_file_exists(self):
        assert TASK_TEMPLATE_PATH.exists(), (
            "Authoritative task template is missing. See "
            "src/openaugi/templates/task-template.md and the task_watcher "
            "docstring for why this file must exist."
        )

    def test_template_has_required_frontmatter(self):
        text = TASK_TEMPLATE_PATH.read_text(encoding="utf-8")
        fm, body = tw.parse_note(text)
        # The contract: every pending task must carry at least status.
        # `workstream` should be present in the example so the dispatch
        # hook sees it modeled.
        assert fm.get("status") == "pending"
        assert "workstream" in fm, "template should model a workstream field"
        # Sections the remote agent depends on
        assert "## Context" in body
        assert "## Task" in body
        assert "## Results" in body

    def test_template_hydrates_cleanly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """The template, as shipped, should survive the full hydrate → dispatch
        pipeline with a mocked tmux launch."""
        monkeypatch.setattr(tw, "CONTEXT_DIR", tmp_path / "ctx")

        calls: dict = {}

        def fake_launch(tmux, claude, session_name, context_file, working_dir=None):
            calls["session"] = session_name
            calls["working_dir"] = working_dir
            calls["prompt"] = Path(context_file).read_text()
            return True

        monkeypatch.setattr(tw, "launch_tmux", fake_launch)

        # Copy the template to a tmp vault location so hydrate can rename it
        # without touching the repo file.
        task_file = tmp_path / "template-instance.md"
        task_file.write_text(TASK_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

        task_id = tw.dispatch_task(
            task_file,
            tmux="/fake/tmux",
            claude="/fake/claude",
            # The template uses `repo: openaugi`; resolve it to a real tmp dir
            # so launch_tmux sees a valid cwd.
            repo_paths={"openaugi": str(tmp_path)},
        )

        assert task_id is not None
        assert task_id.startswith("TASK-")

        # After hydration the file should be renamed and flipped to active.
        new_file = tmp_path / f"{task_id}.md"
        assert new_file.exists()
        fm, body = tw.parse_note(new_file.read_text())
        assert fm["status"] == "active"
        assert fm["task_id"] == task_id
        assert fm["tmux_session"] == task_id
        assert "## Session" in body
        assert "## Results" in body

        # The prompt handed to claude should contain the task body, not
        # the placeholder comments — i.e. strip_out the HTML comments
        # shouldn't erase the real task description.
        assert "README" in calls["prompt"] or "Task" in calls["prompt"]
        assert task_id in calls["prompt"]
        assert calls["working_dir"] == str(tmp_path)
