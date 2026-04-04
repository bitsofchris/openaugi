"""StreamManager — read/write workstream files in OpenAugi/Streams/.

Workstreams are markdown files with YAML frontmatter that track persistent
threads of work. Each stream has a Context section, LEFT OFF marker, and
an append-only Log.

File format:
    ---
    stream: Display Name
    status: active
    last_active: '2026-04-03'
    linked_sessions:
      - session-id-1
    ---

    ## Context
    ...

    ## LEFT OFF
    ...

    ## Log
    - 2026-04-03: ...
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_OPENAUGI_ROOT = "OpenAugi"
_STREAMS_FOLDER = "Streams"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Convert a display name to a slug: lowercase, hyphens, no special chars."""
    return _SLUG_RE.sub("-", name.lower()).strip("-")


def _parse_stream_file(text: str) -> dict:
    """Parse a stream markdown file into structured data.

    Returns dict with keys: frontmatter (dict), context (str),
    left_off (str), log (str).
    """
    fm: dict = {}
    body = text

    # Extract frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                fm = {}
            body = parts[2]

    # Extract sections
    context = _extract_section(body, "Context")
    left_off = _extract_section(body, "LEFT OFF")
    log = _extract_section(body, "Log")

    return {
        "frontmatter": fm,
        "context": context,
        "left_off": left_off,
        "log": log,
    }


def _extract_section(body: str, heading: str) -> str:
    """Extract content under a ## heading, up to the next ## heading or EOF."""
    pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _serialize_stream(fm: dict, context: str, left_off: str, log: str) -> str:
    """Serialize stream data back to markdown."""
    fm_str = yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
    parts = [
        f"---\n{fm_str}\n---",
        f"\n## Context\n{context}\n" if context else "\n## Context\n\n",
        f"\n## LEFT OFF\n{left_off}\n" if left_off else "\n## LEFT OFF\n\n",
        f"\n## Log\n{log}\n" if log else "\n## Log\n\n",
    ]
    return "".join(parts)


class StreamManager:
    """Manage workstream files in {vault_path}/OpenAugi/Streams/."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.streams_dir = self.vault_path / _OPENAUGI_ROOT / _STREAMS_FOLDER

    def list_streams(self, status: str | None = None) -> dict:
        """List all streams with summary info.

        Args:
            status: Filter by "active" or "done". None for all.

        Returns:
            {"streams": [...], "count": int}
        """
        if not self.streams_dir.exists():
            return {"streams": [], "count": 0}

        streams = []
        for path in sorted(self.streams_dir.glob("*.md")):
            parsed = _parse_stream_file(path.read_text(encoding="utf-8"))
            fm = parsed["frontmatter"]

            if status and fm.get("status") != status:
                continue

            left_off = parsed["left_off"]
            streams.append(
                {
                    "slug": path.stem,
                    "stream": fm.get("stream", path.stem),
                    "status": fm.get("status", "active"),
                    "last_active": fm.get("last_active", ""),
                    "linked_sessions": fm.get("linked_sessions", []),
                    "left_off_preview": left_off[:150] if left_off else "",
                }
            )

        # Sort: active first, then by last_active descending
        streams.sort(
            key=lambda s: (s["status"] != "active", s["last_active"] or ""), reverse=False
        )
        streams.sort(key=lambda s: s["status"] != "active")

        return {"streams": streams, "count": len(streams)}

    def get_stream_context(self, stream: str) -> dict:
        """Load full stream state for resuming work.

        Args:
            stream: Slug or display name (tries slug first, then fuzzy match).

        Returns:
            Full stream data or {"status": "error", "reason": "..."}.
        """
        path = self._find_stream(stream)
        if path is None:
            return {"status": "error", "reason": f"Stream not found: {stream}"}

        parsed = _parse_stream_file(path.read_text(encoding="utf-8"))
        fm = parsed["frontmatter"]

        return {
            "slug": path.stem,
            "stream": fm.get("stream", path.stem),
            "status": fm.get("status", "active"),
            "last_active": fm.get("last_active", ""),
            "linked_sessions": fm.get("linked_sessions", []),
            "context": parsed["context"],
            "left_off": parsed["left_off"],
            "log": parsed["log"],
        }

    def make_stream(
        self,
        name: str,
        context: str = "",
        status: str = "active",
    ) -> dict:
        """Create a new workstream.

        Args:
            name: Display name (e.g. "Product Management").
            context: Initial Context section content.
            status: "active" or "done".

        Returns:
            {"status": "created", "path": str, "slug": str}
            {"status": "error", "reason": str}
        """
        name = name.strip()
        if not name:
            return {"status": "error", "reason": "Name cannot be empty"}

        slug = slugify(name)
        if not slug:
            return {"status": "error", "reason": f"Could not generate slug from name: {name}"}

        self.streams_dir.mkdir(parents=True, exist_ok=True)

        filepath = self.streams_dir / f"{slug}.md"
        if filepath.exists():
            return {"status": "error", "reason": f"Stream '{slug}' already exists"}

        fm = {
            "stream": name,
            "status": status,
            "last_active": date.today().isoformat(),
            "linked_sessions": [],
        }

        content = _serialize_stream(fm, context, "", "")
        filepath.write_text(content, encoding="utf-8")
        logger.info("Created stream: %s", filepath)

        return {
            "status": "created",
            "path": str(filepath),
            "vault_relative": str(filepath.relative_to(self.vault_path)),
            "slug": slug,
        }

    def update_stream(
        self,
        stream: str,
        left_off: str | None = None,
        context: str | None = None,
        log: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Update a stream. All params optional — does whatever you pass it.

        Args:
            stream: Slug or display name.
            left_off: Replace LEFT OFF section content.
            context: Replace Context section content.
            log: Append to Log section (timestamped).
            session_id: Add to linked_sessions (deduped).
            status: Update status ("active" or "done").

        Returns:
            {"status": "updated", "path": str, "slug": str}
            {"status": "error", "reason": str}
        """
        path = self._find_stream(stream)
        if path is None:
            return {"status": "error", "reason": f"Stream not found: {stream}"}

        text = path.read_text(encoding="utf-8")
        parsed = _parse_stream_file(text)
        fm = parsed["frontmatter"]

        if left_off is not None:
            parsed["left_off"] = left_off

        if context is not None:
            parsed["context"] = context

        if log is not None:
            today = date.today().isoformat()
            entry = f"- {today}: {log}"
            if parsed["log"]:
                parsed["log"] = parsed["log"] + "\n" + entry
            else:
                parsed["log"] = entry

        if session_id is not None:
            sessions = fm.get("linked_sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            fm["linked_sessions"] = sessions

        if status is not None:
            fm["status"] = status

        fm["last_active"] = date.today().isoformat()

        content = _serialize_stream(fm, parsed["context"], parsed["left_off"], parsed["log"])
        path.write_text(content, encoding="utf-8")
        logger.info("Updated stream: %s", path)

        return {
            "status": "updated",
            "path": str(path),
            "vault_relative": str(path.relative_to(self.vault_path)),
            "slug": path.stem,
        }

    def append_to_log(self, slug: str, text: str) -> bool:
        """Append a line to a stream's Log section. Returns True if successful."""
        path = self.streams_dir / f"{slug}.md"
        if not path.exists():
            return False

        file_text = path.read_text(encoding="utf-8")
        parsed = _parse_stream_file(file_text)
        fm = parsed["frontmatter"]

        today = date.today().isoformat()
        entry = f"- {today}: {text}"
        if parsed["log"]:
            parsed["log"] = parsed["log"] + "\n" + entry
        else:
            parsed["log"] = entry

        fm["last_active"] = date.today().isoformat()
        content = _serialize_stream(fm, parsed["context"], parsed["left_off"], parsed["log"])
        path.write_text(content, encoding="utf-8")
        return True

    def _find_stream(self, stream: str) -> Path | None:
        """Find a stream file by slug or fuzzy name match.

        Tries exact slug first, then case-insensitive match on stream: frontmatter.
        """
        # Try exact slug
        path = self.streams_dir / f"{stream}.md"
        if path.exists():
            return path

        # Try slugified version
        slug = slugify(stream)
        if slug != stream:
            path = self.streams_dir / f"{slug}.md"
            if path.exists():
                return path

        # Fuzzy match on stream: frontmatter field
        if not self.streams_dir.exists():
            return None

        query = stream.lower()
        for candidate in self.streams_dir.glob("*.md"):
            try:
                parsed = _parse_stream_file(candidate.read_text(encoding="utf-8"))
                name = parsed["frontmatter"].get("stream", "")
                if name.lower() == query:
                    return candidate
            except Exception:
                continue

        return None
