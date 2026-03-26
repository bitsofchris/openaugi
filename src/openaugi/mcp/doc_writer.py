"""VaultWriter — write markdown documents to the OpenAugi folder in the vault.

All agent writes are scoped to `{vault_path}/OpenAugi/{subfolder}/`.
The subfolder is agent-chosen (e.g. "Docs", "Notes", "Research") but
cannot escape the OpenAugi root — no `..`, no absolute paths.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Characters not valid in Obsidian note titles
_INVALID_TITLE_RE = re.compile(r'[/\\:*?"<>|#^[\]{}]')
_OPENAUGI_ROOT = "OpenAugi"


class VaultWriter:
    """Write markdown documents to {vault_path}/OpenAugi/{subfolder}/."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)

    def write_document(
        self,
        title: str,
        description: str,
        content: str,
        subfolder: str = "Notes",
    ) -> dict:
        """Create a markdown note in OpenAugi/{subfolder}/.

        Args:
            title: Note title (becomes filename). Must be valid Obsidian title.
            description: One-line summary — goes in frontmatter, used for scanning.
            content: Markdown body. Frontmatter is auto-generated.
            subfolder: Subfolder under OpenAugi/ (e.g. "Notes", "Docs", "Research").
                       Defaults to "Notes". Cannot escape OpenAugi/ root.

        Returns:
            {"status": "created", "path": str, "vault_relative": str, "title": str}
            {"status": "error", "reason": str}
        """
        title = title.strip()
        if not title:
            return {"status": "error", "reason": "Title cannot be empty"}
        if _INVALID_TITLE_RE.search(title):
            invalid = _INVALID_TITLE_RE.findall(title)
            return {"status": "error", "reason": f"Title contains invalid characters: {invalid}"}

        folder = self._resolve_folder(subfolder)
        if folder is None:
            return {
                "status": "error",
                "reason": (
                    f"Invalid subfolder '{subfolder}'. "
                    "Must be a relative path with no '..' components."
                ),
            }

        filepath = folder / f"{title}.md"
        if filepath.exists():
            return {
                "status": "error",
                "reason": f"Note already exists: {filepath.relative_to(self.vault_path)}",
            }

        folder.mkdir(parents=True, exist_ok=True)

        created = datetime.now().isoformat(timespec="seconds")
        note = (
            f"---\n"
            f"type: document\n"
            f"description: {description}\n"
            f"created: {created}\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{content.strip()}\n"
        )
        filepath.write_text(note, encoding="utf-8")
        logger.info("Wrote document: %s", filepath)

        return {
            "status": "created",
            "path": str(filepath),
            "vault_relative": str(filepath.relative_to(self.vault_path)),
            "title": title,
        }

    def write_thread(self, topic: str, description: str, content: str) -> dict:
        """Write a session thread note to OpenAugi/Threads/YYYY-MM-DD - {topic}.md.

        Handles filename collisions by appending -2, -3, etc.

        Returns:
            {"status": "created", "path": str, "vault_relative": str, "title": str}
            {"status": "error", "reason": str}
        """
        topic = topic.strip()
        if not topic:
            return {"status": "error", "reason": "Topic cannot be empty"}
        if _INVALID_TITLE_RE.search(topic):
            invalid = _INVALID_TITLE_RE.findall(topic)
            return {"status": "error", "reason": f"Topic contains invalid characters: {invalid}"}

        folder = self._resolve_folder("Threads")
        if folder is None:
            return {"status": "error", "reason": "Could not resolve Threads folder"}

        folder.mkdir(parents=True, exist_ok=True)

        today = date.today().isoformat()
        base_title = f"{today} - {topic}"
        filepath = self._unique_path(folder, base_title)

        created = datetime.now().isoformat(timespec="seconds")
        note = (
            f"---\n"
            f"type: thread\n"
            f"description: {description}\n"
            f"created: {created}\n"
            f"---\n\n"
            f"# {topic}\n\n"
            f"{content.strip()}\n"
        )
        filepath.write_text(note, encoding="utf-8")
        logger.info("Wrote thread: %s", filepath)

        title = filepath.stem
        return {
            "status": "created",
            "path": str(filepath),
            "vault_relative": str(filepath.relative_to(self.vault_path)),
            "title": title,
        }

    def _unique_path(self, folder: Path, base_title: str) -> Path:
        """Return a non-colliding path, appending -2, -3 as needed."""
        candidate = folder / f"{base_title}.md"
        if not candidate.exists():
            return candidate
        counter = 2
        while True:
            candidate = folder / f"{base_title}-{counter}.md"
            if not candidate.exists():
                return candidate
            counter += 1

    def _resolve_folder(self, subfolder: str) -> Path | None:
        """Resolve subfolder to an absolute path under OpenAugi/.

        Returns None if the subfolder tries to escape (contains '..', is absolute, etc.).
        """
        # Normalize: strip leading slashes, collapse separators
        subfolder = subfolder.strip().strip("/").replace("\\", "/")
        if not subfolder:
            subfolder = "Docs"

        # Reject any path traversal attempts
        parts = Path(subfolder).parts
        if any(p == ".." for p in parts):
            return None
        if Path(subfolder).is_absolute():
            return None

        return self.vault_path / _OPENAUGI_ROOT / subfolder
