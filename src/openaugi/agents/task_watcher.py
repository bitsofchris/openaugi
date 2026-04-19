"""Task Watcher — polls OpenAugi/Tasks/ for pending tasks, hydrates them,
and launches Claude Code agents in named tmux sessions.

Workflow:
  1. The file watcher detects vault changes and ingests them. If any block
     carries a `zzz:` instruction, `pipeline/dispatch.py` writes a task
     file to <vault>/OpenAugi/Tasks/ with `status: pending`. Users and
     mobile capture can also write task files directly.
  2. This watcher scans that folder. Each .md file that has been stable
     for `settle` seconds is picked up.
  3. The file is hydrated: `task_id`, `created`, `tmux_session`, and
     `## Session` / `## Results` sections are added. `status` flips to
     `active`. The file is renamed to match the task_id.
  4. The working directory is resolved from `working_dir` / `working-dir`
     / `repo` in frontmatter — either an absolute path, or a short name
     that maps to a path via `OpenAugi/Repos.md`'s `repos:` dict.
  5. A prompt is built: the augi-agent skill file (source of truth for
     agent behavior) + task_id + task body + linked notes.
  6. A detached tmux session is created (with optional `-c <cwd>`), the
     shell is allowed to settle, and the claude CLI is launched via
     send-keys so the user's login-shell PATH and aliases apply.

No LLM calls in this module. Dispatch only.

## Task file format — the contract

The zzz dispatch hook (writer) and this watcher (reader) agree on one file
format. The authoritative, annotated version lives in ONE place:

    src/openaugi/templates/task-template.md

Change that file and `test_task_template_hydrates_cleanly` breaks until
the watcher keeps up. A minimal pending task looks like:

    ---
    status: pending
    source_block_id: <block id>
    source_note: "[[<source note title>]]"
    ---

    # <Human-readable task title>

    ## Context
    <source block content, verbatim>

    ## Task
    <self-contained description of what the remote agent should do>

    ## Human Todo
    <empty; remote agent appends items here if it needs you>

    ## Results
    <empty; remote agent fills this in when finished>

Hydration adds `task_id`, `created`, `tmux_session`, and a `## Session`
block with the `tmux attach` command. It does NOT add defaults for
`workstream`, `priority`, etc. �� those are the writer's responsibility.

## Agent skill

The agent's behavior is governed by `templates/augi-agent.md`, which is
copied to `<vault>/OpenAugi/AGENT/augi-agent.md` on init. Edit the vault
copy to change how the agent handles tasks — not this Python code.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_TASKS_FOLDER = "OpenAugi/Tasks"
DEFAULT_REPOS_NOTE = "OpenAugi/Repos.md"
DEFAULT_POLL_INTERVAL = 5.0  # seconds between scans
DEFAULT_SETTLE_TIME = 30.0  # seconds a file must be unchanged before processing

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Frontmatter keys that all mean "working directory"
WORKING_DIR_KEYS = ("working_dir", "working-dir", "repo")

# Where to look for the tmux and claude binaries if `which` misses them.
TMUX_SEARCH_PATHS = [
    "/opt/homebrew/bin/tmux",  # Apple Silicon Homebrew
    "/usr/local/bin/tmux",  # Intel Homebrew
    "/usr/bin/tmux",  # system install
]
CLAUDE_SEARCH_PATHS = [
    Path.home() / ".claude" / "local" / "claude",
    Path("/usr/local/bin/claude"),
    Path("/opt/homebrew/bin/claude"),
]

# Temp dir for prompt context files passed to claude via `$(cat ...)`
CONTEXT_DIR = Path(tempfile.gettempdir()) / "openaugi-tasks"


# ── Frontmatter helpers ────────────────────────────────────────────────────


def parse_note(text: str) -> tuple[dict, str]:
    """Split note into (frontmatter_dict, body). Returns ({}, text) if none."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    if not isinstance(fm, dict):
        return {}, text
    body = text[m.end() :]
    return fm, body


def rebuild_note(fm: dict, body: str) -> str:
    """Reconstruct a note from frontmatter dict and body."""
    fm_str = yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{fm_str}\n---\n{body}"


# ── Hydration ──────────────────────────────────────────────────────────────


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len]


def generate_task_id(filepath: Path) -> str:
    """Generate TASK-YYYY-MM-DD-Slug from filename. Preserves existing TASK-* ids."""
    stem = filepath.stem
    if stem.startswith("TASK-"):
        return stem
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(stem)
    return f"TASK-{date_str}-{slug}"


def extract_wiki_links(text: str) -> list[str]:
    """Extract [[wiki-link]] targets from text, unique and in order."""
    return list(dict.fromkeys(WIKILINK_RE.findall(text)))


def hydrate_note(filepath: Path) -> tuple[str, str, Path]:
    """Hydrate a pending task note and rename it to TASK-*.

    Adds task_id, created, tmux_session to frontmatter. Sets status=active.
    Injects Session + Results sections if missing. Renames the file to
    match the task_id.

    Returns (task_id, session_name, new_filepath).
    """
    text = filepath.read_text()
    fm, body = parse_note(text)

    task_id = fm.get("task_id") or generate_task_id(filepath)
    session_name = task_id
    now = datetime.now().isoformat(timespec="seconds")

    fm["task_id"] = task_id
    fm["status"] = "active"
    fm.setdefault("created", now)
    fm["tmux_session"] = session_name

    # No default workstream / priority. The writer (zzz dispatch or the
    # user) owns those fields. If they're missing, they stay missing — the
    # watcher logs the task as-is rather than labeling it under a surprise
    # workstream the user didn't choose.

    # Inject Session section if not present
    if "## Session" not in body:
        session_block = f"\n## Session\n\n```bash\ntmux attach -t {session_name}\n```\n"
        if "## Results" in body:
            body = body.replace("## Results", f"{session_block}\n## Results")
        else:
            body = body.rstrip() + f"\n{session_block}\n## Results\n\n"

    # Ensure ## Results exists
    if "## Results" not in body:
        body = body.rstrip() + "\n\n## Results\n\n"

    new_text = rebuild_note(fm, body)

    new_filepath = filepath.parent / f"{task_id}.md"
    if new_filepath != filepath:
        filepath.rename(new_filepath)
    new_filepath.write_text(new_text)

    return task_id, session_name, new_filepath


# ── Repo paths ─────────────────────────────────────────────────────────────


def load_repo_paths(vault: Path, repos_note: str = DEFAULT_REPOS_NOTE) -> dict[str, str]:
    """Load name→path mapping from `<vault>/<repos_note>` frontmatter.

    The note should have YAML frontmatter with a `repos` dict:

        ---
        repos:
          openaugi: /Users/me/repos/openaugi
          my-site: /Users/me/repos/my-site
        ---

    Keys are lowercased so lookups are case-insensitive. Missing note or
    malformed frontmatter → empty dict (warn and continue).
    """
    repos_file = vault / repos_note
    if not repos_file.exists():
        logger.warning("Repos note not found: %s", repos_file)
        return {}
    try:
        text = repos_file.read_text()
        fm, _ = parse_note(text)
        repos = fm.get("repos", {})
        if not isinstance(repos, dict):
            logger.warning("`repos` field in %s is not a dict", repos_file)
            return {}
        return {str(k).lower(): str(v) for k, v in repos.items()}
    except Exception as e:
        logger.error("Error loading repos note %s: %s", repos_file, e)
        return {}


def resolve_working_dir(fm: dict, repo_paths: dict[str, str]) -> str | None:
    """Resolve the working directory for a task from its frontmatter.

    Checks `working_dir`, `working-dir`, and `repo` keys in that order.
    Short names are looked up in `repo_paths` (case-insensitive). Absolute
    paths are used as-is. Anything else is rejected with a warning.
    """
    for key in WORKING_DIR_KEYS:
        value = fm.get(key)
        if not value:
            continue
        value = str(value).strip()
        resolved = repo_paths.get(value.lower())
        if resolved:
            return resolved
        if os.path.isabs(value):
            return value
        logger.warning("Unknown repo '%s' — not in repos note and not an absolute path", value)
        return None
    return None


# ── Binary detection ───────────────────────────────────────────────────────


def detect_tmux() -> str:
    """Find the tmux binary. Raises FileNotFoundError if not installed."""
    import shutil

    found = shutil.which("tmux")
    if found:
        return found
    for p in TMUX_SEARCH_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError("tmux not found. Install with: brew install tmux")


def detect_claude() -> str:
    """Find the claude CLI binary. Raises FileNotFoundError if not installed."""
    import shutil

    found = shutil.which("claude")
    if found:
        return found
    for p in CLAUDE_SEARCH_PATHS:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise FileNotFoundError(
        "claude CLI not found. Install Claude Code from https://claude.com/claude-code"
    )


# ── Agent dispatch ─────────────────────────────────────────────────────────


SKILL_FILE_RELATIVE = "OpenAugi/AGENT/augi-agent.md"


def build_prompt(
    task_id: str,
    body: str,
    links: list[str],
    skill_file: Path | None = None,
) -> str:
    """Build the initial prompt handed to the Claude agent in its tmux session.

    Injects the augi-agent skill file reference so the agent reads it first,
    then provides the task body and a finish instruction.
    """
    lines: list[str] = []

    # Skill file — source of truth for agent behavior
    if skill_file and skill_file.exists():
        lines.append(f"Read your skill file first:\n  {skill_file}\n")

    lines.append(f"You are working on task {task_id}.\n")

    if links:
        joined = ", ".join(f"[[{lnk}]]" for lnk in links)
        lines.append(f"Linked notes (pull via openaugi MCP): {joined}\n")

    lines.append(f"{body.strip()}\n")
    lines.append(
        f"When done, edit OpenAugi/Tasks/{task_id}.md: fill in `## Results` "
        f"and set frontmatter `status: done`.\n"
    )

    return "\n".join(lines)


def write_context_file(task_id: str, prompt: str) -> Path:
    """Write the prompt to a temp file so claude can read it via `$(cat ...)`."""
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTEXT_DIR / f"task-{task_id}-context.md"
    path.write_text(prompt)
    return path


def wait_for_shell_ready(tmux: str, session_name: str, max_attempts: int = 10) -> None:
    """Poll a tmux pane until the shell prompt appears, with a short cap."""
    for _ in range(max_attempts):
        time.sleep(0.2)
        try:
            result = subprocess.run(
                [tmux, "capture-pane", "-t", session_name, "-p"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                return
        except subprocess.SubprocessError:
            pass
    # Proceed anyway — better than hanging forever.


def launch_tmux(
    tmux: str,
    claude: str,
    session_name: str,
    context_file: Path,
    working_dir: str | None = None,
) -> bool:
    """Launch claude in a named detached tmux session.

    Strategy (same as v1 / the openaugi-obsidian-plugin):
      1. Create a detached session with a login shell (no command yet).
      2. Wait for the shell prompt to appear.
      3. Send the claude command via send-keys, using `$(cat file)` so the
         prompt is not limited by argv length.

    This ensures the user's PATH/aliases are available and the session
    survives if the command exits. Returns True if launched, False if the
    session already existed.
    """
    # Check if session already exists
    result = subprocess.run(
        [tmux, "has-session", "-t", session_name],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        logger.warning("tmux session %s already exists, skipping", session_name)
        return False

    new_cmd = [tmux, "new-session", "-d", "-s", session_name]
    if working_dir and os.path.isdir(working_dir):
        new_cmd += ["-c", working_dir]
    subprocess.run(new_cmd, check=True)

    wait_for_shell_ready(tmux, session_name)

    ctx = shlex.quote(str(context_file))
    agent_cmd = f'{claude} "$(cat {ctx})"'
    subprocess.run(
        [tmux, "send-keys", "-t", session_name, agent_cmd, "Enter"],
        check=True,
    )
    return True


# ── Main loop ──────────────────────────────────────────────────────────────


def scan_pending(tasks_dir: Path, settle: float = DEFAULT_SETTLE_TIME) -> list[Path]:
    """Find task files with `status: pending` that have been stable for `settle` seconds.

    Files modified more recently than `settle` seconds ago are skipped —
    this gives Obsidian Sync / file-based capture time to finish writing.
    """
    if not tasks_dir.exists():
        return []
    pending: list[Path] = []
    now = time.time()
    for f in tasks_dir.glob("*.md"):
        try:
            text = f.read_text()
            fm, _ = parse_note(text)
            if fm.get("status") != "pending":
                continue
            age = now - f.stat().st_mtime
            if age < settle:
                logger.debug(
                    "Skipping %s — modified %ds ago (settle=%ds)",
                    f.name,
                    int(age),
                    int(settle),
                )
                continue
            pending.append(f)
        except Exception as e:
            logger.error("Error reading %s: %s", f.name, e)
    return pending


def dispatch_task(
    filepath: Path,
    tmux: str,
    claude: str,
    repo_paths: dict[str, str],
    vault_path: Path | None = None,
) -> str | None:
    """Hydrate a single pending task and launch it in tmux.

    Returns the task_id on success, None on skip (e.g. session already exists).
    Raises on hydrate / launch errors so the caller can record the failure.
    """
    logger.info("Found pending task: %s", filepath.name)

    task_id, session_name, new_path = hydrate_note(filepath)
    logger.info("Hydrated → %s", task_id)

    text = new_path.read_text()
    fm, body = parse_note(text)
    links = extract_wiki_links(body)

    # Resolve the augi-agent skill file
    skill_file = None
    if vault_path:
        candidate = vault_path / SKILL_FILE_RELATIVE
        if candidate.exists():
            skill_file = candidate

    prompt = build_prompt(task_id, body, links, skill_file=skill_file)

    work_dir = resolve_working_dir(fm, repo_paths)
    if work_dir:
        logger.info("Working dir: %s", work_dir)

    ctx_file = write_context_file(task_id, prompt)

    launched = launch_tmux(tmux, claude, session_name, ctx_file, working_dir=work_dir)
    if launched:
        logger.info("Launched tmux session: %s", session_name)
        return task_id
    logger.warning("Skipped launch for %s", task_id)
    return None


def watch_tasks(
    vault_path: str | Path,
    tasks_folder: str = DEFAULT_TASKS_FOLDER,
    repos_note: str = DEFAULT_REPOS_NOTE,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    settle: float = DEFAULT_SETTLE_TIME,
) -> None:
    """Watch `<vault>/<tasks_folder>/` for pending tasks and dispatch them.

    Blocks until interrupted (Ctrl+C). Binary detection happens once at
    startup so a missing tmux or claude CLI fails loudly before the loop.
    """
    vault = Path(vault_path).resolve()
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault path does not exist: {vault}")

    tmux = detect_tmux()
    claude = detect_claude()
    logger.info("tmux:   %s", tmux)
    logger.info("claude: %s", claude)

    tasks_dir = vault / tasks_folder
    repo_paths = load_repo_paths(vault, repos_note)
    if repo_paths:
        logger.info("Loaded %d repo paths: %s", len(repo_paths), ", ".join(repo_paths.keys()))

    logger.info("Watching %s (poll=%.1fs, settle=%.1fs)", tasks_dir, poll_interval, settle)
    logger.info("Waiting for tasks with status: pending...")

    processed: set[str] = set()

    while True:
        try:
            pending = scan_pending(tasks_dir, settle=settle)
            for filepath in pending:
                key = str(filepath)
                if key in processed:
                    continue
                try:
                    dispatch_task(filepath, tmux, claude, repo_paths, vault_path=vault)
                except Exception as e:
                    logger.error("Dispatch failed for %s: %s", filepath.name, e)
                processed.add(key)

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error("Error in poll loop: %s", e)
            time.sleep(poll_interval)
