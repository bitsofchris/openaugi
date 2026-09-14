#!/bin/bash
# Write the weekly substack-batch task file. The OpenAugi task watcher picks it
# up and runs an agent session that applies the `substack-batch` lens.
#
# Scheduled by ~/Library/LaunchAgents/com.openaugi.substack.plist (Fri 06:30).
# Manual run: scripts/write-substack-task.sh
set -euo pipefail

VAULT="${OPENAUGI_VAULT:-$(python3 -c 'from openaugi.config import load_config, resolve_vault_path; print(resolve_vault_path(None, load_config()) or "")' 2>/dev/null)}"
[ -n "$VAULT" ] || { echo "no vault: set OPENAUGI_VAULT or [vault] default_path in the openaugi config" >&2; exit 1; }
DAY="$(date +%Y-%m-%d)"
TASKS="$VAULT/OpenAugi/Tasks"
TASK="$TASKS/TASK-$DAY-substack-batch.md"

# One rebuild per day. Reruns are the user's call — delete the task file.
[ -e "$TASK" ] && { echo "substack task already written for $DAY"; exit 0; }

mkdir -p "$TASKS"
cat > "$TASK" <<EOF
---
status: pending
working_dir: $VAULT
---

# Substack queue — $DAY

## Context

Scheduled weekly rebuild. The queue answers "what from this week could I
publish?" — draft-ready note candidates, best first, each a scannable card.

## Task

Apply the \`substack-batch\` lens (\`OpenAugi/AGENT/lenses/substack-batch.md\`).
Follow its Process and Hard rules exactly.

Before writing anything, read the CURRENT
\`OpenAugi/Views/View - Substack Queue.md\`:

- Harvest every ticked \`posted\` / \`skip\` / \`sent\` box and every \`aaa:\` line.
- Read the existing \`## Recently decided\` list. **Never re-propose anything
  in it.**
- Ticked cards become new decided-list lines dated $DAY, carrying their
  \`aaa:\` text as the reason.

Then overwrite the cards, and rewrite \`## Recently decided\` as today's new
lines plus every carried line dated within 30 days. Drop anything older.

The boxes are a log, not a trigger. Never post anything.

## Human Todo

## Results
EOF

echo "wrote $TASK"
