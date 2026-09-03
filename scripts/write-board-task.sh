#!/bin/bash
# Write the daily currency-board task file. The OpenAugi task watcher picks it
# up and runs an agent session that applies the `currency-board` lens.
#
# Scheduled by ~/Library/LaunchAgents/com.openaugi.board.plist (06:00 daily).
# Manual run: scripts/write-board-task.sh
set -euo pipefail

VAULT="${OPENAUGI_VAULT:-$HOME/Documents/ZK Home}"
DAY="$(date +%Y-%m-%d)"
TASKS="$VAULT/OpenAugi/Tasks"
TASK="$TASKS/TASK-$DAY-currency-board.md"

# One board per day. If today's task or board already exists, do nothing —
# reruns are the user's call, via the plugin or by deleting the board.
[ -e "$TASK" ] && { echo "board task already written for $DAY"; exit 0; }
[ -e "$VAULT/OpenAugi/Board/$DAY - Board.md" ] && { echo "board already built for $DAY"; exit 0; }

mkdir -p "$TASKS"
cat > "$TASK" <<EOF
---
status: pending
working_dir: $VAULT
---

# Currency board — $DAY

## Context

Scheduled daily build. The board is the one surface that promises currency:
where each thread left off, the next concrete moves, at most three things
needing human judgment, and what drifted.

## Task

Apply the \`currency-board\` lens (\`OpenAugi/AGENT/lenses/currency-board.md\`).
Follow its Process and Hard rules exactly.

Before building, read \`OpenAugi/Board/.board-state.json\`: never re-propose an
item whose state is \`done\`, \`not-doing\` or \`someday\`, honor every \`reason\`
literally, and use \`appearances\` for the staleness flag (an item on its third
board gets one plain line, never a repeated nag).

Write \`OpenAugi/Board/$DAY - Board.md\`, then overwrite
\`OpenAugi/Views/View - Board.md\` with a link and embed of it.

## Human Todo

## Results
EOF

echo "wrote $TASK"
