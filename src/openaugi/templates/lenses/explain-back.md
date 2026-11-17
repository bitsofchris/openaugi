---
kind: engine
name: explain-back
description: >-
  "Do I understand what this chat built?" — a short stepped HTML explainer of the chat's own work, traced through the real code, then five say-it-back questions graded and kept in the anchor note's Recall table.
scope: >-
  This conversation, the code it changed and its diff, the tests, and the anchor note's Outcome.
trigger: on-demand
target: >-
  note — an HTML page under OpenAugi/YYYY/MM/DD/, pinned in the anchor note's Agent Log; graded questions as rows of the ## Recall table there.
---

# explain-back

Explain back to me what this chat built, so I understand it well enough to review it, make the next change without an agent, and say what would break. Then make me recall it.

1. Ask me first, one question at a time and only what I have not said: which part I want to own (the whole change, one mechanism, one decision), and what I will make at the end (the next change, or a review of this one, done without an agent).
2. Sources, in this order of trust: the code as it is now and the diff (git log, git diff), the tests and whether they ran, the anchor note's Outcome, then this conversation. The chat's own summary of its work is a claim to check against the diff, not a source. Where the chat and the code disagree, follow the code and say so.
3. Write one self-contained HTML page of five to eight steps, one idea each, with the step number always in view: what was wrong or missing before; a map of the files and functions that changed, one line each, naming the one to read first; the mechanism, tracing one real input from this work through the new code with its actual values; each decision the chat made and what it ruled out, with a box that asks for my pick and why before it shows the answer; what would break, and which test would catch it; the limits and what was left undone. Define each term where it is first used. Every name and number must match the code as it is now; list on the page anything unverified.
4. Save the page in the notes folder under OpenAugi/YYYY/MM/DD/ and pin it in the anchor note's Agent Log as a line of the form - file: `<path>` · explainer: <what it explains>.
5. Then ask me five say-it-back questions about why and how, never about names, one per message, and wait for my answer before showing anything. Before asking each, write down the two to four claims a complete answer makes, and grade my answer strictly against them: recalled (every claim met), partial (the core is there, a claim is missing), missed. No praise: one line on what was missing.
6. Keep each graded question as a row of the `## Recall` table, the last block of the anchor note's `# Agent Log` section (the table the Augi pane's Recall card keeps; make the heading and table if they are missing): `| id | question | full answer | grades | due | mix-up |`, with the next free `qN`, the claims joined by ` · `, grades as `YYYY-MM-DD grade`, comma separated, oldest first, and the mix-up in one line or empty. Due is the grade's day plus the question's rung on the 1, 3, 7, 21 day ladder: a new question starts on 1 day, recalled moves up a rung, partial keeps it, missed goes back to 1. Compute dates with the shell.
