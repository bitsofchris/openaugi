---
name: user-guide
description: Day-to-day manual for using OpenAugi — entry points, the capture-to-view loop, the five trust rules, how to trigger a pass, and the lens system in brief. Durable reference, kept current; chronological build history lives in docs/plans/master-plan.md, not here.
---

# User Guide

## When to use this doc

You want to know where to look, how the loop works, or how to trigger
something — without reading code or the build history. For "what's next /
what shipped when," go to [docs/plans/master-plan.md](../plans/master-plan.md)
instead — that STATUS header is the chronological record; this doc is not.

The design commitments underneath everything here — the capture grammar,
the truth/index/cache/render layer model, the trust model, promotion —
live in [core-principles.md](core-principles.md). Read that first if
you're designing or changing the system rather than just using it.

## Entry points — where to start

| Context | Start here | What it gives you |
|---|---|---|
| Re-entry — morning, or sitting back down | `View - Board` | Where each thread left off · 1–2 next moves per lane · ≤3 things needing your judgment · what drifted. Built unprompted at 06:00; answered with checkboxes ([currency-board.md](currency-board.md)) |
| Daily life (vault) | `View - Dashboard` | What moved per area · task rollup · **Pending your answer** (the only review you owe) |
| Any thread | The AMOC/PMOC/MOC note itself | Your head + your journal + the transcluded view (recap + remote-capture feed) |
| On your phone | OpenAugi Mobile (augi app) | Capture → bridge writes to your real vault → ingest treats it like any note. `zzz:` in a block dispatches a task |
| Obsidian palette | Augi commands (plugin) | "Run review pass" / "Process dashboard" / "Distill selection" — each writes a pending task file |
| Repo / next Claude session | `CLAUDE.md` → [master-plan.md](../plans/master-plan.md) | The milestone sequence with a STATUS header updated every session |
| How does X work? | [review-pass.md](review-pass.md) | The write-back loop manual: tags vs links, grammar, running a pass |
| Debugging | `~/.openaugi/logs/openaugi.log` | Ingest/MCP crashes |

## The loop, in one line

**capture (anywhere, messy) → ingest → blocks + links (SQLite) → review
pass: route + tag (DB only) → views regenerate → Dashboard nominations →
you answer → agent assembles.**

## The rules that make it trustworthy

1. **Truth is yours, append-only.** Agents never edit your notes. Nothing
   is ever deleted — things demote by ceasing to materialize.
2. **Everything the agent writes is a cache.** Views under
   `OpenAugi/Views/` regenerate freely; a wrong view is a stale cache
   entry, not damage. No review needed.
3. **Review gates structure only.** New tags, new containers, merges:
   agent nominates on the Dashboard → you command → agent assembles.
4. **Classification is tags** (your closed taxonomy, either author);
   **membership is links** — `routed_to` edges in the DB, really just
   auto-filing: a block linked to the container(s) it belongs to, or
   none. See [data-model.md](data-model.md).
5. **Routes are re-derived, never migrated** (decided 2026-07-09). The
   DB is a disposable projection over your files, not truth. Editing a
   routed block changes its content-hash identity, so its route drops
   BY DESIGN and the block re-enters the review queue as new — the next
   pass re-files it. Durable intent belongs in the text as an `aaa:`
   line, which survives every edit because it rides inside the content.
6. **Refresh by tier.** Routing: every pass. Log sections: free,
   always. Recaps: only when they'd change.

## Daily use

**No behavior change.** Capture as always: daily notes, inbox, MOC
journals, random notes. Grammar when you want it: `qqq` splits blocks ·
`zzz: <task>` dispatches an agent immediately · `aaa: <routing hint>`
tells the next pass where a block belongs.

**Trigger a pass — whenever you want, no schedule:**
- `openaugi review` in any terminal (writes a task file; the watcher runs it)
- `openaugi review --dashboard-only` — just execute your nomination answers
- Obsidian palette: **Augi: Run review pass** / **Process dashboard** / **Distill selection**
- Say **"run the review pass"** / **"process the dashboard"** in a Claude session, or write `zzz: run the review pass` in any note

**After a pass (~5 minutes):** open the Dashboard, read what moved, scan
the task rollup, answer **Pending your answer**. Every nomination
carries a stable anchor and an `- answer:` slot. Wrong routing? Say so,
or drop an `aaa:` next time — corrections are the tuning signal.

**On demand:** `"distill X"` → one curated note with provenance,
placement nominated on the Dashboard. Persist only what you'll reuse;
in-chat synthesis stays free and unpersisted.

**The one thing that arrives without being asked:** the currency board,
built at 06:00 into `OpenAugi/Board/<date> - Board.md` and pointed at by
`View - Board`. Read it at re-entry — start of day, or sitting back down
after an interruption. Answering is three checkboxes per item — **done**
/ **not doing** / **someday** — plus an optional `aaa: <why>` line.
Ticking is the entire interaction: the janitor records it and the next
board never re-proposes it, honoring a `not doing` reason literally.
Untouched items stay open and get carried; on an item's third board you
get one plain staleness line, never a repeated nag. Manual:
[currency-board.md](currency-board.md).

## The lens system, in brief

A **lens** = a saved question applied to your data: scope + intent →
derived artifact. Every lens is one markdown file in
`OpenAugi/AGENT/lenses/` — no engine, no code; the agent follows the
generic mechanics in `augi-agent.md`. Full spec format, apply/create
grammar, and the view/note/dashboard output-mode rules:
[lenses.md](lenses.md).

- **Use one:** "apply lens NAME" or "apply lens NAME to SCOPE" in chat,
  as `zzz:` in any note, or (some) as a mobile chip.
- **Add one:** say "new lens NAME: INTENT" from anywhere, or write the
  `.md` file yourself. Tune by editing the file; delete by deleting it.

Current lenses (canonical list: `OpenAugi/AGENT/lenses/` or
[lenses.md](lenses.md); snapshot below may drift — the folder is truth):

| Lens | Answers |
|---|---|
| **currency-board** | "Where did I leave off, and what's the next concrete thing?" — per lane, plus ≤3 judgment items and what drifted. Scheduled daily; answered with checkboxes |
| **morning-briefing** | *(superseded by currency-board — mirror-only, no answer channel)* "What matters today?" — yesterday distilled, open loops due, pending nominations, one resurfaced thought |
| **open-loops** | "What did I say I'd do and never close?" — commitments and aging questions, checkbox to close |
| **echoes** | "Have I thought this before?" — current thinking matched against older notes; recognition, not summary |
| **idea-lineage** | "How did my thinking on X evolve?" — one topic's full biography: earliest mention → revisions → strongest form → dead branches |
| **decision-audit** | "What am I deciding, and what does my own evidence say?" — audits one live decision against your own prior thinking; never recommends |
| **emerging** | "What is trying to emerge that isn't explicit yet?" — area-by-area snapshot |
| **content-pipeline** | "What's close to shippable?" — every content idea, seed → near-ready |
| **cluster-weather** | "Which themes are growing or dying?" — concept-cluster growth/death, nominated for promotion |
| **nuggets** | Stand-alone insights in recent working notes, nominated for promotion |
| **distill** | Gather everything on a topic, write ONE curated note with provenance |

## Current system state

M1–M3 shipped. **M4 (routing quality) and M5 (lens system) are CLOSED**
(Chris's call, 2026-07-09) — no more usage gating on them. **Surface
decision: no web app.** The two owned surfaces are the mobile app and
Obsidian (vault files are the API); the plugin stays a thin task-file
writer. **M8 (data lake) is open**: the source-attribution firewall
(`[vault.source_rules]`, `source/*` tags) and the idea-lineage lens
shipped; Google Drive import and the curator are next. Full history and
what's next right now: [master-plan.md](../plans/master-plan.md) STATUS
header — read that first in any new session.

## Related docs

- [core-principles.md](core-principles.md) — **the skeleton**: capture grammar, truth/index/cache/render, trust model, promotion — the four invariants under everything in this guide (read first when designing, not just using)
- [master-plan.md](../plans/master-plan.md) — the sequence + STATUS header (start every session here)
- [lenses.md](lenses.md) — lens spec format, apply/create mechanics, output modes
- [currency-board.md](currency-board.md) — the scheduled board you read at re-entry, its checkbox contract, and the janitor that makes the next board honor your answers
- [review-pass.md](review-pass.md) — the write-back loop manual
- [data-model.md](data-model.md) — blocks, links, `routed_to`, taxonomy
- [clustering.md](clustering.md) — clustering + cluster weather
- [docs/plans/from-capture-to-jarvis.md](../plans/from-capture-to-jarvis.md) — longer-horizon vision
