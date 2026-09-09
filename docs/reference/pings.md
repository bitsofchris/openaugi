# Pings — experience sampling into the daily note

**Name:** pings · `scripts/ping_stats.py` · the `ping-read` lens

**Description:** A phone-side check-in (iOS Shortcut) appends one
structured line per prompt to the day's daily note, roughly every 100
minutes. A Sunday lens reads only those lines, runs the counter, and hands
back cross-tabs plus a few yes/no questions. Cue during the day, synthesis
on the weekend. Nothing in this is a tracker app; the vault is the store.

**When to use:** when the question is "what am I doing on autopilot?" and
retrospective journaling can't answer it, because the moments worth
catching are the ones you don't write about. Not for anything that needs
a diagnosis; the read counts and asks, it never concludes.

## How it works

The line grammar, written by the Shortcut into
`_private/0-Fleeting-Inbox/YYYY-MM-DD.md`:

```
- HH:MM ping: level=1..5 mood=low|ok|wired place=home|office|errands|others|other event=none|bite|snack|play|tap|scroll|sip note=<free>
- HH:MM event: event=<kind> acted=yes|no level=.. mood=.. place=.. note=<free>
```

`ping` is a scheduled prompt; `event` is the on-demand button. `doing` is
the last key and may be several words.

The counter:

```bash
python3 scripts/ping_stats.py --vault "~/Documents/ZK Home" --days 7
```

Prints a markdown table (event rate overall, by time of day, by energy, by
mind) and bullets (event kinds, mind-elsewhere rate, what preceded events,
level when the event was food, event taps acted on). `--json` for the raw
summary, `--end YYYY-MM-DD` to pin the window. It only counts.

The lens (`OpenAugi/AGENT/lenses/ping-read.md`, trigger `every 7d`) runs
the counter, pastes the numbers, writes at most three observations and
three yes/no questions with `aaa:` lines into
`OpenAugi/Notes/<date> - Ping Read.md`, and the currency board carries the
questions. The lens's hard rules: script numbers win, structured lines are
the only input, no diagnosis, under-claim.

The card, schedule, Shortcut recipe, and the experiment's end date live
in the vault: `OpenAugi/Plans/Plan - Pings - Experience Sampling.md`.

Tests: `tests/test_ping_stats.py`.
