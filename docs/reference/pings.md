---
name: pings
description: Experience sampling into the daily note. A phone-side prompt appends one structured `- [HH:MM] <kind>: key=value …` line per check-in; `scripts/ping_stats.py` cross-tabs those lines over a window (hit rate of one target key by time of day and by every other key, values discovered from the data); a vault lens interprets the numbers. The field vocabulary lives in the vault lens, never in the repo.
---

# Pings

**Name:** pings · `scripts/ping_stats.py` · a vault lens that invokes it

**Description:** A phone-side check-in (an iOS Shortcut, a widget, anything
that can append a line) writes one structured line per prompt into the
day's daily note. On a schedule, a lens reads only those lines, runs the
counter, and hands back cross-tabs plus a few yes/no questions. Cue during
the day, synthesis later. Nothing in this is a tracker app; the vault is the
store.

**When to use:** when the question is "what am I doing on autopilot?" and
retrospective journaling can't answer it, because the moments worth
catching are the ones you don't write about. Not for anything that needs a
diagnosis; the read counts and asks, it never concludes.

## How it works

### The line grammar

Written into `<daily-dir>/YYYY-MM-DD.md` (default `_private/0-Fleeting-Inbox`,
override with `--daily-dir`):

```
- [HH:MM] <kind>: key=value key=value …
```

- `<kind>` is one word before the colon. Only kinds you declare on the
  command line count; every other `word:` line in the note is prose.
- The time is optional. A free-form line of a declared kind with no
  `key=value` pairs (`- event: ate a cookie, felt low`) still counts as one
  event of that kind with no fields.
- Any keys, any values. Keys and values are lower-cased. The last value may
  run to several words (`note=dinner prep`).

A neutral example:

```
- 09:20 ping: level=2 mood=low place=office event=snack note=email
- 15:10 event: event=snack acted=no mood=low place=office
```

**The vocabulary is yours, and it lives in the vault.** Which kinds you
write, which keys they carry, what the values are called, and which
contrasts are worth reporting all belong to the lens file under
`<vault>/OpenAugi/AGENT/lenses/` that invokes the script (a `kind: personal`
file). The repo carries no field list, no value list and no schedule.

### The counter

```bash
python3 scripts/ping_stats.py --days 7 \
  --scheduled ping --ondemand event --target event --free note
```

| Flag | Meaning | Default |
|---|---|---|
| `--scheduled KIND` | kind(s) written on a timer; repeatable or comma-separated | `ping` |
| `--ondemand KIND` | kind(s) written when the user presses a button | `event` |
| `--target KEY` | the key whose presence is the thing being rated | `event` |
| `--absent VALUE` | target value(s) that mean "nothing happened" | `none` |
| `--free KEY` | a free-text key to report "what preceded a hit" from | off |
| `--vault`, `--daily-dir` | where the daily notes are | config `[vault] default_path`, `_private/0-Fleeting-Inbox` |
| `--days`, `--end` | the window | 7, today |
| `--json` | the raw summary instead of markdown | markdown |

A *hit* is a line whose target key is present with a value that is not
empty and not an absent word. Rates run over the scheduled lines only, since
on-demand lines are self-selected; value counts run over every line. The
markdown output is one table and a few bullets:

- hit rate overall, by time of day (morning / afternoon / evening), and by
  every value of every other key found in the data
- the target's values, and each key's values, as counts
- for every key whose values are all numbers, the mean per target value
- with `--free`: the free-text values right before a hit, and at all
  scheduled lines
- the on-demand lines' values (e.g. an `acted=` key)

It only counts. The lens does the reading.

### The read

A lens file with an `every Nd` trigger runs the counter, pastes the numbers
verbatim, writes at most a few one-sentence observations that each point at
a number, and at most a few hypotheses phrased as yes/no questions with the
board's answer grammar. The lens's hard rules: script numbers win,
structured lines are the only input, no diagnosis, under-claim. See
[lenses.md](lenses.md) for the lens contract and
[currency-board.md](currency-board.md) for how questions are carried.

Tests: `tests/test_ping_stats.py` (neutral vocabulary).
