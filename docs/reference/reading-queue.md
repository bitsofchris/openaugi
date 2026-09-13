---
name: reading-queue
description: One reading queue, and it is Readwise Reader. Flagged notes are pushed as Reader documents under a daily cap; the highlights you make come back onto the note that produced them. Manual commands — nothing is scheduled.
---

# Reading queue — augi's prose in Reader, your marks back in the vault

**When to use:** an agent wrote something you want to *read*, not process —
a research synthesis, an answer to a question you asked. It goes to Reader,
you read it on your phone, and your highlights land back on the note.

Design record: `<vault>/OpenAugi/Plans/Plan - Reading Queue in Readwise.md`.

## The round trip

```
a note in OpenAugi/            reading_queue: true in its frontmatter
   ↓  openaugi reading push    POST /save/  url=https://augi.local/note/<sha8>
Reader                          phone, couch, offline — location: later
   ↓  you read it and highlight
   ↓  openaugi reading harvest GET /list/?category=highlight → parent.source_url
the same note                   ## Read in Reader — <date>
   ↓  ingest
the graph                       your marks, on augi's text, next to the original
```

## The three decisions, and how they were made

**The gate is a frontmatter flag, not a folder.** `reading_queue: true`, and
nothing else ships. A folder gate would fight the existing routing (research
already belongs in `Research/`), and deciding at dispatch time means deciding
before you know whether the output is worth reading. The *rule for when an
agent sets the flag* is deliberately prose in the vault's `augi-agent.md`, not
code — if the wrong things start showing up, the fix is one sentence, not a
rebuild.

**The join key is a fabricated URL.** Reader requires a `url` but does not
require it to resolve, so we mint `https://augi.local/note/<sha8-of-vault-path>`.
It is stable, it makes a re-push an in-place update instead of a duplicate
(`POST /save/` returns 201 on create, 200 on update), and it comes back on
every read as `source_url` — so a highlight traces to its note with no mapping
table. `key_from_url` also accepts the `augi://note/<sha8>` spelling, which
survives a hand-made `curl` test.

**Highlights land on the source note, not in a mirror.** One artifact per
idea. The alternative — letting the official Readwise plugin sync augi's own
documents back into `_private/2-Reference/Readwise/` — re-enters augi's output
into the graph stamped `source/readwise`, i.e. classifies our own writing as
external reading material, and duplicates every pushed note in retrieval.
**Suppress it with an exclude glob** in `~/.openaugi/config.toml`:

```toml
[vault]
exclude_patterns = ["_private/2-Reference/Readwise/*/Augi — *.md"]
```

## Commands

```bash
openaugi reading push --dry-run --show-html   # what would ship, and how it renders
openaugi reading push                         # ship it (max --cap, default 2)
openaugi reading harvest --dry-run            # what came back
openaugi reading harvest                      # append it to the notes
openaugi reading status                       # flagged / pushed / last harvest
```

Nothing runs on a schedule. Both commands are safe to re-run: push skips notes
whose content has not changed since the last push, harvest skips highlights it
has already appended.

**The token** comes from `READWISE_TOKEN` in the environment and from nowhere
else. A copy also sits in the Obsidian plugin's `data.json`; this code does not
read it, so the credential stays in one place.

## The guardrails, and why each exists

| Guardrail | Why |
|---|---|
| **Cap of 2/day** (`--cap`) | A queue that grows faster than it is read is the failure mode of every abandoned queue. Over-cap notes wait for tomorrow. |
| **`location: "later"`** | augi's output goes *into* the queue, not to the front of it — it never jumps things you deliberately saved. |
| **Machinery folders never ship** | `Tasks/`, `Sessions/`, `Archive/`, `Compiled/`, `Context/` are skipped even if flagged by hand. |
| **Broken frontmatter fails closed** | A note whose YAML does not parse has no flag, so nothing ships. |
| **Writes only under `OpenAugi/`** | Harvest refuses to append outside it, in code, not just by convention. |
| **One failed push does not fail the run** | The rest of the batch still ships; the failure is reported. |
| **`[[Wikilinks]]` render as bold** | They are dead in Reader. Anything whose value is its link graph is not reading-shaped output and should not have passed the gate. |

## Code

| File | Does |
|---|---|
| `src/openaugi/reading/note.py` | The join key, the frontmatter gate, markdown → HTML (dependency-free) |
| `src/openaugi/reading/reader_api.py` | Reader API client — `save`, `list_documents`, paging, one 429 backoff |
| `src/openaugi/reading/push.py` | Candidate selection, the cap, the payload, the ledger |
| `src/openaugi/reading/harvest.py` | Highlights → parent → `source_url` → note, append, dedupe |
| `src/openaugi/cli/main.py` | `openaugi reading push / harvest / status` |
| `tests/test_reading.py` | Everything above, against a fake Reader — no network |

The ledger is the `reading_queue` records collection
([records.md](records.md)): one row per note keyed by sha8, holding the vault
path, the last-pushed content hash, the Reader id and the highlight ids already
harvested. `reading_queue_state/harvest` holds the harvest watermark. Both are
machinery — delete them and the only cost is one redundant re-push per note.

## Not built yet (deliberately)

These wait on the taste test in the design record — *does reading augi's prose
in Reader feel like reading, or like doing inbox?* If the answer is "inbox",
delete `src/openaugi/reading/` and nothing else is affected.

- **The auto-gate rule** in `augi-agent.md` — for now, `reading_queue: true` is
  set by hand or by an explicit `zzz: ...and put this in my reading queue`.
- **Scheduling.** No launchd job; both commands are run by hand.
- **The board line** (`## Reading queue — pushed 2 · read 1 · 3 highlights came back`).
- **Highlights → `feedback-log.ndjson`.** Which paragraphs of a research note
  the user actually marked is the sharpest signal this system has about what is
  worth writing; it should be logged alongside the board ticks.
