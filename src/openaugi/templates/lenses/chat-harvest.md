---
kind: engine
name: chat-harvest
description: >-
  What was worth keeping from yesterday's AI chats — at most one concise note, in my own words, with where it goes and what it links to. Ideas and learning, never the coding.
scope: >-
  Yesterday's human turns from my local AI chat transcripts (Claude Code ~/.claude/projects, Codex ~/.codex/sessions), extracted by scripts/session_harvest.py in the openaugi repo. Anchored on MY prompts and questions; assistant replies are context only, never the source of a claim. Excludes sessions augi dispatched to itself, and excludes the mechanics of coding — debugging, file edits, test runs, "make it do X".
trigger: on-demand   # Sunday only, inside the weekly pass; weekday boards skip it
target: >-
  board — a `## Worth keeping` section merged into today's OpenAugi/Board/YYYY-MM-DD - Board.md using the two-box proposal grammar (`do` / `no`, `<!-- propose:keep-<slug> -->`). The candidate note is written out IN FULL in the brief, so ticking `do` dispatches an agent that saves exactly what they read. Applied standalone, it appends the same section to today's board if one exists; with no board, it reports the candidate in session and writes nothing.
---

# Chat Harvest

## Intent

A lot of the user's thinking now happens in chat windows and dies there. This
lens is the one pass that asks: **of everything I said to an AI yesterday,
is there one thing my second brain should own?**

The bar is high on purpose: one candidate per day, or none. Zero is the
common, correct answer.

**Anchored on their prompts.** The unit of interest is what *they* asked,
claimed, noticed, or decided — their questions are the record of what they were
working out. The assistant's reply is context that helps you understand the
question; it is never the thing being saved. A note that is mostly the
model's prose has failed this lens, however good the prose.

**Not the coding.** Debugging, file edits, "run the tests", "fix that
import", scaffolding — none of it. What survives is the *discussion*
layer: a question they kept circling, a concept they worked out in their own
words, a distinction they drew, a design principle they stated, a decision and
its reason, an idea they had while doing something else.

Voice: **theirs, quoted.** The note is built out of their sentences wherever
they exist. You compress and connect; you do not upgrade their phrasing into
something smoother than they said.

## Process

1. **Extract the window.** From the openaugi repo:

   ```bash
   python3 <openaugi repo>/scripts/session_harvest.py --day <yesterday>   # repo path: OpenAugi/AGENT/Repos.md
   ```

   Defaults are tuned for this lens: yesterday's local day, agent-dispatched
   sessions excluded, trivial turns ("ok", "do it") dropped, each turn
   verbatim with a slice of the longest reply it drew. Add `--days 3` when
   catching up after a gap; never widen past 7 — this is a daily pass, not
   an archive sweep.

   No sessions, or nothing above the bar → **write nothing** and say so in
   one line. An empty section is noise; a forced note is worse.

2. **Read for the discussion layer.** Walk their turns in order and mark the
   ones that are thinking rather than instructing. The signals that count:

   - a question they asked and then *kept asking* in different words
   - a claim in their own voice ("I think X is really Y because…")
   - a distinction, model, or analogy they built to explain something to themselves
   - a decision with a reason attached
   - an idea that arrived sideways, unrelated to what the session was for
   - a piece of self-observation about how they work

   Signals that do NOT count: task instructions, corrections to the agent,
   approvals, anything where the interesting content is the model's answer
   rather than their question.

3. **Pick ONE.** Rank the marked turns by: does it recur across the day ·
   does it connect to something already in the vault · would they lose it if
   nobody wrote it down. Take the top one. **If nothing clears the bar, stop
   here** — report zero.

4. **Route it before you draft it.** Search the graph (`get_context`,
   `search`) for what already covers this. Read
   [[routing]] (OpenAugi/AGENT/routing.md) and follow it:

   - **Already exists** → do not propose a duplicate. If the chat *added*
     something to it, propose an append instead, naming the exact note.
   - **Continues a live thread** → append to that thread's mirror
     (`OpenAugi/Threads/MIRROR - <source>.md`) in the dated-append format.
   - **Genuinely standalone atomic idea** → a new note in `OpenAugi/Notes/`.
   - **Research synthesis / plan / durable reference** → `OpenAugi/Research/`,
     `OpenAugi/Plans/`, `OpenAugi/Docs/` respectively.

   Name **2–4 real links** — notes you confirmed exist, not plausible titles.

5. **Draft the note in full.** Under 150 words. Structure:
   the idea in one or two lines · their own words quoted with the date · the
   links. That is the whole note. If it needs more than 150 words it is a
   research task, not a harvest — say that instead and propose nothing.

6. **Offer it on the board.** Merge a `## Worth keeping` section into today's
   board note, above the collapsed details block — Sunday boards only; weekday boards do not carry `## Worth keeping`. Read the `proposals` block of
   `OpenAugi/Board/.board-state.json` first and **never re-offer anything
   marked `declined`** — nor the same substance under a new key, which is the
   same violation wearing a hat.

   ```markdown
   ## Worth keeping

   *From yesterday's chats. `do` saves it exactly as written below; `no` means never again.*

   - **<the idea in one line>** — <new note `OpenAugi/Notes/<Title>.md` | append to [[Note]]>
       ↳ Save this, verbatim, to `<exact path>` (or: append this dated block to [[Note]]
         in the `### YYYY-MM-DD (augi)` format from OpenAugi/AGENT/routing.md). Tag it
         a `- [ ] seen` box. Do not expand it, do not add sections, do not restate it in
         better prose — this text is the note:
         ---
         <the full note, under 150 words, their words quoted, links inline>
         ---
         Source: <session title> (`<project>`), <date> — `<resume command>`
       - [ ] do
       - [ ] no
       aaa:
       <!-- propose:keep-<stable-kebab-slug> -->
   ```

   The `↳` brief is handed to an agent verbatim when they tick `do`, so it must
   read as a complete instruction to someone who was not in the session.

7. **Update the lens index** — upsert the `chat-harvest` row in the `## Lenses`
   section of `View - Dashboard.md`: today's date, a link to today's board,
   and "1 candidate waiting" or "nothing above the bar". Touch nothing else
   on the Dashboard.

## Hard rules

- **One candidate per run. Never two.** The cap is the feature. Zero is a
  valid, frequent answer and costs nothing; a padded second note costs the
  section its credibility.
- **Under 150 words.** Longer means it is not a note yet.
- **Their words, not yours.** Quote them. Where you must paraphrase, keep their
  register — never smooth a rough sentence into a polished one and attribute
  it to them.
- **Never save the coding.** If the only candidate is about making something
  work, there is no candidate.
- **The AI's answer is never the note.** It can supply a fact they asked for;
  the note is still built around their question.
- **Never propose a duplicate.** Search first; an append to the existing note
  beats a second note on the same idea, every time.
- **Links must exist.** A `[[link]]` to a note that isn't there is a broken
  promise the board can't keep.
- **Never write the note itself.** This lens *offers*; the `do` tick is what
  saves. The user decides what their second brain owns.
- **Never re-offer a `declined` proposal**, in any wording.
- **Never write `.board-state.json`.** The janitor owns it.
- **This lens never touches raw notes.** Appends go to `OpenAugi/` mirrors,
  never to the source note, unless the user says otherwise.
