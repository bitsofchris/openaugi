---
name: research-agent (template)
description: >
  TEMPLATE — copied to <vault>/OpenAugi/AGENT/research-agent.md on `openaugi init`.
  The vault copy is the live version the agent reads. Edit there, not here.
  Instructions for the research task type — NotebookLM CLI, source ingestion, cited knowledge extraction.
---

# Research Agent

Extension to `augi-agent.md` for research tasks. When the task file contains
`zzz: research <topic>` or `zzz: ingest <urls>`, follow these instructions.

## Tools

You have the `nlm` CLI (NotebookLM) on PATH:

```
nlm --help    # full command reference
```

### Quick reference

```bash
# Notebooks
nlm notebook list                              # list all notebooks
nlm notebook create "<topic>"                  # create a new notebook
nlm notebook query <notebook_id> "question"    # cited Q&A against all sources

# Sources — add to a notebook
nlm source add <notebook_id> --url <url>                    # add a web URL
nlm source add <notebook_id> --url <a> --url <b> --url <c>  # bulk add URLs
nlm source add <notebook_id> --youtube <url>                # add YouTube video
nlm source add <notebook_id> --youtube <a> --youtube <b>    # bulk add videos
nlm source add <notebook_id> --file document.pdf            # upload local file
nlm source add <notebook_id> --wait                         # wait for processing

# Sources — inspect
nlm source list <notebook_id>                  # list sources in a notebook
nlm source content <notebook_id> <source_id>   # fetch full transcript/content

# Research — NotebookLM finds sources for you
nlm research start "query" --notebook-id <id> --mode fast   # ~30s, ~10 sources
nlm research start "query" --notebook-id <id> --mode deep   # ~5min, ~40 sources
nlm research start "query" --title "New Notebook" --auto-import  # create + import
nlm research status                            # check progress
nlm research import                            # import discovered sources
```

## Process

### When given a topic (`zzz: research <topic>`)

1. **Check existing notebooks** — `nlm notebook list`. Reuse if one exists for this topic.
2. **Create notebook** if needed — `nlm notebook create "<topic>"`.
3. **Let NotebookLM find sources** — `nlm research start "<topic>" --notebook-id <id> --mode deep --auto-import`. This finds ~40 sources automatically. While that runs (~5 min), proceed to step 4.
4. **Add your own sources** — web search for YouTube videos, papers, articles the auto-research may miss. Add via `nlm source add --url/--youtube`.
5. **Query the notebook** — run 5-10 key questions:
   - "What are the foundational concepts in <topic>?"
   - "What are the main approaches and their tradeoffs?"
   - "What are the key papers and who are the main researchers?"
   - "What are the open problems and active research directions?"
   - "What practical applications exist?"
   - Plus topic-specific questions based on what you learn.
6. **Write output** to the vault.

### When given URLs or a deep research report (`zzz: ingest <urls>`)

This covers explicit URLs and pasting results from Claude deep research or
Perplexity reports that contain source links.

1. **Extract all URLs** from the block content — papers, articles, YouTube
   videos, anything linkable. Deep research reports often have citations
   with URLs inline or in a references section.
2. **Check existing notebooks** — does a relevant one exist? If ambiguous,
   create a new one named after the topic.
3. **Add all URLs** — `nlm source add <id> --url <a> --url <b> ...` (bulk).
   Use `--youtube` for YouTube links. Use `--wait` to confirm processing.
4. **Run extraction queries** — 3-5 questions targeting what the user likely
   wants from these sources. If the deep research report had a clear thesis
   or question, use that as the first query.
5. **Write output** to the vault.

## Output conventions

Write all output under `Research/<topic>/` in the vault.

### File layout

```
Research/<topic>/
├── index.md          — overview, key concepts, links to Q&A files
└── qa/
    ├── concepts.md   — foundational concepts
    ├── approaches.md — methods and tradeoffs
    └── ...           — one file per major question
```

### Frontmatter

Every file gets:

```yaml
---
source/notebook: true
topic/<slug>: true
notebook_id: <id>
---
```

### Q&A file format

```markdown
# <Question>

<Answer with [N] citation markers from NotebookLM>

QQQ

<Next distinct claim or section>

QQQ

<Continue...>

## Sources cited
- [1] <source title / URL>
- [2] ...
```

Use `QQQ` between distinct claims or sections. This is load-bearing — the
vault splitter uses it to create separate blocks per claim, which makes
cross-claim retrieval work.

Keep `[N]` citation markers from NotebookLM output as-is. They trace back
to the source content. Don't resolve them to wikilinks — the vault adapter
handles linking.

### index.md format

```markdown
---
source/notebook: true
topic/<slug>: true
notebook_id: <id>
---

# Research: <Topic>

## Key concepts
- ...

## Sources loaded
- <list of channels/URLs loaded into the notebook>

## Q&A files
- [[concepts]] — foundational concepts
- [[approaches]] — methods and tradeoffs
- ...

## Notebook
NotebookLM notebook ID: `<id>`. Open in browser or query via
`nlm notebook query <id> "your question"` for follow-up research.
```

## Limits and constraints

- **300 sources per notebook** — NotebookLM hard limit. Use multiple notebooks for very large topics.
- **Indexing delay** — after bulk loading, sources take a few minutes to become queryable. If queries return empty results right after loading, wait and retry.
- **Session expiry** — `nlm` auth sessions last <24h. If auth fails, note it in `## Human Todo` with `status: needs-input`.
- **Google account risk** — heavy automation may trigger Google account flags. Space out bulk loads if doing many.

## When done

Fill in `## Results` on the task file with:
- Notebook ID and source count
- List of Q&A files written
- Any sources that failed to load
- Suggested follow-up questions

Set `status: done`.
