---
name: vault-config-agent
description: An LLM agent that samples a vault, discovers folder structure and naming conventions, and generates vault-specific ingestion and embedding config. Replaces manual hardcoding of exclude patterns, granularity rules, and cleaning rules. Runs on first init and periodically as the vault evolves.
---

# Vault Config Agent

## Problem

The rules that make embeddings useful — which folders to skip, which notes to embed as a
whole document, what naming conventions mean — are currently hardcoded in `DEFAULT_EXCLUDE_PATTERNS`
and `vault.py`. We discovered them manually by browsing the vault, finding broken blocks
(Docker `#` comments treated as headings, Readwise clippings cluttering search, empty
checkbox-only template sections, etc.) and patching them one by one.

This doesn't scale. Every vault is different. The agent should do this exploration work.

---

## Goal

Ship an `openaugi configure` command (and a periodic background refresh) that:

1. Samples the vault — folder by folder, stratified, compact representation
2. Sends samples to an LLM with a structured analysis prompt
3. Parses the response into a `VaultConfig` Pydantic model
4. Writes the config to `~/.openaugi/vault_config.json`
5. The ingest + embed pipeline reads from this config instead of hardcoded defaults

The LLM should discover the same rules a human expert would find by spending 20 minutes
exploring the vault.

---

## Eval Baseline

We manually explored the vault and produced the following rules. A good config agent run
against this vault should reproduce all or most of these. This is our test case.

### Exclude patterns (skip entirely at ingest)

```python
"**/4-Tech Notes/**"    # command cheatsheets (Docker, etc.) — reference, not thought
"**/Instapaper/**"      # imported web clippings
"**/Readwise/**"        # imported highlights
"**/Snipd/**"           # imported podcast clips
"*.excalidraw.md"       # Obsidian drawing files — binary-ish JSON, not prose
```

Standard system excludes (not vault-specific, always applied):
```python
".obsidian/**", ".git/**", ".smart-env/**", ".trash/**",
"templates/**", "OpenAugi/Compiled/**"
```

### Granularity rules (whole-doc vs split-by-heading)

| Pattern | Rule | Reason |
|---|---|---|
| `WK - YY-MM-DD` | document-level | Weekly reflections are one coherent unit; heading-splitting creates disconnected Q&A fragments |

### Content cleaning (applied at embed time, not stored)

- Strip image embeds `![alt](url)` — Readwise cover images, screenshots
- Strip raw URLs `https://...` — no semantic content
- Strip markdown link syntax `[text](url)` — keep anchor text
- Strip bold/italic markers, checkbox markers, blockquote markers, horizontal rules

### Block filtering (skip blocks with no meaningful content)

- Sections containing only `- [ ]` empty checkboxes — unfilled template placeholders
- Sections containing only `---` horizontal rules — section separators with no body

### What the agent should detect from samples alone

A good agent sampling 3-5 files per folder should observe:

- `2-Reference/Readwise/` — notes contain `author:`, `url:`, `readwise-assets` image URLs,
  structured highlight format. → **exclude**
- `2-Reference/Instapaper/` — same pattern as Readwise. → **exclude**
- `2-Reference/Snipd/` — podcast transcription format, imported. → **exclude**
- `4-Tech Notes/` — Docker commands, shell references, code-heavy, no original thought. → **exclude**
- `Excalidraw/` or `*.excalidraw.md` — JSON-heavy content. → **exclude**
- `5-Journals/Weekly/WK - *` — consistent heading structure (reflection questions),
  2-digit date in filename. → **document-level granularity**, **parse WK date**
- Template notes — headings present, body is all `- [ ]` items. → **filter empty blocks**

---

## Architecture

### Components

```
src/openaugi/configure/
    sampler.py      — walk vault, collect folder stats + file samples
    analyzer.py     — call LLM, parse structured response → VaultConfig
    config.py       — VaultConfig Pydantic model, load/save to disk
    cli.py          — `openaugi configure` command
```

### VaultConfig model

```python
class FolderRule(BaseModel):
    pattern: str                    # glob, e.g. "**/Readwise/**"
    action: Literal["exclude", "document_level"]
    reason: str                     # LLM's explanation, shown to user

class VaultConfig(BaseModel):
    vault_path: str
    generated_at: str               # ISO timestamp
    exclude_patterns: list[str]     # merged with DEFAULT_EXCLUDE_PATTERNS at runtime
    document_level_patterns: list[str]  # filename glob patterns → no heading split
    min_block_chars: int            # skip blocks shorter than this
    folder_rules: list[FolderRule]  # full rule set with LLM reasoning
    raw_llm_response: str           # saved for debugging / re-evaluation
```

Saved to `~/.openaugi/vault_config.json`. Loaded by `parse_vault` and the embed pipeline.

### Sampler

```python
def sample_vault(vault_path: Path, files_per_folder: int = 4) -> VaultSample:
    """Walk vault. For each folder: collect file count, avg size, date range,
    and N representative file snippets (first 400 chars after frontmatter strip).
    Returns a compact structure suitable for the LLM prompt."""
```

Key design decisions:
- **Stratified by folder** — not random across all files. Each folder gets its own sample
  regardless of size. This surfaces a 5-file `Snipd/` folder just as well as a 500-file
  `Readwise/` folder.
- **Compact representation** — show folder name, file count, and 4 snippets. Total prompt
  should be well under 8k tokens for any realistic vault.
- **Strip frontmatter before sampling** — show the LLM the note body, not YAML metadata noise.

### Analyzer prompt

Single-turn, structured output. The prompt shows:

```
You are analyzing an Obsidian vault to generate ingestion and embedding configuration.

Here is the vault structure with sample notes from each folder:

[FOLDER SAMPLES]

For each folder, decide:
1. Should it be EXCLUDED from ingestion entirely? (imported content, system files, binary-ish)
2. Should notes be embedded as WHOLE DOCUMENTS instead of split by headings?
   (reflection journals, coherent single-topic notes)
3. Are there filename patterns that indicate special note types?

Also identify:
- Any content patterns to strip before embedding (boilerplate phrases, template headers)
- A sensible minimum block length for this vault (in characters)

Respond as JSON matching this schema: [SCHEMA]
```

Use structured output / tool_use to enforce the schema. No free-form prose in the response.

### Periodic refresh

Config is refreshed when:
1. User runs `openaugi configure` explicitly
2. Ingest detects the config is older than `refresh_after_days` (default: 30)
3. Vault has grown significantly (>20% more files than when config was generated)

On refresh: re-run the sampler + analyzer, diff the new config against the old one,
show the user what changed, ask for confirmation before overwriting.

---

## CLI

```bash
# First-time setup
openaugi configure --vault /path/to/vault

# Force refresh
openaugi configure --vault /path/to/vault --refresh

# Show current config without changing it
openaugi configure --show

# Skip LLM, just write defaults
openaugi configure --vault /path/to/vault --no-llm
```

On first `openaugi ingest`, if no config exists, prompt the user to run configure first
(or auto-run it if `--auto-configure` is passed).

---

## Integration with ingest + embed pipeline

`parse_vault` currently takes `exclude_patterns: list[str] | None`. Extend to accept a
`VaultConfig` object:

```python
def parse_vault(
    vault_path,
    config: VaultConfig | None = None,
    exclude_patterns: list[str] | None = None,  # kept for backward compat / tests
    ...
)
```

When `config` is present:
- `exclude_patterns` = `DEFAULT_EXCLUDE_PATTERNS + config.exclude_patterns`
- `document_level_patterns` fed into `_parse_file` alongside `_extract_wk_date`
- `min_block_chars` used in the meaningful-content filter

The embed pipeline reads `config.min_block_chars` and any vault-specific strip rules.

---

## Testing / eval

The eval harness:

```python
# tests/test_configure.py

EXPECTED_EXCLUDES = {
    "**/Readwise/**",
    "**/Instapaper/**",
    "**/Snipd/**",
    "**/4-Tech Notes/**",
}

EXPECTED_DOCUMENT_LEVEL = {
    "WK - *",   # or any pattern that catches weekly reflection naming
}

def test_analyzer_against_fixture_vault(fixture_vault_path, mock_llm):
    """LLM response for the fixture vault should produce the expected rules."""
    ...

def test_sampler_produces_all_folders(fixture_vault_path):
    """Sampler returns at least one sample per top-level folder."""
    ...
```

For integration eval (not in CI — requires real vault + API key):

```bash
openaugi configure --vault ~/zk-backup --refresh --dry-run
# Manually verify output matches EXPECTED_EXCLUDES and EXPECTED_DOCUMENT_LEVEL
```

---

## Implementation order

1. `config.py` — `VaultConfig` model + load/save. No LLM needed. Start here.
2. `sampler.py` — vault walk + stratified sampling. Testable without LLM.
3. `analyzer.py` — LLM call + response parsing. Unit-testable with mock LLM.
4. `cli.py` — wire together, add `openaugi configure` command.
5. Integration into `parse_vault` and embed pipeline.
6. Periodic refresh logic in the ingest runner.

---

## Open questions

- **How often to refresh?** 30 days is a guess. Vault structure changes slowly.
  Could also trigger on "folder count changed" heuristic.
- **User override?** Should the user be able to pin rules so a refresh can't remove them?
  Probably yes — a `pinned_excludes` list in the config that the agent never touches.
- **Model choice?** Haiku is probably fine for this — it's a classification/extraction task,
  not reasoning. Cheap and fast.
- **What to do with note_type_hints?** The LLM will likely identify `YT - *`, `PMOC - *`,
  `BOOK - *` etc. as note types. We don't use these yet but should capture them in config
  for future features (type-aware retrieval, type-specific prompts in augi-agent).
