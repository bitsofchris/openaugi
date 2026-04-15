#!/usr/bin/env python3
"""Phase 1: Sample blocks, label via local Claude agent, write labeled.json.

Strategy:
  Pass 1 — Taxonomy: show Claude ~30 sampled blocks, ask it to propose coarse
            (5-8) and fine (30-50) taxonomies from the data. Saves taxonomy.json.
  Pass 2 — Labeling: label all blocks in batches of 10, calling `claude -p`
            for each batch. Results saved incrementally — safe to interrupt
            and re-run (already-labeled blocks are skipped).

Block sampling is deterministic: same --seed always picks the same blocks.

After running, open output/labeled.json and manually correct any labels.
Disagreements between your judgement and Claude's are the most informative
signal — note them. Then run 02_linear_probe.py.

Usage:
    python 01_sample_and_label.py
    python 01_sample_and_label.py --db ~/.openaugi/openaugi.db --n 200 --seed 42
    python 01_sample_and_label.py --taxonomy-json output/taxonomy.json  # skip taxonomy pass
    python 01_sample_and_label.py --label-only   # re-run labeling with existing taxonomy
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

DEFAULT_DB = Path.home() / ".openaugi" / "openaugi.db"
OUT_DIR = Path(__file__).parent / "output"
TAXONOMY_SAMPLE_SIZE = 30
LABEL_BATCH_SIZE = 10


# ── DB ────────────────────────────────────────────────────────────────────────


def load_blocks(db_path: Path, n: int, seed: int) -> list[dict]:
    """Load n data_blocks with embeddings from SQLite, deterministically sampled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Pull all candidates, then sample in Python with a fixed seed
        # so --seed always yields the same block set regardless of SQLite RANDOM() behavior.
        rows = conn.execute(
            """
            SELECT
                id,
                content,
                title,
                block_time,
                json_extract(metadata, '$.source_path') AS source_path,
                embedding
            FROM blocks
            WHERE kind = 'data_block'
              AND embedding IS NOT NULL
              AND length(content) > 50
            ORDER BY id
            """,
        ).fetchall()
    finally:
        conn.close()

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))

    blocks = []
    for row in sampled:
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        date = row["block_time"][:7] if row["block_time"] else None
        blocks.append(
            {
                "block_id": row["id"],
                "text": (row["content"] or "").strip(),
                "source_path": row["source_path"] or "",
                "date": date,
                "embedding_3072": emb.tolist(),
            }
        )
    return blocks


# ── Agent calls ───────────────────────────────────────────────────────────────


def _find_claude() -> str:
    # Check PATH first, then common install locations
    found = shutil.which("claude")
    if found:
        return found
    fallbacks = [
        Path.home() / ".claude" / "local" / "claude",
        Path("/usr/local/bin/claude"),
        Path("/opt/homebrew/bin/claude"),
    ]
    for p in fallbacks:
        if p.exists():
            return str(p)
    print("ERROR: `claude` CLI not found. Install Claude Code.")
    sys.exit(1)


def _call_claude(claude_bin: str, prompt: str) -> str:
    """Call `claude -p <prompt>` and return stdout. Exits on non-zero return code."""
    result = subprocess.run(
        [claude_bin, "-p", prompt, "--output-format", "text"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}\nstderr: {result.stderr.strip()}")
    return result.stdout.strip()


def _parse_json_from_response(raw: str) -> object:
    """Extract JSON from a response that may include prose or markdown fences."""
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    if "```" in raw:
        for block in raw.split("```"):
            stripped = block.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    # Find first [ or { and try from there
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not extract JSON from response:\n{raw[:500]}")


# ── Pass 1: Taxonomy ──────────────────────────────────────────────────────────


def discover_taxonomy(claude_bin: str, sample: list[dict]) -> dict:
    print(f"\n[Pass 1] Discovering taxonomy from {len(sample)} sampled blocks...")

    blocks_text = "\n\n".join(f"[{i + 1}] {b['text'][:400]}" for i, b in enumerate(sample))

    prompt = textwrap.dedent(f"""\
        You are analyzing a personal journal from a second brain. Each entry is an
        atomic idea or reflection. Read these {len(sample)} blocks carefully, then
        propose two label taxonomies that emerge from the actual content — do not
        impose generic categories.

        BLOCKS:
        {blocks_text}

        ---

        Propose:

        1. COARSE taxonomy (5–8 categories): broad life/thinking domains visible
           in this corpus. Examples of the *kind* of thing this might be (not prescriptive):
           "productivity systems", "relationships", "health & body", "creative work",
           "learning", "philosophy/worldview", "career", "environment/place"

        2. FINE taxonomy (30–50 themes): specific recurring ideas, tensions, or patterns.
           Examples of the *kind* of thing (not prescriptive):
           "tension between structure and spontaneity", "morning routines and energy",
           "creative resistance", "note-taking workflows", "compounding knowledge"

        Return ONLY a JSON object, no other text:
        {{
          "coarse": ["label1", "label2", ...],
          "fine": ["theme1", "theme2", ...]
        }}
    """)

    raw = _call_claude(claude_bin, prompt)
    taxonomy = _parse_json_from_response(raw)

    print(f"  Coarse ({len(taxonomy['coarse'])}): {taxonomy['coarse']}")
    print(f"  Fine ({len(taxonomy['fine'])}): first 5 = {taxonomy['fine'][:5]}")
    return taxonomy


# ── Pass 2: Labeling ──────────────────────────────────────────────────────────


def label_batch(
    claude_bin: str,
    batch: list[dict],
    taxonomy: dict,
    batch_num: int,
    total_batches: int,
) -> list[dict]:
    coarse_list = "\n".join(f"  - {c}" for c in taxonomy["coarse"])
    fine_list = "\n".join(f"  - {f}" for f in taxonomy["fine"])

    blocks_text = "\n\n".join(f"[{b['block_id']}]\n{b['text'][:500]}" for b in batch)

    prompt = textwrap.dedent(f"""\
        You are labeling personal journal entries against an established taxonomy.
        For each block, pick the single best-fitting coarse label and fine label.
        If nothing fits perfectly, use the closest option — do not invent new labels.

        COARSE LABELS (pick one per block):
{coarse_list}

        FINE LABELS (pick one per block):
{fine_list}

        BLOCKS TO LABEL:
        {blocks_text}

        Return ONLY a JSON array, no other text:
        [
          {{"block_id": "...", "coarse_label": "...", "fine_label": "..."}},
          ...
        ]
    """)

    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} blocks)...", end="", flush=True)
    raw = _call_claude(claude_bin, prompt)
    parsed = _parse_json_from_response(raw)

    # unwrap if model returned {"results": [...]} instead of [...]
    if isinstance(parsed, dict):
        for key in ("results", "labels", "blocks", "data"):
            if key in parsed:
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)}")

    print(f" ok ({len(parsed)} labeled)")
    return parsed


# ── Incremental save/resume ───────────────────────────────────────────────────


def load_existing_labels(out_path: Path) -> dict[str, dict]:
    """Load any labels already saved to disk (for resume after interrupt)."""
    if not out_path.exists():
        return {}
    try:
        with open(out_path) as f:
            existing = json.load(f)
        labeled = {
            r["block_id"]: {"coarse_label": r["coarse_label"], "fine_label": r["fine_label"]}
            for r in existing
            if r.get("coarse_label") and r["coarse_label"] != "NEEDS_REVIEW"
        }
        if labeled:
            print(f"  Resuming: {len(labeled)} blocks already labeled, skipping them.")
        return labeled
    except (json.JSONDecodeError, KeyError):
        return {}


def save_labeled(out_path: Path, blocks: list[dict], labels_by_id: dict[str, dict]) -> None:
    """Write current state to disk. Called after every batch."""
    labeled = []
    for b in blocks:
        lbl = labels_by_id.get(
            b["block_id"], {"coarse_label": "NEEDS_REVIEW", "fine_label": "NEEDS_REVIEW"}
        )
        labeled.append(
            {
                "block_id": b["block_id"],
                "text": b["text"],
                "source_path": b["source_path"],
                "date": b["date"],
                "coarse_label": lbl["coarse_label"],
                "fine_label": lbl["fine_label"],
                "embedding_3072": b["embedding_3072"],
            }
        )
    with open(out_path, "w") as f:
        json.dump(labeled, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--n", type=int, default=200, help="Number of blocks to sample")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic sampling"
    )
    parser.add_argument("--out", default=str(OUT_DIR / "labeled.json"))
    parser.add_argument(
        "--taxonomy-json", default=None, help="Skip taxonomy pass: load from this file instead"
    )
    parser.add_argument(
        "--label-only",
        action="store_true",
        help="Skip sampling — re-use blocks already in --out, re-run labeling",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    claude_bin = _find_claude()
    print(f"Using claude at: {claude_bin}")

    # ── Load / sample blocks ──────────────────────────────────────────────────
    if args.label_only and out_path.exists():
        print(f"--label-only: loading blocks from {out_path}")
        with open(out_path) as f:
            existing = json.load(f)
        blocks = [
            {
                "block_id": r["block_id"],
                "text": r["text"],
                "source_path": r["source_path"],
                "date": r["date"],
                "embedding_3072": r["embedding_3072"],
            }
            for r in existing
        ]
    else:
        print(f"Loading {args.n} blocks from {db_path} (seed={args.seed})...")
        blocks = load_blocks(db_path, args.n, args.seed)
        if len(blocks) < args.n:
            print(f"  Warning: only {len(blocks)} blocks available (requested {args.n})")
        print(f"  Loaded {len(blocks)} blocks")

    if not blocks:
        print("ERROR: no blocks found")
        sys.exit(1)

    # ── Pass 1: Taxonomy ──────────────────────────────────────────────────────
    taxonomy_path = out_path.parent / "taxonomy.json"

    if args.taxonomy_json:
        with open(args.taxonomy_json) as f:
            taxonomy = json.load(f)
        print(f"[Pass 1] Loaded taxonomy from {args.taxonomy_json}")
    elif taxonomy_path.exists() and not args.label_only:
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        print(f"[Pass 1] Loaded existing taxonomy from {taxonomy_path}")
    else:
        rng = random.Random(args.seed)
        taxonomy_sample = rng.sample(blocks, min(TAXONOMY_SAMPLE_SIZE, len(blocks)))
        taxonomy = discover_taxonomy(claude_bin, taxonomy_sample)
        with open(taxonomy_path, "w") as f:
            json.dump(taxonomy, f, indent=2)
        print(f"  Saved → {taxonomy_path}")

    # ── Pass 2: Labeling ──────────────────────────────────────────────────────
    labels_by_id = load_existing_labels(out_path)
    unlabeled = [b for b in blocks if b["block_id"] not in labels_by_id]

    print(f"\n[Pass 2] Labeling {len(unlabeled)} blocks in batches of {LABEL_BATCH_SIZE}...")

    batches = [
        unlabeled[i : i + LABEL_BATCH_SIZE] for i in range(0, len(unlabeled), LABEL_BATCH_SIZE)
    ]
    total_batches = len(batches)
    failed_batches = 0

    for i, batch in enumerate(batches):
        try:
            results = label_batch(claude_bin, batch, taxonomy, i + 1, total_batches)
            for r in results:
                labels_by_id[r["block_id"]] = {
                    "coarse_label": r.get("coarse_label", "NEEDS_REVIEW"),
                    "fine_label": r.get("fine_label", "NEEDS_REVIEW"),
                }
        except Exception as e:
            print(f"\n  ERROR on batch {i + 1}: {e}")
            failed_batches += 1
            for b in batch:
                labels_by_id[b["block_id"]] = {
                    "coarse_label": "NEEDS_REVIEW",
                    "fine_label": "NEEDS_REVIEW",
                }

        # Save after every batch — safe to interrupt
        save_labeled(out_path, blocks, labels_by_id)

    # ── Summary ───────────────────────────────────────────────────────────────
    needs_review = sum(1 for v in labels_by_id.values() if v["coarse_label"] == "NEEDS_REVIEW")
    print(f"\n✓ {len(blocks)} blocks → {out_path}")
    if needs_review:
        print(f"  {needs_review} blocks marked NEEDS_REVIEW")
    if failed_batches:
        print(f"  {failed_batches}/{total_batches} batches failed — re-run to retry")

    from collections import Counter

    coarse_counts = Counter(v["coarse_label"] for v in labels_by_id.values())
    print("\nCoarse label distribution:")
    for label, count in coarse_counts.most_common():
        bar = "█" * (count // max(1, len(blocks) // 40))
        print(f"  {label:<40} {count:3d}  {bar}")

    print(f"\nNext: review {out_path}, correct any mislabels, then run 02_linear_probe.py")


if __name__ == "__main__":
    main()
