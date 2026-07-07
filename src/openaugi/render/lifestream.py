"""Lifestream — merged chronological block stream + activity heat strip.

The first M6 render surface: every data block across all sources on one
time axis, with a commit-graph-style heat strip and client-side filters
(search, area, day). Answers what the Dashboard can't: the *shape* of
recent weeks — density, gaps, where activity lives.

Output is one self-contained HTML file (data inlined as JSON, no server,
no dependencies). Default target: `<vault>/OpenAugi/render/lifestream.html`
— derived and regenerable like everything else the agent writes.
"""

# ruff: noqa: E501 — the embedded HTML/CSS/JS template has long lines by nature

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

OUTPUT_RELPATH = "OpenAugi/render/lifestream.html"
DEFAULT_DAYS = 180
MAX_BLOCKS = 20000
SNIPPET_LEN = 240

_WS_RE = re.compile(r"\s+")


def _snippet(content: str | None) -> str:
    text = _WS_RE.sub(" ", content or "").strip()
    return text[: SNIPPET_LEN - 1] + "…" if len(text) > SNIPPET_LEN else text


def _norm_tag(tag: str) -> str:
    return tag.lstrip("#")


def build_lifestream_data(
    store: SQLiteStore, days: int = DEFAULT_DAYS, limit: int = MAX_BLOCKS
) -> dict:
    """Assemble the lifestream payload: user blocks by day, newest first.

    Excludes `OpenAugi/`-sourced blocks (derived output, not life). Tags
    merge the user's own (`block.tags`) with agent-applied `augi_tags`.
    """
    items = []
    for b in store.get_recent_blocks(days=days, kind="data_block", limit=limit):
        src = b.metadata.get("source_path", "")
        if src.startswith("OpenAugi/"):
            continue
        tags = [_norm_tag(t) for t in b.tags]
        for t in b.metadata.get("augi_tags", []) or []:
            t = _norm_tag(t)
            if t not in tags:
                tags.append(t)
        area = next((t.split("/", 1)[1] for t in tags if t.startswith("area/")), None)
        items.append(
            {
                "d": (b.block_time or "")[:10],
                "s": _snippet(b.content),
                "src": src,
                "tags": tags,
                "a": area,
            }
        )
    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "days": days,
        "blocks": items,
    }


def render_lifestream(
    store: SQLiteStore,
    vault_path: str | Path,
    days: int = DEFAULT_DAYS,
    out: str | Path | None = None,
) -> Path:
    """Write the lifestream HTML. Returns the output path."""
    data = build_lifestream_data(store, days=days)
    out_path = Path(out) if out else Path(vault_path) / OUTPUT_RELPATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out_path.write_text(_TEMPLATE.replace("/*__DATA__*/null", payload))
    logger.info(
        "Wrote lifestream: %s (%d blocks, %dd window)", out_path, len(data["blocks"]), days
    )
    return out_path


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenAugi — Lifestream</title>
<style>
  :root{--bg:#0f1117;--panel:#161a23;--panel2:#1c2130;--border:#2a3145;
    --text:#d8dce8;--dim:#8b93a7;--accent:#e8a33d;--accent2:#5eb3f6;--green:#6fc48a;
    --mono:'SF Mono',Menlo,Consolas,monospace}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;padding:0 0 80px}
  .wrap{max-width:1080px;margin:0 auto;padding:0 24px}
  header{padding:32px 0 16px;border-bottom:1px solid var(--border);margin-bottom:16px}
  h1{font-size:22px;font-weight:700}h1 span{color:var(--accent)}
  .sub{color:var(--dim);font-size:13px;margin-top:4px}
  #heat{display:flex;gap:3px;margin:18px 0 6px;overflow-x:auto;padding-bottom:6px}
  .week{display:flex;flex-direction:column;gap:3px}
  .day{width:11px;height:11px;border-radius:2px;background:var(--panel2);cursor:pointer}
  .day.l1{background:#274a33}.day.l2{background:#2f6b44}.day.l3{background:#43975e}.day.l4{background:#6fc48a}
  .day.sel{outline:2px solid var(--accent)}
  #controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0 22px}
  #q{background:var(--panel);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:8px 12px;font-size:14px;width:260px}
  .chip{font:600 12px var(--mono);border:1px solid var(--border);border-radius:14px;padding:4px 11px;cursor:pointer;color:var(--dim);background:var(--panel)}
  .chip.on{color:var(--bg);background:var(--accent2);border-color:var(--accent2)}
  #count{color:var(--dim);font-size:13px;margin-left:auto}
  .dayhead{font:600 13px var(--mono);color:var(--accent2);margin:26px 0 8px;position:sticky;top:0;background:var(--bg);padding:6px 0;border-bottom:1px solid var(--border)}
  .dayhead .n{color:var(--dim);font-weight:400}
  .blk{background:var(--panel);border:1px solid var(--border);border-radius:9px;padding:10px 14px;margin:8px 0}
  .blk .meta{color:var(--dim);font:11.5px var(--mono);margin-top:6px;display:flex;flex-wrap:wrap;gap:10px}
  .blk .meta .tag{color:var(--green)}
  #more{display:block;margin:28px auto;background:var(--panel2);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:10px 22px;cursor:pointer;font-size:14px}
  mark{background:rgba(232,163,61,.35);color:inherit;border-radius:2px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>OpenAugi — <span>lifestream</span></h1>
  <div class="sub" id="gen"></div>
  <div id="heat"></div>
  <div id="controls">
    <input id="q" type="search" placeholder="search blocks…">
    <span id="chips"></span>
    <span id="count"></span>
  </div>
</header>
<div id="stream"></div>
<button id="more" hidden>show more</button>
</div>
<script>
const DATA = /*__DATA__*/null;
const PAGE = 400;
let fArea = null, fDay = null, q = '', shown = PAGE;

const el = id => document.getElementById(id);
el('gen').textContent = `${DATA.blocks.length} blocks · last ${DATA.days} days · generated ${DATA.generatedAt.slice(0,10)} · derived + regenerable (openaugi render)`;

// ── heat strip ──
const counts = {};
for (const b of DATA.blocks) if (b.d) counts[b.d] = (counts[b.d]||0)+1;
const end = new Date(); const start = new Date(end - DATA.days*864e5);
start.setDate(start.getDate() - start.getDay()); // align to Sunday
const max = Math.max(1, ...Object.values(counts));
const lvl = c => c===0?0:c<=max*.25?1:c<=max*.5?2:c<=max*.75?3:4;
let heatHtml = '', wk = [];
for (let d = new Date(start); d <= end; d.setDate(d.getDate()+1)) {
  const iso = d.toISOString().slice(0,10), c = counts[iso]||0;
  wk.push(`<div class="day l${lvl(c)}" data-d="${iso}" title="${iso} — ${c} block${c===1?'':'s'}"></div>`);
  if (d.getDay()===6) { heatHtml += `<div class="week">${wk.join('')}</div>`; wk=[]; }
}
if (wk.length) heatHtml += `<div class="week">${wk.join('')}</div>`;
el('heat').innerHTML = heatHtml;
el('heat').onclick = e => {
  const d = e.target.dataset.d; if (!d) return;
  fDay = (fDay === d) ? null : d; shown = PAGE; render();
};

// ── area chips ──
const areas = [...new Set(DATA.blocks.map(b=>b.a).filter(Boolean))].sort();
el('chips').innerHTML = areas.map(a=>`<button class="chip" data-a="${a}">${a}</button>`).join(' ');
el('chips').onclick = e => {
  const a = e.target.dataset.a; if (!a) return;
  fArea = (fArea === a) ? null : a; shown = PAGE; render();
};
el('q').oninput = e => { q = e.target.value.toLowerCase(); shown = PAGE; render(); };

const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
const hi = s => q ? esc(s).replace(new RegExp(q.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&'),'gi'), m=>`<mark>${m}</mark>`) : esc(s);

function render() {
  document.querySelectorAll('#heat .day').forEach(d=>d.classList.toggle('sel', d.dataset.d===fDay));
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on', c.dataset.a===fArea));
  const match = DATA.blocks.filter(b =>
    (!fArea || b.a===fArea) && (!fDay || b.d===fDay) &&
    (!q || (b.s+' '+b.src+' '+b.tags.join(' ')).toLowerCase().includes(q)));
  el('count').textContent = `${match.length} block${match.length===1?'':'s'}`;
  let html = '', lastDay = null;
  for (const b of match.slice(0, shown)) {
    if (b.d !== lastDay) {
      const n = match.filter(x=>x.d===b.d).length;
      html += `<div class="dayhead">${b.d||'undated'} <span class="n">· ${n}</span></div>`;
      lastDay = b.d;
    }
    html += `<div class="blk">${hi(b.s)||'<span class="dim">(empty)</span>'}<div class="meta"><span>${esc(b.src)}</span>${b.tags.map(t=>`<span class="tag">#${esc(t)}</span>`).join('')}</div></div>`;
  }
  el('stream').innerHTML = html;
  el('more').hidden = match.length <= shown;
  el('more').textContent = `show more (${match.length - shown} remaining)`;
}
el('more').onclick = () => { shown += PAGE; render(); };
render();
</script>
</body>
</html>
"""
