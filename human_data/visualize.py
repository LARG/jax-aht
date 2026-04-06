"""
human_data/visualize.py

Detects low-effort players in collected LBF (Level-Based Foraging) game data.

Terminology:
  Session    = one player sitting down to play (identified by session ID)
  Game       = one round within a session = one JSON file (up to 8 per session)
  Trajectory = the sequence of actions/states recorded within a single game

Produces:
  human_data/plots/effort_analysis.html  — interactive paginated dashboard
  (also prints a flagged-session report to stdout)

Usage (on server):
  conda activate jax-aht
  cd ~/aht/jax-aht
  python human_data/visualize.py
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("human_data/collected_data")
PLOTS_DIR = Path("human_data/plots")

NOOP_ACTION     = 5
FLAG_SCORE      = 0.0
FLAG_NOOP       = 0.60
BORDERLINE_NOOP = 0.45
MIN_LEVELS      = 1
PAGE_SIZE       = 15   # sessions per page

# ── Load & aggregate ──────────────────────────────────────────────────────────

def load_episodes(data_dir: Path) -> list[dict]:
    episodes, missing = [], []
    for fpath in sorted(data_dir.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            data["_file"] = fpath.name
            episodes.append(data)
        except (json.JSONDecodeError, OSError) as e:
            missing.append((fpath.name, str(e)))
    if missing:
        print(f"[warn] Could not load {len(missing)} file(s):")
        for name, err in missing:
            print(f"  {name}: {err}")
    print(f"[info] Loaded {len(episodes)} games from {data_dir}")
    return episodes


def compute_session_stats(episodes: list[dict]) -> list[dict]:
    sessions = defaultdict(lambda: {
        "scores": [], "noop_counts": [], "total_steps": [],
        "num_levels": 0, "level_details": [], "timestamps": []
    })

    for ep in episodes:
        sid         = ep.get("session_id", "unknown")
        human_score = ep.get("total_rewards", {}).get("agent_0", 0.0)
        trajectory  = ep.get("trajectory", [])
        total_steps = len(trajectory)
        noop_count  = sum(1 for s in trajectory if s.get("human_action") == NOOP_ACTION)
        timestamp   = ep.get("timestamp", "")

        sess = sessions[sid]
        sess["scores"].append(human_score)
        sess["noop_counts"].append(noop_count)
        sess["total_steps"].append(total_steps)
        sess["num_levels"] += 1
        sess["timestamps"].append(timestamp)
        sess["level_details"].append({
            "score":      human_score,
            "noop_count": noop_count,
            "steps":      total_steps,
            "noop_rate":  noop_count / total_steps if total_steps else 0.0,
            "timestamp":  timestamp,
            "file":       ep.get("_file", ""),
        })

    rows = []
    for sid, s in sessions.items():
        if s["num_levels"] < MIN_LEVELS:
            continue
        total_actions = sum(s["total_steps"])
        total_noops   = sum(s["noop_counts"])
        avg_score     = float(np.mean(s["scores"]))
        noop_rate     = total_noops / total_actions if total_actions else 0.0

        if avg_score <= FLAG_SCORE and noop_rate >= FLAG_NOOP:
            status = "flagged"
        elif avg_score <= FLAG_SCORE and noop_rate < FLAG_NOOP:
            status = "zero_active"
        elif noop_rate >= BORDERLINE_NOOP and avg_score <= 0.15:
            status = "borderline"
        else:
            status = "ok"

        # sort level details by timestamp
        level_details = sorted(s["level_details"], key=lambda x: x["timestamp"])

        rows.append({
            "session_id":    sid,
            "session_short": sid[:8],
            "num_levels":    s["num_levels"],
            "avg_score":     avg_score,
            "score_std":     float(np.std(s["scores"])),
            "noop_rate":     noop_rate,
            "status":        status,
            "level_details": level_details,
        })

    rows.sort(key=lambda x: x["avg_score"])
    return rows


# ── HTML generation ───────────────────────────────────────────────────────────

def score_to_color(score: float, max_score: float = 0.5) -> str:
    """Map score to a green shade."""
    pct = min(score / max_score, 1.0) if max_score > 0 else 0
    r = int(220 - pct * 150)
    g = int(120 + pct * 120)
    b = int(100 - pct * 60)
    return f"rgb({r},{g},{b})"

def noop_to_color(noop_rate: float) -> str:
    """Map noop rate to red/yellow/blue."""
    pct = min(noop_rate, 1.0)
    if pct >= FLAG_NOOP:
        return "#e05050"
    elif pct >= BORDERLINE_NOOP:
        return "#e09020"
    else:
        r = int(80  + pct * 100)
        g = int(120 - pct * 40)
        b = int(200 - pct * 100)
        return f"rgb({r},{g},{b})"

def status_badge(status: str) -> str:
    cfg = {
        "flagged":    ("REMOVE",      "#c0392b", "#fff"),
        "zero_active":("REVIEW",      "#d68910", "#fff"),
        "borderline": ("BORDERLINE",  "#ca6f1e", "#fff"),
        "ok":         ("OK",          "#1e8449", "#fff"),
    }
    label, bg, fg = cfg[status]
    return f'<span class="badge" style="background:{bg};color:{fg}">{label}</span>'

def bar_html(value: float, max_val: float = 0.5, width_pct: int = 100) -> str:
    pct = min(value / max_val, 1.0) * width_pct if max_val > 0 else 0
    color = score_to_color(value, max_val)
    return (f'<div class="bar-wrap">'
            f'<div class="bar" style="width:{pct:.1f}%;background:{color}"></div>'
            f'<span class="bar-label">{value:.3f}</span>'
            f'</div>')

def level_sparkline(levels: list[dict]) -> str:
    """Mini per-level score dots."""
    dots = []
    for i, lv in enumerate(levels):
        color = score_to_color(lv["score"])
        ncolor = noop_to_color(lv["noop_rate"])
        tip = (f"Game {i+1}: score={lv['score']:.3f}, "
               f"noop={lv['noop_rate']:.0%}, steps={lv['steps']}")
        dots.append(
            f'<div class="spark-dot" title="{tip}" '
            f'style="background:{color};border:2px solid {ncolor}"></div>'
        )
    return f'<div class="sparkline">{"".join(dots)}</div>'

def render_level_detail_table(levels: list[dict]) -> str:
    rows = []
    for i, lv in enumerate(levels):
        nc = noop_to_color(lv["noop_rate"])
        sc = score_to_color(lv["score"])
        rows.append(
            f'<tr>'
            f'<td>Game {i+1}</td>'
            f'<td>{bar_html(lv["score"], 0.5, 80)}</td>'
            f'<td style="color:{nc};font-weight:bold">{lv["noop_rate"]:.1%}</td>'
            f'<td>{lv["steps"]}</td>'
            f'<td>{lv["noop_count"]}</td>'
            f'</tr>'
        )
    return (
        '<table class="detail-table">'
        '<thead><tr>'
        '<th>#</th><th>Score</th><th>Noop rate</th>'
        '<th>Steps (trajectory length)</th><th>Noop steps</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )

def build_html(sessions: list[dict], generated: str) -> str:
    n_flagged    = sum(1 for s in sessions if s["status"] == "flagged")
    n_borderline = sum(1 for s in sessions if s["status"] == "borderline")
    n_review     = sum(1 for s in sessions if s["status"] == "zero_active")
    n_ok         = sum(1 for s in sessions if s["status"] == "ok")
    total_levels = sum(s["num_levels"] for s in sessions)
    avg_score    = float(np.mean([s["avg_score"] for s in sessions]))
    avg_noop     = float(np.mean([s["noop_rate"] for s in sessions]))

    # Build session rows JSON for JS pagination
    session_rows_html = []
    for s in sessions:
        sid   = s["session_id"]
        short = s["session_short"]
        stat  = s["status"]
        badge = status_badge(stat)
        bar   = bar_html(s["avg_score"])
        spark = level_sparkline(s["level_details"])
        nc    = noop_to_color(s["noop_rate"])
        std_s = f"±{s['score_std']:.3f} score" if s["num_levels"] > 1 else "—"
        detail_tbl = render_level_detail_table(s["level_details"])

        row_class = {
            "flagged":    "row-flagged",
            "zero_active":"row-review",
            "borderline": "row-borderline",
            "ok":         "",
        }[stat]

        session_rows_html.append(
            f'<div class="session-row {row_class}" data-status="{stat}">'
            f'  <div class="row-main" onclick="toggleDetail(this)">'
            f'    <div class="col-id">'
            f'      <span class="sid-full" title="{sid}">{short}…</span>'
            f'      {badge}'
            f'    </div>'
            f'    <div class="col-bar">{bar}</div>'
            f'    <div class="col-noop" style="color:{nc};font-weight:bold">'
            f'      {s["noop_rate"]:.1%}'
            f'    </div>'
            f'    <div class="col-levels">{s["num_levels"]} / 8</div>'
            f'    <div class="col-std">{std_s}</div>'
            f'    <div class="col-spark">{spark}</div>'
            f'    <div class="col-expand">▼</div>'
            f'  </div>'
            f'  <div class="row-detail" style="display:none">'
            f'    <div class="detail-inner">'
            f'      <div class="detail-sid">Full session ID: <code>{sid}</code></div>'
            f'      {detail_tbl}'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )

    rows_html = "\n".join(session_rows_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LBF Human Player Effort Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    background: #f4f5f7;
    color: #222;
  }}

  /* ── Header ── */
  .header {{
    background: #1a1f2e;
    color: white;
    padding: 18px 28px 14px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  .header h1 {{ font-size: 21px; font-weight: 700; letter-spacing: 0.3px; }}
  .header .subtitle {{
    font-size: 13.5px; color: #9aa; margin-top: 4px;
  }}
  .header-meta {{
    display: flex; gap: 24px; margin-top: 10px; flex-wrap: wrap;
  }}
  .stat-pill {{
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 13px;
    display: flex; align-items: center; gap: 7px;
  }}
  .stat-pill .val {{ font-weight: 700; font-size: 17px; }}
  .stat-pill.flagged .val {{ color: #e05050; }}
  .stat-pill.borderline .val {{ color: #e09020; }}
  .stat-pill.review .val {{ color: #d4ac0d; }}
  .stat-pill.ok .val {{ color: #52be80; }}

  /* ── Legend ── */
  .legend {{
    background: white;
    border-bottom: 1px solid #e0e0e0;
    padding: 8px 28px;
    display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
  }}
  .legend-title {{ font-weight: 600; color: #555; font-size: 11.5px; }}
  .legend-item {{
    display: flex; align-items: center; gap: 5px;
    font-size: 11.5px; color: #444;
  }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 2px;
  }}

  /* ── Controls ── */
  .controls {{
    background: white;
    border-bottom: 1px solid #e8e8e8;
    padding: 8px 28px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }}
  .controls label {{ font-size: 12px; color: #555; font-weight: 500; }}
  .filter-btn {{
    padding: 4px 12px;
    border-radius: 14px;
    border: 1.5px solid #ccc;
    background: white;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.15s;
  }}
  .filter-btn.active {{
    border-color: #2d6cdf;
    background: #2d6cdf;
    color: white;
    font-weight: 600;
  }}
  .sort-select {{
    padding: 4px 8px;
    border-radius: 6px;
    border: 1.5px solid #ccc;
    font-size: 12px;
    background: white;
    cursor: pointer;
  }}

  /* ── Table header ── */
  .table-header {{
    display: grid;
    grid-template-columns: 200px 1fr 90px 80px 70px 120px 24px;
    gap: 0;
    padding: 7px 28px;
    background: #eef0f4;
    border-bottom: 2px solid #d0d4dc;
    font-weight: 700;
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    position: sticky;
    top: 102px;
    z-index: 90;
  }}

  /* ── Session rows ── */
  .session-list {{ padding: 0 12px 12px; }}

  .session-row {{
    background: white;
    border-radius: 6px;
    margin: 4px 0;
    border: 1px solid #e8e8e8;
    overflow: hidden;
    transition: box-shadow 0.15s;
  }}
  .session-row:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .session-row.row-flagged {{
    border-left: 4px solid #c0392b;
    background: #fff8f8;
  }}
  .session-row.row-review {{
    border-left: 4px solid #d4ac0d;
    background: #fffdf0;
  }}
  .session-row.row-borderline {{
    border-left: 4px solid #ca6f1e;
    background: #fff9f2;
  }}

  .row-main {{
    display: grid;
    grid-template-columns: 200px 1fr 90px 80px 70px 120px 24px;
    gap: 0;
    padding: 9px 16px;
    cursor: pointer;
    align-items: center;
  }}
  .row-main:hover {{ background: rgba(0,0,0,0.02); }}

  .col-id {{
    display: flex; align-items: center; gap: 7px; min-width: 0;
  }}
  .sid-full {{
    font-family: monospace;
    font-size: 12px;
    color: #333;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .badge {{
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 10px;
    white-space: nowrap;
    flex-shrink: 0;
  }}

  /* bar */
  .bar-wrap {{
    display: flex; align-items: center; gap: 6px;
    padding-right: 10px;
  }}
  .bar {{
    height: 14px;
    border-radius: 3px;
    min-width: 2px;
    transition: width 0.3s;
  }}
  .bar-label {{
    font-size: 11.5px;
    font-weight: 600;
    color: #333;
    white-space: nowrap;
  }}

  .col-noop, .col-levels, .col-std {{
    font-size: 12px;
    padding: 0 4px;
  }}
  .col-std {{ color: #777; }}

  /* sparkline */
  .sparkline {{
    display: flex; gap: 3px; align-items: center; flex-wrap: wrap;
  }}
  .spark-dot {{
    width: 20px; height: 20px;
    border-radius: 5px;
    cursor: help;
    transition: transform 0.15s, box-shadow 0.15s;
    border-width: 3px !important;
    flex-shrink: 0;
  }}
  .spark-dot:hover {{
    transform: scale(1.55);
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    z-index: 10;
    position: relative;
  }}
  .scale-dot {{
    width: 17px; height: 17px;
    border-radius: 4px;
    border: 2px solid transparent;
    display: inline-block;
    flex-shrink: 0;
  }}
  .legend-divider {{
    width: 1px; background: #ddd; height: 38px; margin: 0 10px; align-self: center;
  }}

  .col-expand {{
    font-size: 11px; color: #aaa; text-align: center;
    transition: transform 0.2s;
  }}
  .col-expand.open {{ transform: rotate(180deg); }}

  /* ── Detail panel ── */
  .row-detail {{
    border-top: 1px solid #eee;
    background: #fafafa;
  }}
  .detail-inner {{
    padding: 12px 20px 14px;
  }}
  .detail-sid {{
    font-size: 11.5px;
    color: #666;
    margin-bottom: 10px;
  }}
  .detail-sid code {{
    font-size: 11.5px;
    background: #eee;
    padding: 1px 5px;
    border-radius: 3px;
    user-select: all;
  }}

  .detail-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .detail-table th {{
    background: #eef0f4;
    padding: 5px 10px;
    text-align: left;
    font-size: 11px;
    color: #555;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}
  .detail-table td {{
    padding: 5px 10px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
  }}
  .detail-table tr:last-child td {{ border-bottom: none; }}
  .detail-table tr:hover td {{ background: #f5f7ff; }}

  /* ── Pagination ── */
  .pagination {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    padding: 16px;
    background: white;
    border-top: 1px solid #e8e8e8;
    margin-top: 8px;
    border-radius: 0 0 8px 8px;
  }}
  .page-btn {{
    padding: 5px 14px;
    border: 1.5px solid #ccc;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.15s;
  }}
  .page-btn:hover:not(:disabled) {{
    background: #2d6cdf;
    color: white;
    border-color: #2d6cdf;
  }}
  .page-btn:disabled {{
    opacity: 0.35;
    cursor: not-allowed;
  }}
  .page-info {{
    font-size: 12.5px;
    color: #555;
    min-width: 120px;
    text-align: center;
  }}
  .page-dots {{ display: flex; gap: 5px; }}
  .page-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #ddd;
    cursor: pointer;
    transition: background 0.15s;
  }}
  .page-dot.active {{ background: #2d6cdf; }}

  /* ── Empty state ── */
  .empty-state {{
    text-align: center;
    padding: 40px;
    color: #999;
    font-size: 13px;
  }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 12px;
    font-size: 11px;
    color: #aaa;
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>LBF Human Player Effort Analysis</h1>
  <div class="subtitle">
    Each row = one player session &nbsp;·&nbsp;
    Bar = avg score per game &nbsp;·&nbsp;
    Click any row to expand per-level breakdown &nbsp;·&nbsp;
    Generated {generated}
  </div>
  <div class="header-meta">
    <div class="stat-pill">
      <span>Sessions</span><span class="val">{len(sessions)}</span>
    </div>
    <div class="stat-pill">
      <span>Total games</span><span class="val">{total_levels}</span>
    </div>
    <div class="stat-pill flagged">
      <span>⚑ Remove</span><span class="val">{n_flagged}</span>
    </div>
    <div class="stat-pill borderline">
      <span>~ Borderline</span><span class="val">{n_borderline}</span>
    </div>
    <div class="stat-pill review">
      <span>? Review</span><span class="val">{n_review}</span>
    </div>
    <div class="stat-pill ok">
      <span>✓ OK</span><span class="val">{n_ok}</span>
    </div>
    <div class="stat-pill">
      <span>Avg score</span><span class="val">{avg_score:.3f}</span>
    </div>
    <div class="stat-pill">
      <span>Avg noop</span><span class="val">{avg_noop:.1%}</span>
    </div>
  </div>
</div>

<!-- Legend -->
<div class="legend">
  <span class="legend-title">ROW STATUS</span>
  <div class="legend-item">
    <div class="legend-dot" style="background:#c0392b"></div>
    &#9873; REMOVE &mdash; score = 0 AND noop &ge; 60%
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#d4ac0d"></div>
    ? REVIEW &mdash; zero score but actively moving
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#ca6f1e"></div>
    ~ BORDERLINE &mdash; high noop + low score
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#1e8449"></div>
    &#10003; OK &mdash; normal player
  </div>

  <div class="legend-divider"></div>
  <span class="legend-title">SPARK DOTS</span>

  <div class="legend-item" style="flex-direction:column;align-items:flex-start;gap:3px">
    <span style="font-size:10px;color:#777;font-weight:600;text-transform:uppercase;letter-spacing:.4px">Fill = score (0 &rarr; 0.5)</span>
    <div style="display:flex;align-items:center;gap:3px">
      <div class="scale-dot" style="background:rgb(220,80,80)"></div>
      <div class="scale-dot" style="background:rgb(210,140,60)"></div>
      <div class="scale-dot" style="background:rgb(190,185,55)"></div>
      <div class="scale-dot" style="background:rgb(130,200,65)"></div>
      <div class="scale-dot" style="background:rgb(70,220,60)"></div>
      <span style="font-size:10px;color:#888;margin-left:4px">low &rarr; high</span>
    </div>
  </div>

  <div class="legend-item" style="flex-direction:column;align-items:flex-start;gap:3px">
    <span style="font-size:10px;color:#777;font-weight:600;text-transform:uppercase;letter-spacing:.4px">Border = noop rate (0% &rarr; 100%)</span>
    <div style="display:flex;align-items:center;gap:3px">
      <div class="scale-dot" style="background:#eee;border:3px solid rgb(80,120,200)"></div>
      <div class="scale-dot" style="background:#eee;border:3px solid rgb(120,130,170)"></div>
      <div class="scale-dot" style="background:#eee;border:3px solid rgb(180,140,60)"></div>
      <div class="scale-dot" style="background:#eee;border:3px solid rgb(210,100,50)"></div>
      <div class="scale-dot" style="background:#eee;border:3px solid #e05050"></div>
      <span style="font-size:10px;color:#888;margin-left:4px">active &rarr; idle</span>
    </div>
  </div>

  <div class="legend-item" style="flex-direction:column;align-items:flex-start;gap:4px">
    <span style="font-size:10px;color:#777;font-weight:600;text-transform:uppercase;letter-spacing:.4px">Examples</span>
    <div style="display:flex;gap:5px;align-items:center">
      <div class="scale-dot" style="background:rgb(70,220,60);border:3px solid rgb(80,120,200)"></div>
      <span style="font-size:10px;color:#333">Good: scored well, stayed active</span>
    </div>
    <div style="display:flex;gap:5px;align-items:center;margin-top:2px">
      <div class="scale-dot" style="background:rgb(220,80,80);border:3px solid #e05050"></div>
      <span style="font-size:10px;color:#333">Bad: scored nothing, mostly idle</span>
    </div>
  </div>
</div>

<!-- Floating tooltip -->
<div id="sparkTooltip" style="
  display:none; position:fixed;
  background:#1a1f2e; color:white;
  padding:9px 13px; border-radius:8px;
  font-size:12px; line-height:1.7;
  pointer-events:none; z-index:9999;
  box-shadow:0 4px 18px rgba(0,0,0,0.35);
  min-width:180px;
"></div>

<!-- Controls -->
<div class="controls">
  <span class="legend-title">FILTER</span>
  <button class="filter-btn active" onclick="setFilter('all', this)">All ({len(sessions)})</button>
  <button class="filter-btn" onclick="setFilter('flagged', this)">⚑ Remove ({n_flagged})</button>
  <button class="filter-btn" onclick="setFilter('borderline', this)">~ Borderline ({n_borderline})</button>
  <button class="filter-btn" onclick="setFilter('zero_active', this)">? Review ({n_review})</button>
  <button class="filter-btn" onclick="setFilter('ok', this)">✓ OK ({n_ok})</button>
  &nbsp;&nbsp;
  <label>Sort by</label>
  <select class="sort-select" onchange="setSort(this.value)">
    <option value="score-asc">Score (low → high)</option>
    <option value="score-desc">Score (high → low)</option>
    <option value="noop-desc">Noop rate (high → low)</option>
    <option value="noop-asc">Noop rate (low → high)</option>
    <option value="levels-desc">Levels completed (most → least)</option>
  </select>
</div>

<!-- Table header -->
<div class="table-header">
  <div>Session ID</div>
  <div>Avg score per game <span style="font-weight:400;text-transform:none">(max ≈ 0.5)</span></div>
  <div>Noop rate</div>
  <div>Games played</div>
  <div>Score std dev</div>
  <div>Per-level scores</div>
  <div></div>
</div>

<!-- Session list -->
<div class="session-list" id="sessionList">
{rows_html}
</div>

<!-- Pagination -->
<div class="pagination" id="pagination">
  <button class="page-btn" id="prevBtn" onclick="changePage(-1)">← Prev</button>
  <div class="page-dots" id="pageDots"></div>
  <span class="page-info" id="pageInfo"></span>
  <button class="page-btn" id="nextBtn" onclick="changePage(1)">Next →</button>
</div>

<div class="footer">
  LBF Human Player Effort Analysis &nbsp;·&nbsp;
  LARG @ UT Austin &nbsp;·&nbsp;
  Generated {generated}
</div>

<script>
const PAGE_SIZE = """ + str(PAGE_SIZE) + """;
let currentPage = 0;
let currentFilter = 'all';
let currentSort = 'score-asc';

function getAllRows() {
  return Array.from(document.querySelectorAll('.session-row'));
}

function getVisibleRows() {
  return getAllRows().filter(r => {
    if (currentFilter === 'all') return true;
    return r.dataset.status === currentFilter;
  });
}

function sortRows(rows) {
  rows.sort((a, b) => {
    switch(currentSort) {
      case 'score-asc':   return parseFloat(a.dataset.score||0) - parseFloat(b.dataset.score||0);
      case 'score-desc':  return parseFloat(b.dataset.score||0) - parseFloat(a.dataset.score||0);
      case 'noop-desc':   return parseFloat(b.dataset.noop||0)  - parseFloat(a.dataset.noop||0);
      case 'noop-asc':    return parseFloat(a.dataset.noop||0)  - parseFloat(b.dataset.noop||0);
      case 'levels-desc': return parseInt(b.dataset.levels||0)  - parseInt(a.dataset.levels||0);
      default: return 0;
    }
  });
  return rows;
}

function render() {
  const list = document.getElementById('sessionList');
  const all  = getAllRows();

  all.forEach(r => r.style.display = 'none');

  let visible = getVisibleRows();
  visible = sortRows(visible);

  const totalPages = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  currentPage = Math.min(currentPage, totalPages - 1);

  const start = currentPage * PAGE_SIZE;
  const page  = visible.slice(start, start + PAGE_SIZE);

  page.forEach(r => {
    r.style.display = '';
    list.appendChild(r);
  });

  document.getElementById('prevBtn').disabled = currentPage === 0;
  document.getElementById('nextBtn').disabled = currentPage >= totalPages - 1;
  document.getElementById('pageInfo').textContent =
    'Page ' + (currentPage+1) + ' of ' + totalPages + '  (' + visible.length + ' sessions)';

  const dots = document.getElementById('pageDots');
  dots.innerHTML = '';
  const maxDots = Math.min(totalPages, 8);
  for (let i = 0; i < maxDots; i++) {
    const d = document.createElement('div');
    d.className = 'page-dot' + (i === currentPage ? ' active' : '');
    (function(p) { d.onclick = function() { currentPage = p; render(); }; })(i);
    dots.appendChild(d);
  }

  let empty = document.getElementById('emptyState');
  if (page.length === 0) {
    if (!empty) {
      empty = document.createElement('div');
      empty.id = 'emptyState';
      empty.className = 'empty-state';
      empty.textContent = 'No sessions match this filter.';
      list.appendChild(empty);
    }
  } else if (empty) {
    empty.remove();
  }
}

function changePage(dir) {
  const visible    = getVisibleRows();
  const totalPages = Math.ceil(visible.length / PAGE_SIZE);
  currentPage = Math.max(0, Math.min(currentPage + dir, totalPages - 1));
  render();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function setFilter(f, btn) {
  currentFilter = f;
  currentPage   = 0;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
}

function setSort(val) {
  currentSort = val;
  currentPage = 0;
  render();
}

function toggleDetail(rowMain) {
  const detail = rowMain.nextElementSibling;
  const expand = rowMain.querySelector('.col-expand');
  const isOpen = detail.style.display !== 'none';
  detail.style.display = isOpen ? 'none' : 'block';
  expand.classList.toggle('open', !isOpen);
}

document.querySelectorAll('.session-row').forEach(row => {
  const barLabel = row.querySelector('.bar-label');
  const noopEl   = row.querySelector('.col-noop');
  const levelsEl = row.querySelector('.col-levels');
  row.dataset.score  = barLabel  ? parseFloat(barLabel.textContent)       : 0;
  row.dataset.noop   = noopEl    ? parseFloat(noopEl.textContent) / 100   : 0;
  row.dataset.levels = levelsEl  ? parseInt(levelsEl.textContent)          : 0;
});

render();

// Floating rich tooltip for spark dots
(function() {
  var tip = document.getElementById('sparkTooltip');
  var storedTitles = {};

  // Store all titles before they get removed
  document.querySelectorAll('.spark-dot').forEach(function(dot, idx) {
    storedTitles[idx] = dot.getAttribute('title') || '';
    dot.setAttribute('data-tip-idx', idx);
    dot.removeAttribute('title');
  });

  document.addEventListener('mouseover', function(e) {
    var dot = e.target.closest('.spark-dot');
    if (!dot) return;
    var raw = storedTitles[dot.getAttribute('data-tip-idx')] || '';
    // raw format: "Level N: score=X, noop=Y%, steps=Z"
    var colonIdx = raw.indexOf(':');
    var header = colonIdx > -1 ? raw.slice(0, colonIdx) : raw;
    var rest   = colonIdx > -1 ? raw.slice(colonIdx + 1).trim() : '';
    var html   = '<strong>' + header + '</strong><br>';
    rest.split(',').forEach(function(part) {
      var kv = part.trim().split('=');
      if (kv.length === 2) {
        html += '<span style="color:#aac">' + kv[0].trim() + '</span>: <strong>' + kv[1].trim() + '</strong><br>';
      }
    });
    tip.innerHTML = html;
    tip.style.display = 'block';
  });

  document.addEventListener('mousemove', function(e) {
    if (tip.style.display === 'block') {
      var x = e.clientX + 16;
      var y = e.clientY - 10;
      if (x + 200 > window.innerWidth) x = e.clientX - 210;
      tip.style.left = x + 'px';
      tip.style.top  = y + 'px';
    }
  });

  document.addEventListener('mouseout', function(e) {
    if (e.target.closest('.spark-dot')) {
      tip.style.display = 'none';
    }
  });
})();
</script>
</body>
</html>"""


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(sessions: list[dict]) -> None:
    flagged    = [s for s in sessions if s["status"] == "flagged"]
    borderline = [s for s in sessions if s["status"] == "borderline"]
    review     = [s for s in sessions if s["status"] == "zero_active"]
    zero_score = [s for s in sessions if s["avg_score"] <= FLAG_SCORE]

    print("\n" + "=" * 70)
    print("  LBF HUMAN DATA — EFFORT ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Total sessions    : {len(sessions)}")
    print(f"  Total games       : {sum(s['num_levels'] for s in sessions)}")
    print(f"  Zero-score sess.  : {len(zero_score)}  ({len(zero_score)/len(sessions):.1%})")
    print(f"  Flagged (remove)  : {len(flagged)}   score=0 AND noop >= {FLAG_NOOP:.0%}")
    print(f"  Borderline        : {len(borderline)}   high noop, low score")
    print(f"  Review            : {len(review)}   zero score but low noop")
    print("=" * 70)

    for label, subset in [
        ("FLAGGED — recommended for removal", flagged),
        ("BORDERLINE — manual review suggested", borderline),
        ("REVIEW — zero score but actively moving", review),
    ]:
        if not subset:
            continue
        print(f"\n  {label}")
        print(f"  {'Session ID':<38} {'Levels':>6} {'AvgScore':>9} {'NoopRate':>9}")
        print(f"  {'-'*38} {'-'*6} {'-'*9} {'-'*9}")
        for s in sorted(subset, key=lambda x: -x["noop_rate"]):
            print(f"  {s['session_id']:<38} {s['num_levels']:>6} "
                  f"{s['avg_score']:>9.3f} {s['noop_rate']:>9.1%}")

    scores = [s["avg_score"] for s in sessions]
    noops  = [s["noop_rate"] for s in sessions]
    print("\n  Score distribution across sessions:")
    print(f"    {'mean':>8}: {np.mean(scores):.4f}")
    print(f"    {'std':>8}: {np.std(scores):.4f}")
    print(f"    {'min':>8}: {np.min(scores):.4f}")
    print(f"    {'25%':>8}: {np.percentile(scores,25):.4f}")
    print(f"    {'50%':>8}: {np.percentile(scores,50):.4f}")
    print(f"    {'75%':>8}: {np.percentile(scores,75):.4f}")
    print(f"    {'max':>8}: {np.max(scores):.4f}")
    print("\n  Noop rate distribution across sessions:")
    print(f"    {'mean':>8}: {np.mean(noops):.4f}")
    print(f"    {'std':>8}: {np.std(noops):.4f}")
    print(f"    {'min':>8}: {np.min(noops):.4f}")
    print(f"    {'25%':>8}: {np.percentile(noops,25):.4f}")
    print(f"    {'50%':>8}: {np.percentile(noops,50):.4f}")
    print(f"    {'75%':>8}: {np.percentile(noops,75):.4f}")
    print(f"    {'max':>8}: {np.max(noops):.4f}")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Run from repo root:  cd ~/jax-aht"
        )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(DATA_DIR)
    if not episodes:
        raise RuntimeError("No episodes loaded — check DATA_DIR path.")

    sessions  = compute_session_stats(episodes)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    html      = build_html(sessions, generated)

    out_path = PLOTS_DIR / "effort_analysis.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[html] Saved → {out_path}")
    print(f"[html] SCP to local: scp -J kanishk@agate.cs.utexas.edu "
          f"~/aht/jax-aht/{out_path} ~/Desktop/")

    print_report(sessions)


if __name__ == "__main__":
    main()