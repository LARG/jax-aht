"""
human_data/visualize.py

Analyzes collected LBF (Level-Based Foraging) gameplay sessions.
Flags potentially low-effort or bad-faith players and provides
an interactive dashboard with per-session episode replay.

Terminology:
  Session    = one player sitting down to play (identified by session ID)
  Game       = one round within a session = one JSON file (up to 8 per session)
  Trajectory = the sequence of actions/states recorded within a single game

Produces:
  human_data/plots/effort_analysis.html  — interactive dashboard with replay

Usage:
  conda activate jax-aht
  cd ~/aht/jax-aht
  python human_data/visualize.py
"""

import json
import html as html_lib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("human_data/collected_data_prolific")
PLOTS_DIR = Path("human_data/plots")

NOOP_ACTION      = 5
PAGE_SIZE        = 15

# Flagging thresholds — tune these based on researcher feedback
FLAG_SCORE       = 0.0    # avg score at or below this is suspicious
FLAG_NOOP        = 0.60   # noop rate at or above this is suspicious
BORDERLINE_NOOP  = 0.55   # borderline noop threshold
LOOP_WINDOW      = 6      # detect repeated action loops within this window
LOOP_THRESHOLD   = 0.75   # fraction of steps that are the same action = loop
MIN_DURATION     = 5.0    # seconds — games shorter than this are suspicious
MIN_LEVELS       = 1

ACTION_NAMES = {0: "None", 1: "Up", 2: "Right", 3: "Down", 4: "Left", 5: "Noop"}

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


def detect_loop(actions: list[int], window: int = LOOP_WINDOW,
                threshold: float = LOOP_THRESHOLD) -> bool:
    """True if any sliding window is dominated by one repeated action."""
    if len(actions) < window:
        return False
    for i in range(len(actions) - window + 1):
        w = actions[i:i + window]
        most_common = max(set(w), key=w.count)
        if w.count(most_common) / window >= threshold:
            return True
    return False


def compute_session_stats(episodes: list[dict]) -> list[dict]:
    sessions = defaultdict(lambda: {
        "scores": [], "noop_counts": [], "total_steps": [],
        "num_games": 0, "game_details": [], "timestamps": [],
        "durations": [], "loop_flags": [],
    })

    for ep in episodes:
        sid         = ep.get("session_id", "unknown")
        human_score = ep.get("total_rewards", {}).get("agent_0", 0.0)
        trajectory  = ep.get("trajectory", [])
        total_steps = len(trajectory)
        timestamp   = ep.get("timestamp", "")
        duration    = ep.get("duration", None)
        grid_size   = ep.get("grid_size", 7)
        num_fruits  = ep.get("num_fruits", 3)
        agent_type  = ep.get("agent_type", "unknown")

        human_actions = [
            s.get("human_action") for s in trajectory
            if s.get("human_action") is not None
        ]
        noop_count = sum(1 for a in human_actions if a == NOOP_ACTION)
        noop_rate  = noop_count / len(human_actions) if human_actions else 0.0
        has_loop   = detect_loop(human_actions)

        # Store full trajectory for replay (compact — only what renderer needs)
        replay_frames = []
        for s in trajectory:
            state = s.get("state", {})
            replay_frames.append({
                "step":            s.get("step", 0),
                "human_action":    s.get("human_action"),
                "ai_action":       s.get("ai_action"),
                "agent_positions": state.get("agent_positions", []),
                "food_positions":  state.get("food_positions", []),
                "food_eaten":      state.get("food_eaten", []),
                "food_levels":     state.get("food_levels", []),
                "agent_levels":    state.get("agent_levels", []),
                "rewards":         s.get("rewards", {}),
            })

        sess = sessions[sid]
        sess["scores"].append(human_score)
        sess["noop_counts"].append(noop_count)
        sess["total_steps"].append(total_steps)
        sess["num_games"] += 1
        sess["timestamps"].append(timestamp)
        sess["durations"].append(duration)
        sess["loop_flags"].append(has_loop)
        sess["game_details"].append({
            "score":         human_score,
            "noop_count":    noop_count,
            "noop_rate":     noop_rate,
            "steps":         total_steps,
            "timestamp":     timestamp,
            "duration":      duration,
            "grid_size":     grid_size,
            "num_fruits":    num_fruits,
            "agent_type":    agent_type,
            "has_loop":      has_loop,
            "file":          ep.get("_file", ""),
            "replay_frames": replay_frames,
        })

    rows = []
    for sid, s in sessions.items():
        if s["num_games"] < MIN_LEVELS:
            continue

        total_actions = sum(s["total_steps"])
        total_noops   = sum(s["noop_counts"])
        avg_score     = float(np.mean(s["scores"]))
        noop_rate     = total_noops / total_actions if total_actions else 0.0
        any_loop      = any(s["loop_flags"])
        durations     = [d for d in s["durations"] if d is not None]
        short_games   = sum(1 for d in durations if d < MIN_DURATION)

        # Flagging — err on the side of eager
        sus_reasons = []
        if avg_score <= FLAG_SCORE and noop_rate >= FLAG_NOOP:
            sus_reasons.append("zero score + high noop")
        if any_loop:
            sus_reasons.append("repetitive action loop detected")
        if short_games >= 2:
            sus_reasons.append(f"{short_games} very short games (<{MIN_DURATION}s)")
        if avg_score <= FLAG_SCORE and noop_rate < FLAG_NOOP:
            sus_reasons.append("zero score (actively moving)")

        if len(sus_reasons) >= 2 or (sus_reasons and "zero score + high noop" in sus_reasons):
            status = "flagged"
        elif sus_reasons:
            status = "borderline"
        elif noop_rate >= BORDERLINE_NOOP and avg_score <= 0.10:
            status = "borderline"
        else:
            status = "ok"

        game_details = sorted(s["game_details"], key=lambda x: x["timestamp"])

        rows.append({
            "session_id":    sid,
            "session_short": sid[:8],
            "num_games":     s["num_games"],
            "avg_score":     avg_score,
            "score_std":     float(np.std(s["scores"])),
            "noop_rate":     noop_rate,
            "any_loop":      any_loop,
            "short_games":   short_games,
            "sus_reasons":   sus_reasons,
            "status":        status,
            "game_details":  game_details,
        })

    rows.sort(key=lambda x: x["avg_score"])
    return rows


# ── HTML helpers ──────────────────────────────────────────────────────────────

def e(s) -> str:
    """HTML-escape a value for safe interpolation."""
    return html_lib.escape(str(s), quote=True)

def score_to_color(score: float, max_score: float = 0.5) -> str:
    pct = min(score / max_score, 1.0) if max_score > 0 else 0
    r = int(220 - pct * 150)
    g = int(120 + pct * 120)
    b = int(100 - pct * 60)
    return f"rgb({r},{g},{b})"

def noop_to_color(noop_rate: float) -> str:
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
        "flagged":   ("FLAGGED",    "#c0392b", "#fff"),
        "borderline":("BORDERLINE", "#ca6f1e", "#fff"),
        "ok":        ("OK",         "#1e8449", "#fff"),
    }
    label, bg, fg = cfg.get(status, ("UNKNOWN", "#888", "#fff"))
    return f'<span class="badge" style="background:{e(bg)};color:{e(fg)}">{e(label)}</span>'

def bar_html(value: float, max_val: float = 0.5, width_pct: int = 100) -> str:
    pct   = min(value / max_val, 1.0) * width_pct if max_val > 0 else 0
    color = score_to_color(value, max_val)
    return (f'<div class="bar-wrap">'
            f'<div class="bar" style="width:{pct:.1f}%;background:{e(color)}"></div>'
            f'<span class="bar-label">{value:.3f}</span>'
            f'</div>')

def level_sparkline(games: list[dict]) -> str:
    dots = []
    for i, gm in enumerate(games):
        color  = score_to_color(gm["score"])
        ncolor = noop_to_color(gm["noop_rate"])
        loop   = " ⟳" if gm["has_loop"] else ""
        tip    = (f"Game {i+1}: score={gm['score']:.3f}, "
                  f"noop={gm['noop_rate']:.0%}, steps={gm['steps']}{loop}")
        dots.append(
            f'<div class="spark-dot{" spark-loop" if gm["has_loop"] else ""}" '
            f'title="{e(tip)}" '
            f'style="background:{e(color)};border:2px solid {e(ncolor)}"></div>'
        )
    return f'<div class="sparkline">{"".join(dots)}</div>'

def render_game_detail_table(games: list[dict]) -> str:
    rows = []
    for i, gm in enumerate(games):
        nc      = noop_to_color(gm["noop_rate"])
        loop_ic = ' <span title="Repetitive action loop detected" style="color:#e05050">⟳</span>' if gm["has_loop"] else ""
        dur_s   = f"{gm['duration']:.1f}s" if gm["duration"] is not None else "—"
        rows.append(
            f'<tr>'
            f'<td>Game {i+1}{loop_ic}</td>'
            f'<td>{bar_html(gm["score"], 0.5, 80)}</td>'
            f'<td style="color:{nc};font-weight:bold">{gm["noop_rate"]:.1%}</td>'
            f'<td>{gm["steps"]}</td>'
            f'<td>{dur_s}</td>'
            f'<td>{e(gm["grid_size"])}×{e(gm["grid_size"])}, {e(gm["num_fruits"])} fruits</td>'
            f'</tr>'
        )
    return (
        '<table class="detail-table">'
        '<thead><tr>'
        '<th>#</th><th>Score</th><th>Noop rate</th>'
        '<th>Steps (trajectory length)</th><th>Duration</th><th>Config</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def build_html(sessions: list[dict], generated: str) -> str:
    import json as json_mod

    n_flagged    = sum(1 for s in sessions if s["status"] == "flagged")
    n_borderline = sum(1 for s in sessions if s["status"] == "borderline")
    n_ok         = sum(1 for s in sessions if s["status"] == "ok")
    total_games  = sum(s["num_games"] for s in sessions)
    avg_score    = float(np.mean([s["avg_score"] for s in sessions]))
    avg_noop     = float(np.mean([s["noop_rate"] for s in sessions]))

    # Build session rows
    session_rows_html = []
    for s in sessions:
        sid        = s["session_id"]
        short      = s["session_short"]
        stat       = s["status"]
        badge      = status_badge(stat)
        bar        = bar_html(s["avg_score"])
        spark      = level_sparkline(s["game_details"])
        nc         = noop_to_color(s["noop_rate"])
        std_s      = f"±{s['score_std']:.3f}" if s["num_games"] > 1 else "—"
        detail_tbl = render_game_detail_table(s["game_details"])
        reasons    = "; ".join(s["sus_reasons"]) if s["sus_reasons"] else "none"
        loop_warn  = ' <span class="loop-warn" title="Repetitive loop detected in one or more games">⟳ loop</span>' if s["any_loop"] else ""

        row_class = {"flagged": "row-flagged", "borderline": "row-borderline", "ok": ""}[stat]

        # Replay data — embed trajectory JSON per game (compact)
        replay_games = []
        for gm in s["game_details"]:
            replay_games.append({
                "grid_size":     gm["grid_size"],
                "num_fruits":    gm["num_fruits"],
                "score":         gm["score"],
                "noop_rate":     round(gm["noop_rate"], 3),
                "frames":        gm["replay_frames"],
            })
        replay_json = e(json_mod.dumps(replay_games))

        session_rows_html.append(
            f'<div class="session-row {e(row_class)}" data-status="{e(stat)}"'
            f' data-sid="{e(sid)}">'
            f'  <div class="row-main" onclick="toggleDetail(this)">'
            f'    <div class="col-id">'
            f'      <span class="sid-full" title="{e(sid)}">{e(short)}…</span>'
            f'      {badge}{loop_warn}'
            f'    </div>'
            f'    <div class="col-bar">{bar}</div>'
            f'    <div class="col-noop" style="color:{e(nc)};font-weight:bold">'
            f'      {s["noop_rate"]:.1%}'
            f'    </div>'
            f'    <div class="col-games">{s["num_games"]} / 8</div>'
            f'    <div class="col-std">{e(std_s)}</div>'
            f'    <div class="col-spark">{spark}</div>'
            f'    <div class="col-expand">▼</div>'
            f'  </div>'
            f'  <div class="row-detail" style="display:none">'
            f'    <div class="detail-inner">'
            f'      <div class="detail-meta">'
            f'        <div>Full session ID: <code>{e(sid)}</code></div>'
            f'        <div class="sus-reasons">Flagged for: <strong>{e(reasons)}</strong></div>'
            f'      </div>'
            f'      {detail_tbl}'
            f'      <button class="replay-btn" '
            f'onclick="openReplay({replay_json}, {e(repr(sid))})">▶ Replay all games</button>'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )

    rows_html = "\n".join(session_rows_html)

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px; background: #f4f5f7; color: #222;
  }
  .header {
    background: #1a1f2e; color: white;
    padding: 18px 28px 14px;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .header h1 { font-size: 21px; font-weight: 700; letter-spacing: 0.3px; }
  .header .subtitle { font-size: 13.5px; color: #9aa; margin-top: 4px; }
  .header-meta { display: flex; gap: 24px; margin-top: 10px; flex-wrap: wrap; }
  .stat-pill {
    background: rgba(255,255,255,0.08); border-radius: 6px;
    padding: 6px 14px; font-size: 13px;
    display: flex; align-items: center; gap: 7px;
  }
  .stat-pill .val { font-weight: 700; font-size: 17px; }
  .stat-pill.flagged .val { color: #e05050; }
  .stat-pill.borderline .val { color: #e09020; }
  .stat-pill.ok .val { color: #52be80; }

  /* Legend */
  .legend {
    background: white; border-bottom: 1px solid #e0e0e0;
    padding: 8px 28px;
    display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  }
  .legend-title { font-weight: 600; color: #555; font-size: 11.5px; text-transform: uppercase; letter-spacing: .5px; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 11.5px; color: #444; }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; }
  .legend-divider { width: 1px; background: #ddd; height: 38px; margin: 0 10px; align-self: center; }
  .scale-dot { width: 17px; height: 17px; border-radius: 4px; border: 2px solid transparent; display: inline-block; flex-shrink: 0; }

  /* Controls */
  .controls {
    background: white; border-bottom: 1px solid #e8e8e8;
    padding: 8px 28px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  .controls label { font-size: 12px; color: #555; font-weight: 500; }
  .filter-btn {
    padding: 4px 12px; border-radius: 14px;
    border: 1.5px solid #ccc; background: white;
    cursor: pointer; font-size: 12px; transition: all 0.15s;
  }
  .filter-btn.active { border-color: #2d6cdf; background: #2d6cdf; color: white; font-weight: 600; }
  .sort-select {
    padding: 4px 8px; border-radius: 6px;
    border: 1.5px solid #ccc; font-size: 12px; background: white; cursor: pointer;
  }

  /* Table header */
  .table-header {
    display: grid;
    grid-template-columns: 210px 1fr 90px 80px 70px 140px 24px;
    padding: 7px 28px;
    background: #eef0f4; border-bottom: 2px solid #d0d4dc;
    font-weight: 700; font-size: 11px; color: #555;
    text-transform: uppercase; letter-spacing: 0.5px;
    position: sticky; top: 102px; z-index: 90;
  }

  /* Session rows */
  .session-list { padding: 0 12px 12px; }
  .session-row {
    background: white; border-radius: 6px; margin: 4px 0;
    border: 1px solid #e8e8e8; overflow: hidden; transition: box-shadow 0.15s;
  }
  .session-row:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .session-row.row-flagged { border-left: 4px solid #c0392b; background: #fff8f8; }
  .session-row.row-borderline { border-left: 4px solid #ca6f1e; background: #fff9f2; }

  .row-main {
    display: grid;
    grid-template-columns: 210px 1fr 90px 80px 70px 140px 24px;
    padding: 9px 16px; cursor: pointer; align-items: center;
  }
  .row-main:hover { background: rgba(0,0,0,0.02); }

  .col-id { display: flex; align-items: center; gap: 6px; min-width: 0; flex-wrap: wrap; }
  .sid-full { font-family: monospace; font-size: 12px; color: #333; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 10px; white-space: nowrap; flex-shrink: 0; }
  .loop-warn { font-size: 10px; color: #e05050; font-weight: 600; white-space: nowrap; }

  .bar-wrap { display: flex; align-items: center; gap: 6px; padding-right: 10px; }
  .bar { height: 14px; border-radius: 3px; min-width: 2px; }
  .bar-label { font-size: 11.5px; font-weight: 600; color: #333; white-space: nowrap; }

  .col-noop, .col-games, .col-std { font-size: 12px; padding: 0 4px; }
  .col-std { color: #777; }

  .sparkline { display: flex; gap: 3px; align-items: center; flex-wrap: wrap; }
  .spark-dot {
    width: 20px; height: 20px; border-radius: 5px; cursor: help;
    transition: transform 0.15s, box-shadow 0.15s;
    border-width: 3px !important; flex-shrink: 0;
  }
  .spark-dot:hover { transform: scale(1.55); box-shadow: 0 2px 10px rgba(0,0,0,0.3); z-index: 10; position: relative; }
  .spark-loop { outline: 2px solid #e05050; outline-offset: 1px; }

  .col-expand { font-size: 11px; color: #aaa; text-align: center; transition: transform 0.2s; }
  .col-expand.open { transform: rotate(180deg); }

  /* Detail panel */
  .row-detail { border-top: 1px solid #eee; background: #fafafa; }
  .detail-inner { padding: 12px 20px 14px; }
  .detail-meta { font-size: 11.5px; color: #666; margin-bottom: 10px; display: flex; gap: 20px; flex-wrap: wrap; }
  .detail-meta code { font-size: 11.5px; background: #eee; padding: 1px 5px; border-radius: 3px; user-select: all; }
  .sus-reasons { font-size: 11.5px; }
  .sus-reasons strong { color: #c0392b; }

  .detail-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 12px; }
  .detail-table th { background: #eef0f4; padding: 5px 10px; text-align: left; font-size: 11px; color: #555; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }
  .detail-table td { padding: 5px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
  .detail-table tr:last-child td { border-bottom: none; }
  .detail-table tr:hover td { background: #f5f7ff; }

  .replay-btn {
    background: #2d6cdf; color: white;
    border: none; border-radius: 6px;
    padding: 7px 18px; font-size: 12.5px; font-weight: 600;
    cursor: pointer; transition: background 0.15s;
  }
  .replay-btn:hover { background: #1a50b0; }

  /* Pagination */
  .pagination {
    display: flex; justify-content: center; align-items: center;
    gap: 10px; padding: 16px; background: white;
    border-top: 1px solid #e8e8e8; margin-top: 8px;
  }
  .page-btn {
    padding: 5px 14px; border: 1.5px solid #ccc; border-radius: 6px;
    background: white; cursor: pointer; font-size: 12px; font-weight: 600;
    transition: all 0.15s;
  }
  .page-btn:hover:not(:disabled) { background: #2d6cdf; color: white; border-color: #2d6cdf; }
  .page-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .page-info { font-size: 12.5px; color: #555; min-width: 120px; text-align: center; }
  .page-dots { display: flex; gap: 5px; }
  .page-dot { width: 8px; height: 8px; border-radius: 50%; background: #ddd; cursor: pointer; transition: background 0.15s; }
  .page-dot.active { background: #2d6cdf; }
  .empty-state { text-align: center; padding: 40px; color: #999; font-size: 13px; }
  .footer { text-align: center; padding: 12px; font-size: 11px; color: #aaa; }

  /* ── Replay modal ── */
  #replayModal {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.75); z-index: 1000;
    overflow-y: auto;
  }
  #replayModal.open { display: flex; flex-direction: column; align-items: center; padding: 20px; }
  .modal-box {
    background: #1a1f2e; color: white;
    border-radius: 12px; width: 100%; max-width: 1100px;
    padding: 20px; position: relative;
  }
  .modal-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px; flex-wrap: wrap; gap: 10px;
  }
  .modal-title { font-size: 15px; font-weight: 700; }
  .modal-sid { font-size: 11px; color: #7a8; font-family: monospace; }
  .modal-close {
    background: rgba(255,255,255,0.1); border: none; color: white;
    border-radius: 6px; padding: 5px 12px; cursor: pointer; font-size: 13px;
  }
  .modal-close:hover { background: rgba(255,255,255,0.2); }

  /* Controls bar */
  .replay-controls {
    display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
  }
  .ctrl-btn {
    background: rgba(255,255,255,0.12); border: none; color: white;
    border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px; font-weight: 600;
    transition: background 0.15s;
  }
  .ctrl-btn:hover { background: rgba(255,255,255,0.22); }
  .ctrl-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .step-info { font-size: 12px; color: #9aa; min-width: 80px; }
  .speed-label { font-size: 12px; color: #9aa; }
  .speed-select {
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    color: white; border-radius: 5px; padding: 4px 8px; font-size: 12px;
  }
  .step-slider { flex: 1; min-width: 100px; max-width: 300px; accent-color: #2d6cdf; }

  /* Game grid */
  .games-grid {
    display: grid;
    gap: 12px;
  }
  .game-card {
    background: rgba(255,255,255,0.05);
    border-radius: 8px; padding: 10px;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .game-card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px; font-size: 11.5px; color: #9aa;
  }
  .game-card-header .gc-score { font-weight: 700; color: #52be80; }
  .game-card-header .gc-noop { color: #e09020; }
  .game-card-header .gc-action { color: #aad; font-family: monospace; }

  /* Grid canvas */
  .grid-canvas-wrap { position: relative; width: 100%; }
  canvas.grid-canvas { width: 100%; display: block; border-radius: 4px; image-rendering: pixelated; }

  /* Action log strip */
  .action-strip {
    margin-top: 6px; display: flex; gap: 2px; flex-wrap: wrap; min-height: 16px;
  }
  .action-pip {
    width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0;
  }
"""

    # ── HTML body ─────────────────────────────────────────────────────────────
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LBF Human Player Effort Analysis</title>
<style>{css}</style>
</head>
<body>

<div class="header">
  <h1>LBF Human Player Effort Analysis</h1>
  <div class="subtitle">
    Each row = one player session &nbsp;·&nbsp;
    Bar = avg score per game &nbsp;·&nbsp;
    Click any row to expand &nbsp;·&nbsp;
    Click ▶ Replay to step through all games &nbsp;·&nbsp;
    Generated {e(generated)}
  </div>
  <div class="header-meta">
    <div class="stat-pill"><span>Sessions</span><span class="val">{len(sessions)}</span></div>
    <div class="stat-pill"><span>Total games</span><span class="val">{total_games}</span></div>
    <div class="stat-pill flagged"><span>⚑ Flagged</span><span class="val">{n_flagged}</span></div>
    <div class="stat-pill borderline"><span>~ Borderline</span><span class="val">{n_borderline}</span></div>
    <div class="stat-pill ok"><span>✓ OK</span><span class="val">{n_ok}</span></div>
    <div class="stat-pill"><span>Avg score</span><span class="val">{avg_score:.3f}</span></div>
    <div class="stat-pill"><span>Avg noop</span><span class="val">{avg_noop:.1%}</span></div>
  </div>
</div>

<div class="legend">
  <span class="legend-title">Row status</span>
  <div class="legend-item"><div class="legend-dot" style="background:#c0392b"></div>&#9873; FLAGGED — suspicious play detected</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ca6f1e"></div>~ BORDERLINE — some suspicious signals</div>
  <div class="legend-item"><div class="legend-dot" style="background:#1e8449"></div>&#10003; OK — normal player</div>
  <div class="legend-item" style="color:#e05050;gap:4px">&#9; ⟳ loop = repetitive action loop detected in a game</div>

  <div class="legend-divider"></div>
  <span class="legend-title">Spark dots</span>
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
    <span style="font-size:10px;color:#777;font-weight:600;text-transform:uppercase;letter-spacing:.4px">Border = noop rate</span>
    <div style="display:flex;align-items:center;gap:3px">
      <div class="scale-dot" style="background:#eee;border:3px solid rgb(80,120,200)"></div>
      <div class="scale-dot" style="background:#eee;border:3px solid rgb(180,140,60)"></div>
      <div class="scale-dot" style="background:#eee;border:3px solid #e05050"></div>
      <span style="font-size:10px;color:#888;margin-left:4px">active &rarr; idle</span>
    </div>
  </div>
</div>

<div class="controls">
  <span class="legend-title">Filter</span>
  <button class="filter-btn active" onclick="setFilter('all',this)">All ({len(sessions)})</button>
  <button class="filter-btn" onclick="setFilter('flagged',this)">⚑ Flagged ({n_flagged})</button>
  <button class="filter-btn" onclick="setFilter('borderline',this)">~ Borderline ({n_borderline})</button>
  <button class="filter-btn" onclick="setFilter('ok',this)">✓ OK ({n_ok})</button>
  &nbsp;&nbsp;
  <label>Sort by</label>
  <select class="sort-select" onchange="setSort(this.value)">
    <option value="score-asc">Score (low → high)</option>
    <option value="score-desc">Score (high → low)</option>
    <option value="noop-desc">Noop rate (high → low)</option>
    <option value="noop-asc">Noop rate (low → high)</option>
    <option value="games-desc">Games played (most → least)</option>
  </select>
</div>

<div class="table-header">
  <div>Session ID</div>
  <div>Avg score per game <span style="font-weight:400;text-transform:none">(max ≈ 0.5)</span></div>
  <div>Noop rate</div>
  <div>Games played</div>
  <div>Score std dev</div>
  <div>Per-game scores</div>
  <div></div>
</div>

<div class="session-list" id="sessionList">
{rows_html}
</div>

<div class="pagination" id="pagination">
  <button class="page-btn" id="prevBtn" onclick="changePage(-1)">← Prev</button>
  <div class="page-dots" id="pageDots"></div>
  <span class="page-info" id="pageInfo"></span>
  <button class="page-btn" id="nextBtn" onclick="changePage(1)">Next →</button>
</div>

<div class="footer">
  LBF Human Player Effort Analysis &nbsp;·&nbsp; LARG @ UT Austin &nbsp;·&nbsp; Generated {e(generated)}
</div>

<!-- Floating tooltip -->
<div id="sparkTooltip" style="
  display:none; position:fixed; background:#1a1f2e; color:white;
  padding:9px 13px; border-radius:8px; font-size:12px; line-height:1.7;
  pointer-events:none; z-index:9999; box-shadow:0 4px 18px rgba(0,0,0,0.35); min-width:180px;
"></div>

<!-- Replay modal -->
<div id="replayModal">
  <div class="modal-box">
    <div class="modal-header">
      <div>
        <div class="modal-title">&#9654; Episode Replay</div>
        <div class="modal-sid" id="replaySid"></div>
      </div>
      <button class="modal-close" onclick="closeReplay()">✕ Close</button>
    </div>
    <div class="replay-controls">
      <button class="ctrl-btn" id="btnFirst" onclick="replayGoTo(0)">&#9664;&#9664; First</button>
      <button class="ctrl-btn" id="btnPrev"  onclick="replayStep(-1)">&#9664; Prev</button>
      <button class="ctrl-btn" id="btnPlay"  onclick="replayTogglePlay()">&#9654; Play</button>
      <button class="ctrl-btn" id="btnNext"  onclick="replayStep(1)">Next &#9654;</button>
      <button class="ctrl-btn" id="btnLast"  onclick="replayGoTo(-1)">Last &#9654;&#9654;</button>
      <span class="step-info" id="stepInfo">Step 0 / 0</span>
      <input type="range" class="step-slider" id="stepSlider" min="0" value="0" oninput="replayGoTo(parseInt(this.value))">
      <span class="speed-label">Speed:</span>
      <select class="speed-select" id="speedSelect">
        <option value="600">0.5×</option>
        <option value="300" selected>1×</option>
        <option value="150">2×</option>
        <option value="80">4×</option>
      </select>
    </div>
    <div class="games-grid" id="gamesGrid"></div>
  </div>
</div>

""" + """
<script>
// ── Pagination ──────────────────────────────────────────────────────────────
const PAGE_SIZE = """ + str(PAGE_SIZE) + """;
let currentPage = 0, currentFilter = 'all', currentSort = 'score-asc';

function getAllRows() { return Array.from(document.querySelectorAll('.session-row')); }
function getVisibleRows() {
  return getAllRows().filter(r => currentFilter === 'all' || r.dataset.status === currentFilter);
}
function sortRows(rows) {
  rows.sort((a,b) => {
    switch(currentSort) {
      case 'score-asc':   return parseFloat(a.dataset.score||0)-parseFloat(b.dataset.score||0);
      case 'score-desc':  return parseFloat(b.dataset.score||0)-parseFloat(a.dataset.score||0);
      case 'noop-desc':   return parseFloat(b.dataset.noop||0)-parseFloat(a.dataset.noop||0);
      case 'noop-asc':    return parseFloat(a.dataset.noop||0)-parseFloat(b.dataset.noop||0);
      case 'games-desc':  return parseInt(b.dataset.games||0)-parseInt(a.dataset.games||0);
      default: return 0;
    }
  });
  return rows;
}
function render() {
  const list = document.getElementById('sessionList');
  getAllRows().forEach(r => r.style.display='none');
  let visible = sortRows(getVisibleRows());
  const totalPages = Math.max(1, Math.ceil(visible.length/PAGE_SIZE));
  currentPage = Math.min(currentPage, totalPages-1);
  const page = visible.slice(currentPage*PAGE_SIZE, (currentPage+1)*PAGE_SIZE);
  page.forEach(r => { r.style.display=''; list.appendChild(r); });
  document.getElementById('prevBtn').disabled = currentPage===0;
  document.getElementById('nextBtn').disabled = currentPage>=totalPages-1;
  document.getElementById('pageInfo').textContent =
    'Page '+(currentPage+1)+' of '+totalPages+'  ('+visible.length+' sessions)';
  const dots = document.getElementById('pageDots');
  dots.innerHTML='';
  for(let i=0;i<Math.min(totalPages,8);i++){
    const d=document.createElement('div');
    d.className='page-dot'+(i===currentPage?' active':'');
    (function(p){d.onclick=function(){currentPage=p;render();};})(i);
    dots.appendChild(d);
  }
  let empty=document.getElementById('emptyState');
  if(page.length===0){
    if(!empty){empty=document.createElement('div');empty.id='emptyState';empty.className='empty-state';empty.textContent='No sessions match this filter.';list.appendChild(empty);}
  } else if(empty) empty.remove();
}
function changePage(dir) {
  const v=getVisibleRows(), tp=Math.ceil(v.length/PAGE_SIZE);
  currentPage=Math.max(0,Math.min(currentPage+dir,tp-1));
  render(); window.scrollTo({top:0,behavior:'smooth'});
}
function setFilter(f,btn) {
  currentFilter=f; currentPage=0;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active'); render();
}
function setSort(val) { currentSort=val; currentPage=0; render(); }
function toggleDetail(rowMain) {
  const detail=rowMain.nextElementSibling;
  const expand=rowMain.querySelector('.col-expand');
  const isOpen=detail.style.display!=='none';
  detail.style.display=isOpen?'none':'block';
  expand.classList.toggle('open',!isOpen);
}

// Attach data attrs for sorting
document.querySelectorAll('.session-row').forEach(row => {
  const barLabel=row.querySelector('.bar-label');
  const noopEl=row.querySelector('.col-noop');
  const gamesEl=row.querySelector('.col-games');
  row.dataset.score  = barLabel  ? parseFloat(barLabel.textContent) : 0;
  row.dataset.noop   = noopEl    ? parseFloat(noopEl.textContent)/100 : 0;
  row.dataset.games  = gamesEl   ? parseInt(gamesEl.textContent) : 0;
});
render();

// ── Spark dot tooltip ───────────────────────────────────────────────────────
(function(){
  var tip=document.getElementById('sparkTooltip');
  var stored={};
  document.querySelectorAll('.spark-dot').forEach(function(dot,idx){
    stored[idx]=dot.getAttribute('title')||'';
    dot.setAttribute('data-tip-idx',idx);
    dot.removeAttribute('title');
  });
  document.addEventListener('mouseover',function(e){
    var dot=e.target.closest('.spark-dot');
    if(!dot) return;
    // raw format: "Game N: score=X, noop=Y%, steps=Z"
    var raw=stored[dot.getAttribute('data-tip-idx')]||'';
    var ci=raw.indexOf(':');
    var header=ci>-1?raw.slice(0,ci):raw;
    var rest=ci>-1?raw.slice(ci+1).trim():'';
    var html='<strong>'+header+'</strong><br>';
    rest.split(',').forEach(function(p){
      var kv=p.trim().split('=');
      if(kv.length===2) html+='<span style="color:#aac">'+kv[0].trim()+'</span>: <strong>'+kv[1].trim()+'</strong><br>';
    });
    tip.innerHTML=html; tip.style.display='block';
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.display==='block'){
      var x=e.clientX+16, y=e.clientY-10;
      if(x+200>window.innerWidth) x=e.clientX-210;
      tip.style.left=x+'px'; tip.style.top=y+'px';
    }
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('.spark-dot')) tip.style.display='none';
  });
})();

// ── Replay engine ────────────────────────────────────────────────────────────
var replayGames=[], replayStep_=0, replayPlaying=false, replayTimer=null;
var CELL_COLORS = {bg:'#2d4a3e', grid:'#3a5a4a', human:'#4fc3f7', ai:'#ef9a9a', fruit:'#ffb74d', eaten:'#555'};
var ACTION_NAMES = {0:'None',1:'Up',2:'Right',3:'Down',4:'Left',5:'Noop',null:'—'};
var NOOP_COLOR   = '#e05050';
var ACTION_COLORS= {0:'#888',1:'#81d4fa',2:'#81d4fa',3:'#81d4fa',4:'#81d4fa',5:NOOP_COLOR};

function openReplay(games, sid) {
  replayGames = games;
  replayStep_ = 0;
  replayPlaying = false;
  clearInterval(replayTimer);
  document.getElementById('replaySid').textContent = 'Session: '+sid;
  document.getElementById('btnPlay').textContent = '▶ Play';

  // Build game cards
  var grid = document.getElementById('gamesGrid');
  grid.innerHTML = '';
  var n = games.length;
  var cols = n<=4?n:Math.ceil(Math.sqrt(n));
  grid.style.gridTemplateColumns = 'repeat('+cols+', 1fr)';

  games.forEach(function(gm,i){
    var card = document.createElement('div');
    card.className = 'game-card';
    card.id = 'gamecard-'+i;
    card.innerHTML =
      '<div class="game-card-header">'
      +'<span>Game '+(i+1)+'</span>'
      +'<span class="gc-score">Score: '+gm.score.toFixed(3)+'</span>'
      +'<span class="gc-noop">Noop: '+(gm.noop_rate*100).toFixed(0)+'%</span>'
      +'<span class="gc-done" id="done-'+i+'" style="display:none;color:#52be80;font-weight:700">✓ Done</span>'
      +'<span class="gc-action" id="action-'+i+'">—</span>'
      +'</div>'
      +'<div class="grid-canvas-wrap">'
      +'<canvas class="grid-canvas" id="canvas-'+i+'" width="'+(gm.grid_size*32)+'" height="'+(gm.grid_size*32)+'"></canvas>'
      +'</div>'
      +'<div class="action-strip" id="strip-'+i+'"></div>';
    grid.appendChild(card);
  });

  // Set up slider
  var maxSteps = Math.max.apply(null, games.map(function(g){return g.frames.length-1;}));
  var slider = document.getElementById('stepSlider');
  slider.max = maxSteps; slider.value = 0;

  renderReplayFrame(0);
  document.getElementById('replayModal').classList.add('open');
}

function closeReplay() {
  clearInterval(replayTimer); replayPlaying=false;
  document.getElementById('replayModal').classList.remove('open');
}

function renderReplayFrame(stepIdx) {
  replayStep_ = stepIdx;
  var maxSteps = Math.max.apply(null, replayGames.map(function(g){return g.frames.length-1;}));
  document.getElementById('stepInfo').textContent = 'Step '+stepIdx+' / '+maxSteps;
  document.getElementById('stepSlider').value = stepIdx;

  replayGames.forEach(function(gm,i){
    var frames = gm.frames;
    var isDone = stepIdx >= frames.length;
    var fi = Math.min(stepIdx, frames.length-1);
    var frame = frames[fi];
    drawGrid(i, gm.grid_size, frame, isDone, stepIdx===fi&&fi===frames.length-1);

    // Done badge
    var doneEl = document.getElementById('done-'+i);
    if(doneEl) doneEl.style.display = isDone ? 'inline' : 'none';
    var card = document.getElementById('gamecard-'+i);
    if(card) card.style.opacity = isDone ? '0.7' : '1';
    // Action label
    var ha = stepIdx<frames.length ? frame.human_action : null;
    var aa = stepIdx<frames.length ? frame.ai_action : null;
    var aname = ACTION_NAMES[ha] !== undefined ? ACTION_NAMES[ha] : '—';
    document.getElementById('action-'+i).textContent =
      'Human: '+aname + (aa!==null?' | AI: '+(ACTION_NAMES[aa]||'—'):'');

    // Action strip — color all steps up to current
    var strip = document.getElementById('strip-'+i);
    strip.innerHTML='';
    for(var j=0;j<frames.length;j++){
      var pip=document.createElement('div');
      pip.className='action-pip';
      var act=frames[j].human_action;
      pip.style.background = j===stepIdx ? 'white' : (ACTION_COLORS[act]||'#888');
      pip.style.opacity = j>stepIdx ? '0.3' : '1';
      pip.title='Step '+j+': '+(ACTION_NAMES[act]||'—');
      strip.appendChild(pip);
    }
  });
}

function drawGrid(gameIdx, gridSize, frame, isDone, isLastFrame) {
  var canvas = document.getElementById('canvas-'+gameIdx);
  if(!canvas) return;
  var ctx = canvas.getContext('2d');
  var W=canvas.width, H=canvas.height;
  var cell = W/gridSize;

  // Background — dim if game is done
  ctx.fillStyle = isDone ? '#1e2d26' : CELL_COLORS.bg;
  ctx.fillRect(0,0,W,H);

  // Grid lines
  ctx.strokeStyle = isDone ? '#283d30' : CELL_COLORS.grid;
  ctx.lineWidth = 0.5;
  for(var i=0;i<=gridSize;i++){
    ctx.beginPath(); ctx.moveTo(i*cell,0); ctx.lineTo(i*cell,H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,i*cell); ctx.lineTo(W,i*cell); ctx.stroke();
  }

  // Food — fade eaten fruits, and at last frame fade ALL collected ones more visibly
  var fpos = frame.food_positions||[], featen=frame.food_eaten||[], flvl=frame.food_levels||[];
  fpos.forEach(function(fp,fi){
    var eaten=featen[fi];
    var px=fp[1]*cell, py=fp[0]*cell;
    var pad=cell*0.15;
    if(eaten) {
      // Faded eaten fruit — slightly more visible at game end
      ctx.globalAlpha = isDone ? 0.25 : 0.35;
      ctx.fillStyle = CELL_COLORS.fruit;
      ctx.beginPath();
      ctx.roundRect(px+pad, py+pad, cell-pad*2, cell-pad*2, 3);
      ctx.fill();
      // Checkmark
      ctx.globalAlpha = isDone ? 0.4 : 0.5;
      ctx.fillStyle = '#aaffaa';
      ctx.font='bold '+(cell*0.4)+'px sans-serif';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('✓', px+cell/2, py+cell/2);
      ctx.globalAlpha = 1.0;
    } else {
      ctx.globalAlpha = isDone ? 0.5 : 1.0;
      ctx.fillStyle = CELL_COLORS.fruit;
      ctx.beginPath();
      ctx.roundRect(px+pad, py+pad, cell-pad*2, cell-pad*2, 3);
      ctx.fill();
      ctx.fillStyle='rgba(0,0,0,0.6)';
      ctx.font='bold '+(cell*0.38)+'px sans-serif';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(flvl[fi]||'', px+cell/2, py+cell/2);
      ctx.globalAlpha = 1.0;
    }
  });

  // Agents — fade if done
  var apos=frame.agent_positions||[], alvl=frame.agent_levels||[];
  ctx.globalAlpha = isDone ? 0.45 : 1.0;
  apos.forEach(function(ap,ai){
    var color = ai===0 ? CELL_COLORS.human : CELL_COLORS.ai;
    var px=ap[1]*cell, py=ap[0]*cell;
    var r = cell/2 - cell*0.08;
    // Clean circle — no level badge
    ctx.fillStyle=color;
    ctx.beginPath();
    ctx.arc(px+cell/2, py+cell/2, r, 0, Math.PI*2);
    ctx.fill();
    // Subtle ring
    ctx.strokeStyle='rgba(255,255,255,0.3)';
    ctx.lineWidth=1.5;
    ctx.stroke();
    // Letter
    ctx.fillStyle='rgba(0,0,0,0.8)';
    ctx.font='bold '+(cell*0.42)+'px sans-serif';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(ai===0?'H':'A', px+cell/2, py+cell/2);
    // Small level indicator — cleaner, bottom-right inside circle
    ctx.fillStyle='rgba(255,255,255,0.7)';
    ctx.font=(cell*0.24)+'px sans-serif';
    ctx.fillText(''+( alvl[ai]||''), px+cell*0.72, py+cell*0.72);
  });
  ctx.globalAlpha = 1.0;

  // DONE overlay
  if(isDone) {
    ctx.fillStyle='rgba(0,0,0,0.38)';
    ctx.fillRect(0,0,W,H);
    ctx.fillStyle='rgba(255,255,255,0.92)';
    ctx.font='bold '+(cell*1.1)+'px sans-serif';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('✓', W/2, H/2 - cell*0.6);
    ctx.fillStyle='rgba(200,255,200,0.9)';
    ctx.font='bold '+(cell*0.55)+'px sans-serif';
    ctx.fillText('DONE', W/2, H/2 + cell*0.3);
  }
}

function replayStep(dir) {
  var maxSteps=Math.max.apply(null,replayGames.map(function(g){return g.frames.length-1;}));
  var next=Math.max(0,Math.min(replayStep_+dir,maxSteps));
  renderReplayFrame(next);
}
function replayGoTo(idx) {
  var maxSteps=Math.max.apply(null,replayGames.map(function(g){return g.frames.length-1;}));
  if(idx<0) idx=maxSteps;
  renderReplayFrame(Math.max(0,Math.min(idx,maxSteps)));
}
function replayTogglePlay() {
  replayPlaying=!replayPlaying;
  document.getElementById('btnPlay').textContent=replayPlaying?'⏸ Pause':'▶ Play';
  if(replayPlaying){
    var maxSteps=Math.max.apply(null,replayGames.map(function(g){return g.frames.length-1;}));
    var speed=parseInt(document.getElementById('speedSelect').value)||300;
    replayTimer=setInterval(function(){
      if(replayStep_>=maxSteps){ clearInterval(replayTimer); replayPlaying=false; document.getElementById('btnPlay').textContent='▶ Play'; return; }
      replayStep(1);
    },speed);
  } else { clearInterval(replayTimer); }
}

// Close modal on background click
document.getElementById('replayModal').addEventListener('click',function(e){
  if(e.target===this) closeReplay();
});
// Keyboard controls
document.addEventListener('keydown',function(e){
  if(!document.getElementById('replayModal').classList.contains('open')) return;
  if(e.key==='ArrowRight'||e.key==='ArrowDown') replayStep(1);
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp') replayStep(-1);
  else if(e.key===' '){e.preventDefault();replayTogglePlay();}
  else if(e.key==='Escape') closeReplay();
});
</script>
</body>
</html>"""

    return body


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(sessions: list[dict]) -> None:
    flagged    = [s for s in sessions if s["status"] == "flagged"]
    borderline = [s for s in sessions if s["status"] == "borderline"]
    zero_score = [s for s in sessions if s["avg_score"] <= FLAG_SCORE]

    print("\n" + "=" * 70)
    print("  LBF HUMAN DATA — EFFORT ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Total sessions    : {len(sessions)}")
    print(f"  Total games       : {sum(s['num_games'] for s in sessions)}")
    print(f"  Zero-score sess.  : {len(zero_score)}  ({len(zero_score)/len(sessions):.1%})")
    print(f"  Flagged           : {len(flagged)}")
    print(f"  Borderline        : {len(borderline)}")
    print("=" * 70)

    for label, subset in [
        ("FLAGGED — recommended for removal", flagged),
        ("BORDERLINE — manual review suggested", borderline),
    ]:
        if not subset:
            continue
        print(f"\n  {label}")
        print(f"  {'Session ID':<38} {'Games':>5} {'AvgScore':>9} {'NoopRate':>9}  Reasons")
        print(f"  {'-'*38} {'-'*5} {'-'*9} {'-'*9}  {'-'*30}")
        for s in sorted(subset, key=lambda x: -x["noop_rate"]):
            reasons = "; ".join(s["sus_reasons"]) if s["sus_reasons"] else "—"
            print(f"  {s['session_id']:<38} {s['num_games']:>5} "
                  f"{s['avg_score']:>9.3f} {s['noop_rate']:>9.1%}  {reasons}")

    scores = [s["avg_score"] for s in sessions]
    noops  = [s["noop_rate"] for s in sessions]
    print("\n  Score distribution across sessions:")
    for stat, val in [("mean", np.mean(scores)), ("std", np.std(scores)),
                      ("min", np.min(scores)), ("25%", np.percentile(scores,25)),
                      ("50%", np.percentile(scores,50)), ("75%", np.percentile(scores,75)),
                      ("max", np.max(scores))]:
        print(f"    {stat:>8}: {val:.4f}")
    print("\n  Noop rate distribution across sessions:")
    for stat, val in [("mean", np.mean(noops)), ("std", np.std(noops)),
                      ("min", np.min(noops)), ("25%", np.percentile(noops,25)),
                      ("50%", np.percentile(noops,50)), ("75%", np.percentile(noops,75)),
                      ("max", np.max(noops))]:
        print(f"    {stat:>8}: {val:.4f}")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            "Run from repo root:  cd ~/aht/jax-aht"
        )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    episodes  = load_episodes(DATA_DIR)
    if not episodes:
        raise RuntimeError("No episodes loaded — check DATA_DIR path.")

    sessions  = compute_session_stats(episodes)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    html      = build_html(sessions, generated)

    out_path  = PLOTS_DIR / "effort_analysis.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[html] Saved → {out_path}")
    print_report(sessions)


if __name__ == "__main__":
    main()