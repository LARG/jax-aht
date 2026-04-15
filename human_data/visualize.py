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

# Action mapping (from app.py / README controls):
#   0 = Noop/Wait (Q key)   1 = Up    2 = Right
#   3 = Down                4 = Left  5 = Collect/Load (SPACE)
NOOP_ACTION      = 0   # Q key — true idle/wait action
COLLECT_ACTION   = 5   # SPACE — food collection attempt
PAGE_SIZE        = 15
REQUIRED_GAMES   = 8   # expected number of games per complete session

# Flagging thresholds — tune these based on researcher feedback
FLAG_SCORE       = 0.0    # avg score at or below this is suspicious
FLAG_IDLE        = 0.35   # idle rate (Q + unproductive SPACE) at or above this → flagged
IDLE_SEQ_LEN     = 6      # consecutive idle actions (Q or SPACE) of this length = suspicious
IDLE_SEQ_THRESH  = 0.75   # fraction of window that must be idle to count as a sequence
MIN_LEVELS       = 1

ACTION_NAMES = {0: "Noop(Q)", 1: "Up", 2: "Down", 3: "Left", 4: "Right", 5: "Load(SPACE)"}

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


def detect_idle_sequence(actions: list[int], food_eaten_per_step: list[bool],
                         window: int = IDLE_SEQ_LEN,
                         threshold: float = IDLE_SEQ_THRESH) -> bool:
    """
    True if any sliding window is dominated by idle actions.
    Idle = Q (noop) OR SPACE (collect) when no fruit was collected that step.
    This catches both Q-spammers and SPACE-spammers who aren't actually collecting.
    """
    if len(actions) < window:
        return False
    idle = [
        a == NOOP_ACTION or (a == COLLECT_ACTION and not food_eaten_per_step[i])
        for i, a in enumerate(actions)
    ]
    for i in range(len(idle) - window + 1):
        w = idle[i:i + window]
        if sum(w) / window >= threshold:
            return True
    return False


def compute_session_stats(episodes: list[dict]) -> list[dict]:
    sessions = defaultdict(lambda: {
        "scores": [], "noop_counts": [], "total_steps": [],
        "num_games": 0, "game_details": [], "timestamps": [],
        "durations": [], "loop_flags": [], "idle_rates": [],
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

        # Reaction times: elapsed[i] - elapsed[i-1] in milliseconds
        reaction_times_ms = []
        prev_elapsed = None
        for s in trajectory:
            if s.get("human_action") is None:
                prev_elapsed = s.get("elapsed", None)
                continue
            curr_elapsed = s.get("elapsed", None)
            if curr_elapsed is not None and prev_elapsed is not None:
                dt_ms = (curr_elapsed - prev_elapsed) * 1000
                if dt_ms > 0:
                    reaction_times_ms.append(dt_ms)
            prev_elapsed = curr_elapsed

        # Track which SPACE presses actually collected fruit
        food_eaten_per_action = []
        prev_eaten = None
        for s in trajectory:
            ha = s.get("human_action")
            if ha is None:
                continue
            curr_eaten = tuple(s.get("state", {}).get("food_eaten", []))
            collected  = prev_eaten is not None and curr_eaten != prev_eaten
            food_eaten_per_action.append(collected)
            prev_eaten = curr_eaten

        noop_count       = sum(1 for a in human_actions if a == NOOP_ACTION)
        wasted_load_count= sum(
            1 for a, collected in zip(human_actions, food_eaten_per_action)
            if a == COLLECT_ACTION and not collected
        )
        idle_count  = noop_count + wasted_load_count
        idle_rate   = idle_count / len(human_actions) if human_actions else 0.0
        noop_count  = noop_count  # keep for display
        noop_rate   = noop_count / len(human_actions) if human_actions else 0.0
        has_loop    = detect_idle_sequence(human_actions, food_eaten_per_action)

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
            "score":            human_score,
            "noop_count":       noop_count,
            "noop_rate":        noop_rate,
            "idle_rate":        idle_rate,
            "wasted_load":      wasted_load_count,
            "steps":            total_steps,
            "timestamp":        timestamp,
            "duration":         duration,
            "grid_size":        grid_size,
            "num_fruits":       num_fruits,
            "agent_type":       agent_type,
            "has_loop":          has_loop,
            "file":              ep.get("_file", ""),
            "replay_frames":     replay_frames,
            "reaction_times_ms": reaction_times_ms,
            "median_rt_ms":      float(np.median(reaction_times_ms)) if reaction_times_ms else None,
            "pct10_rt_ms":       float(np.percentile(reaction_times_ms, 10)) if reaction_times_ms else None,
        })

    rows = []
    for sid, s in sessions.items():
        if s["num_games"] < MIN_LEVELS:
            continue

        total_actions  = sum(s["total_steps"])
        total_noops    = sum(s["noop_counts"])
        avg_score      = float(np.mean(s["scores"]))
        noop_rate      = total_noops / total_actions if total_actions else 0.0
        # Idle rate = (Q presses + unproductive SPACE presses) / total actions
        idle_rate      = float(np.mean([gd["idle_rate"] for gd in s["game_details"]]))
        any_loop       = any(s["loop_flags"])
        zero_games     = sum(1 for sc in s["scores"] if sc <= FLAG_SCORE)
        short_games   = 0  # reserved for future reaction-time metric

        # Aggregate reaction times across all games in session
        all_rts = []
        for gd in s["game_details"]:
            all_rts.extend(gd.get("reaction_times_ms", []))
        session_median_rt = float(np.median(all_rts)) if all_rts else None
        session_min_rt    = float(np.min(all_rts))    if all_rts else None
        session_pct10_rt  = float(np.percentile(all_rts, 10)) if all_rts else None
        rt_suspicious     = session_pct10_rt is not None and session_pct10_rt < 150

        # Flagging — err on the side of eager
        sus_reasons = []

        # High idle rate (Q presses + wasted SPACE as fraction of total actions)
        # Note: wasted SPACE alone is unreliable — player may be attempting
        # to collect a fruit that requires combined agent levels. Only flag
        # if the overall idle rate is very high.
        if idle_rate >= FLAG_IDLE:
            sus_reasons.append(f"high idle rate ({idle_rate:.0%} of actions were Q or wasted SPACE)")

        # High Q-key rate alone (pure waiting, no ambiguity)
        noop_only_rate = noop_rate  # Q presses / total actions
        if noop_only_rate >= 0.15:
            sus_reasons.append(f"high Q-key rate ({noop_only_rate:.0%} of actions were pure waits)")

        # Low score — exact zero average
        if avg_score <= FLAG_SCORE:
            sus_reasons.append("zero average score")

        # Many zero-score games even if avg > 0 (one lucky game shouldn't save them)
        if zero_games >= 5 and s["num_games"] >= 6:
            sus_reasons.append(f"{zero_games}/{s['num_games']} games scored zero")

        # (Short game duration check removed — will be replaced by
        #  per-step reaction time metric which is more reliable)

        # Deduplicate
        sus_reasons = list(dict.fromkeys(sus_reasons))

        # Disqualified = didn't finish all games (separate from effort flagging)
        if s["num_games"] < REQUIRED_GAMES:
            status = "disqualified"
        elif sus_reasons:
            status = "flagged"
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
            "idle_rate":     idle_rate,
            "any_loop":      any_loop,
            "short_games":   short_games,
            "zero_games":    zero_games,
            "median_rt_ms":  session_median_rt,
            "pct10_rt_ms":   session_pct10_rt,
            "rt_suspicious": rt_suspicious,
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

def noop_to_color(rate: float) -> str:
    pct = min(rate, 1.0)
    if pct >= FLAG_IDLE:
        return "#e05050"
    elif pct >= 0.20:
        return "#e09020"
    else:
        r = int(80  + pct * 100)
        g = int(120 - pct * 40)
        b = int(200 - pct * 100)
        return f"rgb({r},{g},{b})"

def status_badge(status: str) -> str:
    cfg = {
        "flagged":      ("FLAGGED",       "#c0392b", "#fff"),
        "disqualified": ("DISQUALIFIED",  "#7b1fa2", "#fff"),
        "ok":           ("OK",            "#1e8449", "#fff"),
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
        ic      = noop_to_color(gm["idle_rate"])
        loop_ic = ' <span title="Idle sequence detected" style="color:#e05050">⟳</span>' if gm["has_loop"] else ""
        dur_s   = f"{gm['duration']:.1f}s" if gm["duration"] is not None else "—"
        med_rt  = gm.get("median_rt_ms")
        p10_rt  = gm.get("pct10_rt_ms")
        rt_str  = f"{med_rt:.0f}ms" if med_rt is not None else "—"
        rt_color = "#e05050" if (p10_rt is not None and p10_rt < 150) else "#555"
        rows.append(
            f'<tr>'
            f'<td>Game {i+1}{loop_ic}</td>'
            f'<td>{bar_html(gm["score"], 0.5, 80)}</td>'
            f'<td style="color:{ic};font-weight:bold">{gm["idle_rate"]:.1%}</td>'
            f'<td style="color:#888;font-size:11px">'
            f'Q:{gm["noop_count"]} SPACE-wasted:{gm["wasted_load"]}</td>'
            f'<td>{gm["steps"]}</td>'
            f'<td>{dur_s}</td>'
            f'<td style="color:{rt_color};font-weight:bold">{e(rt_str)}</td>'
            f'<td>{e(gm["grid_size"])}×{e(gm["grid_size"])}, {e(gm["num_fruits"])} fruits</td>'
            f'</tr>'
        )
    return (
        '<table class="detail-table">'
        '<thead><tr>'
        '<th>#</th><th>Score</th><th>Idle rate</th>'
        '<th>Breakdown (Q presses / wasted SPACE)</th>'
        '<th>Steps</th><th>Duration</th><th>Median RT</th><th>Config</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def build_html(sessions: list[dict], generated: str) -> str:
    import json as json_mod

    n_flagged      = sum(1 for s in sessions if s["status"] == "flagged")
    n_disqualified = sum(1 for s in sessions if s["status"] == "disqualified")
    n_ok           = sum(1 for s in sessions if s["status"] == "ok")
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
        nc         = noop_to_color(s["idle_rate"])
        std_s      = f"±{s['score_std']:.3f}" if s["num_games"] > 1 else "—"
        detail_tbl = render_game_detail_table(s["game_details"])
        reasons    = "; ".join(s["sus_reasons"]) if s["sus_reasons"] else "none"
        loop_warn  = ' <span class="loop-warn" title="Repetitive loop detected in one or more games">⟳ loop</span>' if s["any_loop"] else ""
        s_med_rt   = s.get("median_rt_ms")
        s_p10_rt   = s.get("pct10_rt_ms")
        s_rt_warn  = s.get("rt_suspicious", False)
        s_rt_disp  = f"{s_med_rt:.0f}ms" if s_med_rt is not None else "—"
        s_p10_disp = f"{s_p10_rt:.0f}ms" if s_p10_rt is not None else "—"
        s_rt_col   = "#e05050" if s_rt_warn else "#888"

        row_class = {
            "flagged": "row-flagged",
            "disqualified": "row-disqualified", "ok": ""
        }.get(stat, "")

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
            f'        <div style="font-size:11.5px;color:{e(s_rt_col)};margin-top:4px">'
            f'&#9201; Median reaction time: <strong>{e(s_rt_disp)}</strong>'
            f' &nbsp;&#183;&nbsp; 10th pct: <strong>{e(s_p10_disp)}</strong>'
            f'{"" if not s_rt_warn else " &nbsp;<span style=&#39;color:#e05050;font-weight:700&#39;>&#9888; suspiciously fast (p10 &lt; 150ms)</span>"}'
            f'        </div>'
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
  .session-row.row-disqualified { border-left: 4px solid #7b1fa2; background: #fdf4ff; }
  .stat-pill.disqualified .val { color: #ba68c8; }

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
    <div class="stat-pill disqualified"><span>✗ Disqualified</span><span class="val">{n_disqualified}</span></div>
    <div class="stat-pill ok"><span>✓ OK</span><span class="val">{n_ok}</span></div>
    <div class="stat-pill"><span>Avg score</span><span class="val">{avg_score:.3f}</span></div>
    <div class="stat-pill"><span>Avg noop</span><span class="val">{avg_noop:.1%}</span></div>
  </div>
</div>

<div class="legend">
  <span class="legend-title">Row status</span>
  <div class="legend-item"><div class="legend-dot" style="background:#c0392b"></div>&#9873; FLAGGED — suspicious play, review before using data</div>
  <div class="legend-item"><div class="legend-dot" style="background:#7b1fa2"></div>&#10007; DISQUALIFIED — incomplete session (&lt;8 games), exclude from analysis</div>
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
  <button class="filter-btn" onclick="setFilter('disqualified',this)">✗ Disqualified ({n_disqualified})</button>
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
  <div>Idle rate</div>
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
var ACTION_NAMES = {0:'Noop(Q)',1:'Up',2:'Down',3:'Left',4:'Right',5:'Load(SPACE)',null:'—'};
var NOOP_COLOR   = '#e05050';
var ACTION_COLORS= {0:NOOP_COLOR,1:'#81d4fa',2:'#81d4fa',3:'#81d4fa',4:'#81d4fa',5:'#81c784'};

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




def build_scatter_html(sessions: list[dict], generated: str) -> str:
    """
    Standalone scatter plot: median reaction time (x) vs avg score (y).
    One dot per session, colored by status, sized by number of games.
    Helps visualize whether thinking time correlates with performance.
    """
    import json as json_mod

    # Build data points
    points = []
    for s in sessions:
        if s.get("median_rt_ms") is None:
            continue
        points.append({
            "sid":       s["session_id"],
            "short":     s["session_short"],
            "med_rt":    round(s["median_rt_ms"], 1),
            "p10_rt":    round(s["pct10_rt_ms"], 1) if s["pct10_rt_ms"] else None,
            "score":     round(s["avg_score"], 4),
            "noop":      round(s["noop_rate"], 4),
            "idle":      round(s["idle_rate"], 4),
            "games":     s["num_games"],
            "status":    s["status"],
            "rt_warn":   s.get("rt_suspicious", False),
            "reasons":   "; ".join(s["sus_reasons"]) if s["sus_reasons"] else "none",
        })

    points_json = json_mod.dumps(points)

    status_colors = {
        "flagged":      "#e05050",
        "disqualified": "#ba68c8",
        "ok":           "#52be80",
    }

    color_map = json_mod.dumps(status_colors)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LBF Reaction Time vs Score</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f4f5f7; color: #222; font-size: 13px; }}
  .header {{ background: #1a1f2e; color: white; padding: 16px 28px; }}
  .header h1 {{ font-size: 18px; font-weight: 700; }}
  .header .sub {{ font-size: 12.5px; color: #9aa; margin-top: 4px; }}
  .container {{ padding: 20px 28px; }}
  .chart-wrap {{ background: white; border-radius: 8px; padding: 20px;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.08); position: relative; }}
  canvas {{ display: block; width: 100%; cursor: crosshair; }}
  .tooltip {{
    position: fixed; background: #1a1f2e; color: white;
    padding: 9px 13px; border-radius: 8px; font-size: 12px;
    line-height: 1.7; pointer-events: none; z-index: 999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3); display: none; min-width: 200px;
  }}
  .legend {{ display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .threshold-note {{ font-size: 11.5px; color: #888; margin-top: 8px; }}
  .stats-row {{ display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 12px 18px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .stat-card .label {{ font-size: 11px; color: #888; text-transform: uppercase;
                       letter-spacing: .5px; font-weight: 600; }}
  .stat-card .val {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
  .controls {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }}
  .ctrl-label {{ font-size: 12px; color: #555; font-weight: 600; }}
  select {{ padding: 4px 8px; border-radius: 6px; border: 1.5px solid #ccc;
            font-size: 12px; background: white; cursor: pointer; }}
  input[type=range] {{ accent-color: #2d6cdf; }}
</style>
</head>
<body>
<div class="header">
  <h1>Reaction Time vs Performance — LBF Human Players</h1>
  <div class="sub">
    Each dot = one player session &nbsp;·&nbsp;
    X axis = median time between actions &nbsp;·&nbsp;
    Y axis = avg score per game &nbsp;·&nbsp;
    Generated {e(generated)}
  </div>
</div>
<div style="background:#fff8e1;border-left:4px solid #f9a825;padding:12px 28px;font-size:12.5px;color:#555;line-height:1.7">
  <strong>How to read this chart:</strong>
  Dots further <strong>left</strong> = players who acted quickly between steps.
  Dots further <strong>right</strong> = players who paused longer between actions.
  Dots higher up = better average score.
  <strong>Red dots</strong> = flagged sessions &nbsp;·&nbsp;
  <strong>Red ring</strong> = suspiciously fast (10th percentile reaction time &lt; 150ms — possible key-holding or bot).
  The blue trend line shows whether thinking time correlates with performance — a flat line (r ≈ 0) means no strong relationship.
  Use the X-axis dropdown to switch between median reaction time and the 10th percentile (fastest actions).
</div>
<div class="container">
  <div class="stats-row" id="statsRow"></div>
  <div class="controls">
    <span class="ctrl-label">X axis:</span>
    <select id="xAxis" onchange="redraw()">
      <option value="med_rt">Median reaction time (ms)</option>
      <option value="p10_rt">10th pct reaction time (ms) — fastest actions</option>
    </select>
    <span class="ctrl-label" style="margin-left:12px">X max:</span>
    <input type="range" id="xMax" min="500" max="10000" step="500" value="5000" oninput="redraw()">
    <span id="xMaxLabel" style="font-size:12px;color:#555;min-width:60px">5000ms</span>
  </div>
  <div class="chart-wrap">
    <canvas id="scatter" height="500"></canvas>
    <div class="legend" id="legend"></div>
    <div class="threshold-note">
      <strong>Red dashed line</strong> = 150ms — on the p10 view, sessions left of this line had their fastest 10% of actions under 150ms, which is suspicious (possible key-holding or automation).
      <strong>Dot size</strong> = number of games played (larger = more games).
      <strong>Jitter</strong> applied to reduce overlap — hover dots for exact values.
    </div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const POINTS = {points_json};
const STATUS_COLORS = {color_map};
const SUSPICIOUS_RT = 150;

const PAD = {{top:40, right:40, bottom:60, left:70}};

function getCanvas() {{ return document.getElementById('scatter'); }}
function getCtx()    {{ return getCanvas().getContext('2d'); }}

function getXField() {{
  return document.getElementById('xAxis').value;
}}
function getXMax() {{
  const v = parseInt(document.getElementById('xMax').value);
  document.getElementById('xMaxLabel').textContent = v + 'ms';
  return v;
}}

function dataToCanvas(x, y, xMin, xMax, yMin, yMax, W, H) {{
  const cx = PAD.left + (x - xMin) / (xMax - xMin) * (W - PAD.left - PAD.right);
  const cy = H - PAD.bottom - (y - yMin) / (yMax - yMin) * (H - PAD.top - PAD.bottom);
  return [cx, cy];
}}

function redraw() {{
  const canvas = getCanvas();
  const W = canvas.offsetWidth;
  canvas.width  = W * window.devicePixelRatio;
  canvas.height = 500 * window.devicePixelRatio;
  canvas.style.height = '500px';
  const ctx = getCtx();
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  const H = 500;

  const xField = getXField();
  const xMax   = getXMax();
  const xMin   = 0;
  const yMin   = 0;
  const yMax   = 0.55;

  // Background
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 1;
  for (let y = 0; y <= 0.5; y += 0.1) {{
    const [,cy] = dataToCanvas(0, y, xMin, xMax, yMin, yMax, W, H);
    ctx.beginPath(); ctx.moveTo(PAD.left, cy); ctx.lineTo(W-PAD.right, cy); ctx.stroke();
    ctx.fillStyle='#999'; ctx.font='11px sans-serif'; ctx.textAlign='right';
    ctx.fillText(y.toFixed(1), PAD.left-6, cy+4);
  }}
  for (let x = 0; x <= xMax; x += xMax/10) {{
    const [cx,] = dataToCanvas(x, 0, xMin, xMax, yMin, yMax, W, H);
    ctx.beginPath(); ctx.moveTo(cx, PAD.top); ctx.lineTo(cx, H-PAD.bottom); ctx.stroke();
    ctx.fillStyle='#999'; ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(Math.round(x)+'ms', cx, H-PAD.bottom+16);
  }}

  // Suspicious RT threshold line (only for p10 x-axis)
  if (xField === 'p10_rt' && SUSPICIOUS_RT <= xMax) {{
    const [cx,] = dataToCanvas(SUSPICIOUS_RT, 0, xMin, xMax, yMin, yMax, W, H);
    ctx.strokeStyle='#e05050'; ctx.lineWidth=1.5; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(cx, PAD.top); ctx.lineTo(cx, H-PAD.bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#e05050'; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
    ctx.fillText('150ms', cx, PAD.top-6);
  }}

  // Max score reference line
  {{
    const [,cy] = dataToCanvas(0, 0.5, xMin, xMax, yMin, yMax, W, H);
    ctx.strokeStyle='#1a6e2e'; ctx.lineWidth=1.2; ctx.setLineDash([6,4]);
    ctx.beginPath(); ctx.moveTo(PAD.left, cy); ctx.lineTo(W-PAD.right, cy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#1a6e2e'; ctx.font='11px sans-serif'; ctx.textAlign='left';
    ctx.fillText('max score 0.5', PAD.left+4, cy-5);
  }}

  // Axes labels
  ctx.fillStyle='#333'; ctx.font='bold 12px sans-serif';
  ctx.textAlign='center';
  ctx.fillText(xField==='med_rt'?'Median reaction time (ms)':'10th percentile reaction time (ms)',
               PAD.left + (W-PAD.left-PAD.right)/2, H-8);
  ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Avg score per game', 0, 0); ctx.restore();

  // Dots — with deterministic jitter to reduce overlap
  const valid = POINTS.filter(p => p[xField] !== null && p[xField] <= xMax);
  // Deterministic jitter based on session short id hash
  function hashJitter(str, scale) {{
    let h = 0;
    for (let i = 0; i < str.length; i++) h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
    return ((h & 0xffff) / 0xffff - 0.5) * scale;
  }}
  valid.forEach(p => {{
    const jx = hashJitter(p.short + 'x', xMax * 0.012);
    const jy = hashJitter(p.short + 'y', 0.018);
    const [cx, cy] = dataToCanvas(p[xField] + jx, p.score + jy, xMin, xMax, yMin, yMax, W, H);
    const r = 5 + p.games * 1.0;
    const col = STATUS_COLORS[p.status] || '#888';
    ctx.globalAlpha = 0.78;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.fill();
    if (p.rt_warn) {{
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#e05050'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(cx, cy, r+4, 0, Math.PI*2); ctx.stroke();
    }}
    ctx.globalAlpha = 1;
  }});

  // Trend line (simple linear regression on ok sessions)
  const okPts = valid.filter(p => p.status==='ok' && p[xField] !== null);
  if (okPts.length > 3) {{
    const xs = okPts.map(p => p[xField]);
    const ys = okPts.map(p => p.score);
    const n  = xs.length;
    const mx = xs.reduce((a,b)=>a+b,0)/n;
    const my = ys.reduce((a,b)=>a+b,0)/n;
    const num = xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0);
    const den = xs.reduce((s,x)=>s+(x-mx)**2,0);
    if (den > 0) {{
      const slope = num/den, intercept = my - slope*mx;
      const x1=xMin, x2=xMax;
      const y1=slope*x1+intercept, y2=slope*x2+intercept;
      const [cx1,cy1] = dataToCanvas(x1,y1,xMin,xMax,yMin,yMax,W,H);
      const [cx2,cy2] = dataToCanvas(x2,y2,xMin,xMax,yMin,yMax,W,H);
      ctx.strokeStyle='rgba(45,108,223,0.5)'; ctx.lineWidth=2; ctx.setLineDash([8,5]);
      ctx.beginPath(); ctx.moveTo(cx1,cy1); ctx.lineTo(cx2,cy2); ctx.stroke();
      ctx.setLineDash([]);
      // Correlation label
      const r2 = num / Math.sqrt(den * ys.reduce((s,y)=>s+(y-my)**2,0));
      ctx.fillStyle='rgba(45,108,223,0.8)'; ctx.font='11px sans-serif'; ctx.textAlign='right';
      ctx.fillText('r = '+r2.toFixed(2)+' (OK sessions)', W-PAD.right-4, PAD.top+14);
    }}
  }}
}}

// Tooltip
const tooltip = document.getElementById('tooltip');
getCanvas().addEventListener('mousemove', function(e) {{
  const canvas = getCanvas();
  const rect   = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const W  = canvas.offsetWidth;
  const H  = 500;
  const xField = getXField();
  const xMax   = getXMax();

  let closest = null, minDist = 20;
  const valid = POINTS.filter(p => p[xField] !== null && p[xField] <= xMax);
  valid.forEach(p => {{
    const [cx,cy] = dataToCanvas(p[xField], p.score, 0, xMax, 0, 0.55, W, H);
    const d = Math.hypot(mx-cx, my-cy);
    if (d < minDist) {{ minDist=d; closest=p; }}
  }});

  if (closest) {{
    const statusLabel = closest.status.charAt(0).toUpperCase()+closest.status.slice(1);
    tooltip.innerHTML =
      '<strong>'+closest.short+'…</strong><br>'+
      '<span style="color:#aac">Status</span>: <strong>'+statusLabel+'</strong><br>'+
      '<span style="color:#aac">Score</span>: <strong>'+closest.score.toFixed(3)+'</strong><br>'+
      '<span style="color:#aac">Median RT</span>: <strong>'+closest.med_rt+'ms</strong><br>'+
      '<span style="color:#aac">10th pct RT</span>: <strong>'+(closest.p10_rt||'—')+'ms</strong><br>'+
      '<span style="color:#aac">Games</span>: <strong>'+closest.games+'</strong><br>'+
      (closest.reasons!=='none'?'<span style="color:#e09090">'+closest.reasons+'</span>':'');
    tooltip.style.display = 'block';
    let tx = e.clientX+14, ty = e.clientY-10;
    if (tx+210>window.innerWidth) tx=e.clientX-220;
    tooltip.style.left=tx+'px'; tooltip.style.top=ty+'px';
  }} else {{
    tooltip.style.display='none';
  }}
}});
getCanvas().addEventListener('mouseleave', ()=>tooltip.style.display='none');

// Stats cards
function buildStats() {{
  const ok = POINTS.filter(p=>p.status==='ok'&&p.med_rt!==null);
  const flagged = POINTS.filter(p=>p.status==='flagged'&&p.med_rt!==null);
  const allRts  = POINTS.filter(p=>p.med_rt!==null).map(p=>p.med_rt).sort((a,b)=>a-b);
  const medAll  = allRts[Math.floor(allRts.length/2)]||0;
  const suspicious = POINTS.filter(p=>p.rt_warn).length;

  const cards = [
    {{label:'Sessions with RT data', val: POINTS.filter(p=>p.med_rt!==null).length}},
    {{label:'Median RT (all sessions)', val: Math.round(medAll)+'ms'}},
    {{label:'Median RT (OK sessions)', val: Math.round(ok.map(p=>p.med_rt).sort((a,b)=>a-b)[Math.floor(ok.length/2)]||0)+'ms'}},
    {{label:'Suspicious (p10 < 150ms)', val: suspicious, color: suspicious>0?'#e05050':undefined}},
  ];
  const row = document.getElementById('statsRow');
  cards.forEach(c=>{{
    const d=document.createElement('div'); d.className='stat-card';
    d.innerHTML='<div class="label">'+c.label+'</div>'+
      '<div class="val"'+(c.color?' style="color:'+c.color+'"':'')+'>'+(c.val||0)+'</div>';
    row.appendChild(d);
  }});
}}

// Legend
function buildLegend() {{
  const items = [
    {{color:'#52be80', label:'OK'}},
    {{color:'#e05050', label:'Flagged'}},
    {{color:'#ba68c8', label:'Disqualified'}},
    {{color:'rgba(45,108,223,0.5)', label:'Trend line (OK sessions)', dash:true}},
  ];
  // Red ring item
  const ringItem = document.createElement('div');
  ringItem.className = 'legend-item';
  ringItem.innerHTML =
    '<svg width="22" height="22" style="flex-shrink:0"><circle cx="11" cy="11" r="8" fill="rgba(45,180,100,0.5)" stroke="#e05050" stroke-width="2.5"/></svg>' +
    'Red ring = p10 RT &lt; 150ms (suspiciously fast)';
  document.getElementById('legend').appendChild(ringItem);
  const leg = document.getElementById('legend');
  items.forEach(it=>{{
    const d=document.createElement('div'); d.className='legend-item';
    d.innerHTML='<div class="legend-dot" style="background:'+it.color+
      (it.dash?';border-radius:0;height:3px;width:20px':'')+'"></div>'+it.label;
    leg.appendChild(d);
  }});
  const sz = document.createElement('div'); sz.className='legend-item';
  sz.style.marginLeft='12px'; sz.style.color='#888';
  sz.textContent='Dot size = number of games played';
  leg.appendChild(sz);
}}

window.addEventListener('resize', redraw);
buildStats();
buildLegend();
redraw();
</script>
</body>
</html>"""


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(sessions: list[dict]) -> None:
    flagged      = [s for s in sessions if s["status"] == "flagged"]
    disqualified = [s for s in sessions if s["status"] == "disqualified"]
    zero_score   = [s for s in sessions if s["avg_score"] <= FLAG_SCORE]

    print("\n" + "=" * 70)
    print("  LBF HUMAN DATA — EFFORT ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Total sessions    : {len(sessions)}")
    print(f"  Total games       : {sum(s['num_games'] for s in sessions)}")
    print(f"  Zero-score sess.  : {len(zero_score)}  ({len(zero_score)/len(sessions):.1%})")
    print(f"  Flagged           : {len(flagged)}  (review before using data)")
    print(f"  Disqualified      : {len(disqualified)}  (incomplete, exclude from analysis)")
    print(f"  OK                : {len(sessions)-len(flagged)-len(disqualified)}")
    print("=" * 70)

    for label, subset in [
        ("FLAGGED — review before using", flagged),
        ("DISQUALIFIED — incomplete sessions (<8 games)", disqualified),
    ]:
        if not subset:
            continue
        print(f"\n  {label}")
        print(f"  {'Session ID':<38} {'Games':>5} {'AvgScore':>9} {'IdleRate':>9}  Reasons")
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

    scatter_html = build_scatter_html(sessions, generated)
    scatter_path = PLOTS_DIR / "reaction_time_scatter.html"
    scatter_path.write_text(scatter_html, encoding="utf-8")
    print(f"[html] Saved → {scatter_path}")

    print_report(sessions)


if __name__ == "__main__":
    main()