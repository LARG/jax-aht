"""Create variance plots from Weights & Biases runs using Plotly.

Example:
    python plotting/neurips.py \
        --entity my_entity \
        --project my_project \
        --metric Eval/EgoReturn \
        --metric-aliases eval/ego_return test/ego_return \
        --tags neurips exp_v2 \
        --output results/neurips_variance.html
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import wandb


# Set these once for your common workspace defaults.
DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jeffreychen287-the-university-of-texas-at-austin")
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "NEURIPS")


@dataclass
class RunSeries:
    """Single run's x/y series and metadata."""

    run_id: str
    run_name: str
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull W&B runs and create Plotly variance plots grouped by tag."
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=DEFAULT_WANDB_ENTITY,
        help=(
            "W&B entity/team. Defaults to DEFAULT_WANDB_ENTITY or WANDB_ENTITY env var "
            "if set."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help=(
            "W&B project. Defaults to DEFAULT_WANDB_PROJECT or WANDB_PROJECT env var "
            "if set."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Primary metric key in W&B history to plot on the y-axis",
    )
    parser.add_argument(
        "--metric-aliases",
        nargs="*",
        default=None,
        help=(
            "Optional fallback metric keys treated as equivalent to --metric. "
            "For each history row, the first available key in priority order is used."
        ),
    )
    parser.add_argument(
        "--x-key",
        type=str,
        default="train_step",
        help="History key to use for x-axis (default: train_step)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Required tags. A run is included only if it contains all provided tags.",
    )
    parser.add_argument(
        "--run-filters",
        type=str,
        default="{}",
        help='JSON dict passed to wandb.Api().runs(filters=...). Example: "{\"group\": \"neurips\"}"',
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1000,
        help="Maximum runs to pull from W&B (default: 1000)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker threads used to fetch W&B history (default: 4)",
    )
    parser.add_argument(
        "--min-runs-per-point",
        type=int,
        default=1,
        help="Minimum number of runs required at an x value to include it (default: 1)",
    )
    parser.add_argument(
        "--show-individual-runs",
        action="store_true",
        help="Overlay each run as a faint line",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="W&B Variance Plot",
        help="Figure title",
    )
    parser.add_argument(
        "--show-plot-title",
        action="store_true",
        help="Show plot titles in figures and the combined HTML report (default: hidden)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="neurips_variance.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--png-output",
        type=str,
        default=None,
        help="Optional PNG output path (requires kaleido)",
    )
    parser.add_argument(
        "--data-cache",
        type=str,
        default="plotting_data/neurips_cache.pkl",
        help="Optional pickle file for processed series. Existing cache skips W&B requests unless --refresh-data-cache is set.",
    )
    parser.add_argument(
        "--refresh-data-cache",
        action="store_true",
        help="Ignore existing --data-cache and refetch data from W&B.",
    )
    return parser.parse_args()


def run_has_all_tags(run: Any, required_tags: list[str] | None) -> bool:
    if not required_tags:
        return True
    run_tags = {str(tag) for tag in (getattr(run, "tags", []) or [])}
    return all(str(tag) in run_tags for tag in required_tags)


def _to_float_array(values: list[Any]) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim == 0:
        out = np.asarray([float(out)])
    return out


def fetch_run_series(
    run: Any,
    metric_keys: list[str],
    x_key: str,
    mode: str = "first_available",
) -> list[RunSeries]:
    # Accept either a single wandb Run (has scan_history) or a collection of runs.
    if hasattr(run, "scan_history"):
        runs_to_process = [run]
    else:
        runs_to_process = list(run)

    output: list[RunSeries] = []

    for single_run in runs_to_process:
        if mode == "per_metric":
            for metric_key in metric_keys:
                records = single_run.scan_history(keys=[x_key, metric_key], page_size=10000)

                x_values: list[float] = []
                y_values: list[float] = []
                for row in records:
                    xv = row.get(x_key)
                    yv = row.get(metric_key)
                    if xv is None or yv is None:
                        continue
                    x_values.append(xv)
                    y_values.append(yv)

                if not x_values:
                    continue

                x = _to_float_array(x_values)
                y = _to_float_array(y_values)
                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                if x.size == 0:
                    continue

                dedup: dict[float, float] = {}
                for xv, yv in zip(x, y):
                    dedup[float(xv)] = float(yv)

                xs = np.asarray(sorted(dedup.keys()), dtype=float)
                ys = np.asarray([dedup[val] for val in xs], dtype=float)

                output.append(
                    RunSeries(
                        run_id=str(single_run.id),
                        run_name=f"{single_run.name} | {metric_key}",
                        x=xs,
                        y=ys,
                    )
                )
        else:
            records = single_run.scan_history(keys=[x_key] + metric_keys, page_size=10000)

            x_values: list[float] = []
            y_values: list[float] = []
            for row in records:
                xv = row.get(x_key)
                yv = None
                for metric_key in metric_keys:
                    candidate = row.get(metric_key)
                    if candidate is not None:
                        yv = candidate
                        break
                if xv is None or yv is None:
                    continue
                x_values.append(xv)
                y_values.append(yv)

            if not x_values:
                continue

            x = _to_float_array(x_values)
            y = _to_float_array(y_values)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if x.size == 0:
                continue

            # Keep only the last logged value for each x in this run.
            dedup: dict[float, float] = {}
            for xv, yv in zip(x, y):
                dedup[float(xv)] = float(yv)

            xs = np.asarray(sorted(dedup.keys()), dtype=float)
            ys = np.asarray([dedup[val] for val in xs], dtype=float)

            output.append(
                RunSeries(
                    run_id=str(single_run.id),
                    run_name=str(single_run.name),
                    x=xs,
                    y=ys,
                )
            )

    return output


def _agent_count_from_label(agent_label: str) -> int:
    return int(agent_label.split()[0])


def _expand_metric_for_ppo_agents(metric_name: str, agent_count: int) -> list[str]:
    metric_tail = metric_name[len("Train/") :] if metric_name.startswith("Train/") else metric_name
    return [f"Train/Agent_{idx}_Proj/{metric_tail}" for idx in range(1, agent_count + 1)]


def _flatten_series_groups(series_groups: list[Any]) -> list[RunSeries]:
    flattened: list[RunSeries] = []
    for item in series_groups:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _custom_sort_legend_keys(group_keys: list[str]) -> list[str]:
    """Sort legend keys so 'Agent N' entries sort numerically by N, not alphabetically."""
    def sort_key(key: str) -> tuple:
        # Extract agent number if present (e.g., "Agent 10" -> 10)
        parts = key.split(" | ")
        if len(parts) > 0 and parts[-1].startswith("Agent "):
            try:
                agent_num = int(parts[-1].replace("Agent ", ""))
                # Return (base_part, 1, agent_num) so agents sort together numerically
                base = " | ".join(parts[:-1])
                return (base, 1, agent_num)
            except ValueError:
                pass
        # Non-agent entries: return (key, 0, 0) so they sort before agents
        return (key, 0, 0)

    return sorted(group_keys, key=sort_key)


def _strip_agent_count_label(group_name: str, agent_count_label: str) -> str:
    """Remove '| {agent_count_label}' from a legend name."""
    suffix = f" | {agent_count_label}"
    return group_name.replace(suffix, "")


def aggregate_condition(series_list: list[RunSeries], min_runs_per_point: int) -> dict[str, np.ndarray]:
    by_x: dict[float, list[float]] = defaultdict(list)
    for series in series_list:
        for xv, yv in zip(series.x, series.y):
            by_x[float(xv)].append(float(yv))

    x_sorted = np.asarray(sorted(by_x.keys()), dtype=float)
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    filtered_x: list[float] = []

    for xv in x_sorted:
        ys = np.asarray(by_x[float(xv)], dtype=float)
        if ys.size < min_runs_per_point:
            continue
        filtered_x.append(float(xv))
        means.append(float(np.mean(ys)))
        stds.append(float(np.std(ys)))
        counts.append(int(ys.size))

    return {
        "x": np.asarray(filtered_x, dtype=float),
        "mean": np.asarray(means, dtype=float),
        "std": np.asarray(stds, dtype=float),
        "count": np.asarray(counts, dtype=int),
    }


def make_variance_figure(
    grouped_series: dict[str, list[RunSeries]],
    metric: str,
    x_key: str,
    title: str,
    min_runs_per_point: int,
    show_individual_runs: bool,
    y_axis_title: str,
    agent_count_label: str | None = None,
    show_plot_title: bool = False,
) -> go.Figure:
    fig = go.Figure()
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    sorted_keys = _custom_sort_legend_keys(list(grouped_series.keys()))
    for idx, group_name in enumerate(sorted_keys):
        raw_series_list = grouped_series[group_name]
        # Normalize nested lists (e.g., PPO per-run per-agent collections)
        if any(isinstance(item, list) for item in raw_series_list):
            series_list = _flatten_series_groups(raw_series_list)
        else:
            series_list = raw_series_list

        # Strip agent-count label for cleaner legend.
        display_name = group_name
        if agent_count_label is not None:
            display_name = _strip_agent_count_label(group_name, agent_count_label)

        color = palette[idx % len(palette)]

        if show_individual_runs:
            for series in series_list:
                fig.add_trace(
                    go.Scatter(
                        x=series.x,
                        y=series.y,
                        mode="lines",
                        line={"color": color, "width": 1},
                        opacity=0.15,
                        name=f"{display_name} run",
                        legendgroup=display_name,
                        showlegend=False,
                        hovertemplate=(
                            "Group: "
                            + display_name
                            + "<br>Run: "
                            + series.run_name
                            + "<br>"
                            + x_key
                            + ": %{x}<br>"
                            + metric
                            + ": %{y}<extra></extra>"
                        ),
                    )
                )

        agg = aggregate_condition(series_list, min_runs_per_point=min_runs_per_point)
        if agg["x"].size == 0:
            continue

        upper = agg["mean"] + agg["std"]
        lower = agg["mean"] - agg["std"]

        fig.add_trace(
            go.Scatter(
                x=agg["x"],
                y=upper,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=display_name,
                name=f"{display_name} +1 std",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=agg["x"],
                y=lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba{tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}",
                hoverinfo="skip",
                showlegend=False,
                legendgroup=display_name,
                name=f"{display_name} -1 std",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=agg["x"],
                y=agg["mean"],
                mode="lines",
                line={"color": color, "width": 3},
                name=f"{display_name} (n={len(series_list)})",
                legendgroup=display_name,
                customdata=np.stack([agg["std"], agg["count"]], axis=-1),
                hovertemplate=(
                    "Group: "
                    + display_name
                    + "<br>"
                    + x_key
                    + ": %{x}<br>Mean: %{y:.4f}<br>Std: %{customdata[0]:.4f}"
                    + "<br>Runs at x: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title if show_plot_title else None,
        template="plotly_white",
        xaxis_title="Update Step",
        yaxis_title=y_axis_title,
        legend_title="Legend",
        hovermode="x unified",
        width=1100,
        height=650,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    metric_keys = [args.metric] + [m for m in (args.metric_aliases or []) if m != args.metric]

    # Recon env uses N in {2,3,4,5} 
    agents = ["2 agents", "3 agents", "4 agents", "5 agents"]

    if not args.entity or not args.project:
        raise ValueError(
            "W&B entity/project not set. Pass --entity/--project, set WANDB_ENTITY and "
            "WANDB_PROJECT, or edit DEFAULT_WANDB_ENTITY/DEFAULT_WANDB_PROJECT in this script."
        )

    try:
        run_filters = json.loads(args.run_filters)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --run-filters: {exc}") from exc

    path = f"{args.entity}/{args.project}"


    def _n(agent_label: str) -> str:
        """'2 agents' -> '2'"""
        return agent_label.split()[0]

    def _group_tag(agent_label: str, condition: str) -> str:
        n = _n(agent_label)
        if condition == "no_law":
            return f"CoopRecon_Compare_No_law_{n}Agent"
        elif condition == "law_0.0":
            return f"CoopRecon_Compare_Law_0.0_{n}Agent"
        elif condition == "law_0.1":
            return f"CoopRecon_Compare_Law_0.1_{n}Agent"
        elif condition == "law_0.2":
            return f"CoopRecon_Compare_Law_{n}Agent"
        elif condition == "cur_0.0":
            return f"CoopRecon_Compare_Curriculum_law_0_0_{n}Agent"
        elif condition == "cur_0.1":
            return f"CoopRecon_Compare_Curriculum_law_0_1_{n}Agent"
        elif condition == "cur_0.2":
            return f"CoopRecon_Compare_Curriculum_law_0_2_{n}Agent"
        raise ValueError(f"Unknown condition: {condition}")

    def _make_filters(algo: str, condition: str, exclude: list[str] | None = None) -> list[dict]:
        """Build one filter dict per agent for a given algo + law condition."""
        filters = []
        for agent in agents:
            must_have = [agent, algo, _group_tag(agent, condition)]
            f: dict = {"tags": {"$all": must_have}}
            if exclude:
                f["tags"]["$nin"] = exclude
            filters.append(f)
        return filters

    # ------------------------------------------------------------------
    # IPPO
    # ------------------------------------------------------------------
    ippo_no_law_filters   = _make_filters("ippo", "no_law",   exclude=["curriculum"])
    ippo_law_0_0_filters  = _make_filters("ippo", "law_0.0",  exclude=["curriculum"])
    ippo_law_0_1_filters  = _make_filters("ippo", "law_0.1",  exclude=["curriculum"])
    ippo_law_0_2_filters  = _make_filters("ippo", "law_0.2",  exclude=["curriculum"])

    ippo_no_law_runs   = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_no_law_filters]
    ippo_law_0_0_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_law_0_0_filters]
    ippo_law_0_1_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_law_0_1_filters]
    ippo_law_0_2_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_law_0_2_filters]

    # ------------------------------------------------------------------
    # MAPPO
    # ------------------------------------------------------------------
    mappo_no_law_filters  = _make_filters("mappo", "no_law",  exclude=["curriculum"])
    mappo_law_0_0_filters = _make_filters("mappo", "law_0.0", exclude=["curriculum"])
    mappo_law_0_1_filters = _make_filters("mappo", "law_0.1", exclude=["curriculum"])
    mappo_law_0_2_filters = _make_filters("mappo", "law_0.2", exclude=["curriculum"])

    mappo_no_law_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_no_law_filters]
    mappo_law_0_0_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_law_0_0_filters]
    mappo_law_0_1_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_law_0_1_filters]
    mappo_law_0_2_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_law_0_2_filters]

    # ------------------------------------------------------------------
    # PPO (independent agents — per-agent metrics tracked separately)
    # ------------------------------------------------------------------
    ppo_no_law_filters  = _make_filters("ppo", "no_law",  exclude=["creppo"])
    ppo_law_0_0_filters = _make_filters("ppo", "law_0.0", exclude=["creppo"])
    ppo_law_0_1_filters = _make_filters("ppo", "law_0.1", exclude=["creppo"])
    ppo_law_0_2_filters = _make_filters("ppo", "law_0.2", exclude=["creppo"])

    ppo_no_law_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ppo_no_law_filters]
    ppo_law_0_0_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ppo_law_0_0_filters]
    ppo_law_0_1_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ppo_law_0_1_filters]
    ppo_law_0_2_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ppo_law_0_2_filters]

    # ------------------------------------------------------------------
    # CREPPO
    # ------------------------------------------------------------------
    creppo_no_law_filters  = _make_filters("creppo", "no_law")
    creppo_law_0_0_filters = _make_filters("creppo", "law_0.0")
    creppo_law_0_1_filters = _make_filters("creppo", "law_0.1")
    creppo_law_0_2_filters = _make_filters("creppo", "law_0.2")

    creppo_no_law_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in creppo_no_law_filters]
    creppo_law_0_0_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in creppo_law_0_0_filters]
    creppo_law_0_1_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in creppo_law_0_1_filters]
    creppo_law_0_2_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in creppo_law_0_2_filters]

    # ------------------------------------------------------------------
    # Curriculum: IPPO + MAPPO (law 0.0 / 0.1 / 0.2 only)
    # ------------------------------------------------------------------
    ippo_curriculum_law_0_0_filters  = _make_filters("ippo",  "cur_0.0", exclude=None)
    ippo_curriculum_law_0_1_filters  = _make_filters("ippo",  "cur_0.1", exclude=None)
    ippo_curriculum_law_0_2_filters  = _make_filters("ippo",  "cur_0.2", exclude=None)
    mappo_curriculum_law_0_0_filters = _make_filters("mappo", "cur_0.0", exclude=None)
    mappo_curriculum_law_0_1_filters = _make_filters("mappo", "cur_0.1", exclude=None)
    mappo_curriculum_law_0_2_filters = _make_filters("mappo", "cur_0.2", exclude=None)

    ippo_curriculum_law_0_0_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_curriculum_law_0_0_filters]
    ippo_curriculum_law_0_1_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_curriculum_law_0_1_filters]
    ippo_curriculum_law_0_2_runs  = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in ippo_curriculum_law_0_2_filters]
    mappo_curriculum_law_0_0_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_curriculum_law_0_0_filters]
    mappo_curriculum_law_0_1_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_curriculum_law_0_1_filters]
    mappo_curriculum_law_0_2_runs = [api.runs(path=path, filters=f, per_page=args.max_runs) for f in mappo_curriculum_law_0_2_filters]




    for i in range(len(agents)):
        print(f"IPPO No Law {agents[i]}: {len(ippo_no_law_runs[i])} runs")
        print(f"IPPO Law 0.0 {agents[i]}: {len(ippo_law_0_0_runs[i])} runs")
        print(f"IPPO Law 0.1 {agents[i]}: {len(ippo_law_0_1_runs[i])} runs")
        print(f"IPPO Law 0.2 {agents[i]}: {len(ippo_law_0_2_runs[i])} runs")
        print(f"MAPPO No Law {agents[i]}: {len(mappo_no_law_runs[i])} runs")
        print(f"MAPPO Law 0.0 {agents[i]}: {len(mappo_law_0_0_runs[i])} runs")
        print(f"MAPPO Law 0.1 {agents[i]}: {len(mappo_law_0_1_runs[i])} runs")
        print(f"MAPPO Law 0.2 {agents[i]}: {len(mappo_law_0_2_runs[i])} runs")
        print(f"PPO No Law {agents[i]}: {len(ppo_no_law_runs[i])} runs")
        print(f"PPO Law 0.0 {agents[i]}: {len(ppo_law_0_0_runs[i])} runs")
        print(f"PPO Law 0.1 {agents[i]}: {len(ppo_law_0_1_runs[i])} runs")
        print(f"PPO Law 0.2 {agents[i]}: {len(ppo_law_0_2_runs[i])} runs")
        print(f"CREPPO No Law {agents[i]}: {len(creppo_no_law_runs[i])} runs")
        print(f"CREPPO Law 0.0 {agents[i]}: {len(creppo_law_0_0_runs[i])} runs")
        print(f"CREPPO Law 0.1 {agents[i]}: {len(creppo_law_0_1_runs[i])} runs")
        print(f"CREPPO Law 0.2 {agents[i]}: {len(creppo_law_0_2_runs[i])} runs")
        print(f"IPPO Curriculum 0.0 {agents[i]}: {len(ippo_curriculum_law_0_0_runs[i])} runs")
        print(f"IPPO Curriculum 0.1 {agents[i]}: {len(ippo_curriculum_law_0_1_runs[i])} runs")
        print(f"IPPO Curriculum 0.2 {agents[i]}: {len(ippo_curriculum_law_0_2_runs[i])} runs")
        print(f"MAPPO Curriculum 0.0 {agents[i]}: {len(mappo_curriculum_law_0_0_runs[i])} runs")
        print(f"MAPPO Curriculum 0.1 {agents[i]}: {len(mappo_curriculum_law_0_1_runs[i])} runs")
        print(f"MAPPO Curriculum 0.2 {agents[i]}: {len(mappo_curriculum_law_0_2_runs[i])} runs")

    def collect_metric_series(
        run_collections: list[Any],
        agent_labels: list[str],
        returns_metric: str,
        collisions_metric: str,
        ppo_per_agent_metrics: bool = False,
        max_workers: int = 4,
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        returns_by_agent: dict[str, list[Any]] = {agent: [] for agent in agent_labels}
        collisions_by_agent: dict[str, list[Any]] = {agent: [] for agent in agent_labels}

        print(
            f"Collecting metrics: returns={returns_metric}, collisions={collisions_metric}, "
            f"ppo_per_agent_metrics={ppo_per_agent_metrics}"
        )

        def _collect_for_agent(agent_label: str, runs: Any) -> tuple[str, list[Any], list[Any]]:
            print(f"  Processing {agent_label}: {len(runs)} collected runs")
            if ppo_per_agent_metrics:
                agent_count = _agent_count_from_label(agent_label)
                returns_metric_keys = _expand_metric_for_ppo_agents(returns_metric, agent_count)
                collisions_metric_keys = _expand_metric_for_ppo_agents(collisions_metric, agent_count)
                fetch_mode = "per_metric"
                print(f"    PPO metric keys (returns): {returns_metric_keys}")
                print(f"    PPO metric keys (collisions): {collisions_metric_keys}")
            else:
                returns_metric_keys = [returns_metric]
                collisions_metric_keys = [collisions_metric]
                fetch_mode = "first_available"

            print(f"    Fetch mode: {fetch_mode}")

            if ppo_per_agent_metrics:
                return_series_batch = []
                collision_series_batch = []
                for single_run in runs:
                    single_return_series = fetch_run_series(
                        run=single_run,
                        metric_keys=returns_metric_keys,
                        x_key=args.x_key,
                        mode=fetch_mode,
                    )
                    if single_return_series:
                        return_series_batch.append(single_return_series)

                    single_collision_series = fetch_run_series(
                        run=single_run,
                        metric_keys=collisions_metric_keys,
                        x_key=args.x_key,
                        mode=fetch_mode,
                    )
                    if single_collision_series:
                        collision_series_batch.append(single_collision_series)
            else:
                return_series_batch = fetch_run_series(
                    run=runs,
                    metric_keys=returns_metric_keys,
                    x_key=args.x_key,
                    mode=fetch_mode,
                )
                collision_series_batch = fetch_run_series(
                    run=runs,
                    metric_keys=collisions_metric_keys,
                    x_key=args.x_key,
                    mode=fetch_mode,
                )

            print(f"    Collected return series: {len(return_series_batch)}")
            print(f"    Collected collision series: {len(collision_series_batch)}")

            return agent_label, return_series_batch, collision_series_batch

        worker_count = max(1, min(max_workers, len(agent_labels)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_collect_for_agent, agent_label, runs): agent_label
                for agent_label, runs in zip(agent_labels, run_collections)
            }

            for future in as_completed(future_map):
                agent_label, return_series_batch, collision_series_batch = future.result()
                if return_series_batch:
                    if ppo_per_agent_metrics:
                        # Keep PPO data as list[run][agent_series] by extending with
                        # per-run groups (not adding an extra wrapper list).
                        returns_by_agent[agent_label].extend(return_series_batch)
                    else:
                        returns_by_agent[agent_label].extend(return_series_batch)
                if collision_series_batch:
                    if ppo_per_agent_metrics:
                        collisions_by_agent[agent_label].extend(collision_series_batch)
                    else:
                        collisions_by_agent[agent_label].extend(collision_series_batch)

                print(
                    f"    Totals so far -> {agent_label}: returns={len(returns_by_agent[agent_label])}, "
                    f"collisions={len(collisions_by_agent[agent_label])}"
                )

        return returns_by_agent, collisions_by_agent

    def print_collection_summary(name: str, returns_by_agent: dict[str, list[RunSeries]], collisions_by_agent: dict[str, list[RunSeries]]) -> None:
        print(f"Summary for {name}:")
        for agent_label in agents:
            returns_groups = returns_by_agent.get(agent_label, [])
            collisions_groups = collisions_by_agent.get(agent_label, [])
            if len(returns_groups) > 0 and isinstance(returns_groups[0], list):
                return_shape = [len(group) for group in returns_groups]
            else:
                return_shape = [len(returns_groups)]
            if len(collisions_groups) > 0 and isinstance(collisions_groups[0], list):
                collision_shape = [len(group) for group in collisions_groups]
            else:
                collision_shape = [len(collisions_groups)]
            print(
                f"  {agent_label}: returns={return_shape}, collisions={collision_shape}"
            )

    # Attempt to load previously cached processed series to skip W&B fetches.
    loaded_cache = False
    if args.data_cache and os.path.exists(args.data_cache) and not args.refresh_data_cache:
        print(f"Loading processed data cache: {args.data_cache}")
        with open(args.data_cache, "rb") as f:
            cached = pickle.load(f)
        ippo_no_law_ret    = cached.get("ippo_no_law_ret");    ippo_no_law_col    = cached.get("ippo_no_law_col")
        ippo_law_0_0_ret   = cached.get("ippo_law_0_0_ret");   ippo_law_0_0_col   = cached.get("ippo_law_0_0_col")
        ippo_law_0_1_ret   = cached.get("ippo_law_0_1_ret");   ippo_law_0_1_col   = cached.get("ippo_law_0_1_col")
        ippo_law_0_2_ret   = cached.get("ippo_law_0_2_ret");   ippo_law_0_2_col   = cached.get("ippo_law_0_2_col")
        mappo_no_law_ret   = cached.get("mappo_no_law_ret");   mappo_no_law_col   = cached.get("mappo_no_law_col")
        mappo_law_0_0_ret  = cached.get("mappo_law_0_0_ret");  mappo_law_0_0_col  = cached.get("mappo_law_0_0_col")
        mappo_law_0_1_ret  = cached.get("mappo_law_0_1_ret");  mappo_law_0_1_col  = cached.get("mappo_law_0_1_col")
        mappo_law_0_2_ret  = cached.get("mappo_law_0_2_ret");  mappo_law_0_2_col  = cached.get("mappo_law_0_2_col")
        ppo_no_law_ret     = cached.get("ppo_no_law_ret");     ppo_no_law_col     = cached.get("ppo_no_law_col")
        ppo_law_0_0_ret    = cached.get("ppo_law_0_0_ret");    ppo_law_0_0_col    = cached.get("ppo_law_0_0_col")
        ppo_law_0_1_ret    = cached.get("ppo_law_0_1_ret");    ppo_law_0_1_col    = cached.get("ppo_law_0_1_col")
        ppo_law_0_2_ret    = cached.get("ppo_law_0_2_ret");    ppo_law_0_2_col    = cached.get("ppo_law_0_2_col")
        creppo_no_law_ret  = cached.get("creppo_no_law_ret");  creppo_no_law_col  = cached.get("creppo_no_law_col")
        creppo_law_0_0_ret = cached.get("creppo_law_0_0_ret"); creppo_law_0_0_col = cached.get("creppo_law_0_0_col")
        creppo_law_0_1_ret = cached.get("creppo_law_0_1_ret"); creppo_law_0_1_col = cached.get("creppo_law_0_1_col")
        creppo_law_0_2_ret = cached.get("creppo_law_0_2_ret"); creppo_law_0_2_col = cached.get("creppo_law_0_2_col")
        ippo_cur_0_0_ret   = cached.get("ippo_cur_0_0_ret");   ippo_cur_0_0_col   = cached.get("ippo_cur_0_0_col")
        ippo_cur_0_1_ret   = cached.get("ippo_cur_0_1_ret");   ippo_cur_0_1_col   = cached.get("ippo_cur_0_1_col")
        ippo_cur_0_2_ret   = cached.get("ippo_cur_0_2_ret");   ippo_cur_0_2_col   = cached.get("ippo_cur_0_2_col")
        mappo_cur_0_0_ret  = cached.get("mappo_cur_0_0_ret");  mappo_cur_0_0_col  = cached.get("mappo_cur_0_0_col")
        mappo_cur_0_1_ret  = cached.get("mappo_cur_0_1_ret");  mappo_cur_0_1_col  = cached.get("mappo_cur_0_1_col")
        mappo_cur_0_2_ret  = cached.get("mappo_cur_0_2_ret");  mappo_cur_0_2_col  = cached.get("mappo_cur_0_2_col")
        loaded_cache = True

    RET  = "Train/returned_episode_returns"
    COL  = "Train/returned_episode_collisions"
    MW   = args.max_workers

    def collect(run_collections, label):
        ret, col = collect_metric_series(run_collections=run_collections, agent_labels=agents,
                                         returns_metric=RET, collisions_metric=COL, max_workers=MW)
        print_collection_summary(label, ret, col)
        return ret, col

    def collect_ppo(run_collections, label):
        ret, col = collect_metric_series(run_collections=run_collections, agent_labels=agents,
                                         returns_metric=RET, collisions_metric=COL,
                                         ppo_per_agent_metrics=True, max_workers=MW)
        print_collection_summary(label, ret, col)
        return ret, col

    if not loaded_cache:
        ippo_no_law_ret,    ippo_no_law_col    = collect(ippo_no_law_runs,    "IPPO No Law")
        ippo_law_0_0_ret,   ippo_law_0_0_col   = collect(ippo_law_0_0_runs,   "IPPO Law 0.0")
        ippo_law_0_1_ret,   ippo_law_0_1_col   = collect(ippo_law_0_1_runs,   "IPPO Law 0.1")
        ippo_law_0_2_ret,   ippo_law_0_2_col   = collect(ippo_law_0_2_runs,   "IPPO Law 0.2")
        mappo_no_law_ret,   mappo_no_law_col   = collect(mappo_no_law_runs,   "MAPPO No Law")
        mappo_law_0_0_ret,  mappo_law_0_0_col  = collect(mappo_law_0_0_runs,  "MAPPO Law 0.0")
        mappo_law_0_1_ret,  mappo_law_0_1_col  = collect(mappo_law_0_1_runs,  "MAPPO Law 0.1")
        mappo_law_0_2_ret,  mappo_law_0_2_col  = collect(mappo_law_0_2_runs,  "MAPPO Law 0.2")
        ppo_no_law_ret,     ppo_no_law_col     = collect_ppo(ppo_no_law_runs,   "PPO No Law")
        ppo_law_0_0_ret,    ppo_law_0_0_col    = collect_ppo(ppo_law_0_0_runs,  "PPO Law 0.0")
        ppo_law_0_1_ret,    ppo_law_0_1_col    = collect_ppo(ppo_law_0_1_runs,  "PPO Law 0.1")
        ppo_law_0_2_ret,    ppo_law_0_2_col    = collect_ppo(ppo_law_0_2_runs,  "PPO Law 0.2")
        creppo_no_law_ret,  creppo_no_law_col  = collect(creppo_no_law_runs,  "CREPPO No Law")
        creppo_law_0_0_ret, creppo_law_0_0_col = collect(creppo_law_0_0_runs, "CREPPO Law 0.0")
        creppo_law_0_1_ret, creppo_law_0_1_col = collect(creppo_law_0_1_runs, "CREPPO Law 0.1")
        creppo_law_0_2_ret, creppo_law_0_2_col = collect(creppo_law_0_2_runs, "CREPPO Law 0.2")
        ippo_cur_0_0_ret,   ippo_cur_0_0_col   = collect(ippo_curriculum_law_0_0_runs, "IPPO Curriculum 0.0")
        ippo_cur_0_1_ret,   ippo_cur_0_1_col   = collect(ippo_curriculum_law_0_1_runs, "IPPO Curriculum 0.1")
        ippo_cur_0_2_ret,   ippo_cur_0_2_col   = collect(ippo_curriculum_law_0_2_runs, "IPPO Curriculum 0.2")
        mappo_cur_0_0_ret,  mappo_cur_0_0_col  = collect(mappo_curriculum_law_0_0_runs, "MAPPO Curriculum 0.0")
        mappo_cur_0_1_ret,  mappo_cur_0_1_col  = collect(mappo_curriculum_law_0_1_runs, "MAPPO Curriculum 0.1")
        mappo_cur_0_2_ret,  mappo_cur_0_2_col  = collect(mappo_curriculum_law_0_2_runs, "MAPPO Curriculum 0.2")

    # Save cache
    if args.data_cache and not loaded_cache:
        cache_dir = os.path.dirname(args.data_cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cached = {
            "ippo_no_law_ret": ippo_no_law_ret, "ippo_no_law_col": ippo_no_law_col,
            "ippo_law_0_0_ret": ippo_law_0_0_ret, "ippo_law_0_0_col": ippo_law_0_0_col,
            "ippo_law_0_1_ret": ippo_law_0_1_ret, "ippo_law_0_1_col": ippo_law_0_1_col,
            "ippo_law_0_2_ret": ippo_law_0_2_ret, "ippo_law_0_2_col": ippo_law_0_2_col,
            "mappo_no_law_ret": mappo_no_law_ret, "mappo_no_law_col": mappo_no_law_col,
            "mappo_law_0_0_ret": mappo_law_0_0_ret, "mappo_law_0_0_col": mappo_law_0_0_col,
            "mappo_law_0_1_ret": mappo_law_0_1_ret, "mappo_law_0_1_col": mappo_law_0_1_col,
            "mappo_law_0_2_ret": mappo_law_0_2_ret, "mappo_law_0_2_col": mappo_law_0_2_col,
            "ppo_no_law_ret": ppo_no_law_ret, "ppo_no_law_col": ppo_no_law_col,
            "ppo_law_0_0_ret": ppo_law_0_0_ret, "ppo_law_0_0_col": ppo_law_0_0_col,
            "ppo_law_0_1_ret": ppo_law_0_1_ret, "ppo_law_0_1_col": ppo_law_0_1_col,
            "ppo_law_0_2_ret": ppo_law_0_2_ret, "ppo_law_0_2_col": ppo_law_0_2_col,
            "creppo_no_law_ret": creppo_no_law_ret, "creppo_no_law_col": creppo_no_law_col,
            "creppo_law_0_0_ret": creppo_law_0_0_ret, "creppo_law_0_0_col": creppo_law_0_0_col,
            "creppo_law_0_1_ret": creppo_law_0_1_ret, "creppo_law_0_1_col": creppo_law_0_1_col,
            "creppo_law_0_2_ret": creppo_law_0_2_ret, "creppo_law_0_2_col": creppo_law_0_2_col,
            "ippo_cur_0_0_ret": ippo_cur_0_0_ret, "ippo_cur_0_0_col": ippo_cur_0_0_col,
            "ippo_cur_0_1_ret": ippo_cur_0_1_ret, "ippo_cur_0_1_col": ippo_cur_0_1_col,
            "ippo_cur_0_2_ret": ippo_cur_0_2_ret, "ippo_cur_0_2_col": ippo_cur_0_2_col,
            "mappo_cur_0_0_ret": mappo_cur_0_0_ret, "mappo_cur_0_0_col": mappo_cur_0_0_col,
            "mappo_cur_0_1_ret": mappo_cur_0_1_ret, "mappo_cur_0_1_col": mappo_cur_0_1_col,
            "mappo_cur_0_2_ret": mappo_cur_0_2_ret, "mappo_cur_0_2_col": mappo_cur_0_2_col,
        }
        with open(args.data_cache, "wb") as f:
            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved processed data cache: {args.data_cache}")

    # ------------------------------------------------------------------
    # Build plots per agent group
    # ------------------------------------------------------------------
    collisions_key = COL

    def _make_output_path(base: str, agent_label: str, kind: str) -> str:
        root, ext = os.path.splitext(base)
        safe_label = agent_label.replace(" ", "_")
        return f"{root}_{safe_label}_{kind}{ext}"

    total_valid = 0
    all_figures: list[tuple[str, go.Figure]] = []
    for agent_label in agents:
        agent_count = _agent_count_from_label(agent_label)

        # Returns grouped by algorithm + law condition
        grouped_returns: dict[str, list[RunSeries]] = {
            f"IPPO No Law | {agent_label}":          ippo_no_law_ret[agent_label],
            f"IPPO Law 0.0 | {agent_label}":         ippo_law_0_0_ret[agent_label],
            f"IPPO Law 0.1 | {agent_label}":         ippo_law_0_1_ret[agent_label],
            f"IPPO Law 0.2 | {agent_label}":         ippo_law_0_2_ret[agent_label],
            f"IPPO Curriculum 0.0 | {agent_label}":  ippo_cur_0_0_ret[agent_label],
            f"IPPO Curriculum 0.1 | {agent_label}":  ippo_cur_0_1_ret[agent_label],
            f"IPPO Curriculum 0.2 | {agent_label}":  ippo_cur_0_2_ret[agent_label],
            f"MAPPO No Law | {agent_label}":         mappo_no_law_ret[agent_label],
            f"MAPPO Law 0.0 | {agent_label}":        mappo_law_0_0_ret[agent_label],
            f"MAPPO Law 0.1 | {agent_label}":        mappo_law_0_1_ret[agent_label],
            f"MAPPO Law 0.2 | {agent_label}":        mappo_law_0_2_ret[agent_label],
            f"MAPPO Curriculum 0.0 | {agent_label}": mappo_cur_0_0_ret[agent_label],
            f"MAPPO Curriculum 0.1 | {agent_label}": mappo_cur_0_1_ret[agent_label],
            f"MAPPO Curriculum 0.2 | {agent_label}": mappo_cur_0_2_ret[agent_label],
            f"CREPPO No Law | {agent_label}":        creppo_no_law_ret[agent_label],
            f"CREPPO Law 0.0 | {agent_label}":       creppo_law_0_0_ret[agent_label],
            f"CREPPO Law 0.1 | {agent_label}":       creppo_law_0_1_ret[agent_label],
            f"CREPPO Law 0.2 | {agent_label}":       creppo_law_0_2_ret[agent_label],
        }
        # PPO: per-agent series (each agent tracked separately)
        for law_name, ppo_ret in [
            ("No Law", ppo_no_law_ret), ("Law 0.0", ppo_law_0_0_ret),
            ("Law 0.1", ppo_law_0_1_ret), ("Law 0.2", ppo_law_0_2_ret),
        ]:
            for ai in range(agent_count):
                series_for_agent: list[RunSeries] = []
                for run_series_list in ppo_ret[agent_label]:
                    if isinstance(run_series_list, list) and len(run_series_list) > ai:
                        series_for_agent.append(run_series_list[ai])
                grouped_returns[f"PPO {law_name} | {agent_label} | Agent {ai+1}"] = series_for_agent

        # Collisions grouped identically
        grouped_collisions: dict[str, list[RunSeries]] = {
            f"IPPO No Law | {agent_label}":          ippo_no_law_col[agent_label],
            f"IPPO Law 0.0 | {agent_label}":         ippo_law_0_0_col[agent_label],
            f"IPPO Law 0.1 | {agent_label}":         ippo_law_0_1_col[agent_label],
            f"IPPO Law 0.2 | {agent_label}":         ippo_law_0_2_col[agent_label],
            f"IPPO Curriculum 0.0 | {agent_label}":  ippo_cur_0_0_col[agent_label],
            f"IPPO Curriculum 0.1 | {agent_label}":  ippo_cur_0_1_col[agent_label],
            f"IPPO Curriculum 0.2 | {agent_label}":  ippo_cur_0_2_col[agent_label],
            f"MAPPO No Law | {agent_label}":         mappo_no_law_col[agent_label],
            f"MAPPO Law 0.0 | {agent_label}":        mappo_law_0_0_col[agent_label],
            f"MAPPO Law 0.1 | {agent_label}":        mappo_law_0_1_col[agent_label],
            f"MAPPO Law 0.2 | {agent_label}":        mappo_law_0_2_col[agent_label],
            f"MAPPO Curriculum 0.0 | {agent_label}": mappo_cur_0_0_col[agent_label],
            f"MAPPO Curriculum 0.1 | {agent_label}": mappo_cur_0_1_col[agent_label],
            f"MAPPO Curriculum 0.2 | {agent_label}": mappo_cur_0_2_col[agent_label],
            f"CREPPO No Law | {agent_label}":        creppo_no_law_col[agent_label],
            f"CREPPO Law 0.0 | {agent_label}":       creppo_law_0_0_col[agent_label],
            f"CREPPO Law 0.1 | {agent_label}":       creppo_law_0_1_col[agent_label],
            f"CREPPO Law 0.2 | {agent_label}":       creppo_law_0_2_col[agent_label],
        }
        for law_name, ppo_col in [
            ("No Law", ppo_no_law_col), ("Law 0.0", ppo_law_0_0_col),
            ("Law 0.1", ppo_law_0_1_col), ("Law 0.2", ppo_law_0_2_col),
        ]:
            for ai in range(agent_count):
                series_for_agent = []
                for run_series_list in ppo_col[agent_label]:
                    if isinstance(run_series_list, list) and len(run_series_list) > ai:
                        series_for_agent.append(run_series_list[ai])
                grouped_collisions[f"PPO {law_name} | {agent_label} | Agent {ai+1}"] = series_for_agent



            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved processed data cache: {args.data_cache}")

    # Create separate plots per agent-count group and separate returns/collisions plots.
    collisions_key = "Train/returned_episode_collisions"

    def _make_output_path(base: str, agent_label: str, kind: str) -> str:
        root, ext = os.path.splitext(base)
        safe_label = agent_label.replace(" ", "_")
        return f"{root}_{safe_label}_{kind}{ext}"

    total_valid = 0
    all_figures: list[tuple[str, go.Figure]] = []
    for agent_label in agents:
        agent_count = _agent_count_from_label(agent_label)

        # Build grouped returns series for this agent_label only.
        grouped_returns: dict[str, list[RunSeries]] = {}
        grouped_returns[f"IPPO | {agent_label}"] = ippo_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Centralized) | {agent_label}"] = ippo_centralized_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law) | {agent_label}"] = ippo_social_law_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law + Centralized) | {agent_label}"] = ippo_social_law_centralized_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law Yielding) | {agent_label}"] = ippo_social_law_yielding_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law Yielding + Centralized) | {agent_label}"] = ippo_social_law_yielding_centralized_returns_by_agent[agent_label]
        grouped_returns[f"MAPPO (Centralized) | {agent_label}"] = mappo_centralized_returns_by_agent[agent_label]
        grouped_returns[f"MAPPO (Social Law + Centralized) | {agent_label}"] = mappo_social_law_centralized_returns_by_agent[agent_label]
        grouped_returns[f"MAPPO (Social Law Yielding + Centralized) | {agent_label}"] = mappo_social_law_yielding_centralized_returns_by_agent[agent_label]

        # PPO per-agent groups for returns
        ppo_groups = ppo_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent: list[RunSeries] = []
            for run_series_list in ppo_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"PPO | {agent_label} | Agent {ai+1}"] = series_for_agent

        ppo_sl_groups = ppo_social_law_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in ppo_sl_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"PPO (Social Law) | {agent_label} | Agent {ai+1}"] = series_for_agent

        ppo_sly_groups = ppo_social_law_yielding_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in ppo_sly_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"PPO (Social Law Yielding) | {agent_label} | Agent {ai+1}"] = series_for_agent

        # Build grouped collisions series for this agent_label only.
        grouped_collisions: dict[str, list[RunSeries]] = {}
        grouped_collisions[f"IPPO | {agent_label}"] = ippo_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Centralized) | {agent_label}"] = ippo_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law) | {agent_label}"] = ippo_social_law_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law + Centralized) | {agent_label}"] = ippo_social_law_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law Yielding) | {agent_label}"] = ippo_social_law_yielding_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law Yielding + Centralized) | {agent_label}"] = ippo_social_law_yielding_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"MAPPO (Centralized) | {agent_label}"] = mappo_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"MAPPO (Social Law + Centralized) | {agent_label}"] = mappo_social_law_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"MAPPO (Social Law Yielding + Centralized) | {agent_label}"] = mappo_social_law_yielding_centralized_collisions_by_agent[agent_label]

        # PPO per-agent groups for collisions
        ppo_groups_c = ppo_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in ppo_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"PPO | {agent_label} | Agent {ai+1}"] = series_for_agent

        ppo_sl_groups_c = ppo_social_law_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in ppo_sl_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"PPO (Social Law) | {agent_label} | Agent {ai+1}"] = series_for_agent

        ppo_sly_groups_c = ppo_social_law_yielding_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in ppo_sly_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"PPO (Social Law Yielding) | {agent_label} | Agent {ai+1}"] = series_for_agent

        # Make and save returns figure for this agent group.
        returns_title = f"{args.title} | {agent_label} | Returns"
        returns_fig = make_variance_figure(
            grouped_series=grouped_returns,
            metric=args.metric,
            x_key=args.x_key,
            title=returns_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((returns_title, returns_fig))
        print(f"Prepared returns plot: {returns_title}")
        if args.png_output:
            out_png = _make_output_path(args.png_output, agent_label, "returns")
            returns_fig.write_image(out_png)
            print(f"Saved PNG returns plot to: {out_png}")

        # Make and save collisions figure for this agent group.
        collisions_title = f"{args.title} | {agent_label} | Collisions"
        collisions_fig = make_variance_figure(
            grouped_series=grouped_collisions,
            metric=collisions_key,
            x_key=args.x_key,
            title=collisions_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Collisions",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((collisions_title, collisions_fig))
        print(f"Prepared collisions plot: {collisions_title}")
        if args.png_output:
            out_png_c = _make_output_path(args.png_output, agent_label, "collisions")
            collisions_fig.write_image(out_png_c)
            print(f"Saved PNG collisions plot to: {out_png_c}")

        total_valid += sum(len(series_list) for series_list in grouped_returns.values())

    html_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        f"  <title>{args.title}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    h1 { margin-bottom: 8px; }",
        "    .plot-block { margin: 24px 0 48px 0; }",
        "    .plot-title { margin: 0 0 12px 0; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{args.title}</h1>",
        f"  <p>Total plots: {len(all_figures)}</p>",
    ]

    for idx, (plot_title, fig) in enumerate(all_figures):
        html_parts.append("  <div class=\"plot-block\">")
        html_parts.append(f"    <h2 class=\"plot-title\">{plot_title}</h2>")
        fig_html = fig.to_html(full_html=False, include_plotlyjs=(idx == 0))
        html_parts.append(fig_html)
        html_parts.append("  </div>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"Saved combined interactive plot report to: {args.output}")
    print(f"Valid runs (returns grouped): {total_valid}")
    if args.tags:
        print(f"Required tags (all must be present): {args.tags}")
    if args.metric_aliases:
        print(f"Metric keys used (priority order): {metric_keys}")


if __name__ == "__main__":
    main()
