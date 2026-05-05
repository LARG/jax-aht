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
DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "social-laws-project")
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "NEURIPS-2026-UPDATED")


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
        default=2,
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
        default="neurips_curriculum_variance.html",
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
        default="plotting_data/neurips_curriculum_cache.pkl",
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
        # Normalize nested lists.
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
    agents = ["2 agents", "3 agents", "4 agents", "5 agents", "6 agents", "10 agents"]

    if not args.entity or not args.project:
        raise ValueError(
            "W&B entity/project not set. Pass --entity/--project, set WANDB_ENTITY and "
            "WANDB_PROJECT, or edit DEFAULT_WANDB_ENTITY/DEFAULT_WANDB_PROJECT in plotting/neurips.py."
        )

    try:
        run_filters = json.loads(args.run_filters)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --run-filters: {exc}") from exc

    path = f"{args.entity}/{args.project}"

    # Load processed data cache if available to avoid repeated W&B calls.
    data_loaded_from_cache = False
    if args.data_cache and os.path.exists(args.data_cache) and not args.refresh_data_cache:
        print(f"Loading processed data cache: {args.data_cache}")
        with open(args.data_cache, "rb") as f:
            cached = pickle.load(f)
        # Unpack cached collections
        ippo_social_law_returns_by_agent = cached.get("ippo_social_law_returns_by_agent")
        ippo_social_law_collisions_by_agent = cached.get("ippo_social_law_collisions_by_agent")
        ippo_social_law_centralized_returns_by_agent = cached.get("ippo_social_law_centralized_returns_by_agent")
        ippo_social_law_centralized_collisions_by_agent = cached.get("ippo_social_law_centralized_collisions_by_agent")
        ippo_social_law_yielding_returns_by_agent = cached.get("ippo_social_law_yielding_returns_by_agent")
        ippo_social_law_yielding_collisions_by_agent = cached.get("ippo_social_law_yielding_collisions_by_agent")
        ippo_social_law_yielding_centralized_returns_by_agent = cached.get("ippo_social_law_yielding_centralized_returns_by_agent")
        ippo_social_law_yielding_centralized_collisions_by_agent = cached.get("ippo_social_law_yielding_centralized_collisions_by_agent")
        mappo_social_law_centralized_returns_by_agent = cached.get("mappo_social_law_centralized_returns_by_agent")
        mappo_social_law_centralized_collisions_by_agent = cached.get("mappo_social_law_centralized_collisions_by_agent")
        mappo_social_law_yielding_centralized_returns_by_agent = cached.get("mappo_social_law_yielding_centralized_returns_by_agent")
        mappo_social_law_yielding_centralized_collisions_by_agent = cached.get("mappo_social_law_yielding_centralized_collisions_by_agent")
        data_loaded_from_cache = True
    else:
        api = wandb.Api()
        runs = api.runs(path=path, filters=run_filters, per_page=args.max_runs)

        ippo_social_law_filters = []
    if not data_loaded_from_cache:
        for agent in agents:
            ippo_social_law_filters.append({
                "tags": {
                    "$all": [agent, "marl_comparison", "ippo", "social law", "curriculum", "fixed switch"],
                    "$nin": ["OLD", "centralized", "social law yielding"],
                }
            })

        ippo_social_law_centralized_filters = []
        for agent in agents:
            ippo_social_law_centralized_filters.append({
                "tags": {
                    "$all": [agent, "marl_comparison", "ippo", "social law", "centralized", "curriculum", "fixed switch"],
                    "$nin": ["OLD", "social law yielding"],
                }
            })

        ippo_social_law_yielding_filters = []
        for agent in agents:
            ippo_social_law_yielding_filters.append({
                "tags": {
                    "$all": [agent, "marl_comparison", "ippo", "social law yielding", "curriculum", "fixed switch"],
                    "$nin": ["OLD", "centralized", "social law"],
                }
            })

        ippo_social_law_yielding_centralized_filters = []
        for agent in agents:
            ippo_social_law_yielding_centralized_filters.append({
                "tags": {
                    "$all": [agent, "marl_comparison", "ippo", "social law yielding", "centralized", "curriculum", "fixed switch"],
                    "$nin": ["OLD", "social law"],
                }
            })

        ippo_social_law_runs = []
        for ippo_social_law_filter in ippo_social_law_filters:
            ippo_social_law_runs.append(api.runs(path=path, filters=ippo_social_law_filter, per_page=args.max_runs))

        ippo_social_law_centralized_runs = []
        for ippo_social_law_centralized_filter in ippo_social_law_centralized_filters:
            ippo_social_law_centralized_runs.append(api.runs(path=path, filters=ippo_social_law_centralized_filter, per_page=args.max_runs))

        ippo_social_law_yielding_runs = []
        for ippo_social_law_yielding_filter in ippo_social_law_yielding_filters:
            ippo_social_law_yielding_runs.append(api.runs(path=path, filters=ippo_social_law_yielding_filter, per_page=args.max_runs))

        ippo_social_law_yielding_centralized_runs = []
        for ippo_social_law_yielding_centralized_filter in ippo_social_law_yielding_centralized_filters:
            ippo_social_law_yielding_centralized_runs.append(api.runs(path=path, filters=ippo_social_law_yielding_centralized_filter, per_page=args.max_runs))

        mappo_social_law_centralized_filters = []
        for agent in agents:
            mappo_social_law_centralized_filters.append({
                "tags": {
                    "$all": [agent, "marl_comparison", "mappo", "social law", "centralized", "curriculum", "fixed switch"],
                    "$nin": ["OLD", "social law yielding"],
                }
            })

        mappo_social_law_yielding_centralized_filters = []
        for agent in agents:
            mappo_social_law_yielding_centralized_filters.append({
                "tags": {
                    "$all": [agent, "marl_comparison", "mappo", "social law yielding", "centralized", "curriculum", "fixed switch"],
                    "$nin": ["OLD", "social law"],
                }
            })

        mappo_social_law_centralized_runs = []
        for mappo_social_law_centralized_filter in mappo_social_law_centralized_filters:
            mappo_social_law_centralized_runs.append(api.runs(path=path, filters=mappo_social_law_centralized_filter, per_page=args.max_runs))

        mappo_social_law_yielding_centralized_runs = []
        for mappo_social_law_yielding_centralized_filter in mappo_social_law_yielding_centralized_filters:
            mappo_social_law_yielding_centralized_runs.append(api.runs(path=path, filters=mappo_social_law_yielding_centralized_filter, per_page=args.max_runs))

        for i in range(len(agents)):
            print(f"IPPO (Social Law) {agents[i]} runs: {len(ippo_social_law_runs[i])}")
            print(f"IPPO (Social Law + Centralized) {agents[i]} runs: {len(ippo_social_law_centralized_runs[i])}")
            print(f"IPPO (Social Law Yielding) {agents[i]} runs: {len(ippo_social_law_yielding_runs[i])}")
            print(f"IPPO (Social Law Yielding + Centralized) {agents[i]} runs: {len(ippo_social_law_yielding_centralized_runs[i])}")

            print(f"MAPPO (Social Law + Centralized) {agents[i]} runs: {len(mappo_social_law_centralized_runs[i])}")
            print(f"MAPPO (Social Law Yielding + Centralized) {agents[i]} runs: {len(mappo_social_law_yielding_centralized_runs[i])}")
    else:
        print(f"Using cached data from {args.data_cache}; skipping W&B queries and collection")
    def collect_metric_series(
        run_collections: list[Any],
        agent_labels: list[str],
        returns_metric: str,
        collisions_metric: str,
        max_workers: int = 4,
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        returns_by_agent: dict[str, list[Any]] = {agent: [] for agent in agent_labels}
        collisions_by_agent: dict[str, list[Any]] = {agent: [] for agent in agent_labels}

        print(
            f"Collecting metrics: returns={returns_metric}, collisions={collisions_metric}"
        )

        def _collect_for_agent(agent_label: str, runs: Any) -> tuple[str, list[Any], list[Any]]:
            print(f"  Processing {agent_label}: {len(runs)} collected runs")
            returns_metric_keys = [returns_metric]
            collisions_metric_keys = [collisions_metric]
            print("    Fetch mode: first_available")

            return_series_batch = fetch_run_series(
                run=runs,
                metric_keys=returns_metric_keys,
                x_key=args.x_key,
                mode="first_available",
            )
            collision_series_batch = fetch_run_series(
                run=runs,
                metric_keys=collisions_metric_keys,
                x_key=args.x_key,
                mode="first_available",
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
                    returns_by_agent[agent_label].extend(return_series_batch)
                if collision_series_batch:
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

    ippo_social_law_returns_by_agent, ippo_social_law_collisions_by_agent = collect_metric_series(
        run_collections=ippo_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        max_workers=args.max_workers,
    )
    print_collection_summary("IPPO (Social Law)", ippo_social_law_returns_by_agent, ippo_social_law_collisions_by_agent)
    ippo_social_law_centralized_returns_by_agent, ippo_social_law_centralized_collisions_by_agent = collect_metric_series(
        run_collections=ippo_social_law_centralized_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        max_workers=args.max_workers,
    )
    print_collection_summary("IPPO (Social Law + Centralized)", ippo_social_law_centralized_returns_by_agent, ippo_social_law_centralized_collisions_by_agent)
    ippo_social_law_yielding_returns_by_agent, ippo_social_law_yielding_collisions_by_agent = collect_metric_series(
        run_collections=ippo_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        max_workers=args.max_workers,
    )
    print_collection_summary("IPPO (Social Law Yielding)", ippo_social_law_yielding_returns_by_agent, ippo_social_law_yielding_collisions_by_agent)
    ippo_social_law_yielding_centralized_returns_by_agent, ippo_social_law_yielding_centralized_collisions_by_agent = collect_metric_series(
        run_collections=ippo_social_law_yielding_centralized_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        max_workers=args.max_workers,
    )
    print_collection_summary("IPPO (Social Law Yielding + Centralized)", ippo_social_law_yielding_centralized_returns_by_agent, ippo_social_law_yielding_centralized_collisions_by_agent)
    mappo_social_law_centralized_returns_by_agent, mappo_social_law_centralized_collisions_by_agent = collect_metric_series(
        run_collections=mappo_social_law_centralized_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        max_workers=args.max_workers,
    )
    print_collection_summary("MAPPO (Social Law + Centralized)", mappo_social_law_centralized_returns_by_agent, mappo_social_law_centralized_collisions_by_agent)
    mappo_social_law_yielding_centralized_returns_by_agent, mappo_social_law_yielding_centralized_collisions_by_agent = collect_metric_series(
        run_collections=mappo_social_law_yielding_centralized_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        max_workers=args.max_workers,
    )
    print_collection_summary("MAPPO (Social Law Yielding + Centralized)", mappo_social_law_yielding_centralized_returns_by_agent, mappo_social_law_yielding_centralized_collisions_by_agent)

    # Save processed data cache so future runs can skip W&B queries.
    if not data_loaded_from_cache and args.data_cache:
        cache_dir = os.path.dirname(args.data_cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cached = {
            "ippo_social_law_returns_by_agent": ippo_social_law_returns_by_agent,
            "ippo_social_law_collisions_by_agent": ippo_social_law_collisions_by_agent,
            "ippo_social_law_centralized_returns_by_agent": ippo_social_law_centralized_returns_by_agent,
            "ippo_social_law_centralized_collisions_by_agent": ippo_social_law_centralized_collisions_by_agent,
            "ippo_social_law_yielding_returns_by_agent": ippo_social_law_yielding_returns_by_agent,
            "ippo_social_law_yielding_collisions_by_agent": ippo_social_law_yielding_collisions_by_agent,
            "ippo_social_law_yielding_centralized_returns_by_agent": ippo_social_law_yielding_centralized_returns_by_agent,
            "ippo_social_law_yielding_centralized_collisions_by_agent": ippo_social_law_yielding_centralized_collisions_by_agent,
            "mappo_social_law_centralized_returns_by_agent": mappo_social_law_centralized_returns_by_agent,
            "mappo_social_law_centralized_collisions_by_agent": mappo_social_law_centralized_collisions_by_agent,
            "mappo_social_law_yielding_centralized_returns_by_agent": mappo_social_law_yielding_centralized_returns_by_agent,
            "mappo_social_law_yielding_centralized_collisions_by_agent": mappo_social_law_yielding_centralized_collisions_by_agent,
        }
        with open(args.data_cache, "wb") as f:
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
        # Build grouped returns series for this agent_label only.
        grouped_returns: dict[str, list[RunSeries]] = {}
        grouped_returns[f"IPPO (Social Law) | {agent_label}"] = ippo_social_law_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law + Centralized) | {agent_label}"] = ippo_social_law_centralized_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law Yielding) | {agent_label}"] = ippo_social_law_yielding_returns_by_agent[agent_label]
        grouped_returns[f"IPPO (Social Law Yielding + Centralized) | {agent_label}"] = ippo_social_law_yielding_centralized_returns_by_agent[agent_label]
        grouped_returns[f"MAPPO (Social Law + Centralized) | {agent_label}"] = mappo_social_law_centralized_returns_by_agent[agent_label]
        grouped_returns[f"MAPPO (Social Law Yielding + Centralized) | {agent_label}"] = mappo_social_law_yielding_centralized_returns_by_agent[agent_label]

        # Build grouped collisions series for this agent_label only.
        grouped_collisions: dict[str, list[RunSeries]] = {}
        grouped_collisions[f"IPPO (Social Law) | {agent_label}"] = ippo_social_law_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law + Centralized) | {agent_label}"] = ippo_social_law_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law Yielding) | {agent_label}"] = ippo_social_law_yielding_collisions_by_agent[agent_label]
        grouped_collisions[f"IPPO (Social Law Yielding + Centralized) | {agent_label}"] = ippo_social_law_yielding_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"MAPPO (Social Law + Centralized) | {agent_label}"] = mappo_social_law_centralized_collisions_by_agent[agent_label]
        grouped_collisions[f"MAPPO (Social Law Yielding + Centralized) | {agent_label}"] = mappo_social_law_yielding_centralized_collisions_by_agent[agent_label]

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
