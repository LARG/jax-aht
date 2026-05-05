"""Create variance plots from Weights & Biases runs using Plotly.

Example:
    python plotting/neurips.py \
        --entity my_entity \
        --project my_project \
        --metric Eval/EgoReturn \
        --metric-aliases eval/ego_return test/ego_return \
        --tags neurips exp_v2 \
        --output results/neurips_alpha_variance.html
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import wandb
import time


# Set these once for your common workspace defaults.
DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "social-laws-project")
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "NEURIPS-2026-UPDATED")


RUN_HISTORY_CACHE: dict[str, list[dict[str, Any]]] = {}


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
        default=1,
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
        default="neurips_alpha_variance.html",
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
        default="plotting_data/neurips_alpha_history_cache.pkl",
        help="Optional pickle path for cached W&B history rows.",
    )
    parser.add_argument(
        "--refresh-data-cache",
        action="store_true",
        help="Ignore any existing --data-cache and refetch W&B history.",
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


def _get_run_history(
    single_run: Any, metric_keys: list[str] | None = None, x_key: str | None = None
) -> list[dict[str, Any]]:
    """Fetch a (possibly filtered) run history via `scan_history`.

    To reduce bandwidth and API load, pass `keys` to `scan_history` so W&B
    only returns the requested metric columns and the x-axis key.
    The run-history cache is keyed by run id + requested keys so different
    fetches don't collide.
    """
    run_id = str(single_run.id)
    key_parts: list[str] = [run_id]
    if metric_keys:
        key_parts.append("keys=" + ",".join(sorted(metric_keys)))
    if x_key:
        key_parts.append("x=" + x_key)
    cache_key = "::".join(key_parts)

    cached_history = RUN_HISTORY_CACHE.get(cache_key)
    if cached_history is not None:
        return cached_history

    scan_kwargs: dict[str, Any] = {"page_size": 10000}
    if metric_keys or x_key:
        keys = list(metric_keys or [])
        if x_key and x_key not in keys:
            keys.append(x_key)
        scan_kwargs["keys"] = keys

    history_rows = list(single_run.scan_history(**scan_kwargs))
    time.sleep(35.0)  # Be kind to W&B API and avoid hitting rate limits.
    RUN_HISTORY_CACHE[cache_key] = history_rows
    return history_rows


def _load_history_cache(cache_path: str, *, refresh: bool = False) -> None:
    if refresh or not cache_path or not os.path.exists(cache_path):
        return
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "history_cache" in payload:
        RUN_HISTORY_CACHE.update(payload["history_cache"])
        print(f"Loaded run-history cache from: {cache_path}")


def _save_history_cache(cache_path: str) -> None:
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump({"history_cache": RUN_HISTORY_CACHE}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    print(f"Saved run-history cache to: {cache_path}")


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
                history_rows = _get_run_history(single_run, metric_keys=[metric_key], x_key=x_key)
                x_values: list[float] = []
                y_values: list[float] = []
                for row in history_rows:
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
            # First-available priority across metric keys, fetched one key at a time.
            # Keep the first metric (in input order) that provides a value for each x.
            dedup_by_x: dict[float, float] = {}
            for metric_key in metric_keys:
                history_rows = _get_run_history(single_run, metric_keys=[metric_key], x_key=x_key)
                for row in history_rows:
                    xv = row.get(x_key)
                    yv = row.get(metric_key)
                    if xv is None or yv is None:
                        continue
                    try:
                        xvf = float(xv)
                        yvf = float(yv)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(xvf) or not np.isfinite(yvf):
                        continue
                    if xvf not in dedup_by_x:
                        dedup_by_x[xvf] = yvf

            if not dedup_by_x:
                continue

            xs = np.asarray(sorted(dedup_by_x.keys()), dtype=float)
            ys = np.asarray([dedup_by_x[val] for val in xs], dtype=float)

            output.append(
                RunSeries(
                    run_id=str(single_run.id),
                    run_name=str(single_run.name),
                    x=xs,
                    y=ys,
                )
            )

    time.sleep(14.0)

    return output


def _agent_count_from_label(agent_label: str) -> int:
    return int(agent_label.split()[0])


def _expand_metric_for_ppo_agents(metric_name: str, agent_count: int) -> list[str]:
    metric_tail = metric_name[len("Train/") :] if metric_name.startswith("Train/") else metric_name
    return [f"Train/Agent_{idx}_Proj/{metric_tail}" for idx in range(1, agent_count + 1)]


def _expand_metric_for_joint_optimize_agents(metric_name: str, agent_count: int) -> list[str]:
    if "Agent_1_Optimize" not in metric_name:
        return [metric_name]
    return [metric_name.replace("Agent_1_Optimize", f"Agent_{idx}_Optimize") for idx in range(1, agent_count + 1)]


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


def _final_series_value(series: RunSeries) -> float | None:
    if series.y.size == 0:
        return None
    return float(series.y[-1])


def _mean_std_text(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return f"{float(np.mean(arr)):.3f} ± {float(np.std(arr)):.3f}"


def _compute_alpha_summaries(
    worst_by_agent: dict[str, list[Any]],
    optimal_by_agent: dict[str, list[Any]],
    agent_labels: list[str],
) -> dict[str, dict[str, Any]]:
    per_agent_values_by_label: dict[str, list[list[float]]] = {}
    true_values_by_label: dict[str, float | None] = {}

    for agent_label in agent_labels:
        worst_groups = worst_by_agent.get(agent_label, [])
        optimal_groups = optimal_by_agent.get(agent_label, [])
        agent_count = _agent_count_from_label(agent_label)

        per_agent_values = [[] for _ in range(agent_count)]

        worst_by_run_id: dict[str, list[RunSeries]] = {}
        optimal_by_run_id: dict[str, list[RunSeries]] = {}
        for run_series_list in worst_groups:
            if run_series_list:
                worst_by_run_id[run_series_list[0].run_id] = run_series_list
        for run_series_list in optimal_groups:
            if run_series_list:
                optimal_by_run_id[run_series_list[0].run_id] = run_series_list

        for run_id in sorted(set(worst_by_run_id).intersection(optimal_by_run_id)):
            worst_run_series_list = worst_by_run_id[run_id]
            optimal_run_series_list = optimal_by_run_id[run_id]
            for agent_idx in range(min(agent_count, len(worst_run_series_list), len(optimal_run_series_list))):
                worst_series = worst_run_series_list[agent_idx]
                optimal_series = optimal_run_series_list[agent_idx]
                if not isinstance(worst_series, RunSeries) or not isinstance(optimal_series, RunSeries):
                    continue

                worst_final = _final_series_value(worst_series)
                optimal_final = _final_series_value(optimal_series)
                if worst_final is None or optimal_final is None or optimal_final == 0:
                    continue

                alpha_value = worst_final / optimal_final
                per_agent_values[agent_idx].append(alpha_value)

        # Compute per-agent means, then take the min across agents as the true alpha.
        per_agent_means = [np.mean(np.asarray(values, dtype=float)) if values else None for values in per_agent_values]
        non_none_means = [m for m in per_agent_means if m is not None]
        true_value = min(non_none_means) if non_none_means else None

        per_agent_values_by_label[agent_label] = per_agent_values
        true_values_by_label[agent_label] = true_value

    return {
        "per_agent": per_agent_values_by_label,
        "true": true_values_by_label,
    }


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
    api = wandb.Api(timeout=120)
    metric_keys = [args.metric] + [m for m in (args.metric_aliases or []) if m != args.metric]

    algos = ['CREPPO']
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

    _load_history_cache(args.data_cache, refresh=args.refresh_data_cache)

    path = f"{args.entity}/{args.project}"
    runs = api.runs(path=path, filters=run_filters, per_page=args.max_runs)

    # CREPPO
    creppo_filters = []
    for agent in agents:
        creppo_filters.append({
            "tags": {
                "$all": [agent, "marl_comparison", "creppo"],
                "$nin": ["OLD", "social law", "social law yielding"],
            }
        })

    creppo_social_law_filters = []
    for agent in agents:
        creppo_social_law_filters.append({
            "tags": {
                "$all": [agent, "marl_comparison", "creppo", "social law"],
                "$nin": ["OLD", "social law yielding"],
            }
        })

    creppo_social_law_yielding_filters = []
    for agent in agents:
        creppo_social_law_yielding_filters.append({
            "tags": {
                "$all": [agent, "marl_comparison", "creppo", "social law yielding"],
                "$nin": ["OLD", "social law"],
            }
        })

    creppo_generalized_filters = []
    for agent in agents:
        creppo_generalized_filters.append({
            "tags": {
                "$all": [agent, "social_law_generalization", "creppo"],
                "$nin": ["OLD", "social law", "social law yielding"],
            }
        })

    creppo_generalized_social_law_filters = []
    for agent in agents:
        creppo_generalized_social_law_filters.append({
            "tags": {
                "$all": [agent, "social_law_generalization", "creppo", "social law"],
                "$nin": ["OLD", "social law yielding"],
            }
        })

    creppo_generalized_social_law_yielding_filters = []
    for agent in agents:
        creppo_generalized_social_law_yielding_filters.append({
            "tags": {
                "$all": [agent, "social_law_generalization", "creppo", "social law yielding"],
                "$nin": ["OLD", "social law"],
            }
        })

    creppo_runs = []
    for creppo_filter in creppo_filters:
        creppo_runs.append(api.runs(path=path, filters=creppo_filter, per_page=args.max_runs))

    creppo_social_law_runs = []
    for creppo_social_law_filter in creppo_social_law_filters:
        creppo_social_law_runs.append(api.runs(path=path, filters=creppo_social_law_filter, per_page=args.max_runs))

    creppo_social_law_yielding_runs = []
    for creppo_social_law_yielding_filter in creppo_social_law_yielding_filters:
        creppo_social_law_yielding_runs.append(api.runs(path=path, filters=creppo_social_law_yielding_filter, per_page=args.max_runs))

    creppo_generalized_runs = []
    for creppo_generalized_filter in creppo_generalized_filters:
        creppo_generalized_runs.append(api.runs(path=path, filters=creppo_generalized_filter, per_page=args.max_runs))

    creppo_generalized_social_law_runs = []
    for creppo_generalized_social_law_filter in creppo_generalized_social_law_filters:
        creppo_generalized_social_law_runs.append(api.runs(path=path, filters=creppo_generalized_social_law_filter, per_page=args.max_runs))

    creppo_generalized_social_law_yielding_runs = []
    for creppo_generalized_social_law_yielding_filter in creppo_generalized_social_law_yielding_filters:
        creppo_generalized_social_law_yielding_runs.append(api.runs(path=path, filters=creppo_generalized_social_law_yielding_filter, per_page=args.max_runs))


    for i in range(len(agents)):
        print(f"CREPPO {agents[i]} runs: {len(creppo_runs[i])}")
        print(f"CREPPO (Social Law) {agents[i]} runs: {len(creppo_social_law_runs[i])}")
        print(f"CREPPO (Social Law Yielding) {agents[i]} runs: {len(creppo_social_law_yielding_runs[i])}")
        print(f"CREPPO (Generalized) {agents[i]} runs: {len(creppo_generalized_runs[i])}")
        print(f"CREPPO (Generalized, Social Law) {agents[i]} runs: {len(creppo_generalized_social_law_runs[i])}")
        print(f"CREPPO (Generalized, Social Law Yielding) {agents[i]} runs: {len(creppo_generalized_social_law_yielding_runs[i])}")

    def collect_metric_series(
        run_collections: list[Any],
        agent_labels: list[str],
        returns_metric: str,
        collisions_metric: str | None,
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
                if "Agent_1_Optimize" in returns_metric:
                    returns_metric_keys = _expand_metric_for_joint_optimize_agents(returns_metric, agent_count)
                else:
                    returns_metric_keys = _expand_metric_for_ppo_agents(returns_metric, agent_count)
                if collisions_metric is None:
                    collisions_metric_keys = []
                elif "Agent_1_Optimize" in collisions_metric:
                    collisions_metric_keys = _expand_metric_for_joint_optimize_agents(collisions_metric, agent_count)
                else:
                    collisions_metric_keys = _expand_metric_for_ppo_agents(collisions_metric, agent_count)
                fetch_mode = "per_metric"
                print(f"    CREPPO metric keys (returns): {returns_metric_keys}")
                print(f"    CREPPO metric keys (collisions): {collisions_metric_keys}")
            else:
                returns_metric_keys = [returns_metric]
                collisions_metric_keys = [] if collisions_metric is None else [collisions_metric]
                fetch_mode = "first_available"

            print(f"    Fetch mode: {fetch_mode}")

            if ppo_per_agent_metrics:
                return_series_batch = []
                collision_series_batch = []
                for run_index, single_run in enumerate(runs, start=1):
                    print(f"    [{agent_label}] run {run_index}/{len(runs)}: {single_run.name}")
                    single_return_series = fetch_run_series(
                        run=single_run,
                        metric_keys=returns_metric_keys,
                        x_key=args.x_key,
                        mode=fetch_mode,
                    )
                    if single_return_series:
                        return_series_batch.append(single_return_series)

                    if collisions_metric_keys:
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
                collision_series_batch = (
                    fetch_run_series(
                        run=runs,
                        metric_keys=collisions_metric_keys,
                        x_key=args.x_key,
                        mode=fetch_mode,
                    )
                    if collisions_metric_keys
                    else []
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

    creppo_returns_by_agent, creppo_collisions_by_agent = collect_metric_series(
        run_collections=creppo_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    print_collection_summary("CREPPO", creppo_returns_by_agent, creppo_collisions_by_agent)
    creppo_social_law_returns_by_agent, creppo_social_law_collisions_by_agent = collect_metric_series(
        run_collections=creppo_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    print_collection_summary("CREPPO (Social Law)", creppo_social_law_returns_by_agent, creppo_social_law_collisions_by_agent)
    creppo_social_law_yielding_returns_by_agent, creppo_social_law_yielding_collisions_by_agent = collect_metric_series(
        run_collections=creppo_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    print_collection_summary("CREPPO (Social Law Yielding)", creppo_social_law_yielding_returns_by_agent, creppo_social_law_yielding_collisions_by_agent)
    creppo_generalized_returns_by_agent, creppo_generalized_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    print_collection_summary("CREPPO (Generalized)", creppo_generalized_returns_by_agent, creppo_generalized_collisions_by_agent)
    creppo_generalized_social_law_returns_by_agent, creppo_generalized_social_law_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    print_collection_summary("CREPPO (Generalized, Social Law)", creppo_generalized_social_law_returns_by_agent, creppo_generalized_social_law_collisions_by_agent)
    creppo_generalized_social_law_yielding_returns_by_agent, creppo_generalized_social_law_yielding_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/returned_episode_returns",
        collisions_metric="Train/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    print_collection_summary("CREPPO (Generalized, Social Law Yielding)", creppo_generalized_social_law_yielding_returns_by_agent, creppo_generalized_social_law_yielding_collisions_by_agent)

    # Create separate plots per agent-count group and separate returns/collisions plots.
    # Collect worst-case and best-case metrics for CREPPO variants
    creppo_worst_case_by_agent, creppo_worst_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    creppo_best_case_by_agent, creppo_best_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )

    creppo_sl_worst_case_by_agent, creppo_sl_worst_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    creppo_sl_best_case_by_agent, creppo_sl_best_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )

    creppo_sly_worst_case_by_agent, creppo_sly_worst_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    creppo_sly_best_case_by_agent, creppo_sly_best_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )

    creppo_gen_worst_case_by_agent, creppo_gen_worst_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    creppo_gen_best_case_by_agent, creppo_gen_best_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )

    creppo_gen_sl_worst_case_by_agent, creppo_gen_sl_worst_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    creppo_gen_sl_best_case_by_agent, creppo_gen_sl_best_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_social_law_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )

    creppo_gen_sly_worst_case_by_agent, creppo_gen_sly_worst_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )
    creppo_gen_sly_best_case_by_agent, creppo_gen_sly_best_case_collisions_by_agent = collect_metric_series(
        run_collections=creppo_generalized_social_law_yielding_runs,
        agent_labels=agents,
        returns_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
        collisions_metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
        ppo_per_agent_metrics=True,
        max_workers=args.max_workers,
    )

    alpha_variant_specs: list[tuple[str, list[Any]]] = [
        ("CREPPO", creppo_runs),
        ("CREPPO (Social Law)", creppo_social_law_runs),
        ("CREPPO (Social Law Yielding)", creppo_social_law_yielding_runs),
        ("CREPPO (Generalized)", creppo_generalized_runs),
        ("CREPPO (Generalized, Social Law)", creppo_generalized_social_law_runs),
        ("CREPPO (Generalized, Social Law Yielding)", creppo_generalized_social_law_yielding_runs),
    ]

    alpha_eval_worst_by_variant: dict[str, dict[str, list[Any]]] = {}
    alpha_eval_optimal_by_variant: dict[str, dict[str, list[Any]]] = {}
    alpha_eval_best_by_variant: dict[str, dict[str, list[Any]]] = {}
    alpha_eval_single_return_by_variant: dict[str, dict[str, list[Any]]] = {}
    alpha_eval_single_collisions_by_variant: dict[str, dict[str, list[Any]]] = {}
    alpha_eval_worst_metric = "Eval/Joint/Agent_1_Optimize/WorstCase/Return"
    alpha_eval_optimal_metric = "Eval/Joint/Agent_1_Optimize/OptimalReturn"
    alpha_eval_best_metric = "Eval/Joint/Agent_1_Optimize/BestCase/Return"
    alpha_eval_single_return_metric = "Eval/Single_Agent_Proj_Joint/Return"
    alpha_eval_single_collisions_metric = "Eval/Single_Agent_Proj_Joint/Collisions"
    for variant_name, variant_runs in alpha_variant_specs:
        alpha_eval_worst_by_variant[variant_name], _ = collect_metric_series(
            run_collections=variant_runs,
            agent_labels=agents,
            returns_metric=alpha_eval_worst_metric,
            collisions_metric=None,
            ppo_per_agent_metrics=True,
            max_workers=args.max_workers,
        )
        alpha_eval_optimal_by_variant[variant_name], _ = collect_metric_series(
            run_collections=variant_runs,
            agent_labels=agents,
            returns_metric=alpha_eval_optimal_metric,
            collisions_metric=None,
            ppo_per_agent_metrics=True,
            max_workers=args.max_workers,
        )
        alpha_eval_best_by_variant[variant_name], _ = collect_metric_series(
            run_collections=variant_runs,
            agent_labels=agents,
            returns_metric=alpha_eval_best_metric,
            collisions_metric=None,
            ppo_per_agent_metrics=True,
            max_workers=args.max_workers,
        )
        alpha_eval_single_return_by_variant[variant_name], alpha_eval_single_collisions_by_variant[variant_name] = collect_metric_series(
            run_collections=variant_runs,
            agent_labels=agents,
            returns_metric=alpha_eval_single_return_metric,
            collisions_metric=alpha_eval_single_collisions_metric,
            ppo_per_agent_metrics=False,
            max_workers=args.max_workers,
        )

    alpha_summaries_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant_name, _ in alpha_variant_specs:
        alpha_summaries_by_variant[variant_name] = _compute_alpha_summaries(
            worst_by_agent=alpha_eval_worst_by_variant[variant_name],
            optimal_by_agent=alpha_eval_optimal_by_variant[variant_name],
            agent_labels=agents,
        )

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

        # Build grouped collisions series for this agent_label only.
        grouped_collisions: dict[str, list[RunSeries]] = {}

        # CREPPO per-agent groups for returns
        creppo_groups = creppo_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent: list[RunSeries] = []
            for run_series_list in creppo_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"CREPPO | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_sl_groups = creppo_social_law_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_sl_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"CREPPO (Social Law) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_sly_groups = creppo_social_law_yielding_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_sly_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"CREPPO (Social Law Yielding) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_gen_groups = creppo_generalized_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_gen_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"CREPPO (Generalized) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_gen_sl_groups = creppo_generalized_social_law_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_gen_sl_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"CREPPO (Generalized, Social Law) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_gen_sly_groups = creppo_generalized_social_law_yielding_returns_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_gen_sly_groups:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_returns[f"CREPPO (Generalized, Social Law Yielding) | {agent_label} | Agent {ai+1}"] = series_for_agent

        # CREPPO per-agent groups for collisions
        creppo_groups_c = creppo_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"CREPPO | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_sl_groups_c = creppo_social_law_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_sl_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"CREPPO (Social Law) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_sly_groups_c = creppo_social_law_yielding_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_sly_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"CREPPO (Social Law Yielding) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_gen_groups_c = creppo_generalized_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_gen_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"CREPPO (Generalized) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_gen_sl_groups_c = creppo_generalized_social_law_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_gen_sl_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"CREPPO (Generalized, Social Law) | {agent_label} | Agent {ai+1}"] = series_for_agent

        creppo_gen_sly_groups_c = creppo_generalized_social_law_yielding_collisions_by_agent[agent_label]
        for ai in range(agent_count):
            series_for_agent = []
            for run_series_list in creppo_gen_sly_groups_c:
                if isinstance(run_series_list, list) and len(run_series_list) > ai:
                    series_for_agent.append(run_series_list[ai])
            grouped_collisions[f"CREPPO (Generalized, Social Law Yielding) | {agent_label} | Agent {ai+1}"] = series_for_agent

        # Build grouped worst-case series for this agent_label only.
        grouped_worst_case: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": creppo_worst_case_by_agent[agent_label],
            f"CREPPO (Social Law) | {agent_label}": creppo_sl_worst_case_by_agent[agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": creppo_sly_worst_case_by_agent[agent_label],
            f"CREPPO (Generalized) | {agent_label}": creppo_gen_worst_case_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": creppo_gen_sl_worst_case_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": creppo_gen_sly_worst_case_by_agent[agent_label],
        }

        grouped_worst_case_collisions: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": creppo_worst_case_collisions_by_agent[agent_label],
            f"CREPPO (Social Law) | {agent_label}": creppo_sl_worst_case_collisions_by_agent[agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": creppo_sly_worst_case_collisions_by_agent[agent_label],
            f"CREPPO (Generalized) | {agent_label}": creppo_gen_worst_case_collisions_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": creppo_gen_sl_worst_case_collisions_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": creppo_gen_sly_worst_case_collisions_by_agent[agent_label],
        }

        # Build grouped best-case series for this agent_label only.
        grouped_best_case: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": creppo_best_case_by_agent[agent_label],
            f"CREPPO (Social Law) | {agent_label}": creppo_sl_best_case_by_agent[agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": creppo_sly_best_case_by_agent[agent_label],
            f"CREPPO (Generalized) | {agent_label}": creppo_gen_best_case_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": creppo_gen_sl_best_case_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": creppo_gen_sly_best_case_by_agent[agent_label],
        }

        grouped_best_case_collisions: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": creppo_best_case_collisions_by_agent[agent_label],
            f"CREPPO (Social Law) | {agent_label}": creppo_sl_best_case_collisions_by_agent[agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": creppo_sly_best_case_collisions_by_agent[agent_label],
            f"CREPPO (Generalized) | {agent_label}": creppo_gen_best_case_collisions_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": creppo_gen_sl_best_case_collisions_by_agent[agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": creppo_gen_sly_best_case_collisions_by_agent[agent_label],
        }

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

        # Make and save worst-case figure for this agent group.
        worst_case_title = f"{args.title} | {agent_label} | Worst Case"
        worst_case_fig = make_variance_figure(
            grouped_series=grouped_worst_case,
            metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns",
            x_key=args.x_key,
            title=worst_case_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Worst Case Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((worst_case_title, worst_case_fig))
        print(f"Prepared worst-case plot: {worst_case_title}")
        if args.png_output:
            out_png_wc = _make_output_path(args.png_output, agent_label, "worst_case")
            worst_case_fig.write_image(out_png_wc)
            print(f"Saved PNG worst-case plot to: {out_png_wc}")

        worst_case_collisions_title = f"{args.title} | {agent_label} | Worst Case Collisions"
        worst_case_collisions_fig = make_variance_figure(
            grouped_series=grouped_worst_case_collisions,
            metric="Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions",
            x_key=args.x_key,
            title=worst_case_collisions_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Worst Case Collisions",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((worst_case_collisions_title, worst_case_collisions_fig))
        print(f"Prepared worst-case collisions plot: {worst_case_collisions_title}")
        if args.png_output:
            out_png_wcc = _make_output_path(args.png_output, agent_label, "worst_case_collisions")
            worst_case_collisions_fig.write_image(out_png_wcc)
            print(f"Saved PNG worst-case collisions plot to: {out_png_wcc}")

        # Make and save best-case figure for this agent group.
        best_case_title = f"{args.title} | {agent_label} | Best Case"
        best_case_fig = make_variance_figure(
            grouped_series=grouped_best_case,
            metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns",
            x_key=args.x_key,
            title=best_case_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Best Case Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((best_case_title, best_case_fig))
        print(f"Prepared best-case plot: {best_case_title}")
        if args.png_output:
            out_png_bc = _make_output_path(args.png_output, agent_label, "best_case")
            best_case_fig.write_image(out_png_bc)
            print(f"Saved PNG best-case plot to: {out_png_bc}")

        best_case_collisions_title = f"{args.title} | {agent_label} | Best Case Collisions"
        best_case_collisions_fig = make_variance_figure(
            grouped_series=grouped_best_case_collisions,
            metric="Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions",
            x_key=args.x_key,
            title=best_case_collisions_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Best Case Collisions",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((best_case_collisions_title, best_case_collisions_fig))
        print(f"Prepared best-case collisions plot: {best_case_collisions_title}")
        if args.png_output:
            out_png_bcc = _make_output_path(args.png_output, agent_label, "best_case_collisions")
            best_case_collisions_fig.write_image(out_png_bcc)
            print(f"Saved PNG best-case collisions plot to: {out_png_bcc}")

        # Build grouped eval worst/optimal series for this agent_label only.
        grouped_eval_worst: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": alpha_eval_worst_by_variant["CREPPO"][agent_label],
            f"CREPPO (Social Law) | {agent_label}": alpha_eval_worst_by_variant["CREPPO (Social Law)"][agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": alpha_eval_worst_by_variant["CREPPO (Social Law Yielding)"][agent_label],
            f"CREPPO (Generalized) | {agent_label}": alpha_eval_worst_by_variant["CREPPO (Generalized)"][agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": alpha_eval_worst_by_variant["CREPPO (Generalized, Social Law)"][agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": alpha_eval_worst_by_variant["CREPPO (Generalized, Social Law Yielding)"][agent_label],
        }

        grouped_eval_optimal: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": alpha_eval_optimal_by_variant["CREPPO"][agent_label],
            f"CREPPO (Social Law) | {agent_label}": alpha_eval_optimal_by_variant["CREPPO (Social Law)"][agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": alpha_eval_optimal_by_variant["CREPPO (Social Law Yielding)"][agent_label],
            f"CREPPO (Generalized) | {agent_label}": alpha_eval_optimal_by_variant["CREPPO (Generalized)"][agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": alpha_eval_optimal_by_variant["CREPPO (Generalized, Social Law)"][agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": alpha_eval_optimal_by_variant["CREPPO (Generalized, Social Law Yielding)"][agent_label],
        }

        grouped_eval_best: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": alpha_eval_best_by_variant["CREPPO"][agent_label],
            f"CREPPO (Social Law) | {agent_label}": alpha_eval_best_by_variant["CREPPO (Social Law)"][agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": alpha_eval_best_by_variant["CREPPO (Social Law Yielding)"][agent_label],
            f"CREPPO (Generalized) | {agent_label}": alpha_eval_best_by_variant["CREPPO (Generalized)"][agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": alpha_eval_best_by_variant["CREPPO (Generalized, Social Law)"][agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": alpha_eval_best_by_variant["CREPPO (Generalized, Social Law Yielding)"][agent_label],
        }

        grouped_eval_single_return: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": alpha_eval_single_return_by_variant["CREPPO"][agent_label],
            f"CREPPO (Social Law) | {agent_label}": alpha_eval_single_return_by_variant["CREPPO (Social Law)"][agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": alpha_eval_single_return_by_variant["CREPPO (Social Law Yielding)"][agent_label],
            f"CREPPO (Generalized) | {agent_label}": alpha_eval_single_return_by_variant["CREPPO (Generalized)"][agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": alpha_eval_single_return_by_variant["CREPPO (Generalized, Social Law)"][agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": alpha_eval_single_return_by_variant["CREPPO (Generalized, Social Law Yielding)"][agent_label],
        }

        grouped_eval_single_collisions: dict[str, list[RunSeries]] = {
            f"CREPPO | {agent_label}": alpha_eval_single_collisions_by_variant["CREPPO"][agent_label],
            f"CREPPO (Social Law) | {agent_label}": alpha_eval_single_collisions_by_variant["CREPPO (Social Law)"][agent_label],
            f"CREPPO (Social Law Yielding) | {agent_label}": alpha_eval_single_collisions_by_variant["CREPPO (Social Law Yielding)"][agent_label],
            f"CREPPO (Generalized) | {agent_label}": alpha_eval_single_collisions_by_variant["CREPPO (Generalized)"][agent_label],
            f"CREPPO (Generalized, Social Law) | {agent_label}": alpha_eval_single_collisions_by_variant["CREPPO (Generalized, Social Law)"][agent_label],
            f"CREPPO (Generalized, Social Law Yielding) | {agent_label}": alpha_eval_single_collisions_by_variant["CREPPO (Generalized, Social Law Yielding)"][agent_label],
        }

        eval_worst_title = f"{args.title} | {agent_label} | Eval Worst Return"
        eval_worst_fig = make_variance_figure(
            grouped_series=grouped_eval_worst,
            metric=alpha_eval_worst_metric,
            x_key=args.x_key,
            title=eval_worst_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Eval Worst Case Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((eval_worst_title, eval_worst_fig))
        print(f"Prepared eval worst-return plot: {eval_worst_title}")
        if args.png_output:
            out_png_ew = _make_output_path(args.png_output, agent_label, "eval_worst")
            eval_worst_fig.write_image(out_png_ew)
            print(f"Saved PNG eval worst-return plot to: {out_png_ew}")

        eval_optimal_title = f"{args.title} | {agent_label} | Eval Optimal Return"
        eval_optimal_fig = make_variance_figure(
            grouped_series=grouped_eval_optimal,
            metric=alpha_eval_optimal_metric,
            x_key=args.x_key,
            title=eval_optimal_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Eval Optimal Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((eval_optimal_title, eval_optimal_fig))
        print(f"Prepared eval optimal-return plot: {eval_optimal_title}")
        if args.png_output:
            out_png_eo = _make_output_path(args.png_output, agent_label, "eval_optimal")
            eval_optimal_fig.write_image(out_png_eo)
            print(f"Saved PNG eval optimal-return plot to: {out_png_eo}")

        eval_best_title = f"{args.title} | {agent_label} | Eval Best Case Return"
        eval_best_fig = make_variance_figure(
            grouped_series=grouped_eval_best,
            metric=alpha_eval_best_metric,
            x_key=args.x_key,
            title=eval_best_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Eval Best Case Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((eval_best_title, eval_best_fig))
        print(f"Prepared eval best-case plot: {eval_best_title}")
        if args.png_output:
            out_png_eb = _make_output_path(args.png_output, agent_label, "eval_best")
            eval_best_fig.write_image(out_png_eb)
            print(f"Saved PNG eval best-case plot to: {out_png_eb}")

        eval_single_return_title = f"{args.title} | {agent_label} | Eval Single-Agent Return"
        eval_single_return_fig = make_variance_figure(
            grouped_series=grouped_eval_single_return,
            metric=alpha_eval_single_return_metric,
            x_key=args.x_key,
            title=eval_single_return_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Eval Single-Agent Returns",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((eval_single_return_title, eval_single_return_fig))
        print(f"Prepared eval single-agent return plot: {eval_single_return_title}")
        if args.png_output:
            out_png_esr = _make_output_path(args.png_output, agent_label, "eval_single_return")
            eval_single_return_fig.write_image(out_png_esr)
            print(f"Saved PNG eval single-agent return plot to: {out_png_esr}")

        eval_single_collisions_title = f"{args.title} | {agent_label} | Eval Single-Agent Collisions"
        eval_single_collisions_fig = make_variance_figure(
            grouped_series=grouped_eval_single_collisions,
            metric=alpha_eval_single_collisions_metric,
            x_key=args.x_key,
            title=eval_single_collisions_title,
            min_runs_per_point=args.min_runs_per_point,
            show_individual_runs=args.show_individual_runs,
            y_axis_title="Mean Eval Single-Agent Collisions",
            agent_count_label=agent_label,
            show_plot_title=args.show_plot_title,
        )
        all_figures.append((eval_single_collisions_title, eval_single_collisions_fig))
        print(f"Prepared eval single-agent collisions plot: {eval_single_collisions_title}")
        if args.png_output:
            out_png_esc = _make_output_path(args.png_output, agent_label, "eval_single_collisions")
            eval_single_collisions_fig.write_image(out_png_esc)
            print(f"Saved PNG eval single-agent collisions plot to: {out_png_esc}")

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
        "    .alpha-section { margin-top: 36px; }",
        "    .alpha-section h2 { margin-bottom: 8px; }",
        "    .alpha-note { margin: 0 0 16px 0; color: #444; }",
        "    .alpha-table { border-collapse: collapse; margin: 18px 0 28px 0; width: 100%; }",
        "    .alpha-table th, .alpha-table td { border: 1px solid #d0d0d0; padding: 8px 10px; vertical-align: top; text-align: left; }",
        "    .alpha-table th { background: #f6f6f6; }",
        "    .alpha-agent-title { margin: 18px 0 8px 0; }",
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

    html_parts.append("  <section class=\"alpha-section\">")
    html_parts.append("    <h2>Alpha Summary</h2>")
    html_parts.append(
        "    <p class=\"alpha-note\">"
        "Alpha is computed from the final logged evaluation point only, using "
        "Eval/Joint/Agent_#_Optimize/WorstCase/Return divided by "
        "Eval/Joint/Agent_#_Optimize/OptimalReturn. Each agent cell reports the run-level "
        "alpha mean ± std. The true alpha column is the minimum of the per-agent mean alpha values."
        "</p>"
    )
    for agent_label in agents:
        agent_count = _agent_count_from_label(agent_label)
        html_parts.append(f"    <h3 class=\"alpha-agent-title\">{escape(agent_label)}</h3>")
        html_parts.append("    <table class=\"alpha-table\">")
        header_cells = ["Variant"] + [f"Agent {idx + 1}" for idx in range(agent_count)] + ["True alpha"]
        html_parts.append("      <thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in header_cells) + "</tr></thead>")
        html_parts.append("      <tbody>")
        for variant_name, _ in alpha_variant_specs:
            summary = alpha_summaries_by_variant[variant_name]
            per_agent_values = summary["per_agent"][agent_label]
            true_value = summary["true"][agent_label]
            row_cells = [escape(variant_name)]
            for agent_idx, values in enumerate(per_agent_values):
                row_cells.append(_mean_std_text(values))
            true_alpha = f"{float(true_value):.3f}" if true_value is not None else "n/a"
            row_cells.append(true_alpha)
            html_parts.append("        <tr>" + "".join(f"<td>{cell}</td>" for cell in row_cells) + "</tr>")
        html_parts.append("      </tbody>")
        html_parts.append("    </table>")
    html_parts.append("  </section>")

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

    _save_history_cache(args.data_cache)


if __name__ == "__main__":
    main()
