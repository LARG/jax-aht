"""Create variance plots from Weights & Biases runs using Plotly.

Optimized version of the original script:
  - lists W&B runs once, then groups locally by tags;
  - supports exact narrow scans that pull all points for only the needed metrics;
  - avoids full-history scans over every logged metric unless explicitly requested;
  - reuses the extracted series for returns, collisions, worst/best case, and alpha.

Example:
    python optimized_neurips_plotting.py \
        --entity social-laws-project \
        --project NEURIPS-2026-UPDATED \
        --metric Train/returned_episode_returns \
        --output results/neurips_alpha_variance.html
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html import escape
import json
import os
import pickle
import random
import time
from collections import defaultdict
from typing import Any

import numpy as np
import plotly.graph_objects as go
import wandb

DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "social-laws-project")
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "NEURIPS-2026-UPDATED")

AGENTS = ["2 agents"]#, "3 agents", "4 agents", "5 agents", "6 agents", "10 agents"]

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "CREPPO": {
        "all": ["marl_comparison", "creppo"],
        "nin": ["OLD", "social law", "social law yielding"],
    },
    "CREPPO (Social Law)": {
        "all": ["marl_comparison", "creppo", "social law"],
        "nin": ["OLD", "social law yielding"],
    },
    "CREPPO (Social Law Yielding)": {
        "all": ["marl_comparison", "creppo", "social law yielding"],
        "nin": ["OLD", "social law"],
    },
    "CREPPO (Generalized)": {
        "all": ["social_law_generalization", "creppo"],
        "nin": ["OLD", "social law", "social law yielding"],
    },
    "CREPPO (Generalized, Social Law)": {
        "all": ["social_law_generalization", "creppo", "social law"],
        "nin": ["OLD", "social law yielding"],
    },
    "CREPPO (Generalized, Social Law Yielding)": {
        "all": ["social_law_generalization", "creppo", "social law yielding"],
        "nin": ["OLD", "social law"],
    },
}

TRAIN_RETURN = "Train/returned_episode_returns"
TRAIN_COLLISIONS = "Train/returned_episode_collisions"
WORST_RETURN = "Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_returns"
WORST_COLLISIONS = "Train/Joint/Agent_1_Optimize/WorstCase/returned_episode_collisions"
BEST_RETURN = "Train/Joint/Agent_1_Optimize/BestCase/returned_episode_returns"
BEST_COLLISIONS = "Train/Joint/Agent_1_Optimize/BestCase/returned_episode_collisions"
ALPHA_WORST_EVAL = "Eval/Joint/Agent_1_Optimize/WorstCase/Return"
ALPHA_OPTIMAL_EVAL = "Eval/Joint/Agent_1_Optimize/OptimalReturn"


@dataclass(frozen=True)
class RunSeries:
    run_id: str
    run_name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class RunInfo:
    run: Any
    run_id: str
    run_name: str
    tags: set[str]
    agent_label: str
    variant_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull W&B runs and create Plotly variance plots grouped by tag.")
    parser.add_argument("--entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--metric", type=str, required=True, help="Primary metric label used in hover text/title compatibility.")
    parser.add_argument("--x-key", type=str, default="train_step")
    parser.add_argument("--run-filters", type=str, default="{}")
    parser.add_argument("--max-runs", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=1, help="Number of concurrent run fetches inside each batch. Keep 1 for W&B reliability; try 2 only if stable.")
    parser.add_argument("--history-page-size", type=int, default=10000)
    parser.add_argument(
        "--history-mode",
        choices=["sampled", "targeted", "narrow_full", "single_full"],
        default="targeted",
        help=(
            "sampled uses run.history(samples=..., keys=...) and is fastest but approximate; "
            "targeted/narrow_full use scan_history(keys=[x, metric]) for exact all-points narrow scans; "
            "single_full uses full scan_history() over every metric and is the slow fallback."
        ),
    )
    parser.add_argument("--history-samples", type=int, default=1000, help="Samples per run.history call in sampled mode.")
    parser.add_argument("--history-key-batch-size", type=int, default=1, help="Metric keys per narrow W&B request. Use 1 if sparse logging causes empty plots; try 5-20 for sampled mode if stable.")
    parser.add_argument("--wandb-call-sleep", type=float, default=1.1, help="Sleep before each W&B history request; W&B recommends at least 1 second for large public API pulls.")
    parser.add_argument("--wandb-jitter", type=float, default=0.5, help="Random extra sleep before each W&B request.")
    parser.add_argument("--wandb-max-retries", type=int, default=3, help="Script-level retries for W&B request failures.")
    parser.add_argument("--wandb-backoff-base", type=float, default=2.0, help="Exponential backoff base in seconds.")
    parser.add_argument("--wandb-backoff-max", type=float, default=60.0, help="Maximum exponential backoff sleep in seconds.")
    parser.add_argument("--request-batch-size", type=int, default=10, help="Number of runs to process before pausing. This avoids one large burst of W&B requests.")
    parser.add_argument("--batch-sleep", type=float, default=5.0, help="Sleep between run batches.")
    parser.add_argument("--skip-failed-runs", action="store_true", default=True, help="Skip runs that still fail after retries. Enabled by default.")
    parser.add_argument("--fail-on-failed-runs", action="store_true", help="Stop immediately instead of skipping failed runs.")
    parser.add_argument("--data-cache", type=str, default="plotting_data/neurips_alpha_updated_cache.pkl", help="Pickle path for processed data. Existing cache skips W&B requests unless --refresh-data-cache is set.")
    parser.add_argument("--refresh-data-cache", action="store_true", help="Refetch data and overwrite --data-cache.")
    parser.add_argument(
        "--minimal-history-keys",
        action="store_true",
        help=(
            "Compatibility alias for --history-mode targeted."
        ),
    )
    parser.add_argument("--min-runs-per-point", type=int, default=1)
    parser.add_argument("--show-individual-runs", action="store_true")
    parser.add_argument("--title", type=str, default="W&B Variance Plot")
    parser.add_argument("--show-plot-title", action="store_true")
    parser.add_argument("--output", type=str, default="neurips_alpha_variance.html")
    parser.add_argument("--png-output", type=str, default=None)
    return parser.parse_args()


def _agent_count_from_label(agent_label: str) -> int:
    return int(agent_label.split()[0])


def _expand_metric_for_ppo_agents(metric_name: str, agent_count: int) -> list[str]:
    metric_tail = metric_name[len("Train/") :] if metric_name.startswith("Train/") else metric_name
    return [f"Train/Agent_{idx}_Proj/{metric_tail}" for idx in range(1, agent_count + 1)]


def _expand_metric_for_joint_optimize_agents(metric_name: str, agent_count: int) -> list[str]:
    if "Agent_1_Optimize" not in metric_name:
        return [metric_name]
    return [metric_name.replace("Agent_1_Optimize", f"Agent_{idx}_Optimize") for idx in range(1, agent_count + 1)]


def _metric_keys_for_agent(agent_label: str) -> list[str]:
    n = _agent_count_from_label(agent_label)
    keys: list[str] = []
    for metric in (TRAIN_RETURN, TRAIN_COLLISIONS):
        keys.extend(_expand_metric_for_ppo_agents(metric, n))
    for metric in (WORST_RETURN, WORST_COLLISIONS, BEST_RETURN, BEST_COLLISIONS, ALPHA_WORST_EVAL, ALPHA_OPTIMAL_EVAL):
        keys.extend(_expand_metric_for_joint_optimize_agents(metric, n))
    return list(dict.fromkeys(keys))


def _tags_match(tags: set[str], required: list[str], excluded: list[str]) -> bool:
    return all(tag in tags for tag in required) and not any(tag in tags for tag in excluded)


def _classify_run(run: Any) -> RunInfo | None:
    tags = {str(tag) for tag in (getattr(run, "tags", []) or [])}
    agent_matches = [agent for agent in AGENTS if agent in tags]
    if len(agent_matches) != 1:
        return None
    agent_label = agent_matches[0]

    for variant_name, spec in VARIANT_SPECS.items():
        if _tags_match(tags, required=spec["all"], excluded=spec["nin"]):
            return RunInfo(
                run=run,
                run_id=str(run.id),
                run_name=str(getattr(run, "name", run.id)),
                tags=tags,
                agent_label=agent_label,
                variant_name=variant_name,
            )
    return None


def list_and_group_runs(api: wandb.Api, path: str, run_filters: dict[str, Any], max_runs: int) -> dict[str, dict[str, list[RunInfo]]]:
    """List runs once, then group locally by variant and agent-count tags."""
    base_filter = {"tags": {"$all": ["creppo"], "$nin": ["OLD"]}}
    if run_filters:
        # Keep user-provided filters, but avoid overwriting their fields silently.
        # If the user passes their own tags filter, trust it.
        base_filter = {**base_filter, **run_filters}

    runs = list(api.runs(path=path, filters=base_filter, per_page=max_runs))
    grouped: dict[str, dict[str, list[RunInfo]]] = {
        variant: {agent: [] for agent in AGENTS} for variant in VARIANT_SPECS
    }

    skipped = 0
    for run in runs:
        info = _classify_run(run)
        if info is None:
            skipped += 1
            continue
        grouped[info.variant_name][info.agent_label].append(info)

    print(f"Fetched {len(runs)} candidate runs with one W&B query; grouped {len(runs) - skipped}; skipped {skipped}.")
    for agent in AGENTS:
        for variant in VARIANT_SPECS:
            print(f"{variant} {agent} runs: {len(grouped[variant][agent])}")
    return grouped


def _series_from_rows(rows: list[dict[str, Any]], run_id: str, run_name: str, metric_key: str, x_key: str) -> RunSeries | None:
    dedup: dict[float, float] = {}
    for row in rows:
        xv = row.get(x_key)
        yv = row.get(metric_key)
        if xv is None or yv is None:
            continue
        try:
            xf = float(xv)
            yf = float(yv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf) and np.isfinite(yf):
            dedup[xf] = yf  # keep last value for duplicate x within this run

    if not dedup:
        return None
    xs = np.asarray(sorted(dedup), dtype=float)
    ys = np.asarray([dedup[x] for x in xs], dtype=float)
    return RunSeries(run_id=run_id, run_name=f"{run_name} | {metric_key}", x=xs, y=ys)


def _sleep_with_jitter(base_sleep: float, jitter: float = 0.0) -> None:
    delay = max(0.0, base_sleep) + (random.uniform(0.0, max(0.0, jitter)) if jitter else 0.0)
    if delay > 0:
        time.sleep(delay)


def _is_retryable_wandb_error(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return any(token in text for token in ("429", "rate limit", "httperror", "timeout", "connection", "temporarily", "too many requests"))


def _call_wandb_with_backoff(
    func: Any,
    *,
    label: str,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    call_sleep: float,
    jitter: float,
) -> Any:
    """Call one W&B request with throttle + bounded exponential backoff.

    W&B can otherwise enter long retry loops when rate-limited or when a large
    history request fails. This wrapper spaces requests out and retries only a
    bounded number of times.
    """
    attempts = max(1, max_retries + 1)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        _sleep_with_jitter(call_sleep, jitter)
        try:
            return func()
        except BaseException as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_retryable_wandb_error(exc):
                break
            sleep_for = min(float(backoff_max), float(backoff_base) * (2 ** attempt))
            print(f"W&B request failed for {label} on attempt {attempt + 1}/{attempts}: {exc}. Sleeping {sleep_for:.1f}s.")
            _sleep_with_jitter(sleep_for, jitter)
    assert last_exc is not None
    raise last_exc


def _rows_from_history_dataframe(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    try:
        return df.replace({np.nan: None}).to_dict("records")
    except Exception:
        return []


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fetch_rows_for_keys(
    run: Any,
    keys: list[str],
    *,
    x_key: str,
    page_size: int,
    history_mode: str,
    history_samples: int,
    label: str,
    wandb_max_retries: int,
    wandb_backoff_base: float,
    wandb_backoff_max: float,
    wandb_call_sleep: float,
    wandb_jitter: float,
) -> list[dict[str, Any]]:
    keys = list(dict.fromkeys([x_key] + [key for key in keys if key != x_key]))

    if history_mode == "sampled":
        samples = max(1, int(history_samples))
        def _request() -> list[dict[str, Any]]:
            df = run.history(samples=samples, keys=keys, x_axis=x_key)
            return _rows_from_history_dataframe(df)
        return _call_wandb_with_backoff(
            _request,
            label=label,
            max_retries=wandb_max_retries,
            backoff_base=wandb_backoff_base,
            backoff_max=wandb_backoff_max,
            call_sleep=wandb_call_sleep,
            jitter=wandb_jitter,
        )

    if history_mode in {"targeted", "narrow_full"}:
        def _request() -> list[dict[str, Any]]:
            return list(run.scan_history(keys=keys, page_size=page_size))
        return _call_wandb_with_backoff(
            _request,
            label=label,
            max_retries=wandb_max_retries,
            backoff_base=wandb_backoff_base,
            backoff_max=wandb_backoff_max,
            call_sleep=wandb_call_sleep,
            jitter=wandb_jitter,
        )

    raise ValueError(f"_fetch_rows_for_keys does not support history_mode={history_mode!r}")


def _scan_run_once(
    info: RunInfo,
    x_key: str,
    page_size: int,
    minimal_history_keys: bool,
    history_mode: str,
    history_samples: int,
    history_key_batch_size: int,
    wandb_call_sleep: float,
    wandb_jitter: float,
    wandb_max_retries: int,
    wandb_backoff_base: float,
    wandb_backoff_max: float,
) -> tuple[RunInfo, dict[str, RunSeries]]:
    if minimal_history_keys:
        history_mode = "targeted"

    metric_keys = _metric_keys_for_agent(info.agent_label)
    series_by_metric: dict[str, RunSeries] = {}

    if history_mode == "single_full":
        rows = _call_wandb_with_backoff(
            lambda: list(info.run.scan_history(page_size=page_size)),
            label=f"{info.run_name} full-history",
            max_retries=wandb_max_retries,
            backoff_base=wandb_backoff_base,
            backoff_max=wandb_backoff_max,
            call_sleep=wandb_call_sleep,
            jitter=wandb_jitter,
        )
        for metric_key in metric_keys:
            series = _series_from_rows(rows, info.run_id, info.run_name, metric_key, x_key)
            if series is not None:
                series_by_metric[metric_key] = series
        return info, series_by_metric

    # Narrow mode: request only a few metric columns at a time. This is the safest
    # path when each W&B run contains thousands of unrelated metrics.
    batch_size = max(1, int(history_key_batch_size))
    for metric_batch in _chunked(metric_keys, batch_size):
        rows = _fetch_rows_for_keys(
            info.run,
            metric_batch,
            x_key=x_key,
            page_size=page_size,
            history_mode=history_mode,
            history_samples=history_samples,
            label=f"{info.run_name} keys={metric_batch}",
            wandb_max_retries=wandb_max_retries,
            wandb_backoff_base=wandb_backoff_base,
            wandb_backoff_max=wandb_backoff_max,
            wandb_call_sleep=wandb_call_sleep,
            wandb_jitter=wandb_jitter,
        )
        for metric_key in metric_batch:
            series = _series_from_rows(rows, info.run_id, info.run_name, metric_key, x_key)
            if series is not None:
                series_by_metric[metric_key] = series
    return info, series_by_metric


def fetch_all_series(
    grouped_runs: dict[str, dict[str, list[RunInfo]]],
    x_key: str,
    page_size: int,
    max_workers: int,
    minimal_history_keys: bool,
    history_mode: str,
    history_samples: int,
    history_key_batch_size: int,
    wandb_call_sleep: float,
    wandb_jitter: float,
    wandb_max_retries: int,
    wandb_backoff_base: float,
    wandb_backoff_max: float,
    request_batch_size: int,
    batch_sleep: float,
    skip_failed_runs: bool,
) -> dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]]:
    """Return variant -> agent -> metric -> list of per-run per-agent RunSeries dicts.

    Runs are processed in bounded batches instead of launching one large parallel
    burst. This is slower than unconstrained parallelism, but it avoids W&B rate
    limits and HTTP retry loops much more reliably.
    """
    all_infos = [info for by_agent in grouped_runs.values() for infos in by_agent.values() for info in infos]
    output: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]] = {
        variant: {agent: defaultdict(list) for agent in AGENTS} for variant in VARIANT_SPECS
    }

    if minimal_history_keys:
        history_mode = "targeted"

    workers = max(1, min(max_workers, len(all_infos) or 1))
    batch_size = max(1, int(request_batch_size))
    print(
        f"Fetching histories for {len(all_infos)} grouped runs with history_mode={history_mode}, "
        f"history_samples={history_samples}, key_batch_size={history_key_batch_size}, "
        f"workers={workers}, request_batch_size={batch_size}."
    )

    done = 0
    for batch_start in range(0, len(all_infos), batch_size):
        batch = all_infos[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"Starting W&B batch {batch_num}: runs {batch_start + 1}-{batch_start + len(batch)} of {len(all_infos)}")

        if workers == 1:
            for info in batch:
                try:
                    info, series_by_metric = _scan_run_once(
                        info,
                        x_key=x_key,
                        page_size=page_size,
                        minimal_history_keys=minimal_history_keys,
                        history_mode=history_mode,
                        history_samples=history_samples,
                        history_key_batch_size=history_key_batch_size,
                        wandb_call_sleep=wandb_call_sleep,
                        wandb_jitter=wandb_jitter,
                        wandb_max_retries=wandb_max_retries,
                        wandb_backoff_base=wandb_backoff_base,
                        wandb_backoff_max=wandb_backoff_max,
                    )
                    output[info.variant_name][info.agent_label]["__all__"].append(series_by_metric)
                except BaseException as exc:
                    if not skip_failed_runs:
                        raise
                    print(f"Skipping failed run {info.run_name} ({info.run_id}): {exc}")
                done += 1
                if done % 5 == 0 or done == len(all_infos):
                    print(f"  processed {done}/{len(all_infos)} histories")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _scan_run_once,
                        info,
                        x_key,
                        page_size,
                        minimal_history_keys,
                        history_mode,
                        history_samples,
                        history_key_batch_size,
                        wandb_call_sleep,
                        wandb_jitter,
                        wandb_max_retries,
                        wandb_backoff_base,
                        wandb_backoff_max,
                    )
                    for info in batch
                ]
                for future in as_completed(futures):
                    try:
                        info, series_by_metric = future.result()
                        output[info.variant_name][info.agent_label]["__all__"].append(series_by_metric)
                    except BaseException as exc:
                        if not skip_failed_runs:
                            raise
                        print(f"Skipping failed run after retries: {exc}")
                    done += 1
                    if done % 5 == 0 or done == len(all_infos):
                        print(f"  processed {done}/{len(all_infos)} histories")

        if batch_start + len(batch) < len(all_infos) and batch_sleep > 0:
            print(f"Sleeping {batch_sleep:.1f}s between W&B batches.")
            _sleep_with_jitter(batch_sleep, wandb_jitter)

    return output


def _run_groups_for_metric(
    extracted: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]],
    variant: str,
    agent_label: str,
    base_metric: str,
    expand: str,
) -> list[list[RunSeries]]:
    agent_count = _agent_count_from_label(agent_label)
    if expand == "ppo":
        keys = _expand_metric_for_ppo_agents(base_metric, agent_count)
    elif expand == "joint":
        keys = _expand_metric_for_joint_optimize_agents(base_metric, agent_count)
    else:
        keys = [base_metric]

    groups: list[list[RunSeries]] = []
    for series_by_metric in extracted[variant][agent_label].get("__all__", []):
        group = [series_by_metric[key] for key in keys if key in series_by_metric]
        if group:
            groups.append(group)
    return groups


def _flatten_series_groups(series_groups: list[Any]) -> list[RunSeries]:
    flattened: list[RunSeries] = []
    for item in series_groups:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _custom_sort_legend_keys(group_keys: list[str]) -> list[str]:
    def sort_key(key: str) -> tuple:
        parts = key.split(" | ")
        if parts and parts[-1].startswith("Agent "):
            try:
                return (" | ".join(parts[:-1]), 1, int(parts[-1].replace("Agent ", "")))
            except ValueError:
                pass
        return (key, 0, 0)
    return sorted(group_keys, key=sort_key)


def _strip_agent_count_label(group_name: str, agent_count_label: str) -> str:
    return group_name.replace(f" | {agent_count_label}", "")


def _final_series_value(series: RunSeries) -> float | None:
    return float(series.y[-1]) if series.y.size else None


def _mean_std_text(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return f"{float(np.mean(arr)):.3f} ± {float(np.std(arr)):.3f}"


def _compute_alpha_summaries(
    worst_by_agent: dict[str, list[list[RunSeries]]],
    optimal_by_agent: dict[str, list[list[RunSeries]]],
    agent_labels: list[str],
) -> dict[str, dict[str, Any]]:
    per_agent_values_by_label: dict[str, list[list[float]]] = {}
    true_values_by_label: dict[str, list[float]] = {}
    aggregate_per_agent_by_label: dict[str, list[float | None]] = {}
    aggregate_true_by_label: dict[str, float | None] = {}

    for agent_label in agent_labels:
        agent_count = _agent_count_from_label(agent_label)
        per_agent_values = [[] for _ in range(agent_count)]
        per_agent_worst_finals = [[] for _ in range(agent_count)]
        per_agent_optimal_finals = [[] for _ in range(agent_count)]
        true_values: list[float] = []

        worst_by_run_id = {group[0].run_id: group for group in worst_by_agent.get(agent_label, []) if group}
        optimal_by_run_id = {group[0].run_id: group for group in optimal_by_agent.get(agent_label, []) if group}

        for run_id in sorted(set(worst_by_run_id).intersection(optimal_by_run_id)):
            run_alpha_values: list[float] = []
            worst_group = worst_by_run_id[run_id]
            optimal_group = optimal_by_run_id[run_id]
            for agent_idx in range(min(agent_count, len(worst_group), len(optimal_group))):
                worst_final = _final_series_value(worst_group[agent_idx])
                optimal_final = _final_series_value(optimal_group[agent_idx])
                if worst_final is None or optimal_final in (None, 0):
                    continue
                alpha_value = worst_final / optimal_final
                per_agent_values[agent_idx].append(alpha_value)
                per_agent_worst_finals[agent_idx].append(worst_final)
                per_agent_optimal_finals[agent_idx].append(optimal_final)
                run_alpha_values.append(alpha_value)
            if run_alpha_values:
                true_values.append(min(run_alpha_values))

        aggregate_per_agent: list[float | None] = []
        for agent_idx in range(agent_count):
            worst_values = per_agent_worst_finals[agent_idx]
            optimal_values = per_agent_optimal_finals[agent_idx]
            if not worst_values or not optimal_values:
                aggregate_per_agent.append(None)
                continue
            optimal_mean = float(np.mean(np.asarray(optimal_values, dtype=float)))
            aggregate_per_agent.append(None if optimal_mean == 0 else float(np.mean(np.asarray(worst_values, dtype=float))) / optimal_mean)

        aggregate_true = min([value for value in aggregate_per_agent if value is not None], default=None)
        per_agent_values_by_label[agent_label] = per_agent_values
        true_values_by_label[agent_label] = true_values
        aggregate_per_agent_by_label[agent_label] = aggregate_per_agent
        aggregate_true_by_label[agent_label] = aggregate_true

    return {
        "per_agent": per_agent_values_by_label,
        "true": true_values_by_label,
        "aggregate_per_agent": aggregate_per_agent_by_label,
        "aggregate_true": aggregate_true_by_label,
    }


def aggregate_condition(series_list: list[RunSeries], min_runs_per_point: int) -> dict[str, np.ndarray]:
    by_x: dict[float, list[float]] = defaultdict(list)
    for series in series_list:
        for xv, yv in zip(series.x, series.y):
            by_x[float(xv)].append(float(yv))

    filtered_x: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for xv in sorted(by_x):
        ys = np.asarray(by_x[xv], dtype=float)
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
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, group_name in enumerate(_custom_sort_legend_keys(list(grouped_series.keys()))):
        series_list = _flatten_series_groups(grouped_series[group_name])
        if not series_list:
            continue
        display_name = _strip_agent_count_label(group_name, agent_count_label) if agent_count_label else group_name
        color = palette[idx % len(palette)]

        if show_individual_runs:
            for series in series_list:
                fig.add_trace(go.Scatter(
                    x=series.x,
                    y=series.y,
                    mode="lines",
                    line={"color": color, "width": 1},
                    opacity=0.15,
                    name=f"{display_name} run",
                    legendgroup=display_name,
                    showlegend=False,
                    hovertemplate=(f"Group: {display_name}<br>Run: {series.run_name}<br>{x_key}: %{{x}}<br>{metric}: %{{y}}<extra></extra>"),
                ))

        agg = aggregate_condition(series_list, min_runs_per_point=min_runs_per_point)
        if agg["x"].size == 0:
            continue

        upper = agg["mean"] + agg["std"]
        lower = agg["mean"] - agg["std"]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

        fig.add_trace(go.Scatter(x=agg["x"], y=upper, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, legendgroup=display_name, name=f"{display_name} +1 std"))
        fig.add_trace(go.Scatter(x=agg["x"], y=lower, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=f"rgba{rgb + (0.2,)}", hoverinfo="skip", showlegend=False, legendgroup=display_name, name=f"{display_name} -1 std"))
        fig.add_trace(go.Scatter(
            x=agg["x"],
            y=agg["mean"],
            mode="lines",
            line={"color": color, "width": 3},
            name=f"{display_name} (n={len(series_list)})",
            legendgroup=display_name,
            customdata=np.stack([agg["std"], agg["count"]], axis=-1),
            hovertemplate=(f"Group: {display_name}<br>{x_key}: %{{x}}<br>Mean: %{{y:.4f}}<br>Std: %{{customdata[0]:.4f}}<br>Runs at x: %{{customdata[1]}}<extra></extra>"),
        ))

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


def _per_agent_grouped_series(
    extracted: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]],
    agent_label: str,
    base_metric: str,
    expand: str,
) -> dict[str, list[RunSeries]]:
    grouped: dict[str, list[RunSeries]] = {}
    n = _agent_count_from_label(agent_label)
    for variant in VARIANT_SPECS:
        groups = _run_groups_for_metric(extracted, variant, agent_label, base_metric, expand)
        for ai in range(n):
            series_for_agent = [group[ai] for group in groups if len(group) > ai]
            grouped[f"{variant} | {agent_label} | Agent {ai + 1}"] = series_for_agent
    return grouped


def _variant_grouped_series(
    extracted: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]],
    agent_label: str,
    base_metric: str,
    expand: str,
) -> dict[str, list[RunSeries]]:
    return {
        f"{variant} | {agent_label}": _run_groups_for_metric(extracted, variant, agent_label, base_metric, expand)
        for variant in VARIANT_SPECS
    }


def _make_output_path(base: str, agent_label: str, kind: str) -> str:
    root, ext = os.path.splitext(base)
    return f"{root}_{agent_label.replace(' ', '_')}_{kind}{ext}"


def main() -> None:
    args = parse_args()
    if not args.entity or not args.project:
        raise ValueError("W&B entity/project not set. Pass --entity/--project or set WANDB_ENTITY/WANDB_PROJECT.")

    try:
        run_filters = json.loads(args.run_filters)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --run-filters: {exc}") from exc

    if args.fail_on_failed_runs:
        args.skip_failed_runs = False

    path = f"{args.entity}/{args.project}"
    cache_payload = None
    if args.data_cache and os.path.exists(args.data_cache) and not args.refresh_data_cache:
        print(f"Loading processed data cache: {args.data_cache}")
        with open(args.data_cache, "rb") as f:
            cache_payload = pickle.load(f)
        extracted = cache_payload["extracted"] if isinstance(cache_payload, dict) and "extracted" in cache_payload else cache_payload
    else:
        os.environ.setdefault("WANDB_HTTP_TIMEOUT", "120")
        os.environ.setdefault("WANDB_RETRY_MAX", str(max(0, args.wandb_max_retries)))
        api = wandb.Api(timeout=120)
        grouped_runs = list_and_group_runs(api, path, run_filters, args.max_runs)
        extracted = fetch_all_series(
            grouped_runs=grouped_runs,
            x_key=args.x_key,
            page_size=args.history_page_size,
            max_workers=args.max_workers,
            minimal_history_keys=args.minimal_history_keys,
            history_mode=args.history_mode,
            history_samples=args.history_samples,
            history_key_batch_size=args.history_key_batch_size,
            wandb_call_sleep=args.wandb_call_sleep,
            wandb_jitter=args.wandb_jitter,
            wandb_max_retries=args.wandb_max_retries,
            wandb_backoff_base=args.wandb_backoff_base,
            wandb_backoff_max=args.wandb_backoff_max,
            request_batch_size=args.request_batch_size,
            batch_sleep=args.batch_sleep,
            skip_failed_runs=args.skip_failed_runs,
        )
        if args.data_cache:
            cache_dir = os.path.dirname(args.data_cache)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(args.data_cache, "wb") as f:
                pickle.dump({"extracted": extracted}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved processed data cache: {args.data_cache}")

    # Alpha summaries from the already-extracted eval metrics.
    alpha_summaries_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANT_SPECS:
        worst = {agent: _run_groups_for_metric(extracted, variant, agent, ALPHA_WORST_EVAL, "joint") for agent in AGENTS}
        optimal = {agent: _run_groups_for_metric(extracted, variant, agent, ALPHA_OPTIMAL_EVAL, "joint") for agent in AGENTS}
        alpha_summaries_by_variant[variant] = _compute_alpha_summaries(worst, optimal, AGENTS)

    all_figures: list[tuple[str, go.Figure]] = []
    total_valid = 0
    figure_specs = [
        ("Returns", TRAIN_RETURN, "ppo", "Mean Returns", True, "returns"),
        ("Collisions", TRAIN_COLLISIONS, "ppo", "Mean Collisions", True, "collisions"),
        ("Worst Case", WORST_RETURN, "joint", "Mean Worst Case Returns", False, "worst_case"),
        ("Worst Case Collisions", WORST_COLLISIONS, "joint", "Mean Worst Case Collisions", False, "worst_case_collisions"),
        ("Best Case", BEST_RETURN, "joint", "Mean Best Case Returns", False, "best_case"),
        ("Best Case Collisions", BEST_COLLISIONS, "joint", "Mean Best Case Collisions", False, "best_case_collisions"),
    ]

    for agent_label in AGENTS:
        for kind_title, metric, expand, y_title, per_agent, out_kind in figure_specs:
            grouped = (
                _per_agent_grouped_series(extracted, agent_label, metric, expand)
                if per_agent
                else _variant_grouped_series(extracted, agent_label, metric, expand)
            )
            title = f"{args.title} | {agent_label} | {kind_title}"
            fig = make_variance_figure(
                grouped_series=grouped,
                metric=metric,
                x_key=args.x_key,
                title=title,
                min_runs_per_point=args.min_runs_per_point,
                show_individual_runs=args.show_individual_runs,
                y_axis_title=y_title,
                agent_count_label=agent_label,
                show_plot_title=args.show_plot_title,
            )
            all_figures.append((title, fig))
            print(f"Prepared plot: {title}")
            if out_kind == "returns":
                total_valid += sum(len(series_list) for series_list in grouped.values())
            if args.png_output:
                out_png = _make_output_path(args.png_output, agent_label, out_kind)
                fig.write_image(out_png)
                print(f"Saved PNG plot to: {out_png}")

    html_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        f"  <title>{escape(args.title)}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    h1 { margin-bottom: 8px; }",
        "    .plot-block { margin: 24px 0 48px 0; }",
        "    .plot-title { margin: 0 0 12px 0; }",
        "    .alpha-section { margin-top: 36px; }",
        "    .alpha-note { margin: 0 0 16px 0; color: #444; }",
        "    .alpha-table { border-collapse: collapse; margin: 18px 0 28px 0; width: 100%; }",
        "    .alpha-table th, .alpha-table td { border: 1px solid #d0d0d0; padding: 8px 10px; vertical-align: top; text-align: left; }",
        "    .alpha-table th { background: #f6f6f6; }",
        "    .alpha-agent-title { margin: 18px 0 8px 0; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{escape(args.title)}</h1>",
        f"  <p>Total plots: {len(all_figures)}</p>",
    ]

    for idx, (plot_title, fig) in enumerate(all_figures):
        html_parts.append("  <div class=\"plot-block\">")
        html_parts.append(f"    <h2 class=\"plot-title\">{escape(plot_title)}</h2>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(idx == 0)))
        html_parts.append("  </div>")

    html_parts.append("  <section class=\"alpha-section\">")
    html_parts.append("    <h2>Alpha Summary</h2>")
    html_parts.append(
        "    <p class=\"alpha-note\">Alpha is computed from the final logged evaluation point only, "
        "using Eval/Joint/Agent_#_Optimize/WorstCase/Return divided by "
        "Eval/Joint/Agent_#_Optimize/OptimalReturn. Each cell reports the run-level alpha mean ± std "
        "and the aggregate-final alpha point estimate. The true alpha row is the per-run min across agents.</p>"
    )

    for agent_label in AGENTS:
        agent_count = _agent_count_from_label(agent_label)
        html_parts.append(f"    <h3 class=\"alpha-agent-title\">{escape(agent_label)}</h3>")
        html_parts.append("    <table class=\"alpha-table\">")
        header_cells = ["Variant"] + [f"Agent {idx + 1}" for idx in range(agent_count)] + ["True alpha"]
        html_parts.append("      <thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in header_cells) + "</tr></thead>")
        html_parts.append("      <tbody>")
        for variant in VARIANT_SPECS:
            summary = alpha_summaries_by_variant[variant]
            per_agent_values = summary["per_agent"][agent_label]
            aggregate_per_agent = summary["aggregate_per_agent"][agent_label]
            true_values = summary["true"][agent_label]
            aggregate_true = summary["aggregate_true"][agent_label]
            row_cells = [escape(variant)]
            for agent_idx, values in enumerate(per_agent_values):
                aggregate_value = aggregate_per_agent[agent_idx] if agent_idx < len(aggregate_per_agent) else None
                aggregate_text = f"{aggregate_value:.3f}" if aggregate_value is not None else "n/a"
                row_cells.append(f"run {_mean_std_text(values)}<br />agg {aggregate_text}")
            aggregate_true_text = f"{aggregate_true:.3f}" if aggregate_true is not None else "n/a"
            row_cells.append(f"run {_mean_std_text(true_values)}<br />agg {aggregate_true_text}")
            html_parts.append("        <tr>" + "".join(f"<td>{cell}</td>" for cell in row_cells) + "</tr>")
        html_parts.append("      </tbody>")
        html_parts.append("    </table>")

    html_parts.extend(["  </section>", "</body>", "</html>"])

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"Saved combined interactive plot report to: {args.output}")
    print(f"Valid runs (returns grouped): {total_valid}")


if __name__ == "__main__":
    main()