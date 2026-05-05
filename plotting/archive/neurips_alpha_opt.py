"""Create variance plots from Weights & Biases runs using Plotly.

Optimized version of the original script:
  - lists W&B runs once, then groups locally by tags;
  - scans each run's history once;
  - extracts all needed metrics from that one history pass;
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
import multiprocessing as mp
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
SINGLE_AGENT_EVAL_RETURN = "Eval/Single_Agent_Proj_Joint/Return"
SINGLE_AGENT_EVAL_COLLISIONS = "Eval/Single_Agent_Proj_Joint/Collisions"


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
    parser.add_argument("--max-workers", type=int, default=2, help="Number of run-history fetches to run concurrently. Use 2-4; avoid 8+ with W&B.")
    parser.add_argument("--history-page-size", type=int, default=10000)
    parser.add_argument(
        "--history-mode",
        choices=["targeted", "single_full"],
        default="targeted",
        help=(
            "targeted fetches each needed metric key separately with scan_history(keys=[x, metric]); "
            "single_full falls back to the original full scan_history() behavior."
        ),
    )
    parser.add_argument(
        "--minimal-history-keys",
        action="store_true",
        help="Deprecated compatibility flag. Equivalent to --history-mode targeted.",
    )
    parser.add_argument(
        "--wandb-call-sleep",
        type=float,
        default=1.1,
        help="Minimum sleep before starting each run-history subprocess. Default follows W&B's 1-second guidance.",
    )
    parser.add_argument(
        "--wandb-jitter",
        type=float,
        default=0.5,
        help="Add up to this many random seconds of jitter before each request.",
    )
    parser.add_argument("--wandb-timeout", type=int, default=120, help="Timeout passed to wandb.Api(timeout=...).")
    parser.add_argument("--wandb-retries", type=int, default=0, help="Script-level retries per run after an exception. Keep low to avoid long stalls.")
    parser.add_argument("--wandb-retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--scan-timeout-seconds",
        type=int,
        default=90,
        help="Kill/skip a run-history subprocess after this many seconds. Set 0 to disable subprocess timeout mode.",
    )
    parser.add_argument("--skip-failed-runs", action="store_true", default=True, help="Skip timed-out/failed runs and keep going. Enabled by default.")
    parser.add_argument("--fail-on-failed-runs", action="store_true", help="Stop immediately instead of skipping failed/timed-out runs.")
    parser.add_argument("--data-cache", type=str, default="plotting_data/neurips_alpha_opt.pkl", help="Optional pickle path for processed W&B data.")
    parser.add_argument("--refresh-data-cache", action="store_true", help="Ignore existing --data-cache and fetch W&B data again.")
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
    keys.extend([SINGLE_AGENT_EVAL_RETURN, SINGLE_AGENT_EVAL_COLLISIONS])
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


def _sleep_with_jitter(base_sleep: float, jitter: float) -> None:
    delay = max(0.0, base_sleep) + (random.uniform(0.0, max(0.0, jitter)) if jitter else 0.0)
    if delay > 0:
        time.sleep(delay)


def _series_from_summary(info: RunInfo, metric_key: str, x_key: str) -> RunSeries | None:
    """Fast path for final eval metrics stored in run.summary."""
    try:
        summary = getattr(info.run, "summary", {}) or {}
        if metric_key not in summary:
            return None
        yv = summary.get(metric_key)
        xv = summary.get(x_key, summary.get("_step", 0))
        if yv is None:
            return None
        xf = float(xv) if xv is not None else 0.0
        yf = float(yv)
        if not (np.isfinite(xf) and np.isfinite(yf)):
            return None
        return RunSeries(
            run_id=info.run_id,
            run_name=f"{info.run_name} | {metric_key}",
            x=np.asarray([xf], dtype=float),
            y=np.asarray([yf], dtype=float),
        )
    except Exception:
        return None


def _scan_run_once(
    info: RunInfo,
    x_key: str,
    page_size: int,
    history_mode: str,
    wandb_retries: int = 0,
    wandb_retry_sleep: float = 10.0,
) -> tuple[RunInfo, dict[str, RunSeries]]:
    """Fetch one run. In targeted mode, each metric is requested separately.

    Requesting a single metric at a time avoids W&B's sparse-row behavior where
    scan_history(keys=[x, metric1, metric2, ...]) can return no useful rows if
    those metrics are logged on different rows. It also avoids materializing the
    thousands of unrelated metrics that made single_full time out.
    """
    metric_keys = _metric_keys_for_agent(info.agent_label)
    series_by_metric: dict[str, RunSeries] = {}

    # Alpha table metrics are usually in summary; this avoids scanning eval history.
    for metric_key in _expand_metric_for_joint_optimize_agents(ALPHA_WORST_EVAL, _agent_count_from_label(info.agent_label)):
        series = _series_from_summary(info, metric_key, x_key)
        if series is not None:
            series_by_metric[metric_key] = series
    for metric_key in _expand_metric_for_joint_optimize_agents(ALPHA_OPTIMAL_EVAL, _agent_count_from_label(info.agent_label)):
        series = _series_from_summary(info, metric_key, x_key)
        if series is not None:
            series_by_metric[metric_key] = series

    attempts = max(1, wandb_retries + 1)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            if history_mode == "single_full":
                rows = list(info.run.scan_history(page_size=page_size))
                time.sleep(7.0)
                for metric_key in metric_keys:
                    if metric_key in series_by_metric:
                        continue
                    series = _series_from_rows(rows, info.run_id, info.run_name, metric_key, x_key)
                    if series is not None:
                        series_by_metric[metric_key] = series
                return info, series_by_metric

            if history_mode != "targeted":
                raise ValueError(f"Unknown history_mode: {history_mode}")

            # Targeted: one metric per scan. This is more API calls, but far less data per call.
            for metric_key in metric_keys:
                if metric_key in series_by_metric:
                    continue
                rows = list(info.run.scan_history(keys=[x_key, metric_key], page_size=page_size))
                time.sleep(7.0)
                series = _series_from_rows(rows, info.run_id, info.run_name, metric_key, x_key)
                if series is not None:
                    series_by_metric[metric_key] = series
            return info, series_by_metric
        except BaseException as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_for = wandb_retry_sleep * attempt
            print(f"W&B request failed for {info.run_name} ({info.run_id}) attempt {attempt}/{attempts}: {exc}. Sleeping {sleep_for:.1f}s.")
            time.sleep(max(0.0, sleep_for))

    assert last_exc is not None
    raise RuntimeError(f"Failed to fetch W&B history for {info.run_name} ({info.run_id})") from last_exc


def _scan_run_worker(
    queue: Any,
    run_path: str,
    run_id: str,
    run_name: str,
    agent_label: str,
    variant_name: str,
    x_key: str,
    page_size: int,
    history_mode: str,
    wandb_timeout: int,
    wandb_retries: int,
    wandb_retry_sleep: float,
) -> None:
    try:
        # Keep W&B's own retry behavior from stretching forever where supported.
        os.environ.setdefault("WANDB_HTTP_TIMEOUT", str(wandb_timeout))
        os.environ.setdefault("WANDB_RETRY_MAX", str(max(0, wandb_retries)))
        api = wandb.Api(timeout=wandb_timeout)
        run = api.run(run_path)
        info = RunInfo(
            run=run,
            run_id=run_id,
            run_name=run_name,
            tags=set(),
            agent_label=agent_label,
            variant_name=variant_name,
        )
        _, series = _scan_run_once(
            info=info,
            x_key=x_key,
            page_size=page_size,
            history_mode=history_mode,
            wandb_retries=wandb_retries,
            wandb_retry_sleep=wandb_retry_sleep,
        )
        queue.put(("ok", series))
    except BaseException as exc:
        queue.put(("error", repr(exc)))


def _fetch_all_series_with_process_pool(
    all_infos: list[RunInfo],
    path: str,
    x_key: str,
    page_size: int,
    max_workers: int,
    history_mode: str,
    wandb_timeout: int,
    wandb_retries: int,
    wandb_retry_sleep: float,
    wandb_call_sleep: float,
    wandb_jitter: float,
    scan_timeout_seconds: int,
    skip_failed_runs: bool,
) -> dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]]:
    output: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]] = {
        variant: {agent: defaultdict(list) for agent in AGENTS} for variant in VARIANT_SPECS
    }
    total = len(all_infos)
    workers = max(1, min(max_workers, total or 1))
    ctx = mp.get_context("spawn")
    pending = list(all_infos)
    active: dict[mp.Process, tuple[RunInfo, Any, float]] = {}
    started = 0
    finished = 0

    print(
        f"Using subprocess timeout mode with up to {workers} concurrent run scans; "
        f"each run has {scan_timeout_seconds}s before it is killed/skipped."
    )

    def start_next() -> None:
        nonlocal started
        info = pending.pop(0)
        started += 1
        if wandb_call_sleep > 0 or wandb_jitter > 0:
            print(
                f"  [{started}/{total}] starting {info.run_name} ({info.run_id}) "
                f"after throttle sleep {wandb_call_sleep:.1f}s + jitter up to {wandb_jitter:.1f}s"
            )
            _sleep_with_jitter(wandb_call_sleep, wandb_jitter)
        q = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_scan_run_worker,
            args=(
                q,
                f"{path}/{info.run_id}",
                info.run_id,
                info.run_name,
                info.agent_label,
                info.variant_name,
                x_key,
                page_size,
                history_mode,
                wandb_timeout,
                wandb_retries,
                wandb_retry_sleep,
            ),
        )
        proc.start()
        active[proc] = (info, q, time.monotonic())

    while pending and len(active) < workers:
        start_next()

    while active:
        time.sleep(0.5)
        for proc in list(active):
            info, q, start_time = active[proc]
            timed_out = scan_timeout_seconds > 0 and (time.monotonic() - start_time) > scan_timeout_seconds
            if proc.is_alive() and not timed_out:
                continue

            if timed_out and proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(2)
                msg = f"Skipping timed-out run {info.run_name} ({info.run_id}) after {scan_timeout_seconds}s."
                if not skip_failed_runs:
                    raise TimeoutError(msg)
                print(msg)
            else:
                proc.join()
                if q.empty():
                    msg = f"W&B worker exited without data for {info.run_name} ({info.run_id})."
                    if not skip_failed_runs:
                        raise RuntimeError(msg)
                    print("Skipping failed run: " + msg)
                else:
                    status, payload = q.get()
                    if status == "ok":
                        output[info.variant_name][info.agent_label]["__all__"].append(payload)
                    else:
                        msg = f"W&B worker failed for {info.run_name} ({info.run_id}): {payload}"
                        if not skip_failed_runs:
                            raise RuntimeError(msg)
                        print("Skipping failed run: " + msg)

            del active[proc]
            finished += 1
            if finished % 5 == 0 or finished == total:
                print(f"  processed {finished}/{total} runs")
            while pending and len(active) < workers:
                start_next()

    return output


def fetch_all_series(
    grouped_runs: dict[str, dict[str, list[RunInfo]]],
    x_key: str,
    page_size: int,
    max_workers: int,
    history_mode: str,
    *,
    path: str,
    wandb_timeout: int,
    wandb_retries: int,
    wandb_retry_sleep: float,
    wandb_call_sleep: float,
    wandb_jitter: float,
    scan_timeout_seconds: int,
    skip_failed_runs: bool,
) -> dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]]:
    """Return variant -> agent -> metric -> list of per-run per-agent RunSeries dicts."""
    all_infos = [info for by_agent in grouped_runs.values() for infos in by_agent.values() for info in infos]
    print(f"Scanning histories once for {len(all_infos)} grouped runs with history_mode={history_mode}...")

    if scan_timeout_seconds and scan_timeout_seconds > 0:
        return _fetch_all_series_with_process_pool(
            all_infos=all_infos,
            path=path,
            x_key=x_key,
            page_size=page_size,
            max_workers=max_workers,
            history_mode=history_mode,
            wandb_timeout=wandb_timeout,
            wandb_retries=wandb_retries,
            wandb_retry_sleep=wandb_retry_sleep,
            wandb_call_sleep=wandb_call_sleep,
            wandb_jitter=wandb_jitter,
            scan_timeout_seconds=scan_timeout_seconds,
            skip_failed_runs=skip_failed_runs,
        )

    output: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]] = {
        variant: {agent: defaultdict(list) for agent in AGENTS} for variant in VARIANT_SPECS
    }
    workers = max(1, min(max_workers, len(all_infos) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for info in all_infos:
            _sleep_with_jitter(wandb_call_sleep, wandb_jitter)
            futures.append(executor.submit(_scan_run_once, info, x_key, page_size, history_mode, wandb_retries, wandb_retry_sleep))
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                info, series_by_metric = future.result()
                output[info.variant_name][info.agent_label]["__all__"].append(series_by_metric)
            except BaseException as exc:
                if not skip_failed_runs:
                    raise
                print(f"Skipping failed run after retries: {exc}")
            if done % 10 == 0 or done == len(futures):
                print(f"  scanned {done}/{len(futures)} histories")
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
    ratio_mode: str = "worst_over_optimal",
) -> dict[str, dict[str, Any]]:
    """Compute per-agent and true-alpha summaries.

    ratio_mode="worst_over_optimal" computes WorstCase / OptimalReturn.
    ratio_mode="optimal_over_worst" computes OptimalReturn / WorstCase as a diagnostic.
    """
    if ratio_mode not in {"worst_over_optimal", "optimal_over_worst"}:
        raise ValueError(f"Unknown ratio_mode: {ratio_mode}")

    per_agent_values_by_label: dict[str, list[list[float]]] = {}
    true_values_by_label: dict[str, float | None] = {}

    for agent_label in agent_labels:
        agent_count = _agent_count_from_label(agent_label)
        per_agent_values = [[] for _ in range(agent_count)]

        worst_by_run_id = {group[0].run_id: group for group in worst_by_agent.get(agent_label, []) if group}
        optimal_by_run_id = {group[0].run_id: group for group in optimal_by_agent.get(agent_label, []) if group}

        for run_id in sorted(set(worst_by_run_id).intersection(optimal_by_run_id)):
            worst_group = worst_by_run_id[run_id]
            optimal_group = optimal_by_run_id[run_id]
            for agent_idx in range(min(agent_count, len(worst_group), len(optimal_group))):
                worst_final = _final_series_value(worst_group[agent_idx])
                optimal_final = _final_series_value(optimal_group[agent_idx])
                if worst_final is None or optimal_final is None:
                    continue

                if ratio_mode == "worst_over_optimal":
                    numerator = worst_final
                    denominator = optimal_final
                else:
                    numerator = optimal_final
                    denominator = worst_final

                if denominator == 0:
                    continue

                alpha_value = numerator / denominator
                if np.isfinite(alpha_value):
                    per_agent_values[agent_idx].append(float(alpha_value))

        # Compute per-agent means, then take the min across agents
        per_agent_means = [np.mean(np.asarray(values, dtype=float)) if values else None for values in per_agent_values]
        non_none_means = [mean for mean in per_agent_means if mean is not None]
        true_value = float(min(non_none_means)) if non_none_means else None

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



def _cache_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cache_version": 2,
        "entity": args.entity,
        "project": args.project,
        "x_key": args.x_key,
        "run_filters": args.run_filters,
        "max_runs": args.max_runs,
        "history_page_size": args.history_page_size,
        "history_mode": "targeted" if args.minimal_history_keys else args.history_mode,
    }


def _load_extracted_cache(cache_path: str, expected_metadata: dict[str, Any]) -> dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]]:
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "extracted" not in payload:
        raise ValueError(f"Cache file {cache_path} is not a valid processed-data cache.")
    cached_metadata = payload.get("metadata", {})
    mismatches = [(k, cached_metadata.get(k), v) for k, v in expected_metadata.items() if cached_metadata.get(k) != v]
    if mismatches:
        print("Warning: loaded data cache was created with different data-fetch settings:")
        for key, cached, expected in mismatches:
            print(f"  {key}: cache={cached!r}, current={expected!r}")
        print("If plots look wrong, rerun with --refresh-data-cache.")
    print(f"Loaded processed W&B data cache: {cache_path}")
    return payload["extracted"]


def _save_extracted_cache(cache_path: str, extracted: dict[str, dict[str, dict[str, list[dict[str, RunSeries]]]]], metadata: dict[str, Any]) -> None:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"metadata": metadata, "extracted": extracted}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)
    print(f"Saved processed W&B data cache to: {cache_path}")

def main() -> None:
    args = parse_args()
    if not args.entity or not args.project:
        raise ValueError("W&B entity/project not set. Pass --entity/--project or set WANDB_ENTITY/WANDB_PROJECT.")

    try:
        run_filters = json.loads(args.run_filters)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --run-filters: {exc}") from exc

    history_mode = "targeted" if args.minimal_history_keys else args.history_mode
    skip_failed_runs = args.skip_failed_runs and not args.fail_on_failed_runs
    path = f"{args.entity}/{args.project}"
    cache_metadata = _cache_metadata(args)

    if args.data_cache and os.path.exists(args.data_cache) and not args.refresh_data_cache:
        extracted = _load_extracted_cache(args.data_cache, cache_metadata)
    else:
        api = wandb.Api(timeout=args.wandb_timeout)
        grouped_runs = list_and_group_runs(api, path, run_filters, args.max_runs)
        extracted = fetch_all_series(
            grouped_runs=grouped_runs,
            x_key=args.x_key,
            page_size=args.history_page_size,
            max_workers=args.max_workers,
            history_mode=history_mode,
            path=path,
            wandb_timeout=args.wandb_timeout,
            wandb_retries=args.wandb_retries,
            wandb_retry_sleep=args.wandb_retry_sleep,
            wandb_call_sleep=args.wandb_call_sleep,
            wandb_jitter=args.wandb_jitter,
            scan_timeout_seconds=args.scan_timeout_seconds,
            skip_failed_runs=skip_failed_runs,
        )
        if args.data_cache:
            _save_extracted_cache(args.data_cache, extracted, cache_metadata)

    # Alpha summaries from the already-extracted eval metrics.
    alpha_summaries_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    inverse_alpha_summaries_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANT_SPECS:
        worst = {agent: _run_groups_for_metric(extracted, variant, agent, ALPHA_WORST_EVAL, "joint") for agent in AGENTS}
        optimal = {agent: _run_groups_for_metric(extracted, variant, agent, ALPHA_OPTIMAL_EVAL, "joint") for agent in AGENTS}
        alpha_summaries_by_variant[variant] = _compute_alpha_summaries(worst, optimal, AGENTS, ratio_mode="worst_over_optimal")
        inverse_alpha_summaries_by_variant[variant] = _compute_alpha_summaries(worst, optimal, AGENTS, ratio_mode="optimal_over_worst")

    all_figures: list[tuple[str, go.Figure]] = []
    total_valid = 0
    figure_specs = [
        ("Returns", TRAIN_RETURN, "ppo", "Mean Returns", True, "returns"),
        ("Collisions", TRAIN_COLLISIONS, "ppo", "Mean Collisions", True, "collisions"),
        ("Worst Case", WORST_RETURN, "joint", "Mean Worst Case Returns", False, "worst_case"),
        ("Worst Case Collisions", WORST_COLLISIONS, "joint", "Mean Worst Case Collisions", False, "worst_case_collisions"),
        ("Best Case", BEST_RETURN, "joint", "Mean Best Case Returns", False, "best_case"),
        ("Best Case Collisions", BEST_COLLISIONS, "joint", "Mean Best Case Collisions", False, "best_case_collisions"),
        ("Eval Worst Return", ALPHA_WORST_EVAL, "joint", "Mean Eval Worst Case Returns", False, "eval_worst"),
        ("Eval Optimal Return", ALPHA_OPTIMAL_EVAL, "joint", "Mean Eval Optimal Returns", False, "eval_optimal"),
        ("Eval Single Agent Return", SINGLE_AGENT_EVAL_RETURN, "none", "Mean Eval Single Agent Returns", False, "eval_single_agent_return"),
        ("Eval Single Agent Collisions", SINGLE_AGENT_EVAL_COLLISIONS, "none", "Mean Eval Single Agent Collisions", False, "eval_single_agent_collisions"),
    ]

    for agent_label in AGENTS:
        for kind_title, metric, expand, y_title, per_agent, out_kind in figure_specs:
            # For single-agent eval metrics, only plot for the base agent count (2 agents)
            # since they don't vary per agent and are redundant across all agent counts
            if expand == "none" and agent_label != AGENTS[0]:
                continue
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
        "Eval/Joint/Agent_#_Optimize/OptimalReturn. Each cell reports the run-level alpha mean ± std per agent. "
        "The true alpha column is the minimum of the per-agent mean alpha values.</p>"
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
            true_value = summary["true"][agent_label]
            row_cells = [escape(variant)]
            for agent_idx, values in enumerate(per_agent_values):
                row_cells.append(_mean_std_text(values))
            true_alpha = f"{float(true_value):.3f}" if true_value is not None else "n/a"
            row_cells.append(true_alpha)
            html_parts.append("        <tr>" + "".join(f"<td>{cell}</td>" for cell in row_cells) + "</tr>")
        html_parts.append("      </tbody>")
        html_parts.append("    </table>")

    html_parts.append("    <h2>Alternate Alpha Summary: Optimal / Worst Case</h2>")
    html_parts.append(
        "    <p class=\"alpha-note\">This separate diagnostic table computes alpha as "
        "Eval/Joint/Agent_#_Optimize/OptimalReturn divided by "
        "Eval/Joint/Agent_#_Optimize/WorstCase/Return. The true alpha column is the minimum "
        "of the per-agent mean alpha values.</p>"
    )
    for agent_label in AGENTS:
        agent_count = _agent_count_from_label(agent_label)
        html_parts.append(f"    <h3 class=\"alpha-agent-title\">{escape(agent_label)}</h3>")
        html_parts.append("    <table class=\"alpha-table\">")
        header_cells = ["Variant"] + [f"Agent {idx + 1}" for idx in range(agent_count)] + ["True alpha"]
        html_parts.append("      <thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in header_cells) + "</tr></thead>")
        html_parts.append("      <tbody>")
        for variant in VARIANT_SPECS:
            summary = inverse_alpha_summaries_by_variant[variant]
            per_agent_values = summary["per_agent"][agent_label]
            true_value = summary["true"][agent_label]
            row_cells = [escape(variant)]
            for agent_idx, values in enumerate(per_agent_values):
                row_cells.append(_mean_std_text(values))
            true_alpha = f"{float(true_value):.3f}" if true_value is not None else "n/a"
            row_cells.append(true_alpha)
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