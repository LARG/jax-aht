"""Shared utilities for training-curve plots.

Two responsibilities:
  1. Discover wandb runs by (algorithm, task, tag).
  2. Download + cache the orbax-checkpointed `saved_train_run` and `ego_train_run`
     artifacts so that downstream code can extract per-seed training curves.

We use the artifacts (not `run.scan_history`) because the per-step metrics that
get logged to wandb are already averaged across seeds in `fcp.py`,
`train_ego.py`, etc. The artifacts preserve the seed dimension.
"""
from __future__ import annotations

import os
# Force CPU for orbax restore. Train_run artifacts can be 6+ GB once expanded
# (e.g. cramped_room ego eval has shape (5,1,N,82,20,2)) and crash GPU at load.
# Setting via setdefault so callers can override if they really want GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Redirect wandb's artifact dedup cache out of ~/.cache/wandb (which can hit
# the home-dir quota on this cluster). We don't want the cache anyway — we
# do our own metrics caching at a coarser grain.
os.environ.setdefault("WANDB_CACHE_DIR", "/tmp/wandb-cache-jyliu")
os.environ.setdefault("WANDB_DATA_DIR", "/tmp/wandb-data-jyliu")

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import wandb

DEFAULT_CACHE_DIR = Path("results/figures/cache")


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

# Ego algo names differ between this package and `plot_globals.EGO_BENCHMARK_RUNS`.
_EGO_ALGO_TO_PIN_KEY = {
    "ppo_ego": "ppo_ego",
    "liam_ego": "liam",
    "meliba_ego": "meliba",
}


def _as_id_list(value: Any) -> list[str]:
    """Pinned entries are either a single run id or a list of them."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def pinned_run_ids(algorithm: str, task: str) -> list[str]:
    """Run ids pinned in `plot_globals`, or [] if this (algorithm, task) isn't pinned.

    Teammate-gen algos come from `UNIFIED_BENCHMARK_RUNS[task][algo]`. Ego algos
    come from `EGO_BENCHMARK_RUNS[task][algo][<set>_teammates]`, pooled across
    teammate sets — callers re-stratify by reading each run's config, so the
    grouping in the pin table doesn't need to be preserved here.

    Pinning matters because tag discovery returns whatever currently carries the
    `neurips:benchmark` tag, which drifts as runs are re-launched. The pin table
    is what the rest of the paper figures use.
    """
    from scripts.paper_vis.plot_globals import (
        EGO_BENCHMARK_RUNS,
        UNIFIED_BENCHMARK_RUNS,
    )

    if algorithm in _EGO_ALGO_TO_PIN_KEY:
        by_algo = EGO_BENCHMARK_RUNS.get(task, {})
        by_set = by_algo.get(_EGO_ALGO_TO_PIN_KEY[algorithm], {})
        ids: list[str] = []
        for _teammate_set, value in sorted(by_set.items()):
            ids.extend(_as_id_list(value))
        return ids
    return _as_id_list(UNIFIED_BENCHMARK_RUNS.get(task, {}).get(algorithm))


def find_benchmark_runs(
    algorithm: str,
    task: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    tag: str = "neurips:benchmark",
    state: str | None = "finished",
    use_pinned: bool = True,
) -> list[Any]:
    """Return wandb Run objects matching (tag, algorithm, task).

    Args:
        algorithm: value of `config.algorithm.ALG` (e.g. "fcp", "comedi").
        task: value of `config.TASK_NAME` (e.g. "overcooked-v1/coord_ring").
        state: if not None, only runs with this state. Use None to include all.
        use_pinned: prefer the run ids pinned in `plot_globals` over tag
            discovery. Falls back to the tag query when nothing is pinned for
            this (algorithm, task).
    """
    api = wandb.Api()

    if use_pinned:
        ids = pinned_run_ids(algorithm, task)
        if ids:
            runs = [api.run(f"{entity}/{project}/{rid}") for rid in ids]
            print(f"  [pinned] {algorithm}/{task}: {ids}")
            return runs

    filters: dict = {
        "tags": tag,
        "config.algorithm.ALG": algorithm,
        "config.TASK_NAME": task,
    }
    if state is not None:
        filters["state"] = state
    runs = list(api.runs(f"{entity}/{project}", filters=filters, per_page=100))
    return runs


# ---------------------------------------------------------------------------
# Artifact download + cache
# ---------------------------------------------------------------------------

def _artifact_cache_path(
    cache_dir: Path,
    entity: str,
    project: str,
    run_id: str,
    artifact_kind: str,
    extras_suffix: str = "",
) -> Path:
    """Cache file path for the per-run extracted metrics pytree."""
    name = f"{entity}__{project}__{run_id}__{artifact_kind}{extras_suffix}.pkl"
    return Path(cache_dir) / "train_run_metrics" / name


def fetch_train_run_metrics_cached(
    run: Any,
    artifact_kind: str,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
    reduce_per_update: bool = False,
    extra_top_level_keys: tuple[str, ...] = (),
    tuple_index: int | None = None,
) -> dict:
    """Download + cache the `metrics` subtree of a train_run artifact.

    Args:
        run: wandb Run object (from `find_benchmark_runs` or `api.run(...)`).
        artifact_kind: substring matched against artifact names. Use
            "saved_train_run" for partner training, "ego_train_run" for ego.
        cache_dir: root of the local cache tree.
        reduce_per_update: if True, apply `mask_and_mean` over the trailing
            (rollout, num_envs) axes when leaves still carry them. Required for
            older partner-training artifacts (e.g. FCP cramped_room) that saved
            the un-reduced trajectory metric and would otherwise pickle to tens
            of GB. Newer artifacts (e.g. FCP coord_ring) are already reduced
            and this is a no-op for them.
        extra_top_level_keys: additional keys from the artifact's `out` dict to
            include in the result (e.g. `("final_xp_matrix",)` for COLE). When
            non-empty the return is `{"metrics": <metrics>, **extras}`; when
            empty the return is just the metrics dict (backwards-compatible).
        tuple_index: when the artifact root is a tuple/list (e.g. ROTATE saves
            `(teammate_outs, ego_outs)`), select this index before extracting
            metrics. None (default) treats the root as a dict.

    Returns:
        Either a metrics dict (default) or a dict with both metrics and
        the requested extras (when `extra_top_level_keys` is non-empty).
    """
    extras_suffix = ""
    if extra_top_level_keys:
        extras_suffix = "__" + "_".join(sorted(extra_top_level_keys))
    if tuple_index is not None:
        extras_suffix = f"__idx{tuple_index}{extras_suffix}"
    cache_path = _artifact_cache_path(
        cache_dir, entity, project, run.id, artifact_kind, extras_suffix=extras_suffix,
    )

    if not force_recompute and cache_path.exists():
        print(f"  [cache] loading {artifact_kind} metrics from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    target_artifact = None
    for art in run.logged_artifacts():
        if artifact_kind in art.name:
            target_artifact = art
            break
    if target_artifact is None:
        raise ValueError(
            f"No artifact matching {artifact_kind!r} on run {run.id} "
            f"({entity}/{project}). Logged artifact names: "
            f"{[a.name for a in run.logged_artifacts()]}"
        )

    print(f"  [wandb] downloading {target_artifact.name} ({target_artifact.size / 1e6:.0f} MB)")
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = target_artifact.download(root=tmp)
        # Lazy import to avoid pulling jax for callers that only do discovery.
        from common.save_load_utils import load_train_run
        out = load_train_run(artifact_dir)

    if tuple_index is not None:
        if not isinstance(out, (list, tuple)):
            raise ValueError(
                f"Expected tuple/list root for {target_artifact.name} (tuple_index={tuple_index}), "
                f"got {type(out).__name__}"
            )
        if tuple_index >= len(out):
            raise ValueError(
                f"tuple_index={tuple_index} out of bounds for {target_artifact.name} (len={len(out)})"
            )
        out = out[tuple_index]

    if not isinstance(out, dict) or "metrics" not in out:
        raise ValueError(
            f"Artifact {target_artifact.name} has no 'metrics' top-level key. "
            f"Found: {list(out.keys()) if isinstance(out, dict) else type(out).__name__}"
        )

    metrics = out["metrics"]
    if reduce_per_update:
        metrics = _maybe_reduce_partner_metrics(metrics)
    metrics = _to_numpy(metrics)

    if extra_top_level_keys:
        result: dict = {"metrics": metrics}
        for key in extra_top_level_keys:
            if key not in out:
                raise ValueError(
                    f"Artifact {target_artifact.name} missing requested extra key {key!r}. "
                    f"Top-level keys: {list(out.keys())}"
                )
            result[key] = _to_numpy(out[key])
    else:
        result = metrics

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  [cache] saved {artifact_kind} metrics to {cache_path}")
    return result


def _maybe_reduce_partner_metrics(metrics: dict) -> dict:
    """Reduce raw partner-training metrics to per-update scalars.

    Older IPPO-based partner artifacts (e.g. FCP cramped_room, run 4omnlent)
    saved `metrics` as the full `traj_batch.info` pytree with shape
    (NUM_SEEDS, PARTNER_POP_SIZE, NUM_UPDATES, ROLLOUT_LENGTH, NUM_ENVS).
    The reduction `mask_and_mean` (see `marl/ippo.py:255-259`) was applied
    *only* to the per-step scalar pushed through `io_callback` to wandb,
    not to the value scanned out of training. Newer artifacts apply it
    before scan-out and have shape (NUM_SEEDS, PARTNER_POP_SIZE, NUM_UPDATES).

    This helper detects the older form and applies the same
    mask/sum/divide collapse along the trailing two axes, so the cached
    pickle is small and shapes are uniform across runs.
    """
    rer = metrics.get("returned_episode_returns")
    if rer is None or not hasattr(rer, "ndim") or rer.ndim <= 3:
        return metrics  # already reduced

    rer_arr = np.asarray(rer)
    mask_field = metrics.get("returned_episode")
    if mask_field is None:
        # Without an explicit mask, fall back to plain mean. This is wrong on
        # partial episodes but at least won't blow up. Surface as a warning.
        print(f"  [warn] reducing partner metrics without 'returned_episode' mask — using plain mean")
        mask = np.ones(rer_arr.shape[-2:], dtype=np.float32)
    else:
        mask = np.asarray(mask_field).astype(np.float32)

    out = {}
    trailing = rer_arr.shape[-2:]
    mask_sum = np.maximum(1.0, mask.sum(axis=(-2, -1)))
    for k, v in metrics.items():
        if not hasattr(v, "shape"):
            out[k] = v
            continue
        v_arr = np.asarray(v)
        if v_arr.ndim > 3 and v_arr.shape[-2:] == trailing:
            out[k] = (v_arr * mask).sum(axis=(-2, -1)) / mask_sum
        else:
            out[k] = v_arr
    return out


def _to_numpy(tree):
    """Recursively convert a (possibly nested-dict) pytree of array-likes to numpy."""
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    if hasattr(tree, "__array__"):
        return np.asarray(tree)
    return tree


# ---------------------------------------------------------------------------
# Env-step axis
# ---------------------------------------------------------------------------

def make_env_steps_axis(num_updates: int, total_timesteps: int) -> np.ndarray:
    """Return an x-axis array of env steps, one entry per update.

    Convention: entry `i` is the cumulative env-step count *after* update `i`.
    So `axis[0] = total_timesteps / num_updates` (steps in update 0),
    `axis[-1] = total_timesteps` (full run).
    """
    if num_updates <= 0:
        return np.zeros((0,), dtype=np.int64)
    step_per_update = total_timesteps / num_updates
    return np.arange(1, num_updates + 1) * step_per_update


def get_config_value(run_config: dict, dotted_key: str, default=None):
    """Pull a nested value out of a wandb run config by dot-path."""
    cur = run_config
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# ---------------------------------------------------------------------------
# Per-seed curves — shared dataclass + factory used by every algo's fetch.py
# ---------------------------------------------------------------------------

@dataclass
class CurveData:
    """One per-seed training/eval curve.

    Convention:
      values: shape (num_seeds_or_seed_combos, num_steps).
      env_steps: shape (num_steps,) — same x-axis for every seed.
      n_segments: for curves built by concatenating per-iteration training
        passes (CoMeDi/COLE add one partner per iteration; ROTATE runs one
        open-ended iteration per confederate), the number of such segments.
        None for curves with no inner agent axis (e.g. TrajeDi, ego curves).
        Lets plotting code label the x-axis by teammate index rather than by
        raw env steps.
    """
    values: np.ndarray
    env_steps: np.ndarray
    n_segments: int | None = None


def make_curve(values_2d: np.ndarray, total_env_steps: int,
               n_segments: int | None = None) -> CurveData:
    """Build a CurveData from a (num_seeds, num_steps) array.

    Each algo's fetch.py reduces its raw metric tree down to this 2D shape
    (typically by meaning over partner-pop / eval / agent axes and flattening
    OEL iters). This helper attaches the env-step axis.
    """
    if values_2d.ndim != 2:
        raise ValueError(f"make_curve expects 2D (seeds, steps), got {values_2d.shape}")
    return CurveData(
        values=values_2d,
        env_steps=make_env_steps_axis(values_2d.shape[1], total_env_steps),
        n_segments=n_segments,
    )


def extract_ego_curve(ego_metrics: dict, total_env_steps: int) -> CurveData:
    """Per-seed ego training-return curve from an `ego_train_run` artifact.

    Used by FCP, BRDIV, LBRDIV, CoMeDi (via the partner+ego pipeline) and
    the standalone ego algos via `ego_common.py`. Handles both:
      - 2D shape (NUM_SEEDS, NUM_UPDATES) — standalone ego runs
      - 3D shape (NUM_SEEDS, NUM_EGO_TRAIN_SEEDS, NUM_UPDATES) — partner+ego pipeline
    """
    arr = np.asarray(ego_metrics["returned_episode_returns"])
    if arr.ndim == 2:
        per_seed = arr
    elif arr.ndim == 3:
        n_seeds, n_ego_seeds, n_updates = arr.shape
        per_seed = arr.reshape(n_seeds * n_ego_seeds, n_updates)
    else:
        raise ValueError(
            f"ego returned_episode_returns expected 2D or 3D, got {arr.shape}"
        )
    return make_curve(per_seed, total_env_steps)


def sp_xp_masks(pop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (sp_mask, xp_mask) over the `pop_size**2` flattened pair axis.

    BRDIV and LBRDIV both evaluate every (conf_id, br_id) pair in the
    population. SP pairs are those where `conf_id == br_id`; everything else
    is XP. Mirrors the masking in `BRDiv._get_all_ids` / `LBRDiv._get_all_ids`.
    """
    conf_ids = np.repeat(np.arange(pop_size), pop_size)
    br_ids = np.tile(np.arange(pop_size), pop_size)
    sp_mask = conf_ids == br_ids
    return sp_mask, ~sp_mask


# ---------------------------------------------------------------------------
# Wandb run-index sidecar
# ---------------------------------------------------------------------------

def task_to_safe_filename(task: str) -> str:
    """Plot file stem for a task. Reversed by `safe_filename_to_task`."""
    return task.replace("/", "__")


def safe_filename_to_task(safe: str) -> str:
    return safe.replace("__", "/")


EGO_DISPLAY_NAMES = {
    "ppo_ego": "PPO Ego",
    "liam_ego": "LIAM Ego",
    "meliba_ego": "MeLIBA Ego",
}


def update_wandb_run_index(
    out_dir: Path,
    new_entries: dict[str, dict] | None = None,
    entity: str = "aht-project",
    project: str = "aht-benchmark",
) -> None:
    """Update `out_dir/wandb_runs.json` (source of truth) and refresh
    `out_dir/wandb_runs.md` from it.

    The JSON sidecar is keyed by **plot filename** (so each row uniquely
    corresponds to a file on disk) and stores the human-readable label and
    wandb run id as the value. New entries are merged into the existing JSON
    so consecutive `run.py` invocations on different tasks accumulate.

    Args:
        out_dir: results directory containing the plots.
        new_entries: dict of {filename: {"label": ..., "run_id": ...}}
            produced by the latest run.py invocation. If None, just refresh
            the markdown from the JSON.
    """
    out_dir = Path(out_dir)
    json_path = out_dir / "wandb_runs.json"
    md_path = out_dir / "wandb_runs.md"

    entries: dict[str, dict] = {}
    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            raw = {}
        # Backwards-compat: old format was {task: run_id} (string value).
        # Migrate values that are bare strings into {"label": task, "run_id": v}.
        for k, v in raw.items():
            if isinstance(v, dict):
                entries[k] = v
            elif isinstance(v, str):
                fname = f"{task_to_safe_filename(k)}.png"
                entries[fname] = {"label": k, "run_id": v}
    if new_entries:
        entries.update(new_entries)

    json_path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Wandb run mapping",
        "",
        f"Auto-generated by `scripts/training_curves/{out_dir.name}/run.py`. "
        "Source of truth is `wandb_runs.json`; this file is regenerated from it.",
        "",
        f"Entity: `{entity}` · Project: `{project}`",
        "",
        "| Label | Plot | Wandb run |",
        "| --- | --- | --- |",
    ]
    for fname, meta in sorted(entries.items()):
        label = meta.get("label", fname)
        run_id = meta.get("run_id", "?")
        url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        lines.append(f"| {label} | `{fname}` | <{url}> |")
    lines.append("")

    md_path.write_text("\n".join(lines))
