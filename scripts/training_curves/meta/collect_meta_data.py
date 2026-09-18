"""Collect everything needed for the meta-plot in `aggregate_2/`.

Single pickle output with multiple top-level sections:

    {
      'task': '<task>',
      'ego': {
          'ppo_ego':    {teammate_set: {env_steps, values, mean, std, n_seeds, run_id}, ...},
          'liam_ego':   {...},
          'meliba_ego': {...},
      },
      'teammate_curves': {     # mean over seeds + partners (from existing per-algo fetchers)
          'fcp':     {env_steps, values, mean, std, n_seeds, run_ids},
          'comedi':  {...},
          'cole':    {...},
          'trajedi': {...},
      },
      'fcp_partners': {env_steps, values: (n_partners, num_updates), n_partners, run_ids},
      'xp_matrices': {  # final cross-play matrix, mean over seeds
          'brdiv':  {matrix, pop_size, n_seeds, run_ids},
          'lbrdiv': {...},
          'cole':   {...},
      },
    }

Idempotent. Default task is LBF 12x12.

Run with:
    python -m scripts.training_curves.meta.collect_meta_data
    python -m scripts.training_curves.meta.collect_meta_data --task overcooked-v1/coord_ring
"""

from __future__ import annotations

import argparse
import datetime
import pickle
from pathlib import Path

import numpy as np

from scripts.training_curves.cole.fetch import fetch_cole_curves_for_task
from scripts.training_curves.comedi.fetch import fetch_comedi_curves_for_task
from scripts.training_curves.common import (
    DEFAULT_CACHE_DIR,
    extract_ego_curve,
    fetch_train_run_metrics_cached,
    find_benchmark_runs,
    get_config_value,
)
from scripts.training_curves.fcp.fetch import fetch_fcp_curves_for_task
from scripts.training_curves.rotate.fetch import fetch_rotate_curves_for_task
from scripts.training_curves.trajedi.fetch import fetch_trajedi_curves_for_task

# Artifacts live under results/ (gitignored); this module lives under scripts/
# so it is version-controlled. Path is repo-relative, matching
# `common.DEFAULT_CACHE_DIR`, so run from the repo root.
OUT_DIR = Path("results/figures/training_curves/aggregate_2")
OUT_PICKLE = OUT_DIR / "meta_data.pkl"
OUT_MD = OUT_DIR / "meta_data.md"

EGO_ALGOS = ("ppo_ego", "liam_ego", "meliba_ego")
TEAMMATE_CURVE_ALGOS = (
    ("fcp", fetch_fcp_curves_for_task, "partner"),
    ("comedi", fetch_comedi_curves_for_task, "partner"),
    ("cole", fetch_cole_curves_for_task, "partner"),
    ("trajedi", fetch_trajedi_curves_for_task, "train"),
)
XP_ALGOS = ("brdiv", "lbrdiv", "cole")


KNOWN_TEAMMATE_ALGOS = {"fcp", "brdiv", "lbrdiv", "comedi", "cole", "trajedi", "rotate"}


def teammate_set_from_run(run) -> str:
    """Return e.g. 'fcp' or 'comedi' based on the loaded partner_agent path.

    Path looks like
        partner_teammates/<task>/<teammate_gen_algo>/<label>/<timestamp>/...
    where `<task>` may itself contain a slash (e.g. `lbf/lbf_12x12`). Rather
    than guessing the offset, scan the path segments and return the first one
    that matches a known teammate-gen algorithm name.
    """
    pa = run.config.get("algorithm", {}).get("partner_agent", {})
    for info in pa.values():
        path = info.get("path", "") if isinstance(info, dict) else ""
        for seg in path.split("/"):
            if seg in KNOWN_TEAMMATE_ALGOS:
                return seg
    return "unknown"


# Loss-style metrics we want to capture for the ego algos in addition to
# the training return. LIAM/MeLIBA add autoencoder-related metrics on top of
# the standard PPO ones; the collector stores whichever are present.
LOSS_KEYS_OF_INTEREST = (
    "actor_loss",
    "value_loss",
    "entropy_loss",
    "avg_grad_norm",
    "reconstruction_loss",
    "kl_divergence_loss",
    "elbo_loss",
    "encoder_avg_grad_norm",
    "decoder_avg_grad_norm",
)


def collect_ego_stratified(task: str) -> dict:
    """For each ego algo, group runs by teammate-set and produce per-seed curves.

    Each teammate-set entry stores both the training return curve and any
    loss-style metrics that are present in the artifact's `metrics` tree
    (whichever of LOSS_KEYS_OF_INTEREST exist). Loss arrays have the same
    `(NUM_SEEDS, NUM_UPDATES)` shape as the return curve and share the same
    `env_steps` axis.
    """
    out: dict[str, dict] = {}
    for algo in EGO_ALGOS:
        runs = find_benchmark_runs(algorithm=algo, task=task)
        per_set: dict[str, dict] = {}
        for run in runs:
            teammate = teammate_set_from_run(run)
            metrics = fetch_train_run_metrics_cached(
                run,
                artifact_kind="ego_train_run",
                cache_dir=DEFAULT_CACHE_DIR,
                reduce_per_update=True,
            )
            total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
            curve = extract_ego_curve(metrics, total)
            values = np.asarray(curve.values)
            env_steps = np.asarray(curve.env_steps)

            # Capture loss-style metrics that are present and 2D (seeds, updates).
            losses: dict[str, np.ndarray] = {}
            for k in LOSS_KEYS_OF_INTEREST:
                if k in metrics:
                    arr = np.asarray(metrics[k])
                    if arr.ndim == 2 and arr.shape[1] == values.shape[1]:
                        losses[k] = arr

            if teammate in per_set:
                # Pool seeds across multiple runs for the same teammate set.
                prev = per_set[teammate]
                n = min(prev["values"].shape[1], values.shape[1])
                pooled = np.concatenate([prev["values"][:, :n], values[:, :n]], axis=0)
                # Pool losses keyed by the intersection of present keys.
                merged_losses: dict[str, np.ndarray] = {}
                for k in set(prev.get("losses", {})) & set(losses):
                    merged_losses[k] = np.concatenate(
                        [prev["losses"][k][:, :n], losses[k][:, :n]],
                        axis=0,
                    )
                per_set[teammate] = {
                    "env_steps": env_steps[:n],
                    "values": pooled,
                    "mean": pooled.mean(axis=0),
                    "std": pooled.std(axis=0),
                    "n_seeds": int(pooled.shape[0]),
                    "run_id": prev["run_id"] + "+" + run.id,
                    "losses": merged_losses,
                }
            else:
                per_set[teammate] = {
                    "env_steps": env_steps,
                    "values": values,
                    "mean": values.mean(axis=0),
                    "std": values.std(axis=0),
                    "n_seeds": int(values.shape[0]),
                    "run_id": run.id,
                    "losses": losses,
                }
        out[algo] = per_set
        all_loss_keys = sorted(
            {k for v in per_set.values() for k in v.get("losses", {})}
        )
        print(
            f"  ego/{algo}: teammates={sorted(per_set.keys())}  losses={all_loss_keys}"
        )
    return out


def collect_teammate_curves(task: str) -> dict:
    """Per-algo seed-pooled partner-training-return curves."""
    out: dict[str, dict] = {}
    for algo, fetcher, attr in TEAMMATE_CURVE_ALGOS:
        try:
            run_curves = fetcher(task=task, cache_dir=DEFAULT_CACHE_DIR)
        except ValueError as e:
            print(f"  teammate_curves/{algo}: skip ({e})")
            continue
        per_run_values: list[np.ndarray] = []
        env_steps_ref: np.ndarray | None = None
        run_ids: list[str] = []
        for rc in run_curves:
            cd = getattr(rc, attr)
            per_run_values.append(np.asarray(cd.values))
            run_ids.append(rc.run_id)
            if env_steps_ref is None:
                env_steps_ref = np.asarray(cd.env_steps)
        n_updates = min(v.shape[1] for v in per_run_values)
        per_run_values = [v[:, :n_updates] for v in per_run_values]
        env_steps_ref = env_steps_ref[:n_updates]
        pooled = np.concatenate(per_run_values, axis=0)
        out[algo] = {
            "env_steps": env_steps_ref,
            "values": pooled,
            "mean": pooled.mean(axis=0),
            "std": pooled.std(axis=0),
            "n_seeds": int(pooled.shape[0]),
            # Number of concatenated per-partner training passes, when the algo
            # has one (CoMeDi/COLE). None for flat curves like TrajeDi.
            "n_segments": getattr(run_curves[0], attr).n_segments,
            "run_ids": run_ids,
        }
        print(
            f"  teammate_curves/{algo}: {pooled.shape[0]} seeds, {n_updates} updates, "
            f"n_segments={out[algo]['n_segments']}"
        )
    return out


def _pool_curves(curves: list, run_ids: list[str]) -> dict:
    """Pool a list of per-run CurveData on the seed axis, truncated to the shortest."""
    values = [np.asarray(c.values) for c in curves]
    n_updates = min(v.shape[1] for v in values)
    pooled = np.concatenate([v[:, :n_updates] for v in values], axis=0)
    return {
        "env_steps": np.asarray(curves[0].env_steps)[:n_updates],
        "values": pooled,
        "mean": pooled.mean(axis=0),
        "std": pooled.std(axis=0),
        "n_seeds": int(pooled.shape[0]),
        "n_segments": curves[0].n_segments,
        "run_ids": run_ids,
    }


def collect_rotate(task: str) -> dict | None:
    """ROTATE ego-vs-confederate return and train-regret curves, seed-pooled.

    Both curves come from the same `saved_train_run` artifact but live on
    different x-axes: the return is per *ego* update, the regret is per
    *partner* update, and ROTATE spends a different env-step budget on each
    per open-ended iteration. They're stored separately for that reason.

    `train_regret` is absent from runs predating 2026-08-18; in that case the
    key is omitted and the plot renders an empty regret panel.
    """
    try:
        run_curves = fetch_rotate_curves_for_task(
            task=task, cache_dir=DEFAULT_CACHE_DIR
        )
    except ValueError as e:
        print(f"  rotate: skip ({e})")
        return None

    run_ids = [rc.run_id for rc in run_curves]
    out = {"return": _pool_curves([rc.ego_vs_conf for rc in run_curves], run_ids)}
    print(
        f"  rotate/return: {out['return']['n_seeds']} seeds, "
        f"{out['return']['values'].shape[1]} updates"
    )

    with_regret = [rc for rc in run_curves if rc.train_regret is not None]
    if with_regret:
        out["train_regret"] = _pool_curves(
            [rc.train_regret for rc in with_regret],
            [rc.run_id for rc in with_regret],
        )
        print(
            f"  rotate/train_regret: {out['train_regret']['n_seeds']} seeds, "
            f"{out['train_regret']['values'].shape[1]} updates"
        )
    else:
        print("  rotate/train_regret: unavailable (runs predate regret logging)")
    return out


def collect_fcp_partners(task: str) -> dict | None:
    """FCP per-partner training-return curves, flattened across (seeds, pop)."""
    runs = find_benchmark_runs(algorithm="fcp", task=task)
    if not runs:
        return None
    per_run: list[np.ndarray] = []
    env_steps_ref: np.ndarray | None = None
    run_ids: list[str] = []
    for run in runs:
        metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            cache_dir=DEFAULT_CACHE_DIR,
            reduce_per_update=True,
        )
        arr = np.asarray(metrics["returned_episode_returns"])  # (seeds, pop, updates)
        per_run.append(arr)
        run_ids.append(run.id)
        total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
        es = np.arange(1, arr.shape[-1] + 1) * (total / arr.shape[-1])
        if env_steps_ref is None:
            env_steps_ref = es
    n_updates = min(a.shape[-1] for a in per_run)
    flat = []
    for a in per_run:
        a = a[..., :n_updates]
        n_s, pop, _ = a.shape
        flat.append(a.reshape(n_s * pop, n_updates))
    pooled = np.concatenate(flat, axis=0)
    print(f"  fcp_partners: {pooled.shape[0]} partner curves, {n_updates} updates")
    return {
        "env_steps": env_steps_ref[:n_updates],
        "values": pooled,
        "n_partners": int(pooled.shape[0]),
        "run_ids": run_ids,
    }


def collect_xp_matrix(algo: str, task: str) -> dict | None:
    """Final XP matrix from one teammate-gen algo's eval_ep_last_info."""
    runs = find_benchmark_runs(algorithm=algo, task=task)
    if not runs:
        return None
    per_seed: list[np.ndarray] = []
    run_ids: list[str] = []
    for run in runs:
        if not any("saved_train_run" in a.name for a in run.logged_artifacts()):
            continue
        if algo == "cole":
            # COLE saves the final_xp_matrix as a top-level key.
            res = fetch_train_run_metrics_cached(
                run,
                artifact_kind="saved_train_run",
                cache_dir=DEFAULT_CACHE_DIR,
                reduce_per_update=True,
                extra_top_level_keys=("final_xp_matrix",),
            )
            xp = np.asarray(res["final_xp_matrix"])  # (n_seeds, pop, pop)
        else:
            metrics = fetch_train_run_metrics_cached(
                run,
                artifact_kind="saved_train_run",
                cache_dir=DEFAULT_CACHE_DIR,
                reduce_per_update=True,
            )
            arr = np.asarray(metrics["eval_ep_last_info"]["returned_episode_returns"])
            # (seeds, updates, pop^2, eps, agents) → final + mean over (eps, agents)
            n_seeds, _, n_pairs, _, _ = arr.shape
            pop = round(n_pairs**0.5)
            last = arr[:, -1].mean(axis=(-2, -1))  # (seeds, pop²)
            xp = last.reshape(n_seeds, pop, pop)  # (seeds, pop, pop)
        per_seed.append(xp)
        run_ids.append(run.id)
    if not per_seed:
        return None
    pooled = np.concatenate(per_seed, axis=0)
    print(f"  xp/{algo}: {pooled.shape[0]} seeds, pop_size={pooled.shape[-1]}")
    return {
        "matrix": pooled.mean(axis=0),  # (pop, pop)
        "pop_size": int(pooled.shape[-1]),
        "n_seeds": int(pooled.shape[0]),
        "run_ids": run_ids,
    }


def collect_lbrdiv_lms(task: str) -> dict | None:
    """Per-(i,j) Lagrange-multiplier curves, mean over seeds (matches our LBRDIV plot).

    `lms_horizontal` / `lms_vertical` have shape
        (NUM_SEEDS, NUM_PARTNER_UPDATES, POP_SIZE, POP_SIZE)
    Mean over seeds, then flatten the (pop, pop) → pop² pair axis to give
    one curve per pair: shape (pop², num_updates).
    """
    runs = find_benchmark_runs(algorithm="lbrdiv", task=task)
    if not runs:
        return None
    h_per_run: list[np.ndarray] = []
    v_per_run: list[np.ndarray] = []
    env_steps_ref: np.ndarray | None = None
    run_ids: list[str] = []
    for run in runs:
        metrics = fetch_train_run_metrics_cached(
            run,
            artifact_kind="saved_train_run",
            cache_dir=DEFAULT_CACHE_DIR,
            reduce_per_update=True,
        )
        if "lms_horizontal" not in metrics or "lms_vertical" not in metrics:
            print(f"  lbrdiv_lms/{run.id}: no LMs in metric tree; skipping")
            continue
        h = np.asarray(metrics["lms_horizontal"])  # (seeds, updates, pop, pop)
        v = np.asarray(metrics["lms_vertical"])
        h_per_run.append(h)
        v_per_run.append(v)
        run_ids.append(run.id)
        total = get_config_value(run.config, "algorithm.TOTAL_TIMESTEPS")
        es = np.arange(1, h.shape[1] + 1) * (total / h.shape[1])
        if env_steps_ref is None:
            env_steps_ref = es
    if not h_per_run:
        return None
    n_updates = min(a.shape[1] for a in h_per_run)
    h_pooled = np.concatenate([a[:, :n_updates] for a in h_per_run], axis=0)
    v_pooled = np.concatenate([a[:, :n_updates] for a in v_per_run], axis=0)
    h_seed_mean = h_pooled.mean(axis=0)  # (updates, pop, pop)
    v_seed_mean = v_pooled.mean(axis=0)
    n_updates, pop, _ = h_seed_mean.shape
    h_flat = h_seed_mean.reshape(n_updates, pop * pop).T  # (pop², updates)
    v_flat = v_seed_mean.reshape(n_updates, pop * pop).T
    pair_labels = [f"({i},{j})" for i in range(pop) for j in range(pop)]
    print(
        f"  lbrdiv_lms: pop_size={pop}, {n_updates} updates, n_seeds={h_pooled.shape[0]}"
    )
    return {
        "env_steps": env_steps_ref[:n_updates],
        "horizontal": h_flat,
        "vertical": v_flat,
        "pair_labels": pair_labels,
        "pop_size": int(pop),
        "n_seeds": int(h_pooled.shape[0]),
        "run_ids": run_ids,
    }


def write_manifest(data: dict, md_path: Path) -> None:
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    lines = [
        "# Meta-data manifest (`aggregate_2/meta_data.pkl`)",
        "",
        f"Generated: {today}  ·  task: `{data['task']}`",
        "",
        (
            "Reproducible: re-run `python -m "
            "scripts.training_curves.meta.collect_meta_data`."
        ),
        "",
        "## Sections",
        "",
        "- `task`: the task this snapshot is for.",
        "- `ego[<algo>][<teammate_set>]`: stratified per-seed ego training-return curves.",
        "- `teammate_curves[<algo>]`: seed-pooled partner-training-return curves.",
        (
            "- `fcp_partners`: FCP per-partner training-return curves "
            "  (rows are individual partners, not seed means)."
        ),
        "- `xp_matrices[<algo>]`: final cross-play matrix (seed mean) for brdiv / lbrdiv / cole.",
        (
            "- `rotate['return' | 'train_regret']`: seed-pooled ROTATE curves "
            "(separate x-axes: ego updates vs partner updates)."
        ),
        "",
        "## Coverage",
        "",
        "### Ego (stratified by teammate set)",
        "",
    ]
    for algo, by_set in data["ego"].items():
        for ts, e in by_set.items():
            lines.append(
                f"- {algo} / {ts}: n_seeds={e['n_seeds']}, run_id={e['run_id']}"
            )
    lines += ["", "### Teammate curves", ""]
    for algo, e in data["teammate_curves"].items():
        lines.append(f"- {algo}: n_seeds={e['n_seeds']}, run_ids={e['run_ids']}")
    if data.get("rotate"):
        lines += ["", "### ROTATE", ""]
        for key, e in data["rotate"].items():
            lines.append(f"- {key}: n_seeds={e['n_seeds']}, run_ids={e['run_ids']}")
    if data.get("fcp_partners"):
        e = data["fcp_partners"]
        lines += [
            "",
            "### FCP partners",
            "",
            f"- n_partners={e['n_partners']}, run_ids={e['run_ids']}",
        ]
    lines += ["", "### XP matrices", ""]
    for algo, e in data["xp_matrices"].items():
        lines.append(
            f"- {algo}: pop_size={e['pop_size']}, n_seeds={e['n_seeds']}, "
            f"run_ids={e['run_ids']}"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="lbf/lbf_12x12")
    args = p.parse_args()

    task = args.task
    print(f"Collecting meta-data for task={task!r}")

    data: dict = {"task": task}

    print("\n[ego stratified]")
    data["ego"] = collect_ego_stratified(task)

    print("\n[teammate-gen seed-pooled curves]")
    data["teammate_curves"] = collect_teammate_curves(task)

    print("\n[ROTATE return + train regret]")
    data["rotate"] = collect_rotate(task)

    print("\n[FCP per-partner overlay]")
    fcp = collect_fcp_partners(task)
    if fcp is not None:
        data["fcp_partners"] = fcp

    print("\n[XP matrices]")
    data["xp_matrices"] = {}
    for algo in XP_ALGOS:
        entry = collect_xp_matrix(algo, task)
        if entry is not None:
            data["xp_matrices"][algo] = entry

    print("\n[LBRDIV Lagrange multipliers]")
    data["lbrdiv_lms"] = collect_lbrdiv_lms(task)

    OUT_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(data, f)
    print(f"\nwrote {OUT_PICKLE}  ({OUT_PICKLE.stat().st_size / 1e6:.2f} MB)")

    write_manifest(data, OUT_MD)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
