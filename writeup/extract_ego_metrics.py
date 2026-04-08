#!/usr/bin/env python3
"""Pull eval-return curves out of orbax-saved ego training runs and
dump small JSONs under writeup/logs/.

PPO ego and LIAM ego runs save their full training state to
<run_dir>/ego_train_run via common.save_load_utils.save_train_run.
Eval returns live in metrics["eval_ep_last_info"]["returned_episode_returns"]
with shape (n_ego_seeds, n_updates, n_partners, n_eval_episodes, n_agents).

The orbax checkpoints are multi-GB and results/ is gitignored, so these
small JSONs are the committed evidence for every number in the writeup.

Run from repo root:
    .venv/bin/python writeup/extract_ego_metrics.py
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np  # noqa: E402

from common.save_load_utils import load_train_run  # noqa: E402

OUT_DIR = REPO / "writeup" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract(run_dir: Path) -> dict:
    """Reduce one ego training run to per-iter and per-seed JSON-friendly arrays.

    Per-iter mean curve averages across seeds, partners, eval episodes,
    and agents per game (axes 0, 2, 3, 4), matching the standard
    reduction in ppo_ego.log_metrics. Per-seed finals average over
    (2, 3, 4) and take the last update.

    Standalone ego_agent_training runs save with 5D shape
    (n_ego_seeds, n_updates, n_partners, n_eval, n_agents). The
    teammate_generation FCP pipeline trains an ego internally and saves
    with an extra leading dim for the partner pool seed; squashing the
    leading two dims gets both formats to the same per-seed view.
    """
    ckpt_dir = run_dir / "ego_train_run"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"missing orbax checkpoint at {ckpt_dir}")
    out = load_train_run(str(ckpt_dir))
    eli = out["metrics"]["eval_ep_last_info"]
    arr = np.asarray(eli["returned_episode_returns"])
    if arr.ndim == 6:
        n_pool, n_inner, n_updates, n_partners, n_eval, n_agents = arr.shape
        arr = arr.reshape(n_pool * n_inner, n_updates, n_partners, n_eval, n_agents)
    if arr.ndim != 5:
        raise ValueError(f"unexpected shape {arr.shape}; want 5D or 6D")
    n_seeds, n_updates, n_partners, n_eval, n_agents = arr.shape

    per_iter_mean = arr.mean(axis=(0, 2, 3, 4)).tolist()
    per_seed_curve = arr.mean(axis=(2, 3, 4)).tolist()  # (n_seeds, n_updates)
    per_seed_final = arr[:, -1, :, :, :].mean(axis=(1, 2, 3)).tolist()
    overall_final = float(arr[:, -1, :, :, :].mean())
    overall_max = float(per_iter_mean and max(per_iter_mean))

    return {
        "shape": list(arr.shape),
        "axes": [
            "n_ego_seeds",
            "n_updates",
            "n_partners",
            "n_eval_episodes",
            "n_agents_per_game",
        ],
        "n_ego_seeds": n_seeds,
        "n_updates": n_updates,
        "n_eval_episodes_per_partner": n_eval,
        "per_iter_mean": per_iter_mean,
        "per_seed_curve": per_seed_curve,
        "per_seed_final": per_seed_final,
        "overall_final": overall_final,
        "overall_max": overall_max,
    }


def write_json(name: str, payload: dict) -> Path:
    out_path = OUT_DIR / f"{name}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def discover_phase_runs() -> dict:
    """Walk results/dsse/ and pick up the most recent ego_train_run for
    every phase label, returning {algo_label: run_dir}.

    Picks up standalone ego_agent_training runs (phase_a/b/c/d labels and
    ego_*_fcp_* labels) plus the FCP teammate_generation pipeline's
    internal ego run under results/dsse/fcp/fcp_dsse_2drone_*/.
    """
    found = {}
    label_prefixes = ("phase_", "ego_ppo_fcp_", "ego_liam_fcp_")
    for algo_subdir in ("ppo_ego_s5", "liam_ego_mlp"):
        algo_root = REPO / "results" / "dsse" / algo_subdir
        if not algo_root.exists():
            continue
        for label_dir in sorted(algo_root.iterdir()):
            if not label_dir.is_dir():
                continue
            if not any(label_dir.name.startswith(p) for p in label_prefixes):
                continue
            timestamp_dirs = sorted(p for p in label_dir.iterdir() if p.is_dir())
            if not timestamp_dirs:
                continue
            run_dir = timestamp_dirs[-1]  # most recent
            if not (run_dir / "ego_train_run").exists():
                continue
            name = f"{algo_subdir}__{label_dir.name}"
            found[name] = run_dir

    # FCP teammate_generation pipeline trains its own ego in-place.
    fcp_root = REPO / "results" / "dsse" / "fcp"
    if fcp_root.exists():
        for label_dir in sorted(fcp_root.iterdir()):
            if not label_dir.is_dir():
                continue
            if not label_dir.name.startswith("fcp_dsse"):
                continue
            timestamp_dirs = sorted(p for p in label_dir.iterdir() if p.is_dir())
            if not timestamp_dirs:
                continue
            run_dir = timestamp_dirs[-1]
            if not (run_dir / "ego_train_run").exists():
                continue
            found[f"fcp_internal__{label_dir.name}"] = run_dir
    return found


def main():
    runs = {
        "dsse_ppo_ego_7x7_ndr2": REPO
        / "results/dsse/ppo_ego_s5/ego_ppo_7x7_staghunt_v2/2026-04-06_18-32-01",
        "dsse_liam_ego_7x7_ndr2": REPO
        / "results/dsse/liam_ego_mlp/ego_liam_7x7_staghunt_v2/2026-04-06_18-32-04",
        "dsse_ppo_ego_7x7_ndr1": REPO
        / "results/dsse/ppo_ego_s5/ego_ppo_7x7_baseline_v2/2026-04-06_18-57-01",
        "dsse_liam_ego_7x7_ndr1": REPO
        / "results/dsse/liam_ego_mlp/ego_liam_7x7_baseline_v2/2026-04-06_18-57-04",
    }
    runs.update(discover_phase_runs())
    summary = {}
    for name, path in runs.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        try:
            payload = extract(path)
        except Exception as exc:
            print(f"[fail] {name}: {exc}")
            continue
        out_path = write_json(name, payload)
        summary[name] = {
            "path": str(path.relative_to(REPO)),
            "n_ego_seeds": payload["n_ego_seeds"],
            "n_updates": payload["n_updates"],
            "per_seed_final": payload["per_seed_final"],
            "overall_final": payload["overall_final"],
            "overall_max": payload["overall_max"],
            "json": str(out_path.relative_to(REPO)),
        }
        print(
            f"[ok]   {name}: per_seed_final={payload['per_seed_final']} "
            f"overall_final={payload['overall_final']:.4f} "
            f"overall_max={payload['overall_max']:.4f} "
            f"-> {out_path.relative_to(REPO)}"
        )
    summary_path = OUT_DIR / "dsse_ego_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
