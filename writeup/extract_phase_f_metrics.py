#!/usr/bin/env python3
"""Extract held-out eval metrics for the BRDiv / CoMeDi / LBRDiv
integration check, write a JSON summary, and print a one-line
per-method breakdown.

Looks at the most recent pr_long run for each method, loads
heldout_eval_metrics via common.save_load_utils.load_train_run,
averages over seeds and episodes, and dumps a structured summary
at writeup/logs/phase_f_results.json so the writeup table can
regenerate without rerunning the eval.

Expected returned_episode_returns shape is
(NUM_SEEDS, n_heldout_partners, NUM_EVAL_EPISODES, n_agents);
for the integration-check budget this is (1, 6, 64, 2).

Usage:
    PYTHONPATH=. python writeup/extract_phase_f_metrics.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np

from common.save_load_utils import load_train_run

PARTNER_NAMES = [
    "ippo_seed123_pop3_0",
    "ippo_seed123_pop3_1",
    "ippo_seed123_pop3_2",
    "random_agent",
    "greedy_search_agent",
    "sweep_agent",
]

METHODS = ["brdiv", "comedi", "lbrdiv"]


def latest_run_dir(method: str, label: str) -> Optional[str]:
    pattern = f"results/dsse/{method}/{label}/*/"
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1].rstrip("/")


def summarise(method: str, run_dir: str) -> dict:
    metrics_dir = os.path.join(run_dir, "heldout_eval_metrics")
    metrics = load_train_run(metrics_dir)
    arr = np.array(metrics["returned_episode_returns"])
    if arr.ndim != 4:
        raise ValueError(
            f"{method}: expected returned_episode_returns to be 4D "
            f"(NUM_SEEDS, num_partners, NUM_EVAL_EPISODES, num_agents); "
            f"got shape {arr.shape}"
        )

    per_partner = arr.mean(axis=(0, 2, 3))
    if per_partner.shape[0] != len(PARTNER_NAMES):
        raise ValueError(
            f"{method}: expected {len(PARTNER_NAMES)} held-out partners, "
            f"got {per_partner.shape[0]}"
        )

    ippo_mean = float(per_partner[:3].mean())
    heuristics_mean = float(per_partner[3:].mean())
    overall = float(per_partner.mean())

    targets_found = np.array(metrics.get("targets_found", np.zeros_like(arr)))
    all_found = np.array(metrics.get("all_found", np.zeros_like(arr)))
    nonzero_episodes = int((arr != 0).sum())

    return {
        "method": method,
        "run_dir": run_dir,
        "shape": list(arr.shape),
        "per_partner": {
            name: float(per_partner[i]) for i, name in enumerate(PARTNER_NAMES)
        },
        "ippo_mean": ippo_mean,
        "heuristics_mean": heuristics_mean,
        "overall": overall,
        "diagnostics": {
            "nonzero_episodes": nonzero_episodes,
            "total_episodes": int(arr.size),
            "targets_found_total": int(targets_found.sum()),
            "all_found_total": int(all_found.sum()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="pr_long",
        help="run label to look up under results/dsse/<method>/<label>/",
    )
    parser.add_argument(
        "--out",
        default="writeup/logs/phase_f_results.json",
        help="path to write the JSON summary",
    )
    args = parser.parse_args()

    summaries = []
    for method in METHODS:
        run_dir = latest_run_dir(method, args.label)
        if run_dir is None:
            print(
                f"[WARN] {method}: no run dir found at "
                f"results/dsse/{method}/{args.label}/*/  (skipping)"
            )
            continue
        try:
            s = summarise(method, run_dir)
        except Exception as e:
            print(f"[FAIL] {method}: {e}")
            continue

        summaries.append(s)
        print(
            f"{method:7s} ippo={s['ippo_mean']:.4f} "
            f"heur={s['heuristics_mean']:.4f} "
            f"overall={s['overall']:.4f} "
            f"per_partner={[round(s['per_partner'][n], 4) for n in PARTNER_NAMES]}"
        )
        diag = s["diagnostics"]
        print(
            f"        nonzero_episodes={diag['nonzero_episodes']}/"
            f"{diag['total_episodes']} "
            f"targets_found_total={diag['targets_found_total']} "
            f"all_found_total={diag['all_found_total']}"
        )

    if summaries:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {"label": args.label, "methods": summaries},
                f,
                indent=2,
            )
        print(f"\nWrote summary to {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
