#!/usr/bin/env python3
"""Phase A decision gate.

Reads the three Phase A run dirs (PPO ego, LIAM trained, LIAM frozen),
extracts the per-seed final return for each, and prints a clean verdict.

Decision rule (from scripts/run_phase_a_validity.sh):
  - if mean(LIAM frozen) >= mean(LIAM trained) - 0.5*std(LIAM trained):
        verdict = EXPLORATION_BOUND  -> stop and redesign env
  - else:
        verdict = COORDINATION_SIGNAL -> proceed to Phase B/C
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from common.save_load_utils import load_train_run  # noqa: E402

CONDITIONS = {
    "ppo_ego":     "results/dsse/ppo_ego_s5/phase_a_ppo_ego_5seed",
    "liam_trained": "results/dsse/liam_ego_mlp/phase_a_liam_trained_5seed",
    "liam_frozen":  "results/dsse/liam_ego_mlp/phase_a_liam_frozen_5seed",
}


def latest_run(label_dir: Path) -> Path:
    if not label_dir.exists():
        raise FileNotFoundError(f"label dir missing: {label_dir}")
    timestamps = sorted(p for p in label_dir.iterdir() if p.is_dir())
    if not timestamps:
        raise FileNotFoundError(f"no timestamped runs under {label_dir}")
    return timestamps[-1]


def extract_final_returns(run_dir: Path) -> np.ndarray:
    """Per-seed final return: mean over (partners, eval_eps, agents)."""
    ckpt = run_dir / "ego_train_run"
    out = load_train_run(str(ckpt))
    arr = np.asarray(out["metrics"]["eval_ep_last_info"]["returned_episode_returns"])
    # arr shape: (n_seeds, n_updates, n_partners, n_eval_eps, n_agents)
    final = arr[:, -1, :, :, :].mean(axis=(1, 2, 3))  # -> (n_seeds,)
    return final


def main():
    results = {}
    for name, rel in CONDITIONS.items():
        try:
            run_dir = latest_run(REPO / rel)
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}")
            continue
        finals = extract_final_returns(run_dir)
        results[name] = {
            "run": str(run_dir.relative_to(REPO)),
            "per_seed": finals.tolist(),
            "mean": float(finals.mean()),
            "std": float(finals.std(ddof=0)),
            "n": int(finals.size),
        }

    print("\n=== Phase A: validity ablation results ===")
    for name, r in results.items():
        per_seed_str = ", ".join(f"{x:.4f}" for x in r["per_seed"])
        print(f"  {name:14s} mean={r['mean']:.4f} std={r['std']:.4f}  per_seed=[{per_seed_str}]")

    if not all(k in results for k in ("ppo_ego", "liam_trained", "liam_frozen")):
        print("\n[abort] missing one or more conditions; can't decide")
        sys.exit(2)

    ppo = results["ppo_ego"]
    trained = results["liam_trained"]
    frozen = results["liam_frozen"]

    gap_full = trained["mean"] - ppo["mean"]
    gap_encoder = trained["mean"] - frozen["mean"]
    gap_capacity = frozen["mean"] - ppo["mean"]
    threshold = trained["mean"] - 0.5 * trained["std"]

    print(f"\n  LIAM trained > PPO       : {gap_full:+.4f}  (full LIAM gap)")
    print(f"  LIAM trained > LIAM frozen: {gap_encoder:+.4f}  (encoder learning contribution)")
    print(f"  LIAM frozen  > PPO        : {gap_capacity:+.4f}  (capacity / random feature contribution)")
    print(f"  threshold                 : trained.mean - 0.5*trained.std = {threshold:.4f}")

    if frozen["mean"] >= threshold:
        verdict = "EXPLORATION_BOUND_OR_CAPACITY"
        next_step = "STOP. The LIAM > PPO gap is mostly explained by extra capacity / random features, NOT by encoder learning. Discuss env redesign before launching Phase B/C."
    else:
        verdict = "COORDINATION_SIGNAL"
        next_step = "PROCEED to Phase B (10 seeds) and Phase C (3-partner population). Encoder learning is contributing real signal."

    print(f"\n  VERDICT: {verdict}")
    print(f"  NEXT  : {next_step}")

    # Save a small JSON for the writeup
    out_path = REPO / "writeup" / "logs" / "phase_a_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": results,
        "gap_full": gap_full,
        "gap_encoder": gap_encoder,
        "gap_capacity": gap_capacity,
        "threshold": threshold,
        "verdict": verdict,
        "next_step": next_step,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path.relative_to(REPO)}")

    sys.exit(0 if verdict == "COORDINATION_SIGNAL" else 1)


if __name__ == "__main__":
    main()
