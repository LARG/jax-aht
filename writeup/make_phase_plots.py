#!/usr/bin/env python3
"""Generate JAX-side phase result plots for the DSSE writeup.

Reads `writeup/logs/phase_results_aggregate.json` (produced by
`scripts/aggregate_phase_results.py`) and writes:

  - plots/phase_a_validity.png      Phase A coordination-signal ablation
  - plots/phase_b_ndr1_baseline.png Phase B easy-mode (ndr=1) sanity check
  - plots/phase_d_heldout_gap.png   Phase D trained vs held-out generalization
  - plots/phase_overview.png        Combined 3-panel figure for the writeup

Run from repo root:
    .venv/bin/python writeup/make_phase_plots.py
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "writeup" / "logs"
PLOTS = REPO / "writeup" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PPO_COLOR = "#d47500"
LIAM_COLOR = "#2066a8"
FROZEN_COLOR = "#7a7a7a"


def load_aggregate():
    path = LOGS / "phase_results_aggregate.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(REPO)}; run scripts/aggregate_phase_results.py first"
        )
    with path.open() as f:
        return json.load(f)


def annotate_bar(ax, x, value, std, color, fmt="{:.3f}"):
    label = fmt.format(value)
    if std is not None and std > 0:
        label += f"\n±{fmt.format(std)}"
    ax.text(x, value + (std or 0) + 0.015, label,
            ha="center", va="bottom", fontsize=9, color=color)


def plot_phase_a(agg, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=120)

    rows = [
        ("PPO ego\n(no encoder)",      agg["phase_a_ppo_ego_5seed"],      PPO_COLOR),
        ("LIAM trained\n(full)",        agg["phase_a_liam_trained_5seed"], LIAM_COLOR),
        ("LIAM frozen\n(encoder ablated)", agg["phase_a_liam_frozen_5seed"], FROZEN_COLOR),
    ]
    xs = np.arange(len(rows))
    means = [r[1]["mean"] for r in rows]
    stds = [r[1]["std"] for r in rows]
    colors = [r[2] for r in rows]
    labels = [r[0] for r in rows]

    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.5)
    for x, m, s, c in zip(xs, means, stds, colors):
        annotate_bar(ax, x, m, s, c)

    # Scatter per-seed
    for x, (_, d, color) in zip(xs, rows):
        for v in d["per_seed"]:
            ax.scatter(x, v, color="black", s=18, alpha=0.45, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean episode return (5 seeds)")
    ax.set_title("Phase A: coordination-signal ablation\n(7×7, n_drones_to_rescue=2, 1 partner)")
    ax.set_ylim(0, max(means) + max(stds) + 0.18)
    ax.grid(True, axis="y", alpha=0.3)

    if own_fig:
        fig.tight_layout()
        out = PLOTS / "phase_a_validity.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}")


def plot_phase_b(agg, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=120)

    rows = [
        ("PPO ego",  agg["phase_b_ppo_ndr1_10seed"],  PPO_COLOR),
        ("LIAM ego", agg["phase_b_liam_ndr1_10seed"], LIAM_COLOR),
    ]
    xs = np.arange(len(rows))
    means = [r[1]["mean"] for r in rows]
    stds = [r[1]["std"] for r in rows]
    colors = [r[2] for r in rows]
    labels = [r[0] for r in rows]

    ax.bar(xs, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.5)
    for x, m, s, c in zip(xs, means, stds, colors):
        annotate_bar(ax, x, m, s, c, fmt="{:.3f}")

    for x, (_, d, _) in zip(xs, rows):
        for v in d["per_seed"]:
            ax.scatter(x, v, color="black", s=18, alpha=0.45, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean episode return (10 seeds)")
    ax.set_title("Phase B: easy-mode sanity\n(7×7, n_drones_to_rescue=1, 1 partner)")
    ax.set_ylim(0, max(means) + max(stds) + 0.25)
    ax.grid(True, axis="y", alpha=0.3)

    if own_fig:
        fig.tight_layout()
        out = PLOTS / "phase_b_ndr1_baseline.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}")


def plot_phase_d(agg, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=120)

    rows = [
        ("PPO ego",  agg["phase_d_ppo_pop3_heldout"],  PPO_COLOR),
        ("LIAM ego", agg["phase_d_liam_pop3_heldout"], LIAM_COLOR),
    ]
    xs = np.arange(len(rows))
    width = 0.35

    trained_means = [r[1]["mean"] for r in rows]
    trained_stds = [r[1]["std"] for r in rows]
    heldout_means = [r[1]["heldout_mean"] or 0.0 for r in rows]
    heldout_stds = [r[1]["heldout_std"] or 0.0 for r in rows]
    colors = [r[2] for r in rows]
    labels = [r[0] for r in rows]

    bars_train = ax.bar(xs - width / 2, trained_means, width, yerr=trained_stds,
                        capsize=5, color=colors, edgecolor="black", linewidth=0.5,
                        label="Trained partners")
    bars_held = ax.bar(xs + width / 2, heldout_means, width, yerr=heldout_stds,
                       capsize=5, color=colors, alpha=0.45, edgecolor="black",
                       linewidth=0.5, hatch="//", label="Held-out partners")

    for x, m, s in zip(xs - width / 2, trained_means, trained_stds):
        ax.text(x, m + s + 0.003, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    for x, m, s in zip(xs + width / 2, heldout_means, heldout_stds):
        ax.text(x, m + s + 0.003, f"{m:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean episode return (10 seeds)")
    ax.set_title("Phase D: held-out partner generalization\n(7×7, ndr=2, 3-partner population)")
    ymax = max(trained_means + heldout_means) + max(trained_stds + heldout_stds) + 0.02
    ax.set_ylim(0, max(ymax, 0.05))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    if own_fig:
        fig.tight_layout()
        out = PLOTS / "phase_d_heldout_gap.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}")


def plot_overview(agg):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)
    plot_phase_a(agg, ax=axes[0])
    plot_phase_b(agg, ax=axes[1])
    plot_phase_d(agg, ax=axes[2])
    fig.suptitle(
        "DSSE JAX AHT validity & generalization (Phases A, B, D)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "phase_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def main():
    agg = load_aggregate()
    plot_phase_a(agg)
    plot_phase_b(agg)
    plot_phase_d(agg)
    plot_overview(agg)


if __name__ == "__main__":
    main()
