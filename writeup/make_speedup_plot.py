#!/usr/bin/env python3
"""Generate speedup comparison + JAX NUM_ENVS scaling plots."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Measured data ----
# PyTorch NAHT, 500K env steps, 7x7, 4 drones, 2 targets, ndr=2, IPPO
# From naht_results/gpu_experiments/dsse_ippo_s{42,123,456}.log wall-clocks.
PYTORCH_IPPO_SEC = [367, 363, 363]  # s42, s123, s456 self-play 500K
PYTORCH_POAM_SEC = [448, 440, 440]  # s42, s123, s456 self-play 500K

# JAX jax-aht, 500K env steps, same config, NUM_ENVS=64, seed=42
JAX_IPPO_500K = {16: 20.9, 64: 18.2, 128: 18.9}

# JAX NUM_ENVS scaling, 100K env steps
JAX_SCALE_100K = {1: 23.9, 4: 18.5, 16: 17.4, 64: 17.7, 128: 18.1, 256: 19.5, 512: 22.7}


def plot_speedup():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)

    pt_ippo_mean = float(np.mean(PYTORCH_IPPO_SEC))
    pt_poam_mean = float(np.mean(PYTORCH_POAM_SEC))
    jax_ippo_best = min(JAX_IPPO_500K.values())
    jax_ippo_mean = float(np.mean(list(JAX_IPPO_500K.values())))

    labels = ["PyTorch\nIPPO", "PyTorch\nPOAM", "JAX IPPO\n(NUM_ENVS=64)"]
    times = [pt_ippo_mean, pt_poam_mean, JAX_IPPO_500K[64]]
    colors = ["#d47500", "#d47500", "#2066a8"]

    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel("Wall-clock time for 500K env steps (s)")
    ax.set_title("DSSE training throughput: PyTorch NAHT vs JAX JaxAHT\n7×7 grid, 4 drones, 2 targets, ndr=2")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    # Add headroom above the tallest bar so the speedup annotation
    # has somewhere to live without overlapping anything.
    ax.set_ylim(top=max(times) * 5.0)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.15,
                f"{t:.0f}s", ha="center", fontsize=10, fontweight="bold")

    speedup = pt_ippo_mean / JAX_IPPO_500K[64]
    ax.text(0.5, 0.97, f"JAX is {speedup:.0f}× faster than PyTorch (IPPO, same config)",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="black"))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "speedup_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  speedup = {speedup:.1f}x  (PyTorch IPPO {pt_ippo_mean:.0f}s / JAX IPPO {JAX_IPPO_500K[64]:.1f}s)")


def plot_scaling():
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    nenvs = sorted(JAX_SCALE_100K.keys())
    times = [JAX_SCALE_100K[n] for n in nenvs]
    throughput = [100_000 / t for t in times]

    ax2 = ax.twinx()
    ln1 = ax.plot(nenvs, times, "o-", color="#d47500", linewidth=2,
                  markersize=8, label="Wall-clock time (s)")
    ln2 = ax2.plot(nenvs, throughput, "s--", color="#2066a8", linewidth=2,
                   markersize=7, label="Throughput (steps/s)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("NUM_ENVS (vectorised via jax.vmap)")
    ax.set_ylabel("Wall-clock time for 100K steps (s)", color="#d47500")
    ax2.set_ylabel("Effective throughput (env steps / s)", color="#2066a8")
    ax.set_title("DSSE-JAX IPPO scaling: 100K env steps, 7×7, 4 drones, ndr=2")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nenvs)
    ax.set_xticklabels([str(n) for n in nenvs])
    ax.tick_params(axis="y", labelcolor="#d47500")
    ax2.tick_params(axis="y", labelcolor="#2066a8")

    lines = ln1 + ln2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=True)

    ax.text(0.02, 0.98,
            "Flat wall-clock across 1 to 512 envs is expected:\n"
            "jax.vmap fuses the vectorised step into one\n"
            "GPU kernel, so per-env cost is near-zero.\n"
            "JIT compile dominates at short horizons.",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "jax_scaling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_speedup()
    plot_scaling()
