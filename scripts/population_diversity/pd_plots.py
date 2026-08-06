# PD scalar math (det(K), PCA) and figure savers (heatmap, PCA scatter).
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)


def population_diversity(theta: np.ndarray) -> Dict[str, Any]:
    """population diversity = det(k) on per-feature max-normalized theta."""
    # normalize per feature like ZSC-Eval
    col_max = theta.max(axis=0, keepdims=True)
    theta_norm = theta / (col_max + 1e-3)

    K = theta_norm @ theta_norm.T
    # det(K), the ZSC-Eval PD metric
    det = float(np.linalg.det(K))
    # log|det| + sign for our analysis, not the zsc-eval value
    sign, logdet = np.linalg.slogdet(K)

    n_agents, n_features = theta_norm.shape
    rank = int(np.linalg.matrix_rank(theta_norm))
    eigvals = np.linalg.eigvalsh(K)
    degenerate = rank < n_agents
    if degenerate:
        log.warning(
            "PD det(K) is degenerate: rank(theta)=%d < N=%d (D=%d); det is ~0/undefined. "
            "Use rank / cosine / PCA, not det_K.",
            rank, n_agents, n_features,
        )

    # cosine sim for the heatmaps, on normalized theta
    norms = np.linalg.norm(theta_norm, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    unit = theta_norm / norms_safe
    cosine = unit @ unit.T

    return {
        "det_K": det,
        "log_det_K": float(logdet),
        "sign": int(sign),
        "n_agents": n_agents,
        "n_features": n_features,
        "rank": rank,
        "degenerate": bool(degenerate),
        "eigvals_K": eigvals.tolist(),
        "K": K.tolist(),
        "cosine": cosine.tolist(),
        "unit_theta": unit.tolist(),
        # raw and normalized theta kept separate
        "theta_raw": theta.tolist(),
        "theta_norm": theta_norm.tolist(),
        "col_max": col_max.ravel().tolist(),
    }


def pca_2d(theta: np.ndarray) -> np.ndarray:
    """center theta and project onto the top 2 principal components."""
    centered = theta - theta.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def save_pca_plot(theta, names, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    coords = pca_2d(theta)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    def family_of(name):
        n = name.lower()
        if any(h in n for h in ["iggi","piers","flawed","outer","van_den_bergh","smartbot","cautious","internal","entitled","greedy","selfish","random_agent","heuristic","onion","plate","independent"]):
            return "heuristic"
        if "obl_r2d2" in n: return "obl_r2d2"
        if "bc_lstm" in n or "hdr" in n: return "human_proxy"
        if any(p in n for p in ["brdiv","lbrdiv","comedi","trajedi","cole"]): return "rl_pop"
        if "ippo" in n or "fcp" in n: return "rl_sp"
        return "other"

    family_colors = {
        "heuristic":    "#d62728",
        "rl_pop":       "#2ca02c",
        "rl_sp":        "#1f77b4",
        "human_proxy":  "#ff7f0e",
        "obl_r2d2":     "#9467bd",
        "other":        "#7f7f7f",
    }
    families = [family_of(n) for n in names]
    colors = [family_colors[f] for f in families]

    ax.scatter(coords[:, 0], coords[:, 1], s=140, alpha=0.85,
               edgecolor="k", linewidth=1.0, c=colors, zorder=3)

    texts = []
    for i, name in enumerate(names):
        texts.append(ax.text(
            coords[i, 0], coords[i, 1],
            name.replace("_serious", "").replace("_long_6e7", ""),
            fontsize=7, alpha=0.9, zorder=4,
        ))
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5, alpha=0.6),
            expand_points=(1.2, 1.4),
            expand_text=(1.05, 1.2),
            force_points=0.5,
            force_text=0.6,
        )
    except ImportError:
        pass

    used = []
    for f in families:
        if f not in used:
            used.append(f)
    handles = [mpatches.Patch(color=family_colors[f], label=f.replace("_", " ")) for f in used]
    ax.legend(handles=handles, loc="best", frameon=True, edgecolor="0.7", fancybox=False)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_cosine_heatmap(cosine, names, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    def family_of(name):
        n = name.lower()
        if any(h in n for h in ["iggi","piers","flawed","outer","van_den_bergh","smartbot","cautious","internal","entitled","greedy","selfish","random_agent","heuristic","onion","plate","independent"]):
            return "heuristic"
        if "obl_r2d2" in n: return "obl_r2d2"
        if "bc_lstm" in n or "hdr" in n: return "human_proxy"
        if any(p in n for p in ["brdiv","lbrdiv","comedi","trajedi","cole"]): return "rl_pop"
        if "ippo_s5_op" in n: return "rl_sp_op"
        if "ippo_s5" in n: return "rl_sp_s5"
        if "ippo_mlp" in n or "ippo" in n or "fcp" in n: return "rl_sp_mlp"
        return "other"

    family_order = ["heuristic", "obl_r2d2", "human_proxy", "rl_sp_mlp",
                    "rl_sp_s5", "rl_sp_op", "rl_pop", "other"]
    fam = [family_of(n) for n in names]
    order = sorted(range(len(names)), key=lambda i: (family_order.index(fam[i]), names[i]))
    cosine_r = cosine[np.ix_(order, order)]
    names_r = [names[i] for i in order]
    fam_r = [fam[i] for i in order]

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(7, n * 0.4)))

    off_diag = cosine_r[~np.eye(len(cosine_r), dtype=bool)]
    vmin = float(np.percentile(off_diag, 1))
    vmax = 1.0
    im = ax.imshow(cosine_r, vmin=vmin, vmax=vmax, cmap="RdYlBu_r",
                   aspect="equal", interpolation="nearest")

    boundaries = [i for i in range(1, n) if fam_r[i] != fam_r[i-1]]
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", linewidth=0.8)
        ax.axhline(b - 0.5, color="black", linewidth=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        [s.replace("_serious","").replace("_long_6e7","") for s in names_r],
        rotation=45, ha="right",
    )
    ax.set_yticklabels(
        [s.replace("_serious","").replace("_long_6e7","") for s in names_r],
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("cosine similarity", fontsize=10)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


