"""Visualize convention clusters produced by cluster_conventions.py.

Reads a PD features.csv plus the clusters.csv assignment file and renders a single
multi-panel figure:
  (a) dendrogram of the agglomerative clustering (average linkage, cosine distance),
  (b) 2D MDS embedding of the cosine-distance matrix, colored by cluster, medoids ringed,
  (c) cosine-similarity heatmap reordered by the dendrogram leaf order.

Example:
    python scripts/convention_analysis/plot_convention_clusters.py \\
        --features results/conv/pd_coord_ring/overcooked-coord_ring/features_derived.csv \\
        --clusters results/conv/eval/coord_ring__split/clusters.csv \\
        --title "coord_ring eval teammates" --out coord_ring_clusters.pdf
"""
from __future__ import annotations

import re

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

# Reuse the exact feature-selection + medoid logic the clustering used, so the picture
# and the tables can never drift apart.
from scripts.convention_analysis.cluster_conventions import NON_FEATURE_COLS, cluster_medoid, select_feature_matrix

TAB10 = plt.get_cmap("tab10").colors


def mds_2d(D: np.ndarray, seed: int = 0) -> np.ndarray:
    """Classical (Torgerson) MDS to 2D from a precomputed distance matrix.

    Deterministic and dependency-free (no sklearn). Double-center -0.5 D^2, take the top
    two eigenvectors. seed is accepted only for API symmetry; the result is deterministic.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = (B + B.T) / 2  # enforce symmetry against fp drift
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    top = order[:2]
    L = np.sqrt(np.clip(vals[top], 0, None))
    return vecs[:, top] * L


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--clusters", required=True, type=Path,
                   help="clusters.csv from cluster_conventions.py (agent,set,cluster,...).")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--linkage", default="average")
    p.add_argument("--title", default="Convention clusters")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    feats = pd.read_csv(args.features)
    clus = pd.read_csv(args.clusters)

    # clusters.csv may have dropped teammates (e.g. --exclude); align features to it and
    # preserve clusters.csv row order so labels/positions stay consistent.
    if "agent" not in feats.columns:
        raise ValueError("features.csv has no 'agent' column")
    feats = feats.set_index("agent").loc[clus["agent"]].reset_index()
    labels = clus["cluster"].to_numpy()
    names = clus["agent"].tolist()

    Xz, feature_cols, dropped = select_feature_matrix(feats)
    d = pdist(Xz, metric=args.metric)
    D = squareform(d)
    Z = linkage(d, method=args.linkage)
    order = leaves_list(Z)

    uniq = sorted(np.unique(labels))
    color_of = {c: TAB10[i % len(TAB10)] for i, c in enumerate(uniq)}
    medoids = {c: cluster_medoid(D, np.where(labels == c)[0]) for c in uniq}

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.2], wspace=0.28)

    # (a) dendrogram
    ax0 = fig.add_subplot(gs[0, 0])
    short = [re.sub(r"\s*\(0, (\d+)\)$", r" #\1", n) for n in names]
    dendrogram(Z, labels=short, ax=ax0, color_threshold=0, above_threshold_color="0.6",
               leaf_font_size=7)
    ax0.set_title("(a) dendrogram (cosine, average)")
    ax0.set_ylabel("merge distance")
    for lbl in ax0.get_xticklabels():
        idx = short.index(lbl.get_text())
        lbl.set_color(color_of[labels[idx]])

    # (b) MDS embedding
    ax1 = fig.add_subplot(gs[0, 1])
    XY = mds_2d(D)
    for c in uniq:
        m = labels == c
        ax1.scatter(XY[m, 0], XY[m, 1], s=55, color=color_of[c], edgecolor="white",
                    linewidth=0.6, label=f"cluster {c} (n={int(m.sum())})", zorder=3)
    # ring + label the medoids
    for c in uniq:
        mi = medoids[c]
        ax1.scatter(XY[mi, 0], XY[mi, 1], s=180, facecolor="none", edgecolor="black",
                    linewidth=1.6, zorder=4)
        ax1.annotate(short[mi], (XY[mi, 0], XY[mi, 1]), fontsize=7, xytext=(4, 4),
                     textcoords="offset points")
    ax1.set_title("(b) MDS of cosine distances\n(black ring = cluster medoid)")
    ax1.set_xlabel("MDS-1"); ax1.set_ylabel("MDS-2")
    ax1.legend(fontsize=6, loc="best", framealpha=0.9)

    # (c) cosine-similarity heatmap, dendrogram-ordered
    ax2 = fig.add_subplot(gs[0, 2])
    S = 1.0 - D  # cosine similarity
    So = S[np.ix_(order, order)]
    im = ax2.imshow(So, cmap="viridis", vmin=np.percentile(S, 2), vmax=1.0)
    ax2.set_title("(c) cosine similarity (leaf-ordered)")
    ax2.set_xticks(range(len(order)))
    ax2.set_yticks(range(len(order)))
    ax2.set_xticklabels([short[i] for i in order], rotation=90, fontsize=5)
    ax2.set_yticklabels([short[i] for i in order], fontsize=5)
    for tick, i in zip(ax2.get_xticklabels(), order):
        tick.set_color(color_of[labels[i]])
    for tick, i in zip(ax2.get_yticklabels(), order):
        tick.set_color(color_of[labels[i]])
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(args.title, fontsize=13, y=1.02)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}\nwrote {png}")
    print(f"features used: {len(feature_cols)}  dropped: {dropped}")
    print(f"medoids: " + ", ".join(f"c{c}={short[medoids[c]]}" for c in uniq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
