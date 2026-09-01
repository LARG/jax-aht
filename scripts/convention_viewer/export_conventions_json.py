"""Export a 2-D convention map for the viewer from population-diversity features.

Input is the `features.csv` written by
`scripts/population_diversity/compute_population_diversity.py --full-heldout --br-paired`
for one task. Reproduces that pipeline's normalization (drop metadata and
near-constant columns, z-score, cosine distances), embeds the teammates with
classical MDS, marks per-cluster medoids, and writes `conventions_<slug>.json`
for the site.

Cluster ids are read from a `--clusters` CSV (any file with `agent` and `cluster`
columns, e.g. the output of a clustering built on the population-diversity code
in scripts/population_diversity/) when one is given; otherwise they are computed
here with average-linkage agglomerative clustering at a fixed `--num-clusters`.

  python scripts/convention_viewer/export_conventions_json.py --task counter_circuit \
      --features results/population_diversity/overcooked-counter_circuit_full_brpaired/features.csv \
      --clusters results/convention_clusters/counter_circuit/clusters.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convention_viewer.tasks import TASKS, sanitize

NON_FEATURE = {"agent", "env", "pairing", "num_episodes", "mean_final_score", "mean_episode_length"}


def mds_2d(D: np.ndarray) -> np.ndarray:
    """Classical (Torgerson) MDS on a distance matrix."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = (B + B.T) / 2
    vals, vecs = np.linalg.eigh(B)
    top = np.argsort(vals)[::-1][:2]
    return vecs[:, top] * np.sqrt(np.clip(vals[top], 0, None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--features", required=True, type=Path, help="features.csv from a --br-paired PD run")
    ap.add_argument("--clusters", type=Path, default=None,
                    help="CSV with 'agent' and 'cluster' columns; overrides the fallback clustering")
    ap.add_argument("--num-clusters", type=int, default=4,
                    help="k for the fallback clustering, used only without --clusters")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "site")
    args = ap.parse_args()

    feats = pd.read_csv(args.features)
    clus = None
    if args.clusters is not None:
        clus = pd.read_csv(args.clusters)
        # the cluster file may cover a subset of teammates; align features to it and keep its order.
        feats = feats.set_index("agent").loc[clus["agent"]].reset_index()
    names = feats["agent"].tolist()
    cols = [c for c in feats.columns
            if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(feats[c])]
    X = feats[cols].to_numpy(float)
    X = X[:, X.std(axis=0) > 1e-12]
    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    D = squareform(pdist(Xz, metric="cosine"))
    XY = mds_2d(D)

    if clus is not None:
        labels = clus["cluster"].to_numpy()
    else:
        k = min(args.num_clusters, len(names))
        labels = fcluster(linkage(squareform(D, checks=False), method="average"), t=k, criterion="maxclust")

    medoids = set()
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        medoids.add(int(idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]))

    scores = feats["mean_final_score"] if "mean_final_score" in feats else None
    pairing = feats["pairing"] if "pairing" in feats else None
    points = [{
        "name": names[i],
        "key": sanitize(names[i]),
        "x": round(float(XY[i, 0]), 5),
        "y": round(float(XY[i, 1]), 5),
        "cluster": int(labels[i]),
        "is_medoid": i in medoids,
        "score": None if scores is None else round(float(scores.iloc[i]), 3),
        "pairing": None if pairing is None else str(pairing.iloc[i]),
    } for i in range(len(names))]
    points.sort(key=lambda p: (p["cluster"], p["name"]))

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"conventions_{args.task}.json"
    out_path.write_text(json.dumps({"task": args.task, "points": points}, indent=1))
    print("wrote", out_path)
    print("clusters:", {int(c): int((labels == c).sum()) for c in np.unique(labels)})
    print("medoids:", sorted(names[i] for i in medoids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
