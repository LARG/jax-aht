"""Cluster teammates by their behavioral-feature vectors; writes clusters.csv, labels_by_k.csv, cluster_summary.md.

    python scripts/convention_analysis/cluster_conventions.py \
        --features results/conv_v12/pd_coord_ring/overcooked-coord_ring/features_derived.csv \
        --set-labels coord_ring --groups split --exclude-groups throughput \
        --k-tolerance 0.05 --min-cluster-size 2 --output-dir results/tmp/coord_ring
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

from scripts.convention_analysis import feature_groups

log = logging.getLogger("cluster_conventions")

# Metadata / outcome columns, never features.
NON_FEATURE_COLS = {"agent", "env", "num_episodes", "mean_final_score", "mean_episode_length"}
SCORE_COL = "mean_final_score"


def load_features(paths: List[Path], set_labels: Optional[List[str]]) -> pd.DataFrame:
    """Load and concatenate features.csv files, tagging each with its population."""
    if set_labels and len(set_labels) != len(paths):
        raise ValueError(f"--set-labels has {len(set_labels)} entries but --features has {len(paths)}")

    frames = []
    for i, p in enumerate(paths):
        if not p.exists():
            raise FileNotFoundError(f"features file not found: {p}")
        df = pd.read_csv(p)
        df["set"] = set_labels[i] if set_labels else p.parent.name
        frames.append(df)
        log.info("loaded %d teammates from %s (set=%s)", len(df), p, df["set"].iloc[0])

    out = pd.concat(frames, ignore_index=True)
    # Disambiguate identically-named agents coming from different populations.
    if out["agent"].duplicated().any():
        out["agent"] = out["set"] + ":" + out["agent"].astype(str)
    return out


def select_feature_matrix(df: pd.DataFrame, drop_near_constant: bool = True
                          ) -> Tuple[np.ndarray, List[str], List[str]]:
    """Pick numeric feature columns, drop near-constant ones, z-score the rest."""
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != "set" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("no numeric feature columns found")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        bad = [feature_cols[j] for j in range(X.shape[1]) if not np.isfinite(X[:, j]).all()]
        raise ValueError(f"non-finite values in feature columns: {bad}")

    dropped: List[str] = []
    if drop_near_constant:
        std = X.std(axis=0)
        keep_mask = std > 1e-12
        dropped = [c for c, k in zip(feature_cols, keep_mask) if not k]
        feature_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
        X = X[:, keep_mask]
        if dropped:
            log.info("dropped %d near-constant features: %s", len(dropped), dropped)

    if X.shape[1] == 0:
        raise ValueError("all features were near-constant; nothing to cluster on")

    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    return Xz, feature_cols, dropped


def silhouette_from_distances(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient from a precomputed distance matrix (singletons contribute 0)."""
    n = len(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return float("nan")

    sils = np.zeros(n)
    for i in range(n):
        own = labels == labels[i]
        own_others = own.copy()
        own_others[i] = False
        if own_others.sum() == 0:
            sils[i] = 0.0  # singleton
            continue
        a = D[i, own_others].mean()
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            b = min(b, D[i, labels == c].mean())
        sils[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(sils.mean())


def cluster_medoid(D: np.ndarray, idx: np.ndarray) -> int:
    """Index (into the full array) of the cluster member minimizing total in-cluster distance."""
    sub = D[np.ix_(idx, idx)]
    return int(idx[int(np.argmin(sub.sum(axis=1)))])


def distinguishing_features(Xz: np.ndarray, feature_cols: List[str], member_idx: np.ndarray,
                            top_n: int = 5) -> List[Tuple[str, float]]:
    """Features whose in-cluster mean deviates most from the population mean (z units)."""
    means = Xz[member_idx].mean(axis=0)
    order = np.argsort(-np.abs(means))[:top_n]
    return [(feature_cols[j], float(means[j])) for j in order]


def score_ci(values: np.ndarray) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Mean + 95% bootstrap CI via the repo's rliable machinery, if available."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), None
    if values.size < 2:
        return float(values.mean()), None
    try:
        from common.stat_utils import compute_aggregate_stat_and_ci_per_task
        point, interval = compute_aggregate_stat_and_ci_per_task(
            values.reshape(-1, 1), "mean", return_interval_est=True
        )
        return float(np.asarray(point).squeeze()), (
            float(np.asarray(interval).ravel()[0]), float(np.asarray(interval).ravel()[1])
        )
    except Exception as e:  # pragma: no cover - CI is a nicety, never fatal
        log.warning("rliable CI unavailable (%s); reporting mean only", e)
        return float(values.mean()), None


def apply_group_weights(Xz: np.ndarray, feature_cols: List[str], grp: Dict[str, str]
                        ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, int]]:
    """Scale each z-scored column by 1 / sqrt(number of columns in its group)."""
    counts: Dict[str, int] = {}
    for c in feature_cols:
        counts[grp[c]] = counts.get(grp[c], 0) + 1
    w = np.array([1.0 / np.sqrt(counts[grp[c]]) for c in feature_cols])
    return Xz * w[None, :], {c: float(v) for c, v in zip(feature_cols, w)}, counts


def group_distance_shares(X: np.ndarray, feature_cols: List[str], grp: Dict[str, str]
                          ) -> Dict[str, float]:
    """Fraction of total pairwise squared Euclidean distance carried by each group."""
    n = X.shape[0]
    per_col = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        d = X[:, j][:, None] - X[:, j][None, :]
        per_col[j] = (d ** 2).sum() / 2.0
    tot = per_col.sum()
    out: Dict[str, float] = {}
    for c, v in zip(feature_cols, per_col):
        out[grp[c]] = out.get(grp[c], 0.0) + float(v / tot) if tot > 0 else 0.0
    return out


def run(df: pd.DataFrame, metric: str, linkage_method: str, k: Optional[int],
        k_min: int, k_max: int, top_n: int, k_tolerance: float = 0.0,
        min_cluster_size: int = 1, group_mode: str = "none") -> Dict:
    Xz, feature_cols, dropped = select_feature_matrix(df)
    grp: Dict[str, str] = {}
    group_w: Dict[str, float] = {}
    group_counts: Dict[str, int] = {}
    shares_before: Dict[str, float] = {}
    shares_after: Dict[str, float] = {}
    try:  # group shares are reported even without weighting, when every column has a group
        grp = feature_groups.group_of(feature_cols)
        shares_before = group_distance_shares(Xz, feature_cols, grp)
    except KeyError:
        if group_mode != "none":
            raise
        grp = {}
    singleton_dropped: List[str] = []
    if group_mode != "none":
        # drop groups reduced to a single column
        counts0: Dict[str, int] = {}
        for c in feature_cols:
            counts0[grp[c]] = counts0.get(grp[c], 0) + 1
        singleton_dropped = [c for c in feature_cols if counts0[grp[c]] == 1]
        if singleton_dropped:
            log.info("dropping single-column groups: %s", singleton_dropped)
            keep_s = [c not in singleton_dropped for c in feature_cols]
            Xz = Xz[:, keep_s]
            feature_cols = [c for c, k in zip(feature_cols, keep_s) if k]
            grp = {c: grp[c] for c in feature_cols}
            shares_before = group_distance_shares(Xz, feature_cols, grp)
        Xz, group_w, group_counts = apply_group_weights(Xz, feature_cols, grp)
        shares_after = group_distance_shares(Xz, feature_cols, grp)
        log.info("group weighting (%s): %s", group_mode,
                 {g: f"m={m} w={1/np.sqrt(m):.2f}" for g, m in sorted(group_counts.items())})
        log.info("share of squared distance before -> after: %s",
                 {g: f"{shares_before[g]:.2f}->{shares_after[g]:.2f}" for g in sorted(shares_before)})
    # report z-deviations on the unweighted scale
    Xz_report = Xz
    if group_w:
        Xz_report = Xz_report / np.array([group_w[c] for c in feature_cols])[None, :]
    n = Xz.shape[0]
    if n < 3:
        raise ValueError(f"need >=3 teammates to cluster, got {n}")

    D = squareform(pdist(Xz, metric=metric))
    Z = linkage(pdist(Xz, metric=metric), method=linkage_method)

    # cut_tree, not fcluster(maxclust): maxclust can return fewer clusters than requested
    k_hi = min(k_max, n - 1)
    sweep: List[Tuple[int, float]] = []
    labels_by_k: Dict[int, np.ndarray] = {}
    for kk in range(max(2, k_min), k_hi + 1):
        lab = cut_tree(Z, n_clusters=kk).ravel() + 1
        sweep.append((kk, silhouette_from_distances(D, lab)))
        labels_by_k[kk] = lab

    if k is None:
        finite = [(kk, s) for kk, s in sweep if np.isfinite(s)]
        if not finite:
            k = 2
            log.warning("silhouette undefined for every k (population too small?); "
                        "defaulting to k=2 -- treat the partition as unvalidated")
        else:
            best = max(s for _, s in finite)
            # finest partition within tolerance of the best silhouette
            candidates = [kk for kk, s in finite if s >= best - k_tolerance]
            if min_cluster_size > 1:
                def _min_size(kk: int) -> int:
                    lab = cut_tree(Z, n_clusters=kk).ravel()
                    return int(np.bincount(lab).min())
                sized = [kk for kk in candidates if _min_size(kk) >= min_cluster_size]
                if not sized:
                    # re-run the tolerance rule over cuts satisfying the size floor
                    sized_all = [(kk, s) for kk, s in finite if _min_size(kk) >= min_cluster_size]
                    if sized_all:
                        best_sized = max(s for _, s in sized_all)
                        sized = [kk for kk, s in sized_all if s >= best_sized - k_tolerance]
                        log.warning("no k within tolerance satisfies min-cluster-size=%d; "
                                    "restricting to sized cuts %s (best sized silhouette %.3f)",
                                    min_cluster_size, [kk for kk, _ in sized_all], best_sized)
                    else:
                        log.warning("no candidate k satisfies min-cluster-size=%d; "
                                    "ignoring the guard", min_cluster_size)
                if sized:
                    candidates = sized
            k = max(candidates) if k_tolerance > 0 else max(finite, key=lambda t: t[1])[0]
            log.info("auto-selected k=%d (silhouette %.3f, best %.3f, tolerance %.3f)",
                     k, dict(sweep)[k], best, k_tolerance)

    labels = cut_tree(Z, n_clusters=k).ravel() + 1
    order = leaves_list(Z)

    clusters = []
    for c in sorted(np.unique(labels)):
        member_idx = np.where(labels == c)[0]
        medoid = cluster_medoid(D, member_idx)
        scores = df[SCORE_COL].to_numpy()[member_idx] if SCORE_COL in df.columns else np.array([])
        mean_score, ci = score_ci(scores)
        clusters.append({
            "cluster": int(c),
            "size": int(len(member_idx)),
            "members": [str(x) for x in df["agent"].to_numpy()[member_idx]],
            "sets": sorted({str(x) for x in df["set"].to_numpy()[member_idx]}),
            "medoid": str(df["agent"].to_numpy()[medoid]),
            "mean_return": mean_score,
            "return_ci": ci,
            "distinguishing": distinguishing_features(Xz_report, feature_cols, member_idx, top_n),
        })

    # competence confound check
    eta_sq = float("nan")
    if SCORE_COL in df.columns:
        y = df[SCORE_COL].to_numpy(dtype=np.float64)
        if y.std() > 1e-12:
            grand = y.mean()
            ss_between = sum(len(np.where(labels == c)[0]) * (y[labels == c].mean() - grand) ** 2
                             for c in np.unique(labels))
            ss_total = ((y - grand) ** 2).sum()
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")

    return {
        "k": int(k), "labels": labels, "sweep": sweep, "clusters": clusters,
        "labels_by_k": labels_by_k,
        "feature_cols": feature_cols, "dropped": dropped, "order": order,
        "eta_sq_return": eta_sq, "n": n, "metric": metric, "linkage": linkage_method,
        "group_mode": group_mode, "groups": grp, "group_counts": group_counts, "group_w": group_w,
        "group_shares_before": shares_before, "group_shares_after": shares_after,
        "silhouette": float(dict(sweep).get(int(k), float("nan"))),
    }


def write_markdown(res: Dict, df: pd.DataFrame, out: Path) -> None:
    L: List[str] = []
    L.append("# Convention clusters\n")
    L.append(f"- teammates: **{res['n']}**  |  features used: **{len(res['feature_cols'])}**"
             f"  |  distance: `{res['metric']}`  |  linkage: `{res['linkage']}`")
    L.append(f"- selected **k = {res['k']}**")
    if res["dropped"]:
        L.append(f"- dropped {len(res['dropped'])} near-constant features: "
                 f"{', '.join(f'`{d}`' for d in res['dropped'])}")
    if res.get("group_mode", "none") != "none":
        L.append(f"- **group weighting** (`{res['group_mode']}`): each column scaled by 1/sqrt(group size) "
                 "so every a-priori group contributes equally to expected squared distance")
    if res.get("group_shares_before"):
        counts = {g: sum(1 for c in res["groups"].values() if c == g) for g in res["group_shares_before"]}
        L.append("")
        L.append("| group | columns | share of sq. distance, unweighted | weighted |")
        L.append("|---|---|---|---|")
        for g in sorted(counts):
            after = res["group_shares_after"].get(g, res["group_shares_before"][g])
            L.append(f"| {g} | {counts[g]} | {res['group_shares_before'][g]:.2f} | {after:.2f} |")
    L.append("")

    L.append("## Silhouette sweep\n")
    L.append("| k | mean silhouette |")
    L.append("|---|---|")
    for kk, s in res["sweep"]:
        mark = "  **<- selected**" if kk == res["k"] else ""
        L.append(f"| {kk} | {s:.3f}{mark} |")
    L.append("")
    L.append("> Silhouette ranges from -1 to 1; values near 0 mean the clusters overlap heavily "
             "and the partition is weakly supported.\n")

    eta = res["eta_sq_return"]
    L.append("## Competence-confound check\n")
    if np.isfinite(eta):
        L.append(f"Fraction of return variance explained by cluster identity (eta^2): **{eta:.3f}**.")
        L.append("")
        L.append("> High eta^2 means the clustering is largely recovering *how good* teammates are "
                 "rather than *what convention* they play. Low eta^2 supports a convention reading.\n")
    else:
        L.append("Return variance unavailable or degenerate; confound check skipped.\n")

    L.append("## Clusters\n")
    L.append("| cluster | size | populations | medoid (record this one) | mean return [95% CI] |")
    L.append("|---|---|---|---|---|")
    for c in res["clusters"]:
        ci = c["return_ci"]
        ci_s = f"{c['mean_return']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{c['mean_return']:.2f}"
        L.append(f"| {c['cluster']} | {c['size']} | {', '.join(c['sets'])} | `{c['medoid']}` | {ci_s} |")
    L.append("")

    L.append("## Distinguishing features per cluster\n")
    L.append("Deviation of the cluster mean from the population mean, in standard deviations.\n")
    for c in res["clusters"]:
        L.append(f"### Cluster {c['cluster']} (n={c['size']}, medoid `{c['medoid']}`)\n")
        L.append("| feature | z-deviation |")
        L.append("|---|---|")
        for name, z in c["distinguishing"]:
            g = f" ({res['groups'][name]})" if res.get("groups") else ""
            L.append(f"| `{name}`{g} | {z:+.2f} |")
        L.append("")
        L.append(f"Members: {', '.join('`' + m + '`' for m in c['members'])}\n")
        L.append("_Convention description (fill in after watching the medoid rollout):_ TODO\n")

    out.write_text("\n".join(L))
    log.info("wrote %s", out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True, type=Path,
                   help="One or more features.csv paths.")
    p.add_argument("--set-labels", nargs="+", default=None,
                   help="Population label per --features entry.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "correlation"],
                   help="Distance on z-scored features.")
    p.add_argument("--linkage", default="average", choices=["average", "complete", "single", "ward"],
                   help="ward requires --metric euclidean.")
    p.add_argument("--k", type=int, default=None, help="Number of clusters; default = best silhouette.")
    p.add_argument("--min-cluster-size", type=int, default=1,
                   help="Reject candidate k's producing any cluster smaller than this.")
    p.add_argument("--k-tolerance", type=float, default=0.0,
                   help="Select the largest k whose silhouette is within this margin of the max.")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--top-n", type=int, default=5, help="Distinguishing features to list per cluster.")
    p.add_argument("--drop-features", nargs="*", default=None,
                   help="Feature columns to remove before clustering.")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Drop teammates whose name contains any of these substrings.")
    p.add_argument("--groups", default="none", choices=["none", "split"],
                   help="Feature-group weighting mode (feature_groups.py).")
    p.add_argument("--exclude-groups", nargs="*", default=[],
                   help="Drop every column of these feature groups.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.linkage == "ward" and args.metric != "euclidean":
        p.error("--linkage ward requires --metric euclidean")

    df = load_features(args.features, args.set_labels)
    drop_features = list(args.drop_features or [])
    drop_features += [c for c in feature_groups.DERIVE_INPUTS if c not in drop_features]
    if args.exclude_groups:
        allg = feature_groups.groups()
        bad = [g for g in args.exclude_groups if g not in allg]
        if bad:
            p.error(f"--exclude-groups unknown: {bad}; choose from {sorted(allg)}")
        drop_features += [c for g in args.exclude_groups for c in allg[g] if c not in drop_features]
    if drop_features:
        present = [c for c in drop_features if c in df.columns]
        missing = [c for c in drop_features if c not in df.columns]
        if missing:
            log.warning("--drop-features not found in columns (ignored): %s", missing)
        if present:
            log.info("dropping %d screened-out features: %s", len(present), present)
            df = df.drop(columns=present)
    if args.exclude:
        mask = df["agent"].apply(lambda a: not any(s in str(a) for s in args.exclude))
        dropped_names = df.loc[~mask, "agent"].tolist()
        if dropped_names:
            log.info("excluded %d teammates: %s", len(dropped_names), dropped_names)
        df = df[mask].reset_index(drop=True)

    res = run(df, args.metric, args.linkage, args.k, args.k_min, args.k_max, args.top_n,
              k_tolerance=args.k_tolerance, min_cluster_size=args.min_cluster_size,
              group_mode=args.groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_df = df[["agent", "set"]].copy()
    out_df["cluster"] = res["labels"]
    if SCORE_COL in df.columns:
        out_df[SCORE_COL] = df[SCORE_COL]
    out_df.to_csv(args.output_dir / "clusters.csv", index=False)
    # every cut of the same dendrogram
    byk = df[["agent"]].copy()
    for kk, lab in sorted(res["labels_by_k"].items()):
        byk[f"k{kk}"] = lab
    byk.to_csv(args.output_dir / "labels_by_k.csv", index=False)
    write_markdown(res, df, args.output_dir / "cluster_summary.md")
    (args.output_dir / "features_used.txt").write_text("\n".join(res["feature_cols"]) + "\n")
    if res["group_w"]:
        (args.output_dir / "feature_weights.txt").write_text(
            "".join(f"{c} {w:.6f}\n" for c, w in res["group_w"].items()))

    print(f"\nk={res['k']}  n={res['n']}  features={len(res['feature_cols'])}")
    for c in res["clusters"]:
        print(f"  cluster {c['cluster']}: n={c['size']:2d}  medoid={c['medoid']}")
    print(f"\nwrote {args.output_dir}/clusters.csv and cluster_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
