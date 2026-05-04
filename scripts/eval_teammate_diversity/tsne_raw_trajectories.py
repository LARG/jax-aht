"""Raw t-SNE on val_episodes trajectories — no classifier, no learned embeddings.

Each trajectory is aggregated into a single feature vector and then projected
with t-SNE.  Three aggregation modes are supported:
  - mean   : mean over timesteps (default, robust to variable-length episodes)
  - meanstd: concatenate mean and std over timesteps
  - flatten: pad to max length then flatten (large but lossless)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DEFAULT_TASK_NAME = "lbf/lbf_7x7_nolevels"
DEFAULT_DATA_DIR = None
DEFAULT_AGG = "mean"
DEFAULT_PERPLEXITY = 30
DEFAULT_MAX_SAMPLES_PER_CLASS = 500


# --------------------------------------------------------------------------- #
# Label display helpers (kept in sync with plot_tsne_trajectory.py)           #
# --------------------------------------------------------------------------- #

_DISPLAY_NAMES = {
    "brdiv-conf1_0": "brdiv1-0",
    "brdiv-conf1_1": "brdiv1-1",
    "brdiv-conf1_2": "brdiv1-2",
    "brdiv-conf2_0": "brdiv2-0",
    "brdiv-conf2_1": "brdiv2-1",
    "ippo_mlp": "ippo-mlp",
    "ippo_mlp_s2c0": "ippo-s2c0",
    "seq_agent_col": "seq-col",
    "seq_agent_farthest": "seq-far",
    "seq_agent_lexi": "seq-lexi",
    "seq_agent_nearest": "seq-near",
    "seq_agent_rcol": "seq-rcol",
    "seq_agent_rlexi": "seq-rlexi",
}

_AGENT_MARKERS = {
    "ippo": "o",
    "brdiv": "^",
    "comedi": "s",
    "lbrdiv": "v",
    "seq": "P",
}


def _agent_type(agent_name):
    for prefix in ("lbrdiv", "brdiv", "comedi", "ippo", "seq"):
        if agent_name.startswith(prefix):
            return prefix
    return None


def _display_name(label):
    br_marker = "_br_for_"
    agent = label[: label.index(br_marker)] if br_marker in label else label
    return _DISPLAY_NAMES.get(agent, agent)


def _marker(label):
    br_marker = "_br_for_"
    agent = label[: label.index(br_marker)] if br_marker in label else label
    atype = _agent_type(agent)
    return _AGENT_MARKERS.get(atype, "o")


# --------------------------------------------------------------------------- #
# Feature extraction                                                           #
# --------------------------------------------------------------------------- #

def filter_diagonal_episodes(episodes_with_labels):
    """Keep only episodes where an agent plays against its own best response.

    Labels have the form '<agent>_br_for_<br_suffix>'; diagonal entries are
    those where the BR was specifically trained for that agent.  This mirrors
    _is_specific_br_pair in trajectory_collection.py: normalize hyphens to
    underscores and allow a trailing numeric-only index (e.g. ippo_mlp vs
    br_for_ippo_mlp_0).  Labels without the marker are dropped.
    """
    br_marker = "_br_for_"
    filtered = []
    for traj, label in episodes_with_labels:
        if br_marker not in label:
            continue
        agent, br_suffix = label.split(br_marker, 1)
        norm = agent.replace("-", "_")
        if br_suffix == norm:
            filtered.append((traj, label))
        elif br_suffix.startswith(norm + "_"):
            rest = br_suffix[len(norm) + 1:]
            if rest and all(c.isdigit() or c == "_" for c in rest):
                filtered.append((traj, label))
    return filtered


def aggregate_episodes(episodes_with_labels, agg, max_samples_per_class):
    """Return (features_dict, unique_labels) where features_dict maps label -> (N, D) array."""
    by_label: dict[str, list[np.ndarray]] = {}
    for traj, label in episodes_with_labels:
        by_label.setdefault(label, []).append(np.asarray(traj, dtype=np.float32))

    rng = np.random.default_rng(42)
    features_dict: dict[str, np.ndarray] = {}

    if agg == "flatten":
        max_len = max(ep.shape[0] for eps in by_label.values() for ep in eps)

    for label, eps in by_label.items():
        if max_samples_per_class is not None and len(eps) > max_samples_per_class:
            idx = rng.choice(len(eps), size=max_samples_per_class, replace=False)
            eps = [eps[i] for i in idx]

        vecs = []
        for ep in eps:
            if agg == "mean":
                vecs.append(ep.mean(axis=0))
            elif agg == "meanstd":
                vecs.append(np.concatenate([ep.mean(axis=0), ep.std(axis=0)]))
            elif agg == "flatten":
                padded = np.zeros((max_len, ep.shape[1]), dtype=np.float32)
                padded[: ep.shape[0]] = ep
                vecs.append(padded.ravel())
            else:
                raise ValueError(f"Unknown aggregation: {agg!r}")
        features_dict[label] = np.stack(vecs)

    unique_labels = sorted(features_dict.keys())
    return features_dict, unique_labels


# --------------------------------------------------------------------------- #
# t-SNE + plot                                                                 #
# --------------------------------------------------------------------------- #

def run_and_plot(features_dict, unique_labels, perplexity, save_path, title):
    all_features = np.concatenate([features_dict[l] for l in unique_labels], axis=0)
    n_samples = len(all_features)
    eff_perplexity = min(perplexity, max(5, n_samples // 3))

    print(f"Running t-SNE on {n_samples} points, dim={all_features.shape[1]}, perplexity={eff_perplexity}")
    tsne = TSNE(n_components=2, perplexity=eff_perplexity, random_state=42)
    embeddings = tsne.fit_transform(all_features)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Assign one color per display name so all raw labels sharing a name look identical.
    unique_display = list(dict.fromkeys(_display_name(l) for l in unique_labels))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    display_color = {dn: color_cycle[i % len(color_cycle)] for i, dn in enumerate(unique_display)}
    seen_display: set[str] = set()

    offset = 0
    for label in unique_labels:
        n = len(features_dict[label])
        sl = embeddings[offset : offset + n]
        dname = _display_name(label)
        legend_label = dname if dname not in seen_display else "_nolegend_"
        seen_display.add(dname)
        ax.scatter(sl[:, 0], sl[:, 1], label=legend_label, marker=_marker(label), color=display_color[dname], alpha=0.6, s=20)
        offset += n

    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10, markerscale=1.2, handlelength=1.0, borderpad=0.4, labelspacing=0.3)
    leg.get_frame().set_alpha(0.4)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("t-SNE 1", fontsize=20)
    ax.set_ylabel("t-SNE 2", fontsize=20)
    ax.tick_params(axis="both", labelsize=17)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = Path(save_path).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  |  {pdf_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(task_name, data_dir, agg, perplexity, max_samples_per_class, out_dir):
    if data_dir is None:
        data_dir = f"results/{task_name}/trajectory_data"
    if out_dir is None:
        out_dir = f"results/{task_name}/tsne_raw"

    val_path = Path(data_dir) / "val_episodes.pkl"
    if not val_path.exists():
        raise FileNotFoundError(f"val_episodes.pkl not found at {val_path}")

    print(f"Loading {val_path} ...")
    with open(val_path, "rb") as f:
        data = pickle.load(f)
    episodes_with_labels = data["episodes"]
    print(f"Loaded {len(episodes_with_labels)} episodes.")

    episodes_with_labels = filter_diagonal_episodes(episodes_with_labels)
    print(f"After diagonal filter: {len(episodes_with_labels)} episodes (agent vs its own BR only).")

    features_dict, unique_labels = aggregate_episodes(episodes_with_labels, agg, max_samples_per_class)
    print("Class breakdown:")
    for label in unique_labels:
        print(f"  {label}: {len(features_dict[label])} episodes, feat_dim={features_dict[label].shape[1]}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(out_dir) / f"tsne_raw_{agg}.png")
    title = f"t-SNE (raw trajectories, agg={agg})"
    run_and_plot(features_dict, unique_labels, perplexity, save_path, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw t-SNE on val_episodes trajectories.")
    parser.add_argument("--task_name", type=str, default=DEFAULT_TASK_NAME)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory with val_episodes.pkl (default: results/<task_name>/trajectory_data)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: results/<task_name>/tsne_raw)")
    parser.add_argument("--agg", type=str, default=DEFAULT_AGG, choices=["mean", "meanstd", "flatten"], help="How to aggregate each trajectory into a single vector")
    parser.add_argument("--perplexity", type=int, default=DEFAULT_PERPLEXITY)
    parser.add_argument("--max_samples_per_class", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLASS, help="Cap episodes per class (0 = no cap)")
    args = parser.parse_args()

    main(
        task_name=args.task_name,
        data_dir=args.data_dir,
        agg=args.agg,
        perplexity=args.perplexity,
        max_samples_per_class=args.max_samples_per_class if args.max_samples_per_class > 0 else None,
        out_dir=args.out_dir,
    )
