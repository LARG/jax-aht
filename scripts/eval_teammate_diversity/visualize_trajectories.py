"""Visualize trajectory classifier performance."""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flax.training.train_state import TrainState
import optax
from sklearn.metrics import classification_report

from trajectory_classifier import (
    create_classifier,
    make_classifier_eval_step,
    pad_labeled_episodes,
)
from trajectory_collection import (
    collect_pair_trajectories,
    get_agent_pair_configs,
)
from plot_tsne_trajectory import plot_tsne
from common.save_load_utils import load_train_run
from envs import make_env


def _is_specific_best_response(agent_name, br_name):
    """Return true if br_name is the specific BR for agent_name."""
    if not br_name.startswith("br_for_"):
        return False
    suffix = br_name[len("br_for_"):]
    norm_agent = agent_name.replace("-", "_")
    if suffix == norm_agent:
        return True
    prefix = norm_agent + "_"
    if suffix.startswith(prefix):
        # Only a numeric seed/index suffix (digits and underscores) is allowed,
        # not another agent's name that shares the same prefix (e.g. ippo_mlp_s2c0).
        rest = suffix[len(prefix):]
        return all(c.isdigit() or c == "_" for c in rest)
    return False


def _filter_agent_br_pairs(pairs):
    return [
        (agent_name, agent_cfg, br_name, br_cfg)
        for agent_name, agent_cfg, br_name, br_cfg in pairs
        if _is_specific_best_response(agent_name, br_name)
    ]

def plot_pair_confusion_matrix(predictions, all_true_labels, label_to_idx, idx_to_label, output_file):
    """Plot and save a column-normalized 169×169 confusion matrix heatmap.

    Entry (i, j) = P(classifier predicts pair i | true label is pair j).
    """
    pair_labels = {k: v for k, v in label_to_idx.items() if "_br_for_" in k}
    sorted_names = sorted(pair_labels.keys())
    sorted_indices = [pair_labels[n] for n in sorted_names]
    n = len(sorted_names)

    idx_map = {orig: new for new, orig in enumerate(sorted_indices)}

    cm = np.zeros((n, n), dtype=float)
    for true_lbl, pred_lbl in zip(all_true_labels, predictions):
        if true_lbl in idx_map and pred_lbl in idx_map:
            cm[idx_map[pred_lbl], idx_map[true_lbl]] += 1

    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_norm = cm / col_sums

    fig, ax = plt.subplots(figsize=(32, 30))
    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=sorted_names,
        yticklabels=sorted_names,
        annot=False,
        cbar_kws={"label": "P(predicted | true)", "shrink": 0.6},
    )
    ax.set_xlabel("True label", fontsize=12)
    ax.set_ylabel("Predicted label", fontsize=12)
    ax.set_title(f"Pair confusion matrix ({n}×{n})", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=3)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=3)

    plt.tight_layout()

    output_path = Path(output_file)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Pair confusion matrix saved to {output_path} and {pdf_path}")


# Config — paths are relative to the repo root (two levels up from this script)
_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = str(_REPO_ROOT / "results/lbf_7x7_nolevels/trajectory_data")
DEFAULT_MODEL_DIR = str(_REPO_ROOT / "results/lbf_7x7_nolevels/models")
DEFAULT_MODEL_FILE = "trajectory_classifier"
DEFAULT_OUTPUT_FILE = str(_REPO_ROOT / "results/lbf_7x7_nolevels/tsne_trajectory_visualization.png")
DEFAULT_LATENTS_FILE = str(_REPO_ROOT / "results/lbf_7x7_nolevles/latents.pkl")


def collect_latents(
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    model_file=DEFAULT_MODEL_FILE,
    latents_file=DEFAULT_LATENTS_FILE,
):
    """Encode heldout episodes with the trained autoencoder and save latents to disk."""
    data_path = Path(data_dir)
    model_path = Path(model_dir) / model_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train_classifier.py first."
        )

    print(f"Loading trained model from {model_path}...")
    restored = load_train_run(str(model_path))

    params = restored["params"]
    config = restored["config"]
    hidden_dim = int(config["hidden_dim"])
    latent_dim = int(config["latent_dim"])
    obs_dim = int(config["obs_dim"])
    max_seq_len = int(config["max_seq_len"])
    num_classes = int(config["num_classes"])
    label_to_idx = {k: int(v) for k, v in restored["label_to_idx"].items()}

    print(f"Model config: hidden_dim={hidden_dim}, latent_dim={latent_dim}, obs_dim={obs_dim}, max_seq_len={max_seq_len}, num_classes={num_classes}")

    model = create_classifier(obs_dim, max_seq_len, hidden_dim, num_classes, latent_dim)
    tx = optax.adam(1e-3)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    eval_step = make_classifier_eval_step(model)

    def make_encoder_eval_step(model):
        @jax.jit
        def encode_step(params, x, mask):
            return jax.vmap(
                lambda x_i, m_i: model.apply(params, x_i, m_i, method=model.encode)
            )(x, mask)
        return encode_step

    encode_step = make_encoder_eval_step(model)

    val_path = data_path / "val_episodes.pkl"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation episodes not found at {val_path}. Run collect_trajectories.py first.")

    print(f"Loading validation trajectories from {val_path}...")
    with open(val_path, "rb") as f:
        data = pickle.load(f)
    test_episodes = data["episodes"]

    # Use the training label_to_idx so integer labels align with classifier logit indices.
    padded_episodes, masks, labels, _, _ = pad_labeled_episodes(test_episodes, label_to_idx=label_to_idx)

    print("Evaluating classifier on test data...")
    all_logits = []
    all_latents = []
    all_true_labels = []
    batch_size = 64
    N = padded_episodes.shape[0]
    num_batches = (N + batch_size - 1) // batch_size
    print(f"Evaluating in {num_batches} batches of up to {batch_size} examples")

    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch_x = padded_episodes[i:end_idx]
        batch_mask = masks[i:end_idx]
        batch_y = labels[i:end_idx]

        logits = eval_step(params, batch_x, batch_mask)
        latents = encode_step(params, batch_x, batch_mask)
        all_logits.append(np.array(logits))
        all_latents.append(np.array(latents))
        all_true_labels.append(np.array(batch_y))

    all_logits = np.concatenate(all_logits, axis=0)
    all_latents = np.concatenate(all_latents, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    predictions = np.argmax(all_logits, axis=1)

    # Build idx_to_label from the training mapping (same mapping used for labels above).
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # For the t-SNE, only include trajectories where an agent plays against its
    # specific best-response (not cross-BR pairs).  The val set now contains all
    # pairs, so we filter here rather than in collection.
    latents_dict = {}
    for label_name, label_idx in label_to_idx.items():
        br_marker = "_br_for_"
        if br_marker not in label_name:
            continue
        split_pos = label_name.index(br_marker)
        agent_name = label_name[:split_pos]
        br_name = label_name[split_pos + 1:]
        if _is_specific_best_response(agent_name, br_name):
            latents = all_latents[all_true_labels == label_idx]
            if len(latents) == 0:
                print(f"ERROR: No trajectory data found for '{label_name}' — agent or its best response was not collected. Skipping from t-SNE.")
                continue
            latents_dict[label_name] = latents
    print(f"Collected latents for {len(latents_dict)} agent-BR pairs: {sorted(latents_dict.keys())}")

    save_data = {
        "latents_dict": latents_dict,
        "predictions": predictions,
        "all_true_labels": all_true_labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "num_classes": num_classes,
    }
    Path(latents_file).parent.mkdir(parents=True, exist_ok=True)
    with open(latents_file, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Latents saved to {latents_file}")
    return save_data


def plot(
    latents_file=DEFAULT_LATENTS_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
    plot_title="t-SNE",
):
    """Load saved latents and produce confusion matrix + t-SNE plots."""
    if not Path(latents_file).exists():
        raise FileNotFoundError(
            f"Latents file not found at {latents_file}. Run with --collect first."
        )

    print(f"Loading latents from {latents_file}...")
    with open(latents_file, "rb") as f:
        save_data = pickle.load(f)

    latents_dict = save_data["latents_dict"]
    predictions = save_data["predictions"]
    all_true_labels = save_data["all_true_labels"]
    label_to_idx = save_data["label_to_idx"]
    idx_to_label = save_data["idx_to_label"]
    num_classes = save_data["num_classes"]

    accuracy = np.mean(predictions == all_true_labels)
    print(f"Accuracy: {accuracy:.4f}")

    unique_labels = np.unique(np.concatenate([all_true_labels, predictions]))
    label_names = [idx_to_label[i] for i in unique_labels]

    print(f"Creating t-SNE visualization from latent encodings...")
    plot_tsne(latents_dict, save_path=output_file, title=plot_title)

    cm_output = Path(output_file).with_name("pair_confusion_matrix.png")
    print(f"Creating pair confusion matrix heatmap...")
    plot_pair_confusion_matrix(predictions, all_true_labels, label_to_idx, idx_to_label, cm_output)

    print("\nClassification Report:")
    print(classification_report(all_true_labels, predictions, labels=unique_labels, target_names=label_names))

    print(f"Evaluation complete. t-SNE saved to {output_file}")


def main(
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    model_file=DEFAULT_MODEL_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
    latents_file=DEFAULT_LATENTS_FILE,
    plot_only=False,
    plot_title="t-SNE",
):
    if not plot_only:
        collect_latents(
            data_dir=data_dir,
            model_dir=model_dir,
            model_file=model_file,
            latents_file=latents_file,
        )
    plot(latents_file=latents_file, output_file=output_file, plot_title=plot_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory classifier performance.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing trajectory data")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory containing trained model")
    parser.add_argument("--model_file", type=str, default=DEFAULT_MODEL_FILE, help="Trained model filename")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output visualization filename")
    parser.add_argument("--latents_file", type=str, default=DEFAULT_LATENTS_FILE, help="Path to save/load encoded latents")
    parser.add_argument("--plot-only", action="store_true", help="Skip data collection and plot from saved latents file")
    parser.add_argument("--plot_title", type=str, default="t-SNE", help="Title for the t-SNE plot")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_file=args.model_file,
        output_file=args.output_file,
        latents_file=args.latents_file,
        plot_only=args.plot_only,
        plot_title=args.plot_title,
    )
