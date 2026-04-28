"""Visualize trajectory classifier performance."""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from evaluation.trajectory_autoencoder import (
    create_classifier,
    make_classifier_eval_step,
    pad_labeled_episodes,
)
from evaluation.trajectory_collection import (
    collect_pair_trajectories,
    get_agent_pair_configs,
)
from evaluation.trajectory_plot import plot_tsne
from common.agent_loader_from_config import initialize_rl_agent_from_config
from envs import make_env


def _is_specific_best_response(agent_name, br_name):
    """Return true if br_name is the specific BR for agent_name."""
    if not br_name.startswith("br_for_"):
        return False
    suffix = br_name[len("br_for_") :]
    return suffix == agent_name or suffix.startswith(agent_name + "_")


def _filter_agent_br_pairs(pairs):
    return [
        (agent_name, agent_cfg, br_name, br_cfg)
        for agent_name, agent_cfg, br_name, br_cfg in pairs
        if _is_specific_best_response(agent_name, br_name)
    ]

# Config
DEFAULT_DATA_DIR = "results/lbf/trajectory_data"
DEFAULT_MODEL_DIR = "results/lbf/autoencoder_models"
DEFAULT_MODEL_FILE = "autoencoder.pkl"
DEFAULT_OUTPUT_FILE = "results/lbf/tsne_trajectory_visualization.png"
DEFAULT_LATENTS_FILE = "results/lbf/latents.pkl"


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
            f"Trained model not found at {model_path}. Run train_autoencoder.py first."
        )

    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    params = checkpoint["params"]
    config = checkpoint["config"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    obs_dim = config["obs_dim"]
    max_seq_len = config["max_seq_len"]
    num_classes = config["num_classes"]
    label_to_idx = config["label_to_idx"]

    print(f"Model config: {config}")

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

    heldout_path = data_path / "heldout_episodes.pkl"
    if not heldout_path.exists():
        raise FileNotFoundError("Heldout episodes not found. Collect test data first.")

    print(f"Loading test trajectories from {heldout_path}...")
    with open(heldout_path, "rb") as f:
        data = pickle.load(f)
    test_episodes = data["episodes"]

    padded_episodes, masks, labels, _, _ = pad_labeled_episodes(test_episodes)

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

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    latents_dict = {}
    for label_name, label_idx in label_to_idx.items():
        br_marker = "_br_for_"
        if br_marker not in label_name:
            continue
        split_pos = label_name.index(br_marker)
        agent_name = label_name[:split_pos]
        br_name = label_name[split_pos + 1:]
        if _is_specific_best_response(agent_name, br_name):
            latents_dict[label_name] = all_latents[all_true_labels == label_idx][:100]

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

    label_names = [idx_to_label[i] for i in range(num_classes)]

    cm = confusion_matrix(all_true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = output_file.replace('.png', '_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"Creating t-SNE visualization from latent encodings...")
    plot_tsne(latents_dict, save_path=output_file)

    print("\nClassification Report:")
    print(classification_report(all_true_labels, predictions, target_names=label_names))

    print(f"Evaluation complete. Confusion matrix saved to {cm_path}, t-SNE saved to {output_file}")


def main(
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    model_file=DEFAULT_MODEL_FILE,
    output_file=DEFAULT_OUTPUT_FILE,
    latents_file=DEFAULT_LATENTS_FILE,
    collect=True,
    plot_only=False,
    env_name="lbf",
    k=5,
    num_envs=256,
    rollout_steps=128,
):
    if not plot_only:
        collect_latents(
            data_dir=data_dir,
            model_dir=model_dir,
            model_file=model_file,
            latents_file=latents_file,
        )
    plot(latents_file=latents_file, output_file=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory classifier performance.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing trajectory data")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory containing trained model")
    parser.add_argument("--model_file", type=str, default=DEFAULT_MODEL_FILE, help="Trained model filename")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output visualization filename")
    parser.add_argument("--latents_file", type=str, default=DEFAULT_LATENTS_FILE, help="Path to save/load encoded latents")
    parser.add_argument("--plot-only", action="store_true", help="Skip data collection and plot from saved latents file")
    parser.add_argument("--env_name", type=str, default="lbf", help="Environment name")
    parser.add_argument("--k", type=int, default=5, help="Number of rollouts per agent pair")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--rollout_steps", type=int, default=128, help="Steps per rollout")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_file=args.model_file,
        output_file=args.output_file,
        latents_file=args.latents_file,
        plot_only=args.plot_only,
        env_name=args.env_name,
        k=args.k,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
    )
