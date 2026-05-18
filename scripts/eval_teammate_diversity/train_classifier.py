"""Train the trajectory classifier on saved trajectory data."""

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
import yaml

from trajectory_classifier import (
    init_classifier,
    make_classifier_train_step,
    train_classifier,
    pad_labeled_episodes,
)
from common.save_load_utils import save_train_run
# Config
DEFAULT_TASK_NAME = "lbf/lbf_7x7_nolevels"
DEFAULT_DATA_DIR = None  # auto-derived from task_name when not provided
DEFAULT_MODEL_DIR = None  # auto-derived from task_name when not provided
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LATENT_DIM = 16
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_NUM_EPOCHS = 200
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_SAMPLES_PER_CLASS = 5000


def _load_task_config(task_name):
    config_path = Path("evaluation/configs/task") / f"{task_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_obs_dim(env_name, env_kwargs=None):
    """Get observation dimension for an environment."""
    from envs import make_env
    env = make_env(env_name, env_kwargs or {})
    return env.observation_space("agent_0").shape[0]


def main(
    task_name=DEFAULT_TASK_NAME,
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    hidden_dim=DEFAULT_HIDDEN_DIM,
    latent_dim=DEFAULT_LATENT_DIM,
    learning_rate=DEFAULT_LEARNING_RATE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    max_samples_per_class=DEFAULT_MAX_SAMPLES_PER_CLASS,
):
    """Train classifier on saved trajectories."""
    cfg = _load_task_config(task_name)
    env_name = cfg["ENV_NAME"]
    env_kwargs = cfg.get("ENV_KWARGS") or {}

    if data_dir is None:
        data_dir = f"results/{task_name}/trajectory_data"
    if model_dir is None:
        model_dir = f"results/{task_name}/models"

    data_path = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load trajectories
    train_path = data_path / "train_episodes.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"Train episodes not found at {train_path}. Run collect_trajectories.py first.")

    print(f"Loading trajectories from {train_path}...")
    with open(train_path, "rb") as f:
        data = pickle.load(f)
    episodes_with_labels = data["episodes"]
    pair_labels = data["pair_labels"]
    print(f"Loaded {len(episodes_with_labels)} train pairwise episodes.")

    obs_dim = get_obs_dim(env_name, env_kwargs)
    padded_episodes, masks, labels, max_seq_len, label_to_idx = pad_labeled_episodes(
        episodes_with_labels, max_samples_per_class=max_samples_per_class
    )

    # Load val episodes for overfitting diagnostics (optional)
    val_padded, val_masks, val_labels = None, None, None
    val_path = data_path / "val_episodes.pkl"
    if val_path.exists():
        print(f"Loading val episodes from {val_path}...")
        with open(val_path, "rb") as f:
            val_data = pickle.load(f)
        val_padded, val_masks, val_labels, _, _ = pad_labeled_episodes(
            val_data["episodes"], label_to_idx=label_to_idx
        )
        print(f"Loaded {len(val_data['episodes'])} val episodes.")
    else:
        print("No val_episodes.pkl found; skipping test accuracy tracking.")

    num_classes = len(label_to_idx)

    rng = jax.random.PRNGKey(42)
    rng, train_state, model = init_classifier(
        rng,
        obs_dim,
        max_seq_len,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        learning_rate=learning_rate,
        latent_dim=latent_dim,
    )
    train_step = make_classifier_train_step(model)

    print(f"Training classifier on {len(episodes_with_labels)} episodes (padded length {max_seq_len})")
    print(f"Model config: hidden_dim={hidden_dim}, latent_dim={latent_dim}, num_classes={num_classes}")
    rng, train_state, losses, train_accs, test_accs = train_classifier(
        rng,
        train_state,
        train_step,
        padded_episodes,
        masks,
        labels,
        num_epochs=num_epochs,
        batch_size=batch_size,
        test_padded=val_padded,
        test_masks=val_masks,
        test_labels=val_labels,
    )

    print("Training complete. Saving model...")

    # Wrap scalar config values and label_to_idx values as numpy arrays so
    # orbax can serialize them alongside the params pytree.
    checkpoint = {
        "params": train_state.params,
        "config": {
            "hidden_dim": np.array(hidden_dim, dtype=np.int32),
            "latent_dim": np.array(latent_dim, dtype=np.int32),
            "obs_dim": np.array(obs_dim, dtype=np.int32),
            "max_seq_len": np.array(max_seq_len, dtype=np.int32),
            "num_classes": np.array(num_classes, dtype=np.int32),
        },
        "label_to_idx": {k: np.array(v, dtype=np.int32) for k, v in label_to_idx.items()},
    }

    save_path = save_train_run(checkpoint, model_dir, "trajectory_classifier")
    print(f"Model saved to {save_path}")

    # Save training losses and accuracies
    loss_file = model_path / "training_losses.npy"
    np.save(loss_file, np.array(losses))
    print(f"Training losses saved to {loss_file}")

    epochs = np.arange(1, num_epochs + 1)
    has_test = len(test_accs) > 0

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Classifier Training Loss Curve')
    plt.grid(True)
    loss_plot_file = model_path / "loss_curve.png"
    plt.savefig(loss_plot_file)
    plt.close()
    print(f"Loss curve saved to {loss_plot_file}")

    # Plot train (and test) accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Train')
    if has_test:
        plt.plot(epochs, test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('LSTM Classifier Train/Test Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_file = model_path / "accuracy_curve.png"
    plt.savefig(acc_plot_file)
    plt.close()
    print(f"Accuracy curve saved to {acc_plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trajectory classifier on saved data.")
    parser.add_argument("--task_name", type=str, default=DEFAULT_TASK_NAME, help="Task name (e.g. lbf/lbf_7x7_nolevels)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing trajectory data (default: results/<task_name>/trajectory_data)")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory to save trained model (default: results/<task_name>/models)")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM, help="Latent dimension")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--max_samples_per_class", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLASS,
                        help="Max training episodes per class (0 = use all data)")

    args = parser.parse_args()
    main(
        task_name=args.task_name,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class if args.max_samples_per_class > 0 else None,
    )
