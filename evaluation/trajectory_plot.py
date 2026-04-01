import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(latents_dict, save_path="tsne_trajectories.png", perplexity=30):
    all_latents = np.concatenate(list(latents_dict.values()), axis=0)
    labels = []
    for label, lat in latents_dict.items():
        labels.extend([label] * len(lat))

    n_samples = len(all_latents)
    perplexity = min(perplexity, max(5, n_samples // 4))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(all_latents)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = list(latents_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    offset = 0
    for i, label in enumerate(unique_labels):
        n = len(latents_dict[label])
        ax.scatter(
            embeddings[offset:offset+n, 0],
            embeddings[offset:offset+n, 1],
            c=[colors[i]],
            label=label,
            alpha=0.6,
            s=20,
        )
        offset += n

    ax.legend()
    ax.set_title("t-SNE of Trajectory Latents")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
