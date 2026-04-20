import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(latents_dict, save_path="tsne_trajectories.png", perplexity=30):
    # Validate input
    if not latents_dict:
        print("ERROR: latents_dict is empty!")
        return
    
    # Count total points
    total_points = sum(len(lat) for lat in latents_dict.values())
    if total_points == 0:
        print("ERROR: No data points in latents_dict!")
        return
    
    all_latents = np.concatenate(list(latents_dict.values()), axis=0)
    labels = []
    for label, lat in latents_dict.items():
        labels.extend([label] * len(lat))

    print(f"\n=== PLOT TSNE DEBUG ===")
    print(f"Number of categories: {len(latents_dict)}")
    print(f"Category breakdown:")
    for label, lat in latents_dict.items():
        print(f"  {label}: {len(lat)} points, shape {lat.shape}")
    print(f"Total points: {len(all_latents)}")
    print(f"Labels length: {len(labels)}")
    print(f"Data shape: {all_latents.shape}")
    
    # Ensure we have enough samples for t-SNE
    if len(all_latents) < 2:
        print("ERROR: Need at least 2 data points for t-SNE!")
        return

    n_samples = len(all_latents)
    # Perplexity must be less than n_samples / 3, and we want at least perplexity=5
    effective_perplexity = min(perplexity, max(5, n_samples // 3))
    print(f"Using perplexity: {effective_perplexity} (requested: {perplexity}, n_samples: {n_samples})")

    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42)
    embeddings = tsne.fit_transform(all_latents)
    print(f"t-SNE output shape: {embeddings.shape}")

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = list(latents_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    offset = 0
    plotted_count = 0
    for i, label in enumerate(unique_labels):
        n = len(latents_dict[label])
        embedding_slice = embeddings[offset:offset+n, 0:2]
        if embedding_slice.shape[0] > 0:
            ax.scatter(
                embedding_slice[:, 0],
                embedding_slice[:, 1],
                c=[colors[i]],
                label=label,
                alpha=0.6,
                s=20,
            )
            plotted_count += embedding_slice.shape[0]
        offset += n

    print(f"Actually plotted {plotted_count} points")
    ax.legend()
    ax.set_title("t-SNE of Trajectory Latents")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
    print(f"======================\n")
