# TODO: implement / use PCA (scikit learn?)
# @ Aditya

from typing import List
import jax
import jax.numpy as jnp

def PCA(points: List[jax.Array]) -> List[jax.Array]:
    # take the input (list of points in N-d space) and map to a list of 2d using PCA
    # we should implement using jax because this is time sensitive?

    # number of components in the final PCA (by default 2)
    n_components = 2

    # Stack the list of 1D points into a (num_points, num_dims) matrix
    X = jnp.stack(points)

    # Center the data (zero mean)
    X_mean = jnp.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute covariance matrix
    cov_matrix = jnp.cov(X_centered, rowvar=False)

    # Eigen decomposition (symmetric matrix)
    eigvals, eigvecs = jnp.linalg.eigh(cov_matrix)

    # Select top n_components eigenvectors
    sorted_indices = jnp.argsort(eigvals)[::-1]
    top_vectors = eigvecs[:, sorted_indices[:n_components]]

    # Project the data
    X_pca = X_centered @ top_vectors

    # Return as list of 2D JAX arrays
    return [X_pca[i] for i in range(X_pca.shape[0])]  

# TODO: implement / use t-SNE
# @ Aditya

def tSNE(points: List[jax.Array]) -> List[jax.Array]:
    # take the input (list of points in N-d space) and map to a list of 2d using tSNE
    # we should implement using jax because this is time sensitive?
    return None    

# TODO: implement / use t-SNE
# @ Aditya

import numpy as np
import matplotlib as plt

def visualize(points: List[jax.Array]):
     # Convert to NumPy for plotting
    points_np = np.stack([np.array(p) for p in points])
    x, y = points_np[:, 0], points_np[:, 1]

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=50)

    plt.title("Agent Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()