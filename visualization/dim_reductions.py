# TODO: implement / use PCA (scikit learn?)
# @ Aditya

from typing import List
import jax
import jax.numpy as jnp
from jax import vmap

def PCA(points: List[jax.Array]) -> List[jax.Array]:
    # take the input (list of points in N-d space) and map to a list of 2d using PCA
    # we should implement using jax because this is time sensitive?

    # Number of components to reduce points into (by default 2)
    n_components = 2

    # Flatten each N-dimensional array
    flat_arrays = [jnp.ravel(x) for x in points]
    
    # Stack into 2D data matrix: (num_samples, features)
    X = jnp.stack(flat_arrays)

    # Center the data
    X_mean = jnp.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute covariance matrix (features x features)
    cov_matrix = jnp.cov(X_centered, rowvar=False)

    # Eigen decomposition (symmetric matrices => use eigh)
    eigvals, eigvecs = jnp.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = jnp.argsort(eigvals)[::-1]
    top_components = eigvecs[:, sorted_indices[:n_components]]

    # Project data
    X_pca = jnp.dot(X_centered, top_components)

    # Split back into list of JAX arrays
    return [X_pca[i] for i in range(X_pca.shape[0])]    

# TODO: implement / use t-SNE
# @ Aditya

def PCA(points: List[jax.Array]) -> List[jax.Array]:
    # take the input (list of points in N-d space) and map to a list of 2d using tSNE
    # we should implement using jax because this is time sensitive?
    return None    

# TODO: implement / use t-SNE
# @ Aditya

import matplotlib

def visualize(points: List[jax.Array]):
    # generate a plot
    return