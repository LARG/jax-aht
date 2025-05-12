import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def do_PCA(data: np.ndarray):
  # Standardize the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)

  # Apply PCA
  pca = PCA(n_components=2)  # Choose the number of components
  pca.fit(scaled_data)
  pca_data = pca.transform(scaled_data)
  # Explained variance ratio
  # explained_variance = pca.explained_variance_ratio_
  # print("Explained variance ratio:", explained_variance)
  return pca_data

def visualize_PCA(pca_data):
  # Visualize the results (for 2D data)
  if pca_data.shape[1] == 2:
    plt.scatter(pca_data[:, 0], pca_data[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Results")
    plt.show()
  else:
    print("Invalid PCA data given, must be 2D")