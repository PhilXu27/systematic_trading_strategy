from utils.path_info import part_1_data_path, part_1_results_path
from pathlib import Path
from os.path import exists

import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

print("Loading HMM data...")

data_hmm_file_path = Path(part_1_data_path, "data_hmm.csv")
if exists(data_hmm_file_path):
    data_hmm = pd.read_csv(data_hmm_file_path)
else:
    url = 'https://drive.google.com/uc?id=1YBEwyZblijkXiMad3mqqmaWwgQpiVlKl'
    data_hmm = pd.read_csv(url)
    data_hmm.to_csv(data_hmm_file_path)

X = data_hmm.values
print("✓ Loaded HMM data")
# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1])
plt.title('HMM Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.savefig(Path(part_1_results_path, "hmm_plot.png"))

# Assume X is your data matrix (rows are i.i.d. observations)
# Example: X = np.array([[...], [...], ...])

# Fit GMM with 3 components
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Extract learned parameters
pi = gmm.weights_         # Mixture weights (π)
mu = gmm.means_           # Mean vectors (μ)
Sigma = gmm.covariances_  # Covariance matrices (Σ)

# Print parameters
print("Mixture weights (π):", pi)
print("Mean vectors (μ):", mu)
print("Covariance matrices (Σ):", Sigma)
