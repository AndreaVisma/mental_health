import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

# Ultra-light version using KMeans on sampled data
distance_matrix_path = ("C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\dtw_distance_M_3_2010_noNorm.npy")

def load_and_sample(matrix_path, sample_size=1000):
    """Load matrix and sample for quick testing"""
    matrix = np.load(matrix_path, mmap_mode='r')
    n = min(sample_size, matrix.shape[0])
    indices = np.random.choice(matrix.shape[0], n, replace=False)
    return matrix[indices][:, indices], indices


# Load data
dtw_distances, sampled_indices = load_and_sample(distance_matrix_path, 2000)
print(f"Working with {dtw_distances.shape[0]} samples")

k_values = range(2, 12)
inertias = []
silhouettes = []

for k in tqdm(k_values):
    # Use KMeans on distance matrix (requires conversion)
    from sklearn.metrics.pairwise import pairwise_distances_argmin_min

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(dtw_distances)

    inertias.append(kmeans.inertia_)

    if k > 1:  # Silhouette requires at least 2 clusters
        silhouette_avg = silhouette_score(dtw_distances, labels, metric='precomputed')
        silhouettes.append(silhouette_avg)
    else:
        silhouettes.append(0)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(k_values, inertias, 'bo-')
ax1.set_title('Elbow Method')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')

ax2.plot(k_values[1:], silhouettes[1:], 'ro-')
ax2.set_title('Silhouette Score')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show(block = True)