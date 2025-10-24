

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm
import os

# --- Paths ---
distance_matrix_path = (
    "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\dtw_distance_F_3_2310_noNorm.npy"
)


# --- Utility ---
def load_and_sample(matrix_path, sample_size=1000):
    """Load DTW distance matrix and take a random sample for quick testing"""
    matrix = np.load(matrix_path, mmap_mode="r")
    n = min(sample_size, matrix.shape[0])
    indices = np.random.choice(matrix.shape[0], n, replace=False)
    return matrix[np.ix_(indices, indices)], indices


# --- Load data ---
dtw_distances, sampled_indices = load_and_sample(distance_matrix_path, 2000)
print(f"Working with {dtw_distances.shape[0]} samples")

# --- Hierarchical Clustering setup ---
k_values = range(2, 12)
linkage_methods = ['ward', 'complete', 'average', 'single']  # Different linkage criteria
best_silhouette = -1
best_k = 2
best_method = 'complete'

# Convert distance matrix to condensed form (required by scipy linkage)
condensed_dist = squareform(dtw_distances, checks=False)

print("Performing hierarchical clustering with different linkage methods...")

for method in linkage_methods:
    print(f"\n--- Testing linkage method: {method} ---")

    # Perform hierarchical clustering
    # Note: 'ward' method requires Euclidean distances, others work with precomputed
    if method == 'ward':
        # For ward linkage, we need to ensure the input is appropriate
        # You might want to use a different approach or skip ward for distance matrices
        Z = linkage(condensed_dist, method=method, metric='euclidean')
    else:
        Z = linkage(condensed_dist, method=method, metric='precomputed')

    silhouettes = []

    for k in tqdm(k_values, desc=f"{method}"):
        # Cut dendrogram to get k clusters
        labels = fcluster(Z, k, criterion='maxclust')

        # Calculate silhouette score
        if k > 1:
            silhouette_avg = silhouette_score(dtw_distances, labels, metric="precomputed")
            silhouettes.append(silhouette_avg)

            # Track best configuration
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k = k
                best_method = method
        else:
            silhouettes.append(0)

    # Plot for this linkage method
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(k_values[1:], silhouettes[1:], 'o-')
    plt.title(f'Silhouette Score ({method} linkage)')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Plot dendrogram for the best k (truncated for readability)
    from scipy.cluster.hierarchy import dendrogram

    plt.title(f'Dendrogram ({method} linkage)')
    dendrogram(Z, truncate_mode='lastp', p=min(20, len(k_values)), show_leaf_counts=True)

    plt.tight_layout()
    plt.show()

print(f"\nBest configuration: k={best_k}, method='{best_method}', silhouette={best_silhouette:.4f}")

# --- Final clustering with best parameters ---
print(f"\nPerforming final clustering with best parameters...")
if best_method == 'ward':
    Z_final = linkage(condensed_dist, method=best_method, metric='euclidean')
else:
    Z_final = linkage(condensed_dist, method=best_method, metric='precomputed')

final_labels = fcluster(Z_final, best_k, criterion='maxclust')

# --- Plot final results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Silhouette analysis
from scipy.cluster.hierarchy import dendrogram

dendrogram(Z_final, truncate_mode='lastp', p=min(30, best_k * 2), ax=ax1)
ax1.set_title(f'Final Dendrogram ({best_method} linkage, k={best_k})')

# Cluster distribution
unique, counts = np.unique(final_labels, return_counts=True)
ax2.bar(unique, counts)
ax2.set_title('Cluster Size Distribution')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Number of Points')

plt.tight_layout()
plt.show(block = True)

print(f"\nCluster sizes: {dict(zip(unique, counts))}")
print(f"Final silhouette score: {best_silhouette:.4f}")