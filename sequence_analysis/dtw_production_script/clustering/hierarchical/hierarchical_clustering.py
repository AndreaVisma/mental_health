import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import os
import sys

# --- CONFIGURATION ---
distance_matrix_path = ("c://git-projects//mental_health//sequence_analysis//common_disease_trajectories//"
                        "Simons_distances//distance_matrices//dtw_distance_matrix_MEN_ONLY_F_4DIGITS.npy")
# Hierarchical clustering can be slow/memory intensive for plotting large N
MAX_SAMPLES = 10_000
DEFAULT_K = 5


# --- FUNCTIONS ---
def load_distance_matrix(matrix_path, max_samples=None):
    """Load and validate distance matrix, optionally subsampling."""
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Distance matrix not found at: {matrix_path}")

    dtw_distances = np.load(matrix_path)

    if dtw_distances.shape[0] != dtw_distances.shape[1]:
        raise ValueError("Distance matrix must be square.")

    print(f"Loaded DTW distance matrix with shape: {dtw_distances.shape}")

    if max_samples is not None and dtw_distances.shape[0] > max_samples:
        print(f"Subsampling matrix to {max_samples}x{max_samples} for clustering analysis.")
        return dtw_distances[:max_samples, :max_samples]

    return dtw_distances


def get_merge_distances(Z):
    """
    Extracts the merge distance (height) and corresponding number of clusters (k)
    from the linkage matrix Z.
    """
    N = len(Z) + 1  # Total number of points

    # Z[:, 2] contains the distance (height) at which the merge occurred
    distances = Z[:, 2]

    # The number of clusters remaining after the i-th merge is N - (i + 1)
    # We want k to go from N down to 2.
    k_values = np.arange(N, 1, -1)

    # When k=N, distance is 0.0 (no merges). We start from k=N-1 down to k=2.
    # The plot shows the distance for the merge that reduces k to k-1.

    # We plot (N - i) clusters vs. the distance Z[i-1, 2]
    # For k=N-1 (first merge), the distance is distances[0].
    # For k=2 (last merge), the distance is distances[-1].

    # We want the plot to start high (low k) and go down (high k), or vice versa.

    # Let's plot k vs. the distance that created those k clusters (starting from N)

    # Number of clusters (k) after the merge (i=0 to N-2)
    k_after_merge = N - np.arange(len(distances)) - 1

    # We discard the first N-2 entries (which are not visually meaningful)
    # and focus on the last 50 merges (k=51 down to k=2) for better visualization
    start_index = max(0, len(distances) - 50)

    # Distances are already sorted ascendingly by SciPy's linkage.
    # We reverse them to plot k=2 (highest distance) down to k=51 (lowest distance)
    # k_plot: [2, 3, 4, ..., 51]
    # distance_plot: [Z[-1, 2], Z[-2, 2], ..., Z[N-51, 2]]
    k_plot = k_after_merge[start_index:]
    distance_plot = distances[start_index:]

    return k_plot, distance_plot


def find_optimal_k_distance_jump(Z):
    """
    Identifies the largest jump in distance (dendrogram height) to suggest optimal k.
    """
    distances = Z[:, 2]

    # Calculate the difference (jump) between successive merging distances
    distance_jumps = np.diff(distances)

    if len(distance_jumps) == 0:
        return DEFAULT_K

    # The index of the largest jump suggests the best cut point.
    largest_jump_index = np.argmax(distance_jumps)

    N = len(Z) + 1
    # Number of clusters (k) = N - (index of the largest jump)
    optimal_k = N - largest_jump_index

    # Ensure k is a valid number
    return max(2, min(N - 1, optimal_k))


# --- MAIN EXECUTION ---
try:
    # Load the DTW distance matrix
    dtw_distances = load_distance_matrix(distance_matrix_path, max_samples=MAX_SAMPLES)
    N = dtw_distances.shape[0]

    # Convert to condensed form for SciPy
    condensed_dist_matrix = squareform(dtw_distances)

    # 1. Perform Hierarchical Linkage (using 'average' for non-Euclidean DTW distance)
    Z = linkage(condensed_dist_matrix, method='average')
    print(f"Linkage matrix (Z) created with shape: {Z.shape}")

    # 2. Get data for the Information Loss Plot
    k_plot, distance_plot = get_merge_distances(Z)

    # 3. Find Optimal K for the plot annotation
    optimal_k_suggestion = find_optimal_k_distance_jump(Z)
    distance_threshold = Z[N - optimal_k_suggestion, 2]  # Height at the suggested cut

    # 4. Extract Clusters for a brief printout
    clusters = fcluster(Z, optimal_k_suggestion, criterion='maxclust')
    print(f"\nExtracted {optimal_k_suggestion} clusters. Cluster counts:\n{np.unique(clusters, return_counts=True)}")


except FileNotFoundError as e:
    print(e)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

# --- VISUALIZATION (Information Loss Plot) ---
plt.figure(figsize=(10, 6))

# Plot the distance against the number of clusters (k)
plt.plot(k_plot, distance_plot, marker='o', linestyle='-',
         color='tab:red', linewidth=2, markersize=5)

# Highlight the suggested cut point (the largest jump)
plt.axvline(x=optimal_k_suggestion, color='k', linestyle='--', alpha=0.7)
plt.axhline(y=distance_threshold, color='k', linestyle='--', alpha=0.7)

plt.title("Information Preservation vs. Number of Clusters (DTW Hierarchical)", fontsize=14)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Distance/Dissimilarity at Merge (Information Loss)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_plot[::max(1, len(k_plot) // 10)])  # Make sure x-ticks aren't too crowded
plt.legend([f'Suggested Cut: k={optimal_k_suggestion}'], loc='upper right')
plt.gca().invert_xaxis()  # Plot k decreasing, as is traditional for this view

plt.tight_layout()
plt.show()

print("\nHierarchical clustering information loss plot complete.")