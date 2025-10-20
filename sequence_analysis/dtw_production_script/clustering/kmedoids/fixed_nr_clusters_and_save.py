

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

# Ultra-light version using KMeans on sampled data
distance_matrix_path = ("c://git-projects//mental_health//sequence_analysis//common_disease_trajectories//"
                        "Simons_distances//distance_matrices//dtw_distance_matrix_MEN_ONLY_F_4DIGITS.npy")


def load_and_sample(matrix_path, sample_size=2000):
    """Load matrix and sample for quick testing"""
    matrix = np.load(matrix_path, mmap_mode='r')
    n = min(sample_size, matrix.shape[0])
    indices = np.random.choice(matrix.shape[0], n, replace=False)
    return matrix[indices][:, indices], indices


# Load data
dtw_distances, sampled_indices = load_and_sample(distance_matrix_path, 10_000)
print(f"Working with {dtw_distances.shape[0]} samples")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
labels = kmeans.fit_predict(dtw_distances)
np.save(
    "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\clustering\\kmedoids\\labels_5_clusters.npy",
    labels)
print("Labels for 10_000 boys saved!")