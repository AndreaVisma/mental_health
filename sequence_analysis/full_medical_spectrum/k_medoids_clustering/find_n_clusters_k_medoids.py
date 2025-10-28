


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids   # <-- NEW import
from tqdm import tqdm
import os

gender = "F"
if gender == "F":
    sex_id = 2
else:
    sex_id = 1

sample = True
sample_size = 20_000

# --- Paths ---
distance_matrix_path = (
    f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\dtw_distance_{gender}_sampled_{sample}_noNorm.npy"
)

# --- Utility ---
def load_and_sample(matrix_path, sample_size=1000):
    """Load DTW distance matrix and take a random sample for quick testing"""
    matrix = np.load(matrix_path, mmap_mode="r")
    n = min(sample_size, matrix.shape[0])
    indices = np.random.choice(matrix.shape[0], n, replace=False)
    return matrix[np.ix_(indices, indices)], indices


# --- Load data ---
dtw_distances, sampled_indices = load_and_sample(distance_matrix_path, sample_size)
print(f"Working with {dtw_distances.shape[0]} samples")

# --- Clustering setup ---
k_values = range(2, 15)
inertias = []
silhouettes = []

for k in tqdm(k_values):
    # KMedoids with precomputed distances
    kmedoids = KMedoids(
        n_clusters=k,
        metric="precomputed",   # important
        init="k-medoids++",
        random_state=42,
        max_iter=300,
        method="alternate",
    )

    labels = kmedoids.fit_predict(dtw_distances)

    # Inertia equivalent: total distance to medoids
    inertias.append(kmedoids.inertia_)

    # Silhouette score
    if k > 1:
        silhouette_avg = silhouette_score(dtw_distances, labels, metric="precomputed")
        silhouettes.append(silhouette_avg)
    else:
        silhouettes.append(0)

# --- Plot results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_values, inertias, "bo-")
ax1.set_title("Elbow Method (K-Medoids)")
ax1.set_xlabel("k")
ax1.set_ylabel("Total Distance to Medoids")

ax2.plot(k_values[1:], silhouettes[1:], "ro-")
ax2.set_title("Silhouette Score (K-Medoids)")
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show(block=True)

