
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

# -------------------
# Configuration
# -------------------
distance_matrix_path = (
    "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\dtw_distance_F_3_2310_noNorm.npy"
)
output_dir = (
    "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\clustering\\kmedoids\\"
)

# ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# -------------------
# Parameters
# -------------------
sample_size = 20_000   # adjust as needed (2000–5000 usually fine)
n_clusters = 4
random_state = 42

# -------------------
# Load and subsample
# -------------------
print("Loading DTW distance matrix (memory-mapped)...")
dtw_distances = np.load(distance_matrix_path, mmap_mode="r")
n_total = dtw_distances.shape[0]
print(f"Full dataset has {n_total} samples")

# Randomly choose subset of indices
rng = np.random.default_rng(random_state)
sample_indices = rng.choice(n_total, size=min(sample_size, n_total), replace=False)
print(f"Subsampled {len(sample_indices)} samples")

# Extract the corresponding submatrix
dtw_sub = dtw_distances[np.ix_(sample_indices, sample_indices)]
print(f"Submatrix shape: {dtw_sub.shape}")

# -------------------
# K-Medoids clustering
# -------------------
print("Running K-Medoids clustering...")
kmedoids = KMedoids(
    n_clusters=n_clusters,
    metric="precomputed",
    init="k-medoids++",
    random_state=random_state,
    max_iter=300,
    method="alternate",
)

labels = kmedoids.fit_predict(dtw_sub)

# -------------------
# Evaluation
# -------------------
try:
    silhouette_avg = silhouette_score(dtw_sub, labels, metric="precomputed")
    print(f"Silhouette score: {silhouette_avg:.4f}")
except Exception as e:
    print("Silhouette computation failed:", e)

# -------------------
# Save outputs
# -------------------
labels_path = os.path.join(output_dir, f"labels_{n_clusters}_clusters_subsample_F.npy")
indices_path = os.path.join(output_dir, f"indices_subsample_F.npy")

np.save(labels_path, labels)
np.save(indices_path, sample_indices)

print(f"✅ Saved labels to:   {labels_path}")
print(f"✅ Saved indices to:  {indices_path}")
print("Done!")
