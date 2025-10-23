

import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids   # <-- NEW import
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

# Ultra-light version using KMeans on sampled data
distance_matrix_path = ("C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\dtw_distance_M_3_2010_noNorm.npy")

# Load data
dtw_distances = np.load(distance_matrix_path, mmap_mode='r')
print(f"Working with {dtw_distances.shape[0]} samples")

kmedoids = KMedoids(
    n_clusters=4,
    metric="precomputed",  # important
    init="k-medoids++",
    random_state=42,
    max_iter=300,
    method="alternate",
)
labels = kmedoids.fit_predict(dtw_distances)
np.save(
    "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\clustering\\kmedoids\\labels_4_clusters.npy",
    labels)
print("Labels for all the boys saved!")