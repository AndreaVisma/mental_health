import json
import pandas as pd
from itertools import chain
from collections import Counter
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import re
from numba import jit, prange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import os

plot_folder = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\figures\\"

####
# metadata to select subsample
metadata = pd.read_csv("C:\\Data\\my_datasets\\medical\\patients_metadata.csv")
males = metadata.query("sex_id == 1")["patient_no"].tolist()

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences.parquet")
seq_df = seq_df[seq_df.patient_no.isin(males)]
seq_df.rename(columns={"code": "sequence"}, inplace=True)

seq_df['patient_no'] = seq_df['patient_no'].astype(int)

# Optimized unique codes extraction
unique_codes = sorted(set(chain.from_iterable(seq_df['sequence'])))

def unique_in_order_ONLYF(seq):
    """Remove duplicates while preserving order, keep only F codes"""
    seen = set()
    new_seq = []
    for code in seq:
        if code.startswith("F"):
            if code not in seen:
                seen.add(code)
                new_seq.append(code)
    return new_seq

def four_to_three_digits(seq):
    """Remove duplicates while preserving order"""
    return [x[:3] for x in seq]


seq_df['f_sequence_only_new'] = seq_df['sequence'].apply(four_to_three_digits).apply(unique_in_order_ONLYF)

# Data subset preparation
seq_df['length'] = seq_df['f_sequence_only_new'].str.len()
seq_df = seq_df[seq_df['length'] >= 2]

tot_pats = len(seq_df)
unique_seq = len({tuple(seq) for seq in seq_df['f_sequence_only_new']})

####
print(f"Total patients in sample: {tot_pats}")
print(f"Unique sequences: {round(100 * unique_seq / tot_pats, 2)}%")
####

seq_df_m = seq_df.copy().iloc[:10_000]
seq_df_m_small = (seq_df_m[seq_df_m["f_sequence_only_new"].apply(
    lambda x: any(c.startswith("F") for c in x) and len(x) >= 2)]
                  .dropna())


def build_cost_matrix(seq_df, method="npmi", plot_probabilities=False):
    """
    Build a cost matrix for diagnosis substitution based on co-occurrences or nPMI.
    Returns a dense numpy array for Numba compatibility.
    """
    # --- 1. Setup and mappings ---
    all_codes = sorted(set(chain.from_iterable(seq_df['f_sequence_only_new'])))
    code_to_index = {code: i for i, code in enumerate(all_codes)}
    n_codes = len(all_codes)

    # --- 2. Marginal counts ---
    code_counts = Counter(chain.from_iterable(seq_df['f_sequence_only_new']))
    counts_array = np.array([code_counts[c] for c in all_codes], dtype=np.float64)

    # --- 3. Co-occurrence matrix ---
    co_occurrence_matrix = np.zeros((n_codes, n_codes), dtype=np.int64)
    sequences_to_use = seq_df['f_sequence_only_new'].tolist()

    for seq in tqdm(sequences_to_use, desc="Calculating co-occurrences"):
        unique_in_seq = list(set(seq))
        indices = [code_to_index[c] for c in unique_in_seq]
        n_unique = len(indices)

        for i in range(n_unique):
            idx_i = indices[i]
            co_occurrence_matrix[idx_i, idx_i] += 1
            for j in range(i + 1, n_unique):
                idx_j = indices[j]
                co_occurrence_matrix[idx_i, idx_j] += 1
                co_occurrence_matrix[idx_j, idx_i] += 1

    # --- 4. Probability computations ---
    total_sequences = len(sequences_to_use)
    p_x = counts_array / total_sequences
    p_xy = co_occurrence_matrix / total_sequences

    # --- 5. Cost matrix calculation ---
    cost_matrix = np.zeros((n_codes, n_codes), dtype=np.float64)

    if method == "npmi":
        eps = 1e-12
        p_prod = np.outer(p_x, p_x)
        valid_mask = (p_xy > 0) & (p_prod > 0)

        ratio = np.zeros_like(p_xy)
        ratio[valid_mask] = p_xy[valid_mask] / p_prod[valid_mask]

        pmi = np.full_like(p_xy, -10)
        pmi[valid_mask] = np.log2(ratio[valid_mask])

        npmi = np.full_like(pmi, -1)
        npmi[valid_mask] = pmi[valid_mask] / (-np.log2(p_xy[valid_mask]))

        cost_matrix = 1 - ((npmi + 1) / 2)
        cost_matrix[~np.isfinite(cost_matrix)] = 0.0
        cost_matrix = np.clip(cost_matrix, 0, 1)

    elif method == "cooccurrence":
        for i in tqdm(range(n_codes), desc="Building cost matrix (co-occurrence)"):
            count_a = counts_array[i]
            if count_a > 0:
                p_b_given_a = co_occurrence_matrix[i, :] / count_a
            else:
                p_b_given_a = np.zeros(n_codes)

            p_a_given_b = np.divide(co_occurrence_matrix[i, :], counts_array,
                                    out=np.zeros(n_codes), where=counts_array > 0)

            cost_matrix[i, :] = 1 - np.maximum(p_a_given_b, p_b_given_a)
    else:
        raise ValueError("method must be either 'cooccurrence' or 'npmi'")

    np.fill_diagonal(cost_matrix, 0.0)
    return cost_matrix, code_to_index, all_codes


# Build the cost matrix
print("Building cost matrix...")
cost_matrix, code_to_index, all_codes = build_cost_matrix(seq_df_m_small, method="npmi")

# Convert sequences to indices
sequences_indices = [np.array([code_to_index[c] for c in seq], dtype=np.int32)
                     for seq in seq_df_m_small["f_sequence_only_new"]]

print(f"Number of sequences: {len(sequences_indices)}")
print(f"Cost matrix shape: {cost_matrix.shape}")


# HIGHLY OPTIMIZED DTW IMPLEMENTATION
@jit(nopython=True, fastmath=True, cache=True)
def dtw_custom_cost_fast(s1, s2, cost_matrix):
    """
    Ultra-fast DTW implementation using Numba with custom cost matrix.
    Uses integer indices for direct array access - much faster than dictionary lookups.
    """
    n = len(s1)
    m = len(s2)

    # Use two rows instead of full matrix to save memory
    prev_row = np.full(m + 1, np.inf)
    curr_row = np.full(m + 1, np.inf)
    prev_row[0] = 0

    for i in range(1, n + 1):
        curr_row[0] = np.inf
        for j in range(1, m + 1):
            # Direct array access - much faster than dictionary lookups
            cost = cost_matrix[s1[i - 1], s2[j - 1]]
            curr_row[j] = cost + min(prev_row[j],  # insertion
                                     curr_row[j - 1],  # deletion
                                     prev_row[j - 1])  # match
        # Swap rows
        prev_row, curr_row = curr_row, prev_row

    final_dtw_distance = prev_row[m]

    # Normalization Step: Divide by the average sequence length
    normalization_factor = (len(s1) + len(s2)) / 2

    # Avoid division by zero
    if normalization_factor == 0:
        return 0.0

    normalized_distance = final_dtw_distance / normalization_factor
    return normalized_distance


def chunked_dtw_matrix(sequences_indices, cost_matrix, chunk_size=500, output_file=None):
    """
    Calculate DTW distance matrix in chunks to avoid memory issues.
    Uses memory-mapped arrays to store results on disk.
    """
    N = len(sequences_indices)

    if output_file is None:
        output_file = "dtw_distance_matrix_chunked.npy"

    print(f"Creating memory-mapped array of size {N}x{N}...")
    print(f"This will require approximately {N * N * 4 / (1024 ** 3):.2f} GB of disk space")

    # Initialize output file with memmap for large arrays (using float32 to save space)
    dist_matrix = np.memmap(output_file, dtype=np.float32, mode='w+', shape=(N, N))

    # Initialize with zeros on diagonal
    print("Initializing diagonal with zeros...")
    for i in tqdm(range(N)):
        dist_matrix[i, i] = 0.0

    # Process in chunks
    total_chunks = (N + chunk_size - 1) // chunk_size
    print(f"Processing {total_chunks} chunks of size {chunk_size}...")

    for i in tqdm(range(0, N, chunk_size), desc="Processing row chunks"):
        i_end = min(i + chunk_size, N)

        for j in tqdm(range(0, N, chunk_size), desc=f"Rows {i}-{i_end}", leave=False):
            j_end = min(j + chunk_size, N)

            # Process this chunk
            for idx_i in range(i, i_end):
                s1 = sequences_indices[idx_i]
                for idx_j in range(j, j_end):
                    if idx_j > idx_i:  # Only compute upper triangle
                        dist = dtw_custom_cost_fast(s1, sequences_indices[idx_j], cost_matrix)
                        dist_matrix[idx_i, idx_j] = dist
                        dist_matrix[idx_j, idx_i] = dist  # Symmetric matrix

    # Flush to disk
    dist_matrix.flush()
    print("Calculation completed and flushed to disk!")
    return dist_matrix


def optimized_dtw_matrix(sequences_indices, cost_matrix, chunk_size=1000):
    """
    Optimized single-threaded DTW calculation with chunking.
    Alternative implementation without memory mapping.
    """
    N = len(sequences_indices)
    dist_matrix = np.zeros((N, N), dtype=np.float32)

    # Fill diagonal with zeros
    for i in range(N):
        dist_matrix[i, i] = 0.0

    # Process in chunks to manage memory
    for i in tqdm(range(0, N, chunk_size), desc="DTW Calculation"):
        i_end = min(i + chunk_size, N)

        for idx_i in range(i, i_end):
            s1 = sequences_indices[idx_i]
            for idx_j in range(idx_i + 1, N):  # Only compute upper triangle
                dist = dtw_custom_cost_fast(s1, sequences_indices[idx_j], cost_matrix)
                dist_matrix[idx_i, idx_j] = dist
                dist_matrix[idx_j, idx_i] = dist  # Symmetric matrix

    return dist_matrix


# Choose your preferred method based on available memory
print("Calculating DTW distances...")

# Method 1: Memory-mapped chunked processing (recommended for large datasets)
output_path = "C:\\git-projects\\mental_health\\sequence_analysis\\common_disease_trajectories\\Simons_distances\\distance_matrices\\dtw_distance_matrix_MEN_ONLY_F_4DIGITS.dat"

# Adjust chunk_size based on your available RAM
# Smaller chunk_size = less memory usage but slower
# Larger chunk_size = more memory usage but faster
chunk_size = 5000

try:
    dist_full = chunked_dtw_matrix(sequences_indices, cost_matrix,
                                   chunk_size=chunk_size,
                                   output_file=output_path)
    print("DTW calculation completed using memory-mapped chunks!")

except MemoryError:
    print("Memory error with chunked processing. Trying with smaller chunk size...")
    chunk_size = 100
    dist_full = chunked_dtw_matrix(sequences_indices, cost_matrix,
                                   chunk_size=chunk_size,
                                   output_file=output_path)

# If you want to convert the memory-mapped file to a regular .npy file:
print("Converting memory-mapped file to .npy format...")
dist_array = np.memmap(output_path, dtype=np.float32, mode='r', shape=(len(sequences_indices), len(sequences_indices)))
np.save(
    "C:\\git-projects\\mental_health\\sequence_analysis\\common_disease_trajectories\\Simons_distances\\distance_matrices\\dtw_distance_matrix_MEN_ONLY_F_4DIGITS.npy",
    dist_array)

# Save patient IDs
ids = np.array(seq_df_m_small["patient_no"].tolist())
np.save(
    "C:\\git-projects\\mental_health\\sequence_analysis\\common_disease_trajectories\\Simons_distances\\distance_matrices\\patient_ids_MEN_ONLY_F_4DIGITS.npy",
    ids)
print("Patient IDs saved!")

print("All operations completed successfully!")