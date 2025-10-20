

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


seq_df['f_sequence_only_new'] = seq_df['sequence'].apply(unique_in_order_ONLYF)

# Data subset preparation
seq_df['length'] = seq_df['f_sequence_only_new'].str.len()
seq_df = seq_df[seq_df['length'] >= 2]

tot_pats = len(seq_df)
unique_seq = len({tuple(seq) for seq in seq_df['f_sequence_only_new']})

####
print(f"Total patients in sample: {tot_pats}")
print(f"Unique sequences: {round(100 * unique_seq / tot_pats, 2)}%")
####

seq_df_m = seq_df.copy()
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
cost_matrix, code_to_index, all_codes = build_cost_matrix(seq_df_m_small, method="npmi")

# Convert sequences to indices
sequences_indices = [np.array([code_to_index[c] for c in seq], dtype=np.int32)
                     for seq in seq_df_m_small["f_sequence_only_new"]]


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


def calculate_dtw_pair(args):
    """Calculate DTW for a single pair - for parallel processing"""
    i, j, sequences, cost_matrix = args
    if i == j:
        return i, j, 0.0
    else:
        dist = dtw_custom_cost_fast(sequences[i], sequences[j], cost_matrix)
        return i, j, dist


def parallel_dtw_matrix(sequences_indices, cost_matrix, n_jobs=None):
    """Calculate DTW distance matrix in parallel"""
    if n_jobs is None:
        n_jobs = mp.cpu_count() - 1  # Leave one core free

    N = len(sequences_indices)
    dist_matrix = np.zeros((N, N))

    # Prepare arguments for parallel processing
    tasks = []
    for i in range(N):
        for j in range(i, N):
            tasks.append((i, j, sequences_indices, cost_matrix))

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(calculate_dtw_pair, tasks),
                            total=len(tasks), desc="DTW Calculation"))

    # Fill the matrix
    for i, j, dist in results:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

    return dist_matrix


# Single-threaded version for testing
def optimized_dtw_matrix(sequences_indices, cost_matrix):
    """Optimized single-threaded DTW calculation"""
    N = len(sequences_indices)
    dist_matrix = np.zeros((N, N))

    for i in tqdm(range(N), desc="DTW Calculation"):
        s1 = sequences_indices[i]
        for j in range(i, N):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist = dtw_custom_cost_fast(s1, sequences_indices[j], cost_matrix)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    return dist_matrix


# Choose your preferred method:
print("Calculating DTW distances...")

# For testing with a small subset first:
test_size = min(100, len(sequences_indices))
test_sequences = sequences_indices[:test_size]
test_dist = optimized_dtw_matrix(test_sequences, cost_matrix)
print(f"Test completed with {test_size} sequences")

# For full dataset - use parallel version for larger datasets
if len(sequences_indices) > 1000:
    print("Using parallel processing for large dataset...")
    dist_full = parallel_dtw_matrix(sequences_indices, cost_matrix)
else:
    print("Using single-threaded processing...")
    dist_full = optimized_dtw_matrix(sequences_indices, cost_matrix)

print("DTW calculation completed!")

# Save results
np.save(
    "C:\\git-projects\\mental_health\\sequence_analysis\\common_disease_trajectories\\Simons_distances\\distance_matrices\\dtw_distance_matrix_MEN_ONLY_F_4DIGITS.npy",
    dist_full)
print("DTW distances matrix saved ;)")

ids = np.array(seq_df_m_small["patient_no"].tolist())
np.save(
    "C:\\git-projects\\mental_health\\sequence_analysis\\common_disease_trajectories\\Simons_distances\\distance_matrices\\patient_ids_MEN_ONLY_F_4DIGITS.npy",
    ids)
print("patient IDs saved ;)")