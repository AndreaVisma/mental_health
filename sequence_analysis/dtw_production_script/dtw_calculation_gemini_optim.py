import json
import pandas as pd
from itertools import chain
from collections import Counter
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from numba import jit, prange
import os
import sys
# from scipy.cluster.hierarchy import linkage, fcluster # Not used in main run
# from scipy.spatial.distance import squareform # Not used in main run
# import re # Not used
# import matplotlib.pyplot as plt # Not used in main run

# --- CONFIGURATION ---
plot_folder = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\figures\\"
output_path = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\dtw_distance_matrix_MEN_ONLY_F_3DIGITS.npy"
patient_ids_path = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\patient_ids_MEN_ONLY_F_4DIGITS.npy"
metadata_path = "C:\\Data\\my_datasets\\medical\\patients_metadata.csv"
sequence_data_path = "C:\\Data\\my_datasets\\medical\\diagnosis_sequences.parquet"

# --- DATA LOADING AND PRE-PROCESSING (Optimized) ---

# metadata to select subsample
print("Loading metadata and filtering...")
metadata = pd.read_csv(metadata_path)
males = metadata.query("sex_id == 1")["patient_no"].tolist()

# Load data
seq_df = pd.read_parquet(sequence_data_path)
seq_df = seq_df[seq_df.patient_no.isin(males)].copy()
seq_df.rename(columns={"code": "sequence"}, inplace=True)
seq_df['patient_no'] = seq_df['patient_no'].astype(int)


def four_to_three_digits_optimized(seq):
    """Truncate to 3 digits."""
    return [x[:3] for x in seq]

def unique_in_order_ONLYF_optimized(seq):
    """Remove duplicates while preserving order, keep only F codes (mental health)."""
    seen = set()
    new_seq = []
    for code in seq:
        # Check F prefix and uniqueness simultaneously
        if code.startswith("F") and code not in seen:
            seen.add(code)
            new_seq.append(code)
    return new_seq

print("Applying sequence transformations (optimized list comprehension)...")
# Chain the transformations using a single list comprehension for better performance
# 1. Truncate codes, 2. Filter F codes and keep unique in order
sequences_transformed = [
    unique_in_order_ONLYF_optimized(four_to_three_digits_optimized(seq))
    for seq in tqdm(seq_df['sequence'], desc="Transforming sequences")
]
seq_df['f_sequence_only_new'] = sequences_transformed


# Data subset preparation
seq_df['length'] = seq_df['f_sequence_only_new'].str.len()
seq_df_m_small = seq_df[seq_df['length'] >= 2].copy()

tot_pats = len(seq_df_m_small)
unique_seq = len({tuple(seq) for seq in seq_df_m_small['f_sequence_only_new']})

####
print(f"\nTotal patients in final sample: {tot_pats}")
print(f"Unique sequences: {round(100 * unique_seq / tot_pats, 2)}%")
####

# --- COST MATRIX BUILDING (Optimized Co-occurrence) ---

def build_cost_matrix(seq_df, method="npmi"):
    """
    Build a cost matrix for diagnosis substitution based on co-occurrences or nPMI.
    Returns a dense numpy array for Numba compatibility.
    """
    print("--- Building Cost Matrix ---")
    # --- 1. Setup and mappings ---
    all_codes = sorted(set(chain.from_iterable(seq_df['f_sequence_only_new'])))
    code_to_index = {code: i for i, code in enumerate(all_codes)}
    n_codes = len(all_codes)
    sequences_to_use = seq_df['f_sequence_only_new'].tolist()
    total_sequences = len(sequences_to_use)

    # --- 2. Marginal counts ---
    code_counts = Counter(chain.from_iterable(sequences_to_use))
    counts_array = np.array([code_counts[c] for c in all_codes], dtype=np.float64)

    # --- 3. Co-occurrence matrix (Vectorized Optimization) ---
    co_occurrence_matrix = np.zeros((n_codes, n_codes), dtype=np.int64)

    print("Calculating co-occurrences (vectorized)...")

    # Pre-calculate unique indices for all sequences
    all_unique_indices = [np.unique([code_to_index[c] for c in seq])
                          for seq in sequences_to_use]

    for unique_indices in tqdm(all_unique_indices, desc="Vectorizing co-occurrences"):

        # Create a dense indicator vector for codes present in the sequence
        present_mask = np.zeros(n_codes, dtype=bool)
        present_mask[unique_indices] = True

        # Vectorized outer product: adds 1 to co_occurrence_matrix[i, j]
        # if both code i and code j are present in the sequence.
        co_occurrence_matrix += np.outer(present_mask, present_mask)

        # --- 4. Probability computations ---
    p_x = counts_array / total_sequences
    p_xy = co_occurrence_matrix / total_sequences

    # --- 5. Cost matrix calculation (NPMI) ---
    cost_matrix = np.zeros((n_codes, n_codes), dtype=np.float64)

    if method == "npmi":
        eps = 1e-12
        p_prod = np.outer(p_x, p_x)

        # Avoid log(0) and division by 0
        valid_mask = (p_xy > 0) & (p_prod > 0)

        ratio = np.zeros_like(p_xy)
        ratio[valid_mask] = p_xy[valid_mask] / p_prod[valid_mask]

        pmi = np.full_like(p_xy, -10.0) # Low value for unconnected pairs
        pmi[valid_mask] = np.log2(ratio[valid_mask])

        npmi = np.full_like(pmi, -1.0) # Default for P(XY)=0 or P(X)P(Y)=0

        # Calculate NPMI only where P(XY) > 0
        npmi_mask = p_xy > 0
        npmi[npmi_mask] = pmi[npmi_mask] / (-np.log2(p_xy[npmi_mask] + eps)) # Added eps for safety

        # Map NPMI (-1 to 1) to cost (0 to 1). 1 -> 0 (Cost), -1 -> 1 (Cost)
        cost_matrix = 1 - ((npmi + 1) / 2)

        # Handle potential edge cases (should be rare with good eps handling)
        cost_matrix[~np.isfinite(cost_matrix)] = 1.0
        cost_matrix = np.clip(cost_matrix, 0.0, 1.0) # Ensure it's between 0 and 1

    else:
        raise ValueError("method must be 'npmi'")

    np.fill_diagonal(cost_matrix, 0.0)
    print("Cost matrix built successfully.")
    return cost_matrix, code_to_index, all_codes

# Build the cost matrix
cost_matrix, code_to_index, all_codes = build_cost_matrix(seq_df_m_small, method="npmi")

# Convert sequences to indices
sequences_indices = [np.array([code_to_index[c] for c in seq], dtype=np.int32)
                     for seq in seq_df_m_small["f_sequence_only_new"]]

print(f"Number of sequences: {len(sequences_indices)}")
print(f"Cost matrix shape: {cost_matrix.shape}")

# --- DTW CALCULATION (Numba + Parallelization) ---

# HIGHLY OPTIMIZED DTW IMPLEMENTATION
@jit(nopython=True, fastmath=True, cache=True)
def dtw_custom_cost_fast(s1, s2, cost_matrix):
    """
    Ultra-fast DTW implementation using Numba with custom cost matrix.
    Uses integer indices for direct array access.
    """
    n = len(s1)
    m = len(s2)

    if n == 0 or m == 0:
        return 0.0

    # Use two rows instead of full matrix to save memory
    prev_row = np.full(m + 1, np.inf)
    curr_row = np.full(m + 1, np.inf)
    prev_row[0] = 0.0 # Start point cost

    for i in range(1, n + 1):
        curr_row[0] = np.inf
        for j in range(1, m + 1):
            # Direct array access
            cost = cost_matrix[s1[i - 1], s2[j - 1]]

            # DTW recurrence relation: Match/Substitution, Deletion, Insertion
            curr_row[j] = cost + min(prev_row[j],    # Deletion (move from i-1 to i, j stays)
                                     curr_row[j - 1],  # Insertion (move from j-1 to j, i stays)
                                     prev_row[j - 1]) # Match/Substitution (move from i-1, j-1 to i, j)

        # Swap rows
        prev_row[:] = curr_row[:] # Numba-friendly assignment

    final_dtw_distance = prev_row[m]

    # Normalization Step: Divide by the average sequence length
    normalization_factor = (n + m) / 2

    # Normalization helps compare distances between sequences of different lengths
    normalized_distance = final_dtw_distance / normalization_factor
    return normalized_distance

# --- PARALLEL WORKER AND EXECUTOR ---
def dtw_worker_task(s1, s2, cost_matrix):
    """Worker function for parallel DTW calculation, receiving only sequence data."""
    # dtw_custom_cost_fast is already Numba-jitted
    return dtw_custom_cost_fast(s1, s2, cost_matrix)


def parallel_chunked_dtw_matrix(sequences_indices, cost_matrix, output_file, chunk_size=5000, max_workers=None):
    """
    Calculates DTW distance matrix in chunks using ProcessPoolExecutor and memory mapping.
    Avoids building the large list of tasks in RAM.
    """
    N = len(sequences_indices)

    if max_workers is None:
        max_workers = mp.cpu_count()
    print(f"Using {max_workers} processes and chunk size {chunk_size}...")

    # Initialize output file with memmap for large arrays
    dist_matrix = np.memmap(output_file, dtype=np.float32, mode='w+', shape=(N, N))

    # Initialize with zeros on diagonal
    for i in tqdm(range(N), desc="Initializing diagonal"):
        dist_matrix[i, i] = 0.0

    total_chunks = (N + chunk_size - 1) // chunk_size
    print(f"Processing in chunks of size {chunk_size}...")

    # Use a shared (global) reference for cost_matrix in the process pool
    # This avoids copying it for every single task.

    for i in tqdm(range(0, N, chunk_size), desc="Processing row chunks"):
        i_end = min(i + chunk_size, N)

        # Inner loop: Iterate over the remaining columns for the current row chunk
        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)

            # --- Build a small, local batch of tasks for this chunk ---
            batch_tasks = []

            # Only iterate over the upper triangle or the square diagonal chunk (i <= j)
            for idx_i in range(i, i_end):
                s1 = sequences_indices[idx_i]
                for idx_j in range(j, j_end):
                    if idx_j > idx_i:  # Only compute the strictly upper triangle
                        s2 = sequences_indices[idx_j]
                        batch_tasks.append((idx_i, idx_j, s1, s2, cost_matrix))

            if not batch_tasks:
                continue

            # --- Execute the batch in parallel ---
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use executor.map to execute the small batch
                results = list(executor.map(dtw_worker_for_batch, batch_tasks))

            # --- Write results directly to the memory-mapped array ---
            for (idx_i, idx_j, _, _, _), dist in zip(batch_tasks, results):
                dist_matrix[idx_i, idx_j] = dist
                dist_matrix[idx_j, idx_i] = dist  # Symmetric

            # Flush every inner chunk to ensure progress is saved
            dist_matrix.flush()

    # Final flush to disk
    dist_matrix.flush()
    print("Calculation completed and flushed to disk!")
    return dist_matrix


# Modify the worker to handle the task structure from the batch
def dtw_worker_for_batch(task):
    """Worker function for parallel DTW calculation used by the chunking logic."""
    idx_i, idx_j, s1, s2, cost_matrix = task
    # We only return the distance here; the indices are held in batch_tasks
    return dtw_custom_cost_fast(s1, s2, cost_matrix)


if __name__ == "__main__":
    # --- MAIN EXECUTION BLOCK ---

    # Define the output path for the memory-mapped file
    # Note: Use a reliable path that exists.
    mmapped_output_path = output_path.replace(".npy", ".dat")

    print("Calculating DTW distances (Parallelized Memory-Mapped Chunks)...")

    # Configuration for execution
    CHUNK_SIZE = 5000
    num_cores = mp.cpu_count()  # Use all available cores

    # Check if we can safely reduce CHUNK_SIZE if N is small
    N = len(sequences_indices)
    safe_chunk_size = min(N, CHUNK_SIZE)

    print(f"Total sequences (N): {N}")
    print(f"Using {num_cores} cores with CHUNK_SIZE={safe_chunk_size}")

    try:
        # Run the parallel, chunked, and memory-mapped DTW calculation
        dist_mmap = parallel_chunked_dtw_matrix(
            sequences_indices,
            cost_matrix,
            output_file=mmapped_output_path,
            chunk_size=safe_chunk_size,
            max_workers=num_cores
        )

        # Convert the memory-mapped file to a regular .npy file after calculation
        print("\nConverting memory-mapped file to .npy format...")
        # Reading the data from disk into a regular NumPy array
        dist_full = np.array(dist_mmap, copy=True)
        np.save(output_path, dist_full)
        print(f"Final distance matrix saved to {output_path}")

        # Delete the memory-mapped file reference (optional cleanup)
        del dist_mmap

    except Exception as e:
        print(f"\nAn error occurred during parallel DTW calculation: {e}")
        sys.exit(1)

    # Save patient IDs
    ids = np.array(seq_df_m_small["patient_no"].tolist())
    print(f"Saving patient IDs to {patient_ids_path}")
    np.save(patient_ids_path, ids)
    print("Patient IDs saved!")

    print("All operations completed successfully! 🎉")