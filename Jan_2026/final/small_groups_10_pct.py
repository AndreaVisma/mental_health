

import json
import pandas as pd
from itertools import chain
from collections import Counter
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import re
from numba import njit, prange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from scipy import sparse

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences_only_f.parquet")

seq_df['patient_no'] = seq_df['patient_no'].astype(int)

import pyreadr
result = pyreadr.read_r('c://data//my_datasets//medical//patients_data_full.rds')
elmas_metadata = result[None] # extract the pandas data frame
elmas_metadata["patient_no"] = elmas_metadata["patient_no"].astype(int)
elmas_metadata["mortality"].fillna(0, inplace = True)

seq_df = seq_df.merge(elmas_metadata[["patient_no", "age_central_date"]], on = "patient_no", how = "left")
seq_df["age"] = (seq_df["age_central_date"] - 1) * 5
seq_df["age_group_1"] = "young 0-39"
seq_df.loc[seq_df.age > 39, "age_group_1"] ="midlife 39-64"
seq_df.loc[seq_df.age > 64, "age_group_1"] ="old 65+"

## blocks
blocks = pd.read_csv("C://users//Andrea Vismara//downloads//Blocks_All.csv")

dict_blocks = dict(zip(blocks["icd_code"], blocks["block_name"]))
dict_blocks['A90'] = "A92-A99"

def transform_sequences_block(seq):
    seen = set()
    out = []
    for x in [dict_blocks[i] for i in seq]:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

seq_df['sequence_unique_blocks'] = seq_df['sequence_uniques'].apply(transform_sequences_block)
seq_df['len_block_seq'] = seq_df['sequence_unique_blocks'].apply(lambda x: len(set(x)))
seq_df = seq_df[seq_df.len_block_seq >=2]

####
sex_dict = {1:'male', 2:'female'}
for age_group in seq_df.age_group_1.unique():
    for sex_id in seq_df.sex_id.unique():
        mini_df = seq_df[(seq_df.age_group_1 == age_group) & (seq_df.sex_id == sex_id)]
        print(f"Age: {age_group}, sex: {sex_dict[sex_id]}")
        print(f"patients in group: {len(mini_df)}")
        print("===========")


##########
def build_cost_matrix(seq_df, method="npmi", plot_probabilities=False):
    """
    Build a cost matrix for diagnosis substitution based on co-occurrences or nPMI.
    Returns a dense numpy array for Numba compatibility.
    """
    # --- 1. Setup and mappings ---
    all_codes = sorted(set(chain.from_iterable(seq_df['sequence_unique_blocks'])))
    code_to_index = {code: i for i, code in enumerate(all_codes)}
    n_codes = len(all_codes)

    # --- 2. Marginal counts ---
    code_counts = Counter(chain.from_iterable(seq_df['sequence_unique_blocks']))
    counts_array = np.array([code_counts[c] for c in all_codes], dtype=np.float64)

    # --- 3. Co-occurrence matrix ---
    co_occurrence_matrix = np.zeros((n_codes, n_codes), dtype=np.int64)
    sequences_to_use = seq_df['sequence_unique_blocks'].tolist()

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
        #
        # mask = np.where(npmi <=0, True, False)
        cost_matrix = 1 - ((npmi + 1) / 2)
        # cost_matrix[mask] = 1
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

def dtw_distance(s1, s2, cost_matrix, normalize=False):
    """
    Simple and readable DTW with a custom cost matrix.

    s1, s2: sequences of integer indices
    cost_matrix: 2D array, cost_matrix[a, b] = cost of matching a with b
    normalize: divide final distance by average sequence length
    """
    n, m = len(s1), len(s2)

    # Two rows for dynamic programming
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            cost = cost_matrix[s1[i - 1], s2[j - 1]]
            curr[j] = cost + min(
                prev[j],      # insertion
                curr[j - 1],  # deletion
                prev[j - 1]   # match
            )
        prev, curr = curr, prev

    distance = prev[m]

    if normalize:
        length = (n + m) / 2
        return distance / length if length > 0 else 0.0

    return distance

def dtw_distance_matrix(sequences, cost_matrix, normalize=False):
    """
    Compute full DTW distance matrix for a list of sequences.
    """
    N = len(sequences)
    dist = np.zeros((N, N), dtype=np.float32)

    for i in tqdm(range(N)):
        for j in range(i + 1, N):
            d = dtw_distance(
                sequences[i],
                sequences[j],
                cost_matrix,
                normalize=normalize
            )
            dist[i, j] = d
            dist[j, i] = d  # symmetry

    return dist


@njit
def dtw_distance_numba(s1, s2, cost_matrix, normalize):
    n = len(s1)
    m = len(s2)

    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            cost = cost_matrix[s1[i - 1], s2[j - 1]]
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    d = prev[m]
    if normalize:
        length = (n + m) * 0.5
        return d / length if length > 0 else 0.0
    return d

@njit(parallel=True)
def dtw_distance_matrix_numba(sequences, cost_matrix, normalize):
    N = len(sequences)
    dist = np.zeros((N, N), dtype=np.float32)

    for i in prange(N):
        for j in range(i + 1, N):
            d = dtw_distance_numba_path_len_norm(sequences[i], sequences[j], cost_matrix, normalize)
            dist[i, j] = d
            dist[j, i] = d
    return dist

##############
# normalise by path length
@njit
def dtw_distance_numba_path_len_norm(s1, s2, cost_matrix, normalize):
    n = len(s1)
    m = len(s2)

    prev_cost = np.full(m + 1, np.inf)
    curr_cost = np.full(m + 1, np.inf)

    prev_len = np.zeros(m + 1, dtype=np.int32)
    curr_len = np.zeros(m + 1, dtype=np.int32)

    prev_cost[0] = 0.0
    prev_len[0] = 0

    for i in range(1, n + 1):
        curr_cost[0] = np.inf
        curr_len[0] = 0

        for j in range(1, m + 1):
            cost = cost_matrix[s1[i - 1], s2[j - 1]]

            # find best predecessor
            c_up = prev_cost[j]
            c_left = curr_cost[j - 1]
            c_diag = prev_cost[j - 1]

            if c_diag <= c_up and c_diag <= c_left:
                curr_cost[j] = cost + c_diag
                curr_len[j] = prev_len[j - 1] + 1
            elif c_up <= c_left:
                curr_cost[j] = cost + c_up
                curr_len[j] = prev_len[j] + 1
            else:
                curr_cost[j] = cost + c_left
                curr_len[j] = curr_len[j - 1] + 1

        prev_cost, curr_cost = curr_cost, prev_cost
        prev_len, curr_len = curr_len, prev_len

    d = prev_cost[m]
    L = prev_len[m]

    if normalize:
        return d / L if L > 0 else 0.0
    return d
##############

sex_dict = {1:'male', 2:'female'}
for age_group in tqdm(['young 0-39', 'midlife 39-64']):
    for sex_id in tqdm([1, 2]):
        mini_df = seq_df[(seq_df.age_group_1 == age_group) & (seq_df.sex_id == sex_id)]
        # subsample = mini_df.sample(int(len(mini_df) * 0.4))
        subsample = mini_df.copy()
        ids = subsample["patient_no"].tolist()

        print(f"Building cost matrix for {sex_dict[sex_id]}s, age {age_group}...")
        cost_matrix, code_to_index, all_codes = build_cost_matrix(mini_df, method="npmi")

        sequences_indices = [np.array([code_to_index[c] for c in seq], dtype=np.int32)
                             for seq in subsample["sequence_unique_blocks"]]

        d = dtw_distance_matrix_numba(sequences_indices, cost_matrix, normalize=True)
        np.save(
            f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\dtw_full_{sex_dict[sex_id]}_{age_group}_norm_PATH_LEN.npy",
            d)
        # np.save(
        #     f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\IDS_full_{sex_dict[sex_id]}_{age_group}.npy",
        #     ids)
        print('done')