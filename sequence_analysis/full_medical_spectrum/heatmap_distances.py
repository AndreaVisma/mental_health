

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

plots_path = "C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\figures\\"

gender = "F"
if gender == "F":
    sex_id = 2
else:
    sex_id = 1

sample = True
sample_size = 20_000
####
# metadata to select subsample
metadata = pd.read_csv("C:\\Data\\my_datasets\\medical\\patients_metadata.csv")
females = metadata.query(f"sex_id == {sex_id}")["patient_no"].tolist()

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences.parquet")
seq_df = seq_df[seq_df.patient_no.isin(females)]
seq_df.rename(columns={"code": "sequence"}, inplace=True)

seq_df['patient_no'] = seq_df['patient_no'].astype(int)

def four_to_three_digits(seq):
    """Remove duplicates while preserving order"""
    return [x[:3] for x in seq]

seq_df['sequence'] = seq_df['sequence'].apply(four_to_three_digits)

# Optimized unique codes extraction
unique_codes = sorted(set(chain.from_iterable(seq_df['sequence'])))
code_counts = Counter(chain.from_iterable(seq_df['sequence']))
top_100_codes = [c for c, _ in code_counts.most_common(100)]

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

def unique_in_order_top100_and_F(seq):
    """Keep unique codes in original order, restricted to F-codes or top 100 frequent codes."""
    seen = set()
    out = []
    for code in seq:
        if code.startswith("F") or code in top_100_codes:
            if code not in seen:
                seen.add(code)
                out.append(code)
    return out

def keep_f(seq):
    if any(code.startswith("F") for code in seq):
        return True
    else:
        return False

seq_df['f_sequence_only_new'] = seq_df['sequence'].apply(unique_in_order_top100_and_F)
seq_df["has_F"] = seq_df['f_sequence_only_new'] .apply(keep_f)

# Data subset preparation
seq_df['length'] = seq_df['f_sequence_only_new'].str.len()
seq_df = seq_df[(seq_df['length'] >= 3) & (seq_df.has_F == True)]

tot_pats = len(seq_df)
unique_seq = len({tuple(seq) for seq in seq_df['f_sequence_only_new']})

####
print(f"Total patients in sample: {tot_pats}")
print(f"Unique sequences: {round(100 * unique_seq / tot_pats, 2)}%")
####

if sample:
    seq_df_m = seq_df.copy().sample(sample_size)
else:
    seq_df_m = seq_df.copy()

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
cost_matrix, code_to_index, all_codes = build_cost_matrix(seq_df_m, method="npmi")

## plot heatmap

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

fig = px.imshow(cost_matrix,
                labels=dict(x="Diagnoses", y="Diagnoses", color="Dissimilarity"),
                x=all_codes,
                y=all_codes, color_continuous_scale='speed'
               )
fig.update_xaxes(side="top")
fig.write_html(plots_path + "dissimilarity_matrix.html")
fig.show()