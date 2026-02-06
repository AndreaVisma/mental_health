
import pandas as pd
from itertools import chain
from collections import Counter
import numpy as np
from tqdm import tqdm

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

#####################

def compute_cost_npmi(seq_df):
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
    p_prod = np.outer(p_x, p_x)
    valid_mask = (p_xy > 0) & (p_prod > 0)

    ratio = np.zeros_like(p_xy)
    ratio[valid_mask] = p_xy[valid_mask] / p_prod[valid_mask]

    pmi = np.full_like(p_xy, -10)
    pmi[valid_mask] = np.log2(ratio[valid_mask])

    npmi = np.full_like(pmi, -1)
    npmi[valid_mask] = pmi[valid_mask] / (-np.log2(p_xy[valid_mask]))
    #
    cost_matrix = 1 - ((npmi + 1) / 2)
    cost_matrix[~np.isfinite(cost_matrix)] = 0.0
    cost_matrix = np.clip(cost_matrix, 0, 1)

    np.fill_diagonal(cost_matrix, np.nan)
    return cost_matrix, code_to_index, all_codes


#########
dict_npmis = {}
sex_dict = {1:'male', 2:'female'}
for age_group in tqdm(['young 0-39', 'midlife 39-64', 'old 65+']):
    for sex_id in tqdm([1, 2]):
        mini_df = seq_df[(seq_df.age_group_1 == age_group) & (seq_df.sex_id == sex_id)]
        subsample = mini_df.copy()
        ids = subsample["patient_no"].tolist()

        print(f"Building cost matrix for {sex_dict[sex_id]}s, age {age_group}...")
        cost_matrix, code_to_index, all_codes = compute_cost_npmi(mini_df)
        dict_npmis[(sex_id, age_group)] = (cost_matrix, code_to_index, all_codes)

###############
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import os

output_dir = "npmi_matrices"
os.makedirs(output_dir, exist_ok=True)

def plot_cost_matrix(cost_matrix, labels, title):
    fig = go.Figure(
        data=go.Heatmap(
            z=cost_matrix,
            x=labels,
            y=labels,
            colorscale="Brwnyl",
            colorbar=dict(title="Dissimilarity"),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=900,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=9),
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(size=9),
            automargin=True
        )
    )

    return fig

figures = {}

for (sex_id, age_group), (cost_matrix, code_to_index, all_codes) in dict_npmis.items():
    sex_label = sex_dict[sex_id]
    title = f"NPMI-based dissimilarity matrix ({sex_label}, {age_group})"

    fig = plot_cost_matrix(
        cost_matrix=cost_matrix,
        labels=all_codes,
        title=title
    )

    figures[(sex_id, age_group)] = fig
    fig.show()


for (sex_id, age_group), fig in figures.items():
    sex_label = sex_dict[sex_id].lower()
    age_label = age_group.replace(" ", "_").replace("+", "plus")
    filename = f"npmi_cost_{sex_label}_{age_label}.html"

    fig.write_html(os.path.join(output_dir, filename))