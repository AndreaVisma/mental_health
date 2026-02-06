

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
from math import log2

metadata = pd.read_csv("C:\\Data\\my_datasets\\medical\\patients_metadata.csv")

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences_only_f.parquet")

##############
#
from collections import defaultdict

position_counts = defaultdict(Counter)

for seq in seq_df['sequence_uniques']:
    for i, code in enumerate(seq):
        position_counts[i][code] += 1

position_df = (
    pd.DataFrame(position_counts)
      .fillna(0)
      .T
)

# normalize
position_prop = position_df.div(position_df.sum(axis=1), axis=0)

# plot top diagnoses only
top_codes = position_df.sum().sort_values(ascending=False).head(10).index
position_prop[top_codes].plot(kind='area', stacked=True, figsize=(10,5))
plt.title("State distribution by sequence position")
plt.xlabel("Position in sequence")
plt.ylabel("Proportion")
plt.show(block = True)

#stratify
dict_young_mid_old = dict(zip(sorted(seq_df.ag_id.unique()),
                     ["young 0-39"] * 8 + ["midlife 39-64"] * 5 + ["old 65+"] * 6))
seq_df["age_group_1"] = seq_df.ag_id.map(dict_young_mid_old)

def make_transitions(seq):
    return list(zip(seq, seq[1:]))

seq_df['transitions'] = seq_df['sequence_uniques'].apply(make_transitions)

all_transitions = seq_df['transitions'].explode()
transition_counts = all_transitions.value_counts().head(10)

print(transition_counts)

# 1. Calculate the proportions per group
group_totals = (
    seq_df.groupby(['sex_id', 'age_group_1', 'is_foreign'])
    .size()
    .reset_index(name='total_patients_in_group')
)

# 2. Count unique patients per transition (the numerator)
# 'index' here refers to the original patient ID/row index
patient_transitions = (
    seq_df[['sex_id', 'age_group_1', 'is_foreign', 'transitions']]
    .explode('transitions')
    .dropna(subset=['transitions'])
    .reset_index()
    .drop_duplicates(subset=['index', 'transitions']) # Ensure one patient = one count per transition
)

transition_counts = (
    patient_transitions
    .groupby(['sex_id', 'age_group_1', 'is_foreign', 'transitions'])
    .size()
    .reset_index(name='patient_count')
)

# 3. Merge the counts with the totals
final_df = pd.merge(
    transition_counts,
    group_totals,
    on=['sex_id', 'age_group_1', 'is_foreign']
)

# 4. Calculate the percentage
final_df['patient_percentage (%)'] = (final_df['patient_count'] / final_df['total_patients_in_group']) * 100

# 5. Get the Top 5 for each group
final_df = (
    final_df.sort_values(['sex_id', 'age_group_1', 'is_foreign', 'patient_percentage (%)'], ascending=[True, True, True, False])
    .groupby(['sex_id', 'age_group_1', 'is_foreign'])
    .head(5)
    .round(2)
)

print(final_df)
final_df.to_excel("top_transitions_per_group.xlsx")

##########
#
from collections import Counter

transitions = Counter()

for seq in seq_df['sequence_uniques']:
    for a, b in zip(seq[:-1], seq[1:]):
        transitions[(a, b)] += 1

trans_df = (
    pd.DataFrame([
        {'from': k[0], 'to': k[1], 'count': v}
        for k, v in transitions.items()
    ])
)
trans_df['prob'] = (
    trans_df
    .groupby('from')['count']
    .transform(lambda x: x / x.sum())
)
trans_df = trans_df[trans_df['prob'] > 0.1]
trans_df = trans_df.sort_values('prob', ascending = False).head(100)

import networkx as nx

G = nx.DiGraph()

for _, row in trans_df.iterrows():
    G.add_edge(
        row['from'],
        row['to'],
        weight=row['prob']
    )

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k=0.5, seed=42)

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k=0.5, seed=42)

edges = G.edges()
weights = [G[u][v]['weight'] * 5 for u,v in edges]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=800,
    width=weights,      # <-- correct argument
    alpha=0.7
)
plt.title("Diagnostic transition network")
plt.show(block = True)

######
edge_alphas = [
    min(1, G[u][v]['weight'] * 10) for u, v in edges
]

for (u, v), w, a in zip(edges, weights, edge_alphas):
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v)],
        width=w,
        alpha=a
    )

nx.draw_networkx_nodes(G, pos, node_size=800)
nx.draw_networkx_labels(G, pos)

plt.title("Diagnostic transition network")
plt.show(block = True)