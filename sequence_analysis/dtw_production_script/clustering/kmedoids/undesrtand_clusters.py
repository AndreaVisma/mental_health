

import pyreadr
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

ids = np.load("C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\patient_ids_M_3_2010.npy")

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences.parquet")
seq_df = seq_df[seq_df.patient_no.isin(ids)]
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

seq_df_m = seq_df.copy()
seq_df_m_small = (seq_df_m[seq_df_m["f_sequence_only_new"].apply(
    lambda x: any(c.startswith("F") for c in x) and len(x) >= 2)]
                  .dropna())

###############################
##### CLUSTERS

labels = np.load("C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\clustering\\kmedoids\\labels_4_clusters.npy")
seq_df_m_small['cluster'] = labels

result = pyreadr.read_r('C:\\Data\\my_datasets\\medical\\patients_data.rds') # also works for RData
metadata = result[None]

dem_df = seq_df_m_small.merge(metadata, on = "patient_no", how = "left")
dem_df.dropna(inplace = True)


def map_group_index_to_median_age(group_index_float):
    """
    Takes a float representing the age group index (1.0 to 19.0)
    and returns the median age of that group, rounded to the nearest integer.
    """
    # Round the float input to the nearest integer index

    # Median age for a 5-year group [A to A+4] is A + 2

    if 1 <= group_index_float <= 18:
        # Calculate the starting age of the group: (Index - 1) * 5
        # Example: Index 1 -> (1-1)*5 = 0 (Group 0-4, Median 2)
        # Example: Index 11 -> (11-1)*5 = 50 (Group 50-54, Median 52)
        starting_age = (group_index_float - 1) * 5
        median_age = starting_age + 2
        return median_age

    elif group_index_float >= 19:
        # Group "90 Jahre und älter".
        # A common convention for the median of this open-ended group is 92.
        return 92

    else:
        # Handle inputs outside the defined range
        return None

dem_df["age_nr"] = dem_df["age_central_date"].apply(map_group_index_to_median_age)


############ cluster description

# 1. Number of clusters
n_clusters = len(np.unique(labels))
print(f"\n1. NUMBER OF CLUSTERS: {n_clusters}")

# 2. Size (number of patients) of clusters
cluster_sizes = dem_df['cluster'].value_counts().sort_index()
print(f"\n2. CLUSTER SIZES (Number of patients):")
print("-" * 40)
for cluster_id, size in cluster_sizes.items():
    percentage = (size / len(dem_df)) * 100
    print(f"Cluster {cluster_id}: {size:,} patients ({percentage:.1f}%)")

# 3. Number of unique sequences in clusters
print(f"\n3. UNIQUE SEQUENCES PER CLUSTER:")
print("-" * 40)
for cluster_id in sorted(dem_df['cluster'].unique()):
    cluster_data = dem_df[dem_df['cluster'] == cluster_id]
    unique_sequences = cluster_data['f_sequence_only_new'].apply(tuple).nunique()
    sequences_per_patient = cluster_data['f_sequence_only_new'].apply(len).mean()
    print(f"Cluster {cluster_id}: {unique_sequences:,} unique sequences")
    print(f"          Average sequence length: {sequences_per_patient:.1f} diagnoses")

# 4. Age distribution of clusters
print(f"\n4. AGE DISTRIBUTION BY CLUSTER:")
print("-" * 40)

# Basic age statistics
age_stats = dem_df.groupby('cluster')['age_nr'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('25%', lambda x: x.quantile(0.25)),
    ('median', 'median'),
    ('75%', lambda x: x.quantile(0.75)),
    ('max', 'max')
]).round(2)

print(age_stats)

# Statistical tests for age differences
from scipy import stats
if n_clusters > 1:
    print(f"\n5. STATISTICAL TESTS FOR AGE DIFFERENCES:")
    print("-" * 40)

    # ANOVA test
    cluster_groups = [dem_df[dem_df['cluster'] == i]['age_central_date'] for i in sorted(dem_df['cluster'].unique())]
    f_stat, p_value = stats.f_oneway(*cluster_groups)
    print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("→ Statistically significant age differences between clusters (p < 0.05)")
    else:
        print("→ No statistically significant age differences between clusters")


import seaborn as sns

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Cluster Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Cluster sizes
axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', edgecolor='navy')
axes[0, 0].set_title('Cluster Sizes (Number of Patients)')
axes[0, 0].set_xlabel('Cluster ID')
axes[0, 0].set_ylabel('Number of Patients')
for i, v in enumerate(cluster_sizes.values):
    axes[0, 0].text(i, v + max(cluster_sizes.values) * 0.01, f'{v:,}', ha='center', va='bottom')

# Plot 2: Age distribution by cluster
cluster_data = [dem_df[dem_df['cluster'] == i]['age_nr'] for i in sorted(dem_df['cluster'].unique())]
box_plot = axes[0, 1].boxplot(cluster_data, labels=[f'Cluster {i}' for i in sorted(dem_df['cluster'].unique())])
axes[0, 1].set_title('Age Distribution by Cluster')
axes[0, 1].set_ylabel('Age')

# Plot 3: Age density by cluster
for cluster_id in sorted(dem_df['cluster'].unique()):
    cluster_ages = dem_df[dem_df['cluster'] == cluster_id]['age_nr']
    sns.kdeplot(cluster_ages, label=f'Cluster {cluster_id}', ax=axes[1, 0])
axes[1, 0].set_title('Age Density by Cluster')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()

# Plot 4: Unique sequences vs cluster size
unique_seqs = []
for cluster_id in sorted(dem_df['cluster'].unique()):
    cluster_data = dem_df[dem_df['cluster'] == cluster_id]
    unique_sequences = cluster_data['f_sequence_only_new'].apply(tuple).nunique()
    unique_seqs.append(unique_sequences)

axes[1, 1].scatter(cluster_sizes.values, unique_seqs, s=100, alpha=0.7)
axes[1, 1].set_title('Unique Sequences vs Cluster Size')
axes[1, 1].set_xlabel('Cluster Size (Number of Patients)')
axes[1, 1].set_ylabel('Number of Unique Sequences')
for i, (size, unique_count) in enumerate(zip(cluster_sizes.values, unique_seqs)):
    axes[1, 1].annotate(f'Cluster {i}', (size, unique_count),
                        xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.show(block = True)




# # Additional detailed report
# print(f"\n6. DETAILED CLUSTER PROFILES:")
# print("=" * 50)
#
# for cluster_id in sorted(dem_df['cluster'].unique()):
#     cluster_data = dem_df[dem_df['cluster'] == cluster_id]
#     print(f"\n--- CLUSTER {cluster_id} ---")
#     print(f"Patients: {len(cluster_data):,}")
#     print(f"Unique sequences: {cluster_data['f_sequence_only_new'].apply(tuple).nunique():,}")
#     print(f"Age: {cluster_data['age_central_date'].mean():.1f} ± {cluster_data['age_central_date'].std():.1f} years")
#     print(
#         f"Age range: {cluster_data['age_central_date'].min():.0f} - {cluster_data['age_central_date'].max():.0f} years")
#
#     # Most common sequence patterns (top 3)
#     sequence_counts = cluster_data['f_sequence_only_new'].apply(tuple).value_counts().head(3)
#     print("Most common sequence patterns:")
#     for i, (seq, count) in enumerate(sequence_counts.items(), 1):
#         percentage = (count / len(cluster_data)) * 100
#         print(f"  {i}. {seq[:5]}{'...' if len(seq) > 5 else ''} - {count} patients ({percentage:.1f}%)")
#
# # Save the analysis to CSV
# output_data = []
# for cluster_id in sorted(dem_df['cluster'].unique()):
#     cluster_data = dem_df[dem_df['cluster'] == cluster_id]
#     output_data.append({
#         'cluster': cluster_id,
#         'n_patients': len(cluster_data),
#         'percentage': (len(cluster_data) / len(dem_df)) * 100,
#         'unique_sequences': cluster_data['f_sequence_only_new'].apply(tuple).nunique(),
#         'mean_age': cluster_data['age_central_date'].mean(),
#         'std_age': cluster_data['age_central_date'].std(),
#         'min_age': cluster_data['age_central_date'].min(),
#         'max_age': cluster_data['age_central_date'].max(),
#         'median_age': cluster_data['age_central_date'].median()
#     })
#
# output_df = pd.DataFrame(output_data)
# output_df.to_csv('cluster_analysis_summary.csv', index=False)
# print(f"\nDetailed analysis saved to 'cluster_analysis_summary.csv'")
#
# print(f"\n" + "=" * 60)
# print("ANALYSIS COMPLETE")
# print("=" * 60)
# print(f"Total patients analyzed: {len(dem_df):,}")
# print(f"Number of clusters: {n_clusters}")
# print(f"Overall age range: {dem_df['age_central_date'].min():.0f} - {dem_df['age_central_date'].max():.0f} years")
# print(f"Overall mean age: {dem_df['age_central_date'].mean():.1f} ± {dem_df['age_central_date'].std():.1f} years")