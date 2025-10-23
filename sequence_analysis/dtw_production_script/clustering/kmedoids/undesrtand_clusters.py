

import pyreadr
import json
import pandas as pd
from itertools import chain
from collections import Counter
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# PATHS
# -------------------------
plot_folder = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\figures\\"

# Metadata & input
metadata_path = "C:\\Data\\my_datasets\\medical\\patients_metadata.csv"
ids_path = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\matrices_store\\patient_ids_M_3_2010.npy"
seq_path = "C:\\Data\\my_datasets\\medical\\diagnosis_sequences.parquet"

# Cluster results (from subsample)
labels_path = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\clustering\\kmedoids\\labels_4_clusters_subsample.npy"
indices_path = "C:\\git-projects\\mental_health\\sequence_analysis\\dtw_production_script\\clustering\\kmedoids\\indices_subsample.npy"

# -------------------------
# LOAD DATA
# -------------------------
metadata = pd.read_csv(metadata_path)
males = metadata.query("sex_id == 1")["patient_no"].tolist()
ids = np.load(ids_path)

# diagnosis sequences
seq_df = pd.read_parquet(seq_path)
seq_df = seq_df[seq_df.patient_no.isin(ids)]
seq_df.rename(columns={"code": "sequence"}, inplace=True)
seq_df['patient_no'] = seq_df['patient_no'].astype(int)

# -------------------------
# PREPROCESSING
# -------------------------
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
    """Truncate codes to 3 digits"""
    return [x[:3] for x in seq]

seq_df['f_sequence_only_new'] = (
    seq_df['sequence'].apply(four_to_three_digits)
                      .apply(unique_in_order_ONLYF)
)

# Filter valid sequences
seq_df['length'] = seq_df['f_sequence_only_new'].str.len()
seq_df = seq_df[seq_df['length'] >= 2]

tot_pats = len(seq_df)
unique_seq = len({tuple(seq) for seq in seq_df['f_sequence_only_new']})
print(f"Total patients in sample: {tot_pats}")
print(f"Unique sequences: {round(100 * unique_seq / tot_pats, 2)}%")

seq_df_m_small = seq_df[
    seq_df["f_sequence_only_new"].apply(lambda x: any(c.startswith("F") for c in x) and len(x) >= 2)
].dropna()

# -------------------------
# LOAD SUBSAMPLE LABELS + INDICES
# -------------------------
labels = np.load(labels_path)
subsample_indices = np.load(indices_path)

# Note: indices refer to rows in your DTW matrix, which align with the original patient ID array (`ids`)
sampled_patient_ids = np.array(ids)[subsample_indices]

# -------------------------
# MERGE LABELS INTO SEQUENCE DATAFRAME
# -------------------------
seq_df_m_small_sub = seq_df_m_small[seq_df_m_small["patient_no"].isin(sampled_patient_ids)].copy()
print(f"Subsample size after merge: {len(seq_df_m_small_sub)} patients")

# Attach labels based on patient order
# We assume sampled_patient_ids order matches labels order (as saved during clustering)
label_df = pd.DataFrame({
    "patient_no": sampled_patient_ids.astype(int),
    "cluster": labels
})
seq_df_m_small_sub = seq_df_m_small_sub.merge(label_df, on="patient_no", how="inner")

# -------------------------
# MERGE WITH METADATA
# -------------------------
result = pyreadr.read_r('C:\\Data\\my_datasets\\medical\\patients_data.rds')  # loads RDS
metadata_r = result[None]
dem_df = seq_df_m_small_sub.merge(metadata_r, on="patient_no", how="left")
print(dem_df.isna().sum())
dem_df.dropna(inplace=True)

# -------------------------
# AGE MAPPING
# -------------------------
def map_group_index_to_median_age(group_index_float):
    """Convert 5-year age group index to median age."""
    if 0 <= group_index_float <= 19:
        starting_age = (group_index_float - 1) * 5
        return starting_age + 2
    elif group_index_float >= 19:
        return 92
    else:
        return None

dem_df["age_nr"] = dem_df["age_central_date"].apply(map_group_index_to_median_age)

# -------------------------
# CLUSTER DESCRIPTIONS
# -------------------------
n_clusters = len(np.unique(labels))
print(f"\n1. NUMBER OF CLUSTERS: {n_clusters}")

cluster_sizes = dem_df['cluster'].value_counts().sort_index()
print(f"\n2. CLUSTER SIZES (Number of patients):")
for cluster_id, size in cluster_sizes.items():
    percentage = (size / len(dem_df)) * 100
    print(f"Cluster {cluster_id}: {size:,} patients ({percentage:.1f}%)")

print(f"\n3. UNIQUE SEQUENCES PER CLUSTER:")
for cluster_id in sorted(dem_df['cluster'].unique()):
    cluster_data = dem_df[dem_df['cluster'] == cluster_id]
    unique_sequences = cluster_data['f_sequence_only_new'].apply(tuple).nunique()
    avg_len = cluster_data['f_sequence_only_new'].apply(len).mean()
    print(f"Cluster {cluster_id}: {unique_sequences:,} unique sequences (avg len: {avg_len:.1f})")

print(f"\n4. AGE DISTRIBUTION BY CLUSTER:")
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

if n_clusters > 1:
    print(f"\n5. STATISTICAL TESTS FOR AGE DIFFERENCES:")
    cluster_groups = [dem_df[dem_df['cluster'] == i]['age_central_date'] for i in sorted(dem_df['cluster'].unique())]
    f_stat, p_value = stats.f_oneway(*cluster_groups)
    print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("→ Statistically significant age differences between clusters (p < 0.05)")
    else:
        print("→ No statistically significant age differences between clusters")

# -------------------------
# VISUALIZATION
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Cluster Analysis (Subsample)', fontsize=16, fontweight='bold')

# Cluster sizes
axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', edgecolor='navy')
axes[0, 0].set_title('Cluster Sizes')
axes[0, 0].set_xlabel('Cluster ID')
axes[0, 0].set_ylabel('Patients')

# Age boxplot
cluster_data = [dem_df[dem_df['cluster'] == i]['age_nr'] for i in sorted(dem_df['cluster'].unique())]
axes[0, 1].boxplot(cluster_data)
axes[0, 1].set_title('Age Distribution by Cluster')
axes[0, 1].set_ylabel('Age')

# Age density
for cid in sorted(dem_df['cluster'].unique()):
    sns.kdeplot(dem_df[dem_df['cluster'] == cid]['age_nr'], label=f'Cluster {cid}', ax=axes[1, 0])
axes[1, 0].set_title('Age Density by Cluster')
axes[1, 0].legend()

# Unique sequences vs cluster size
unique_seqs = []
for cid in sorted(dem_df['cluster'].unique()):
    cluster_data = dem_df[dem_df['cluster'] == cid]
    unique_sequences = cluster_data['f_sequence_only_new'].apply(tuple).nunique()
    unique_seqs.append(unique_sequences)

axes[1, 1].scatter(cluster_sizes.values, unique_seqs, s=100, alpha=0.7)
axes[1, 1].set_title('Unique Sequences vs Cluster Size')
axes[1, 1].set_xlabel('Cluster Size')
axes[1, 1].set_ylabel('Unique Sequences')
for i, (size, uniq) in enumerate(zip(cluster_sizes.values, unique_seqs)):
    axes[1, 1].annotate(f'Cluster {i}', (size, uniq), xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.show(block=True)

##################
#
