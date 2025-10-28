


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

#############
## GENDER choice
gender = "F"
if gender == "F":
    sex_id = 2
    col = "red"
else:
    sex_id = 1
    col = "blue"

sample = True
sample_size = 20_000
n_clusters = 9
#####

# -------------------------
# PATHS
# -------------------------
plots_path = "C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\figures\\"

# Metadata & input
metadata_path = "C:\\Data\\my_datasets\\medical\\patients_metadata.csv"
ids_path = f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\patient_ids_{gender}_sampled_{sample}.npy"
seq_path = "C:\\Data\\my_datasets\\medical\\diagnosis_sequences.parquet"

# Cluster results (from subsample)
labels_path = f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\labels_{n_clusters}_clusters_subsample_{gender}_sample_{sample_size}.npy"
indices_path = f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\indices_{n_clusters}_subsample_{gender}_sample_{sample_size}.npy"

# -------------------------
# LOAD DATA
# -------------------------
metadata = pd.read_csv(metadata_path)
males = metadata.query(f"sex_id == {sex_id}")["patient_no"].tolist()
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

seq_df['sequence'] = seq_df['sequence'].apply(four_to_three_digits)

# Optimized unique codes extraction
unique_codes = sorted(set(chain.from_iterable(seq_df['sequence'])))
code_counts = Counter(chain.from_iterable(seq_df['sequence']))
top_100_codes = [c for c, _ in code_counts.most_common(100)]

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

seq_df['f_sequence_only_new'] = (
    seq_df['sequence'].apply(unique_in_order_top100_and_F)
)

# Filter valid sequences
seq_df['length'] = seq_df['f_sequence_only_new'].str.len()
seq_df = seq_df[seq_df['length'] >= 3]

tot_pats = len(seq_df)
unique_seq = len({tuple(seq) for seq in seq_df['f_sequence_only_new']})
print(f"Total patients in sample: {tot_pats}")
print(f"Unique sequences: {round(100 * unique_seq / tot_pats, 2)}%")

seq_df_m_small = seq_df.copy()


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
fig.savefig(plots_path + f"\\{gender}_age_description_{n_clusters}_clusters.png")
plt.show(block=True)

##################

from wordcloud import WordCloud

####################
## Word clouds
# Ensure sequences are iterable
def flatten(seq_list):
    """Flatten a list of lists."""
    return [item for sublist in seq_list for item in sublist]

print("\n6. WORD CLOUDS FOR MOST FREQUENT DIAGNOSES PER CLUSTER:")

fig, axes = plt.subplots(
    nrows=int(np.ceil(n_clusters / 3)),
    ncols=min(3, n_clusters),
    figsize=(18, 6 * np.ceil(n_clusters / 3))
)
axes = axes.flatten() if n_clusters > 1 else [axes]

for i, cid in enumerate(sorted(dem_df['cluster'].unique())):
    cluster_data = dem_df[dem_df['cluster'] == cid]

    # Flatten diagnosis sequences
    diagnoses = flatten(cluster_data['f_sequence_only_new'].tolist())

    # Count frequencies
    freq_counter = Counter(diagnoses)
    most_common = freq_counter.most_common(15)
    print(f"Cluster {cid}: Top diagnoses → {[d for d, _ in most_common]}")

    # Generate word cloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=100,
        contour_color='steelblue',
        contour_width=1
    ).generate_from_frequencies(freq_counter)

    # Plot
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(f"Cluster {cid}", fontsize=14, fontweight='bold')

# Hide unused axes if clusters < grid size
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Most Frequent Diagnoses per Cluster', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
fig.savefig(plots_path + f"\\{gender}_diagnosis_wordclouds_{n_clusters}_clusters.png")
plt.show(block=True)
####################

# most common diagnoses
from itertools import combinations
from collections import Counter

def extract_ordered_combinations(seq, n):
    """Extract all ordered n-combinations (non-consecutive) from a sequence."""
    return list(combinations(seq, n))

print("\n7. MOST FREQUENT DIAGNOSES PER CLUSTER:")
print("-" * 40)

top_n = 5
for cluster_id, cluster_data in dem_df.groupby('cluster'):
    all_codes = list(chain.from_iterable(cluster_data['f_sequence_only_new']))
    counter = Counter(all_codes)
    top_codes = counter.most_common(top_n)
    print(f"\nCluster {cluster_id} — Top {top_n} F-codes:")
    for code, freq in top_codes:
        print(f"  {code}: {freq} occurrences")

print("\n8. MOST FREQUENT DIAGNOSIS SEQUENCES PER CLUSTER:")
print("-" * 40)

top_n = 5
for cluster_id, cluster_data in dem_df.groupby('cluster'):
    pairs = Counter()
    triplets = Counter()
    for seq in cluster_data['f_sequence_only_new']:
        pairs.update(extract_ordered_combinations(seq, 2))
        triplets.update(extract_ordered_combinations(seq, 3))

    print(f"\nCluster {cluster_id} — Top {top_n} 2-code Sequences:")
    for seq, freq in pairs.most_common(top_n):
        print(f"  {seq}: {freq}")

    print(f"\nCluster {cluster_id} — Top {top_n} 3-code Sequences:")
    for seq, freq in triplets.most_common(top_n):
        print(f"  {seq}: {freq}")

##################
# RADAR PLOTS

from math import pi
top_100_chapters = list(set([x[:2] for x in top_100_codes]))

def get_f_chapter_probs(df):
    chapters = [f"F{i}" for i in range(0, 10)]
    chapter_probs = {}
    for ch in chapters:
        # Patient-level: has any diagnosis from this chapter
        has_ch = df['f_sequence_only_new'].apply(lambda seq: any(s.startswith(ch) for s in seq))
        chapter_probs[ch] = has_ch.mean()
    return chapter_probs

def get_all_chapter_probs(df):
    chapters = top_100_chapters
    chapter_probs = {}
    for ch in chapters:
        # Patient-level: has any diagnosis from this chapter
        has_ch = df['f_sequence_only_new'].apply(lambda seq: any(s.startswith(ch) for s in seq))
        chapter_probs[ch] = has_ch.mean()
    return chapter_probs

cluster_profiles = {}
for cluster_id, cluster_data in dem_df.groupby('cluster'):
    cluster_profiles[cluster_id] = get_all_chapter_probs(cluster_data)

# Convert to DataFrame
radar_df = pd.DataFrame(cluster_profiles).T
radar_df.index.name = "Cluster"

#############
# RADAR comparison with population

# Whole population profile
population_profile = get_all_chapter_probs(dem_df)

# Combine into a single DataFrame
radar_df = pd.DataFrame(cluster_profiles).T
radar_df.loc["Population"] = population_profile
radar_df.index.name = "Group"

radar_df = radar_df[list(sorted(radar_df.columns))]

print(radar_df.round(3))

def plot_radar_with_population(df, cluster_id, normalize=False, save=True):
    """
    Draws a radar chart comparing one cluster vs the overall population.
    - df: DataFrame with clusters + 'Population' as rows, F-chapters as columns
    - cluster_id: which cluster to plot (must exist in df.index)
    - normalize: if True, scale so outer radius = 1
    """
    categories = list(df.columns)
    cluster_values = df.loc[cluster_id].values.astype(float)
    population_values = df.loc["Population"].values.astype(float)

    # Normalization: across *all groups*
    if normalize:
        max_val = df.values.max()
        cluster_values = cluster_values / max_val
        population_values = population_values / max_val

    # close the loop
    cluster_values = cluster_values.tolist() + [cluster_values[0]]
    population_values = population_values.tolist() + [population_values[0]]
    N = len(categories)

    # angle setup
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # plotting
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ["0.2", "0.4", "0.6", "0.8", "1.0"],
               color="grey", size=8)
    plt.ylim(0, 1)

    # cluster profile
    ax.plot(angles, cluster_values, linewidth=2, linestyle='solid', label=f"Cluster {cluster_id}", color = col)
    ax.fill(angles, cluster_values, alpha=0.25, color = col)

    # population baseline
    ax.plot(angles, population_values, linewidth=2, linestyle='dashed', color='black', label='Population')

    plt.title(f"Cluster {cluster_id} vs Population", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    if save:
        plt.savefig(plots_path + f"radar_plots//COMP_{gender}_{cluster_id}_all", bbox_inches="tight", dpi=300)
        plt.show(block = True)
    else:
        plt.show(block = True)

for cluster_id in radar_df.index:
    if cluster_id == "Population":
        continue
    plot_radar_with_population(radar_df, cluster_id, save=True)

#########################
# Look at migrants
df_nats = pd.read_csv("C:\\Data\\my_datasets\\medical\\table_nationality.csv", sep = ";", encoding='latin-1')
dict_nats = dict(zip(df_nats.nationality_id, df_nats.code))

print("\n6. MIGRANT REPRESENTATION BY CLUSTER:")
print("-" * 40)

dem_df['is_migrant'] = dem_df['nationality_id'].apply(lambda x: 0 if x == 1 else 1)
dem_df['nationality_code'] = dem_df['nationality_id'].map(dict_nats)

migrant_stats = (
    dem_df.groupby('cluster')['is_migrant']
    .agg(['count', 'sum'])
    .rename(columns={'count': 'n_patients', 'sum': 'n_migrants'})
)
migrant_stats['pct_migrants'] = (migrant_stats['n_migrants'] / migrant_stats['n_patients']) * 100
print(migrant_stats.round(2))

plt.figure(figsize=(8,5))
plt.bar(migrant_stats.index, migrant_stats['pct_migrants'], color='tomato')
plt.title("Migrant Representation by Cluster")
plt.xlabel("Cluster")
plt.ylabel("% Migrants")
plt.show(block = True)

print("\n7. TOP MIGRANT NATIONALITIES PER CLUSTER:")
print("-" * 50)

# filter out Austrians (nationality_id == 1)
migrants_df = dem_df[dem_df['nationality_id'] != 1]

top_n = 10  # how many to print per cluster

for cluster_id, cluster_data in migrants_df.groupby('cluster'):
    nat_counts = cluster_data['nationality_code'].value_counts()
    top_nats = nat_counts.head(top_n)
    total_migrants = nat_counts.sum()

    print(f"\nCluster {cluster_id} — Top {top_n} Migrant Nationalities:")
    for nat, count in top_nats.items():
        pct = (count / total_migrants) * 100
        print(f"  {nat:<15} {count:>5} migrants ({pct:>5.1f}%)")

#####
# Population-level nationality distribution (excluding Austrians)
pop_nat_counts = migrants_df['nationality_code'].value_counts()
pop_nat_ratios = pop_nat_counts / pop_nat_counts.sum()

comparison_list = []

for cluster_id, cluster_data in migrants_df.groupby('cluster'):
    nat_counts = cluster_data['nationality_code'].value_counts()
    nat_ratios = nat_counts / nat_counts.sum()

    for nat, cluster_ratio in nat_ratios.items():
        if nat in pop_nat_ratios:
            comparison_list.append({
                'cluster': cluster_id,
                'nationality': nat,
                'cluster_ratio': cluster_ratio,
                'population_ratio': pop_nat_ratios[nat],
                'overrepresentation': cluster_ratio / pop_nat_ratios[nat]
            })

comparison_df = pd.DataFrame(comparison_list)

min_size = 15
valid_nats = pop_nat_counts[pop_nat_counts >= min_size].index

comparison_df_filtered = comparison_df[comparison_df['nationality'].isin(valid_nats)]

import seaborn as sns
plt.figure(figsize=(10,6))
sns.barplot(
    data=comparison_df_filtered,
    x='overrepresentation', y='nationality', hue='cluster',
    dodge=True
)
plt.axvline(1, color='black', linestyle='--', lw=1)
plt.title(f"Nationality Overrepresentation (≥ {min_size} total migrants)")
plt.xlabel("Overrepresentation (Cluster Ratio / Population Ratio)")
plt.ylabel("Nationality")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show(block = True)