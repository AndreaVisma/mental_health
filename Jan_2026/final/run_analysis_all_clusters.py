
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
import pyreadr
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids   # <-- NEW import
from tqdm import tqdm
import os
import math

outfile = "c://git-projects//mental_health//cluster_info//"

dict_clusters_nr = {
('young 0-39', 1) : 10,
('midlife 39-64', 1) : 8,
('old 65+', 1) : 9,
('young 0-39', 2) : 11,
('midlife 39-64', 2) : 21
# ('old 65+', 2) : 8
}
#####

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences_only_f.parquet")

seq_df['patient_no'] = seq_df['patient_no'].astype(int)

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

#######
#Most common nationalities
top_12_nats = seq_df[seq_df.country != "not found"].country.value_counts().reset_index().country.iloc[1:13].tolist()

#######
sex_dict = {1:'male', 2:'female'}
for key, val in tqdm(dict_clusters_nr.items(), total = len(dict_clusters_nr.keys())):
    age_group = key[0]
    sex_id = key[1]

    print(f"Clustering {sex_dict[sex_id]}s, {age_group} ...")
    try:
        os.mkdir(outfile + f"{sex_dict[sex_id]}_{age_group}//")
    except:
        pass

    d = np.load(
        f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\dtw_40pct_{sex_dict[sex_id]}_{age_group}_norm.npy")
    ids = np.load(
        f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\IDS_40pct_{sex_dict[sex_id]}_{age_group}.npy")

    mini_df = seq_df[seq_df.patient_no.isin(ids)].copy()
    mini_df.patient_no = mini_df.patient_no.astype("category")
    mini_df.patient_no = mini_df.patient_no.cat.set_categories(list(ids))
    mini_df.sort_values(["patient_no"], inplace=True)

    #### pick the value I think i correct (11?)
    kmedoids = KMedoids(
        n_clusters=val,
        metric="precomputed",
        init="k-medoids++",
        random_state=42,
        max_iter=300,
        method="alternate",
    )

    labels = kmedoids.fit_predict(d)
    mini_df["cluster"] = labels

    #### demographics
    df_dem = mini_df[
        ["patient_no", "cluster", "sequence_unique_blocks", "is_foreign", "age", "ag_id", "country"]].copy()
    df_dem = df_dem.merge(elmas_metadata[["patient_no", "mortality", "num_days", "num_stays"]],
                          on="patient_no", how="left")

    n_clusters = len(np.unique(labels))
    print(f"\n1. NUMBER OF CLUSTERS: {n_clusters}")

    cluster_sizes = df_dem['cluster'].value_counts().sort_index()
    print(f"\n2. CLUSTER SIZES (Number of patients):")
    for cluster_id, size in cluster_sizes.items():
        percentage = (size / len(df_dem)) * 100
        print(f"Cluster {cluster_id}: {size:,} patients ({percentage:.1f}%)")

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(15, 12))
    # Cluster sizes
    ax.bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', edgecolor='navy')
    ax.set_title('Cluster Sizes')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Patients')
    plt.tight_layout()
    fig.savefig(outfile + f"{sex_dict[sex_id]}_{age_group}//n_people_per_cluster.png")
#     plt.show(block=True)

    ###
    clusters = sorted(df_dem['cluster'].unique())
    n_clusters = len(clusters)

    n_cols = 4
    n_rows = math.ceil(n_clusters / n_cols)

    if age_group == 'young 0-39':
        bins = np.linspace(-1, 39, 9)
    elif age_group == 'midlife 39-64':
        bins = np.linspace(39, 64, 6)
    else:
        bins = np.linspace(64, 99, 8)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True)

    fig.suptitle(
        f'Age Distributions by Cluster ({sex_dict[sex_id]}, {age_group})',
        fontsize=16, fontweight='bold'
    )

    axes = axes.flatten()

    for ax, cid in zip(axes, clusters):
        ages = df_dem.loc[df_dem['cluster'] == cid, 'age']
        ax.hist(
            ages,
            bins=bins,
            edgecolor='black',
            alpha=0.7
        )
        ax.set_title(f'Cluster {cid}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Patients')

    # Remove unused subplots
    for ax in axes[n_clusters:]:
        ax.remove()

    plt.tight_layout()
    fig.savefig(outfile + f"{sex_dict[sex_id]}_{age_group}//age_hist_clusters.png")
#     plt.show(block=True)

    ###
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Cluster Analysis ({sex_dict[sex_id]}, {age_group})', fontsize=16, fontweight='bold')

    sns.boxplot(x='cluster', y='num_days', data=df_dem, ax=axes[0, 0])
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Number of Hospital Days')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Hospital Days')

    sns.boxplot(x='cluster', y='num_stays', data=df_dem, ax=axes[0, 1])
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Number of Hospital Stays')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Hospital Stays')

    avg_mortality = df_dem.groupby('cluster')['mortality'].mean().sort_index() * 100

    axes[1, 0].bar(avg_mortality.index, avg_mortality.values, color='red')

    axes[1, 0].set_title('Mortality')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Average mortality')

    avg_migrants = df_dem.groupby('cluster')['is_foreign'].mean().sort_index() * 100

    axes[1, 1].bar(avg_migrants.index, avg_migrants.values, color='green')

    axes[1, 1].set_title('Migrants')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Percentage foreigners')

    plt.tight_layout()
    fig.savefig(outfile + f"{sex_dict[sex_id]}_{age_group}//other_descr_clusters.png")
#     plt.show(block=True)

    ############
    table = (df_dem.groupby("cluster")
             [["age", "mortality", "num_days", "num_stays", "is_foreign"]]
             .mean().reset_index())
    table["n_people"] = df_dem.groupby("cluster").size()
    total = (["total"] +
             df_dem[["age", "mortality", "num_days", "num_stays", "is_foreign"]].mean().tolist()
             + [len(df_dem)])

    table.loc[-1] = total  # adding a row
    table.index = table.index + 1  # shifting index
    table = table.sort_index()  # sorting by index
    table.to_excel(
        outfile + f"{sex_dict[sex_id]}_{age_group}//{sex_dict[sex_id]}_{age_group}_clusters_means_summary.xlsx",
        index=False)

    ###########
    ## diagnoses
    from collections import Counter


    def get_top_blocks(sequences, top_n=5):
        """
        sequences: iterable of lists
        """
        all_blocks = []
        for seq in sequences:
            all_blocks.extend(seq)
        return Counter(all_blocks).most_common(top_n)


    def get_top_transitions(sequences, top_n=5):
        """
        transitions are adjacent pairs within each sequence
        """
        transitions = []
        for seq in sequences:
            transitions.extend(zip(seq[:-1], seq[1:]))
        return Counter(transitions).most_common(top_n)


    results = {}

    for cluster_id, group in df_dem.groupby('cluster'):
        sequences = group['sequence_unique_blocks']

        top_blocks = get_top_blocks(sequences, top_n=5)
        top_transitions = get_top_transitions(sequences, top_n=5)

        results[cluster_id] = {
            'top_blocks': top_blocks,
            'top_transitions': top_transitions
        }

    rows = []

    for cluster_id, res in results.items():
        for block, count in res['top_blocks']:
            rows.append({
                'cluster': cluster_id,
                'type': 'block',
                'item': block,
                'count': count
            })
        for (src, dst), count in res['top_transitions']:
            rows.append({
                'cluster': cluster_id,
                'type': 'transition',
                'item': f'{src} -> {dst}',
                'count': count
            })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.merge(table[["cluster", "n_people"]], on="cluster", how="left")
    summary_df["freq"] = 100 * summary_df["count"] / summary_df["n_people"]

    summary_df.to_excel(
        outfile + f"{sex_dict[sex_id]}_{age_group}//{sex_dict[sex_id]}_{age_group}_top_diagnoses_summary.xlsx",
        index=False)

    ############################
    from highlight_text import fig_text

    ### plot citizenship
    fig, axes = plt.subplots(
        3, 4, figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True)

    text = 'The iris dataset contains 3 species:\n<setosa>, <versicolor>, and <virginica>'
    fig_text(
        s=f'Percentage of foreigners by Cluster ({sex_dict[sex_id]}, {age_group})\n'
          f'<(in red the average number of patients of the nationality in the age bracket)>',
        x=.5, y=0.98,
        fontsize=16,
        color='black',
        highlight_textprops=[{"color": 'red', 'fontweight': 'bold'}],
        ha='center'
    )

    axes = axes.flatten()

    for ax, nat in zip(axes, top_12_nats):
        avg_nat = 100 * len(df_dem[df_dem.country == nat]) / len(df_dem)
        ax.axhline(y=avg_nat, color='r', linestyle='-')
        # n_people_tot = df_dem.groupby("cluster").size().sort_index().reset_index()
        # n_people_tot["tot"] = len(df_dem)
        # n_people_tot["pct"] = 100 * n_people_tot[0] / n_people_tot["tot"]
        n_people = df_dem[df_dem.country == nat].groupby("cluster").size().sort_index().reset_index()
        n_people["tot"] = df_dem.groupby("cluster").size().sort_index()
        n_people["pct"] = 100 * n_people[0] / n_people["tot"]
        # ax.bar(n_people_tot.cluster, n_people_tot.pct)
        ax.bar(n_people.cluster, n_people.pct)
        ax.set_title(f'{nat}')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('pct patients (%)')

    fig.savefig(outfile + f"{sex_dict[sex_id]}_{age_group}//nationalities_cluster_percentage.png")
#     plt.show(block=True)

    ############

    ### plot citizenship
    fig, axes = plt.subplots(
        3, 4, figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True)

    fig_text(
        s=f'Percentage of foreigners by Cluster ({sex_dict[sex_id]}, {age_group})\n'
          f'<(in blue the percentage of patients in the cluster)>\n'
          f'<(in orange the percentage of the foreign population in the cluster)>',
        x=.5, y=0.98,
        fontsize=16,
        color='black',
        highlight_textprops=[{"color": 'skyblue', 'fontweight': 'bold'},
                             {"color": 'orange', 'fontweight': 'bold'}],
        ha='center'
    )

    axes = axes.flatten()

    for ax, nat in zip(axes, top_12_nats):
        avg_nat = 100 * len(df_dem[df_dem.country == nat]) / len(df_dem)
        # ax.axhline(y=avg_nat, color='r', linestyle='-')
        n_people_tot = df_dem.groupby("cluster").size().sort_index().reset_index()
        n_people_tot["tot"] = len(df_dem)
        n_people_tot["pct"] = 100 * n_people_tot[0] / n_people_tot["tot"]
        n_people_tot['nat'] = df_dem[df_dem.country == nat].groupby("cluster").size().sort_index()
        n_people_tot["tot_nat"] = len(df_dem[df_dem.country == nat])
        n_people_tot["pct_nat"] = 100 * n_people_tot["nat"] / n_people_tot["tot_nat"]
        ax.bar(n_people_tot.cluster, n_people_tot.pct)
        ax.bar(n_people_tot.cluster, n_people_tot.pct_nat, alpha=0.4)
        ax.set_title(f'{nat}')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('pct patients (%)')

    fig.savefig(outfile + f"{sex_dict[sex_id]}_{age_group}//nationalities_cluster_percentage_2.png")
#     plt.show(block=True)


    ########
    # most common trajectories
    def unique_sublists(input_list):
        # Create an empty dictionary 'result' to store the sublists and their counts
        result = {}
        # Iterate through the sublists in 'input_list'
        for l in input_list:
            # Use a tuple representation of the sublist as a key to the 'result' dictionary
            # and append 1 to the list associated with that key
            result.setdefault(tuple(l), list()).append(1)
        # Iterate through the items in 'result' and replace the list of counts with their sum
        for a, b in result.items():
            result[a] = sum(b)
        return result


    dfs = []
    n_unique_trajs = []
    n_unique_pats = []
    avg_len_traj = []
    cluster_ids = np.sort(df_dem.cluster.unique())
    for cluster_id in cluster_ids:
        df_cluster = df_dem[df_dem.cluster == cluster_id]
        unique_sub = unique_sublists(df_cluster.sequence_unique_blocks.tolist())
        n_unique_traj = len(unique_sub)
        n_patients = len(df_cluster)
        avg_len = np.mean([len(x) for x in df_cluster.sequence_unique_blocks.tolist()])
        mct = pd.DataFrame(unique_sub.items(), columns=['Trajectory', 'Nr patients'])
        mct["pct patients in cluster"] = 100 * mct["Nr patients"] / n_patients
        mct["cluster"] = cluster_id
        mct = mct.sort_values("pct patients in cluster", ascending=False).head(10)
        dfs.append(mct)
        n_unique_trajs.append(n_unique_traj)
        n_unique_pats.append(n_patients)
        avg_len_traj.append(avg_len)
    mct_all = pd.concat(dfs)
    dict_clust_traj = dict(zip(cluster_ids, n_unique_trajs))
    df_un_traj = pd.DataFrame(dict_clust_traj.items(), columns=['cluster', 'nr unique trajectories'])
    df_un_traj["nr patients"] = n_unique_pats
    df_un_traj["unique traj per patient"] = df_un_traj['nr unique trajectories'] / df_un_traj['nr patients']
    df_un_traj["avg trajectory length"] = avg_len_traj

    df_un_traj.to_excel(
        outfile + f"{sex_dict[sex_id]}_{age_group}//{sex_dict[sex_id]}_{age_group}_nr_unique_trajectories.xlsx",
        index=False)
    mct_all.to_excel(
        outfile + f"{sex_dict[sex_id]}_{age_group}//{sex_dict[sex_id]}_{age_group}_top_10_most_common_trajectories.xlsx",
        index=False)
    mct_all.to_csv(
        outfile + f"{sex_dict[sex_id]}_{age_group}//{sex_dict[sex_id]}_{age_group}_top_10_most_common_trajectories.csv",
        index=False)
    ########

####