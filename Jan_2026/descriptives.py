

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

metadata = pd.read_csv("C:\\Data\\my_datasets\\medical\\patients_metadata.csv")

# Load data
seq_df = pd.read_parquet("C:\\Data\\my_datasets\\medical\\diagnosis_sequences_only_f.parquet")

#age_groupings
dict_young_mid_old = dict(zip(sorted(seq_df.ag_id.unique()),
                     ["young 0-39"] * 8 + ["midlife 39-64"] * 5 + ["old 65+"] * 6))
seq_df["age_group_1"] = seq_df.ag_id.map(dict_young_mid_old)

dict_age_groups = dict(zip(sorted(seq_df.ag_id.unique()),
                     ["no"] * 4 + ["young 18-44"] * 5 + ["midlife 45-64"] * 4 + ["old 65+"] * 6))
seq_df["age_group_2"] = seq_df.ag_id.map(dict_age_groups)

##### descr
print(f"n patients: {len(seq_df)}")
print(f"Pct migrants: {100 * len(seq_df[seq_df.is_foreign == True]) / len(seq_df)}")
print(f"Pct males: {100 * len(seq_df[seq_df.sex_id == 1]) / len(seq_df)}")



####################3

summary_table_1 = seq_df[['patient_no', 'length', 'sex_id', 'age_group_1', 'is_foreign', 'n_f_codes']].groupby(['sex_id', 'age_group_1']).agg(
    {'patient_no' : 'count', 'length' : 'mean', 'is_foreign' : 'mean', 'n_f_codes' : 'mean'}
).reset_index()
summary_table_1.age_group_1 = pd.Categorical(summary_table_1.age_group_1,
                      categories=["young 0-39","midlife 39-64","old 65+"],
                      ordered=True)
summary_table_1.sort_values('age_group_1', inplace=True)
summary_table_1.to_excel('summary_sequences_groups_1.xlsx', index = False)

summary_table_2 = seq_df[['patient_no', 'length', 'sex_id', 'age_group_2', 'is_foreign', 'n_f_codes']].groupby(['sex_id', 'age_group_2']).agg(
    {'patient_no' : 'count', 'length' : 'mean', 'is_foreign' : 'mean', 'n_f_codes' : 'mean'}
).reset_index()
summary_table_2.age_group_2 = pd.Categorical(summary_table_2.age_group_2,
                      categories=["no", "young 18-44","midlife 45-64","old 65+"],
                      ordered=True)
summary_table_2.sort_values('age_group_2', inplace=True)
summary_table_2.to_excel('summary_sequences_groups_2.xlsx', index = False)

################

######
unique_codes = sorted(set(chain.from_iterable(seq_df['sequence'])))
codes_a_to_n = sorted([code for code in unique_codes if 'A' <= code[0] <= 'N'])

def process_sequence_A_N(seq):
    new_seq = []
    new_seq_append = new_seq.append
    for code in seq:
        if code in codes_a_to_n:
            new_seq_append(code)
    return new_seq

seq_df['sequence'] = seq_df['sequence'].apply(process_sequence_A_N)

def process_sequence_unique(seq):
    seen = set()
    new_seq = []

    seen_add = seen.add
    new_seq_append = new_seq.append
    for code in seq:
        if code in codes_a_to_n and code not in seen:
            seen_add(code)
            new_seq_append(code)
    return new_seq
seq_df['sequence_uniques'] = seq_df['sequence'].apply(process_sequence_unique)

##############

# Full sequence length
seq_df['seq_len'] = seq_df['sequence'].apply(len)

# number of unique diagnoses
seq_df['n_unique'] = seq_df['sequence_uniques'].apply(lambda x: len(set(x)))

# repetition rate
seq_df['repetition_rate'] = 1 - seq_df['n_unique'] / seq_df['seq_len']

def shannon_entropy(seq):
    counts = Counter(seq)
    probs = np.array(list(counts.values())) / len(seq)
    return -np.sum(probs * np.log2(probs))

seq_df['entropy'] = seq_df['sequence'].apply(shannon_entropy)

###############
import seaborn as sns
len_counts = seq_df.groupby("is_foreign")['n_unique'].value_counts().reset_index()

sns.barplot(len_counts, x="n_unique", y="count", hue = "is_foreign")
plt.show(block = True)

#################3
## blocks
blocks = pd.read_csv("C://users//Andrea Vismara//downloads//Blocks_All.csv")

dict_blocks = dict(zip(blocks["icd_code"], blocks["block_name"]))
dict_blocks['A90'] = "A92-A99"

def transform_sequences_block(seq):
    new_seq = list(set([dict_blocks[i] for i in seq]))
    return new_seq

seq_df['sequence_unique_blocks'] = seq_df['sequence_uniques'].apply(transform_sequences_block)

def process_sequence_unique(seq):
    seen = set()
    new_seq = []

    seen_add = seen.add
    new_seq_append = new_seq.append
    for code in seq:
        if code not in seen:
            seen_add(code)
            new_seq_append(code)
    return new_seq
seq_df['sequence_unique_blocks'] = seq_df['sequence_unique_blocks'].apply(process_sequence_unique)

seq_df['len_blocks_sequence'] = seq_df['sequence_unique_blocks'].apply(len)

import seaborn as sns
len_counts = seq_df.groupby("is_foreign")['len_blocks_sequence'].value_counts().reset_index()

sns.barplot(len_counts, x="len_blocks_sequence", y="count", hue = "is_foreign")
plt.show(block = True)

#####################
## FREQUENCY OF F CODES
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# 1. Count how many patients have each code
# We flatten the list of lists. Since the column is 'sequence_uniques',
# each code appears at most once per patient.
all_codes = [code for sublist in seq_df['sequence_uniques'] for code in sublist]
code_counts = Counter(all_codes)

# 2. Create a frequency DataFrame
freq_df = pd.DataFrame.from_dict(code_counts, orient='index', columns=['patient_count']).reset_index()
freq_df.rename(columns={'index': 'diagnostic_code'}, inplace=True)

# 3. Calculate the percentage (frequency)
total_patients = len(seq_df)
freq_df['percentage'] = (freq_df['patient_count'] / total_patients) * 100

# 4. Sort and select the top N (e.g., top 20) for the chart
freq_df_plot = freq_df.sort_values(by='percentage', ascending=False).head(20)

# 5. Plot using Plotly
fig = px.bar(
    freq_df_plot,
    x='diagnostic_code',
    y='percentage',
    title='Top 20 Most Frequent Diagnostic Codes (all codes)',
    labels={'percentage': 'Percentage of Patients (%)', 'diagnostic_code': 'Diagnostic Code'},
    text_auto='.1f' # Adds percentage labels on top of bars
)

fig.update_layout(xaxis_tickangle=-45, xaxis={'categoryorder':'total descending'})
fig.show()

freq_df_f = freq_df[freq_df.diagnostic_code.str.startswith("F")].sort_values(by='percentage', ascending=False).head(20)

# 5. Plot using Plotly
fig = px.bar(
    freq_df_f,
    x='diagnostic_code',
    y='percentage',
    title='Top 20 Most Frequent Diagnostic Codes (only F codes)',
    labels={'percentage': 'Percentage of Patients (%)', 'diagnostic_code': 'Diagnostic Code'},
    text_auto='.1f' # Adds percentage labels on top of bars
)

fig.update_layout(xaxis_tickangle=-45, xaxis={'categoryorder':'total descending'})
fig.show()

#######
#
seq_df["sex_id"] = seq_df["sex_id"].map({1:'male', 2:'female'})
exploded_df = seq_df.explode('sequence_uniques').rename(columns={'sequence_uniques': 'diagnostic_code'})

def plot_stratified_codes(df, original_df, stratify_col, only_f = False, top_n=10):
    """
    df: the exploded dataframe
    original_df: the original seq_df (to get total patient counts per group)
    stratify_col: the column to stratify by (e.g., 'sex')
    """
    # 1. Calculate total patients in each stratum (Denominator)
    group_totals = original_df.groupby(stratify_col).size().reset_index(name='total_in_group')

    # 2. Count occurrences of each code per stratum (Numerator)
    # Since codes were unique per patient in the list, count() = patient count
    counts = df.groupby([stratify_col, 'diagnostic_code']).size().reset_index(name='patient_count')

    # 3. Merge and calculate percentage
    merged = pd.merge(counts, group_totals, on=stratify_col)
    merged['percentage'] = (merged['patient_count'] / merged['total_in_group']) * 100

    # 4. Filter for Top N codes (based on overall frequency) to keep chart readable
    if only_f:
        top_codes = df[df.diagnostic_code.str.startswith("F")]['diagnostic_code'].value_counts().nlargest(top_n).index
    else:
        top_codes = df['diagnostic_code'].value_counts().nlargest(top_n).index
    plot_df = merged[merged['diagnostic_code'].isin(top_codes)]

    # 5. Plot
    fig = px.bar(
        plot_df,
        x='diagnostic_code',
        y='percentage',
        color=stratify_col,
        barmode='group',
        title=f'Top {top_n} Diagnostic Codes by {stratify_col.capitalize()}',
        labels={'percentage': 'Frequency (% of Patients in group)', 'diagnostic_code': 'Code'},
        text_auto='.1f'
    )

    fig.update_layout(xaxis={'categoryorder': 'total descending'}, legend_title=stratify_col)
    fig.show()

# Execute the plots
# Replace 'sex', 'age_group', and 'is_foreigner' with your actual column names
for col in ['sex_id', 'age_group_1', 'is_foreign']:
    if col in seq_df.columns:
        plot_stratified_codes(exploded_df, seq_df, col, only_f = True)
