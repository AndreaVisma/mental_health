

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