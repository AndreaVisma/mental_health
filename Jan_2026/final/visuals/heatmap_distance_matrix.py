
import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids
import math
outfile = "c://git-projects//mental_health//cluster_info//norm_path_len//"

######################################
sex_id = 1
age_group = 'young 0-39'
sex_dict = {1:'male', 2:'female'}
try:
    os.mkdir(outfile + f"{sex_dict[sex_id]}_{age_group}//")
except:
    pass

##################################
d = np.load(f"C:\\git-projects\\mental_health\\sequence_analysis\\full_medical_spectrum\\store\\dtw_full_{sex_dict[sex_id]}_{age_group}_norm_PATH_LEN.npy")

d_mini = d[:20, :20].copy()

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

def plot_dtw_matrix(dtw_mat, title):
    # np.fill_diagonal(dtw_mat, np.nan)
    fig = go.Figure(
        data=go.Heatmap(
            z=dtw_mat,
            colorscale="RdBu_r",
            colorbar=dict(title="DTW distance"),
        )
    )

    fig.update_layout(
        title=title,
        template="simple_white",
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

fig = plot_dtw_matrix(
    dtw_mat=d_mini,
    title="DTW distances matrix"
)
fig.show()
