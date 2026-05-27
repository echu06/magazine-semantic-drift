import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import defaultdict
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import time
from time_parser import time_parser





def compute_drift(subset_df: pd.DataFrame, time_col: str = "time_bin", min_n: int = 5) -> pd.DataFrame:
    """
    Compute consecutive-bin cosine drift for one slice of data.
    `time_col` should contain pd.Timestamp values (from time_parser).
    Returns a DataFrame with columns: from_bin, to_bin, cosine_drift.
    """
    bins = sorted(subset_df[time_col].unique())   # Timestamps sort correctly
    if len(bins) < 2:
        return pd.DataFrame()

    centroids = {}
    for b in bins:
        rows = subset_df[subset_df[time_col] == b]
        if len(rows) < min_n:
            continue
        vecs = np.stack(rows["embedding"].values)
        centroids[b] = vecs.mean(axis=0)

    valid_bins = sorted(centroids.keys())
    if len(valid_bins) < 2:
        return pd.DataFrame()

    records = []
    for b1, b2 in zip(valid_bins, valid_bins[1:]):
        sim = cosine_similarity([centroids[b1]], [centroids[b2]])[0][0]
        records.append({
            "from_bin":     b1,           # pd.Timestamp — plot-ready
            "to_bin":       b2,
            "cosine_drift": round(1 - sim, 4),
        })
    return pd.DataFrame(records)

