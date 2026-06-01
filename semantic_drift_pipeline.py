import pandas as pd
import numpy as np
import multiprocessing as mp
import tempfile, shutil, gc, os
from sentence_transformers import SentenceTransformer
from typing import Literal, Optional
import torch
from sklearn.metrics.pairwise import cosine_similarity
from embeddings_package import BGEEmbedder
from time_parser import time_parser
from drift_model import compute_drift
import os






'''
Pipeline Arguments: 

–– 1. Loading in Data  ──────────────────
df: Dataframe you want to calculate drift for (word search data)


–– 2. Time Bins ──────────────────
num_of_months: monthly bins
num_of_years: by default it's 1 year, but can calculate with multiple years
date_col: Name of your date column (ensure it's in the format YYYY-MM (year month) or YYYY for the function to run)


–– 3. Embeddings Model Presets ──────────────────
embed_path: Where you want your embeddings saved to
model: The model size. Either "small" or "large"
chunks: The amount of chunks you want to have your embeddings use
batches: Total number of batches the embeddings will take to make.
text_col_name: Name of the column where the text occurs. 

–– 4. Semantic Drift ──────────────────
drift_path: The path you want to save the semantic drift data to.

'''



def pipeline(df: pd.DataFrame, num_of_months: int, num_of_years: 1, 
             date_col: str, text_col_name: str,
             embed_path: str, drift_path: str, 
             model: Literal["large", "small"] = "small", chunks = 300, batches = 64):


    # ── 2. EMBED: Create new column for embeddings──────────────────────────────────────────────────────


    if os.path.exists(embed_path):
        embeddings = np.load(embed_path, allow_pickle=True)
        df["embedding"] = list(embeddings) 
    
    else:
        embedder = BGEEmbedder(model_size= model, n_cores=2)
        df = embedder.embed(
            df=df,
            text_col= text_col_name,
            chunk_size= chunks,
            batch_size= batches,
            save_path=embed_path
        )


    # ── 3. TIME PARSE ─────────────────────────────────────────────────
    df = time_parser(df, date_col_name= date_col, month_bin = num_of_months, year_bin = num_of_years)

    # ── 4. MELT LABEL COLUMNS ─────────────────────────────────────────
    group_cols = [
        "Gun Violence",
        "LGBTQ Viewpoints",
        "Right Wing Viewpoints",
        "Immigrant and POC Viewpoints",
    ]

    df_melted = df.melt(
        id_vars=["time_bin", "Keyword", "Sentence Position", text_col_name, "embedding"],
        value_vars=group_cols,
        var_name="keyword_group",
        value_name="is_member"
    )


    df_melted = df_melted[df_melted["is_member"] == 1].drop(columns="is_member").reset_index(drop=True)

    print(f"Total labeled context windows: {len(df_melted)}")
    print(df_melted["keyword_group"].value_counts())

    # ── 5. COMPUTE DRIFT PER KEYWORD GROUP ───────────────────────────
    all_drift = []

    for kg, group in df_melted.groupby("keyword_group"):
        drift = compute_drift(group, time_col="time_bin", min_n=5)
        if not drift.empty:
            drift["keyword_group"] = kg
            all_drift.append(drift)

    drift_df = pd.concat(all_drift).reset_index(drop=True)

    # ── 6. SAVE ───────────────────────────────────────────────────────
    # Save drift results
    drift_df.to_csv(drift_path, index=False)
    print(f"\n✓ Saved: {drift_path}")
    print("\nTop drift moments:")
    print(drift_df.sort_values("cosine_drift", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    # ── 1. Christianity Today ───────────────────────────────────────────────────────
    Christianity_today = pd.read_csv('Word Search Data/FinalToday_word_search.csv')
    Christianity_today['date'] = pd.to_numeric(Christianity_today['Year'])


    pipeline(df = Christianity_today, text_col_name = 'Surrounding Text', date_col='date',
             embed_path="Embeddings/ChristianityToday_embeddings.npy",model = "small",
             chunks = 300, batches = 64, num_of_months=3, num_of_years=1, 
             drift_path = 'Semantic Drift Data/ChristianityToday_Drift.csv' )

    # ── 2. WORLD Magazine ───────────────────────────────────────────────────────
    world = pd.read_csv('Word Search Data/World_word_search.csv')
    world['Date'] = pd.to_datetime(world['Date'])
    world['date'] = world['Date'].dt.strftime('%Y %m')

    pipeline(df = world, text_col_name = 'Surrounding Text', date_col='date',
             embed_path="Embeddings/World_embeddings.npy",model = "small",
             chunks = 300, batches = 64, num_of_months=3, num_of_years=1, 
             drift_path = 'Semantic Drift Data/World_Drift.csv' )
    
    # ── 3. Chick Tracts ──────────────────────────────────────────────────────

    chick_tracts = pd.read_csv('Word Search Data/ChickTracks_word_search.csv')
    chick_tracts = chick_tracts[chick_tracts["Year"].notna()].copy()
    chick_tracts["date"] = chick_tracts["Year"].astype(float).astype(int).astype(str)


    pipeline(df = chick_tracts, text_col_name = 'Surrounding Text', date_col='date',
            embed_path="Embeddings/ChickTracts_embeddings.npy",model = "small",
            chunks = 300, batches = 64, num_of_months=3, num_of_years=1, 
            drift_path = 'Semantic Drift Data/ChickTracts_Drift.csv' )
    