import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time_parser import time_parser



colors = {
    "Gun Violence": "#e74c3c",
    "LGBTQ Viewpoints": "#9b59b6",
    "Right Wing Viewpoints": "#2980b9",
    "Immigrant and POC Viewpoints": "#27ae60",
    "Female Outlook": "#FFF947"
}


def score_roberta(texts, tokenizer, model, device, batch_size=32):
    all_scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiment"):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        scores = softmax(output.logits.cpu().numpy(), axis=1)

        for s in scores:
            all_scores.append({
                "roberta_negative": round(float(s[0]), 4),
                "roberta_neutral": round(float(s[1]), 4),
                "roberta_positive": round(float(s[2]), 4),
                "roberta_compound": round(float(s[2]) - float(s[0]), 4)
            })

    return all_scores

def run_pipeline(df: pd.DataFrame, timebins_month: int, timebins_year: int, magazine_name: str, time_col: str):

    # ── 2. TIME PARSE ─────────────────────────────────────────────────
    df = time_parser(df, date_col_name= time_col, month_bin=timebins_month, year_bin= timebins_year)

    # ── 3. MELT LABEL COLUMNS ─────────────────────────────────────────
    group_cols = [
        "Gun Violence",
        "LGBTQ Viewpoints",
        "Right Wing Viewpoints",
        "Immigrant and POC Viewpoints",
    ]

    df_melted = df.melt(
        id_vars=["time_bin", "Keyword", "Sentence Position", "Surrounding Text"],
        value_vars=group_cols,
        var_name="keyword_group",
        value_name="is_member"
    )

    df_melted = (
        df_melted[df_melted["is_member"] == 1]
        .drop(columns="is_member")
        .reset_index(drop=True)
    )

    print(f"Total labeled context windows: {len(df_melted)}")
    print(df_melted["keyword_group"].value_counts())

    # ── 4. LOAD RoBERTa MODEL ────────────────────────────────────────
    print("\nLoading RoBERTa sentiment model...")
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    # ── 5. SCORE SENTIMENT ───────────────────────────────────────────
    if "roberta_compound" in df_melted.columns:
        print("✓ RoBERTa scores found, skipping scoring step...")
    else:
        print("\nScoring all labeled context windows with RoBERTa...")
        texts = df_melted["Surrounding Text"].fillna("").tolist()
        scores = score_roberta(texts, tokenizer, model, device, batch_size=32)
        score_df = pd.DataFrame(scores)
        df_melted = pd.concat([df_melted, score_df], axis=1)

    # ── 6. AGGREGATE SENTIMENT OVER TIME ─────────────────────────────
    sentiment_roberta = (
        df_melted.groupby(["keyword_group", "time_bin"])
        .agg(
            avg_compound=("roberta_compound", "mean"),
            std_compound=("roberta_compound", "std"),
            avg_positive=("roberta_positive", "mean"),
            avg_negative=("roberta_negative", "mean"),
            avg_neutral=("roberta_neutral", "mean"),
            n=("roberta_compound", "count")
        )
        .reset_index()
        .sort_values(["keyword_group", "time_bin"])
    )

    # ── 7. SAVE ──────────────────────────────────────────────────────
    sentiment_roberta.to_csv(f"Roberta Data/{magazine_name}_roberta_sentiment_over_time.csv", index=False)
    print(f"✓ Saved: {magazine_name}_roberta_sentiment_over_time.csv")
    print("\nTop positive periods:")
    print(
        sentiment_roberta.sort_values("avg_compound", ascending=False)
        .head(10)
        .to_string(index=False)
    )


# ── 0. MAIN PIPELINE ───────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. LOAD AND MANAGE DATA ───────────────────────────────────────────────────────
    charisma_data = pd.read_csv('Word Search Data/CharismaFinal_word_search.csv')
    charisma_data["Date"] = pd.to_datetime(charisma_data["Date"])
    charisma_data["date"] = charisma_data["Date"].dt.strftime("%Y %m")
    run_pipeline(df = charisma_data, timebins_month=3, time_col='date', magazine_name = "Charisma", timebins_year=1)


    christianity_today = pd.read_csv('Word Search Data/FinalToday_word_search.csv')
    christianity_today['date'] = pd.to_numeric(christianity_today['Year'])
    run_pipeline(df = christianity_today, timebins_month=3, time_col='date', magazine_name = "Christianity Today", timebins_year=1)

    chick_tracts = pd.read_csv('Word Search Data/ChickTracks_word_search.csv')
    chick_tracts = chick_tracts[chick_tracts["Year"].notna()].copy()
    chick_tracts["date"] = chick_tracts["Year"].astype(float).astype(int).astype(str)
    run_pipeline(df = chick_tracts, timebins_month=3, time_col='date', magazine_name = "Chick Tracts", timebins_year=1)
