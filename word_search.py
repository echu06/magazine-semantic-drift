import pandas as pd
import spacy
import nltk
from pathlib import Path
import logging

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    filename="error_log.txt",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)



class keyword_panel_builder:

    def __init__(self, topic_groups: dict):
        """
        topic_groups: a dict where keys are topic names and values are lists of keywords.
        Example:
            {
                "Gun Violence": ["gun", "rifle", "shotgun"],
                "LGBTQ":        ["gay", "lesbian", "homosexual"],
            }
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.topic_groups = topic_groups 

        # Flat keyword -> [topic1, topic2, ...] mapping (a keyword can belong to multiple topics)
        self.keyword_to_topics = {}
        for topic, keywords in topic_groups.items():
            for kw in keywords:
                self.keyword_to_topics.setdefault(kw.lower(), []).append(topic)

        self.all_topics = list(topic_groups.keys()) 

    def _get_surrounding_sentences(self, text: str, keyword: str, window: int = 2) -> list[dict]:
        """
        text: Simply just the article/magazine/text snippet you want to analyze for key words
        keyword: the keyword to search for
        window: Along the text given, how many sentences do you want your context windows to have before and after. 
        """
        sentences = sent_tokenize(text) # pulls all possible sentences and classifies them in order. [sentence1, sentence2, ....]
        matches = [] # create a list for all of our matches. 

        for sent_idx, sentence in enumerate(sentences):     # grab numerical id and the sentences
            if keyword.lower() in sentence.lower():         # if we find keywords are there, then begin grabbing context windows. 
                start = max(0, sent_idx - window)           # grab either the first sentence if there are no sentences before the keyword. 
                end = min(len(sentences), sent_idx + window + 1)    # Grab the sentence that is closest after the keyword sentence. 

                matches.append({                            # Create a tibble of information/observation. 
                    "Keyword":           keyword,           # The keyword we found. 
                    "Sentence Position": sent_idx,          # Just the position of the keyword and where it's located
                    "Surrounding Text":  " ".join(sentences[start:end]),  # Join sentences 2 before and 2 after (if possible). 
                })

        return matches

    def _scan_row(self, text: str) -> list[dict]:
        """
        Scans a single text string for all keyword matches.
        Returns a list of dicts, one per match, with binary topic columns
        and surrounding sentence context.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        all_matches = []

        for kw, topics in self.keyword_to_topics.items():
            sentence_matches = self._get_surrounding_sentences(text, kw)        #run through the text with a keyword. 

            for match in sentence_matches:
                topic_flags = {t: int(t in topics) for t in self.all_topics}    # Soon after, append each match with the topic flag that we designated. 
                all_matches.append({
                    **match,
                    **topic_flags
                })

        return all_matches

    def build_panel(self, df: pd.DataFrame, text_col: str = "Text", meta_cols: list = None) -> pd.DataFrame: 
        """
        Iterates over each row of the input dataframe, scans the text column,
        and returns a long-format panel with one row per keyword match.
        All original columns are preserved.
        """
        all_rows = []

        if meta_cols is None:
            meta_cols = [col for col in df.columns if col != text_col]
        else:
            # Ensure only valid columns are used
            meta_cols = [col for col in meta_cols if col in df.columns and col != text_col]

        for _, record in df.iterrows():
            matches = self._scan_row(record[text_col])
            for match in matches:
                meta = {col: record[col] for col in meta_cols}
                all_rows.append({**meta, **match})

        panel_df = pd.DataFrame(all_rows)

        # Final column order: metadata -> keyword info -> context -> topic flags
        col_order = (meta_cols
                     + ["Keyword", "Sentence Position", "Surrounding Text"]
                     + self.all_topics)
        
        panel_df = panel_df[[c for c in col_order if c in panel_df.columns]]

        return panel_df

    def run(self, input_dataframe: pd.DataFrame, output_path: str, text_col: str = "Text"):
        """
        Full pipeline: Builds panel, saves to CSV.

        REQUIRED: Need to manually load in data before with proper format for pipeline to run. 
        """

        print(f"Columns found: {list(input_dataframe.columns)}")

        print(f"Scanning {len(input_dataframe)} rows for keywords...")
        panel_df = self.build_panel(input_dataframe, text_col=text_col)

        panel_df.to_csv(output_path, index=False)
        print(f"Done. {len(panel_df)} keyword matches written to {output_path}")
        return panel_df


if __name__ == "__main__":

    topic_groups = {
        "Gun Violence": [
            "gun", "rifle", "shotgun", "firearm", "pistol", "revolver",
            "handgun", "carbine", "bullet", "ak-47", "ar-15", "m16",
            "glock", "colt", "remington", "winchester", "mossberg",
            "semiautomatic", "assault rifle", "sniper rifle", "machine gun",
            "submachine gun", "grenade launcher"
        ],
        "LGBTQ Viewpoints": [
        "gay", "gays", "gay rights", "gay community",
        "lesbian", "lesbians", "lgb", "lgbt",
        "homosexual", "homosexuals", "homosexuality",
        "sodomy", "sodomite", "queer", "queers",
        "bisexual", "bisexuals",
        "aids", "gay plague", "bathhouse"
        ],
        "Right Wing Viewpoints": [
            "republican", "republicans", "conservative", "conservatives",
            "right wing", "right-wing", "far right", "radical right",
            "trump", "jd vance", "maga", "gop"
        ],
        "Immigrant and POC Viewpoints": [
            "black", "asian", "hispanic", "chinese", "mexican", "immigrant", "korean", 
            "japanese", "latino", "latina", "outsider", "foreigner", "migrant", "migrants", "refugee", "refugees"
        ], 
        "Female Outlook": [
            "female", "woman", "women", "wife", "girl", "girls",
            "mother", "mom", "daughter", "lady", "ladies", "pregnant"
            ]
    }

    pipeline = keyword_panel_builder(topic_groups=topic_groups)
    # df = pd.read_parquet("atlantic_articles.parquet", engine="pyarrow") 

    panel = pipeline.run(
        input_dataframe= pd.read_csv("ChickTracks.csv"), 
        output_path="ChickTracks_word_search.csv",
        text_col="Responses"
    )

    print(panel.head(10))


