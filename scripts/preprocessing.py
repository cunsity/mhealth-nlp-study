"""Preprocessing utilities for mHealth narrative analysis."""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """Basic text cleaning for review narratives."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Create a clean_text column from raw review text."""
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    return df
