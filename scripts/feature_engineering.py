"""Feature engineering for mHealth narrative analysis."""

from typing import Iterable, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_IMPROVEMENT_KEYWORDS: List[str] = [
    "improved",
    "better",
    "relief",
    "pain free",
    "pain-free",
    "recovered",
    "recovery",
    "reduced pain",
    "symptoms reduced",
    "helped"
]


def create_improvement_label(text: str, keywords: Iterable[str] = DEFAULT_IMPROVEMENT_KEYWORDS) -> int:
    """Generate a binary keyword-derived improvement indicator."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return int(any(keyword in text_lower for keyword in keywords))


def add_improvement_label(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Add keyword-derived improvement_label column."""
    df = df.copy()
    df["improvement_label"] = df[text_col].apply(create_improvement_label)
    return df


def build_tfidf(train_texts, test_texts, max_features: int = 1000, ngram_range=(1, 1), min_df: int = 2):
    """Fit TF-IDF on training data and transform train/test text."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test
