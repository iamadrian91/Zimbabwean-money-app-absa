"""Corpus loading and experiment splits."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (APPS, HOLDOUT_FRACTION, PROCESSED_DIR, RAW_REVIEWS, SEED,
                     SHORT_REVIEW_MAX_WORDS, stars_to_label)
from .text import clean, split_clauses, split_sentences

REVIEWS_FILE = PROCESSED_DIR / "reviews.csv"
SENTENCES_FILE = PROCESSED_DIR / "sentences.csv"


def load_raw() -> pd.DataFrame:
    """Load reviews and drop empty ones."""
    df = pd.read_csv(RAW_REVIEWS, parse_dates=["at"], encoding="utf-8-sig")
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""].reset_index(drop=True)
    return df


def assign_subsets(df: pd.DataFrame) -> pd.DataFrame:
    """Add label and subset columns."""
    df = df.copy()
    df["label"] = df["score"].map(stars_to_label)
    df["n_words"] = df["content"].str.split().str.len()
    df["is_short"] = df["n_words"] <= SHORT_REVIEW_MAX_WORDS
    df["app_type"] = df["app"].map(lambda a: APPS[a]["type"])

    df["subset"] = ""
    train_mask = df["split"] == "train"
    tr_idx, ho_idx = train_test_split(
        df.index[train_mask], test_size=HOLDOUT_FRACTION, random_state=SEED,
        stratify=df.loc[train_mask, "label"])
    df.loc[tr_idx, "subset"] = "train"
    df.loc[ho_idx, "subset"] = "holdout"
    df.loc[df["app"] == "EcoCash Super App", "subset"] = "test_known"
    df.loc[(df["split"] == "test") & (df["app"] != "EcoCash Super App"), "subset"] = "test_unseen"
    assert (df["subset"] != "").all()
    return df


def build_reviews(save: bool = True) -> pd.DataFrame:
    df = assign_subsets(load_raw())
    df["clean"] = df["content"].map(clean)
    df["clean_unmasked"] = df["content"].map(lambda t: clean(t, mask_brands=False))
    if save:
        df.to_csv(REVIEWS_FILE, index=False, encoding="utf-8-sig")
    return df


def build_sentences(reviews: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """One row per clause, with sentence and clause ids."""
    rows = []
    for r in reviews.itertuples(index=False):
        for i, sent in enumerate(split_sentences(r.content)):
            for j, clause in enumerate(split_clauses(sent)):
                rows.append({
                    "reviewId": r.reviewId, "app": r.app, "brand": r.brand,
                    "subset": r.subset, "score": r.score, "label": r.label, "at": r.at,
                    "sent_id": f"{r.reviewId}_{i}", "clause_id": f"{r.reviewId}_{i}_{j}",
                    "sentence": sent, "clause": clause,
                    "sentence_clean": clean(sent), "clause_clean": clean(clause),
                })
    out = pd.DataFrame(rows)
    if save:
        out.to_csv(SENTENCES_FILE, index=False, encoding="utf-8-sig")
    return out


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(REVIEWS_FILE, parse_dates=["at"], encoding="utf-8-sig",
                       keep_default_na=False, na_values={"appVersion": [""]})


def load_sentences() -> pd.DataFrame:
    return pd.read_csv(SENTENCES_FILE, parse_dates=["at"], encoding="utf-8-sig",
                       keep_default_na=False)
