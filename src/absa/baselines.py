"""Baselines: majority class, VADER and saved transformer predictions."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import TABLES_DIR

TRANSFORMER_PREDICTIONS = TABLES_DIR / "transformer_predictions.csv"
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def vader_label(text: str, threshold: float = 0.05) -> str:
    """Standard VADER rule: compound >= 0.05 positive, <= -0.05 negative."""
    c = _vader().polarity_scores(str(text))["compound"]
    if c >= threshold:
        return "positive"
    if c <= -threshold:
        return "negative"
    return "neutral"


def vader_compound(text: str) -> float:
    return _vader().polarity_scores(str(text))["compound"]


def transformer_predictions() -> pd.DataFrame | None:
    """Saved transformer predictions (key column 'key'), or None if not run yet."""
    if TRANSFORMER_PREDICTIONS.exists():
        return pd.read_csv(TRANSFORMER_PREDICTIONS, encoding="utf-8-sig")
    return None
