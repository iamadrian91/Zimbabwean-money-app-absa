"""Project settings: paths, apps, splits and labels."""

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_REVIEWS = DATA_DIR / "raw_reviews.csv"
LEXICON_DIR = DATA_DIR / "lexicons"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

for _d in (PROCESSED_DIR, GOLD_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42

# Apps
# split: train/test side; test_group: known brand vs unseen brands
APPS = {
    "EcoCash":           {"id": "com.ecocash",             "brand": "EcoCash",  "type": "mobile money", "split": "train"},
    "InnBucks":          {"id": "com.innbucks.customer",   "brand": "InnBucks", "type": "mobile money", "split": "train"},
    "CBZ Touch":         {"id": "com.mode.cbz.ui",         "brand": "CBZ",      "type": "banking",      "split": "train"},
    "EcoCash Super App": {"id": "com.ecocash.superapp",    "brand": "EcoCash",  "type": "mobile money", "split": "test", "test_group": "known_brand"},
    "O'mari":            {"id": "com.hypercubesoft.omari", "brand": "O'mari",   "type": "mobile money", "split": "test", "test_group": "unseen_brand"},
    "OneMoney":          {"id": None,                      "brand": "OneMoney", "type": "mobile money", "split": "test", "test_group": "unseen_brand"},
    "MyZB":              {"id": None,                      "brand": "ZB",       "type": "banking",      "split": "test", "test_group": "unseen_brand"},
    "CABS":              {"id": None,                      "brand": "CABS",     "type": "banking",      "split": "test", "test_group": "unseen_brand"},
}

TRAIN_APPS = [a for a, v in APPS.items() if v["split"] == "train"]
TEST_APPS = [a for a, v in APPS.items() if v["split"] == "test"]

# share of training-app reviews held out
HOLDOUT_FRACTION = 0.20

# Labels
LABELS = ["negative", "neutral", "positive"]


def stars_to_label(score: int) -> str:
    """1-2 stars negative, 3 neutral, 4-5 positive."""
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


# Analysis window
# app comparisons use reviews from this date
ANALYSIS_START = "2024-01-01"

# short-review threshold (words)
SHORT_REVIEW_MAX_WORDS = 3
