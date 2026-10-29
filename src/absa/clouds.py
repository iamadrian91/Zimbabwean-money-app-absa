"""Sentiment-coloured word clouds."""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import matplotlib
import numpy as np
import pandas as pd

from .config import SEED
from .text import KEEP_WORDS, clean, stopwords, tokenize

# frequent, uninformative words
UNINFORMATIVE = {
    "app", "apps", "application", "applications", "brandname", "use", "using", "used", "one", "get", "got",
    "please", "also", "even", "still", "would", "could", "can", "will", "like", "just", "really", "much",
    "every", "time", "make", "makes", "thing", "things", "way", "need", "want", "know", "now", "today",
    "zim", "zimbabwe", "guys", "okay", "ok", "thanks", "thank", "you", "lot", "many", "since", "able",
} | (KEEP_WORDS - {"not", "no", "never", "cannot"})
NEGATORS = {"not", "no", "never", "cannot"}
SENTIMENT_VALUE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

# cloud font
_MANROPE = os.path.join(os.path.dirname(__file__), "..", "..", "app", "static", "fonts", "Manrope-ExtraBold.ttf")
FONT = _MANROPE if os.path.exists(_MANROPE) else os.path.join(
    matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")


def _content_words(clean_text: str) -> list[str]:
    sw = stopwords()
    out = []
    for t in tokenize(clean_text):
        if t in NEGATORS:
            out.append(t)
        elif (len(t) >= 3 and t.isalpha() and t not in sw and t not in UNINFORMATIVE):
            out.append(t)
    return out


def word_stats(opinions: pd.DataFrame, min_count: int = 4, pair_min_count: int = 8) -> pd.DataFrame:
    """Per word or common word pair: sentence count (n) and mean sentiment (net)."""
    sent = (opinions.assign(v=opinions["sentiment"].map(SENTIMENT_VALUE))
                    .groupby(["reviewId", "sentence_no"], as_index=False)
                    .agg(sentence=("sentence", "first"), v=("v", "mean")))
    counts, sums = Counter(), defaultdict(float)
    pair_counts, pair_sums = Counter(), defaultdict(float)
    for text, v in zip(sent["sentence"], sent["v"]):
        words = _content_words(clean(text))
        for w in set(x for x in words if x not in NEGATORS):
            counts[w] += 1
            sums[w] += v
        pairs = {f"{a} {b}" for a, b in zip(words, words[1:]) if b not in NEGATORS}
        for p in pairs:
            pair_counts[p] += 1
            pair_sums[p] += v
    rows = [{"word": w, "n": n, "net": sums[w] / n} for w, n in counts.items() if n >= min_count]
    rows += [{"word": p, "n": n, "net": pair_sums[p] / n} for p, n in pair_counts.items() if n >= pair_min_count]
    if not rows:
        return pd.DataFrame(columns=["word", "n", "net"])
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def _rgb(net: float) -> str:
    """Diverging colour for a net value in [-1, 1]."""
    neg, mid, pos = np.array([214, 61, 60]), np.array([150, 148, 140]), np.array([34, 102, 196])
    t = float(np.clip(net, -1, 1))
    c = mid + (neg - mid) * (-t) if t < 0 else mid + (pos - mid) * t
    return "rgb({},{},{})".format(*c.astype(int))


def make_cloud(stats: pd.DataFrame, width: int = 1200, height: int = 560, max_words: int = 90,
               background: str | None = None):
    """Render the word cloud as a PIL image."""
    from wordcloud import WordCloud
    top = stats.head(max_words)
    freqs = dict(zip(top["word"], top["n"]))
    colours = dict(zip(top["word"], top["net"]))
    wc = WordCloud(
        width=width, height=height, font_path=FONT, random_state=SEED,
        background_color=background, mode="RGBA" if background is None else "RGB",
        prefer_horizontal=0.92,
        relative_scaling=0.5,
        min_font_size=11, max_font_size=110, margin=4, collocations=False,
        color_func=lambda word, **_: _rgb(colours.get(word, 0.0)),
    )
    wc.generate_from_frequencies(freqs)
    return wc.to_image()
