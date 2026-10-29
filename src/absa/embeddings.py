"""FastText embeddings: training, lexicon expansion candidates and mean-vector features."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.base import BaseEstimator, TransformerMixin

from .config import MODELS_DIR, SEED
from .text import remove_stopwords, stopwords, tokenize

# fasttext_train: classifier features; fasttext_all: lexicon expansion
FASTTEXT_FILE = MODELS_DIR / "fasttext_train.model"
FASTTEXT_ALL_FILE = MODELS_DIR / "fasttext_all.model"


def train_fasttext(clean_texts, vector_size: int = 100, epochs: int = 40) -> FastText:
    """Train skip-gram FastText on cleaned texts."""
    sentences = [tokenize(t) for t in clean_texts]
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=2,
        sg=1,
        min_n=3, max_n=5,
        bucket=50_000,
        epochs=epochs,
        seed=SEED, workers=1,
    )
    return model


def expansion_candidates(model: FastText, lexicon, clean_texts, topn: int = 12,
                         min_sim: float = 0.55, min_freq: int = 3) -> pd.DataFrame:
    """Score candidate aspect terms by similarity to the seed terms."""
    freq = Counter(t for txt in clean_texts for t in tokenize(txt))
    known = {w for terms in lexicon.terms.values() for t in terms
             if not t.startswith("re:") for w in t.split()}
    sw = stopwords()
    rows = []
    for aspect, terms in lexicon.terms.items():
        seeds = [t for t in terms if not t.startswith("re:") and " " not in t
                 and t in model.wv.key_to_index and freq[t] >= min_freq]
        hits: dict[str, list[float]] = {}
        for s in seeds:
            for w, sim in model.wv.most_similar(s, topn=topn):
                if (sim >= min_sim and w not in known and w not in sw and w.isalpha()
                        and len(w) >= 3 and freq[w] >= min_freq):
                    hits.setdefault(w, []).append(sim)
        for w, sims in hits.items():
            rows.append({"aspect": aspect, "candidate": w, "n_seeds": len(sims),
                         "mean_sim": float(np.mean(sims)), "freq": freq[w],
                         "score": float(np.mean(sims)) * np.log1p(len(sims))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # keep each word's best-scoring aspect
    out = out.sort_values("score", ascending=False).drop_duplicates("candidate")
    return out.sort_values(["aspect", "score"], ascending=[True, False]).reset_index(drop=True)


class MeanEmbedding(BaseEstimator, TransformerMixin):
    """Mean FastText vector of a cleaned text (stopwords removed)."""

    def __init__(self, model_path: str = str(FASTTEXT_FILE)):
        self.model_path = model_path

    def fit(self, X, y=None):
        self.model_ = FastText.load(self.model_path)
        return self

    def transform(self, X):
        wv = self.model_.wv
        out = np.zeros((len(X), wv.vector_size), dtype=np.float32)
        for i, text in enumerate(X):
            toks = remove_stopwords(tokenize(text)) or tokenize(text)
            if toks:
                out[i] = np.mean([wv[t] for t in toks], axis=0)
        return out
