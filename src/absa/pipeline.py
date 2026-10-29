"""End-to-end aspect-based sentiment analyser."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from .aspects import AspectLexicon
from .config import MODELS_DIR
from .scope import nlp, scopes_for
from .text import clean, split_sentences

FINAL_MODEL = MODELS_DIR / "sentiment_final.joblib"
EVAL_MODEL = MODELS_DIR / "sentiment_eval.joblib"


@dataclass
class AspectOpinion:
    sentence_no: int
    sentence: str
    aspect: str
    term: str
    scope: str
    sentiment: str
    confidence: float


class ABSAnalyzer:
    def __init__(self, model=None, lexicon: AspectLexicon | None = None,
                 strategy: str = "dependency", model_path=FINAL_MODEL):
        self.model = model if model is not None else joblib.load(model_path)
        self.lexicon = lexicon or AspectLexicon()
        self.strategy = strategy

    def _predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(texts)
            classes = self.model.classes_
            return classes[proba.argmax(1)], proba.max(1)
        pred = self.model.predict(texts)
        return pred, np.ones(len(texts))

    def analyse(self, review: str) -> list[AspectOpinion]:
        units = []
        sentences = split_sentences(review)
        cleaned = [clean(s) for s in sentences]
        need_parse = self.strategy == "dependency"
        docs = list(nlp().pipe(cleaned)) if need_parse else [None] * len(cleaned)
        for i, (sent, c, doc) in enumerate(zip(sentences, cleaned, docs)):
            matches = self.lexicon.find(c)
            if not matches:
                units.append((i, sent, "general", "", c))
                continue
            seen = set()
            for m, scope in zip(matches, scopes_for(c, matches, self.strategy, doc)):
                if m.aspect in seen:
                    continue
                seen.add(m.aspect)
                units.append((i, sent, m.aspect, m.term, scope))
        if not units:
            return []
        labels, conf = self._predict([u[4] for u in units])
        return [AspectOpinion(u[0], u[1], u[2], u[3], u[4], str(l), float(p))
                for u, l, p in zip(units, labels, conf)]

    def analyse_many(self, reviews: pd.DataFrame, text_col: str = "content",
                     keep_cols=("reviewId", "app", "brand", "at", "score")) -> pd.DataFrame:
        """Analyse a DataFrame of reviews; one row per aspect opinion."""
        rows, scopes = [], []
        texts = reviews[text_col].tolist()
        all_sents = [split_sentences(t) for t in texts]
        flat = [clean(s) for sents in all_sents for s in sents]
        docs = iter(nlp().pipe(flat, batch_size=256)) if self.strategy == "dependency" else None
        for (_, r), sents in zip(reviews.iterrows(), all_sents):
            meta = {k: r[k] for k in keep_cols if k in r}
            for i, s in enumerate(sents):
                c = clean(s)
                doc = next(docs) if docs is not None else None
                matches = self.lexicon.find(c)
                if not matches:
                    rows.append({**meta, "sentence_no": i, "sentence": s, "aspect": "general", "term": ""})
                    scopes.append(c)
                    continue
                seen = set()
                for m, scope in zip(matches, scopes_for(c, matches, self.strategy, doc)):
                    if m.aspect in seen:
                        continue
                    seen.add(m.aspect)
                    rows.append({**meta, "sentence_no": i, "sentence": s, "aspect": m.aspect, "term": m.term})
                    scopes.append(scope)
        out = pd.DataFrame(rows)
        labels, conf = self._predict(scopes)
        out["scope"] = scopes
        out["sentiment"] = labels
        out["confidence"] = conf
        return out


def to_frame(opinions: list[AspectOpinion]) -> pd.DataFrame:
    return pd.DataFrame([asdict(o) for o in opinions])
