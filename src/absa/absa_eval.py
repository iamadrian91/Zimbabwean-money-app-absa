"""Aspect-level evaluation against the gold set: detection, sentiment and end-to-end."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABELS
from .evaluate import scores
from .scope import nlp, scopes_for
from .text import clean


def sentence_table(gold_long: pd.DataFrame) -> pd.DataFrame:
    """One row per gold sentence: its human aspect set and per-aspect sentiment."""
    rows = []
    for gid, g in gold_long.groupby("gold_id", sort=False):
        real = g[~g["aspect"].isin(["none", "other", ""])]
        rows.append({
            "gold_id": gid, "sentence": g["sentence"].iloc[0], "subset": g["subset"].iloc[0],
            "app": g["app"].iloc[0], "review_label": g["label"].iloc[0], "sampling": g["sampling"].iloc[0],
            "gold_aspects": set(real["aspect"]),
            "gold_pairs": set(zip(real["aspect"], real["sentiment"])),
            "has_other": (g["aspect"] == "other").any(),
            "sentence_sentiment": _sentence_sentiment(g["sentiment"]),
        })
    out = pd.DataFrame(rows)
    out["clean"] = out["sentence"].map(clean)
    return out


def _sentence_sentiment(sents: pd.Series) -> str:
    vals = [s for s in sents if s in LABELS]
    if not vals:
        return ""
    if len(set(vals)) == 1:
        return vals[0]
    return "mixed"


def detection_scores(table: pd.DataFrame, lexicon, aspects: list[str] | None = None) -> tuple[dict, pd.DataFrame]:
    """Micro P/R/F1 over (sentence, aspect) pairs + a per-aspect table."""
    aspects = aspects or lexicon.aspects
    pred = table["clean"].map(lambda c: set(lexicon.tag(c)))
    tp = fp = fn = 0
    per = {a: {"tp": 0, "fp": 0, "fn": 0} for a in aspects}
    for g, p in zip(table["gold_aspects"], pred):
        for a in aspects:
            if a in p and a in g:
                tp += 1; per[a]["tp"] += 1
            elif a in p:
                fp += 1; per[a]["fp"] += 1
            elif a in g:
                fn += 1; per[a]["fn"] += 1
    micro = _prf(tp, fp, fn)
    rows = []
    for a, c in per.items():
        rows.append({"aspect": a, "gold": c["tp"] + c["fn"], "predicted": c["tp"] + c["fp"],
                     **_prf(c["tp"], c["fp"], c["fn"])})
    per_aspect = pd.DataFrame(rows).set_index("aspect")
    micro["macro_f1"] = float(per_aspect.loc[per_aspect["gold"] > 0, "f1"].mean())
    return micro, per_aspect


def _prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def aspect_units(table: pd.DataFrame, lexicon, strategy: str) -> pd.DataFrame:
    """One row per human (sentence, aspect, sentiment) pair with its opinion scope."""
    rows = []
    docs = list(nlp().pipe(table["clean"])) if strategy == "dependency" else [None] * len(table)
    for r, doc in zip(table.itertuples(index=False), docs):
        matches = lexicon.find(r.clean)
        scopes = scopes_for(r.clean, matches, strategy, doc) if matches else []
        first = {}
        for m, s in zip(matches, scopes):
            first.setdefault(m.aspect, s)
        for aspect, sentiment in sorted(r.gold_pairs):   # deterministic order
            rows.append({"gold_id": r.gold_id, "subset": r.subset, "aspect": aspect,
                         "gold_sentiment": sentiment, "review_label": r.review_label,
                         "sentence": r.sentence, "detected": aspect in first,
                         "scope": first.get(aspect, r.clean), "strategy": strategy})
    return pd.DataFrame(rows)


def end_to_end(table: pd.DataFrame, analyzer) -> dict:
    """Tuple-level P/R/F1: predicted (aspect, sentiment) vs human (aspect, sentiment)."""
    tp = fp = fn = 0
    for r in table.itertuples(index=False):
        ops = analyzer.analyse(r.sentence)
        pred = {(o.aspect, o.sentiment) for o in ops if o.aspect != "general"}
        gold = set(r.gold_pairs)
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    return _prf(tp, fp, fn)


def sentiment_scores(units: pd.DataFrame, pred_col: str) -> dict:
    d = units[units["gold_sentiment"].isin(LABELS)]
    return scores(d["gold_sentiment"], d[pred_col])
