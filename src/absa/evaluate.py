"""Metrics, bootstrap confidence intervals, McNemar test and confusion matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .config import LABELS, SEED


def scores(y_true, y_pred, labels=LABELS) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    for lab, pi, ri, fi in zip(labels, p, r, f):
        out[f"f1_{lab}"] = float(fi)
        out[f"precision_{lab}"] = float(pi)
        out[f"recall_{lab}"] = float(ri)
    return out


def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1", n_boot: int = 1000,
                 labels=LABELS, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if metric == "macro_f1":
            vals.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
        else:
            vals.append(accuracy_score(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired predictions."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def confusion(y_true, y_pred, labels=LABELS, normalize: bool = False) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return pd.DataFrame(cm, index=[f"true {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])


def score_table(y_true, preds: dict, labels=LABELS, ci: bool = True) -> pd.DataFrame:
    """One row per model: accuracy, macro-P/R/F1 (+95% CI), per-class F1."""
    rows = []
    for name, y_pred in preds.items():
        s = scores(y_true, y_pred, labels)
        row = {"model": name, **{k: s[k] for k in ("n", "accuracy", "macro_precision",
                                                   "macro_recall", "macro_f1")}}
        if ci:
            lo, hi = bootstrap_ci(y_true, y_pred, labels=labels)
            row["macro_f1_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        for lab in labels:
            row[f"f1_{lab}"] = s[f"f1_{lab}"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
