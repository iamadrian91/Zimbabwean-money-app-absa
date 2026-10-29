"""Transformer baseline (cardiffnlp/twitter-roberta-base-sentiment-latest), inference only.

Requires torch and transformers. Writes reports/tables/transformer_predictions.csv.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from absa.baselines import TRANSFORMER_MODEL, TRANSFORMER_PREDICTIONS  # noqa: E402
from absa.data import load_reviews  # noqa: E402
from absa.gold import GOLD_LABELLED  # noqa: E402


def main():
    from transformers import pipeline
    print(f"Loading {TRANSFORMER_MODEL} (downloads ~500 MB the first time)...")
    clf = pipeline("sentiment-analysis", model=TRANSFORMER_MODEL, tokenizer=TRANSFORMER_MODEL,
                   truncation=True, max_length=256, device=-1)

    def predict(texts, what):
        t0 = time.time()
        out = clf(list(texts), batch_size=32)
        print(f"  {what}: {len(texts)} texts in {time.time() - t0:.0f}s")
        return [o["label"].lower() for o in out]

    rows = []
    reviews = load_reviews()
    test = reviews[reviews["subset"].isin(["holdout", "test_known", "test_unseen"])]
    preds = predict(test["content"].astype(str).tolist(), "test reviews")
    rows += [{"key": k, "pred": p} for k, p in zip(test["reviewId"], preds)]

    if GOLD_LABELLED.exists():
        from absa.absa_eval import aspect_units, sentence_table
        from absa.aspects import AspectLexicon
        from absa.gold import load_labelled
        table = sentence_table(load_labelled())
        lex = AspectLexicon()
        for strategy in ("sentence", "clause", "dependency"):
            u = aspect_units(table, lex, strategy)
            p = predict(u["scope"].tolist(), f"gold scopes ({strategy})")
            rows += [{"key": f"{g}|{a}|{strategy}", "pred": x} for g, a, x in zip(u["gold_id"], u["aspect"], p)]
    else:
        print("  (gold_labelled.xlsx not found: aspect-level transformer scores skipped)")

    pd.DataFrame(rows).to_csv(TRANSFORMER_PREDICTIONS, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} predictions to {TRANSFORMER_PREDICTIONS}")


if __name__ == "__main__":
    main()
