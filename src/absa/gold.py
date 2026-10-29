"""Gold set: sampling, labelling sheet and loading labels."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GOLD_DIR, SEED

GOLD_SHEET = GOLD_DIR / "gold_to_label.xlsx"
GOLD_KEY = GOLD_DIR / "gold_key.csv"
# labelled sheet (override with ABSA_GOLD_FILE)
GOLD_LABELLED = Path(os.environ.get("ABSA_GOLD_FILE", GOLD_DIR / "gold_labelled.xlsx"))

QUOTAS = {"holdout": 70, "test_known": 60, "test_unseen": 120}
TARGETED_SHARE = 0.80
SENTIMENTS = ["positive", "negative", "neutral"]
N_SLOTS = 3


def sample_gold(sentences: pd.DataFrame, reviews: pd.DataFrame, lexicon) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sent = sentences.drop_duplicates("sent_id").copy()
    sent = sent[sent["sentence"].str.split().str.len() >= 3]
    sent["aspects"] = sent["sentence_clean"].map(lexicon.tag)
    picked = []

    for subset, quota in QUOTAS.items():
        pool = sent[sent["subset"] == subset]
        used_reviews: set = set()
        chosen: list = []
        n_targeted = round(quota * TARGETED_SHARE)

        # targeted: round-robin over aspects, rarest first
        by_aspect = {a: pool[pool["aspects"].map(lambda l, a=a: a in l)] for a in lexicon.aspects}
        order = sorted(by_aspect, key=lambda a: len(by_aspect[a]))
        while len(chosen) < n_targeted and any(len(v) for v in by_aspect.values()):
            for a in order:
                cand = by_aspect[a]
                cand = cand[~cand["reviewId"].isin(used_reviews) & ~cand["sent_id"].isin(chosen)]
                by_aspect[a] = cand
                if len(cand) == 0:
                    continue
                row = cand.iloc[rng.integers(len(cand))]
                chosen.append(row["sent_id"]); used_reviews.add(row["reviewId"])
                if len(chosen) >= n_targeted:
                    break

        # random
        rest = pool[~pool["reviewId"].isin(used_reviews)]
        n_random = quota - len(chosen)
        idx = rng.choice(len(rest), size=min(n_random, len(rest)), replace=False)
        chosen += rest.iloc[idx]["sent_id"].tolist()

        part = pool[pool["sent_id"].isin(chosen)].copy()
        part["sampling"] = np.where(part["sent_id"].isin(chosen[:n_targeted]), "targeted", "random")
        picked.append(part)

    gold = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)
    gold.insert(0, "gold_id", [f"G{i:03d}" for i in range(1, len(gold) + 1)])
    full = reviews.set_index("reviewId")["content"]
    gold["full_review"] = gold["reviewId"].map(full)
    return gold


def write_sheet(gold: pd.DataFrame, lexicon) -> None:
    """Write the labelling sheet and key file."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    gold[["gold_id", "sent_id", "reviewId", "app", "subset", "score", "label", "sampling",
          "sentence"]].to_csv(GOLD_KEY, index=False, encoding="utf-8-sig")

    wb = Workbook()
    ws = wb.active
    ws.title = "Label"
    headers = ["gold_id", "sentence", "full_review"]
    for k in range(1, N_SLOTS + 1):
        headers += [f"aspect_{k}", f"sentiment_{k}"]
    headers += ["notes"]
    ws.append(headers)
    for r in gold.itertuples():
        ws.append([r.gold_id, r.sentence, r.full_review] + [""] * (2 * N_SLOTS + 1))

    head_fill = PatternFill("solid", fgColor="1C1C1E")
    thin = Side(style="thin", color="D1D1D6")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = head_fill
        c.alignment = Alignment(horizontal="center", vertical="center")
    widths = {"A": 8, "B": 60, "C": 70, "J": 30}
    for col in "DEFGHI":
        widths[col] = 15
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
        row[1].font = Font(bold=True, size=11)
        row[2].font = Font(color="8E8E93", size=9)
    ws.freeze_panes = "D2"

    last = len(gold) + 1
    aspect_list = '"' + ",".join(["none"] + lexicon.aspects + ["other"]) + '"'
    dv_a = DataValidation(type="list", formula1=aspect_list, allow_blank=True)
    dv_s = DataValidation(type="list", formula1='"' + ",".join(SENTIMENTS) + '"', allow_blank=True)
    ws.add_data_validation(dv_a); ws.add_data_validation(dv_s)
    for k, col in enumerate("DFH"):
        dv_a.add(f"{col}2:{col}{last}")
    for col in "EGI":
        dv_s.add(f"{col}2:{col}{last}")
    todo = PatternFill("solid", fgColor="FFF4CE")
    ws.conditional_formatting.add(f"D2:E{last}", FormulaRule(formula=["D2=\"\""], fill=todo))
    ws.conditional_formatting.add(f"E2:E{last}", FormulaRule(formula=["E2=\"\""], fill=todo))

    # Guide tab
    g = wb.create_sheet("Guide")
    lines = [
        ("HOW TO LABEL (about 250 rows, roughly 1 to 1.5 hours)", True),
        ("1. Read the bold SENTENCE. The grey full review is only there for context.", False),
        ("2. aspect_1 / sentiment_1: pick the main aspect the sentence talks about and how the writer feels about it.", False),
        ("3. If the sentence talks about a second or third aspect, fill aspect_2/sentiment_2 (and _3).", False),
        ("4. If the sentence mentions NO specific aspect (e.g. 'Great app'), choose aspect_1 = none and still give sentiment_1 for the sentence overall.", False),
        ("5. Use 'other' for a real aspect that is not in the list (and write it in notes).", False),
        ("", False),
        ("SENTIMENT RULES", True),
        ("positive  = praise, satisfaction ('sending money is easy').", False),
        ("negative  = complaint, failure, frustration, OR a request to fix something ('please fix the login').", False),
        ("neutral   = facts, questions or suggestions with no complaint ('it would be nice to have dark mode').", False),
        ("Judge the SENTENCE, not the star rating (you cannot see the stars on purpose).", False),
        ("Shona/Ndebele: label by meaning, e.g. 'chakadhakwa' (malfunctioning) = performance, negative.", False),
        ("", False),
        ("EXAMPLES", True),
        ("'App keeps crashing after the update'  ->  performance/negative,  updates/negative", False),
        ("'Easy to use and sending money is fast' ->  usability/positive,  transactions/positive", False),
        ("'I never receive the OTP'               ->  login/negative", False),
        ("'Charges are too high but the app is good' -> fees/negative", False),
        ("'Thank you'                             ->  none/positive", False),
        ("", False),
        ("ASPECTS", True),
    ]
    for text, bold in lines:
        g.append([text]); g.cell(g.max_row, 1).font = Font(bold=bold, size=12 if bold else 11)
    g.append(["code", "name", "what it covers"])
    for c in g[g.max_row]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = head_fill
    for a in lexicon.aspects:
        g.append([a, lexicon.info[a]["name"], lexicon.info[a]["description"]])
    g.append(["none", "No aspect", "General opinion with no specific aspect ('good app', 'thank you')."])
    g.append(["other", "Other", "A real aspect not listed above. Write it in the notes column."])
    g.column_dimensions["A"].width = 16; g.column_dimensions["B"].width = 28; g.column_dimensions["C"].width = 100
    wb.move_sheet("Guide", offset=-1)
    wb.active = 1
    wb.save(GOLD_SHEET)


def load_labelled(path=GOLD_LABELLED) -> pd.DataFrame:
    """Load labels as one row per (gold_id, aspect)."""
    lab = pd.read_excel(path, sheet_name="Label", dtype=str).fillna("")
    key = pd.read_csv(GOLD_KEY, encoding="utf-8-sig")
    rows = []
    for r in lab.itertuples(index=False):
        d = r._asdict()
        for k in range(1, N_SLOTS + 1):
            a = d[f"aspect_{k}"].strip().lower()
            s = d[f"sentiment_{k}"].strip().lower()
            if a or s:
                rows.append({"gold_id": d["gold_id"], "aspect": a or "none", "sentiment": s})
    long = pd.DataFrame(rows)
    return long.merge(key, on="gold_id", how="left")
