"""Opinion scope strategies: sentence, clause and dependency."""

from __future__ import annotations

from functools import lru_cache

import re

from .text import CONTRAST_WORDS, split_clauses

# drop the leading contrast word before scoring
_RE_LEADING_CONTRAST = re.compile(r"^(?:" + "|".join(CONTRAST_WORDS) + r")\b\s*")


def strip_contrast(clause: str) -> str:
    out = _RE_LEADING_CONTRAST.sub("", clause)
    return out if len(out.split()) >= 2 else clause

# clause-level dependency labels
CLAUSE_DEPS = {"ROOT", "conj", "advcl", "ccomp", "relcl", "parataxis", "xcomp"}
CLAUSE_POS = {"VERB", "AUX", "ADJ"}
STRATEGIES = ("sentence", "clause", "dependency")


@lru_cache(maxsize=1)
def nlp():
    """Load spaCy (NER disabled)."""
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


def _clause_head(token):
    """Nearest clause head above the aspect token."""
    t = token
    while True:
        if t.dep_ in CLAUSE_DEPS and t.pos_ in CLAUSE_POS:
            return t
        if t.head.i == t.i:
            return t
        t = t.head


def dependency_scope(clean_sentence: str, start: int, end: int, doc=None) -> str:
    """Text of the clause governing the aspect term at characters [start, end)."""
    doc = doc if doc is not None else nlp()(clean_sentence)
    targets = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
    if not targets:
        return clean_sentence
    head = _clause_head(targets[0])
    # drop nested clauses without the aspect
    keep = set()
    for t in head.subtree:
        keep.add(t.i)
    for child in head.children:
        if child.dep_ in CLAUSE_DEPS - {"xcomp"} and child.pos_ in CLAUSE_POS \
                and not any(t.i == targets[0].i for t in child.subtree):
            for t in child.subtree:
                keep.discard(t.i)
    words = [doc[i].text for i in sorted(keep)]
    scope = " ".join(words)
    return scope if len(words) >= 2 else clean_sentence


def scopes_for(clean_sentence: str, matches, strategy: str = "dependency", doc=None) -> list[str]:
    """One opinion scope (text) per aspect match, under the chosen strategy."""
    if strategy == "sentence":
        return [clean_sentence for _ in matches]
    if strategy == "clause":
        clauses = split_clauses(clean_sentence)
        out, pos = [], 0
        bounds = []
        for c in clauses:
            i = clean_sentence.find(c, pos)
            bounds.append((i, i + len(c))); pos = max(pos, i + len(c))
        for m in matches:
            hit = next((c for c, (s, e) in zip(clauses, bounds) if s <= m.start < e), clean_sentence)
            out.append(strip_contrast(hit))
        return out
    if strategy == "dependency":
        doc = doc if doc is not None else nlp()(clean_sentence)
        return [dependency_scope(clean_sentence, m.start, m.end, doc) for m in matches]
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
