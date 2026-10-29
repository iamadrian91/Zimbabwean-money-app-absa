"""Text preprocessing: cleaning rules, tokenisation, sentence and clause splitting."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache

import emoji

from .config import LEXICON_DIR

with open(LEXICON_DIR / "normalisation.json", encoding="utf-8") as f:
    _NORM = json.load(f)


def _word_map_regex(mapping: dict) -> re.Pattern:
    """Whole-word regex for the keys of a mapping, longest first."""
    keys = sorted(mapping, key=len, reverse=True)
    return re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w'])")


_CONTRACTIONS = _NORM["contractions"]
_TEXTSPEAK = _NORM["textspeak"]
_JOINED = _NORM["joined_words"]
_GLOSSARY = _NORM["local_glossary"]
_RE_CONTRACTIONS = _word_map_regex(_CONTRACTIONS)
_RE_TEXTSPEAK = _word_map_regex(_TEXTSPEAK)
_RE_JOINED = _word_map_regex(_JOINED)
_RE_GLOSSARY = _word_map_regex(_GLOSSARY)
_RE_BRANDS = re.compile(r"\b(?:" + "|".join(_NORM["brand_patterns"]) + r")\b")

# generic contraction endings
_RE_NT = re.compile(r"n't\b")
_RE_SUFFIXES = [
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'s\b"), ""),
]

# PII patterns (rule 10)
_RE_PHONE = re.compile(r"(?:\+?263\s?|\b0)7[1378]\d[\s-]?\d{3}[\s-]?\d{3}\b")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")

_RE_ELONGATED = re.compile(r"([a-z])\1{2,}")
_RE_EXCLAIM = re.compile(r"!+")
_RE_QUESTION = re.compile(r"\?+")
_RE_NON_WORD = re.compile(r"[^a-z0-9_!? ]+")
_RE_SPACES = re.compile(r"\s+")
_RE_TOKEN = re.compile(r"[a-z0-9_]+|[!?]")


def mask_pii(text: str) -> str:
    """Replace phone numbers, e-mails and links with placeholder tokens."""
    text = _RE_URL.sub(" urltoken ", text)
    text = _RE_EMAIL.sub(" emailtoken ", text)
    return _RE_PHONE.sub(" phonetoken ", text)


def emojis_to_words(text: str) -> str:
    """Replace emojis with word tokens."""
    return emoji.demojize(text, delimiters=(" emj_", " "))


def clean(text: str, mask_brands: bool = True, contractions: bool = True,
          slang: bool = True, emojis: bool = True, local: bool = True) -> str:
    """Apply the cleaning rules. Flags are for the ablation study."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    text = mask_pii(text)                          # rule 10
    if emojis:
        text = emojis_to_words(text)
    text = text.lower()                            # rule 2

    if contractions:                               # rule 3
        text = _RE_CONTRACTIONS.sub(lambda m: _CONTRACTIONS[m.group(1)], text)
        text = _RE_NT.sub(" not", text)
        for pattern, repl in _RE_SUFFIXES:
            text = pattern.sub(repl, text)

    if slang:
        text = _RE_ELONGATED.sub(r"\1\1", text)    # rule 6
        text = _RE_JOINED.sub(lambda m: _JOINED[m.group(1)], text)        # rule 5
        text = _RE_TEXTSPEAK.sub(lambda m: _TEXTSPEAK[m.group(1)], text)  # rule 4

    if local:                                      # rule 12
        text = _RE_GLOSSARY.sub(lambda m: f"{m.group(1)} {_GLOSSARY[m.group(1)]}", text)

    if mask_brands:                                # rule 7
        text = _RE_BRANDS.sub(" brandname ", text)

    text = _RE_EXCLAIM.sub(" ! ", text)
    text = _RE_QUESTION.sub(" ? ", text)
    text = _RE_NON_WORD.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split CLEANED text into tokens (words, numbers, emoji tokens, ! and ?)."""
    return _RE_TOKEN.findall(text)


# Stopwords
# rule 11: keep negations and contrast words
KEEP_WORDS = {
    "not", "no", "nor", "never", "none", "nothing", "nobody", "cannot",
    "but", "however", "although", "though", "yet", "very", "too", "only",
    "against", "off", "down", "again", "more", "most", "few",
}


@lru_cache(maxsize=1)
def stopwords() -> frozenset:
    from nltk.corpus import stopwords as sw
    words = {w.replace("'", "") for w in sw.words("english")}
    return frozenset(words - KEEP_WORDS)


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = stopwords()
    return [t for t in tokens if t not in sw]


# Sentence and clause splitting (for aspect-level analysis)
_RE_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[a-z]{2}[.!?])(?=[A-Z])"
    r"|\.{2,}\s*"
    r"|\n+"
)

# contrast markers (English, Shona, Ndebele)
CONTRAST_WORDS = [
    "but", "however", "although", "though", "except", "whereas", "yet",
    "otherwise", "unfortunately", "nevertheless", "asi", "kodwa",
]
_RE_CONTRAST = re.compile(r"(?i)[,;]?\s*\b(?=(?:" + "|".join(CONTRAST_WORDS) + r")\b)")
_RE_HAS_LETTER = re.compile(r"[A-Za-z]")


def split_sentences(text: str) -> list[str]:
    """Split a raw review into sentences, keeping the original wording."""
    if not isinstance(text, str):
        return []
    parts = (p.strip(" .,;-") for p in _RE_SENT_BOUNDARY.split(text))
    return [p for p in parts if p and _RE_HAS_LETTER.search(p)]


def split_clauses(sentence: str) -> list[str]:
    """Split a sentence at contrast words."""
    clauses: list[str] = []
    for part in (p.strip(" ,;-") for p in _RE_CONTRAST.split(sentence)):
        if not part:
            continue
        # merge one-word tails
        if len(part.split()) < 2 and clauses:
            clauses[-1] = f"{clauses[-1]} {part}"
        else:
            clauses.append(part)
    return clauses or [sentence]


# Lemmatisation
def lemmatize(clean_texts, batch_size: int = 256) -> list[str]:
    """spaCy lemmas for cleaned texts."""
    from .scope import nlp
    nlp_ = nlp()
    out = []
    with nlp_.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
        for doc in nlp_.pipe(clean_texts, batch_size=batch_size):
            out.append(" ".join(t.lemma_.lower() for t in doc))
    return out
