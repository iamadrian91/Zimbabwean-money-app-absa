"""Lexicon-based aspect detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import LEXICON_DIR

SEED_FILE = LEXICON_DIR / "aspects_seed.json"
EXPANDED_FILE = LEXICON_DIR / "aspects_expanded.json"


def _term_to_regex(term: str) -> str:
    """Convert a lexicon entry to a regex fragment."""
    if term.startswith("re:"):
        return term[3:]
    words = [re.escape(w) for w in term.split()]
    if len(words) == 1 and not term.endswith("s"):
        return words[0] + r"(?:s|es)?"
    return r"\s+".join(words)


def _compile(terms: list[str]) -> re.Pattern:
    # longest terms first
    parts = sorted({_term_to_regex(t) for t in terms}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


@dataclass
class Match:
    aspect: str
    term: str
    start: int
    end: int


class AspectLexicon:
    """Loads the lexicon files and tags cleaned text with aspects."""

    def __init__(self, use_expansion: bool = True,
                 seed_file: Path = SEED_FILE, expanded_file: Path = EXPANDED_FILE):
        with open(seed_file, encoding="utf-8") as f:
            raw = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

        # optional expansion terms
        expansion = {}
        if use_expansion and expanded_file.exists():
            with open(expanded_file, encoding="utf-8") as f:
                expansion = json.load(f).get("accepted", {})

        self.info = {}
        self.terms = {}
        self._patterns = {}
        self._exclude = {}
        for aspect, spec in raw.items():
            terms = spec["terms"] + spec.get("local", []) + expansion.get(aspect, [])
            self.info[aspect] = {k: spec[k] for k in ("name", "level", "description")}
            self.terms[aspect] = terms
            self._patterns[aspect] = _compile(terms)
            if spec.get("exclude"):
                self._exclude[aspect] = _compile(spec["exclude"])

    @property
    def aspects(self) -> list[str]:
        return list(self._patterns)

    def name(self, aspect: str) -> str:
        return self.info[aspect]["name"]

    def find(self, clean_text: str) -> list[Match]:
        """All aspect-term matches in a CLEANED text, in reading order."""
        found = []
        for aspect, pattern in self._patterns.items():
            excluded = []
            if aspect in self._exclude:
                excluded = [m.span() for m in self._exclude[aspect].finditer(clean_text)]
            for m in pattern.finditer(clean_text):
                # drop matches inside exclusions
                if any(s <= m.start() and m.end() <= e for s, e in excluded):
                    continue
                found.append(Match(aspect, m.group(0), m.start(), m.end()))
        return sorted(found, key=lambda m: m.start)

    def tag(self, clean_text: str) -> list[str]:
        """Unique aspects mentioned in a cleaned text (reading order)."""
        seen = []
        for m in self.find(clean_text):
            if m.aspect not in seen:
                seen.append(m.aspect)
        return seen
