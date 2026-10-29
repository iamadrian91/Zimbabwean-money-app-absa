"""Download NLTK data and the spaCy model if missing."""

from __future__ import annotations

import subprocess
import sys

NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "words": "corpora/words",
}
SPACY_MODEL = "en_core_web_sm"


def ensure_resources(verbose: bool = False) -> None:
    import nltk
    for pkg, path in NLTK_PACKAGES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=not verbose)

    import spacy.util
    if not spacy.util.is_package(SPACY_MODEL):
        print(f"Downloading spaCy model {SPACY_MODEL} (about 12 MB, one time only)...")
        subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)
