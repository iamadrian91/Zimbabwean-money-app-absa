"""Feature representations and classifiers."""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import SEED
from .embeddings import MeanEmbedding
from .text import tokenize


def _word_tfidf():
    return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, lowercase=False)


FEATURES = {
    "bow": lambda: CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), min_df=2, binary=True),
    "tfidf_word": _word_tfidf,
    "tfidf_char": _char_tfidf,
    "tfidf_word_char": lambda: FeatureUnion([("word", _word_tfidf()), ("char", _char_tfidf())]),
    "fasttext": lambda: make_pipeline(MeanEmbedding(), StandardScaler()),
}

FEATURE_NAMES = {
    "bow": "Bag of Words", "tfidf_word": "TF-IDF (words)", "tfidf_char": "TF-IDF (characters)",
    "tfidf_word_char": "TF-IDF (words + characters)", "fasttext": "FastText (mean vector)",
}
CLASSIFIER_NAMES = {"nb": "Naive Bayes", "svm": "Linear SVM", "logreg": "Logistic Regression"}

PARAM_GRIDS = {
    "nb": {"clf__alpha": [0.1, 0.3, 1.0]},
    "svm": {"clf__C": [0.03, 0.1, 0.3, 1.0]},
    "logreg": {"clf__C": [0.3, 1.0, 3.0, 10.0]},
}


def make_classifier(name: str, feature: str, balanced: bool = True):
    weight = "balanced" if balanced else None
    if name == "nb":
        return GaussianNB() if feature == "fasttext" else ComplementNB()
    if name == "svm":
        return LinearSVC(class_weight=weight, random_state=SEED)
    if name == "logreg":
        return LogisticRegression(class_weight=weight, max_iter=3000, random_state=SEED)
    raise ValueError(name)


def make_pipeline_for(feature: str, classifier: str, balanced: bool = True) -> Pipeline:
    """Feature-classifier pipeline."""
    return Pipeline([("features", FEATURES[feature]()),
                     ("clf", make_classifier(classifier, feature, balanced))])


def param_grid(feature: str, classifier: str) -> dict:
    if classifier == "nb" and feature == "fasttext":
        return {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]}
    return PARAM_GRIDS[classifier]


def top_features(pipeline: Pipeline, n: int = 15) -> dict:
    """Top weighted word features per class."""
    feats = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {}
    names = feats.get_feature_names_out()
    out = {}
    coefs = clf.coef_
    classes = clf.classes_
    for k, cls in enumerate(classes):
        order = coefs[k].argsort()[::-1][:n]
        out[cls] = [(names[i], float(coefs[k][i])) for i in order]
    return out
