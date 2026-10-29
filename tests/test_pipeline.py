"""Tests for preprocessing, splitting, aspect detection and opinion scope."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absa.aspects import AspectLexicon  # noqa: E402
from absa.text import (clean, remove_stopwords, split_clauses,  # noqa: E402
                       split_sentences, tokenize)


# Cleaning rules
@pytest.mark.parametrize("raw, expected", [
    ("It doesn't work", "it does not work"),          # rule 3
    ("I cant login", "i cannot login"),
    ("won't open", "will not open"),
    ("pls fix", "please fix"),                        # rule 4
    ("it nolonger opens", "it no longer opens"),      # rule 5
    ("goood app", "good app"),                        # rule 6
])
def test_normalisation(raw, expected):
    assert clean(raw) == expected


@pytest.mark.parametrize("brand", ["EcoCash", "eco cash", "Innbucks", "lnnbucks", "CBZ Touch",
                                   "my zb", "O'mari", "O’mari", "OneMoney"])
def test_brands_are_masked(brand):                    # rule 7
    assert clean(f"{brand} is slow") == "brandname is slow"


def test_device_brands_are_kept():
    assert "samsung" in clean("Not working on my Samsung")


def test_pii_is_masked():                             # rule 10
    out = clean("call 0772 123 456 or +263772123456 or mail me@test.com")
    assert out.count("phonetoken") == 2 and "emailtoken" in out
    assert "0772" not in out


def test_emojis_become_tokens():
    assert "emj_thumbs_up" in clean("good 👍")


def test_exclamations_kept_as_tokens():
    assert tokenize(clean("useless!!!")) == ["useless", "!"]


def test_local_glossary():                            # rule 12
    assert clean("chakadhakwa") == "chakadhakwa malfunctioning"
    assert clean("App haishande") == "app haishande not working"


def test_negations_survive_stopword_removal():        # rule 11
    toks = remove_stopwords(tokenize(clean("The app is not working but it is fast")))
    assert "not" in toks and "but" in toks and "the" not in toks


# Sentence and clause splitting
def test_sentence_split_handles_missing_space():
    assert split_sentences("Great app.Improvements still needed") == ["Great app", "Improvements still needed"]


def test_sentence_split_drops_emoji_only():
    assert split_sentences("👍👍") == []


@pytest.mark.parametrize("sentence, n", [
    ("good app but slow", 2),
    ("limit apa asi the app is quite fast", 2),
    ("Improvements still needed though", 1),
])
def test_clause_split(sentence, n):
    assert len(split_clauses(sentence)) == n


# Aspect detection
@pytest.fixture(scope="module")
def lex():
    return AspectLexicon()


@pytest.mark.parametrize("text, aspect", [
    ("I never receive the OTP", "login"),
    ("Charges are too high", "fees"),
    ("app keeps crashing", "performance"),
    ("it does not open since the update", "updates"),
    ("chakadhakwa app iyi", "performance"),
    ("asi itai kuti tiwane kuswiper", "cards"),
    ("Ikuramba kuloga in newifi", "connectivity"),
])
def test_aspect_found(lex, text, aspect):
    assert aspect in lex.tag(clean(text))


@pytest.mark.parametrize("text, not_aspect", [
    ("I never receive the OTP", "transactions"),
    ("it sends me the code", "transactions"),
    ("My sim card is roaming", "cards"),
    ("Rate this app 5 stars", "fees"),
])
def test_false_positives_avoided(lex, text, not_aspect):
    assert not_aspect not in lex.tag(clean(text))


# Opinion scope
def test_clause_scope_drops_leading_contrast_word(lex):
    from absa.scope import scopes_for
    c = clean("Easy to use but the charges are too high")
    scopes = dict((m.aspect, s) for m, s in zip(lex.find(c), scopes_for(c, lex.find(c), "clause")))
    assert scopes["fees"] == "the charges are too high"
    assert scopes["usability"] == "easy to use"
