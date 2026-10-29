"""MariLens Streamlit app. Run: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(APP_DIR))

import components as ui  # noqa: E402
from absa.config import ANALYSIS_START, APPS, PROCESSED_DIR, TABLES_DIR  # noqa: E402

st.set_page_config(page_title="MariLens", page_icon=str(APP_DIR / "static" / "icon.png"),
                   layout="wide", initial_sidebar_state="collapsed")
st.html(f"<style>{(APP_DIR / 'style.css').read_text(encoding='utf-8')}</style>")

MIN_MENTIONS = 8
SENT_VALUE = {"negative": -1, "neutral": 0, "positive": 1}


def show(fragment: str) -> None:
    """Render an HTML fragment."""
    st.markdown(fragment, unsafe_allow_html=True)


# Data and models, loaded once and cached
@st.cache_data(show_spinner=False)
def load_opinions() -> pd.DataFrame:
    path = PROCESSED_DIR / "absa_opinions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["at"], encoding="utf-8-sig", keep_default_na=False)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    path = PROCESSED_DIR / "reviews.csv"
    if not path.exists():
        return pd.DataFrame(columns=["reviewId", "app", "content", "score", "at"])
    return pd.read_csv(path, parse_dates=["at"], encoding="utf-8-sig", keep_default_na=False,
                       usecols=["reviewId", "app", "content", "score", "at"])


@st.cache_resource(show_spinner=False)
def get_analyzer():
    from absa.pipeline import ABSAnalyzer
    strategy = "clause"
    summary = TABLES_DIR / "06_summary.json"
    if summary.exists():
        strategy = json.load(open(summary))["aspect_sentiment_best_scope"]
    return ABSAnalyzer(strategy=strategy)


@st.cache_resource(show_spinner=False)
def get_lexicon():
    from absa.aspects import AspectLexicon
    return AspectLexicon()


@st.cache_data(show_spinner=False)
def cloud_png(op: pd.DataFrame, side: str) -> tuple[bytes, list[str]]:
    """Word cloud PNG and its word list."""
    from absa.clouds import make_cloud, word_stats
    stats = word_stats(op, min_count=3, pair_min_count=5)
    if side == "Complaints":
        stats = stats[stats["net"] < -0.2]
    elif side == "Praise":
        stats = stats[stats["net"] > 0.2]
    if stats.empty:
        return b"", []
    buf = io.BytesIO()
    make_cloud(stats, width=900, height=520, max_words=70).save(buf, format="PNG")
    return buf.getvalue(), stats["word"].head(40).tolist()


# Helpers
def aspect_table(op: pd.DataFrame) -> pd.DataFrame:
    """Per aspect: counts of each sentiment, total n and net score (aspect opinions only)."""
    a = op[op["aspect"] != "general"]
    cols = ["negative", "neutral", "positive"]
    if a.empty:
        return pd.DataFrame(columns=cols + ["n", "net", "name"])
    t = a.groupby("aspect")["sentiment"].value_counts().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
    t["n"] = t[cols].sum(axis=1)
    t["net"] = (t["positive"] - t["negative"]) / t["n"]
    lex = get_lexicon()
    t["name"] = [lex.name(x) if x in lex.info else x for x in t.index]
    return t.sort_values("n", ascending=False)


def overall_net(op: pd.DataFrame) -> float:
    a = op[op["aspect"] != "general"]
    return float(a["sentiment"].map(SENT_VALUE).mean()) if len(a) else 0.0


def card_for(app: str, op: pd.DataFrame, n_reviews: int, selected=False, big=False, seed=0) -> str:
    t = aspect_table(op)
    t = t[t["n"] >= MIN_MENTIONS]
    best = t.loc[t["net"].idxmax(), "name"] if len(t) and t["net"].max() > 0 else ""
    worst = t.loc[t["net"].idxmin(), "name"] if len(t) and t["net"].min() < 0 else ""
    kind = APPS[app]["type"].capitalize() if app in APPS else "Live"
    return ui.bank_card(app, kind, overall_net(op), n_reviews, best, worst, selected, big, seed)


def quotes_for(op: pd.DataFrame, aspect: str, want: str | None, k: int = 3) -> list[dict]:
    q = op[op["aspect"] == aspect]
    if want:
        q = q[q["sentiment"] == want]
    q = q[q["sentence"].str.len().between(25, 220)].sort_values("confidence", ascending=False).head(k)
    out = []
    for r in q.itertuples():
        when = pd.Timestamp(r.at).strftime("%b %Y") if pd.notna(r.at) else ""
        stars = f"{int(r.score)} star{'s' if int(r.score) != 1 else ''}" if str(r.score).isdigit() else ""
        out.append({"sentence": r.sentence, "term": r.term, "sentiment": r.sentiment,
                    "meta": ", ".join(x for x in [r.app, when, stars] if x)})
    return out


def statement_rows(op: pd.DataFrame) -> list[dict]:
    t = aspect_table(op)
    t = t[t["n"] >= MIN_MENTIONS]
    rows = []
    for r in t.itertuples():
        want = "negative" if r.net < 0 else "positive"
        rows.append({"name": r.name, "net": r.net, "n": int(r.n), "neg": int(r.negative), "neu": int(r.neutral),
                     "pos": int(r.positive), "quotes": quotes_for(op, r.Index, want)})
    return rows


def feed_items(op: pd.DataFrame, reviews: pd.DataFrame, k: int = 8) -> list[dict]:
    lex = get_lexicon()
    ids = set(op["reviewId"])
    rv = reviews[reviews["reviewId"].isin(ids) & (reviews["content"].str.split().str.len() >= 6)]
    rv = rv.sort_values("at", ascending=False).head(k)
    tags = op[op["aspect"] != "general"].groupby("reviewId")
    out = []
    for r in rv.itertuples():
        t = []
        if r.reviewId in tags.groups:
            g = tags.get_group(r.reviewId).drop_duplicates("aspect")
            t = [(lex.name(a), s) for a, s in zip(g["aspect"], g["sentiment"])][:4]
        out.append({"day": r.at.day, "month": r.at.strftime("%b"), "stars": int(r.score),
                    "text": r.content, "tags": t})
    return out


def word_lookup(op: pd.DataFrame, words: list[str], key: str) -> None:
    """Word lookup: sentences that use a chosen word."""
    if not words:
        return
    word = st.selectbox("Read what people wrote about a word in the cloud", words, key=key)
    hits = op[op["sentence"].str.contains(rf"\b{word}\b", case=False, regex=True)]
    hits = hits.drop_duplicates(["reviewId", "sentence_no"]).sort_values("at", ascending=False).head(4)
    quotes = []
    for r in hits.itertuples():
        quotes.append(f'<div class="mx-quote" style="--q:{ui.POS if r.sentiment == "positive" else ui.NEG if r.sentiment == "negative" else ui.NEU}">'
                      f'{ui._highlight(r.sentence, word)}<small>{ui.esc(r.app)}, {pd.Timestamp(r.at):%b %Y}</small></div>')
    show('<div class="mx-quotes" style="padding-top:6px">' + "".join(quotes) + "</div>")


def app_detail(op: pd.DataFrame, reviews: pd.DataFrame, key: str) -> None:
    """App detail view."""
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        show(ui.panel("Mood statement", "tap a line to read the reviews",
                      ui.statement(statement_rows(op)) or ""))
        show(ui.note(f"Each line is an aspect with at least {MIN_MENTIONS} mentions. The number is its mood score: "
                     "praise minus complaints as a share of mentions, from −100 to +100. The bar shows "
                     "complaints (red), neutral (grey) and praise (blue)."))
    with right:
        show('<div class="mx-h2" style="margin:4px 0 8px">What people say</div>')
        side = st.segmented_control("Word cloud", ["All words", "Complaints", "Praise"], default="All words",
                                    key=f"{key}_cloud", label_visibility="collapsed") or "All words"
        png, words = cloud_png(op, side)
        if png:
            st.image(png, width="stretch")
            show(ui.note("Bigger words are used more often. Red words come mostly from complaints, blue from praise."))
            word_lookup(op, words, key=f"{key}_word")
        else:
            show(ui.note("Not enough text for a word cloud."))
        items = feed_items(op, reviews)
        if items:
            show(ui.panel("Latest reviews", f"{len(items)} most recent", ui.feed(items)))


# Header and navigation
SECTIONS = ["Wallet", "Compare", "Check a review", "Find an app", "How it works"]
top_l, top_r = st.columns([1, 1.6], vertical_alignment="center")
with top_l:
    show(ui.top_bar())
with top_r:
    section = st.segmented_control("Section", SECTIONS, default="Wallet", key="section",
                                   label_visibility="collapsed") or "Wallet"

opinions = load_opinions()
reviews = load_reviews()
if opinions.empty and section in ("Wallet", "Compare"):
    st.warning("No analysis results yet. Run notebook 07_analysis.ipynb first: it writes "
               "data/processed/absa_opinions.csv, which this page reads.")
    st.stop()


# Wallet
if section == "Wallet":
    show(ui.heading("Which money app treats its users best?",
                    "Each card is an app. Its mood score comes from what reviewers wrote about it, aspect by "
                    "aspect, not from the star rating. Choose an app under the cards to open its statement."))
    period = st.segmented_control("Period", ["2024 to now", "All years"], default="2024 to now",
                                  key="period", label_visibility="collapsed") or "2024 to now"
    op_all = opinions if period == "All years" else opinions[opinions["at"] >= ANALYSIS_START]
    apps = list(APPS)
    chosen = st.session_state.get("app_pick") or "EcoCash Super App"
    cards = []
    for i, a in enumerate(apps):
        op_a = op_all[op_all["app"] == a]
        cards.append(card_for(a, op_a, op_a["reviewId"].nunique(), selected=(a == chosen), seed=i))
    show(ui.wallet(cards))
    chosen = st.pills("Open a card", apps, default="EcoCash Super App", key="app_pick") or chosen

    op = op_all[op_all["app"] == chosen]
    rv_ids = op["reviewId"].unique()
    rv = reviews[reviews["reviewId"].isin(rv_ids)]
    stars = rv["score"].astype(float).mean() if len(rv) else float("nan")
    aspect_ops = op[op["aspect"] != "general"]
    show(f'<div class="mx-h2" style="margin-top:14px">{ui.esc(chosen)}</div>')
    show(ui.tiles([("average stars", f"{stars:.2f}" if stars == stars else "n/a"),
                   ("aspect opinions", f"{len(aspect_ops):,}"),
                   ("of them complaints", f"{(aspect_ops['sentiment'] == 'negative').mean():.0%}" if len(aspect_ops) else "n/a")]))
    st.write("")
    app_detail(op, reviews, key="wallet")


# Compare
elif section == "Compare":
    show(ui.heading("Compare apps side by side",
                    "The board works like an exchange-rate board: one row per aspect, one column per app. "
                    "The best app in each row has a gold outline."))
    picked = st.multiselect("Apps (up to four)", list(APPS), default=["EcoCash", "EcoCash Super App", "InnBucks", "MyZB"],
                            max_selections=4)
    if not picked:
        st.info("Choose at least one app.")
        st.stop()
    recent = opinions[opinions["at"] >= ANALYSIS_START]
    show(ui.wallet([card_for(a, recent[recent["app"] == a], recent.loc[recent["app"] == a, "reviewId"].nunique(),
                             seed=list(APPS).index(a)) for a in picked]))
    tables = {a: aspect_table(recent[recent["app"] == a]) for a in picked}
    lex = get_lexicon()
    order = recent[recent["app"].isin(picked) & (recent["aspect"] != "general")]["aspect"].value_counts().index
    rows = []
    for asp in order:
        vals = {a: (float(tables[a].loc[asp, "net"]), int(tables[a].loc[asp, "n"]))
                for a in picked if asp in tables[a].index and tables[a].loc[asp, "n"] >= MIN_MENTIONS}
        if vals:
            rows.append((lex.name(asp), vals))
    show(ui.board(rows, picked))
    show(ui.note(f"Reviews from {ANALYSIS_START[:4]} onwards. Scores run from −100 (all complaints) to +100 (all praise). "
                 f"\"too few\" means fewer than {MIN_MENTIONS} mentions."))


# Check a review
elif section == "Check a review":
    show(ui.heading("Check a review",
                    "Paste a review in English, Shona or a mix. Every aspect it mentions gets its own sentiment."))
    examples = {
        "Mixed review": "The app is easy to use and sending money is fast, but the charges are too high. "
                        "Since the update I never receive the OTP.",
        "Shona mix": "App iyi chakadhakwa, haishande kana ndaupdater. Asi interface yacho is nice.",
        "Praise": "Best banking app in Zim! Paying ZESA and buying airtime takes seconds 👍",
    }
    cols = st.columns(len(examples) + 2)
    for col, (label, text) in zip(cols, examples.items()):
        if col.button(label, type="secondary", width="stretch"):
            st.session_state["review_text"] = text
    left, right = st.columns([1, 1], gap="large")
    with left:
        text = st.text_area("Review", key="review_text", height=170,
                            placeholder="e.g. Good app but it keeps crashing after the latest update")
        st.button("Print receipt", type="primary")
    with right:
        if text.strip():
            with st.spinner("Reading the review..."):
                ops = get_analyzer().analyse(text)
            if ops:
                lex = get_lexicon()
                groups = {}
                for o in ops:
                    g = groups.setdefault(o.sentence_no, {"sentence": o.sentence, "terms": [], "lines": []})
                    g["terms"].append(o.term)
                    g["lines"].append(("General tone" if o.aspect == "general" else lex.name(o.aspect),
                                       o.sentiment, o.confidence))
                scored = [SENT_VALUE[o.sentiment] for o in ops if o.aspect != "general"] or \
                         [SENT_VALUE[o.sentiment] for o in ops]
                show(ui.receipt(list(groups.values()), sum(scored) / len(scored),
                                sum(1 for o in ops if o.aspect != "general")))
                show('<div style="height:14px"></div>' + ui.note("Underlined words triggered the aspect. The percentage is the model's confidence; "
                             "below 50% the result is printed as a lean, not a firm answer."))
            else:
                show(ui.note("No words to analyse. Try a full sentence."))
        else:
            show(ui.note("Your receipt will appear here."))


# Find an app
elif section == "Find an app":
    show(ui.heading("Find any app on the Play Store",
                    "Search by name, fetch its newest reviews from the Zimbabwe store and get its card and "
                    "statement in about a minute. Needs an internet connection."))

    @st.cache_data(ttl=3600, show_spinner=False)
    def search_store(q: str):
        from google_play_scraper import search
        res = search(q, lang="en", country="zw", n_hits=8)
        return [{"id": r["appId"], "title": r["title"]} for r in res if r.get("appId")]

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_reviews(app_id: str, n: int):
        from google_play_scraper import Sort, reviews as gp_reviews
        got, token = [], None
        while len(got) < n:
            batch, token = gp_reviews(app_id, lang="en", country="zw", sort=Sort.NEWEST,
                                      count=min(200, n - len(got)), continuation_token=token)
            if not batch:
                break
            got += batch
        # user fields are not kept
        return pd.DataFrame([{"reviewId": r["reviewId"], "content": r["content"], "score": r["score"],
                              "at": r["at"]} for r in got])

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("App name", placeholder="e.g. Steward Bank, NMB, Mukuru, FBC")
    with c2:
        n_rev = st.selectbox("Reviews to fetch", [100, 200, 400], index=1)
    if query.strip():
        try:
            with st.spinner("Searching the Play Store..."):
                hits = search_store(query.strip())
        except Exception as e:
            st.error(f"The Play Store search failed ({e}). Check the internet connection and try again.")
            hits = []
        if not hits:
            st.info("No app with a usable ID was found. Try another name. (The top search result sometimes has no ID "
                    "because of a known scraper bug.)")
        else:
            labels = [f"{h['title']}  ({h['id']})" for h in hits]
            hit = hits[labels.index(st.selectbox("Choose the app", labels))]
            if st.button(f"Fetch and analyse {n_rev} reviews", type="primary"):
                try:
                    with st.spinner("Fetching reviews (about one page per second)..."):
                        df = fetch_reviews(hit["id"], n_rev)
                except Exception as e:
                    st.error(f"Could not fetch reviews ({e}).")
                    df = pd.DataFrame()
                if df.empty:
                    st.info("This app has no written reviews in the Zimbabwe store.")
                else:
                    df["app"], df["brand"] = hit["title"], hit["title"]
                    with st.spinner("Analysing..."):
                        live = get_analyzer().analyse_many(df)
                    st.session_state["live"] = (hit["title"], live, df)
    if "live" in st.session_state:
        name, live, df = st.session_state["live"]
        show(ui.wallet([card_for(name, live, len(df), big=True, seed=99)]))
        app_detail(live, df, key="live")


# How it works
elif section == "How it works":
    show(ui.heading("How MariLens reads a review"))
    left, right = st.columns([1, 1], gap="large")
    with left:
        show(ui.steps([
            ("Split the review into sentences", "Handles phone typing: missing spaces, \"...\", line breaks."),
            ("Clean the text", "12 rules found in the data: contractions, text-speak, brand names, Shona words."),
            ("Find the aspects", "14 aspects from a 544-term English, Shona and Ndebele lexicon."),
            ("Give each aspect its own clause", "Sentences are split at but, however, asi, kodwa."),
            ("Score each clause", "TF-IDF on words and characters, then logistic regression."),
        ]))
    with right:
        items = []
        main = TABLES_DIR / "05_main_results.csv"
        if main.exists():
            m = pd.read_csv(main)
            ours = m[(m["model"] == "Our model") & (m["test set"] == "Unseen brands")]["macro_f1"]
            vader = m[(m["model"] == "VADER") & (m["test set"] == "Unseen brands")]["macro_f1"]
            if len(ours):
                items += [("review sentiment, unseen apps", f"{ours.iloc[0]:.2f}"), ("VADER on the same", f"{vader.iloc[0]:.2f}")]
        summ = TABLES_DIR / "06_summary.json"
        if summ.exists():
            s = json.load(open(summ))
            best = [r for r in s["aspect_sentiment"] if r["method"] == "our model"
                    and r["scope"] == s["aspect_sentiment_best_scope"]][0]
            items += [("aspect sentiment", f"{best['macro_f1']:.2f}"),
                      ("aspect detection", f"{s['detection']['seed + FastText expansion']['f1']:.2f}"),
                      ("stars agree with people", f"{s['star_vs_human_agreement']:.0%}"),
                      ("hand-labelled sentences", str(s["n_sentences"]))]
        if items:
            show('<div class="mx-h2">Test results (macro-F1 unless stated)</div>')
            show(ui.tiles(items).replace('class="mx-tiles"', 'class="mx-tiles" style="max-width:none;grid-template-columns:repeat(2,1fr)"'))
        show(ui.note("Trained on 3,197 reviews of EcoCash, InnBucks and CBZ Touch. Tested on held-out reviews, the "
                     "EcoCash Super App, and four brands it never saw. Full details are in the notebooks."))
    fig = ROOT / "reports" / "figures" / "07_aspect_app_heatmap.png"
    if fig.exists():
        st.write("")
        st.image(str(fig), width="stretch")
