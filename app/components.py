"""HTML components for the MariLens interface."""

from __future__ import annotations

import html
import math
import re

import numpy as np

POS, NEG, NEU = "#2a78d6", "#d63d3c", "#a3a199"

# Card colour per app
CARD_COLOURS = {
    "EcoCash": "#14532D", "InnBucks": "#0F4C5C", "CBZ Touch": "#3B3F8F", "EcoCash Super App": "#4A1D5E",
    "O'mari": "#7A4E00", "OneMoney": "#1F2933", "MyZB": "#5B2A1A", "CABS": "#2F4F3F",
}
FALLBACK_COLOUR = "#24443A"

# Logo
LOGO = """<svg width="40" height="40" viewBox="0 0 40 40" role="img" aria-label="MariLens logo" style="display:block;flex:none">
<rect width="40" height="40" rx="11" fill="#0F3D2E"/>
<path d="M20.5 8.5a11.5 11.5 0 1 1-9.3 18.3l-3.4 4.4 1.1-6.3A11.5 11.5 0 0 1 20.5 8.5z" fill="#E9B63B"/>
<path d="M14.6 24.6V15.6l5.9 6.6 5.9-6.6v9" fill="none" stroke="#0F3D2E" stroke-width="2.7"
 stroke-linecap="round" stroke-linejoin="round"/></svg>"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def mood(net: float) -> str:
    """Net sentiment as a signed score out of 100."""
    v = round(net * 100)
    return f"+{v}" if v > 0 else ("−" + str(abs(v)) if v < 0 else "0")


def _mix(a: str, b: str, t: float) -> str:
    ra = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    rb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ra, rb))


def tint(net: float) -> tuple[str, str]:
    """Background and text colour for a value on the compare board."""
    t = min(abs(net), 1.0)
    if net < -0.05:
        return _mix("#ffffff", NEG, 0.15 + 0.75 * t), ("#ffffff" if t > 0.5 else "#7a1f1e")
    if net > 0.05:
        return _mix("#ffffff", POS, 0.15 + 0.75 * t), ("#ffffff" if t > 0.5 else "#123f75")
    return "#EEF1EF", "#44554E"


def top_bar() -> str:
    return (f'<div class="mx-top">{LOGO}<div class="mx-word">MariLens'
            f'<small>What Zimbabweans say about their money apps</small></div></div>')


def heading(title: str, lead: str = "") -> str:
    return f'<div class="mx-h1">{esc(title)}</div>' + (f'<p class="mx-lead">{esc(lead)}</p>' if lead else "")


def note(text: str) -> str:
    return f'<div class="mx-note">{text}</div>'


# Bank cards
def _guilloche(seed: int, w: int = 420, h: int = 240) -> str:
    """Wavy background lines for a card."""
    rng = np.random.default_rng(seed)
    a1, a2 = rng.uniform(14, 30), rng.uniform(6, 14)
    f1, f2 = rng.uniform(1.2, 2.2), rng.uniform(3.0, 5.0)
    paths = []
    for k in range(14):
        phase = k * 0.42
        pts = []
        for i in range(0, 61):
            x = i / 60 * w
            y = (h * 0.62 + a1 * math.sin(f1 * 2 * math.pi * i / 60 + phase)
                 + a2 * math.sin(f2 * 2 * math.pi * i / 60 - phase * 1.7) - k * 4)
            pts.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#ffffff" '
                     f'stroke-opacity="{0.05 + 0.006 * k:.3f}" stroke-width="1"/>')
    ring = (f'<circle cx="{w * 0.86:.0f}" cy="{h * 0.18:.0f}" r="{h * 0.62:.0f}" fill="none" '
            f'stroke="#ffffff" stroke-opacity=".06" stroke-width="22"/>')
    return (f'<svg class="mx-guilloche" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'aria-hidden="true">{ring}{"".join(paths)}</svg>')


def bank_card(app: str, app_type: str, net: float, n_reviews: int, best: str, worst: str,
              selected: bool = False, big: bool = False, seed: int = 0) -> str:
    colour = CARD_COLOURS.get(app, FALLBACK_COLOUR)
    cls = "mx-card" + (" on" if selected else "") + (" big" if big else "")
    best_line = f"<span>▲ {esc(best)}</span>" if best else ""
    worst_line = f"<span>▼ {esc(worst)}</span>" if worst else ""
    return (f'<div class="{cls}" style="--c:{colour}">{_guilloche(seed)}'
            f'<div class="mx-row"><div class="mx-name">{esc(app)}</div><div class="mx-type">{esc(app_type)}</div></div>'
            f'<div class="mx-chip"></div>'
            f'<div class="mx-bal"><b>{mood(net)}</b><span>mood score</span></div>'
            f'<div class="mx-foot"><div class="mx-hl">{best_line}{worst_line}</div>'
            f'<div class="mx-count">{n_reviews:,}<br>reviews</div></div></div>')


def wallet(cards: list[str]) -> str:
    return f'<div class="mx-wallet" role="list">{"".join(cards)}</div>'


def tiles(items: list[tuple[str, str]]) -> str:
    return '<div class="mx-tiles">' + "".join(
        f'<div class="mx-tile"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in items) + "</div>"


# Mood statement
def _highlight(sentence: str, term: str) -> str:
    s = esc(sentence)
    if term:
        pat = re.compile(r"\b(" + r"\s+".join(re.escape(esc(w)) for w in str(term).split()) + r")", re.I)
        s = pat.sub(r"<mark>\1</mark>", s, count=1)
    return s


def statement(rows: list[dict]) -> str:
    """rows: name, net, n, neg, neu, pos, quotes[{sentence, term, sentiment, meta}]"""
    out = []
    for r in rows:
        cls = "neg" if r["net"] < -0.1 else "pos" if r["net"] > 0.1 else "mid"
        n = max(r["n"], 1)
        split = (f'<i style="width:{r["neg"] / n:.1%};background:{NEG}"></i>'
                 f'<i style="width:{r["neu"] / n:.1%};background:{NEU}"></i>'
                 f'<i style="width:{r["pos"] / n:.1%};background:{POS}"></i>')
        quotes = "".join(
            f'<div class="mx-quote" style="--q:{NEG if q["sentiment"] == "negative" else POS if q["sentiment"] == "positive" else NEU}">'
            f'{_highlight(q["sentence"], q["term"])}<small>{esc(q["meta"])}</small></div>' for q in r["quotes"])
        out.append(
            f'<details><summary><span class="mx-aname">{esc(r["name"])}</span>'
            f'<span class="mx-amt {cls}">{mood(r["net"])}</span>'
            f'<span class="mx-asub">{r["n"]} mentions: {r["neg"]} complaints, {r["pos"]} praise</span>'
            f'<span class="mx-split">{split}</span></summary>'
            f'<div class="mx-quotes">{quotes or "<div class=mx-note>No example sentences.</div>"}</div></details>')
    return f'<div class="mx-stmt">{"".join(out)}</div>'


def panel(title: str, side: str, body: str) -> str:
    return (f'<div class="mx-panel"><div class="mx-panel-head"><div class="mx-h2">{esc(title)}</div>'
            f'<span>{esc(side)}</span></div>{body}</div>')


# Latest reviews feed
def feed(items: list[dict]) -> str:
    """items: day, month, stars, text, tags[(name, sentiment)]"""
    out = []
    for it in items:
        stars = "★" * it["stars"] + f'<span class="off">{"★" * (5 - it["stars"])}</span>'
        tags = "".join(
            f'<span class="mx-tag"><i style="background:{POS if s == "positive" else NEG if s == "negative" else NEU}"></i>'
            f'{esc(n)}</span>' for n, s in it["tags"])
        out.append(f'<div><div class="mx-date">{it["day"]}<small>{esc(it["month"])}</small></div>'
                   f'<div><div class="mx-text"><span class="mx-stars" aria-label="{it["stars"]} stars">{stars}</span>'
                   f'{esc(it["text"])}</div><div class="mx-tags">{tags}</div></div></div>')
    return f'<div class="mx-feed">{"".join(out)}</div>'


# Compare board
def board(aspects: list[tuple[str, dict]], apps: list[str]) -> str:
    """aspects: (aspect name, {app: (net, n)}); the best app in each row gets a gold outline."""
    head = "<tr><th>Aspect</th>" + "".join(f"<th>{esc(a)}</th>" for a in apps) + "</tr>"
    rows = []
    for name, vals in aspects:
        best = max(vals, key=lambda a: vals[a][0]) if len(vals) > 1 else None
        cells = []
        for a in apps:
            if a not in vals:
                cells.append('<td><div class="mx-cell empty">too few</div></td>')
                continue
            net, n = vals[a]
            bg, fg = tint(net)
            cls = "mx-cell best" if a == best else "mx-cell"
            cells.append(f'<td><div class="{cls}" style="background:{bg};color:{fg}" '
                         f'title="{esc(a)}: {mood(net)} from {n} mentions">{mood(net)}<small>{n} mentions</small></div></td>')
        rows.append(f"<tr><td>{esc(name)}</td>{''.join(cells)}</tr>")
    return f'<div class="mx-panel"><table class="mx-board">{head}{"".join(rows)}</table></div>'


# Receipt
UNCERTAIN = 0.5


def receipt(groups: list[dict], overall: float, n_opinions: int) -> str:
    """groups: sentence, terms[], lines[(aspect name, sentiment, confidence)]"""
    blocks = []
    for g in groups:
        s = esc(g["sentence"])
        for t in g["terms"]:
            if t:
                pat = re.compile(r"\b(" + r"\s+".join(re.escape(esc(w)) for w in t.split()) + r")", re.I)
                s = pat.sub(r"<mark>\1</mark>", s, count=1)
        lines = []
        for name, snt, conf in g["lines"]:
            colour = POS if snt == "positive" else NEG if snt == "negative" else NEU
            label = f"leans {snt}" if conf < UNCERTAIN else snt
            lines.append(f'<div class="mx-rline"><span>{esc(name)}</span><span class="lead"></span>'
                         f'<span class="v" style="color:{colour}">{esc(label)} {conf:.0%}</span></div>')
        blocks.append(f'<div class="mx-rsent"><p>{s}</p>{"".join(lines)}</div>')
    colour = POS if overall > 0.1 else NEG if overall < -0.1 else "#44554E"
    return (f'<div class="mx-receipt"><div class="mx-rhead"><b>Review receipt</b>'
            f'<span>{n_opinions} opinion{"s" if n_opinions != 1 else ""} found</span></div>{"".join(blocks)}'
            f'<div class="mx-rtotal"><span>Overall mood</span><span style="color:{colour}">{mood(overall)}</span></div></div>')


def steps(items: list[tuple[str, str]]) -> str:
    return '<div class="mx-steps">' + "".join(
        f'<div class="mx-step"><div><b>{esc(t)}</b><span>{esc(d)}</span></div></div>' for t, d in items) + "</div>"
