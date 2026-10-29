"""Chart style and helpers."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES_DIR

POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL = "#b4b2a9"
SENTIMENT_COLORS = {"positive": POSITIVE, "neutral": NEUTRAL, "negative": NEGATIVE}
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DIVERGING = LinearSegmentedColormap.from_list(
    "neg_neu_pos", ["#b8312f", NEGATIVE, "#f3b9b5", "#f0efec", "#b7d3f6", POSITIVE, "#184f95"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2, "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    """Save a figure to reports/figures."""
    fig.savefig(FIGURES_DIR / f"{name}.png")


def sentiment_bars(ax, table, order=("negative", "neutral", "positive"), label_min: float = 0.06):
    """Horizontal 100% stacked sentiment bars."""
    left = None
    for s in order:
        vals = table[s].values
        ax.barh(table.index, vals, left=left, color=SENTIMENT_COLORS[s], label=s,
                edgecolor=SURFACE, linewidth=1.5, height=0.7)
        for i, v in enumerate(vals):
            if v >= label_min:
                x = (0 if left is None else left[i]) + v / 2
                ax.text(x, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if s != "neutral" else INK)
        left = vals if left is None else left + vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1)
