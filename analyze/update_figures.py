"""Figures of the November 2026 update doc, for LaTeX (Overleaf).

1. carma_vs_actuated: time lost per trip with learned bids (learner A) compared with actuated
   lights, for every network and lane count, reference setting, 7-day evaluation week, mean of
   10 seeds with 95% intervals. Source: results/analysis/params_quick_cells.csv (plan "quick",
   written by analyze/parameter_analysis.py --plan quick).
2. learned_price_300_days: learner A's price (median winning bid in contested auctions, % of
   starting carma) in 30-day blocks over 300 learning days, 3x3 grid, reference setting, one
   line per lane count, mean of 10 seeds with 95% bands. Source: the archived first run,
   results/analysis/salvage/archive_days.csv (written by analyze/salvage_archive.py).

Writes PDF (vector, for \\includegraphics) and PNG to analyze/figures/update/, plus
figures.tex with a figure environment and caption for each.

    python analyze/update_figures.py            # no titles inside the figures (captions in LaTeX)
    python analyze/update_figures.py --titles   # the doc's titles drawn inside the figures
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from parameter_analysis import (INK, INK2, LANE_COLORS, REFERENCE, TABLES,  # noqa: E402
                                interval, settings_of)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "update")
WIDTH = 6.0                       # inches: the text width of the update's LaTeX layout
GOOD, BAD = "#2f8f5b", "#c8413b"  # muted green (carma better) and red (carma worse)
NETWORKS = [("single", "Single intersection"), ("grid3", "3x3 grid"), ("grid5", "5x5 grid")]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name: str) -> None:
    """PDF for LaTeX and PNG for quick viewing."""
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    plt.close(fig)


def fmt(v: float) -> str:
    """Signed whole percent with a true minus sign: +94%, −30%."""
    return ("+" if v > 0 else "−") + f"{round(abs(v)):d}%"


def lane_word(n: int) -> str:
    return "1 lane" if n == 1 else f"{n} lanes"


# ---- 1. Carma vs actuated lights ----------------------------------------------------------------

def headline_rows() -> pd.DataFrame:
    """Learner A in each network configuration's reference cell (study drivers)."""
    t = pd.read_csv(os.path.join(TABLES, "params_quick_cells.csv"))
    keep = t["strategy"] == "learn"
    for key, value in REFERENCE.items():
        keep &= t[key] == value
    t = t[keep][["network", "lanes", "seeds", "time_lost_vs_actuated", "time_lost_vs_actuated_lo",
                 "time_lost_vs_actuated_hi"]]
    order = {n: i for i, (n, _) in enumerate(NETWORKS)}
    return t.sort_values(["network", "lanes"], key=lambda s: s.map(order) if s.name == "network"
                         else s).reset_index(drop=True)


def fig_headline(titles: bool) -> pd.DataFrame:
    rows = headline_rows()
    names = dict(NETWORKS)
    gap = 1.0                     # room for each network's header line
    ys = [-(i + (i // 3) * gap) for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    for y, (_, r) in zip(ys, rows.iterrows()):
        pct, lo, hi = r.time_lost_vs_actuated, r.time_lost_vs_actuated_lo, r.time_lost_vs_actuated_hi
        good = pct < 0
        ax.barh(y, pct, height=0.6, color=GOOD if good else BAD, alpha=0.6, linewidth=0)
        ax.plot([lo, hi], [y, y], color=INK, lw=0.8)
        ax.text(lo - 2 if good else hi + 2, y, fmt(pct), va="center",
                ha="right" if good else "left", fontsize=8.5, color=INK)
        ax.text(-58, y, lane_word(int(r.lanes)), va="center", ha="right", fontsize=8.5, color=INK2)
        if r.lanes == 1:
            ax.text(-100, y + 0.8, names[r.network], va="center", ha="left", fontsize=8.5,
                    fontweight="bold", color=INK)
    ax.axvline(0, color="#8a897f", lw=0.8)
    ax.text(0, max(ys) + 1.3, "actuated lights", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(-100, 112)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 1.8)
    ax.axis("off")
    ax.text(-100, min(ys) - 1.0, "Green: less time lost than actuated lights. Red: more. "
            "Whiskers: 95% interval across seeds.", fontsize=7.5, color=INK2, va="top")
    if titles:
        ax.set_title("Carma: 30–37% less time lost on one-lane roads, 28–94% more on two "
                     "or three lanes", loc="left", x=0.0, color=INK)
    save(fig, "carma_vs_actuated")
    return rows


# ---- 2. Learned price over 300 days -------------------------------------------------------------

def price_rows() -> pd.DataFrame:
    """30-day block means of the learned price, reference cell, by lane count (95% intervals
    across seeds)."""
    d = pd.read_csv(os.path.join(TABLES, "salvage", "archive_days.csv"))
    d = d[d["phase"] == "day"].copy()
    d["block"] = (d["day"] - 1) // 30 + 1
    b = d.groupby(["cell", "seed", "block"])["price"].mean().reset_index()
    s = pd.DataFrame([settings_of(c) for c in b["cell"].unique()])
    b = b.merge(s, on="cell")
    for key, value in REFERENCE.items():
        if key in b:
            b = b[b[key] == value]
    out = []
    for (lanes, block), g in b.groupby(["lanes", "block"]):
        m, lo, hi = interval(g["price"])
        out.append({"lanes": lanes, "day": 30 * block - 15, "mean": m, "lo": lo, "hi": hi,
                    "seeds": g["seed"].nunique()})
    return pd.DataFrame(out)


def fig_price(titles: bool) -> pd.DataFrame:
    rows = price_rows()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    for lanes, g in rows.groupby("lanes"):
        c = LANE_COLORS[lanes]
        ax.fill_between(g["day"], g["lo"], g["hi"], color=c, alpha=0.15, lw=0)
        ax.plot(g["day"], g["mean"], color=c, lw=1.8, marker="o", ms=3.5)
        first, last = g.iloc[0], g.iloc[-1]
        ax.text(first["day"] - 6, first["mean"], f"{first['mean']:.1f}%", ha="right", va="center",
                fontsize=8.5, color=INK)
        ax.text(last["day"] + 6, last["mean"], f"{lane_word(int(lanes))} {last['mean']:.1f}%",
                ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-25, 345)
    ax.set_ylim(0, 18.5)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("learning day (30-day means)")
    ax.set_yticks([])
    ax.text(-4, 0, "0%", ha="right", va="center", fontsize=8, color=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 300)
    seeds = int(rows["seeds"].min())
    note = ("Source: the first 300-day run on the 3x3 grid (made before the U-turn and driver "
            "fixes), reference setting: moderate demand, 6 trips per driver, 20% urgent trips, lane "
            "bids, 80 starting carma. Price: each day's median winning bid in contested auctions "
            "(the winner displaced a conflicting bid), as % of the carma supply per driver, averaged "
            f"over the 9 intersections; 30-day means of {seeds} seeds per lane count, bands = 95% "
            "intervals across seeds.")
    fig.text(0.0, -0.09, textwrap.fill(note, 135), fontsize=7, color=INK2, va="top")
    if titles:
        ax.set_title("The learned price is still falling after 300 days, 29–41% below the "
                     "first month", loc="left", color=INK)
    save(fig, "learned_price_300_days")
    return rows


# ---- LaTeX snippet ------------------------------------------------------------------------------

def latex(head: pd.DataFrame, price: pd.DataFrame) -> None:
    one = head[head["lanes"] == 1]["time_lost_vs_actuated"]
    multi = head[head["lanes"] > 1]["time_lost_vs_actuated"]
    seeds = int(head["seeds"].min())
    drop = []
    for _, g in price.groupby("lanes"):
        drop.append(100 * (1 - g.iloc[-1]["mean"] / g.iloc[0]["mean"]))
    text = rf"""% Figures for the update (analyze/update_figures.py). Needs \usepackage{{graphicx}}.
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{carma_vs_actuated.pdf}}
    \caption{{Carma cuts time lost per trip by {abs(one.max()):.0f}--{abs(one.min()):.0f}\% against
    actuated lights on one-lane roads, and increases it by {multi.min():.0f}--{multi.max():.0f}\% on
    two- and three-lane roads. Learned bids, reference setting (moderate demand, 6 trips per driver,
    20\% urgent trips, lane bids, 80 starting carma), 7-day evaluation week; bars are means of
    {seeds} seeds, whiskers 95\% intervals.}}
    \label{{fig:carma_vs_actuated}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{learned_price_300_days.pdf}}
    \caption{{The learned price of priority is still falling after 300 days, {min(drop):.0f}--{max(drop):.0f}\%
    below the first month. Median winning bid in contested auctions as a share of starting carma,
    3x3 grid, reference setting, 30-day means of 10 seeds with 95\% bands. From the first 300-day
    run, made before the U-turn and driver fixes.}}
    \label{{fig:learned_price}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--titles", action="store_true", help="draw the doc's titles in the figures")
    args = parser.parse_args()
    head = fig_headline(args.titles)
    price = fig_price(args.titles)
    latex(head, price)
    print(f"Wrote carma_vs_actuated.pdf/.png, learned_price_300_days.pdf/.png and figures.tex to {OUT}")
