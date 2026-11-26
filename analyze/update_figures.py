"""Figures of the November 2026 update doc, for LaTeX (Overleaf).

1. carma_vs_actuated: time lost per trip with learned bids (learner A) compared with actuated
   lights, for every network and lane count, reference setting, 7-day evaluation week, mean of
   10 seeds with 95% intervals. Source: results/analysis/params_quick_cells.csv (plan "quick",
   written by analyze/parameter_analysis.py --plan quick).
2. learned_price_300_days: learner A's price (median winning bid in contested auctions, % of
   starting carma) in 30-day blocks over 300 learning days, one panel per network (single
   intersection, 3x3, 5x5), one line per lane count, reference setting, mean of 10 seeds with
   95% bands, after the U-turn and driver fixes. Source: plan "price300",
   results/analysis/price300/price300_days.csv (written by analyze/price300_days.py).

Writes PDF (vector, for \\includegraphics) and PNG to analyze/figures/update/, plus
figures.tex with a figure environment and caption for each.

    python analyze/update_figures.py            # no titles inside the figures (captions in LaTeX)
    python analyze/update_figures.py --titles   # the doc's titles drawn inside the figures
"""
import argparse
import os

import matplotlib
import matplotlib.ticker  # noqa: F401
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
    ax.text(0, min(ys) - 0.9, "change in time lost per trip compared with actuated lights (%)",
            ha="center", va="top", fontsize=8.5, color=INK2)
    if titles:
        ax.set_title("Carma: 30–37% less time lost on one-lane roads, 28–94% more on two "
                     "or three lanes", loc="left", x=0.0, color=INK)
    save(fig, "carma_vs_actuated")
    return rows


# ---- 2. Learned price over 300 days (plan price300) ---------------------------------------------

def price_rows() -> pd.DataFrame:
    """30-day block means of the learned price per network and lane count (95% intervals across
    seeds)."""
    d = pd.read_csv(os.path.join(TABLES, "price300", "price300_days.csv"))
    d = d[d["phase"] == "day"].copy()
    d["block"] = (d["day"] - 1) // 30 + 1
    b = d.groupby(["cell", "seed", "block"])["price"].mean().reset_index()
    b = b.merge(pd.DataFrame([settings_of(c) for c in b["cell"].unique()]), on="cell")
    out = []
    for (network, lanes, block), g in b.groupby(["network", "lanes", "block"]):
        m, lo, hi = interval(g["price"])
        out.append({"network": network, "lanes": lanes, "day": 30 * block - 15, "block": block,
                    "mean": m, "lo": lo, "hi": hi, "seeds": g["seed"].nunique()})
    return pd.DataFrame(out)


def price_changes(rows: pd.DataFrame) -> pd.DataFrame:
    """Per network and lane count, from the block means: % change from the first to the last
    30 days, % change from the first to the second 30 days, and the mean change per 30 days over
    the last 90 days (points, and % of the last level)."""
    out = []
    for (network, lanes), g in rows.groupby(["network", "lanes"]):
        g = g.set_index("block")["mean"]
        last = g.index.max()
        per30 = (g[last] - g[last - 3]) / 3
        out.append({"network": network, "lanes": lanes, "first": g.iloc[0], "last": g.iloc[-1],
                    "drop_pct": 100 * (1 - g.iloc[-1] / g.iloc[0]),
                    "early_pct": 100 * (1 - g.iloc[1] / g.iloc[0]),
                    "late_pts_per_30d": per30, "late_pct_per_30d": 100 * per30 / g[last]})
    return pd.DataFrame(out)


def spread_labels(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so neighbours are at least `gap` apart, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ys = [values[i] for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    shift = (sum(ys) - sum(values[i] for i in order)) / len(ys)   # re-centre the group
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = ys[k] - shift
    return out


# Intersections with auctions (the grids' four corners are bends with no conflicting movements,
# so the controller skips them): what each panel's price averages, with equal weight.
JUNCTIONS = {"single": "one four-way intersection",
             "grid3": "average of 5 intersections: 1 four-way, 4 T-junctions",
             "grid5": "average of 21 intersections: 9 four-way, 12 T-junctions"}


def fig_price(titles: bool) -> pd.DataFrame:
    """One row per network (stacked, shared x axis), one line per lane count."""
    rows = price_rows()
    panels = [(n, name) for n, name in NETWORKS if n in set(rows["network"])]
    height = 2.05 * len(panels) + 0.4
    fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH, height), squeeze=False,
                             sharex=True, gridspec_kw={"hspace": 0.62})
    for ax, (network, name) in zip(axes[:, 0], panels):
        net = rows[rows["network"] == network]
        ymax = net["hi"].max() * 1.1
        ends, texts = [], []
        for lanes, g in net.groupby("lanes"):
            c = LANE_COLORS[lanes]
            ax.fill_between(g["day"], g["lo"], g["hi"], color=c, alpha=0.15, lw=0)
            ax.plot(g["day"], g["mean"], color=c, lw=1.6, marker="o", ms=3,
                    label=lane_word(int(lanes)))
            ends.append(g.iloc[-1]["mean"])
            texts.append(f"{lane_word(int(lanes))}  {g.iloc[-1]['mean']:.1f}%")
        for y, text in zip(spread_labels(ends, 0.11 * ymax), texts):
            ax.text(304, y, text, ha="left", va="center", fontsize=8, color=INK, clip_on=False)
        ax.axvspan(210, 300, color="#e1e0d9", alpha=0.35, lw=0)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("learning day (30-day means)", labelpad=2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(name, loc="left", fontsize=9, fontweight="bold", color=INK)
        ax.set_title(JUNCTIONS[network], loc="right", fontsize=7.5, color=INK2)
    axes[0, 0].text(255, axes[0, 0].get_ylim()[1], "last 90 days", ha="center", va="top",
                    fontsize=7, color=INK2)
    fig.supylabel("price of priority (% of starting carma)", x=0.055, fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top = axes[0, 0].get_position()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.2 / height),
               ncol=len(labels), frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0)
    if titles:
        ch = price_changes(rows)
        fig.suptitle(f"The learned price falls {ch.drop_pct.min():.0f}–{ch.drop_pct.max():.0f}% "
                     "over 300 days, most of it in the first 100", x=0.0, ha="left",
                     y=1 + 0.3 / height, fontsize=10, color=INK)
    save(fig, "learned_price_300_days")
    return rows


# ---- LaTeX snippet ------------------------------------------------------------------------------

def latex(head: pd.DataFrame, price: pd.DataFrame) -> None:
    one = head[head["lanes"] == 1]["time_lost_vs_actuated"]
    multi = head[head["lanes"] > 1]["time_lost_vs_actuated"]
    seeds = int(head["seeds"].min())
    ch = price_changes(price)
    lo, hi = ch["drop_pct"].min(), ch["drop_pct"].max()
    e_lo, e_hi = ch["early_pct"].min(), ch["early_pct"].max()
    late = -ch["late_pct_per_30d"].min()
    names = {"single": "single intersection", "grid3": "3x3 grid", "grid5": "5x5 grid"}
    nets = [names[n] for n, _ in NETWORKS if n in set(price["network"])]
    nets = ", ".join(nets[:-1]) + " and " + nets[-1] if len(nets) > 1 else nets[0]
    seeds300 = int(price["seeds"].min())
    text = rf"""% Figures for the update (analyze/update_figures.py). Needs \usepackage{{graphicx}}.
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{carma_vs_actuated.pdf}}
    \caption{{Carma cuts time lost per trip by {abs(one.max()):.0f}--{abs(one.min()):.0f}\% against
    actuated lights on one-lane roads, and increases it by {multi.min():.0f}--{multi.max():.0f}\% on
    two- and three-lane roads. Learned bids, reference setting (moderate demand, 6 trips per driver,
    20\% urgent trips, lane bids, 80 starting carma), 7-day evaluation week; bars are means of
    {seeds} seeds (green: less time lost than actuated lights, red: more), whiskers 95\%
    intervals across seeds.}}
    \label{{fig:carma_vs_actuated}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{learned_price_300_days.pdf}}
    \caption{{The learned price of priority falls {lo:.0f}--{hi:.0f}\% from the first to the last
    month of 300 learning days. The fall slows from {e_lo:.0f}--{e_hi:.0f}\% between the first and
    second months to at most {late:.0f}\% a month over the last 90 days; on the grids it has not
    quite stopped. Median winning bid in contested auctions as a share of starting carma, {nets};
    each grid's value is the unweighted mean over its signalised intersections (5 on the 3x3: 1
    four-way and 4 T-junctions; 21 on the 5x5: 9 four-way and 12 T-junctions; the corners are
    bends without auctions). Reference setting, 30-day means of {seeds300} seeds with 95\% bands; shaded: the last 90 days;
    labels: the last 30 days; each panel has its own y scale.}}
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
    print(price_changes(price).round(2).to_string(index=False))
    print(f"Wrote carma_vs_actuated and learned_price_300_days (.pdf/.png) and figures.tex to {OUT}")
