"""Learned bids over 300 days, one setting at a time (plan "sweep300", run overnight 2026-11-21):
how the price of priority and carma's time lost against actuated lights depend on starting carma,
the share of urgent trips, demand, trips per driver and the bidding unit, at 1 and 2 lanes.

sweep300 skips the reference cell, which plan "price300" already ran for 300 days; its actuated
runs come from plan "fixed" (every plan shares the evaluation week's trips, so actuated lights
give identical numbers in all of them). Missing cells (runs still going) are left blank.

Writes results/analysis/params_sweep300_cells.csv and analyze/figures/update/sweep300_price
.pdf/.png, sweep300_time_lost.pdf/.png plus figures_sweep300.tex.

    python analyze/sweep300_analysis.py [workers]
"""
import glob
import os
import sys

import matplotlib
import matplotlib.ticker  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fixed_rules_analysis import OUT, WIDTH, save  # noqa: E402  (also sets the update style)
from parameter_analysis import (INK, INK2, LANE_COLORS, REFERENCE, RESULTS, TABLES,  # noqa: E402
                                cell_table, load, read_folder, settings_of)

NETWORKS = [("single", "Single intersection"), ("grid3", "3x3 grid"), ("grid5", "5x5 grid")]
LANES = (1, 2)
# (axis title, column, levels in order); the reference level of each is REFERENCE[column].
SWEEPS = [
    ("starting carma", "carma", [10, 20, 40, 80, 160, 320]),
    ("trips with low urgency (%)", "low", [20, 50, 80]),
    ("demand", "demand", ["light", "moderate", "heavy"]),
    ("trips per driver per day", "tpd", [2, 6, 10]),
    ("bidding unit", "bidding", ["vehicle", "lane", "phase"]),
]


def evaluations(workers: int) -> tuple[pd.DataFrame, list]:
    """Prices and evaluation rows of sweep300, the reference cells of price300 (1-2 lanes) and the
    fixed plan's actuated runs for the reference, without duplicate baseline days."""
    prices, evals = load("sweep300", workers)
    ref_prices, ref_evals = [], []
    for folder in sorted(glob.glob(os.path.join(RESULTS, "price300", "*_c80"))):
        s = settings_of(folder)
        if not s or s["lanes"] not in LANES:
            continue
        data = read_folder(folder)
        if len(data["prices"]):
            ref_prices.append(data["prices"].assign(**s))
        ref_evals += [{**s, **r} for r in data["eval"]]
    for folder in sorted(glob.glob(os.path.join(RESULTS, "fixed", "baselines", "*"))):
        s = settings_of(folder)
        if s and s["lanes"] in LANES:
            ref_evals += [{**s, **r} for r in read_folder(folder)["eval"]
                          if r["strategy"] == "actuated"]
    prices = pd.concat([prices, *ref_prices], ignore_index=True) if ref_prices else prices
    seen, out = set(), []
    for row in evals + ref_evals:
        key = (row["cell"], row["baseline_key"], row["strategy"], row["seed"], row["eval_day"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return prices, out


def sweep_rows(table: pd.DataFrame, column: str) -> pd.DataFrame:
    keep = pd.Series(True, index=table.index)
    for key, value in REFERENCE.items():
        if key != column and key in table:
            keep &= table[key] == value
    return table[keep]


def figure(table: pd.DataFrame, value: str, band: tuple, ylabel: str, name: str,
           zero: bool = False) -> None:
    """Rows = the five sweeps, columns = networks; a line per lane count with its 95% band (columns
    `band`). The reference level is shaded; each row shares its y scale. Without `zero` (a change
    around 0) the axis starts at 0."""
    t = table[table["strategy"] == "learn"]
    height = 1.55 * len(SWEEPS) + 0.9
    fig, axes = plt.subplots(len(SWEEPS), len(NETWORKS), figsize=(WIDTH, height), squeeze=False,
                             sharey="row", gridspec_kw={"hspace": 0.75, "wspace": 0.12})
    for i, (label, column, levels) in enumerate(SWEEPS):
        x = np.arange(len(levels))
        for j, (network, title) in enumerate(NETWORKS):
            ax = axes[i][j]
            rows = sweep_rows(t[t["network"] == network], column)
            ref = levels.index(REFERENCE[column])
            ax.axvspan(ref - 0.35, ref + 0.35, color="#f0efec", lw=0, zorder=0)
            for lanes in LANES:
                r = rows[rows["lanes"] == lanes].set_index(column).reindex(levels)
                if value not in r or r[value].notna().sum() == 0:
                    continue
                ax.plot(x, r[value].astype(float), color=LANE_COLORS[lanes],
                        lw=1.6, marker="o", ms=3.5, label=f"{lanes} lane" + ("s" if lanes > 1 else ""))
                lo, hi = band
                if lo in r and r[lo].notna().any():
                    ax.fill_between(x, r[lo].astype(float), r[hi].astype(float),
                                    color=LANE_COLORS[lanes], alpha=0.15, lw=0)
            if zero:
                ax.axhline(0, color="#8a897f", lw=0.8)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in levels], fontsize=7)
            ax.set_xlim(-0.5, len(levels) - 0.5)
            ax.tick_params(axis="both", length=0, labelsize=7)
            ax.grid(axis="y", color="#e1e0d9", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel(label, fontsize=7.5, labelpad=2)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: (f"{v:+g}%" if zero and v else f"{v:g}%").replace("-", "−")))
            if i == 0:
                ax.set_title(title, loc="left", fontweight="bold", color=INK, fontsize=8.5)
    found = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            found.setdefault(lab, h)
    top = axes[0, 0].get_position()
    fig.legend(list(found.values()), list(found.keys()), loc="lower left", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.32 / height), borderaxespad=0)
    fig.supylabel(ylabel, x=0.015, fontsize=8.5, color=INK2)
    save(fig, name)
    plt.close(fig)


def latex(seeds: int) -> None:
    text = rf"""% Learned bids over 300 days, one setting at a time (analyze/sweep300_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{sweep300_price.pdf}}
    \caption{{Price of priority after 300 learning days (median contested winning bid as a share of
    starting carma, mean of the last 30 days), changing one setting at a time from the reference
    (shaded: 80 carma, 80\% low-urgency trips, moderate demand, 6 trips per driver, lane bids).
    Mean over the network's auctioned intersections; means of {seeds} seeds, bands 95\% intervals.
    Bids are whole carma, so with 10 starting carma one unit is 10\%.}}
    \label{{fig:sweep300_price}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{sweep300_time_lost.pdf}}
    \caption{{Time lost per trip with learned bids after 300 learning days, compared with actuated
    lights on the same evaluation trips (paired by seed), one setting at a time from the reference
    (shaded). Means of {seeds} seeds, bands 95\% intervals.}}
    \label{{fig:sweep300_time_lost}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_sweep300.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "8"
    if arg == "redraw":   # figures from the saved table
        table = pd.read_csv(os.path.join(TABLES, "params_sweep300_cells.csv"))
    else:
        prices, evals = evaluations(int(arg))
        table = cell_table(prices, evals)
        table.round(4).to_csv(os.path.join(TABLES, "params_sweep300_cells.csv"), index=False)
    figure(table, "price_last_block", ("price_last_lo", "price_last_hi"),
           "price of priority, last 30 days (% of starting carma)", "sweep300_price")
    figure(table, "time_lost_vs_actuated", ("time_lost_vs_actuated_lo", "time_lost_vs_actuated_hi"),
           "change in time lost per trip vs actuated lights (%)", "sweep300_time_lost", zero=True)
    learn = table[table["strategy"] == "learn"]
    latex(int(learn["seeds"].min()) if len(learn) else 0)
    cols = ["network", "lanes", "carma", "low", "demand", "tpd", "bidding", "seeds",
            "price_last_block", "settled_day", "time_lost_vs_actuated", "time_lost_vs_actuated_lo",
            "time_lost_vs_actuated_hi", "cost_vs_actuated", "eval_contested"]
    pd.set_option("display.width", 220)
    print(learn[[c for c in cols if c in learn]].sort_values(cols[:7]).round(1).to_string(index=False))
