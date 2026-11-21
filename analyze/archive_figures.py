"""Figures and table from the archived first `main` run (3x3 with U-turns, SUMO's default
drivers, 300 learning days, 10 seeds; results/params/archive_uturn/): how learner A's price and
the time lost develop over 300 days, and the long-run price in every sweep.
Input: results/analysis/salvage/archive_days.csv (analyze/salvage_archive.py).
Output: analyze/figures/params/archive/, results/analysis/salvage/archive_cells.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from parameter_analysis import (INK, INK2, LANE_COLORS, REFERENCE, SWEEPS, TABLES, _style,  # noqa: E402
                                interval, lanes_label, save, settings_of)

PLAN = "archive"
NOTE = "Archived run: 3x3 grid with U-turns, SUMO default drivers, 10 seeds"


def blocks() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(TABLES, "salvage", "archive_days.csv"))
    d = d[d["phase"] == "day"].copy()
    d["block"] = (d["day"] - 1) // 30 + 1
    b = d.groupby(["cell", "seed", "block"])[["price", "time_lost", "wait", "contested"]].mean().reset_index()
    s = pd.DataFrame([settings_of(c) for c in b["cell"].unique()])
    return b.merge(s, on="cell")


def by_block(b: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for (lanes, block), g in b.groupby(["lanes", "block"]):
        m, lo, hi = interval(g[value])
        rows.append({"lanes": lanes, "day": 30 * block - 15, "mean": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def reference(b: pd.DataFrame) -> pd.DataFrame:
    out = b
    for key, value in REFERENCE.items():
        if key in out:
            out = out[out[key] == value]
    return out


def fig_learning(b: pd.DataFrame) -> None:
    ref = reference(b)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, value, label in ((axes[0], "price", "price (% of starting carma)"),
                             (axes[1], "time_lost", "time lost per trip (s)")):
        t = by_block(ref, value)
        ends = []
        for lanes, g in t.groupby("lanes"):
            c = LANE_COLORS[lanes]
            ax.fill_between(g["day"], g["lo"], g["hi"], color=c, alpha=0.15, lw=0)
            ax.plot(g["day"], g["mean"], color=c, lw=2, marker="o", ms=4, label=lanes_label(lanes))
            ends.append([g["mean"].iloc[-1], lanes, g["day"].iloc[-1]])
        # End labels, pushed apart when lines finish close together.
        ax.set_ylim(bottom=0)
        gap = 0.06 * ax.get_ylim()[1]
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, lanes, x in ends:
            ax.annotate(lanes_label(lanes), (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_xlabel("learning day (30-day block means)", fontsize=9, color=INK2)
        ax.set_ylabel(label, fontsize=9, color=INK2)
        ax.set_xlim(0, 330)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Carma with learned bids over 300 days, reference cell: the price keeps falling, time lost "
                 "does not move", fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, NOTE + "; bands are 95% intervals across seeds.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, PLAN, "archive_learning_reference")


def fig_sweeps(b: pd.DataFrame) -> pd.DataFrame:
    late = b[b["block"] == 10]
    early = b[b["block"] == 2]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    axes = axes.ravel()
    for ax, (label, column, levels, fixed) in zip(axes, SWEEPS):
        for lanes in (1, 2, 3):
            for frame, style, tag in ((late, "-", "days 271-300"), (early, ":", "days 31-60")):
                keep = frame["lanes"] == lanes
                for key, value in {**REFERENCE, **fixed}.items():
                    if key != column:
                        keep &= frame[key] == value
                g = frame[keep]
                xs, ms, los, his = [], [], [], []
                for i, level in enumerate(levels):
                    m, lo, hi = interval(g[g[column] == level]["price"])
                    xs.append(i); ms.append(m); los.append(lo); his.append(hi)
                c = LANE_COLORS[lanes]
                if style == "-":
                    ax.fill_between(xs, los, his, color=c, alpha=0.12, lw=0)
                ax.plot(xs, ms, style, color=c, lw=2 if style == "-" else 1.2, marker="o" if style == "-" else None,
                        ms=4, label=f"{lanes_label(lanes)}, {tag}")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(v) for v in levels], fontsize=8)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].set_ylabel("price (% of starting carma)", fontsize=9, color=INK2)
    axes[4].set_ylabel("price (% of starting carma)", fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, frameon=False, fontsize=8, loc="center left")
    fig.suptitle("Learned price in every sweep: after 300 days (solid, 95% band) and after 60 "
                 "(dotted)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, NOTE + ".", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, PLAN, "archive_sweeps_price")


def table(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, g in b.groupby("cell"):
        row = {k: g[k].iloc[0] for k in ("network", "lanes", "demand", "tpd", "low", "mode",
                                          "bidding", "extend", "carma")}
        row["cell"] = cell
        row["seeds"] = g["seed"].nunique()
        for blk, tag in ((2, "d31_60"), (3, "d61_90"), (10, "d271_300")):
            for v in ("price", "time_lost", "contested"):
                row[f"{v}_{tag}"], row[f"{v}_{tag}_lo"], row[f"{v}_{tag}_hi"] = interval(
                    g[g["block"] == blk][v])
        rows.append(row)
    t = pd.DataFrame(rows)
    t["price_drop_60_to_300_pct"] = 100 * (t["price_d271_300"] - t["price_d31_60"]) / t["price_d31_60"]
    t.round(4).to_csv(os.path.join(TABLES, "salvage", "archive_cells.csv"), index=False)
    return t


if __name__ == "__main__":
    b = blocks()
    fig_learning(b)
    fig_sweeps(b)
    t = table(b)
    print(f"{len(t)} cells; figures in analyze/figures/params/{PLAN}/; table "
          "results/analysis/salvage/archive_cells.csv")
    print("price change from days 31-60 to 271-300, % (median, 10th-90th percentile across cells):",
          round(t["price_drop_60_to_300_pct"].median(), 1),
          tuple(np.round(t["price_drop_60_to_300_pct"].quantile([.1, .9]).values, 1)))
