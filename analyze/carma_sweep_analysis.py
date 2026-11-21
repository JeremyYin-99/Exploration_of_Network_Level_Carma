"""Starting-carma sweep on the 5x5 (learning study with whole units of carma): how the bids drivers
learn, the price of priority at each intersection and travel times change when every driver
starts with 10, 20, 40, 80, 160, 320 or 640 carma.

With whole units the starting amount matters: with 10 carma a driver can only bid 0, 1, ..., 10,
so many bids are equal (ties go to the longest wait); with 640 bids are nearly continuous. If
nothing but the scale mattered, every price would be proportional to the starting carma (a flat
line in the "share of starting carma" panels).

Run from the repo root (after controller/learning_study.py ... --carma-levels ... --integer-carma
--results-name learning_carma, and the signal baselines in learning_carma/baselines):
    python analyze/carma_sweep_analysis.py
    python analyze/carma_sweep_analysis.py --results learning_carma --scenario grid5

Reads results/<results>/c<level>/<scenario>/ and results/<results>/baselines/<scenario>/.
Writes (figures as PNG and PDF in analyze/figures/learning/, tables in results/analysis/):
  carma_sweep_prices            price (median winning bid, contested auctions) against starting
                                carma, in carma and as a share of the starting amount, by junction
                                type, numbers written at every point
  carma_sweep_bids              median bid, share of zero bids, share of contested auctions decided
                                by a tie, bid as a share of the bidder's balance, share of broke
                                bidders, against starting carma
  carma_sweep_time_lost         time lost in the evaluation week against starting carma, with
                                actuated lights and stop signs, and the paired difference to
                                actuated lights at every level
  carma_sweep_by_intersection   learner A's price at every intersection and level, in carma and as a
                                share of the starting amount (numbers in the cells)
  carma_sweep_evolution         learner A's price by day at every level (share of starting carma)
  carma_sweep_node_trends       learner A's price by day at every intersection, panels laid out as
                                the map, with the drift over the last 28 days
  carma_sweep_node_trends_by_level  the same map, one line per starting carma level
  carma_sweep_block_changes     price per 30-day block and the change from the block before
  carma_sweep_node_blocks       the same per intersection (map; table carma_sweep_blocks.csv)
  carma_sweep_level_blocks      price per 30-day block for every starting level
  carma_sweep_node_drift        every intersection's drift over days 63-90 and the last 28 days,
                                with 95% intervals across chains (table carma_sweep_node_trends.csv)
  carma_sweep_node_raw          every chain's unsmoothed daily price at every intersection (map)
  carma_sweep_node_raw_by_level one page per intersection (PDF; PNGs in node_raw/): daily median
                                winning bid in carma, one panel per starting level, one line per seed
  tablescarma_sweep_summary.csv, carma_sweep_by_intersection.csv, carma_sweep_paired.csv,
         carma_sweep_convergence.csv (learner A, per chain: the price-only stopping rule of spec
         section 8c, and the network price and its drift per 28 days at day 90 and at the end)
Per level, the detailed learning-study figures come from
    python analyze/learning_analysis.py --results learning_carma/c<level> --scenarios grid5
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from learning_analysis import LAST, ROOT, TABLES, fmt, network, save, tls_order  # noqa: E402
from paired import paired, show  # noqa: E402

STRATEGIES = {"learn_br": "A: best response (learned)", "front_wait": "front-car wait bid",
              "only_lane_urgency": "lane wait bid"}
COLORS = {"learn_br": "#1f77b4", "front_wait": "#ff7f0e", "only_lane_urgency": "#9467bd"}
JUNCTION_STYLE = {"4-way": "-", "T": "--"}
# Where each line's numbers go (points above +, below -), so labels of nearby lines do not overlap.
LABEL_OFFSET = {("learn_br", "4-way"): 5, ("learn_br", "T"): -11,
                ("front_wait", "4-way"): 5, ("front_wait", "T"): 5,
                ("only_lane_urgency", "4-way"): -11, ("only_lane_urgency", "T"): -11}
# Sideways shift (log2 units) of each strategy in the travel-time figure.
DODGE = {"learn_br": -0.18, "front_wait": 0.0, "only_lane_urgency": 0.18}


def spread(values: list[float], gap: float) -> list[float]:
    """Label heights near `values`, moved apart so neighbours are at least `gap` apart."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed -= (placed - np.array(values, dtype=float)[order]).mean()   # keep them centred
    out = np.empty_like(placed)
    out[order] = placed
    return list(out)


def levels_found(results: str, scenario: str) -> list[int]:
    """Starting-carma levels with results."""
    found = []
    for path in glob.glob(os.path.join(ROOT, "results", results, "c*", scenario)):
        match = re.search(r"/c(\d+)/", path + "/")
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def last_days(folder: str, cid: str) -> list[int]:
    """The last LAST learning days of a chain (from its finished day files)."""
    days = sorted(int(re.search(r"_day(\d+)_run\.csv$", p).group(1))
                  for p in glob.glob(os.path.join(folder, f"{cid}_day*_run.csv")))
    return days[-LAST:]


def load_level(results: str, scenario: str, level: int):
    """(bid rows, whole-day price rows) of the last learning days of every chain at one level."""
    folder = os.path.join(ROOT, "results", results, f"c{level}", scenario)
    bids, prices = [], []
    for name in STRATEGIES:
        for status in glob.glob(os.path.join(folder, f"{name}_s*_status.json")):
            cid = os.path.basename(status)[:-len("_status.json")]
            seed = int(cid.rsplit("_s", 1)[1])
            for day in last_days(folder, cid):
                b = os.path.join(folder, f"{cid}_day{day}_bids.csv.gz")
                p = os.path.join(folder, f"{cid}_day{day}_prices.csv")
                if os.path.exists(b):
                    bids.append(pd.read_csv(b).assign(strategy=name, seed=seed, day=day,
                                                      level=level))
                if os.path.exists(p):
                    t = pd.read_csv(p)
                    prices.append(t[t.scope == "day"].assign(strategy=name, seed=seed, day=day,
                                                             level=level))
    cat = (lambda parts: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
    return cat(bids), cat(prices)


def summarise(bids: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per level, strategy and junction type: bids, ties, broke bidders and the price."""
    rows = []
    for (level, name, junction), p in prices.groupby(["level", "strategy", "junction"]):
        tls = set(p["tls"])
        b = bids[(bids.level == level) & (bids.strategy == name) & bids.tls.isin(tls)]
        contested_wins = b[b["won"] & (b["n_rivals"] > 0)]
        rows.append({
            "level": level, "strategy": name, "junction": junction,
            "price": p["win_q50"].mean(), "price_share": p["win_q50"].mean() / level,
            "bid_median": b["bid"].median(), "bid_mean": b["bid"].mean(),
            "bid_share_of_balance": (b["bid"] / b["balance"].where(b["balance"] > 0)).median(),
            "share_zero_bids": (b["bid"] <= 0).mean(),
            "share_ties": (contested_wins["top_rival"] == contested_wins["winning_bid"]).mean(),
            "share_broke": (b["balance"] < 1).mean(),
            "share_contested": p["share_contested"].mean()})
    return pd.DataFrame(rows)


def evaluation(results: str, scenario: str, levels: list[int]) -> pd.DataFrame:
    """Time lost per (name, seed, evaluation day) for every level's strategies and the signal
    baselines (latest evaluation block of each chain)."""
    rows = []
    folders = [(f"c{level}", level) for level in levels] + [("baselines", None)]
    for sub, level in folders:
        folder = os.path.join(ROOT, "results", results, sub, scenario)
        for path in glob.glob(os.path.join(folder, "*_after*_eval*_cars.csv")):
            match = re.search(r"/(\w+)_s(\d+)_after(\d+)_eval(\d+)_cars\.csv$", path)
            if not match:
                continue
            name = match.group(1)
            label = name if level is None else f"{name}@{level}"
            rows.append({"name": label, "strategy": name, "level": level,
                         "seed": int(match.group(2)), "after": int(match.group(3)),
                         "eval_day": int(match.group(4)),
                         "time_lost": pd.read_csv(path)["time_loss"].mean()})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    latest = table.groupby(["name", "seed"])["after"].transform("max")
    return table[table["after"] == latest]


def prices_figure(summary: pd.DataFrame) -> None:
    """Price against starting carma, in carma (log-log) and as a share of the starting amount."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in STRATEGIES:
        for junction, style in JUNCTION_STYLE.items():
            s = summary[(summary.strategy == name) & (summary.junction == junction)]
            s = s.sort_values("level")
            if s.empty:
                continue
            label = f"{STRATEGIES[name]}, {junction}"
            for ax, column, scale in ((axes[0], "price", 1), (axes[1], "price_share", 100)):
                shown = s[s[column] * scale >= 0.1] if ax is axes[0] else s   # log axis: > 0
                if shown.empty:
                    continue
                ax.plot(shown["level"], shown[column] * scale, ls=style, marker="o",
                        color=COLORS[name], label=label)
                for _, r in shown.iterrows():
                    if r[column] * scale < 0.05:
                        continue                        # a line at 0 needs no number per point
                    ax.annotate(fmt(r[column] * scale), (r["level"], r[column] * scale),
                                xytext=(0, LABEL_OFFSET[(name, junction)]),
                                textcoords="offset points", ha="center", fontsize=7,
                                color=COLORS[name])
    levels = sorted(summary["level"].unique())
    axes[0].plot(levels, levels, color="0.6", lw=0.8, ls=":", label="= starting carma")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    positive = summary["price"][summary["price"] > 0]
    if len(positive):
        axes[0].set_ylim(positive.min() * 0.6, max(levels) * 1.3)
    axes[0].set_ylabel("price (carma, log scale)")
    axes[0].set_title("price in carma (T-junction price of the wait bids is 0: not on a log axis)",
                      fontsize=9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("price (% of starting carma)")
    axes[1].set_title("price as a share of the starting amount (flat = only the scale changes)",
                      fontsize=10)
    for ax in axes:
        ax.set_xticks(levels, [str(v) for v in levels])
        ax.set_xlabel("starting carma per driver (carma, log scale)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Price of priority (median winning bid in contested auctions, last {LAST} "
                 "learning days, mean of 3 seeds) against starting carma", fontsize=10)
    fig.tight_layout()
    save(fig, "carma_sweep_prices")


def bids_figure(summary: pd.DataFrame) -> None:
    """Bids, zero bids, ties, bid as share of balance and broke bidders against starting carma."""
    panels = [("bid_median", "median bid (carma)", 1, False),
              ("share_zero_bids", "bids of 0 (%)", 100, True),
              ("share_ties", "contested auctions decided by a tie (%)", 100, True),
              ("bid_share_of_balance", "bid as % of own balance (median)", 100, True),
              ("share_broke", "bidders with 0 carma (%)", 100, True)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    levels = sorted(summary["level"].unique())
    for ax, (column, title, scale, percent) in zip(axes, panels):
        for name in STRATEGIES:
            for junction, style in JUNCTION_STYLE.items():
                s = summary[(summary.strategy == name) & (summary.junction == junction)]
                s = s.sort_values("level")
                if s.empty:
                    continue
                ax.plot(s["level"], s[column] * scale, ls=style, marker="o", ms=3,
                        color=COLORS[name], label=f"{STRATEGIES[name]}, {junction}")
                for _, r in s.iterrows():
                    value = r[column] * scale
                    if value < 0.5:
                        continue                        # lines at 0: no number per point
                    ax.annotate(f"{value:.0f}" if percent else fmt(value), (r["level"], value),
                                xytext=(0, LABEL_OFFSET[(name, junction)] * 0.8),
                                textcoords="offset points", ha="center", fontsize=6,
                                color=COLORS[name])
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, [str(v) for v in levels], fontsize=7)
        ax.set_xlabel("starting carma per driver (carma, log scale)")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"What drivers bid against starting carma (last {LAST} learning days, all seeds; "
                 "solid 4-way, dashed T-junctions)", fontsize=10)
    fig.tight_layout()
    save(fig, "carma_sweep_bids")


def time_lost_figure(table: pd.DataFrame) -> pd.DataFrame:
    """Evaluation-week time lost against starting carma, with the signal baselines, and the
    paired difference to actuated lights."""
    if table.empty:
        return pd.DataFrame()
    pairs = paired(table, "time_lost", "actuated_lights", units=("seed", "eval_day"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for name in STRATEGIES:
        s = table[table.strategy == name].groupby("level")["time_lost"].mean()
        if s.empty:
            continue
        # Strategies side by side at each level, numbers written upwards above each point.
        shift = 2.0 ** DODGE[name]
        axes[0].plot(s.index * shift, s.values, marker="o", ms=4, color=COLORS[name],
                     label=STRATEGIES[name])
        for level, v in s.items():
            axes[0].annotate(f"{v:.1f}", (level * shift, v), xytext=(0, 5),
                             textcoords="offset points", ha="center", va="bottom", rotation=90,
                             fontsize=7, color=COLORS[name])
        p = pairs[pairs["name"].str.startswith(f"{name}@")].copy()
        if len(p):
            p["level"] = p["name"].str.split("@").str[1].astype(int)
            p = p.sort_values("level")
            axes[1].errorbar(p["level"] * shift, p["mean_diff"],
                             yerr=[p["mean_diff"] - p["ci_low"], p["ci_high"] - p["mean_diff"]],
                             marker="o", ms=4, capsize=2, color=COLORS[name],
                             label=STRATEGIES[name])
            for _, r in p.iterrows():
                axes[1].annotate(f"{r['mean_diff']:+.1f}%", (r["level"] * shift, r["ci_high"]),
                                 xytext=(0, 3), textcoords="offset points", ha="center",
                                 va="bottom", rotation=90, fontsize=7, color=COLORS[name])
    for reference, color in (("actuated_lights", "k"), ("stop_signs", "0.5")):
        r = table[table.strategy == reference]["time_lost"]
        if len(r):
            axes[0].axhline(r.mean(), color=color, ls="--", lw=1,
                            label=f"{reference.replace('_', ' ')} ({r.mean():.1f} s)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylim(top=axes[1].get_ylim()[1] + 1.5)     # room for the numbers above the bars
    axes[0].set_ylim(top=axes[0].get_ylim()[1] + 0.4)
    levels = sorted(table["level"].dropna().unique().astype(int))
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels, [str(v) for v in levels])
        ax.set_xlabel("starting carma per driver (carma, log scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean time lost per trip (s)")
    axes[0].set_title("evaluation week (learning frozen), mean of 3 seeds x 7 days", fontsize=9)
    axes[1].set_ylabel("time lost vs actuated lights (%)")
    axes[1].set_title("paired by (seed, evaluation day), 95% interval", fontsize=9)
    fig.suptitle("Travel time against starting carma", fontsize=10)
    fig.tight_layout()
    save(fig, "carma_sweep_time_lost")
    return pairs


def by_intersection_figure(prices: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Learner A's price at every intersection and level, in carma and as a share of the
    starting amount, numbers in the cells."""
    a = prices[prices.strategy == "learn_br"]
    if a.empty:
        return pd.DataFrame()
    table = a.groupby(["tls", "junction", "level"])["win_q50"].mean().reset_index()
    order = tls_order(table["tls"].unique(), network(scenario))
    carma = table.pivot_table(index="tls", columns="level", values="win_q50").reindex(order)
    share = carma.div(carma.columns.to_numpy(dtype=float), axis=1) * 100
    junctions = dict(table.drop_duplicates("tls")[["tls", "junction"]].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(14, 1.5 + 0.36 * len(order)), layout="constrained")
    for ax, data, label in ((axes[0], carma, "price (carma)"),
                            (axes[1], share, "price (% of starting carma)")):
        values = data.to_numpy(dtype=float)
        shown = np.log1p(values) if ax is axes[0] else values
        image = ax.imshow(shown, aspect="auto", cmap="viridis")
        # Write the real values (the colour of the carma panel is on a log scale).
        for (i, j), v in np.ndenumerate(values):
            if not pd.isna(v):
                r, g, b, _ = image.cmap(image.norm(shown[i, j]))
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7,
                        color="black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white")
        ax.set_xticks(range(len(data.columns)), [str(c) for c in data.columns])
        ax.set_xlabel("starting carma per driver (carma)")
        ax.set_ylabel("intersection (junction type)")
        ax.set_yticks(range(len(order)), [f"{t} ({junctions.get(t, '')})" for t in order],
                      fontsize=7)
        ax.set_title(label, fontsize=10)
    fig.suptitle(f"Learner A: price at every intersection (median winning bid, last {LAST} "
                 "learning days, mean of 3 seeds)", fontsize=10)
    save(fig, "carma_sweep_by_intersection")
    return table


def evolution_figure(results: str, scenario: str, levels: list[int]) -> None:
    """Learner A's network-mean price by day at every level, as a share of the starting amount."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cmap = plt.get_cmap("viridis")
    ends = []                                            # (level, end value, colour, last day)
    for i, level in enumerate(levels):
        folder = os.path.join(ROOT, "results", results, f"c{level}", scenario)
        curves = []
        for path in glob.glob(os.path.join(folder, "learn_br_s*_prices_by_day.csv")):
            p = pd.read_csv(path)
            p = p[p["n_auctions"] >= 50]
            curves.append(p.groupby("day").apply(
                lambda g: np.average(g["win_mean"].fillna(0), weights=g["n_contested"] + 1e-9),
                include_groups=False))
        if not curves:
            continue
        mean = pd.concat(curves, axis=1).mean(axis=1) / level * 100
        color = cmap(i / max(1, len(levels) - 1))
        ax.plot(mean.index, mean.values, color=color, lw=1.2, label=f"start {level}")
        ends.append((level, mean.iloc[-LAST:].mean(), color, mean.index[-1]))
    # End labels, moved apart so they do not overlap, with a thin line to their curve.
    heights = spread([e[1] for e in ends], gap=0.45)
    for (level, end, color, day), y in zip(ends, heights):
        ax.annotate(f"start {level}: {end:.1f}%", (day, end), xytext=(day + 4, y),
                    textcoords="data", fontsize=7, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))
    ax.set_xlabel("day")
    ax.set_ylabel("network-mean price (% of starting carma)")
    ax.set_title("Learner A: price by day at every starting-carma level (mean of 3 seeds; number "
                 f"at the end: mean of the last {LAST} days)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.margins(x=0.2)
    fig.tight_layout()
    save(fig, "carma_sweep_evolution")


MAP_STYLE = {"4-way": ("#1f77b4", "4-way junction"), "T": ("#ff7f0e", "T-junction"),
             "corner": ("#bbbbbb", "corner (no auctions)")}


def drift_per_28_days(net: pd.DataFrame, first: int, last: int) -> float:
    """Least-squares slope of the network price over days first..last, in % of its mean per 28
    days."""
    window = net[(net["day"] >= first) & (net["day"] <= last)]
    if len(window) < 3:
        return np.nan
    slope = np.polyfit(window["day"], window["price"], 1)[0]
    return 100.0 * slope * 28 / window["price"].mean()


def convergence_report(results: str, scenario: str, levels: list[int]) -> pd.DataFrame:
    """Learner A per chain: the price-only stopping rule of spec section 8c (M2, M3 and M4 on 7
    consecutive days, M1 reported but not required), and the network price (mean of the
    intersections' win_rel_mean) and its drift at day 90 and at the chain's last day."""
    sys.path.insert(0, os.path.join(ROOT, "controller"))
    from pathlib import Path
    from convergence import Settings, convergence_table
    from learning_study import learn_stats, price_history
    rule = Settings(m1_max=float("inf"))
    rows = []
    for level in levels:
        folder = Path(ROOT) / "results" / results / f"c{level}" / scenario
        for status in sorted(folder.glob("learn_br_s*_status.json")):
            cid = status.name.replace("_status.json", "")
            prices = price_history(folder, cid)
            if prices.empty:
                continue
            table = convergence_table(prices, learn_stats(folder, cid), rule)
            net = (prices.groupby("day")["win_rel_mean"].mean().rename("price").reset_index())
            end = int(table["day"].max())
            converged = table.loc[table["converged"], "day"]
            rows.append({
                "level": level, "seed": int(cid.split("_s")[-1]), "last_day": end,
                "first_converged": int(converged.min()) if len(converged) else np.nan,
                "converged_at_end": bool(table["converged"].iloc[-1]),
                "share_converged_after_90": table.loc[table["day"] > 90, "converged"].mean(),
                "share_passing_last_30": table.loc[table["day"] > end - 30, "passed"].mean(),
                "price_day90": net.loc[net["day"].between(77, 90), "price"].mean(),
                "price_end": net.loc[net["day"] > end - LAST, "price"].mean(),
                "drift_days_63_90": drift_per_28_days(net, 63, 90),
                "drift_last_28": drift_per_28_days(net, end - 27, end),
                "m1_last": table.loc[table["day"] > end - LAST, "m1"].mean()})
    return pd.DataFrame(rows)


# Intersection classes of the 5x5 used in the write-up: the four T-junctions in the middle of an
# edge price like 4-ways, the other T-junctions (next to a corner) lower.
MID_EDGE = {"C0", "C4", "A2", "E2"}
COARSE_MAX = 10          # levels up to this are shown apart: whole-unit bids are too coarse there


def node_class(tls: str, junction: str) -> str:
    """4-way, T in the middle of an edge, or T next to a corner."""
    if junction == "4-way":
        return "4-way"
    return "T, middle of an edge" if tls in MID_EDGE else "T, next to a corner"


def node_history(results: str, scenario: str, levels: list[int]) -> pd.DataFrame:
    """Learner A's whole-day price at every intersection, day and chain: the median winning bid in
    contested auctions, as % of the starting carma."""
    parts = []
    for level in levels:
        folder = os.path.join(ROOT, "results", results, f"c{level}", scenario)
        for path in glob.glob(os.path.join(folder, "learn_br_s*_prices_by_day.csv")):
            seed = int(re.search(r"_s(\d+)_prices_by_day\.csv$", path).group(1))
            t = pd.read_csv(path, usecols=["day", "tls", "junction", "win_q50",
                                           "share_contested"])
            parts.append(t.assign(level=level, seed=seed, price=100.0 * t["win_q50"] / level))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def chain_drift(g: pd.DataFrame, first: int, last: int) -> float:
    """One chain's least-squares price slope over days first..last, in % of its mean per 28 days."""
    w = g[(g["day"] >= first) & (g["day"] <= last)].dropna(subset=["price"])
    if len(w) < 10 or w["price"].mean() <= 0:
        return np.nan
    return 100.0 * np.polyfit(w["day"], w["price"], 1)[0] * 28 / w["price"].mean()


def node_trends(hist: pd.DataFrame) -> pd.DataFrame:
    """Per intersection: the price in three 14-day windows and the drift over days 63-90 and over
    the last 28 days, each the mean over the chains above COARSE_MAX carma with a 95% interval
    across chains (chains are independent, so this interval is valid although days are not)."""
    from convergence import t975
    end = int(hist["day"].max())
    windows = {"start": (1, 14), "day90": (77, 90), "end": (end - 13, end)}
    drifts = {"drift_63_90": (63, 90), "drift_last28": (end - 27, end)}
    rows = []
    for tls, g in hist.groupby("tls"):
        main, coarse = g[g["level"] > COARSE_MAX], g[g["level"] <= COARSE_MAX]
        junction = g["junction"].iloc[0]
        row = {"tls": tls, "junction": junction, "class": node_class(tls, junction),
               "share_contested": main.loc[main["day"] > 90, "share_contested"].mean(),
               "chains": main.groupby(["level", "seed"]).ngroups}
        for key, (a, b) in windows.items():
            row[f"price_{key}"] = main.loc[main["day"].between(a, b), "price"].mean()
            row[f"price_{key}_coarse"] = coarse.loc[coarse["day"].between(a, b), "price"].mean()
        for key, (a, b) in drifts.items():
            per_chain = np.array([chain_drift(c, a, b)
                                  for _, c in main.groupby(["level", "seed"])])
            per_chain = per_chain[~np.isnan(per_chain)]
            half = (t975(len(per_chain) - 1) * per_chain.std(ddof=1) / np.sqrt(len(per_chain))
                    if len(per_chain) > 1 else np.nan)
            row[key] = per_chain.mean()
            row[f"{key}_lo"], row[f"{key}_hi"] = per_chain.mean() - half, per_chain.mean() + half
            row[f"{key}_coarse"] = np.nanmean([chain_drift(c, a, b)
                                               for _, c in coarse.groupby("seed")])
        rows.append(row)
    return pd.DataFrame(rows)


INK, INK2, GRIDC, REFC = "#0b0b0b", "#52514e", "#e8e7e3", "#9b9a96"
ACCENT, COARSE_C = "#2a78d6", "#8a8985"
CLASS_ORDER = ["4-way", "T, middle of an edge", "T, next to a corner"]
MAP_DPI = 900            # the two trend maps are read zoomed in
# Ordinal blue ramp for the starting levels 20..640 (light to dark, palette steps 250-700);
# 10 carma keeps the grey it has in the other figures.
LEVEL_COLORS = {20: "#86b6ef", 40: "#5598e7", 80: "#2a78d6", 160: "#1c5cab", 320: "#104281",
                640: "#0d366b"}


def panel_labels(ax, x: str, y: str, size: float = 7.5) -> None:
    """Axis labels (with units) and tick numbers on one panel, so every panel reads on its own."""
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.set_xlabel(x, fontsize=size, color=INK2, labelpad=1.5)
    ax.set_ylabel(y, fontsize=size, color=INK2, labelpad=1.5)


def map_key(tls: str) -> tuple:
    """Map order: north row first, west to east."""
    return (-int(tls[1:]), tls[0])


def node_trends_figure(hist: pd.DataFrame, trends: pd.DataFrame) -> None:
    """Learner A's price by day at every intersection, panels laid out as the 5x5 map."""
    end = int(hist["day"].max())
    main, coarse = hist[hist["level"] > COARSE_MAX], hist[hist["level"] <= COARSE_MAX]

    def roll(df):
        return df.rolling(7, center=True, min_periods=4).mean()

    mean = roll(main.groupby(["day", "tls"])["price"].mean().unstack())
    chains = roll(main.pivot_table(index="day", columns=["tls", "level", "seed"],
                                   values="price"))
    q25 = chains.T.groupby(level="tls").quantile(0.25).T
    q75 = chains.T.groupby(level="tls").quantile(0.75).T
    low = roll(coarse.groupby(["day", "tls"])["price"].mean().unstack())
    info = trends.set_index("tls")
    n_chains = int(info["chains"].max())
    ymax = 5 * np.ceil(np.nanmax(q75.values) / 5)
    marks = [(7.5, "price_start"), (83.5, "price_day90"), (end - 6.5, "price_end")]
    flat = int(((info["drift_last28_lo"] <= 0) & (info["drift_last28_hi"] >= 0)).sum())

    fig, axes = plt.subplots(5, 5, figsize=(14.5, 12.8), sharex=True, sharey=True)
    for r in range(5):
        for c in range(5):
            ax, tls = axes[r][c], f"{'ABCDE'[c]}{4 - r}"
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(REFC)
            ax.tick_params(colors=INK2, labelsize=7.5, length=2)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-3, end + 4)
            ax.set_yticks(np.arange(0, ymax + 1, 10))
            ax.set_xticks([0, 90, 180] if end >= 180 else [0, 90])
            if tls not in info.index:
                ax.set_title(f"{tls} · corner", loc="left", fontsize=8.5, color=REFC)
                ax.text(0.5, 0.5, "no auctions", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=REFC)
                ax.axis("off")
                continue
            panel_labels(ax, "day", "price (% of starting carma)")
            row = info.loc[tls]
            ax.grid(axis="y", color=GRIDC, lw=0.6)
            ax.set_axisbelow(True)
            ax.axvline(90, color=REFC, lw=0.8, ls=(0, (3, 3)))
            ax.fill_between(q25.index, q25[tls], q75[tls], color=ACCENT, alpha=0.14, lw=0)
            if tls in low:
                ax.plot(low.index, low[tls], color=COARSE_C, lw=1.0)
            ax.plot(mean.index, mean[tls], color=ACCENT, lw=1.6)
            for x, key in marks:
                ax.plot(x, row[key], "o", ms=4.5, color=ACCENT, mec="white", mew=1.0, zorder=5)
            ax.set_title(f"{tls} · {row['class']} · {row['share_contested']:.0%} contested",
                         loc="left", fontsize=8, color=INK)
            ax.text(0.02, 0.15, "  →  ".join(f"{row[key]:.1f}" for _, key in marks),
                    transform=ax.transAxes, fontsize=7.5, color=INK)
            ax.text(0.02, 0.04, f"last 28 days: {row['drift_last28']:+.0f}% "
                    f"[{row['drift_last28_lo']:+.0f}, {row['drift_last28_hi']:+.0f}]",
                    transform=ax.transAxes, fontsize=7, color=INK2)
    fig.text(0.01, 0.985, "Every intersection's price peaked in the first month, then drifted "
             f"down; over the last 28 days {flat} of {len(info)} are flat",
             fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.958, "Learner A on the 5x5, panels placed as on the map (north at top). Dashed line: day 90, the end of the first run. Price "
             "= median winning bid in contested auctions. Blue: mean of the "
             f"{n_chains} chains at 20-640 carma (7-day mean), band: their middle half; grey: 10 "
             "carma. Dots and the a → b → c line: means of days 1-14, 77-90 and "
             f"{end - 13}-{end}. Bottom line: drift over the last 28 days, % per 28 days "
             "[95% interval across chains].", fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.93), h_pad=2.2, w_pad=1.4)
    save(fig, "carma_sweep_node_trends", dpi=MAP_DPI)


SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (blue, orange, aqua)
RAW_DIR = "node_raw"


def node_raw_figure(hist: pd.DataFrame, trends: pd.DataFrame) -> None:
    """Every chain's unsmoothed daily price at every intersection, panels laid out as the map."""
    end = int(hist["day"].max())
    main = hist[hist["level"] > COARSE_MAX]
    daily = main.groupby(["day", "tls"])["price"].mean().unstack()
    info = trends.set_index("tls")
    n_chains = int(info["chains"].max())
    ymax = 40.0
    above = int((main["price"] > ymax).sum())
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(5, 5, figsize=(14.5, 12.8), sharex=True, sharey=True)
    for r in range(5):
        for c in range(5):
            ax, tls = axes[r][c], f"{'ABCDE'[c]}{4 - r}"
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(REFC)
            ax.tick_params(colors=INK2, labelsize=7.5, length=2)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-3, end + 4)
            ax.set_yticks(np.arange(0, ymax + 1, 10))
            ax.set_xticks([0, 90, 180] if end >= 180 else [0, 90])
            if tls not in info.index:
                ax.set_title(f"{tls} · corner", loc="left", fontsize=8.5, color=REFC)
                ax.text(0.5, 0.5, "no auctions", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=REFC)
                ax.axis("off")
                continue
            panel_labels(ax, "day", "daily price (% of starting carma)")
            row = info.loc[tls]
            pts = main[main["tls"] == tls]
            ax.grid(axis="y", color=GRIDC, lw=0.6)
            ax.set_axisbelow(True)
            ax.axvline(90, color=REFC, lw=0.8, ls=(0, (3, 3)))
            # small horizontal jitter so the 18 chains of one day do not sit in one column
            ax.scatter(pts["day"] + rng.uniform(-0.35, 0.35, len(pts)), pts["price"],
                       s=1.6, color=ACCENT, alpha=0.18, lw=0, rasterized=True)
            ax.plot(daily.index, daily[tls], color="#104281", lw=0.8)
            ax.set_title(f"{tls} · {row['class']}", loc="left", fontsize=8, color=INK)
            ax.text(0.98, 0.95, "  →  ".join(f"{row[k]:.1f}" for k in
                                             ("price_start", "price_day90", "price_end")),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    fig.text(0.01, 0.985, "Raw daily prices: each intersection rises for about a month, sinks "
             "until around day 150, then levels off", fontsize=12.5,
             fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.958, "Learner A on the 5x5, panels placed as on the map (north at top). Dashed line: day 90, the end of the first run. "
             f"Each dot is one chain on one day (no smoothing): the {n_chains} chains at 20-640 "
             "carma. Price = median winning bid in contested auctions that day. Dark line: mean "
             "of the chains that day. Top right: means of days 1-14 → 77-90 → "
             f"{end - 13}-{end}. Horizontal stripes: whole-carma bids at 20 and 40 carma "
             "(one carma = 5% or 2.5%)." + (f" {above} of {len(main):,} points lie above "
                                     f"{ymax:.0f}% and are not shown." if above else ""),
             fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.93), h_pad=2.2, w_pad=1.4)
    save(fig, "carma_sweep_node_raw")


def node_raw_pages(hist: pd.DataFrame) -> None:
    """One page per intersection: the median winning bid in carma by day, one panel per starting
    level, one line per seed. Saved as PNGs in figures/learning/node_raw/ and one multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    from learning_analysis import DPI, OUT
    end = int(hist["day"].max())
    folder = os.path.join(OUT, RAW_DIR)
    os.makedirs(folder, exist_ok=True)
    levels = sorted(hist["level"].unique(), reverse=True)
    seeds = sorted(hist["seed"].unique())
    order = sorted(hist["tls"].unique(), key=lambda t: (CLASS_ORDER.index(
        node_class(t, hist.loc[hist["tls"] == t, "junction"].iloc[0])), map_key(t)))
    with PdfPages(os.path.join(OUT, "carma_sweep_node_raw_by_level.pdf")) as pdf:
        for tls in order:
            g = hist[hist["tls"] == tls]
            cls = node_class(tls, g["junction"].iloc[0])
            fig, axes = plt.subplots(len(levels), 1, figsize=(11, 1.95 * len(levels) + 1.4),
                                     sharex=True)
            for ax, level in zip(axes, levels):
                lv = g[g["level"] == level]
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color(REFC)
                ax.tick_params(colors=INK2, labelsize=7.5, length=2)
                ax.grid(axis="y", color=GRIDC, lw=0.6)
                ax.set_axisbelow(True)
                ax.axvline(90, color=REFC, lw=0.8, ls=(0, (3, 3)))
                for seed, color in zip(seeds, SEED_COLORS):
                    s = lv[lv["seed"] == seed].sort_values("day")
                    ax.plot(s["day"], s["win_q50"], color=color, lw=0.8, alpha=0.9,
                            label=f"seed {seed}")
                last = lv.loc[lv["day"] > end - 14, "win_q50"].mean()
                top = lv["win_q50"].max()
                ax.set_ylim(0, max(1.0, top) * 1.12)
                ax.set_xlim(0, end + 1)
                panel_labels(ax, "day", "median winning bid\n(carma)")
                ax.set_title(f"starting carma {level}   ·   last 14 days: {last:.1f} carma "
                             f"({100 * last / level:.0f}% of start)", loc="left",
                             fontsize=8.5, color=INK)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.985),
                       ncol=len(seeds), frameon=False, fontsize=8.5)
            fig.text(0.01, 0.985, f"{tls} · {cls}: learner A's daily price in carma",
                     fontsize=12, fontweight="bold", color=INK, va="top")
            fig.text(0.01, 0.955, "Raw daily values, no smoothing. Median winning bid in "
                     "contested auctions; each panel has its own scale. Dashed line: day 90, the end "
                     "of the first run.", fontsize=8.5,
                     color=INK2, va="top")
            fig.tight_layout(rect=(0.01, 0, 1, 0.94), h_pad=1.2)
            fig.savefig(os.path.join(folder, f"node_raw_{tls}.png"), dpi=DPI)
            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)


def node_levels_figure(hist: pd.DataFrame) -> None:
    """Learner A's price by day at every intersection, one line per starting carma level (mean of
    its seeds, 7-day mean), panels laid out as the 5x5 map."""
    end = int(hist["day"].max())
    levels = sorted(hist["level"].unique())
    lines = (hist.groupby(["day", "tls", "level"])["price"].mean().unstack(["tls", "level"])
             .rolling(7, center=True, min_periods=4).mean())
    windows = [(1, 14), (77, 90), (end - 13, end)]
    main = [lv for lv in levels if lv > COARSE_MAX]

    def spread(tls, a, b):
        w = hist[(hist["tls"] == tls) & hist["day"].between(a, b) & (hist["level"] > COARSE_MAX)]
        m = w.groupby("level")["price"].mean()
        return m.max() - m.min()

    spreads = {t: [spread(t, a, b) for a, b in windows] for t in hist["tls"].unique()}
    end_spread = np.array([v[-1] for v in spreads.values()])
    w = hist[hist["day"].between(end - 13, end)].groupby(["tls", "level"])["price"].mean().unstack()
    low_everywhere = int((w[COARSE_MAX] < w[main].min(axis=1)).sum()) if COARSE_MAX in w else 0
    ymax = 5 * np.ceil(np.nanmax(lines.values) / 5)
    junction = hist.groupby("tls")["junction"].first()

    fig, axes = plt.subplots(5, 5, figsize=(14.5, 12.8), sharex=True, sharey=True)
    for r in range(5):
        for c in range(5):
            ax, tls = axes[r][c], f"{'ABCDE'[c]}{4 - r}"
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(REFC)
            ax.tick_params(colors=INK2, labelsize=7.5, length=2)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-3, end + 4)
            ax.set_yticks(np.arange(0, ymax + 1, 10))
            ax.set_xticks([0, 90, 180] if end >= 180 else [0, 90])
            if tls not in junction.index:
                ax.set_title(f"{tls} · corner", loc="left", fontsize=8.5, color=REFC)
                ax.text(0.5, 0.5, "no auctions", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=REFC)
                ax.axis("off")
                continue
            panel_labels(ax, "day", "price (% of starting carma)")
            ax.grid(axis="y", color=GRIDC, lw=0.6)
            ax.set_axisbelow(True)
            ax.axvline(90, color=REFC, lw=0.8, ls=(0, (3, 3)))
            for level in levels:
                if (tls, level) not in lines:
                    continue
                color = LEVEL_COLORS.get(level, COARSE_C)
                ax.plot(lines.index, lines[(tls, level)], color=color,
                        lw=0.9 if level <= COARSE_MAX else 1.1,
                        label=f"{level}" + (" (whole-unit bids too coarse)"
                                            if level <= COARSE_MAX else ""))
            ax.set_title(f"{tls} · {node_class(tls, junction[tls])}", loc="left", fontsize=8,
                         color=INK)
            ax.text(0.02, 0.05, "spread, 20-640: " + "  →  ".join(f"{v:.1f}"
                                                                  for v in spreads[tls]),
                    transform=ax.transAxes, fontsize=7.2, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    handles, labels = axes[2][2].get_legend_handles_labels()
    fig.legend(handles, labels, title="starting carma", title_fontsize=8.5, fontsize=8.5,
               loc="upper left", bbox_to_anchor=(0.01, 0.938), ncol=len(levels), frameon=False,
               alignment="left", handlelength=1.8, columnspacing=1.4)
    fig.text(0.01, 0.985, f"Starting carma sets the early price; by day {end} the levels from "
             f"20 to 640 sit within {end_spread.max():.1f} points everywhere", fontsize=12.5,
             fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.958, "Learner A on the 5x5, panels placed as on the map (north at top). Dashed line: day 90, the end of the first run. "
             "Each line: one starting level, mean of its 3 seeds, 7-day mean. Price = median "
             "winning bid in contested auctions. Bottom line: gap between the highest and lowest "
             f"of the 20-640 levels in days 1-14 → 77-90 → {end - 13}-{end}, in points. 10 carma "
             f"ends lowest at {low_everywhere} of {len(junction)} intersections.",
             fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.895), h_pad=2.2, w_pad=1.4)
    save(fig, "carma_sweep_node_trends_by_level", dpi=MAP_DPI)


WINDOW = 30              # days per block in the block-to-block comparison


def block_label(block: int) -> str:
    return f"{block * WINDOW + 1}-{(block + 1) * WINDOW}"


def block_means(hist: pd.DataFrame) -> pd.DataFrame:
    """Each chain's mean price in every complete 30-day block, per intersection, per junction class
    and network-wide (the mean over intersections of each day first)."""
    h = hist.assign(block=(hist["day"] - 1) // WINDOW,
                    cls=[node_class(t, j) for t, j in zip(hist["tls"], hist["junction"])])
    days = h.groupby(["level", "seed", "block"])["day"].nunique()
    full = days[days == WINDOW].index
    parts = []
    for scope, key in (("intersection", "tls"), ("class", "cls"), ("network", None)):
        cols = ["level", "seed", "block", "day"] + ([key] if key else [])
        daily = h.groupby(cols)["price"].mean().reset_index()
        m = daily.groupby([c for c in cols if c != "day"])["price"].mean().reset_index()
        m["unit"] = m[key] if key else "network"
        parts.append(m[["level", "seed", "block", "unit", "price"]].assign(scope=scope))
    out = pd.concat(parts, ignore_index=True)
    return out[out.set_index(["level", "seed", "block"]).index.isin(full)]


def block_changes(means: pd.DataFrame) -> pd.DataFrame:
    """Per scope, unit and chain group (20-640 carma, or 10 carma; also each starting level
    network-wide): the block's mean price and the change from the block before, each with a 95%
    interval across chains. Chains are independent, so the interval is valid although days are
    not; the change is paired within each chain."""
    from convergence import t975
    means = means.assign(group=np.where(means["level"] > COARSE_MAX, "20-640", "10"))
    per_level = means[means["scope"] == "network"].assign(scope="level",
                                                          unit=lambda d: d["level"].astype(str),
                                                          group=lambda d: d["level"].astype(str))
    rows = []
    for (scope, unit, group), g in pd.concat([means, per_level]).groupby(["scope", "unit",
                                                                         "group"]):
        wide = g.pivot_table(index=["level", "seed"], columns="block", values="price")
        for block in wide.columns:
            x = wide[block].dropna()
            n = len(x)
            half = t975(n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            row = {"scope": scope, "unit": unit, "group": group, "block": block,
                   "days": block_label(block), "chains": n, "price": x.mean(),
                   "price_lo": x.mean() - half, "price_hi": x.mean() + half}
            if block - 1 in wide.columns:
                d = (wide[block] - wide[block - 1]).dropna()
                hd = t975(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                row.update(change=d.mean(), change_lo=d.mean() - hd, change_hi=d.mean() + hd,
                           change_pct=100 * d.mean() / wide[block - 1].dropna().mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFC)
    ax.tick_params(colors=INK2, labelsize=7.5, length=2)
    ax.grid(axis="y", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)


def block_changes_figure(changes: pd.DataFrame) -> None:
    """Price per 30-day block (top) and its change from the block before (bottom), network-wide
    and per junction class."""
    units = ["network"] + CLASS_ORDER
    blocks = sorted(changes["block"].unique())
    labels = [block_label(b) for b in blocks]
    sub = changes[changes["scope"].isin(["network", "class"])]
    main = sub[sub["group"] == "20-640"].set_index(["unit", "block"])
    low = sub[sub["group"] == "10"].set_index(["unit", "block"])
    last = blocks[-1]
    net = main.loc[("network", last)]
    biggest = main.xs("network", level="unit")["change"].abs().idxmax()
    pmax = 5 * np.ceil(max(main["price_hi"].max(), low["price_hi"].max()) / 5)
    cmax = np.ceil(max(main["change_hi"].abs().max(), main["change_lo"].abs().max()) + 0.5)

    fig, axes = plt.subplots(2, 4, figsize=(14, 8.4), sharey="row")
    x = np.arange(len(blocks))
    for col, unit in enumerate(units):
        top, bot = axes[0][col], axes[1][col]
        _style(top)
        _style(bot)
        m = main.loc[unit].reindex(blocks)
        lo = low.loc[unit].reindex(blocks) if unit in low.index.get_level_values(0) else None
        if lo is not None:
            top.plot(x + 0.12, lo["price"], color=COARSE_C, lw=1.0, marker="o", ms=3.5,
                     label="10 carma (3 chains)")
        top.errorbar(x, m["price"], yerr=[m["price"] - m["price_lo"], m["price_hi"] - m["price"]],
                     color=ACCENT, lw=1.6, marker="o", ms=5, mec="white", mew=1, capsize=0,
                     elinewidth=1.2, label="20-640 carma (18 chains)")
        top.set_ylim(0, pmax)
        top.set_title("network (all 21 intersections)" if unit == "network" else unit,
                      loc="left", fontsize=9, color=INK)
        for xi, v in zip(x, m["price"]):
            top.text(xi, v + pmax * 0.035, f"{v:.1f}", ha="center", fontsize=7, color=INK)
        bot.axhline(0, color=REFC, lw=0.9)
        has = m["change"].notna().values
        bot.errorbar(x[has], m["change"][has],
                     yerr=[(m["change"] - m["change_lo"])[has], (m["change_hi"] - m["change"])[has]],
                     color=ACCENT, lw=0, marker="o", ms=5, mec="white", mew=1, capsize=0,
                     elinewidth=1.4)
        for xi, v in zip(x[has], m["change"][has]):
            bot.text(xi + 0.14, v, f"{v:+.1f}", va="center", fontsize=7, color=INK)
        bot.set_ylim(-cmax, cmax)
        for ax in (top, bot):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        panel_labels(top, "30-day block (days)", "mean price in the block\n(% of starting carma)")
        panel_labels(bot, "30-day block (days)",
                     "change from the block before\n(percentage points, 95% interval)")
    handles, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], lab[::-1], loc="upper left", bbox_to_anchor=(0.005, 0.9),
               ncol=2, frameon=False, fontsize=8.5)
    verdict = ("indistinguishable from zero" if net["change_lo"] <= 0 <= net["change_hi"]
               else ("still below zero" if net["change_hi"] < 0 else "above zero"))
    seq = ", ".join(f"{v:+.1f}" for v in
                    main.xs("network", level="unit")["change"].dropna().values)
    fig.text(0.01, 0.985, f"Network price change from one 30-day block to the next: {seq} points; "
             f"the last is {verdict}", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.94, "Learner A on the 5x5. Price = median winning bid in contested auctions, % "
             "of starting carma, averaged over each chain's 30 days, then across chains. Intervals "
             "are across chains (independent runs), so day-to-day correlation does not affect "
             f"them. Largest network change: block {block_label(biggest)}.", fontsize=8.5,
             color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0.0, 0.0, 1, 0.86), h_pad=1.6, w_pad=1.2)
    save(fig, "carma_sweep_block_changes", dpi=MAP_DPI)


def node_blocks_figure(changes: pd.DataFrame) -> None:
    """Every intersection's price per 30-day block, panels laid out as the 5x5 map."""
    sub = changes[changes["scope"] == "intersection"]
    main = sub[sub["group"] == "20-640"].set_index(["unit", "block"])
    low = sub[sub["group"] == "10"].set_index(["unit", "block"])
    blocks = sorted(sub["block"].unique())
    last = blocks[-1]
    x = np.arange(len(blocks))
    ymax = 5 * np.ceil(max(main["price_hi"].max(), low["price"].max()) / 5)
    tl = main.xs(last, level="block")
    moving = tl[(tl["change_lo"] > 0) | (tl["change_hi"] < 0)]
    fig, axes = plt.subplots(5, 5, figsize=(14.5, 12.8), sharex=True, sharey=True)
    for r in range(5):
        for c in range(5):
            ax, tls = axes[r][c], f"{'ABCDE'[c]}{4 - r}"
            _style(ax)
            ax.set_ylim(0, ymax)
            ax.set_yticks(np.arange(0, ymax + 1, 10))
            ax.set_xticks(x)
            ax.set_xticklabels([block_label(b).split("-")[1] for b in blocks], fontsize=7)
            if tls not in main.index.get_level_values(0):
                ax.grid(False)
                ax.set_title(f"{tls} · corner", loc="left", fontsize=8.5, color=REFC)
                ax.text(0.5, 0.5, "no auctions", transform=ax.transAxes, ha="center",
                        va="center", fontsize=8, color=REFC)
                ax.axis("off")
                continue
            panel_labels(ax, "30-day block (end day)", "30-day mean price\n(% of starting carma)")
            m = main.loc[tls].reindex(blocks)
            if tls in low.index.get_level_values(0):
                lo = low.loc[tls].reindex(blocks)
                ax.plot(x + 0.12, lo["price"], color=COARSE_C, lw=1.0, marker="o", ms=2.8)
            ax.fill_between(x, m["price_lo"], m["price_hi"], color=ACCENT, alpha=0.15, lw=0)
            ax.plot(x, m["price"], color=ACCENT, lw=1.6, marker="o", ms=4, mec="white", mew=0.8)
            cls = node_class(tls, "4-way" if tls in ("B1", "B2", "B3", "C1", "C2", "C3", "D1",
                                                     "D2", "D3") else "T")
            ax.set_title(f"{tls} · {cls}", loc="left", fontsize=8, color=INK)
            row = m.loc[last]
            ax.text(0.02, 0.04, f"last block: {row['change']:+.1f} pts "
                    f"[{row['change_lo']:+.1f}, {row['change_hi']:+.1f}]"
                    + ("  ←" if tls in moving.index else ""),
                    transform=ax.transAxes, fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    n = len(tl)
    fig.text(0.01, 0.985, f"30-day blocks: in the last block {n - len(moving)} of {n} "
             "intersections changed by an amount indistinguishable from zero",
             fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.958, "Learner A on the 5x5, panels placed as on the map (north at top). Blue: "
             "mean over the 18 chains at 20-640 carma of each chain's 30-day mean price, band: 95% "
             "interval across chains; grey: 10 carma. Bottom line: change from the block before "
             f"to block {block_label(last)}, points [95% interval]; ← marks an interval that "
             "excludes zero.", fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.93), h_pad=2.2, w_pad=1.4)
    save(fig, "carma_sweep_node_blocks", dpi=MAP_DPI)


def level_blocks_figure(changes: pd.DataFrame, hist: pd.DataFrame) -> None:
    """Price per 30-day block for every starting level, per junction class (mean over seeds)."""
    h = hist.assign(block=(hist["day"] - 1) // WINDOW,
                    cls=[node_class(t, j) for t, j in zip(hist["tls"], hist["junction"])])
    days = h.groupby(["level", "seed", "block"])["day"].nunique()
    full = set(days[days == WINDOW].index)
    h = h[[k in full for k in zip(h["level"], h["seed"], h["block"])]]
    m = (h.groupby(["cls", "level", "seed", "block", "day"])["price"].mean()
         .groupby(["cls", "level", "block"]).mean())
    blocks = sorted(h["block"].unique())
    x = np.arange(len(blocks))
    levels = sorted(h["level"].unique())
    gap = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, cls in zip(axes, CLASS_ORDER):
        _style(ax)
        for level in levels:
            y = m.loc[(cls, level)].reindex(blocks)
            ax.plot(x, y, color=LEVEL_COLORS.get(level, COARSE_C), lw=1.4 if level > COARSE_MAX
                    else 1.0, marker="o", ms=3.5, label=str(level))
        ends = [m.loc[(cls, lv, blocks[-1])] for lv in levels if lv > COARSE_MAX]
        starts = [m.loc[(cls, lv, blocks[0])] for lv in levels if lv > COARSE_MAX]
        gap[cls] = (max(starts) - min(starts), max(ends) - min(ends))
        ax.set_title(f"{cls}: gap 20-640 {gap[cls][0]:.1f} → {gap[cls][1]:.1f} pts", loc="left",
                     fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([block_label(b) for b in blocks], rotation=45, ha="right",
                           fontsize=7.5)
        ax.set_ylim(0, 5 * np.ceil(m.max() / 5))
        panel_labels(ax, "30-day block (days)", "mean price in the block\n(% of starting carma)",
                     size=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="starting carma", title_fontsize=8.5, fontsize=8.5,
               loc="upper left", bbox_to_anchor=(0.005, 0.855), ncol=len(levels), frameon=False,
               alignment="left")
    closer = all(gap[c][1] < gap[c][0] for c in CLASS_ORDER)
    lowest = all(m.loc[(c, COARSE_MAX, blocks[-1])] <
                 min(m.loc[(c, lv, blocks[-1])] for lv in levels if lv > COARSE_MAX)
                 for c in CLASS_ORDER) if COARSE_MAX in levels else False
    fig.text(0.01, 0.985, "Price per 30-day block by starting level: the 20-640 levels "
             + ("draw closer at every junction type" if closer else "do not all draw closer")
             + ("; 10 carma ends lowest" if lowest else ""),
             fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, "Learner A on the 5x5. Each line: one starting level, mean of its 3 seeds "
             "and of the intersections of that type. Panel titles: gap between the highest and "
             "lowest of the 20-640 levels, first block → last block. Grey: 10 carma, where "
             "whole-unit bids are too coarse.", fontsize=8.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.71), w_pad=1.0)
    save(fig, "carma_sweep_level_blocks", dpi=MAP_DPI)


def node_drift_figure(trends: pd.DataFrame, end: int) -> None:
    """Every intersection's price drift over days 63-90 and over the last 28 days, with 95%
    intervals across chains."""
    t = trends.copy()
    t["order"] = t["class"].map(CLASS_ORDER.index)
    t = t.sort_values(["order", "tls"], key=lambda s: s if s.name == "order"
                      else s.map(lambda v: map_key(v)))
    ys, labels, heads, y = [], [], [], 0.0
    for cls in CLASS_ORDER:
        heads.append((y, cls))
        y -= 1.0
        for tls in t.loc[t["class"] == cls, "tls"]:
            ys.append(y)
            labels.append(tls)
            y -= 1.0
        y -= 0.4
    t["y"] = ys
    falling = int((t["drift_63_90_hi"] < 0).sum())
    up = int((t["drift_last28_lo"] > 0).sum())
    down = int((t["drift_last28_hi"] < 0).sum())
    fig, ax = plt.subplots(figsize=(9, 8.6))
    ax.axvline(0, color=REFC, lw=1.0)
    ax.grid(axis="x", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    series = [("drift_63_90", 0.17, REFC, "days 63-90"),
              ("drift_last28", -0.17, ACCENT, f"last 28 days ({end - 27}-{end})")]
    for key, dy, color, label in series:
        ax.hlines(t["y"] + dy, t[f"{key}_lo"], t[f"{key}_hi"], color=color, lw=1.6)
        ax.plot(t[key], t["y"] + dy, "o", ms=6, color=color, mec="white", mew=1.2, label=label)
    for row in t.itertuples():
        ax.annotate(f"{row.drift_last28:+.1f}  [{row.drift_last28_lo:+.0f}, "
                    f"{row.drift_last28_hi:+.0f}]", (1.01, row.y - 0.17),
                    xycoords=("axes fraction", "data"), va="center", fontsize=8, color=INK)
    for yy, cls in heads:
        ax.text(-0.01, yy, cls, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9, fontweight="bold", color=INK)
    ax.text(1.01, heads[0][0], "last 28 days", transform=ax.get_yaxis_transform(),
            va="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.set_yticks(t["y"])
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("intersection, grouped by junction type", fontsize=8.5, color=INK2,
                  labelpad=118)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(REFC)
    ax.set_ylim(min(ys) - 0.8, 0.6)
    ax.set_xlabel("price drift, % per 28 days (dot: mean of the chains at 20-640 carma; "
                  "line: 95% interval across chains)", fontsize=8.5, color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK)
    fig.text(0.01, 0.985, f"Over days 63-90, {falling} of {len(t)} intersections were falling;\n"
             f"over the last 28 days {up + down} still move ({up} up, {down} down)",
             fontsize=12, fontweight="bold", color=INK, va="top", linespacing=1.4)
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.92))
    save(fig, "carma_sweep_node_drift")


def network_map_figure(scenario: str) -> None:
    """Where each intersection name sits in the grid: letter = column, number = row."""
    junctions, edges = network(scenario)
    incoming = {j: 0 for j in junctions}
    for _, to in edges:
        incoming[to] += 1
    kind = {j: {4: "4-way", 3: "T"}.get(n, "corner") for j, n in incoming.items()}
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    for a, b in {tuple(sorted(e)) for e in edges}:
        (x1, y1), (x2, y2) = junctions[a], junctions[b]
        ax.plot([x1, x2], [y1, y2], color="0.85", lw=7, solid_capstyle="round", zorder=1)
    for j, (x, y) in junctions.items():
        color, _ = MAP_STYLE[kind[j]]
        ax.text(x, y, j, ha="center", va="center", fontsize=12, fontweight="bold", zorder=3,
                color="white" if kind[j] != "corner" else "0.3",
                bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="white", lw=1.5))
    # One road length, so the scale is known.
    (x1, y1), (x2, y2) = junctions["A0"], junctions["B0"]
    ax.annotate("200 m", ((x1 + x2) / 2, y1), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=8, color="0.4")
    xs = sorted({x for x, _ in junctions.values()})
    ys = sorted({y for _, y in junctions.values()})
    ax.set_xticks(xs, [chr(ord("A") + i) for i in range(len(xs))])
    ax.set_yticks(ys, [str(i) for i in range(len(ys))])
    ax.set_xlabel("column (letter): west → east")
    ax.set_ylabel("row (number): south → north")
    ax.tick_params(length=0, labelsize=10)
    pad = 0.12 * (max(xs) - min(xs))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    counts = pd.Series(kind).value_counts()
    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, color=MAP_STYLE[k][0],
                          label=f"{MAP_STYLE[k][1]} ({counts.get(k, 0)})") for k in MAP_STYLE]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
              fontsize=8, frameon=False)
    ax.set_title(f"{scenario}: intersection names (all {len(junctions)} junctions have a light)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, f"network_map_{scenario}")


def main() -> None:
    """Every figure and table of the sweep."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--results", default="learning_carma")
    parser.add_argument("--scenario", default="grid5")
    args = parser.parse_args()
    network_map_figure(args.scenario)
    levels = levels_found(args.results, args.scenario)
    if not levels:
        print(f"No results in results/{args.results}/c*/{args.scenario}")
        return
    bids, prices = [], []
    for level in levels:
        b, p = load_level(args.results, args.scenario, level)
        bids.append(b)
        prices.append(p)
    bids = [b for b in bids if len(b)]
    prices = [p for p in prices if len(p)]
    if not bids or not prices:
        print("No finished days with bids yet.")
        return
    bids = pd.concat(bids, ignore_index=True)
    prices = pd.concat(prices, ignore_index=True)
    summary = summarise(bids, prices)
    os.makedirs(TABLES, exist_ok=True)
    table = evaluation(args.results, args.scenario, levels)
    if len(table):
        times = table.groupby(["strategy", "level"], dropna=False)["time_lost"].mean()
        summary["time_lost"] = [times.get((r.strategy, r.level), np.nan)
                                for r in summary.itertuples()]
    summary.round(4).to_csv(os.path.join(TABLES, "carma_sweep_summary.csv"), index=False)
    prices_figure(summary)
    bids_figure(summary)
    pairs = time_lost_figure(table)
    if len(pairs):
        pairs.round(4).to_csv(os.path.join(TABLES, "carma_sweep_paired.csv"), index=False)
    by_tls = by_intersection_figure(prices, args.scenario)
    if len(by_tls):
        by_tls.round(4).to_csv(os.path.join(TABLES, "carma_sweep_by_intersection.csv"),
                               index=False)
    evolution_figure(args.results, args.scenario, levels)
    conv = convergence_report(args.results, args.scenario, levels)
    if len(conv):
        conv.round(4).to_csv(os.path.join(TABLES, "carma_sweep_convergence.csv"), index=False)
        print("=== learner A: price-only stopping rule (spec 8c) and network price drift ===")
        print(conv.round(3).to_string(index=False))
        print(f"rule ever met: {conv['first_converged'].notna().sum()}/{len(conv)} chains; "
              f"holds on the last day: {conv['converged_at_end'].sum()}/{len(conv)}; "
              f"drift % per 28 days, mean over chains: days 63-90 "
              f"{conv['drift_days_63_90'].mean():+.1f}, last 28 days "
              f"{conv['drift_last_28'].mean():+.1f} (sd {conv['drift_last_28'].std():.1f})")
    hist = node_history(args.results, args.scenario, levels)
    if len(hist):
        trends = node_trends(hist)
        trends.round(4).to_csv(os.path.join(TABLES, "carma_sweep_node_trends.csv"), index=False)
        node_trends_figure(hist, trends)
        node_levels_figure(hist)
        node_drift_figure(trends, int(hist["day"].max()))
        node_raw_figure(hist, trends)
        node_raw_pages(hist)
        changes = block_changes(block_means(hist))
        changes.round(4).to_csv(os.path.join(TABLES, "carma_sweep_blocks.csv"), index=False)
        block_changes_figure(changes)
        node_blocks_figure(changes)
        level_blocks_figure(changes, hist)
        show_blocks = changes[changes["scope"].isin(["network", "class"])].round(2)
        print(f"=== learner A: {WINDOW}-day blocks (price in % of starting carma; change from the "
              "block before, 95% interval across chains) ===")
        print(show_blocks[["scope", "unit", "group", "days", "chains", "price", "change",
                           "change_lo", "change_hi"]].to_string(index=False))
    show_cols = ["level", "strategy", "junction", "price", "price_share", "bid_median",
                 "share_zero_bids", "share_ties", "bid_share_of_balance", "share_broke"]
    if "time_lost" in summary:
        show_cols.append("time_lost")
    print("=== by starting carma (last 14 learning days; amounts in carma) ===")
    print(summary[show_cols].sort_values(["strategy", "junction", "level"]).round(3)
          .to_string(index=False))
    if len(pairs):
        print("=== evaluation week: time lost vs actuated lights, paired by (seed, day) ===")
        print(show(pairs))


if __name__ == "__main__":
    main()
