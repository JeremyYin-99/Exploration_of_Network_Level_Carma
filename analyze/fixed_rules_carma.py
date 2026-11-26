"""Carma use of the fixed bidding rules (plan "fixed"): how much each rule bids and pays, how far
balances spread, and, in the even mix, which rules gain or lose carma.

From each chain's bid log (one row per bidding group and auction: the front car's bid, its
payment if it won a contested auction, its rebate if it lost) and driver balances:
  bid             mean bid per auction decision (carma; 80 = the starting balance)
  bid_share       mean bid as a share of the bidder's balance (%)
  zero_bids       share of decisions that bid nothing (%)
  paid_per_trip   carma paid per trip (in single-rule runs, equal to the rebates received: the
                  payments go to the losing bidders, so carma is only moved, never created)
  paid_urgent / paid_calm   carma paid per urgent / calm trip
  balance_gini    Gini of driver balances at the end of the evaluation week
  broke           share of drivers with at most 5 carma at the end (%)
Mix only, per rule: mean balance at the end of every day (14 days: 7 warm-up, 7 evaluation).

Writes results/analysis/params_fixed_carma.csv, params_fixed_carma_mix.csv and
analyze/figures/update/fixed_carma_*.pdf/.png, figures_fixed_carma.tex.

    python analyze/fixed_rules_carma.py [workers]
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
import matplotlib.ticker  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fixed_rules_analysis import (COLOR, NAME, NETWORKS, OUT, RULES, WIDTH,  # noqa: E402
                                  save)
from parameter_analysis import INK, INK2, RESULTS, TABLES, gini, interval, settings_of  # noqa: E402

PLAN = "fixed"
CARMA = RULES + ["mix"]
START = 80
FILE_RE = re.compile(r"^(\w+?)_s(\d+)_(day|eval)(\d+)_(bids\.csv\.gz|cars\.csv|drivers\.csv)$")


def read_folder(folder: str) -> tuple[list, list]:
    """Per (strategy, seed): evaluation-week carma use; and the mix's balance per rule per day."""
    s = settings_of(folder)
    files = {}
    for name in os.listdir(folder):
        m = FILE_RE.match(name)
        if m and m[1] in CARMA:
            files.setdefault((m[1], int(m[2])), []).append((m[3], int(m[4]), m[5], name))
    rows, mix_rows = [], []
    for (strategy, seed), items in files.items():
        bids, cars, last = [], [], None
        for phase, day, kind, name in items:
            path = os.path.join(folder, name)
            if kind == "drivers.csv":
                d = pd.read_csv(path, usecols=["driver", "kind", "final_carma"])
                order = day if phase == "day" else 7 + day
                if strategy == "mix":
                    for rule, g in d.groupby("kind"):
                        mix_rows.append({"network": s["network"], "lanes": s["lanes"],
                                         "seed": seed, "day": order, "kind": rule,
                                         "balance": g["final_carma"].mean()})
                if phase == "eval" and (last is None or day > last[0]):
                    last = (day, d)
            elif phase == "eval" and kind == "bids.csv.gz":
                bids.append(pd.read_csv(path, usecols=["kind", "balance", "urgency", "bid",
                                                       "payment", "rebate"]))
            elif phase == "eval" and kind == "cars.csv":
                cars.append(pd.read_csv(path, usecols=["kind", "urgency"]))
        if not bids or not cars or last is None:
            continue
        b, c, d = pd.concat(bids), pd.concat(cars), last[1]
        b["payment"] = b["payment"].clip(lower=0)
        groups = [("all", b, c, d)]
        if strategy == "mix":
            groups += [(rule, b[b["kind"] == rule], c[c["kind"] == rule], d[d["kind"] == rule])
                       for rule in RULES]
        for who, bb, cc, dd in groups:
            urgent_b, urgent_c = bb["urgency"] > 1, cc["urgency"] > 1
            share = (bb["bid"] / bb["balance"]).where(bb["balance"] > 0)
            rows.append({
                "network": s["network"], "lanes": s["lanes"], "strategy": strategy, "who": who,
                "seed": seed, "bid": bb["bid"].mean(), "bid_share": 100 * share.mean(),
                "zero_bids": 100 * (bb["bid"] == 0).mean(),
                "paid_per_trip": bb["payment"].sum() / len(cc),
                "rebate_per_trip": bb["rebate"].sum() / len(cc),
                "paid_urgent": bb.loc[urgent_b, "payment"].sum() / max(urgent_c.sum(), 1),
                "paid_calm": bb.loc[~urgent_b, "payment"].sum() / max((~urgent_c).sum(), 1),
                "urgent_ratio": ((bb.loc[urgent_b, "payment"].sum() / max(urgent_c.sum(), 1))
                                 / (bb.loc[~urgent_b, "payment"].sum() / (~urgent_c).sum())
                                 if bb.loc[~urgent_b, "payment"].sum() > 0 else np.nan),
                "balance": dd["final_carma"].mean(),
                "balance_gini": gini(dd["final_carma"]),
                "broke": 100 * (dd["final_carma"] <= 5).mean()})
    return rows, mix_rows


def summarise(per_seed: pd.DataFrame, keys: list) -> pd.DataFrame:
    cols = [c for c in per_seed.columns if c not in keys + ["seed"]]
    out = []
    for key, g in per_seed.groupby(keys):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)), seeds=g["seed"].nunique())
        for col in cols:
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col])
        out.append(row)
    return pd.DataFrame(out)


# ---- Figures ------------------------------------------------------------------------------------

def panels(height_per: float = 2.0, hspace: float = 0.55):
    height = height_per * len(NETWORKS) + 0.6
    fig, axes = plt.subplots(len(NETWORKS), 1, figsize=(WIDTH, height), squeeze=False,
                             gridspec_kw={"hspace": hspace})
    return fig, axes[:, 0], height


def style(ax, title: str, xlabel: str = "lanes per road") -> None:
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)


def legend(fig, axes, keys, height, ncol=3) -> None:
    found = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    labels = [NAME[k] for k in keys if NAME[k] in found]
    top = axes[0].get_position()
    fig.legend([found[label] for label in labels], labels, loc="lower left",
               bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.3 / height), ncol=ncol, frameon=False,
               fontsize=8, handlelength=1.2, columnspacing=1.2, borderaxespad=0)


def bars(table: pd.DataFrame, value: str, ylabel: str, name: str, keys=CARMA,
         ref: tuple | None = None, fmt=lambda v: f"{v:g}", log: bool = False) -> None:
    """Stacked panels per network; per lane count a group of bars, one per strategy. With
    `log`, dots on a log axis instead (bar lengths would misstate values on a log scale)."""
    fig, axes, height = panels()
    width = 0.8 / len(keys)
    for ax, (network, title) in zip(axes, NETWORKS):
        net = table[table["network"] == network]
        for i, key in enumerate(keys):
            r = net[net["strategy"] == key].sort_values("lanes")
            if r.empty:
                continue
            xs = r["lanes"] + (i - (len(keys) - 1) / 2) * width
            yerr = [r[value] - r[value + "_lo"], r[value + "_hi"] - r[value]]
            if log:
                ax.errorbar(xs, r[value], yerr=yerr, fmt="o", ms=5, color=COLOR[key],
                            mec=INK2, mew=0.4, ecolor=INK, elinewidth=0.6, capsize=0,
                            label=NAME[key])
            else:
                ax.bar(xs, r[value], width * 0.9, color=COLOR[key], label=NAME[key], lw=0)
                ax.errorbar(xs, r[value], yerr=yerr, fmt="none", ecolor=INK, elinewidth=0.6,
                            capsize=0)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100]))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            for x in (1.5, 2.5):
                ax.axvline(x, color="#e1e0d9", lw=0.6)
        if ref:
            ax.axhline(ref[0], color="#8a897f", lw=0.8, ls="--")
            ax.text(3.45, ref[0], ref[1], ha="left", va="center", fontsize=7.5, color=INK2)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["1 lane", "2 lanes", "3 lanes"])
        ax.set_xlim(0.5, 3.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
        style(ax, title)
    legend(fig, axes, keys, height)
    fig.supylabel(ylabel, x=0.02, fontsize=9, color=INK2)
    save(fig, name)


def urgent_calm(table: pd.DataFrame) -> None:
    """How many times more carma a strategy pays per urgent trip than per calm trip."""
    bars(table, "urgent_ratio", "carma paid per urgent trip \u00f7 per calm trip",
         "fixed_carma_urgent_calm", ref=(1, "same"), fmt=lambda v: f"{v:g}\u00d7")


SHORT = {"random": "random share", "fraction": "10% of balance", "wait": "1 per second waited",
         "uwait": "seconds × urgency", "ufrac": "50% urgent / 5% calm"}


def mix_balances(table: pd.DataFrame) -> None:
    """The even mix: mean balance of each rule's drivers at the end of the evaluation week."""
    t = table[(table["strategy"] == "mix") & (table["who"] != "all")].copy()
    t["strategy"] = t["who"]
    bars(t, "balance", "mean balance at the end of the evaluation week (carma)",
         "fixed_carma_mix_balances", keys=RULES, ref=(START, "start (80)"))


def fig_mix_days(mix: pd.DataFrame, lanes: int = 2) -> None:
    """The even mix at one lane count: mean balance of each rule's drivers at the end of every
    day, one stacked panel per network, each line labelled at its end."""
    fig, axes, height = panels(1.8)
    for ax, (network, title) in zip(axes, NETWORKS):
        net = mix[(mix["network"] == network) & (mix["lanes"] == lanes)]
        ends = []
        for kind in RULES:
            g = net[net["kind"] == kind].sort_values("day")
            if g.empty:
                continue
            ax.plot(g["day"], g["balance"], color=COLOR[kind], lw=1.8, marker="o", ms=2.5)
            ends.append((g["balance"].iloc[-1], kind))
        ymax = max(net["balance"].max() * 1.08, START * 1.3)
        ax.set_ylim(0, ymax)
        prev = -1e9
        for value, kind in sorted(ends):
            y = max(value, prev + 0.09 * ymax)
            prev = y
            ax.text(14.3, y, f"{SHORT[kind]}  {value:.0f}", ha="left", va="center", fontsize=7.5,
                    color=INK)
        ax.axhline(START, color="#8a897f", lw=0.8, ls="--")
        ax.text(0.6, START, "start 80", ha="left", va="bottom", fontsize=7, color=INK2)
        ax.axvline(7.5, color="#c3c2b7", lw=0.8)
        ax.text(7.6, ymax, "evaluation week", ha="left", va="top", fontsize=7, color=INK2)
        ax.set_xlim(0.5, 14)
        ax.set_xticks(range(1, 15))
        style(ax, f"{title}, {lanes} lanes", "day (1–7 warm-up, 8–14 evaluation)")
    fig.supylabel("mean balance of the rule's drivers (carma)", x=0.02, fontsize=9, color=INK2)
    save(fig, "fixed_carma_mix_days")


def latex(seeds: int) -> None:
    text = rf"""% Carma use of the fixed bidding rules (analyze/fixed_rules_carma.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_carma_bids.pdf}}
    \caption{{Mean bid per auction decision of each fixed rule and of the even mix, in carma
    (every driver starts with 80). Reference setting, 7-day evaluation week after 7 warm-up days;
    means of {seeds} seeds, whiskers 95\% intervals.}}
    \label{{fig:fixed_carma_bids}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_carma_paid.pdf}}
    \caption{{Carma paid per trip: what changes hands. Winners of contested auctions pay their bid
    and the payment is shared among the losing bidders, so in a single-rule run the carma paid
    equals the rebates received and the total never changes. Settings as in
    Figure~\ref{{fig:fixed_carma_bids}}.}}
    \label{{fig:fixed_carma_paid}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_carma_urgent_calm.pdf}}
    \caption{{How many times more carma each strategy pays per urgent trip than per calm trip
    (1$\times$ = the same). Settings as in Figure~\ref{{fig:fixed_carma_bids}}.}}
    \label{{fig:fixed_carma_urgent_calm}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_carma_balance_gini.pdf}}
    \caption{{How unequal carma balances are at the end of the evaluation week (Gini coefficient
    of driver balances; 0 = everyone holds the same). Settings as in
    Figure~\ref{{fig:fixed_carma_bids}}.}}
    \label{{fig:fixed_carma_balance_gini}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_carma_mix_balances.pdf}}
    \caption{{The even mix: mean balance of each rule's drivers at the end of the evaluation week
    (day 14; every driver started with 80, dashed line). Carma flows from the rules that bid a lot
    to those that bid little and collect rebates. Means of {seeds} seeds, whiskers 95\% intervals.}}
    \label{{fig:fixed_carma_mix_balances}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_carma_mix_days.pdf}}
    \caption{{The even mix with 2 lanes per road: mean balance of each rule's drivers at the end of
    every day, from the first warm-up day to the end of the evaluation week; labels give day 14.
    Means of {seeds} seeds.}}
    \label{{fig:fixed_carma_mix_days}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_fixed_carma.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    folders = sorted(f for f in glob.glob(os.path.join(RESULTS, PLAN, "*_c80")) if os.path.isdir(f))
    with Pool(workers) as pool:
        results = pool.map(read_folder, folders, chunksize=1)
    per_seed = pd.DataFrame([r for rows, _ in results for r in rows])
    mix_days = pd.DataFrame([r for _, rows in results for r in rows])
    table = summarise(per_seed, ["network", "lanes", "strategy", "who"])
    table.to_csv(os.path.join(TABLES, f"params_{PLAN}_carma.csv"), index=False)
    mix = mix_days.groupby(["network", "lanes", "kind", "day"])["balance"].mean().reset_index()
    mix.to_csv(os.path.join(TABLES, f"params_{PLAN}_carma_mix.csv"), index=False)
    whole = table[table["who"] == "all"]
    bars(whole, "bid", "mean bid per auction (carma, log scale)", "fixed_carma_bids", log=True)
    bars(whole, "paid_per_trip", "carma paid per trip (log scale)", "fixed_carma_paid", log=True)
    urgent_calm(whole)
    bars(whole, "balance_gini", "inequality of balances at the end (Gini)", "fixed_carma_balance_gini",
         fmt=lambda v: f"{v:.2f}")
    mix_balances(table)
    fig_mix_days(mix)
    latex(int(whole["seeds"].min()))
    pd.set_option("display.width", 220)
    cols = ["network", "lanes", "strategy", "bid", "bid_share", "zero_bids", "paid_per_trip",
            "paid_urgent", "paid_calm", "balance_gini", "broke"]
    print(whole[cols].round(2).to_string(index=False))
    m = table[(table["strategy"] == "mix") & (table["who"] != "all")]
    print(m[["network", "lanes", "who", "bid", "paid_per_trip", "rebate_per_trip", "balance",
             "broke"]].round(1).to_string(index=False))
