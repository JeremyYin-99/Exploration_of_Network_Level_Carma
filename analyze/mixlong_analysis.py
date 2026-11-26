"""The even mix of the five fixed rules over 200 days (plan "mixlong", decided 2026-11-21): does
carma keep flowing between the rules, and does the delay ranking change as balances drift?

Day by day (from each day's per-rule summary): mean balance and time lost of each rule's drivers.
Evaluation week after day 200 (the same trips as the quick and fixed plans): each rule's time lost
relative to the whole mix, its carma use, and the mix compared with actuated lights (the fixed
plan's actuated runs on the same evaluation days).

Writes results/analysis/params_mixlong_days.csv, params_mixlong_types.csv, params_mixlong_carma.csv
and analyze/figures/update/mixlong_*.pdf/.png, figures_mixlong.tex.

    python analyze/mixlong_analysis.py [workers]
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

from fixed_rules_analysis import COLOR, NAME, NETWORKS, OUT, RULES, mix_types, save  # noqa: E402
from fixed_rules_carma import SHORT, START, bars, panels, style, summarise  # noqa: E402
from fixed_rules_carma import read_folder as carma_folder  # noqa: E402
from parameter_analysis import INK, INK2, RESULTS, TABLES, interval, settings_of  # noqa: E402

PLAN = "mixlong"
WARMUP = 200
SUMMARY_RE = re.compile(r"^mix_s(\d+)_(day|eval)(\d+)_summary\.csv$")


def daily(folder: str) -> list:
    """Per seed and day (1-200 warm-up, 201-207 evaluation): each rule's mean balance at the end
    of its trips and its time lost, absolute and relative to the whole mix that day."""
    s = settings_of(folder)
    rows = []
    for name in os.listdir(folder):
        m = SUMMARY_RE.match(name)
        if not m:
            continue
        day = int(m[3]) + (WARMUP if m[2] == "eval" else 0)
        t = pd.read_csv(os.path.join(folder, name))
        whole = (t["cars"] * t["mean_time_loss"]).sum() / t["cars"].sum()
        for _, r in t.iterrows():
            rows.append({"network": s["network"], "lanes": s["lanes"], "seed": int(m[1]),
                         "day": day, "kind": r["kind"], "balance": r["mean_final_carma"],
                         "time_lost": r["mean_time_loss"],
                         "time_lost_vs_mix": 100 * (r["mean_time_loss"] / whole - 1)})
    return rows


def fig_days(days: pd.DataFrame, value: str, ylabel: str, name: str, lanes: int = 2,
             window: int = 1, ref: tuple | None = None) -> None:
    """One stacked panel per network at one lane count: each rule's daily value (mean over
    seeds, optionally a rolling mean over `window` days), each line labelled at its end."""
    fig, axes, height = panels(1.8)
    for ax, (network, title) in zip(axes, NETWORKS):
        net = days[(days["network"] == network) & (days["lanes"] == lanes)]
        mean = net.groupby(["kind", "day"])[value].mean().reset_index()
        ends = []
        for kind in RULES:
            g = mean[mean["kind"] == kind].sort_values("day")
            if g.empty:
                continue
            y = g[value].rolling(window, min_periods=1, center=True).mean()
            ax.plot(g["day"], y, color=COLOR[kind], lw=1.6)
            ends.append((y.iloc[-1], kind))
        lo, hi = ax.get_ylim()
        prev = -1e9
        for v, kind in sorted(ends):
            y = max(v, prev + 0.09 * (hi - lo))
            prev = y
            label = f"{v:+.0f}" if value == "time_lost_vs_mix" else f"{v:.0f}"
            label = "0" if label in ("-0", "+0") else label.replace("-", "\u2212")
            ax.text(WARMUP + 7 + 4, y, f"{SHORT[kind]}  {label}", ha="left", va="center",
                    fontsize=7.5, color=INK)
        if ref:
            ax.axhline(ref[0], color="#8a897f", lw=0.8, ls="--")
            ax.text(2, ref[0], ref[1], ha="left", va="bottom", fontsize=7, color=INK2)
        ax.axvline(WARMUP + 0.5, color="#c3c2b7", lw=0.8)
        ax.set_xlim(0, WARMUP + 7)
        ax.set_xticks([0, 50, 100, 150, 200])
        style(ax, f"{title}, {lanes} lanes", "day (evaluation week after day 200, right of the line)")
    fig.supylabel(ylabel, x=0.02, fontsize=9, color=INK2)
    save(fig, name)


def latex(seeds: int) -> None:
    text = rf"""% The even mix over 200 days (analyze/mixlong_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{mixlong_balance_days.pdf}}
    \caption{{The even mix of the five fixed rules over 200 days, 2 lanes per road: mean balance of
    each rule's drivers at the end of every day (every driver started with 80, dashed line); the
    evaluation week follows day 200. Means of {seeds} seeds.}}
    \label{{fig:mixlong_balance_days}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{mixlong_time_lost_days.pdf}}
    \caption{{The even mix over 200 days, 2 lanes per road: time lost per trip of each rule's
    drivers relative to the whole mix that day (10-day rolling means of {seeds} seeds).}}
    \label{{fig:mixlong_time_lost_days}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{mixlong_balances.pdf}}
    \caption{{The even mix after 200 days: mean balance of each rule's drivers at the end of the
    evaluation week, every network and lane count (dashed line: the starting 80). Means of
    {seeds} seeds, whiskers 95\% intervals.}}
    \label{{fig:mixlong_balances}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{mixlong_mix_time_lost.pdf}}
    \caption{{The even mix after 200 days, split by rule: time lost per trip of each rule's drivers
    relative to the whole mix in the evaluation week. Means of {seeds} seeds, whiskers 95\%
    intervals.}}
    \label{{fig:mixlong_mix_time_lost}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_mixlong.tex"), "w") as f:
        f.write(text)


def mix_vs_actuated() -> pd.DataFrame:
    """The whole mix's evaluation-week time lost after 200 days, against actuated lights on the
    same evaluation days (the fixed plan's actuated runs) and against the mix after 7 days."""
    def week(plan: str, folder_glob: str, strategy: str) -> pd.DataFrame:
        rows = []
        for folder in glob.glob(os.path.join(RESULTS, plan, folder_glob)):
            s = settings_of(folder)
            for path in glob.glob(os.path.join(folder, f"{strategy}_s*_eval*_cars.csv")):
                seed = int(os.path.basename(path).split("_s")[1].split("_")[0])
                rows.append({"network": s["network"], "lanes": s["lanes"], "seed": seed,
                             "time_lost": pd.read_csv(path, usecols=["time_loss"])["time_loss"].mean()})
        return pd.DataFrame(rows).groupby(["network", "lanes", "seed"])["time_lost"].mean()
    long = week(PLAN, "*_c80", "mix")
    short = week("fixed", "*_c80", "mix")
    act = week("fixed", "baselines/*", "actuated")
    out = []
    for (network, lanes), g in long.groupby(level=["network", "lanes"]):
        both = g.index.intersection(act.index)
        vs_act = interval(100 * (g.loc[both] / act.loc[both] - 1))
        both_s = g.index.intersection(short.index)
        vs_short = interval(100 * (g.loc[both_s] / short.loc[both_s] - 1))
        out.append({"network": network, "lanes": lanes, "mix_vs_actuated": vs_act[0],
                    "mix_vs_actuated_lo": vs_act[1], "mix_vs_actuated_hi": vs_act[2],
                    "day200_vs_day7": vs_short[0], "day200_vs_day7_lo": vs_short[1],
                    "day200_vs_day7_hi": vs_short[2]})
    return pd.DataFrame(out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "redraw":   # the two day-by-day figures, from the saved table
        days = pd.read_csv(os.path.join(TABLES, f"params_{PLAN}_days.csv"))
        fig_days(days, "balance", "mean balance of the rule's drivers (carma)", "mixlong_balance_days",
                 ref=(START, "start 80"))
        fig_days(days, "time_lost_vs_mix", "time lost relative to the whole mix (%, 10-day means)",
                 "mixlong_time_lost_days", window=10, ref=(0, "mix average"))
        sys.exit(0)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    folders = sorted(f for f in glob.glob(os.path.join(RESULTS, PLAN, "*_c80")) if os.path.isdir(f))
    with Pool(workers) as pool:
        days = pd.DataFrame([r for rows in pool.map(daily, folders) for r in rows])
        carma = pool.map(carma_folder, folders)
    days.to_csv(os.path.join(TABLES, f"params_{PLAN}_days.csv"), index=False)
    fig_days(days, "balance", "mean balance of the rule's drivers (carma)", "mixlong_balance_days",
             ref=(START, "start 80"))
    fig_days(days, "time_lost_vs_mix", "time lost relative to the whole mix (%, 10-day means)",
             "mixlong_time_lost_days", window=10, ref=(0, "mix average"))
    per_seed = pd.DataFrame([r for rows, _ in carma for r in rows])
    finished = per_seed.empty is False
    if finished:
        table = summarise(per_seed, ["network", "lanes", "strategy", "who"])
        table.to_csv(os.path.join(TABLES, f"params_{PLAN}_carma.csv"), index=False)
        t = table[table["who"] != "all"].copy()
        t["strategy"] = t["who"]
        bars(t, "balance", "mean balance at the end of the evaluation week (carma)",
             "mixlong_balances", keys=RULES, ref=(START, "start (80)"))
        types = mix_types(PLAN)
        types.to_csv(os.path.join(TABLES, f"params_{PLAN}_types.csv"), index=False)
        from fixed_rules_analysis import grouped_bars
        grouped_bars(types.rename(columns={"kind": "strategy"}), "time_lost_vs_mix", RULES,
                     "time lost per trip relative to the whole mix (%)", "mixlong_mix_time_lost",
                     "mix average")
        latex(int(table["seeds"].min()))
        pd.set_option("display.width", 200)
        print(mix_vs_actuated().round(1).to_string(index=False))
        print(t[["network", "lanes", "who", "bid", "paid_per_trip", "rebate_per_trip", "balance",
                 "broke"]].round(1).to_string(index=False))
        print(types[["network", "lanes", "kind", "time_lost", "time_lost_vs_mix",
                     "cost_vs_mix"]].round(1).to_string(index=False))
    last = days.groupby(["network", "lanes", "kind"])["day"].max().min()
    print(f"days so far: up to {last}")
