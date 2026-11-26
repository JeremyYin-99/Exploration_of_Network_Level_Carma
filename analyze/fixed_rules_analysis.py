"""Fixed bidding rules (plan "fixed", decided 2026-11-20): each of the five fixed rules alone, and
an even split of all five ("mix"), against actuated lights, stop signs and max-pressure, in every
network configuration's reference cell: 10 seeds, 7 warm-up days, then the quick plan's 7-day
evaluation week (the same trips for every strategy).

Writes
  results/analysis/params_fixed_cells.csv      one row per network, lanes and strategy
  results/analysis/params_fixed_mix_types.csv  the mixed run split by each driver's rule
  analyze/figures/update/fixed_rules_*.pdf/.png and figures_fixed.tex (captions for LaTeX)
and checks that the wait rule and the baselines repeat the quick plan's numbers exactly.

    python analyze/fixed_rules_analysis.py [workers]
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

from parameter_analysis import (INK, INK2, RESULTS, TABLES, interval, load,  # noqa: E402
                                read_folder, settings_of, week_per_seed)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "update")
PLAN = "fixed"
WIDTH = 6.0
NETWORKS = [("single", "Single intersection"), ("grid3", "3x3 grid"), ("grid5", "5x5 grid")]
# (key, legend name, colour): the five rules in oranges, the mix in purple, baselines in greys.
STRATEGIES = [
    ("random", "random share of balance", "#f29467"),
    ("fraction", "10% of balance", "#f7c0a0"),
    ("wait", "1 carma per second waited", "#b84a1c"),
    ("uwait", "seconds waited × urgency", "#eb6834"),
    ("ufrac", "50% of balance if urgent, 5% if not", "#7a2e0e"),
    ("learn", "learned bids (60 learning days)", "#2a78d6"),
    ("mix", "even mix of the five rules", "#7b5ea7"),
    ("stop", "all-way stop signs", "#898781"),
    ("maxpressure", "max-pressure lights", "#bdbcb3"),
]
RULES = ["random", "fraction", "wait", "uwait", "ufrac"]
NAME = {k: n for k, n, _ in STRATEGIES}
COLOR = {k: c for k, _, c in STRATEGIES}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


# ---- Tables -------------------------------------------------------------------------------------

def strategy_table(week: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per network, lanes and strategy: evaluation-week means and the paired (by seed) %
    differences from actuated lights, with 95% intervals across seeds."""
    week = week.copy()
    week["network"] = [settings_of(k)["network"] for k in week["baseline_key"]]
    week["lanes"] = [settings_of(k)["lanes"] for k in week["baseline_key"]]
    week["urgent_vs_calm"] = 100 * (week["time_lost_urgent"] / week["time_lost_calm"] - 1)
    price = (prices[prices["phase"] == "eval"].groupby(["network", "lanes", "strategy"])["price"]
             .mean() if len(prices) else pd.Series(dtype=float))
    rows = []
    for (network, lanes), g in week.groupby(["network", "lanes"]):
        ref = g[g["strategy"] == "actuated"].set_index("seed")
        for strategy, s in g.groupby("strategy"):
            s = s.set_index("seed")
            row = {"network": network, "lanes": lanes, "strategy": strategy, "seeds": len(s)}
            for col in ("time_lost", "cost", "time_lost_urgent", "time_lost_calm",
                        "urgent_vs_calm", "cost_gini"):
                row[col], row[col + "_lo"], row[col + "_hi"] = interval(s[col])
            both = s.index.intersection(ref.index)
            for col in ("time_lost", "cost"):
                pct = 100 * (s.loc[both, col] - ref.loc[both, col]) / ref.loc[both, col]
                m, lo, hi = interval(pct)
                row[f"{col}_vs_actuated"], row[f"{col}_vs_actuated_lo"] = m, lo
                row[f"{col}_vs_actuated_hi"] = hi
            row["price"] = price.get((network, lanes, strategy), np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def mix_types(plan: str = PLAN) -> pd.DataFrame:
    """The mixed run split by each driver's rule: evaluation-week time lost and cost per trip of
    each rule's drivers, as means over seeds, and relative to the whole mix (paired by seed)."""
    rows = []
    for folder in sorted(glob.glob(os.path.join(RESULTS, plan, "*_c80"))):
        s = settings_of(folder)
        per_seed = {}
        for path in glob.glob(os.path.join(folder, "mix_s*_eval*_cars.csv")):
            seed = int(os.path.basename(path).split("_s")[1].split("_")[0])
            cars = pd.read_csv(path, usecols=["kind", "time_loss", "urgency"])
            per_seed.setdefault(seed, []).append(cars)
        for seed, days in per_seed.items():
            cars = pd.concat(days)
            cars["cost"] = cars["urgency"] * cars["time_loss"]
            whole = cars[["time_loss", "cost"]].mean()
            for kind, k in cars.groupby("kind"):
                rows.append({"network": s["network"], "lanes": s["lanes"], "kind": kind,
                             "seed": seed, "time_lost": k["time_loss"].mean(),
                             "cost": k["cost"].mean(),
                             "time_lost_vs_mix": 100 * (k["time_loss"].mean() / whole["time_loss"] - 1),
                             "cost_vs_mix": 100 * (k["cost"].mean() / whole["cost"] - 1)})
    per_seed = pd.DataFrame(rows)
    out = []
    for (network, lanes, kind), g in per_seed.groupby(["network", "lanes", "kind"]):
        row = {"network": network, "lanes": lanes, "kind": kind, "seeds": g["seed"].nunique()}
        for col in ("time_lost", "cost", "time_lost_vs_mix", "cost_vs_mix"):
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col])
        out.append(row)
    return pd.DataFrame(out)


def quick_learn_evals() -> list:
    """Learned bids from the quick plan's reference cells (60 learning days, then the same 7-day
    evaluation week on the same trips as the fixed plan, whose actuated runs are identical to the
    quick plan's), in the shape load() returns."""
    rows = []
    for folder in sorted(glob.glob(os.path.join(RESULTS, "quick", "*_moderate_t6_u80-20_lane_c80"))):
        s = settings_of(folder)
        for r in read_folder(folder)["eval"]:
            if r["strategy"] == "learn":
                rows.append({**s, **r})
    return rows


def check_against_quick() -> str:
    """The wait rule and the baselines ran in the quick plan on the same cells, seeds and trips:
    every evaluation day they share must be identical."""
    def days(plan: str) -> pd.Series:
        folders = sorted(glob.glob(os.path.join(RESULTS, plan, "*_moderate_t6_u80-20_lane_c80")))
        folders += sorted(glob.glob(os.path.join(RESULTS, plan, "baselines",
                                                 "*_moderate_t6_u80-20")))
        rows = []
        for folder in folders:
            key = settings_of(folder)["baseline_key"]
            for r in read_folder(folder)["eval"]:
                if r["strategy"] in ("wait", "actuated", "stop", "maxpressure"):
                    rows.append((key, r["strategy"], r["seed"], r["eval_day"], r["time_lost"]))
        frame = pd.DataFrame(rows, columns=["key", "strategy", "seed", "day", "time_lost"])
        return frame.set_index(["key", "strategy", "seed", "day"])["time_lost"]
    ours, quick = days(PLAN), days("quick")
    both = ours.index.intersection(quick.index)
    diff = (ours.loc[both] - quick.loc[both]).abs().max() if len(both) else np.nan
    return (f"{len(both)} evaluation days also in the quick plan (wait rule and baselines); "
            f"largest difference in time lost: {diff:.2e} s")


# ---- Figures ------------------------------------------------------------------------------------

def save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    plt.close(fig)


def grouped_bars(table: pd.DataFrame, value: str, keys: list, ylabel: str, name: str,
                 zero_label: str | None, lanes: tuple = (1, 2, 3)) -> None:
    """One stacked panel per network; on each, a group of bars per lane count in `lanes` (one bar
    per strategy) with 95% whiskers."""
    height = 2.1 * len(NETWORKS) + 0.6
    fig, axes = plt.subplots(len(NETWORKS), 1, figsize=(WIDTH, height), squeeze=False,
                             gridspec_kw={"hspace": 0.55})
    width = 0.8 / len(keys)
    for ax, (network, title) in zip(axes[:, 0], NETWORKS):
        net = table[table["network"] == network]
        for i, key in enumerate(keys):
            r = net[net["strategy"] == key].sort_values("lanes")
            if r.empty:
                continue
            xs = r["lanes"] + (i - (len(keys) - 1) / 2) * width
            ax.bar(xs, r[value], width * 0.9, color=COLOR[key], label=NAME[key], lw=0)
            ax.errorbar(xs, r[value], yerr=[r[value] - r[value + "_lo"],
                                            r[value + "_hi"] - r[value]],
                        fmt="none", ecolor=INK, elinewidth=0.6, capsize=0)
        ax.axhline(0, color="#8a897f", lw=0.8)
        if zero_label:
            ax.text(max(lanes) + 0.45, 0, zero_label, ha="left", va="center", fontsize=7.5,
                    color=INK2)
        ax.set_xticks(list(lanes))
        ax.set_xticklabels([f"{n} lane" + ("s" if n > 1 else "") for n in lanes])
        ax.set_xlim(min(lanes) - 0.5, max(lanes) + 0.5)
        ax.set_xlabel("lanes per road", labelpad=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:+g}%".replace("-", "\u2212") if v else "0%"))
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
    found = {}
    for ax in axes[:, 0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    labels = [NAME[k] for k in keys if NAME[k] in found]
    handles = [found[label] for label in labels]
    top = axes[0, 0].get_position()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.3 / height),
               ncol=2, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.5,
               borderaxespad=0)
    fig.supylabel(ylabel, x=0.02, fontsize=9, color=INK2)
    save(fig, name)


def fig_mix(types: pd.DataFrame) -> None:
    t = types.rename(columns={"kind": "strategy"})
    grouped_bars(t, "time_lost_vs_mix", RULES, "time lost per trip relative to the whole mix (%)",
                 "fixed_rules_mix", "mix average")


# ---- LaTeX --------------------------------------------------------------------------------------

def latex(table: pd.DataFrame, seeds: int) -> None:
    text = rf"""% Figures for the fixed bidding rules (analyze/fixed_rules_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_rules_time_lost.pdf}}
    \caption{{Time lost per trip under each fixed bidding rule, learned bids, an even mix of the
    five rules, stop signs and max-pressure lights, compared with actuated lights (0\%) on the
    same trips. Reference setting (moderate demand, 6 trips per driver, 20\% urgent trips, lane
    bids, 80 starting carma); fixed rules after 7 warm-up days and learned bids after 60
    learning days, then the same 7-day evaluation week; bars are means of {seeds} seeds of the
    paired difference, whiskers 95\% intervals; each panel has its own scale.}}
    \label{{fig:fixed_rules_time_lost}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_rules_cost.pdf}}
    \caption{{Urgency-weighted cost per trip (time lost $\times$ urgency, urgent trips weigh 6
    times calm ones), compared with actuated lights; settings as in
    Figure~\ref{{fig:fixed_rules_time_lost}}.}}
    \label{{fig:fixed_rules_cost}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_rules_urgency.pdf}}
    \caption{{How much less (negative) or more time urgent trips lose than calm ones under each
    rule; actuated lights, stop signs and max-pressure ignore urgency. Settings as in
    Figure~\ref{{fig:fixed_rules_time_lost}}.}}
    \label{{fig:fixed_rules_urgency}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed_rules_mix.pdf}}
    \caption{{The even mix split by rule: time lost per trip of each rule's drivers relative to the
    whole mix, when every driver keeps one of the five rules (a fifth of the drivers each) and
    the car at the head of each lane bids by its own rule. Settings as in
    Figure~\ref{{fig:fixed_rules_time_lost}}.}}
    \label{{fig:fixed_rules_mix}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_fixed.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "8"
    if arg == "redraw":   # figures from the saved tables
        table = pd.read_csv(os.path.join(TABLES, f"params_{PLAN}_cells.csv"))
        types = pd.read_csv(os.path.join(TABLES, f"params_{PLAN}_mix_types.csv"))
    else:
        prices, evals = load(PLAN, int(arg))
        evals += quick_learn_evals()
        week = week_per_seed(evals)
        table = strategy_table(week, prices)
        table.to_csv(os.path.join(TABLES, f"params_{PLAN}_cells.csv"), index=False)
        types = mix_types()
        types.to_csv(os.path.join(TABLES, f"params_{PLAN}_mix_types.csv"), index=False)
        print(check_against_quick())
    carma_and_baselines = [k for k, _, _ in STRATEGIES]
    grouped_bars(table, "time_lost_vs_actuated", carma_and_baselines,
                 "change in time lost per trip vs actuated lights (%)", "fixed_rules_time_lost",
                 "actuated lights")
    grouped_bars(table, "cost_vs_actuated", carma_and_baselines,
                 "change in urgency-weighted cost vs actuated lights (%)", "fixed_rules_cost",
                 "actuated lights")
    COLOR["actuated"], NAME["actuated"] = "#52514e", "actuated lights"
    grouped_bars(table, "urgent_vs_calm", RULES + ["mix", "actuated", "stop", "maxpressure"],
                 "time lost of urgent trips vs calm trips (%)", "fixed_rules_urgency", None)
    if len(types):
        fig_mix(types)
    latex(table, int(table["seeds"].min()))
    cols = ["network", "lanes", "strategy", "seeds", "time_lost", "time_lost_vs_actuated",
            "cost_vs_actuated", "urgent_vs_calm", "cost_gini", "price"]
    pd.set_option("display.width", 200)
    print(table[[c for c in cols if c in table]].round(1).to_string(index=False))
    if len(types):
        print(types[["network", "lanes", "kind", "time_lost", "time_lost_vs_mix",
                     "cost_vs_mix"]].round(1).to_string(index=False))
