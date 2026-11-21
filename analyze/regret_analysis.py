"""Regret-minimisation study (5x5, multi-day): urgency-weighted delay by day for each strategy,
fairness, the learned policy, and (for the mixed populations) how each driver type fares.
python analyze/regret_analysis.py

Inputs: results/regret (written by controller/regret_study.py). Outputs: the figure
regret_days.png in analyze/figures/, results/analysis/regret_by_day.csv and
results/analysis/regret_paired.csv. "cost" is urgency x time lost averaged over cars (lower is
better), "loss" the plain time lost. Printed: the mixed-population comparison, the cost by day,
the days 5-7 averages, the paired comparisons, and the learned policy of one run.

Paired comparisons (see paired.py): every strategy sees the same trips on the same day of a chain,
so each strategy is compared with a reference on every (seed, day) and the differences are
summarised (mean, 95% interval, days won). References: only_lane_urgency (the hand-made lane
bidder) and default_lights. Per-driver fairness (driver_gini) is compared in absolute terms.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paired import paired, show

# Folders, found relative to this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "regret")
OUT = os.path.join(ROOT, "analyze", "figures")
TAB = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True); os.makedirs(TAB, exist_ok=True)


def _gini(v):
    """Gini coefficient of a list of values: 0 = all equal, 1 = one value holds everything."""
    v = np.sort(np.asarray(v, float)); n = len(v)
    return float(2 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum()) - (n + 1) / n) if v.sum() > 0 else 0.0


def load():
    """One row per simulated day of every chain, with cost and the urgent / calm split."""
    rows = []
    for f in glob.glob(os.path.join(RES, "*_run.csv")):
        run = pd.read_csv(f).iloc[0]                                   # run-level numbers
        cars = pd.read_csv(f.replace("_run.csv", "_cars.csv"))         # one row per car
        cars["cost"] = cars["urgency"] * cars["time_loss"]             # urgency-weighted time lost
        urgent, calm = cars[cars["urgency"] > 1], cars[cars["urgency"] <= 1]
        r = dict(name=run["experiment_name"], p=run["p"], seed=run["chain_seed"], day=int(run["day"]),
                 cost=cars["cost"].mean(), loss=cars["time_loss"].mean(),
                 loss_urgent=urgent["time_loss"].mean(), loss_calm=calm["time_loss"].mean(),
                 backlog=run["not_in_network"], jammed=bool(run["jammed"]))
        # Fairness across drivers (only when drivers are persistent).
        if cars["driver"].notna().any():
            r["driver_gini"] = _gini(cars.groupby("driver")["cost"].mean().values)
        rows.append(r)
    return pd.DataFrame(rows)


def load_kinds():
    """Per-driver-type results of the mixed-population runs."""
    rows = []
    for f in glob.glob(os.path.join(RES, "mix_*_run.csv")):
        run = pd.read_csv(f).iloc[0]
        cars = pd.read_csv(f.replace("_run.csv", "_cars.csv"))
        cars["cost"] = cars["urgency"] * cars["time_loss"]
        # The `kind` column says which bidder type drove each trip (learner or heuristic).
        for kind, g in cars.groupby("kind"):
            urgent, calm = g[g["urgency"] > 1], g[g["urgency"] <= 1]
            rows.append(dict(mix=run["experiment_name"], p=run["p"], seed=run["chain_seed"],
                             day=int(run["day"]), kind=kind, trips=len(g), cost=g["cost"].mean(),
                             loss_urgent=urgent["time_loss"].mean(), loss_calm=calm["time_loss"].mean()))
    return pd.DataFrame(rows)


def main():
    """Print the mixed-population, by-day and days 5-7 tables, save regret_by_day.csv, draw
    regret_days.png and show a learned policy.
    """
    # --- mixed populations: does a learner do better than the heuristic drivers it shares roads with?
    k = load_kinds()
    if len(k):
        print("\n=== mixed populations (days 3-7): mean urgency-weighted time lost per driver type ===")
        late = k[k.day >= 3].groupby(["mix", "p", "kind"])[["trips", "cost", "loss_urgent", "loss_calm"]].mean().round(1)
        print(late.to_string())
    d = load()
    pd.set_option("display.width", 250)
    d.groupby(["p", "name", "day"]).mean(numeric_only=True).round(2).to_csv(os.path.join(TAB, "regret_by_day.csv"))
    # --- cost by day and the average over days 5-7 (after the first days' transient), per p
    for p in sorted(d["p"].unique(), reverse=True):
        sub = d[d.p == p]
        print(f"\n=== p={p}: urgency-weighted time lost by day (mean of seeds) ===")
        print(sub.groupby(["name", "day"])["cost"].mean().unstack("day").round(0).to_string())
        late = sub[sub.day >= 5].groupby("name")[["cost", "loss", "loss_urgent", "loss_calm", "backlog", "driver_gini"]].mean().round(2)
        print(f"--- days 5-7 average ---")
        print(late.sort_values("cost").to_string())
    # --- paired comparisons on the same (seed, day): cost and time lost against two references,
    # and per-driver fairness (Gini of the drivers' mean cost) in absolute terms
    tables = []
    for ref in ("only_lane_urgency", "default_lights"):
        for metric in ("cost", "loss"):
            tables.append(paired(d, metric, ref, units=("seed", "day"), by=("p",)))
    if "driver_gini" in d:
        tables.append(paired(d, "driver_gini", "default_lights", units=("seed", "day"), by=("p",),
                             relative=False))
    pt = pd.concat(tables, ignore_index=True)
    pt.round(4).to_csv(os.path.join(TAB, "regret_paired.csv"), index=False)
    for metric, ref, what in (("cost", "only_lane_urgency", "urgency-weighted time lost, % vs only_lane_urgency"),
                              ("cost", "default_lights", "urgency-weighted time lost, % vs default_lights"),
                              ("loss", "default_lights", "time lost (unweighted), % vs default_lights"),
                              ("driver_gini", "default_lights", "per-driver cost Gini, difference vs default_lights")):
        t = pt[(pt.metric == metric) & (pt.reference == ref)]
        if len(t):
            print(f"\n=== paired by (seed, day): {what} ===")
            print(show(t, by=("p",), digits=3 if metric == "driver_gini" else 1))
    # --- figure: cost by day, one panel per demand level (learners solid, the others dashed)
    ps = sorted(d["p"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(ps), figsize=(6.5 * len(ps), 4.8), squeeze=False)
    for ax, p in zip(axes[0], ps):
        for name, g in d[d.p == p].groupby("name"):
            s = g.groupby("day")["cost"].mean()
            ax.plot(s.index, s.values, marker="o", label=name,
                    ls="-" if name.startswith("regret") else "--")
        ax.set_xlabel("day"); ax.set_ylabel("urgency-weighted time lost (s)")
        ax.set_title(f"5x5, p={p}"); ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "regret_days.png"), dpi=130)
    # Learned policy of the last day, averaged over seeds and p.
    for f in sorted(glob.glob(os.path.join(RES, "regret_lane_1.0_p1.00_s0_day7_policy.csv"))):
        pol = pd.read_csv(f)
        print("\nLearned policy (lam0=1, p=1.0, seed 0, day 7): mean fraction of balance bid")
        print(pol.pivot_table(index=["urgency", "wait_bucket"], columns="balance_bucket",
                              values="mean_fraction").round(2).to_string())


if __name__ == "__main__":
    main()
