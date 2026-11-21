"""Urgency study: urgency-weighted delay, urgent vs calm waits, efficiency between the coin floor
and the omniscient ceiling, and fairness across drivers.   python analyze/urgency_analysis.py

Inputs: results/urgency (written by controller/urgency_study.py). Outputs: the figure
urgency_time_lost.png in analyze/figures/, results/analysis/urgency_summary.csv and
results/analysis/urgency_paired.csv (every strategy against each baseline, paired by seed: all
runs share one trip file, and a seed fixes the urgency draws, so runs with the same seed saw the
same traffic; see paired.py).

Terms:
  cost        urgency x time lost, averaged over cars (a second lost by an urgent trip counts 6x)
  efficiency  0 = as bad as the coin benchmark, 1 = as good as the omniscient benchmark
              (misleading at vehicle level, where the omniscient benchmark is not a true ceiling)
  *_gini      inequality (0 = equal) of the drivers' average cost
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paired import paired, show

# Baselines every strategy is compared with, pair by pair.
REFERENCES = ("default_lights", "actuated_lights", "delay_based_lights", "max_pressure", "stop_signs")

# Folders, found relative to this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "urgency")
OUT = os.path.join(ROOT, "analyze", "figures")
TAB = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True); os.makedirs(TAB, exist_ok=True)


def load():
    """One row per run with the urgency-weighted cost and the urgent / calm split."""
    rows = []
    for f in glob.glob(os.path.join(RES, "*_run.csv")):
        run = pd.read_csv(f).iloc[0]                                   # run-level numbers
        cars = pd.read_csv(f.replace("_run.csv", "_cars.csv"))         # one row per car
        cars["cost"] = cars["urgency"] * cars["time_loss"]      # urgency-weighted time lost
        urgent = cars[cars["urgency"] > 1]                      # urgency 6 trips
        calm = cars[cars["urgency"] <= 1]                       # urgency 1 trips
        r = dict(name=run["experiment_name"], p=run["p"], seed=run["seed"], k=run["start_carma"],
                 family=run["family"], backlog=run["not_in_network"],
                 cost=cars["cost"].mean(), wait_urgent=urgent["wait_time"].mean(),
                 wait_calm=calm["wait_time"].mean(), loss_urgent=urgent["time_loss"].mean(),
                 loss_calm=calm["time_loss"].mean(), wait=cars["wait_time"].mean())
        # Fairness across drivers (only when drivers are persistent).
        if cars["driver"].notna().any():
            per = cars.groupby("driver")["cost"].mean()
            r["driver_cost_sd"] = per.std()
            r["driver_cost_gini"] = _gini(per.values)
        rows.append(r)
    return pd.DataFrame(rows)


def _gini(v):
    """Gini coefficient of a list of values: 0 = all equal, 1 = one value holds everything."""
    v = np.sort(np.asarray(v, float)); n = len(v)
    return float((2 * np.sum((np.arange(1, n + 1)) * v)) / (n * v.sum()) - (n + 1) / n) if v.sum() > 0 else 0.0


def main():
    """Average the seeds, compute the efficiency scale, print a table per demand level, save the
    summary and draw urgency_time_lost.png.
    """
    d = load()
    # Label carma strategies with their starting balance, e.g. "only_lane_urgency (k=100)".
    d["label"] = d.apply(lambda r: r["name"] + (f" (k={r['k']:g})" if r["family"] == "carma" else ""), axis=1)
    # Average the seeds.
    g = d.groupby(["p", "label"]).mean(numeric_only=True)
    # efficiency: 0 = coin toss, 1 = omniscient (on urgency-weighted time lost)
    for p in sorted(d["p"].unique()):
        sub = g.loc[p]
        if "urg_coin" in sub.index and "urg_omniscient" in sub.index:
            lo, hi = sub.loc["urg_coin", "cost"], sub.loc["urg_omniscient", "cost"]
            g.loc[(p, slice(None)), "efficiency"] = ((lo - g.loc[p]["cost"]) / (lo - hi)).values
    g = g.round(2)
    g.to_csv(os.path.join(TAB, "urgency_summary.csv"))
    cols = ["cost", "efficiency", "loss_urgent", "loss_calm", "wait_urgent", "wait_calm", "backlog", "driver_cost_gini"]
    pd.set_option("display.width", 250)
    # Print a table per demand level, best (lowest cost) first.
    for p in sorted(d["p"].unique()):
        print(f"\n=== p={p} ===")
        print(g.loc[p][[c for c in cols if c in g.columns]].sort_values("cost").to_string())
    # Paired comparisons: every strategy against each baseline, seed by seed.
    tables = [paired(d, "cost", ref, units=("seed",), by=("p",), name="label")
              for ref in REFERENCES if ref in set(d["label"])]
    if tables:
        pt = pd.concat(tables, ignore_index=True)
        pt.round(4).to_csv(os.path.join(TAB, "urgency_paired.csv"), index=False)
        for ref in ("default_lights", "actuated_lights"):
            t = pt[pt.reference == ref]
            if len(t):
                print(f"\n=== paired by seed: urgency-weighted time lost, % vs {ref} ===")
                print(show(t, by=("p",), name="label"))
    # figure: urgent vs calm time lost per strategy
    ps = sorted(d["p"].unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(7.5 * len(ps), 5), squeeze=False)
    for ax, p in zip(axes[0], ps):
        s = g.loc[p].sort_values("cost")
        y = np.arange(len(s))
        ax.barh(y - 0.2, s["loss_calm"], 0.4, label="calm (u=1)")
        ax.barh(y + 0.2, s["loss_urgent"], 0.4, label="urgent (u=6)")
        ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("mean time lost (s)"); ax.set_title(f"p={p}, sorted by urgency-weighted cost")
        ax.grid(alpha=.3, axis="x")
    axes[0][0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "urgency_time_lost.png"), dpi=130)


if __name__ == "__main__":
    main()
