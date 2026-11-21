"""Lanes study: urgency-weighted time lost vs number of lanes, per strategy and load level.
python analyze/lanes_analysis.py

Inputs: results/lanes (written by controller/lanes_study.py). Outputs: the figure lanes_cost.png in
analyze/figures/, results/analysis/lanes_summary.csv and results/analysis/lanes_paired.csv. "cost"
is urgency x time lost averaged over cars (lower is better); "base" is the load level (the arrival
gap is base / lanes seconds). The paired table compares every strategy with default_lights and
with actuated_lights seed by seed (all strategies at one lane count and load share one trip file;
see paired.py).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd

from paired import paired

# Folders, found relative to this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "lanes")
OUT = os.path.join(ROOT, "analyze", "figures")
TAB = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True); os.makedirs(TAB, exist_ok=True)


def load():
    """One row per run with the urgency-weighted cost, time lost and the backlog."""
    rows = []
    for f in glob.glob(os.path.join(RES, "*_run.csv")):
        run = pd.read_csv(f).iloc[0]                                   # run-level numbers
        cars = pd.read_csv(f.replace("_run.csv", "_cars.csv"))         # one row per car
        cars["cost"] = cars["urgency"] * cars["time_loss"]             # urgency-weighted time lost
        urgent, calm = cars[cars["urgency"] > 1], cars[cars["urgency"] <= 1]
        rows.append(dict(name=run["experiment_name"], lanes=int(run["lanes"]), base=run["base"],
                         seed=run["seed"], cost=cars["cost"].mean(), loss=cars["time_loss"].mean(),
                         loss_urgent=urgent["time_loss"].mean(), loss_calm=calm["time_loss"].mean(),
                         backlog=run["not_in_network"], jammed=bool(run["jammed"]),
                         cars=len(cars)))
    return pd.DataFrame(rows)


def main():
    """Print the cost, time lost and backlog tables per load level, save the summary and draw
    lanes_cost.png.
    """
    d = load()
    # Average the seeds for every (load, strategy, lane count).
    g = d.groupby(["base", "name", "lanes"]).mean(numeric_only=True).round(2)
    g.to_csv(os.path.join(TAB, "lanes_summary.csv"))
    pd.set_option("display.width", 250)
    # Printed tables: strategies as rows, lane counts as columns, one block per load level.
    for base in sorted(d["base"].unique(), reverse=True):
        print(f"\n=== urgency-weighted time lost, load base={base} (p = base/lanes) ===")
        print(g.loc[base]["cost"].unstack("lanes").round(1).to_string())
        print(f"--- mean time lost (s, unweighted) ---")
        print(g.loc[base]["loss"].unstack("lanes").round(1).to_string())
        print(f"--- backlog (cars not finished) ---")
        print(g.loc[base]["backlog"].unstack("lanes").round(0).to_string())
    # Paired by seed against the fixed and the actuated lights: mean % difference in cost, with
    # the number of seeds on which the strategy was better (e.g. "-35% 2/2").
    tables = [paired(d, "cost", ref, units=("seed",), by=("base", "lanes"))
              for ref in ("default_lights", "actuated_lights") if ref in set(d["name"])]
    if tables:
        pt = pd.concat(tables, ignore_index=True)
        pt.round(4).to_csv(os.path.join(TAB, "lanes_paired.csv"), index=False)
        pt["cell"] = pt.apply(lambda r: f"{r.mean_diff:+.0f}% {r.better}/{r.n}", axis=1)
        for ref in pt["reference"].unique():
            for base in sorted(pt["base"].unique(), reverse=True):
                t = pt[(pt.reference == ref) & (pt.base == base)]
                print(f"\n=== paired by seed: cost % vs {ref}, load base={base} (seeds better) ===")
                print(t.pivot(index="name", columns="lanes", values="cell").to_string())
    # Figure: cost against lane count, one panel per load level, log scale on the cost.
    bases = sorted(d["base"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(bases), figsize=(6 * len(bases), 4.8), squeeze=False, sharey=False)
    for ax, base in zip(axes[0], bases):
        for name in g.loc[base].index.get_level_values("name").unique():
            s = g.loc[(base, name)]["cost"]
            ax.plot(s.index, s.values, marker="o", label=name,
                    ls="--" if name in ("urg_coin", "urg_omniscient") else "-")   # benchmarks dashed
        ax.set_yscale("log"); ax.set_xticks(sorted(d["lanes"].unique()))
        ax.set_xlabel("lanes per road"); ax.set_ylabel("urgency-weighted time lost (s)")
        ax.set_title(f"per-lane load base={base}"); ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lanes_cost.png"), dpi=130)


if __name__ == "__main__":
    main()
