"""Stop-sign baseline vs the traffic lights and carma strategies (tasks 1 and 2).

  1. 5x5 grid, demand sweep (v1 runs + the stop_signs runs):  figure stop_signs_grid_demand.png
  2. single intersection, persistence runs + stop signs:      figure stop_signs_single.png
Run from the repo root:  python analyze/stop_sign_analysis.py

Inputs: results/saved/sweeps_v1_single_winner/demand (the old v1 demand sweep), results/sweeps/demand
(the stop-sign runs) and results/persistence. Outputs: figures in analyze/figures/ and tables in
results/analysis/.

Note the wait-time metric (seconds at speed < 0.1 m/s) flatters stop signs, because cars roll
slowly through an all-way stop instead of standing still; see the time-lost figures in the
urgency, lanes and regret analyses for a fairer comparison.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd

# Folders, found relative to this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analyze", "figures")
TABLES = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)


def load(pattern: str, name_col: str, x_col: str) -> pd.DataFrame:
    """One row per run: experiment, x (demand), seed, run stats and the mean wait of arrived cars."""
    rows = []
    for run_file in glob.glob(pattern):
        # The *_run.csv file has one row of run-level numbers (jammed, cars arrived, ...).
        run = pd.read_csv(run_file).iloc[0].to_dict()
        # The matching *_cars.csv has one row per car.
        cars = pd.read_csv(run_file.replace("_run.csv", "_cars.csv"))
        run["name"] = run[name_col]          # which strategy
        run["x"] = run[x_col]                # the demand level on the x axis
        run["mean_wait"] = cars["wait_time"].mean()
        run["p95_wait"] = cars["wait_time"].quantile(0.95)     # the unlucky 5% of cars
        run["mode"] = run.get("mode", "")
        rows.append(run)
    return pd.DataFrame(rows)


def panel(ax, data, ycol, ylabel, highlight="stop_signs"):
    """Plot one metric against demand, one line per strategy (stop signs bold black, default
    lights red, carma strategies dashed)."""
    for name, g in data.groupby("name"):
        m = g.groupby("x")[ycol].mean()      # average over seeds
        stop = name == highlight
        base = name == "default_lights"
        ax.plot(m.index, m.values, marker="o", ms=3, lw=2.6 if stop else 1.6 if base else 1,
                color="black" if stop else "tab:red" if base else None,
                ls="-" if (stop or base) else "--", label=name, zorder=5 if stop else 2)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.3)


def grid_figure():
    """Task 1: the 5x5 grid demand sweep, with the stop-sign runs added to the old v1 sweep."""
    old = load(os.path.join(ROOT, "results/saved/sweeps_v1_single_winner/demand/*_run.csv"),
               "experiment", "value")
    new = load(os.path.join(ROOT, "results/sweeps/demand/stop_signs_*_run.csv"), "experiment", "value")
    d = pd.concat([old, new])
    d = d[d["jammed"] == False]           # a jammed run never finishes: compare drivable levels
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panel(axes[0], d, "mean_wait", "mean wait of arrived cars (s)")
    panel(axes[1], d, "mean_stopped_share", "mean share of cars stopped")
    panel(axes[2], d, "cars_arrived", "cars arrived (of those inserted)")
    for ax in axes:
        ax.set_xlabel("p (s between inserted vehicles; lower = more traffic)")
        ax.invert_xaxis()                 # heavier traffic to the right
    axes[0].legend(fontsize=7)
    fig.suptitle("5x5 grid, non-jammed runs (v1 sweep + stop signs)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stop_signs_grid_demand.png"), dpi=130)
    # Table including the jammed runs: the share of runs that jammed at each demand level.
    allr = pd.concat([old, new])
    t = allr.groupby(["name", "x"]).agg(jam_share=("jammed", "mean"), arrived=("cars_arrived", "mean"),
                                        stopped=("mean_stopped_share", "mean"),
                                        wait=("mean_wait", "mean")).round(2)
    t.to_csv(os.path.join(TABLES, "stop_signs_grid_demand.csv"))
    print(t["jam_share"].unstack("x").to_string())


def single_figure():
    """Task 2: the single intersection over 4 hours of demand (persistent carma)."""
    d = load(os.path.join(ROOT, "results/persistence/*_run.csv"), "experiment_name", "p")
    d = d[(d["mode"] != "reset")]         # persist runs + default lights + stop signs
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panel(axes[0], d, "mean_wait", "mean wait of arrived cars (s)")
    panel(axes[1], d, "p95_wait", "95th percentile wait (s)")
    panel(axes[2], d, "not_in_network", "vehicles never inserted (backlog)")
    for ax in axes:
        ax.set_xlabel("p (s between vehicles)")
        ax.invert_xaxis()
        ax.set_xscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("Single 4-way intersection, 4 h of demand (persistent carma)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stop_signs_single.png"), dpi=130)
    t = d.groupby(["name", "x"]).agg(wait=("mean_wait", "mean"), p95=("p95_wait", "mean"),
                                     backlog=("not_in_network", "mean")).round(1)
    t.to_csv(os.path.join(TABLES, "stop_signs_single.csv"))
    print(t["wait"].unstack("x").to_string())


if __name__ == "__main__":
    grid_figure()
    single_figure()
