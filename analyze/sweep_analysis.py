"""Summaries and plots for the parameter sweeps run by controller/sweeps.py.

    python analyze/sweep_analysis.py carma     # starting carma 50 ... 500
    python analyze/sweep_analysis.py demand    # default lights vs carma strategies, by demand
    python analyze/sweep_analysis.py all

Tables go to analyze/tables/sweeps/ and figures to analyze/figures/sweeps/. Both sweeps can be
analysed while they are still running: the runs that have finished are used.

Terms:
  demand      traffic in vehicles per hour = 3600 / p, where p is the randomTrips period.
  jammed      a run where >= 90% of the cars (at least 100) were stopped for 300 s in a row.
              Jammed runs are ended at that point, so their numbers describe the time up to the jam.
  throughput  cars that reached their destination per hour of simulated time.
  completion  share of the cars due to depart that had arrived when the run ended. Below 100%
              means the network could not keep up (cars still driving or waiting to enter).
              Delay statistics only cover the cars that arrived, so they flatter a network that
              is falling behind; read them together with throughput and completion.
  improvement % change against the default traffic lights at the same demand and seed
              (positive = better than the default lights).
"""
import argparse
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import delay_metrics as dm

SWEEPS_DIR = dm.ANALYZE_DIR.parent / "results" / "sweeps"
TABLES_DIR = dm.ANALYZE_DIR / "tables" / "sweeps"
FIGURES_DIR = dm.ANALYZE_DIR / "figures" / "sweeps"

# Seconds over which the simulation inserts vehicles (the trips run from 0 to this time).
INSERTION_END = 7200

# Metrics where a LOWER value is better, and ones where HIGHER is better.
LOWER_IS_BETTER = ["mean_wait", "p95_wait", "mean_time_loss", "mean_delay_ratio"]
HIGHER_IS_BETTER = ["throughput_per_hour"]

METRIC_LABELS = {
    "mean_wait": "Mean wait time (s)",
    "p95_wait": "95th percentile wait time (s)",
    "mean_time_loss": "Mean time loss (s)",
    "mean_delay_ratio": "Mean delay ratio (1 = no delay)",
    "mean_travel_time": "Mean travel time (s)",
    "throughput_per_hour": "Throughput (cars arrived per hour)",
    "completion_pct": "Completion (% of cars due that arrived)",
    "share_broke_pct": "Cars that ended with < 1 carma (%)",
}


# ---- Loading ----------------------------------------------------------------------------------

def read_runs(sweep: str) -> pd.DataFrame:
    """One row per finished run (the *_run.csv files of a sweep)."""
    frames = []
    for path in sorted((SWEEPS_DIR / sweep).glob("*_run.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_cars(sweep: str, labels) -> pd.DataFrame:
    """The per-car results of the given runs, with a `label` column."""
    frames = []
    for label in labels:
        path = SWEEPS_DIR / sweep / f"{label}_cars.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["label"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return dm.add_derived(pd.concat(frames, ignore_index=True))


def p95(values: pd.Series) -> float:
    """95th percentile of a column (the unlucky 5% of cars)."""
    return values.quantile(0.95)


def share_broke(values: pd.Series) -> float:
    """Percentage of cars whose final balance is below 1 carma (they ran out)."""
    return (values < 1.0).mean() * 100


def per_run_stats(cars: pd.DataFrame, by_kind: bool) -> pd.DataFrame:
    """Delay statistics of each run (and of each bidder type within it if by_kind)."""
    keys = ["label"]
    if by_kind:
        keys.append("kind")
    return cars.groupby(keys).agg(
        cars=("vid", "count"),
        mean_wait=("wait_time", "mean"),
        p95_wait=("wait_time", p95),
        mean_time_loss=("time_loss", "mean"),
        mean_delay_ratio=("delay_ratio", "mean"),
        mean_travel_time=("travel_time", "mean"),
        share_broke_pct=("final_carma", share_broke),
    ).reset_index()


def runs_with_stats(sweep: str, by_kind: bool) -> pd.DataFrame:
    """Run info joined with the delay statistics, one row per run (or run and bidder type)."""
    runs = read_runs(sweep)
    if runs.empty:
        return runs
    cars = read_cars(sweep, runs["label"])
    stats = per_run_stats(cars, by_kind)
    table = runs.merge(stats, on="label", how="inner")
    # Throughput over the time actually simulated, and the share of due cars that arrived.
    table["throughput_per_hour"] = table["cars_arrived"] / table["end_time"] * 3600
    return table


def save_table(table: pd.DataFrame, name: str) -> None:
    """Write a table to results/analysis as a CSV file."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(TABLES_DIR / f"{name}.csv")
    print(f"Saved table {TABLES_DIR / (name + '.csv')}")


def save_figure(fig, name: str) -> None:
    """Save a figure to analyze/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved figure {path}")


# ---- Starting carma sweep ---------------------------------------------------------------------

def analyse_carma() -> None:
    """Starting-carma sweep: tables and figures of delay against the starting balance, per bidder
    type.
    """
    table = runs_with_stats("initial_carma", by_kind=True)
    if table.empty:
        print("No finished runs in results/sweeps/initial_carma yet.")
        return
    metrics = ["mean_wait", "p95_wait", "mean_time_loss", "mean_delay_ratio", "share_broke_pct"]
    # Average over seeds; the spread across seeds shows how much the result moves between runs.
    grouped = table.groupby(["kind", "value"])[metrics]
    mean = grouped.mean()
    spread = grouped.std()
    seeds = table.groupby(["kind", "value"])["seed"].nunique().rename("seeds")
    summary = mean.join(spread, rsuffix="_std").join(seeds)
    save_table(summary, "carma_sweep_by_type")

    kinds = sorted(table["kind"].unique())
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, metrics):
        for index, kind in enumerate(kinds):
            rows = mean.loc[kind]
            errors = spread.loc[kind][metric].fillna(0.0)
            ax.errorbar(rows.index, rows[metric], yerr=errors, marker="o", capsize=3,
                        label=kind, color=colors(index % 10))
        ax.set_xlabel("Starting carma")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3)
    axes.flat[0].legend(title="Bidder type")
    # The last panel: how many carma the cars ended with, as a share of what they started with.
    ax = axes.flat[5]
    cars = read_cars("initial_carma", read_runs("initial_carma")["label"])
    runs = read_runs("initial_carma")[["label", "value"]]
    cars = cars.merge(runs, on="label")
    cars["final_share"] = cars["final_carma"] / cars["start_carma"]
    ending = cars.groupby(["kind", "value"])["final_share"].mean()
    for index, kind in enumerate(kinds):
        ax.plot(ending.loc[kind].index, ending.loc[kind].values * 100, marker="o", label=kind,
                color=colors(index % 10))
    ax.set_xlabel("Starting carma")
    ax.set_ylabel("Final carma as % of starting carma")
    ax.grid(alpha=0.3)
    fig.suptitle("Effect of the starting carma balance (mean over seeds, bars = std across seeds)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "carma_sweep")


# ---- Demand study -----------------------------------------------------------------------------

def demand_table() -> pd.DataFrame:
    """One row per demand-sweep run with throughput, completion and delay measures."""
    table = runs_with_stats("demand", by_kind=False)
    if table.empty:
        return table
    table["demand_per_hour"] = 3600 / table["value"]
    # Cars due to depart by the end of the run (one every p seconds, over the insertion time).
    due = (np.minimum(table["end_time"], INSERTION_END) / table["value"]).clip(lower=1)
    table["completion_pct"] = table["cars_arrived"] / due * 100
    return table


def majority_jammed(group: pd.DataFrame) -> bool:
    """True if most of the runs in the group jammed."""
    return group["jammed"].sum() * 2 > len(group)


def capacity_table(table: pd.DataFrame) -> pd.DataFrame:
    """For each configuration: the heaviest demand it handled without jamming, and the first one
    where it jammed (most seeds jammed)."""
    rows = []
    for name, runs in table.groupby("experiment"):
        ok_p = []
        jam_p = []
        for p, at_p in runs.groupby("value"):
            if majority_jammed(at_p):
                jam_p.append(p)
            else:
                ok_p.append(p)
        last_ok = min(ok_p) if ok_p else np.nan
        first_jam = max(jam_p) if jam_p else np.nan
        rows.append({
            "configuration": name,
            "last_ok_p": last_ok,
            "max_demand_ok_per_hour": 3600 / last_ok if ok_p else np.nan,
            "first_jammed_p": first_jam,
            "jammed_demand_per_hour": 3600 / first_jam if jam_p else np.nan,
            "levels_run": runs["value"].nunique(),
        })
    return pd.DataFrame(rows).set_index("configuration")


def improvement_table(table: pd.DataFrame, control: str = "default_lights") -> pd.DataFrame:
    """Each carma strategy against the default lights, run by run (same p and seed).

    One row per (strategy, p, seed) where both ran. status says what happened:
      compared      neither jammed; the improvement columns are filled in.
      strategy_jam  only the carma strategy jammed.
      default_jam   only the default lights jammed (the strategy survived where the lights did not).
      both_jam      both jammed.
    """
    metrics = LOWER_IS_BETTER + HIGHER_IS_BETTER
    base = table[table["experiment"] == control].set_index(["value", "seed"])
    rows = []
    for name in sorted(table["experiment"].unique()):
        if name == control:
            continue
        for _, run in table[table["experiment"] == name].iterrows():
            key = (run["value"], run["seed"])
            if key not in base.index:
                continue
            reference = base.loc[key]
            row = {"experiment": name, "p": run["value"], "seed": run["seed"]}
            if run["jammed"] and reference["jammed"]:
                row["status"] = "both_jam"
            elif run["jammed"]:
                row["status"] = "strategy_jam"
            elif reference["jammed"]:
                row["status"] = "default_jam"
            else:
                row["status"] = "compared"
                for metric in LOWER_IS_BETTER:
                    row[f"{metric}_improvement_pct"] = (
                        (reference[metric] - run[metric]) / reference[metric] * 100)
                for metric in HIGHER_IS_BETTER:
                    row[f"{metric}_improvement_pct"] = (
                        (run[metric] - reference[metric]) / reference[metric] * 100)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_demand_curves(table: pd.DataFrame) -> None:
    """Delay, throughput and completion against demand, one line per configuration."""
    metrics = ["mean_wait", "p95_wait", "mean_time_loss", "mean_delay_ratio",
               "throughput_per_hour", "completion_pct"]
    configs = sorted(table["experiment"].unique())
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, metric in zip(axes.flat, metrics):
        for index, name in enumerate(configs):
            runs = table[table["experiment"] == name]
            by_p = runs.groupby("demand_per_hour")
            mean = by_p[metric].mean()
            spread = by_p[metric].std().fillna(0.0)
            style = "--" if name == "default_lights" else "-"
            ax.errorbar(mean.index, mean.values, yerr=spread.values, marker="o", markersize=4,
                        capsize=2, linestyle=style, label=name, color=colors(index % 10),
                        linewidth=2.2 if name == "default_lights" else 1.3)
            # Mark the demand levels where this configuration jammed with a cross.
            for demand, at_demand in by_p:
                if majority_jammed(at_demand):
                    ax.plot(demand, at_demand[metric].mean(), marker="X", markersize=11,
                            color=colors(index % 10), markeredgecolor="black")
        ax.set_xlabel("Demand (vehicles per hour)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.3)
    axes.flat[0].legend(title="Configuration", fontsize=8)
    fig.suptitle("Delay and throughput against demand (X = jammed; bars = std across seeds; "
                 "delay is over cars that arrived)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "demand_curves")


def plot_improvement(table: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Heat maps of how much each carma strategy improves on the default lights, per demand level.
    """
    strategies = sorted(comparison["experiment"].unique())
    levels = sorted(table["value"].unique(), reverse=True)
    metrics = [("mean_time_loss", "Mean time loss"), ("p95_wait", "95th percentile wait"),
               ("throughput_per_hour", "Throughput")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.7 * len(strategies) + 3.2), squeeze=False)
    for ax, (metric, title) in zip(axes[0], metrics):
        grid = np.full((len(strategies), len(levels)), np.nan)
        notes = [[""] * len(levels) for _ in strategies]
        for row, name in enumerate(strategies):
            for col, p in enumerate(levels):
                cell = comparison[(comparison["experiment"] == name) & (comparison["p"] == p)]
                if cell.empty:
                    continue
                statuses = set(cell["status"])
                compared = cell[cell["status"] == "compared"]
                if statuses == {"default_jam"}:
                    notes[row][col] = "D"          # the strategy ran where the default jammed
                elif statuses <= {"both_jam"}:
                    notes[row][col] = "J"
                elif statuses == {"strategy_jam"}:
                    notes[row][col] = "S"          # the strategy jammed where the default did not
                elif not compared.empty:
                    values = compared[f"{metric}_improvement_pct"]
                    grid[row, col] = values.mean()
                    # A star where every seed agrees on the direction.
                    consistent = len(values) > 1 and ((values > 0).all() or (values < 0).all())
                    notes[row][col] = f"{values.mean():.0f}" + ("*" if consistent else "")
        image = ax.imshow(grid, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{3600 / p:.0f}" for p in levels], rotation=90, fontsize=8)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=8)
        ax.set_xlabel("Demand (vehicles per hour)")
        ax.set_title(f"{title}: % better than default lights")
        for row in range(len(strategies)):
            for col in range(len(levels)):
                ax.text(col, row, notes[row][col], ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.04)
    fig.suptitle("Do carma strategies beat the default lights?  Green = better, red = worse.  "
                 "* = all seeds agree.  D = default lights jammed, strategy did not.  "
                 "S = strategy jammed, default did not.  J = both jammed.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "demand_improvement")


def plot_capacity(capacity: pd.DataFrame) -> None:
    """Bars of the heaviest demand each configuration handled without a jam, with the first jammed
    level marked.
    """
    capacity = capacity.sort_values("max_demand_ok_per_hour")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(capacity) + 2))
    positions = np.arange(len(capacity))
    ax.barh(positions, capacity["max_demand_ok_per_hour"].fillna(0), color="tab:blue",
            label="heaviest demand without a jam")
    ax.scatter(capacity["jammed_demand_per_hour"], positions, marker="X", color="tab:red", s=80,
               label="first demand level that jammed", zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(capacity.index)
    ax.set_xlabel("Demand (vehicles per hour)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("How much traffic each configuration handles before jamming")
    fig.tight_layout()
    save_figure(fig, "demand_capacity")


def analyse_demand() -> None:
    """Demand sweep: build the tables, print the capacity summary and draw the figures."""
    table = demand_table()
    if table.empty:
        print("No finished runs in results/sweeps/demand yet.")
        return
    columns = ["experiment", "value", "demand_per_hour", "seed", "jammed", "jam_time",
               "cars_arrived", "cars_in_network", "not_in_network", "completion_pct",
               "throughput_per_hour", "mean_wait", "p95_wait", "mean_time_loss",
               "mean_delay_ratio", "mean_travel_time", "mean_stopped_share", "teleports",
               "wall_seconds"]
    save_table(table[columns].sort_values(["experiment", "value", "seed"],
                                          ascending=[True, False, True]).set_index("experiment"),
               "demand_runs")

    capacity = capacity_table(table)
    save_table(capacity, "demand_capacity")
    print("\nWhere each configuration jams:\n" + capacity.round(2).to_string())

    # Average over seeds, per configuration and demand level.
    metrics = LOWER_IS_BETTER + HIGHER_IS_BETTER + ["completion_pct", "mean_stopped_share"]
    by_level = table.groupby(["experiment", "demand_per_hour"])
    summary = by_level[metrics].mean()
    summary["jammed_seeds"] = by_level["jammed"].sum()
    summary["seeds"] = by_level["seed"].nunique()
    save_table(summary, "demand_summary")

    comparison = improvement_table(table)
    if not comparison.empty:
        save_table(comparison.set_index("experiment"), "demand_improvement_by_run")
        compared = comparison[comparison["status"] == "compared"]
        if not compared.empty:
            columns = [c for c in compared.columns if c.endswith("_improvement_pct")]
            by_p = compared.groupby(["experiment", "p"])[columns].mean()
            save_table(by_p, "demand_improvement")
        plot_improvement(table, comparison)
    plot_demand_curves(table)
    plot_capacity(capacity)


def main(argv=None) -> None:
    """Analyse the carma sweep, the demand sweep or both."""
    parser = argparse.ArgumentParser(description="Analyse the parameter sweeps.")
    parser.add_argument("sweep", choices=["carma", "demand", "all"])
    parser.add_argument("--show", action="store_true", help="Also open the figures in windows.")
    args = parser.parse_args(argv)
    if args.sweep in ("carma", "all"):
        analyse_carma()
    if args.sweep in ("demand", "all"):
        analyse_demand()
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
