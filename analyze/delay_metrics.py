"""Delay metrics per bidder type, from the per-car results files (results/*_cars.csv).

Run it to print the tables and save them as CSV files in analyze/tables/:

    python analyze/delay_metrics.py                      # the most recent *_cars.csv
    python analyze/delay_metrics.py results/a_cars.csv   # a specific run
    python analyze/delay_metrics.py a_cars.csv b_cars.csv --thresholds 30 60 120

Giving several files pools their cars together (e.g. runs with different seeds).
plot_delay.py uses the functions in this file to draw the figures.

What the tables show (all per bidder type, `kind`):

  distribution tables   mean, median, 90th and 95th percentile and max of wait time, travel time,
                        time loss and the delay ratio. The mean can hide the worst-off drivers;
                        the upper percentiles don't.
  delay ratio           travel time divided by free-flow time. It corrects for route length: a
                        long route has a long travel time even with no delay at all. Free-flow
                        time is travel time minus SUMO's time loss (time lost against driving at
                        full speed all the way).
  spread table          standard deviation of each measure, the interquartile range of the wait
                        time, and the share of cars that waited longer than each threshold.
  efficiency table      carma spent against wait saved (see efficiency_table below).
"""
import argparse
from pathlib import Path

import pandas as pd

ANALYZE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYZE_DIR.parent / "results"
TABLES_DIR = ANALYZE_DIR / "tables"

# Wait times (seconds) used for the "share of cars waiting longer than ..." columns.
DEFAULT_THRESHOLDS = (30, 60, 120)


# ---- Loading ----------------------------------------------------------------------------------

def latest_cars_file() -> Path:
    """The most recently written results/*_cars.csv."""
    files = sorted(RESULTS_DIR.glob("*_cars.csv"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise SystemExit(f"No *_cars.csv files found in {RESULTS_DIR}. Run a simulation first.")
    return files[-1]


def run_label(paths: list[Path]) -> str:
    """A short name for the run(s), used in titles and output file names."""
    if len(paths) == 1:
        name = paths[0].stem
        if name.endswith("_cars"):
            name = name[:-len("_cars")]
        return name
    return f"pooled_{len(paths)}_runs"


def safe_label(label: str) -> str:
    """The label with characters that are awkward in file names (spaces, colons) replaced."""
    return label.replace(" ", "_").replace(":", "-")


def load_cars(paths: list[Path]) -> pd.DataFrame:
    """Read one or more per-car files into one table, with a `run` column saying where each
    row came from."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["run"] = run_label([path])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_derived(cars: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the tables and plots need: free-flow time, delay ratios, carma spent."""
    cars = cars.copy()
    # Free-flow time: how long the trip would have taken with no delay at all. At least 1 s so
    # the ratios below can never divide by zero.
    cars["free_flow_time"] = (cars["travel_time"] - cars["time_loss"]).clip(lower=1.0)
    # 1.0 means no delay; 1.5 means the trip took half as long again as it needed to.
    cars["delay_ratio"] = cars["travel_time"] / cars["free_flow_time"]
    cars["time_loss_ratio"] = cars["time_loss"] / cars["free_flow_time"]
    # Net carma the car paid over its trip. Negative means it received more than it paid (carma
    # paid by winners is shared out to losing bidders). Older result files have no start_carma.
    if "start_carma" in cars.columns:
        cars["carma_spent"] = cars["start_carma"] - cars["final_carma"]
    return cars


# ---- Tables -----------------------------------------------------------------------------------

def distribution_table(cars: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean, median, 90th and 95th percentile and max of one measure, per bidder type."""
    grouped = cars.groupby("kind")[column]
    return pd.DataFrame({
        "cars": grouped.count(),
        "mean": grouped.mean(),
        "median": grouped.median(),
        "p90": grouped.quantile(0.90),
        "p95": grouped.quantile(0.95),
        "max": grouped.max(),
    })


def spread_table(cars: pd.DataFrame, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """How unevenly delay is spread within each bidder type.

    std_*  standard deviation of the measure.
    iqr_wait_time  75th minus 25th percentile of the wait time (robust to outliers).
    pct_wait_over_Ts  percentage of cars that waited more than T seconds.
    """
    grouped = cars.groupby("kind")
    table = pd.DataFrame({
        "std_travel_time": grouped["travel_time"].std(),
        "std_wait_time": grouped["wait_time"].std(),
        "std_time_loss": grouped["time_loss"].std(),
        "std_delay_ratio": grouped["delay_ratio"].std(),
        "iqr_wait_time": grouped["wait_time"].quantile(0.75) - grouped["wait_time"].quantile(0.25),
    })
    for threshold in thresholds:
        over = cars["wait_time"] > threshold
        table[f"pct_wait_over_{threshold}s"] = over.groupby(cars["kind"]).mean() * 100
    return table


def efficiency_table(cars: pd.DataFrame) -> pd.DataFrame | None:
    """Carma spent against delay saved, per bidder type. None if the data has no start_carma.

    wait_saved_vs_avg    mean wait of ALL cars in the data minus this type's mean wait. Positive
                         means the type waits less than average. (Without a no-bidding reference
                         run, "saved" can only be measured against the other cars.)
    mean_net_carma_spent mean of (start carma - final carma). Can be negative: carma paid by
                         winners is shared among losing bidders, so a type can end with more
                         than it started with.
    wait_saved_per_carma wait_saved_vs_avg divided by mean_net_carma_spent. Only defined for
                         types that spent carma on average (NaN otherwise); unstable when the
                         spend is close to zero, so read it next to the other columns.
    """
    if "carma_spent" not in cars.columns:
        return None
    overall_wait = cars["wait_time"].mean()
    overall_loss = cars["time_loss"].mean()
    grouped = cars.groupby("kind")
    # 1.0 for cars that paid more than they received, 0.0 otherwise, and blank where the carma
    # is unknown (pooled files from before start_carma was recorded), so those stay blank
    # instead of counting as 0%.
    is_payer = (cars["carma_spent"] > 0).astype(float).where(cars["carma_spent"].notna())
    table = pd.DataFrame({
        "cars": grouped.size(),
        "mean_start_carma": grouped["start_carma"].mean(),
        "mean_net_carma_spent": grouped["carma_spent"].mean(),
        "pct_net_payers": is_payer.groupby(cars["kind"]).mean() * 100,
        "mean_wait_time": grouped["wait_time"].mean(),
        "wait_saved_vs_avg": overall_wait - grouped["wait_time"].mean(),
        "time_loss_saved_vs_avg": overall_loss - grouped["time_loss"].mean(),
    })
    spent = table["mean_net_carma_spent"]
    # where(spent > 0) turns the spend into NaN for types that did not spend, so their
    # efficiency is NaN instead of a meaningless negative number.
    table["wait_saved_per_carma"] = table["wait_saved_vs_avg"] / spent.where(spent > 0)
    return table


def build_tables(cars: pd.DataFrame, thresholds=DEFAULT_THRESHOLDS) -> dict[str, pd.DataFrame]:
    """All the tables, in the order they are printed. cars must have add_derived() applied."""
    tables = {
        "wait_time": distribution_table(cars, "wait_time"),
        "travel_time": distribution_table(cars, "travel_time"),
        "time_loss": distribution_table(cars, "time_loss"),
        "delay_ratio": distribution_table(cars, "delay_ratio"),
        "spread": spread_table(cars, thresholds),
    }
    efficiency = efficiency_table(cars)
    if efficiency is not None:
        tables["efficiency"] = efficiency
    return tables


TABLE_TITLES = {
    "wait_time": "Wait time (s)",
    "travel_time": "Travel time (s)",
    "time_loss": "Time loss (s)",
    "delay_ratio": "Delay ratio (travel time / free-flow time; 1.0 = no delay)",
    "spread": "Spread within each bidder type",
    "efficiency": "Carma efficiency",
}


def print_tables(tables: dict[str, pd.DataFrame], label: str) -> None:
    """Print every table for the run, rounded to two decimals."""
    print(f"\nRun: {label}")
    for name, table in tables.items():
        print(f"\n{TABLE_TITLES.get(name, name)}")
        print(table.round(2).to_string())


def save_tables(tables: dict[str, pd.DataFrame], label: str) -> None:
    """Write every table to analyze/tables/ as a CSV file."""
    TABLES_DIR.mkdir(exist_ok=True)
    for name, table in tables.items():
        path = TABLES_DIR / f"{safe_label(label)}_{name}.csv"
        table.round(4).to_csv(path)
    print(f"\nTables saved to {TABLES_DIR}/")


def parse_args(argv=None) -> argparse.Namespace:
    """Read the command line (cars files, thresholds, --no-save)."""
    parser = argparse.ArgumentParser(description="Delay metrics per bidder type.")
    parser.add_argument("files", nargs="*", type=Path,
                        help="*_cars.csv files (several are pooled). Default: the most recent.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS),
                        help="Wait times (s) for the share-of-cars-waiting-longer columns.")
    parser.add_argument("--no-save", action="store_true", help="Print only; don't write CSVs.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Load the cars files, add the derived columns, build the tables, print them and (unless --no-
    save) save them.
    """
    args = parse_args(argv)
    paths = args.files
    if not paths:
        paths = [latest_cars_file()]
    cars = add_derived(load_cars(paths))
    label = run_label(paths)
    if "start_carma" not in cars.columns:
        print("Note: these files have no start_carma column (older run), so the carma "
              "efficiency table is skipped.")
    tables = build_tables(cars, args.thresholds)
    print_tables(tables, label)
    if not args.no_save:
        save_tables(tables, label)


if __name__ == "__main__":
    main()
