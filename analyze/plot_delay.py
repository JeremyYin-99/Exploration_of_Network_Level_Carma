"""Plots of delay per bidder type, from the per-car results files (results/*_cars.csv).

    python analyze/plot_delay.py                      # the most recent *_cars.csv
    python analyze/plot_delay.py results/a_cars.csv   # a specific run
    python analyze/plot_delay.py a_cars.csv b_cars.csv --thresholds 30 60 120 --show

Figures are saved as PNG files in analyze/figures/ (add --show to also open them). Several
files are pooled together. The numbers behind the plots are the tables from delay_metrics.py.

Figures:
  distributions  box plots of wait time, time loss, travel time and the delay ratio.
  cdfs           cumulative distributions of wait time and delay ratio. The right-hand tail is
                 the worst-off drivers; a curve further left is better.
  percentiles    mean, median, 90th and 95th percentile of wait time and time loss.
  thresholds     share of cars that waited longer than each threshold.
  carma          net carma spent against wait time, and wait saved per carma spent.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

# Draw to files only (no window) unless --show was given, so this also works without a display.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import delay_metrics as dm

FIGURES_DIR = dm.ANALYZE_DIR / "figures"

# Panels of the distribution figure: (column, axis label).
DISTRIBUTION_PANELS = [
    ("wait_time", "Wait time (s)"),
    ("time_loss", "Time loss (s)"),
    ("travel_time", "Travel time (s)"),
    ("delay_ratio", "Delay ratio (travel time / free-flow time)"),
]

# Most points drawn per bidder type in the scatter plot, to keep it readable.
SCATTER_MAX_POINTS = 1500


def kind_colors(kinds: list[str]) -> dict[str, tuple]:
    """One fixed colour per bidder type, so a type looks the same in every figure."""
    palette = plt.get_cmap("tab10")
    colors = {}
    for index, kind in enumerate(kinds):
        colors[kind] = palette(index % 10)
    return colors


def values_by_kind(cars, kinds: list[str], column: str) -> list:
    """The values of one column for each bidder type, in the order of `kinds`."""
    data = []
    for kind in kinds:
        data.append(cars.loc[cars["kind"] == kind, column].to_numpy())
    return data


def plot_distributions(cars, kinds, colors, label):
    """Box plots of wait time, travel time, time loss and delay ratio per bidder type."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, axis_label) in zip(axes.flat, DISTRIBUTION_PANELS):
        boxes = ax.boxplot(values_by_kind(cars, kinds, column), whis=(5, 95), showfliers=False,
                           showmeans=True, patch_artist=True)
        # Colour each box by bidder type.
        for patch, kind in zip(boxes["boxes"], kinds):
            patch.set_facecolor(colors[kind])
            patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(kinds) + 1))
        ax.set_xticklabels(kinds, rotation=15)
        ax.set_ylabel(axis_label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Delay by bidder type: {label}")
    fig.text(0.5, 0.005, "Box = 25th-75th percentile, line = median, triangle = mean, "
             "whiskers = 5th-95th percentile", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def plot_cdfs(cars, kinds, colors, label, thresholds):
    """Cumulative distributions of wait time and delay ratio per bidder type, with the thresholds
    marked.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [("wait_time", "Wait time (s)"), ("delay_ratio", "Delay ratio")]
    for ax, (column, axis_label) in zip(axes, panels):
        for kind in kinds:
            values = np.sort(cars.loc[cars["kind"] == kind, column].to_numpy())
            fraction = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, fraction, label=kind, color=colors[kind])
        # Zoom to the 99.5th percentile so a few extreme cars don't squash the curves.
        # The delay ratio starts at 1.0 (no delay); the wait time starts at 0.
        left = 0.0
        if column == "delay_ratio":
            left = 1.0
        ax.set_xlim(left=left, right=cars[column].quantile(0.995))
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Fraction of cars at or below")
        ax.grid(alpha=0.3)
    # Mark the wait thresholds on the wait-time panel.
    for threshold in thresholds:
        axes[0].axvline(threshold, color="grey", linestyle="--", linewidth=0.8)
    axes[0].legend(title="Bidder type")
    fig.suptitle(f"Cumulative distribution of delay: {label}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_percentiles(cars, kinds, label):
    """Grouped bars of the mean, median, 90th and 95th percentile of wait time and time loss per
    bidder type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stats = ["mean", "median", "p90", "p95"]
    stat_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(stats)))
    bar_width = 0.8 / len(stats)
    for ax, (column, axis_label) in zip(axes, [("wait_time", "Wait time (s)"),
                                              ("time_loss", "Time loss (s)")]):
        table = dm.distribution_table(cars, column).reindex(kinds)
        positions = np.arange(len(kinds))
        for index, stat in enumerate(stats):
            bars = ax.bar(positions + (index - 1.5) * bar_width, table[stat], bar_width,
                          label=stat, color=stat_colors[index])
            ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(kinds, rotation=15)
        ax.set_ylabel(axis_label)
        ax.grid(axis="y", alpha=0.3)
    # One legend below both panels, so it doesn't cover any bars.
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(stats), title="Statistic")
    fig.suptitle(f"Mean and upper percentiles by bidder type: {label}")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    return fig


def plot_thresholds(cars, kinds, colors, label, thresholds):
    """Bars showing the share of each type's cars that waited longer than each threshold."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table = dm.spread_table(cars, thresholds).reindex(kinds)
    positions = np.arange(len(thresholds))
    bar_width = 0.8 / len(kinds)
    for index, kind in enumerate(kinds):
        shares = []
        for threshold in thresholds:
            shares.append(table.loc[kind, f"pct_wait_over_{threshold}s"])
        bars = ax.bar(positions + (index - (len(kinds) - 1) / 2) * bar_width, shares, bar_width,
                      label=kind, color=colors[kind])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(positions)
    tick_labels = []
    for threshold in thresholds:
        tick_labels.append(f"> {threshold:g} s")
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Waiting time")
    ax.set_ylabel("Share of cars (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Bidder type")
    ax.set_title(f"Cars waiting longer than each threshold: {label}")
    fig.tight_layout()
    return fig


def plot_carma(cars, kinds, colors, label):
    """Carma spent against wait time (scatter) and the carma each type had at the end."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: each car's net carma spent against how long it waited (a sample per type).
    ax = axes[0]
    for kind in kinds:
        subset = cars[cars["kind"] == kind]
        if len(subset) > SCATTER_MAX_POINTS:
            subset = subset.sample(SCATTER_MAX_POINTS, random_state=0)
        ax.scatter(subset["carma_spent"], subset["wait_time"], s=8, alpha=0.3,
                   color=colors[kind], label=kind)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Net carma spent (start - final); negative = received more than paid")
    ax.set_ylabel("Wait time (s)")
    ax.grid(alpha=0.3)
    ax.legend(title="Bidder type", markerscale=2)

    # Right: wait saved (against the average car) per carma spent, per type.
    ax = axes[1]
    table = dm.efficiency_table(cars).reindex(kinds)
    values = table["wait_saved_per_carma"].fillna(0.0)
    bar_colors = []
    for kind in kinds:
        bar_colors.append(colors[kind])
    bars = ax.bar(range(len(kinds)), values, color=bar_colors)
    # Label each bar with its value, or say why there isn't one.
    for bar, kind in zip(bars, kinds):
        efficiency = table.loc[kind, "wait_saved_per_carma"]
        spent = table.loc[kind, "mean_net_carma_spent"]
        if np.isnan(efficiency):
            text = f"n/a\n(net spend {spent:.1f})"
        else:
            text = f"{efficiency:.2f}\n(net spend {spent:.1f})"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8)
    # Leave room above the tallest bar for its label.
    top = max(values.max(), 0.0)
    ax.set_ylim(min(values.min(), 0.0) * 1.3, top * 1.3 if top > 0 else 1.0)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=15)
    ax.set_ylabel("Seconds of wait saved per carma spent")
    ax.set_title("Wait saved vs the average car, per carma spent\n"
                 "(n/a = the type did not spend carma on average)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Carma and delay: {label}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_figure(fig, label: str, name: str) -> None:
    """Save a figure to analyze/figures/ under the run's label."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{dm.safe_label(label)}_{name}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main(argv=None) -> None:
    """Load the cars files and draw and save every figure (and show them with --show)."""
    parser = argparse.ArgumentParser(description="Plots of delay per bidder type.")
    parser.add_argument("files", nargs="*", type=Path,
                        help="*_cars.csv files (several are pooled). Default: the most recent.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(dm.DEFAULT_THRESHOLDS),
                        help="Wait times (s) marked on the CDF and shown in the threshold bars.")
    parser.add_argument("--show", action="store_true", help="Also open the figures in windows.")
    args = parser.parse_args(argv)

    paths = args.files
    if not paths:
        paths = [dm.latest_cars_file()]
    cars = dm.add_derived(dm.load_cars(paths))
    label = dm.run_label(paths)
    kinds = sorted(cars["kind"].unique())
    colors = kind_colors(kinds)

    save_figure(plot_distributions(cars, kinds, colors, label), label, "distributions")
    save_figure(plot_cdfs(cars, kinds, colors, label, args.thresholds), label, "cdfs")
    save_figure(plot_percentiles(cars, kinds, label), label, "percentiles")
    save_figure(plot_thresholds(cars, kinds, colors, label, args.thresholds), label, "thresholds")
    if "carma_spent" in cars.columns:
        save_figure(plot_carma(cars, kinds, colors, label), label, "carma")
    else:
        print("No start_carma column in these files (older run): skipping the carma figure.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
