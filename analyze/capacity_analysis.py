"""Demand capacity: how much traffic each policy handles before the network jams (plan "capacity"
of controller/parameter_study.py; docs/parameter_spec.md).

Every network with 1 and 2 lanes is run at 1x to 4x moderate demand (trips per day). Policies:
baselines (actuated lights, all-way stop signs, max-pressure), fixed bidding policies (wait rule,
urgency-scaled wait, random share, 10% share) and learner A (30 learning days at every level).
Each policy then plays a 3-day evaluation on reserved trips, 5 seeds.

Terms:
  jammed     an evaluation day on which at least 100 cars were in the network and 90% or more of
             them were stopped for 300 s in a row (carma.run's jam rule); the day ends there.
  handled    a demand level at which at most half of the evaluation days (seeds x days) jammed.
  capacity   the heaviest level handled below the first level that jammed.
  completion cars that arrived / cars that entered or were waiting to enter when the day ended.

    python analyze/capacity_analysis.py          # figures to analyze/figures/capacity/ (PDF+PNG)

Writes results/analysis/capacity_days.csv (one row per evaluation day) and
results/analysis/capacity_cells.csv (one row per network, lanes, policy and level).
"""
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "params", "capacity")
TABLES = os.path.join(ROOT, "results", "analysis")
FIGURES = os.path.join(ROOT, "analyze", "figures", "capacity")

FOLDER_RE = re.compile(r"^(single|grid3|grid5)_L(\d)_moderate_t6_u80-20(_lane_c80)?(?:_x([\d.]+))?$")
DAY_RE = re.compile(r"^(\w+?)_s(\d+)_eval(\d+)_run\.csv$")

NETWORKS = [("single", "Single intersection"), ("grid3", "3x3 grid"), ("grid5", "5x5 grid")]
# Policies in plot order, grouped: baselines, fixed bidding policies, the learned policy.
POLICIES = [
    ("actuated", "actuated traffic lights", "baseline", "#52514e"),
    ("stop", "all-way stop signs", "baseline", "#898781"),
    ("maxpressure", "queue-balancing traffic lights (max-pressure)", "baseline", "#bdbcb3"),
    ("wait", "carma, bid = seconds waited", "fixed", "#b84a1c"),
    ("uwait", "carma, bid = seconds waited \u00d7 urgency", "fixed", "#eb6834"),
    ("random", "carma, random bid (share of balance)", "fixed", "#f29467"),
    ("fraction", "carma, bid 10% of balance", "fixed", "#f7c0a0"),
    ("learn", "carma, learned bids", "learned", "#2a78d6"),
]
NAME = {p: n for p, n, _, _ in POLICIES}
GROUP = {p: g for p, _, g, _ in POLICIES}
COLOR = {p: c for p, _, _, c in POLICIES}
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def scan(folder: str) -> list[dict]:
    """Every finished evaluation day in one cell or baseline folder."""
    m = FOLDER_RE.match(os.path.basename(folder))
    if not m:
        return []
    network, lanes, factor = m[1], int(m[2]), float(m[4] or 1.0)
    rows = []
    for name in os.listdir(folder):
        d = DAY_RE.match(name)
        if not d:
            continue
        run = pd.read_csv(os.path.join(folder, name)).iloc[0]
        cars_path = os.path.join(folder, name.replace("_run.csv", "_cars.csv"))
        cars = pd.read_csv(cars_path) if os.path.exists(cars_path) else pd.DataFrame()
        arrived = int(run["cars_arrived"])
        unfinished = int(run["cars_in_network"]) + int(run["not_in_network"])
        rows.append({
            "network": network, "lanes": lanes, "factor": factor, "policy": d[1],
            "seed": int(d[2]), "day": int(d[3]), "p": float(run["p"]),
            "demand_per_hour": 3600.0 / float(run["p"]),
            "jammed": str(run["jammed"]) == "True",
            "jam_time": float(run["jam_time"]) if str(run.get("jam_time", "")) not in ("", "nan") else np.nan,
            "teleports": int(run["teleports"]), "arrived": arrived,
            "completion": arrived / max(1, arrived + unfinished),
            "time_lost": float(cars["time_loss"].mean()) if len(cars) else np.nan,
        })
    return rows


def read_days() -> pd.DataFrame:
    folders = glob.glob(os.path.join(RESULTS, "*")) + glob.glob(os.path.join(RESULTS, "baselines", "*"))
    with ProcessPoolExecutor(8) as pool:
        rows = [r for rs in pool.map(scan, [f for f in folders if os.path.isdir(f)]) for r in rs]
    return pd.DataFrame(rows)


def cells(days: pd.DataFrame) -> pd.DataFrame:
    """One row per network, lanes, policy and level: jam share, completion, time lost."""
    g = days.groupby(["network", "lanes", "policy", "factor"])
    out = g.agg(demand_per_hour=("demand_per_hour", "mean"), days=("jammed", "size"),
                seeds=("seed", "nunique"), jam_share=("jammed", "mean"),
                completion=("completion", "mean"), teleports=("teleports", "mean")).reset_index()
    ok = days[~days["jammed"]].groupby(["network", "lanes", "policy", "factor"])["time_lost"]
    out = out.merge(ok.mean().rename("time_lost_unjammed").reset_index(), how="left")
    out["handled"] = out["jam_share"] <= 0.5
    return out


def capacity(table: pd.DataFrame) -> pd.DataFrame:
    """Per network, lanes and policy: the heaviest level handled below the first jammed level."""
    rows = []
    for (network, lanes, policy), g in table.groupby(["network", "lanes", "policy"]):
        g = g.sort_values("factor")
        jammed = g[~g["handled"]]
        first_jam = jammed["factor"].min() if len(jammed) else np.nan
        below = g[g["factor"] < first_jam] if len(jammed) else g
        rows.append({"network": network, "lanes": lanes, "policy": policy,
                     "capacity_factor": below["factor"].max() if len(below) else np.nan,
                     "capacity_per_hour": below["demand_per_hour"].max() if len(below) else np.nan,
                     "first_jam_factor": first_jam,
                     "first_jam_per_hour": jammed["demand_per_hour"].min() if len(jammed) else np.nan,
                     "levels": g["factor"].nunique(), "max_factor": g["factor"].max(),
                     "max_per_hour": g["demand_per_hour"].max()})
    return pd.DataFrame(rows)


def save(fig, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES, f"{name}.png"), dpi=300)
    plt.close(fig)


def moderate_per_hour(table: pd.DataFrame, network: str, lanes: int) -> float:
    m = table[(table["network"] == network) & (table["lanes"] == lanes) & (table["factor"] == 1.0)]
    return float(m["demand_per_hour"].mean()) if len(m) else np.nan


def demand_axis(ax, moderate: float, label_moderate: bool) -> None:
    """Demand in vehicles per hour on x (thousands separators), with the moderate level marked."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    if np.isfinite(moderate):
        ax.axvline(moderate, color="#8a897f", lw=0.8, ls=":", zorder=0)
        if label_moderate:
            ax.text(moderate, 1.0, " moderate", transform=ax.get_xaxis_transform(), fontsize=7,
                    color=INK2, va="top", ha="left")


def fig_capacity(table: pd.DataFrame, cap: pd.DataFrame) -> None:
    """Bars: heaviest demand handled without a jam; X: the first level that jammed; arrow: no jam
    up to the heaviest level run."""
    policies = [p for p, _, _, _ in POLICIES if p in set(cap["policy"])]
    ys = {p: -(i + 0.6 * ["baseline", "fixed", "learned"].index(GROUP[p])) for i, p in enumerate(policies)}
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 7.2), sharey=True)
    for row, (network, title) in enumerate(NETWORKS):
        for col, lanes in enumerate((1, 2)):
            ax = axes[row][col]
            c = cap[(cap["network"] == network) & (cap["lanes"] == lanes)].set_index("policy")
            top = float(c["max_per_hour"].max()) if len(c) else 1.0
            xmax = top * 1.15
            for p in policies:
                if p not in c.index:
                    continue
                r = c.loc[p]
                y = ys[p]
                if np.isfinite(r["capacity_per_hour"]):
                    ax.barh(y, r["capacity_per_hour"], height=0.7, color=COLOR[p], linewidth=0)
                if np.isfinite(r["first_jam_per_hour"]):
                    ax.scatter(r["first_jam_per_hour"], y, marker="X", s=28, color="#c8413b", zorder=3,
                               linewidths=0)
                else:
                    ax.annotate("", xy=(r["max_per_hour"] + 0.11 * top, y),
                                xytext=(r["max_per_hour"] + 0.01 * top, y),
                                arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.8})
            ax.set_title(f"{title}, {lanes} lane{'s' if lanes > 1 else ''}", loc="left")
            ax.set_xlim(0, xmax)
            demand_axis(ax, moderate_per_hour(table, network, lanes), label_moderate=True)
            ax.grid(True, axis="x", color="#e1e0d9", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", length=0)
    axes[0][0].set_yticks([ys[p] for p in policies])
    axes[0][0].set_yticklabels([NAME[p] for p in policies])
    for ax in axes[-1]:
        ax.set_xlabel("demand (vehicles per hour)")
    fig.text(0.0, -0.02, "Bars: heaviest demand handled without a jam. Red X: first demand level "
             "that jammed (most of 5 seeds x 3 days). Arrow: no jam up to the heaviest level run.\n"
             "Dotted line: moderate demand (the study's reference level). Levels run: 1, 1.25, 1.5, "
             "1.75, 2, 2.5, 3 and 4 times moderate.", fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "demand_capacity")


def fig_curves(table: pd.DataFrame) -> None:
    """Time lost per trip (days that did not jam) against demand; X where most days jammed."""
    styles = {"baseline": "--", "fixed": "-", "learned": "-"}
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 7.0))
    for row, (network, title) in enumerate(NETWORKS):
        for col, lanes in enumerate((1, 2)):
            ax = axes[row][col]
            t = table[(table["network"] == network) & (table["lanes"] == lanes)]
            for p, name, group, color in POLICIES:
                g = t[t["policy"] == p].sort_values("factor")
                if not len(g):
                    continue
                ok = g[g["handled"]]
                ax.plot(ok["demand_per_hour"], ok["time_lost_unjammed"], styles[group], color=color,
                        lw=2.0 if group == "learned" else 1.2, marker="o", ms=2.5, label=name)
                bad = g[~g["handled"]]
                if len(bad) and len(ok):
                    first_bad = bad.sort_values("factor").iloc[0]["demand_per_hour"]
                    ax.scatter([first_bad], [ok["time_lost_unjammed"].iloc[-1]], marker="X",
                               s=22, color=color, zorder=3, linewidths=0)
            ax.set_title(f"{title}, {lanes} lane{'s' if lanes > 1 else ''}", loc="left")
            if len(t):
                # From the lightest level to the last point or X drawn (levels past every policy's
                # first jam show nothing).
                drawn = []
                for p in t["policy"].unique():
                    g = t[t["policy"] == p].sort_values("factor")
                    bad = g[~g["handled"]]
                    drawn.append(bad.iloc[0]["demand_per_hour"] if len(bad) else g["demand_per_hour"].max())
                lo, hi = float(t["demand_per_hour"].min()), float(max(drawn))
                ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
            demand_axis(ax, moderate_per_hour(table, network, lanes), label_moderate=True)
            ax.set_ylim(bottom=0)
            ax.grid(True, color="#e1e0d9", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("time lost per trip (s)")
    for ax in axes[-1]:
        ax.set_xlabel("demand (vehicles per hour)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.11))
    fig.text(0.0, -0.13, "Lines: mean time lost over days that did not jam. X: the next level, at "
             "which most days jammed (drawn at the last handled value).", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, "demand_curves")


if __name__ == "__main__":
    days = read_days()
    if days.empty:
        raise SystemExit("No finished evaluation days yet in results/params/capacity/.")
    os.makedirs(TABLES, exist_ok=True)
    days.to_csv(os.path.join(TABLES, "capacity_days.csv"), index=False)
    table = cells(days)
    table.round(4).to_csv(os.path.join(TABLES, "capacity_cells.csv"), index=False)
    cap = capacity(table)
    cap.round(3).to_csv(os.path.join(TABLES, "capacity_summary.csv"), index=False)
    fig_capacity(table, cap)
    fig_curves(table)
    pd.set_option("display.width", 200)
    print(cap.to_string(index=False))
    print(f"Figures: {FIGURES}")
