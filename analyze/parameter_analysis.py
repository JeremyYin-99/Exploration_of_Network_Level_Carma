"""Analysis of the parameter study (docs/parameter_spec.md): one results folder in, every table
and figure out.

    python analyze/parameter_analysis.py                 # plan "main"
    python analyze/parameter_analysis.py --plan smoke

Reads results/params/<plan>/:
  <cell>/learn_s<seed>_day<d>_prices.csv     price per intersection per learning day
  <cell>/<strategy>_s<seed>_eval<e>_cars.csv  every trip of the evaluation week
  baselines/<key>/...                         the same for actuated lights and stop signs

Writes
  results/analysis/params_<plan>_cells.csv          one row per cell and strategy
  results/analysis/params_<plan>_prices_by_day.csv  network price per chain and day
  analyze/figures/params/<plan>/*.png|pdf

Conventions. Price = median winning bid of contested winners, % of starting carma, mean over
intersections. Evaluation-week numbers are averaged over the 7 days per seed; intervals are 95%
across seeds (t with seeds - 1 degrees of freedom). Differences from a baseline are paired by
seed: (strategy - baseline) / baseline per seed, then the mean and interval across seeds.
Settled: the first 30-day block k such that the changes into blocks k and k+1 both have
intervals that include 0 and means within 3% of the price (spec, "Seeds and learning length").
"""
import argparse
import glob
import os
import pickle
import re
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "params")
TABLES = os.path.join(ROOT, "results", "analysis")
FIGURES = os.path.join(ROOT, "analyze", "figures", "params")
DPI = 300
BLOCK = 30

CELL_RE = re.compile(r"^(single|grid3|grid5)_L(\d)_(light|moderate|heavy)_t(\d+)_u(\d+)-(\d+)(d?)"
                     r"_(vehicle|lane|phase)(_ext)?_c(\d+)(_sumodrv)?$")
BASE_RE = re.compile(r"^(single|grid3|grid5)_L(\d)_(light|moderate|heavy)_t(\d+)_u(\d+)-(\d+)(d?)(_sumodrv)?$")
CHAIN_RE = re.compile(r"^(\w+?)_s(\d+)_(day|eval)(\d+)_(prices|cars)\.csv$")

NETWORKS = ["single", "grid3", "grid5"]
NETWORK_NAMES = {"single": "single intersection", "grid3": "3x3 grid", "grid5": "5x5 grid"}
REFERENCE = {"demand": "moderate", "tpd": 6, "low": 80, "mode": "trip", "bidding": "lane",
             "extend": False, "carma": 80, "drivers": "lc"}

# Colours (reference palette): lanes as a blue ramp; strategies blue / orange / grey ink.
LANE_COLORS = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
STRATEGY_COLORS = {"learn": "#2a78d6", "wait": "#eb6834", "actuated": "#52514e", "stop": "#898781",
                   "maxpressure": "#bdbcb3"}
STRATEGY_NAMES = {"learn": "carma, learned bids", "wait": "carma, bid = seconds waited",
                  "actuated": "actuated traffic lights", "stop": "all-way stop signs",
                  "maxpressure": "queue-balancing traffic lights (max-pressure)"}
BASELINES = ["actuated", "stop", "maxpressure"]


def t975(df: int) -> float:
    """97.5% point of Student's t."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
             9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 19: 2.093, 29: 2.045}
    if df in table:
        return table[df]
    return 1.96 if df > 29 else table[min(table, key=lambda k: abs(k - df))]


def interval(values) -> tuple[float, float, float]:
    """Mean and 95% interval across independent values."""
    v = pd.Series(values, dtype=float).dropna()
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v.iloc[0]), np.nan, np.nan
    half = t975(len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def gini(values) -> float:
    """Gini coefficient of non-negative values."""
    x = np.sort(np.asarray(pd.Series(values, dtype=float).dropna()))
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def settings_of(folder: str) -> dict | None:
    """Settings from a cell (or baseline) folder name."""
    name = os.path.basename(folder)
    m = CELL_RE.match(name)
    if m:
        return {"cell": name, "network": m[1], "lanes": int(m[2]), "demand": m[3],
                "tpd": int(m[4]), "low": int(m[5]), "mode": "driver" if m[7] else "trip",
                "bidding": m[8], "extend": bool(m[9]), "carma": int(m[10]),
                "drivers": "sumo" if m[11] else "lc",
                "baseline_key": f"{m[1]}_L{m[2]}_{m[3]}_t{m[4]}_u{m[5]}-{m[6]}{m[7]}{m[11] or ''}"}
    m = BASE_RE.match(name)
    if m:
        return {"cell": "", "baseline_key": name, "network": m[1], "lanes": int(m[2]),
                "demand": m[3], "tpd": int(m[4]), "low": int(m[5]),
                "mode": "driver" if m[7] else "trip", "drivers": "sumo" if m[8] else "lc"}
    return None


# ---- Reading one folder (cached) ----------------------------------------------------------------

def read_folder(folder: str) -> dict:
    """Prices per day and evaluation-week rows of every chain in one folder. Cached in
    results/analysis/params_cache/<plan>/, refreshed when the folder's file count changes."""
    files = os.listdir(folder)
    parent = os.path.dirname(folder)
    plan = os.path.basename(parent)
    if plan == "baselines":
        plan = os.path.basename(os.path.dirname(parent)) + "_baselines"
    cache_dir = os.path.join(TABLES, "params_cache", plan)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, os.path.basename(folder) + ".pkl")
    stamp = len(files)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if data.get("stamp") == stamp:
            return data
    price_rows, eval_rows = [], []
    for name in files:
        m = CHAIN_RE.match(name)
        if not m:
            continue
        strategy, seed, phase, day, kind = m[1], int(m[2]), m[3], int(m[4]), m[5]
        path = os.path.join(folder, name)
        if kind == "prices":
            if os.path.getsize(path) <= 1:
                continue
            pr = pd.read_csv(path, usecols=["scope", "win_rel_q50", "share_contested",
                                            "bid_rel_mean"])
            pr = pr[pr["scope"] == "day"]
            if len(pr):
                price_rows.append({"strategy": strategy, "seed": seed, "phase": phase, "day": day,
                                   "price": 100 * pr["win_rel_q50"].mean(),
                                   "contested": pr["share_contested"].mean(),
                                   "bid_mean": 100 * pr["bid_rel_mean"].mean()})
        elif kind == "cars" and phase == "eval":
            cars = pd.read_csv(path, usecols=["driver", "time_loss", "wait_time", "urgency"])
            cars["cost"] = cars["urgency"] * cars["time_loss"]
            urgent = cars["urgency"] > 1
            drivers = path.replace("_cars.csv", "_drivers.csv")
            balance_gini = np.nan
            if os.path.exists(drivers):
                balance_gini = gini(pd.read_csv(drivers, usecols=["final_carma"])["final_carma"])
            per_driver = cars.groupby("driver")["cost"].agg(["sum", "count"])
            eval_rows.append({
                "strategy": strategy, "seed": seed, "eval_day": day, "trips": len(cars),
                "time_lost": cars["time_loss"].mean(), "wait": cars["wait_time"].mean(),
                "cost": cars["cost"].mean(),
                "time_lost_urgent": cars.loc[urgent, "time_loss"].mean(),
                "time_lost_calm": cars.loc[~urgent, "time_loss"].mean(),
                "balance_gini": balance_gini, "per_driver": per_driver})
    data = {"stamp": stamp, "prices": pd.DataFrame(price_rows), "eval": eval_rows}
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def load(plan: str, workers: int) -> tuple[pd.DataFrame, list]:
    """(prices per chain and day, evaluation-week rows) for every folder of a plan."""
    base = os.path.join(RESULTS, plan)
    folders = [f for f in glob.glob(os.path.join(base, "*")) if settings_of(f)]
    folders += [f for f in glob.glob(os.path.join(base, "baselines", "*")) if settings_of(f)]
    with Pool(workers) as pool:
        results = pool.map(read_folder, folders, chunksize=1)
    prices, evals = [], []
    for folder, data in zip(folders, results):
        s = settings_of(folder)
        if len(data["prices"]):
            prices.append(data["prices"].assign(**s))
        for row in data["eval"]:
            evals.append({**s, **row})
    prices = pd.concat(prices, ignore_index=True) if prices else pd.DataFrame()
    return prices, evals


# ---- Tables ------------------------------------------------------------------------------------

def week_per_seed(evals: list) -> pd.DataFrame:
    """Evaluation week per (folder, strategy, seed): mean of the daily numbers, and the Gini of
    per-driver cost (each driver's mean cost per trip over the week)."""
    groups = {}
    for row in evals:
        key = (row["cell"], row["baseline_key"], row["strategy"], row["seed"])
        groups.setdefault(key, []).append(row)
    out = []
    for (cell, bkey, strategy, seed), rows in groups.items():
        per_driver = pd.concat([r["per_driver"] for r in rows]).groupby(level=0).sum()
        g = pd.DataFrame([{k: v for k, v in r.items() if k != "per_driver"} for r in rows])
        out.append({"cell": cell, "baseline_key": bkey, "strategy": strategy, "seed": seed,
                    "days": len(rows), "trips": g["trips"].mean(),
                    "time_lost": g["time_lost"].mean(), "wait": g["wait"].mean(),
                    "cost": g["cost"].mean(), "time_lost_urgent": g["time_lost_urgent"].mean(),
                    "time_lost_calm": g["time_lost_calm"].mean(),
                    "cost_gini": gini(per_driver["sum"] / per_driver["count"]),
                    "balance_gini": g["balance_gini"].mean()})
    return pd.DataFrame(out)


def settling(prices: pd.DataFrame) -> pd.DataFrame:
    """Per learner cell: the settled day and the last block's price and change."""
    rows = []
    learn = prices[(prices["strategy"] == "learn") & (prices["phase"] == "day")]
    for cell, g in learn.groupby("cell"):
        g = g.assign(block=(g["day"] - 1) // BLOCK)
        days_per = g.groupby(["seed", "block"])["day"].nunique()
        full = days_per[days_per == BLOCK].index
        if len(full) == 0:
            continue
        means = g.groupby(["seed", "block"])["price"].mean().loc[full].unstack("block")
        means = means.dropna(axis=1, thresh=max(2, int(0.8 * len(means))))
        blocks = list(means.columns)
        if len(blocks) < 2:
            continue
        changes = {b: interval(means[b] - means[a]) for a, b in zip(blocks[:-1], blocks[1:])}
        settled = None
        for i in range(1, len(blocks) - 1):
            ok = True
            for b in (blocks[i], blocks[i + 1]):
                mean, lo, hi = changes[b]
                if not (lo <= 0 <= hi and abs(mean) <= 0.03 * means[b].mean()):
                    ok = False
            if ok:
                settled = blocks[i]
                break
        last = blocks[-1]
        m, lo, hi = interval(means[last])
        rows.append({"cell": cell, "learn_seeds": means.shape[0], "blocks": len(blocks),
                     "settled_day": np.nan if settled is None else int(settled * BLOCK + 1),
                     "price_last_block": m, "price_last_lo": lo, "price_last_hi": hi,
                     "change_last": changes[last][0], "change_last_lo": changes[last][1],
                     "change_last_hi": changes[last][2],
                     "price_settled": (means[[b for b in blocks if b >= settled]].mean(axis=1)
                                       .mean() if settled is not None else np.nan)})
    return pd.DataFrame(rows)


def cell_table(prices: pd.DataFrame, evals: list) -> pd.DataFrame:
    """One row per (cell, strategy): settings, price and settling (learner A), evaluation week,
    and the paired differences from actuated lights and stop signs."""
    week = week_per_seed(evals)
    base = week[week["strategy"].isin(BASELINES)]
    bids = week[~week["strategy"].isin(BASELINES)]
    settle = settling(prices) if len(prices) else pd.DataFrame(columns=["cell"])
    eval_price = (prices[prices["phase"] == "eval"].groupby(["cell", "strategy"])[
        ["price", "contested", "bid_mean"]].mean() if len(prices) else pd.DataFrame())
    rows = []
    for (cell, strategy), g in bids.groupby(["cell", "strategy"]):
        s = settings_of(cell)
        row = {**s, "strategy": strategy, "seeds": len(g)}
        for col in ("time_lost", "wait", "cost", "time_lost_urgent", "time_lost_calm",
                    "cost_gini", "balance_gini"):
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col])
        if (cell, strategy) in eval_price.index:
            row.update(eval_price.loc[(cell, strategy)].add_prefix("eval_").to_dict())
        for ref in BASELINES:
            r = base[(base["baseline_key"] == s["baseline_key"]) & (base["strategy"] == ref)]
            pair = g.merge(r, on="seed", suffixes=("", "_ref"))
            for col in ("time_lost", "wait", "cost"):
                pct = 100 * (pair[col] - pair[col + "_ref"]) / pair[col + "_ref"]
                m, lo, hi = interval(pct)
                row[f"{col}_vs_{ref}"] = m
                row[f"{col}_vs_{ref}_lo"], row[f"{col}_vs_{ref}_hi"] = lo, hi
            for col in ("time_lost", "wait", "cost", "cost_gini", "time_lost_urgent",
                        "time_lost_calm"):
                row[f"{ref}_{col}"] = r[col].mean() if len(r) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) and len(settle):
        table = table.merge(settle, on="cell", how="left")
        cols = [c for c in settle.columns if c != "cell"]
        table.loc[table["strategy"] != "learn", cols] = np.nan
    return table


# ---- Figures -----------------------------------------------------------------------------------

def _style(ax) -> None:
    """Recessive axes: light grid, no top/right spines."""
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, plan: str, name: str) -> None:
    """Save a figure as PNG and PDF."""
    out = os.path.join(FIGURES, plan)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=DPI)
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def lanes_label(lanes: int) -> str:
    """'1 lane', '2 lanes', ..."""
    return f"{lanes} lane{'s' if lanes > 1 else ''}"


# The one-setting-at-a-time sweeps: (axis label, column, levels, other settings fixed).
SWEEPS = [
    ("starting carma", "carma", [5, 10, 20, 40, 80, 160, 320], {}),
    ("urgency per trip: % of trips with low urgency", "low", [20, 40, 50, 60, 80],
     {"mode": "trip"}),
    ("urgency per driver: % of drivers with low urgency", "low", [20, 40, 50, 60, 80],
     {"mode": "driver"}),
    ("demand (6 trips per driver)", "demand", ["light", "moderate", "heavy"], {"tpd": 6}),
    ("trips per driver per day (moderate demand)", "tpd", [2, 6, 10], {"demand": "moderate"}),
    ("bidding unit, no green extension", "bidding", ["vehicle", "lane", "phase"],
     {"extend": False}),
    ("bidding unit, with green extension", "bidding", ["vehicle", "lane", "phase"],
     {"extend": True}),
]


def sweep_rows(table: pd.DataFrame, column: str, fixed: dict) -> pd.DataFrame:
    """The cells of one sweep: every setting at the reference except `column` (and `fixed`)."""
    keep = pd.Series(True, index=table.index)
    for key, value in {**REFERENCE, **fixed}.items():
        if key != column:
            keep &= table[key] == value
    return table[keep]


def reference_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the reference cell of each network configuration."""
    out = frame
    for key, value in REFERENCE.items():
        if key in out:
            out = out[out[key] == value]
    return out


def sweep_figure(table: pd.DataFrame, plan: str, value: str, ylabel: str, name: str,
                 title: str, zero_line: bool = False, strategy: str = "learn") -> None:
    """Small multiples: one row per sweep, one column per network; one line per lane count."""
    t = table[table["strategy"] == strategy]
    if t.empty or value not in t:
        return
    networks = [n for n in NETWORKS if n in set(t["network"])]
    fig, axes = plt.subplots(len(SWEEPS), len(networks),
                             figsize=(4.3 * len(networks) + 0.6, 2.3 * len(SWEEPS)),
                             squeeze=False, sharey="row")
    for i, (label, column, levels, fixed) in enumerate(SWEEPS):
        for j, network in enumerate(networks):
            ax = axes[i][j]
            _style(ax)
            rows = sweep_rows(t[t["network"] == network], column, fixed)
            x = np.arange(len(levels))
            for lanes in (1, 2, 3):
                r = rows[rows["lanes"] == lanes].set_index(column).reindex(levels)
                if r.empty or r[value].notna().sum() == 0:
                    continue
                ax.plot(x, r[value].astype(float), color=LANE_COLORS[lanes], lw=1.8, marker="o",
                        ms=4, label=lanes_label(lanes))
                lo, hi = f"{value}_lo", f"{value}_hi"
                if lo in r and r[lo].notna().any():
                    ax.fill_between(x, r[lo].astype(float), r[hi].astype(float),
                                    color=LANE_COLORS[lanes], alpha=0.15, lw=0)
            if zero_line:
                ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in levels], fontsize=8)
            if i == 0:
                ax.set_title(NETWORK_NAMES[network], fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=INK2)
            ax.set_xlabel(label, fontsize=8, color=INK2)
    handles, labels = [], []
    for row in axes:
        for ax in row:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, plan, name)


def heatmap_figure(table: pd.DataFrame, plan: str, value: str, name: str, title: str,
                   cbar: str, diverging: bool = False) -> None:
    """Demand x trips per driver, one panel per network configuration (learner A)."""
    t = table[table["strategy"] == "learn"]
    if t.empty or value not in t:
        return
    demands, tpds = ["light", "moderate", "heavy"], [2, 6, 10]
    grids = {}
    for n in NETWORKS:
        for lanes in (1, 2, 3):
            rows = t[(t["network"] == n) & (t["lanes"] == lanes) & (t["low"] == 80) &
                     (t["mode"] == "trip") & (t["bidding"] == "lane") & (~t["extend"]) &
                     (t["carma"] == 80)]
            if rows.empty:
                continue
            grids[(n, lanes)] = rows.pivot_table(index="tpd", columns="demand", values=value) \
                .reindex(index=tpds, columns=demands)
    if not grids:
        return
    values = np.concatenate([g.values.astype(float).ravel() for g in grids.values()])
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    if diverging:
        lim = float(np.nanmax(np.abs(values)))
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    else:
        vmin, vmax, cmap = float(values.min()), float(values.max()), "Blues"
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), squeeze=False)
    im = None
    for i, n in enumerate(NETWORKS):
        for lanes in (1, 2, 3):
            ax = axes[i][lanes - 1]
            grid = grids.get((n, lanes))
            if grid is None:
                ax.axis("off")
                continue
            data = grid.values.astype(float)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            for (a, b), v in np.ndenumerate(data):
                if not np.isnan(v):
                    dark = abs(v - (vmin + vmax) / 2) > 0.3 * (vmax - vmin)
                    ax.text(b, a, f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}", ha="center",
                            va="center", fontsize=8, color="white" if dark else INK)
            ax.set_xticks(range(3))
            ax.set_xticklabels(demands, fontsize=8)
            ax.set_yticks(range(3))
            ax.set_yticklabels(tpds, fontsize=8)
            ax.set_title(f"{NETWORK_NAMES[n]}, {lanes_label(lanes)}", fontsize=9, loc="left",
                         color=INK)
            ax.set_xlabel("demand", fontsize=8, color=INK2)
            ax.set_ylabel("trips per driver per day", fontsize=8, color=INK2)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=cbar)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    save(fig, plan, name)


def settling_figure(prices: pd.DataFrame, plan: str) -> None:
    """Price over learning days for each network configuration's reference cell."""
    learn = prices[(prices["strategy"] == "learn") & (prices["phase"] == "day")]
    ref = reference_rows(learn)
    if ref.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, n in zip(axes, NETWORKS):
        _style(ax)
        for lanes in (1, 2, 3):
            g = ref[(ref["network"] == n) & (ref["lanes"] == lanes)]
            if g.empty:
                continue
            daily = g.groupby("day")["price"].mean()
            smooth = daily.rolling(7, min_periods=1, center=True).mean()
            ax.plot(smooth.index, smooth.values, color=LANE_COLORS[lanes], lw=1.6,
                    label=lanes_label(lanes))
        ax.set_title(NETWORK_NAMES[n], fontsize=10, loc="left", color=INK)
        ax.set_xlabel("learning day", fontsize=8, color=INK2)
    axes[0].set_ylabel("price (% of starting carma),\n7-day mean over seeds", fontsize=8,
                       color=INK2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Price of priority with learned bids over the learning days, reference cell of "
                 "each network",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, plan, "settling_reference")


def strategies_figure(table: pd.DataFrame, plan: str, value: str = "time_lost") -> None:
    """Reference cell of each network configuration: learner A, the wait rule and the
    baselines side by side."""
    ref = reference_rows(table)
    if ref.empty:
        return
    labels = {"time_lost": "time lost per trip (s)", "cost": "urgency-weighted cost per trip (s)",
              "wait": "waiting time per trip (s)"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, n in zip(axes, NETWORKS):
        _style(ax)
        rows = ref[ref["network"] == n]
        x = np.arange(3)
        width = 0.16
        for k, strategy in enumerate(["learn", "wait"] + BASELINES):
            ys, los, his = [], [], []
            for lanes in (1, 2, 3):
                r = rows[rows["lanes"] == lanes]
                if strategy in BASELINES:
                    ys.append(r[f"{strategy}_{value}"].mean() if len(r) else np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
                else:
                    rr = r[r["strategy"] == strategy]
                    ys.append(rr[value].mean() if len(rr) else np.nan)
                    los.append(rr[f"{value}_lo"].mean() if len(rr) else np.nan)
                    his.append(rr[f"{value}_hi"].mean() if len(rr) else np.nan)
            xs = x + (k - 2) * width
            ax.bar(xs, ys, width * 0.92, color=STRATEGY_COLORS[strategy],
                   label=STRATEGY_NAMES[strategy])
            err = np.array([np.subtract(ys, los), np.subtract(his, ys)], dtype=float)
            if np.isfinite(err).any():
                ax.errorbar(xs, ys, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([lanes_label(lanes) for lanes in (1, 2, 3)], fontsize=8)
        ax.set_title(NETWORK_NAMES[n], fontsize=10, loc="left", color=INK)
    axes[0].set_ylabel(f"{labels[value]}, evaluation week", fontsize=8, color=INK2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{labels[value][0].upper() + labels[value][1:]} in the reference cell of each "
                 "network configuration", x=0.01, ha="left", fontsize=12, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, plan, f"reference_{value}")


def fairness_figure(table: pd.DataFrame, plan: str) -> None:
    """Gini of per-driver cost against mean cost: learner A and actuated lights, every cell."""
    t = table[table["strategy"] == "learn"]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _style(ax)
    markers = {"single": "o", "grid3": "s", "grid5": "^"}
    for n in NETWORKS:
        r = t[t["network"] == n]
        if r.empty:
            continue
        ax.scatter(r["cost"], r["cost_gini"], s=18, color=STRATEGY_COLORS["learn"], alpha=0.6,
                   marker=markers[n], label=f"carma, learned bids, {NETWORK_NAMES[n]}")
        ax.scatter(r["actuated_cost"], r["actuated_cost_gini"], s=18,
                   color=STRATEGY_COLORS["actuated"], alpha=0.6, marker=markers[n],
                   label=f"actuated traffic lights, {NETWORK_NAMES[n]}")
    ax.set_xlabel("urgency-weighted cost per trip (s), evaluation week", fontsize=9, color=INK2)
    ax.set_ylabel("Gini of per-driver cost", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("Fairness against efficiency, every cell", fontsize=12, loc="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, plan, "fairness_vs_cost")


def main() -> None:
    """Load a plan's results, write the tables and draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--plan", default="main")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    prices, evals = load(args.plan, args.workers)
    os.makedirs(TABLES, exist_ok=True)
    if len(prices):
        daily = prices.groupby(["cell", "strategy", "seed", "phase", "day"])[
            ["price", "contested", "bid_mean"]].mean().reset_index()
        daily.to_csv(os.path.join(TABLES, f"params_{args.plan}_prices_by_day.csv"), index=False)
    if not evals:
        print("No evaluation weeks finished yet.")
        return
    table = cell_table(prices, evals)
    table.round(4).to_csv(os.path.join(TABLES, f"params_{args.plan}_cells.csv"), index=False)
    print(f"{table['cell'].nunique()} cells with a finished evaluation week; table: "
          f"results/analysis/params_{args.plan}_cells.csv")
    cols = ["network", "lanes", "strategy", "seeds", "time_lost", "time_lost_vs_actuated",
            "time_lost_vs_actuated_lo", "time_lost_vs_actuated_hi", "cost_vs_actuated",
            "time_lost_vs_stop", "eval_price", "eval_contested", "cost_gini",
            "actuated_cost_gini", "settled_day", "price_last_block"]
    ref = reference_rows(table)
    print(ref[[c for c in cols if c in ref.columns]].round(2).to_string(index=False))
    # Driver-model sensitivity: each reference cell with the study's drivers and SUMO's default.
    if "drivers" in table and (table["drivers"] == "sumo").any():
        both = table
        for key, value in REFERENCE.items():
            if key != "drivers":
                both = both[both[key] == value]
        dcols = ["network", "lanes", "drivers", "strategy", "seeds", "time_lost",
                 "actuated_time_lost", "stop_time_lost", "time_lost_vs_actuated",
                 "time_lost_vs_actuated_lo", "time_lost_vs_actuated_hi", "cost_vs_actuated"]
        both = both[[c for c in dcols if c in both.columns]].sort_values(
            ["network", "lanes", "strategy", "drivers"])
        both.round(4).to_csv(os.path.join(TABLES, f"params_{args.plan}_drivers.csv"), index=False)
        print("\nDriver-model check (lc = study drivers, sumo = SUMO default):")
        print(both.round(2).to_string(index=False))
    if len(prices):
        settling_figure(prices, args.plan)
    for value in ("time_lost", "cost", "wait"):
        strategies_figure(table, args.plan, value)
    sweep_figure(table, args.plan, "price_last_block", "price (% of starting carma),\nlast block",
                 "sweep_price", "Price of priority with learned bids in each setting (last 30 learning days)")
    sweep_figure(table, args.plan, "time_lost_vs_actuated", "time lost vs actuated (%)",
                 "sweep_time_lost_vs_actuated", "Carma with learned bids: time lost per trip compared with "
                 "actuated lights (negative = carma better)", zero_line=True)
    sweep_figure(table, args.plan, "cost_vs_actuated", "cost vs actuated (%)",
                 "sweep_cost_vs_actuated", "Carma with learned bids: urgency-weighted cost compared with "
                 "actuated lights (negative = carma better)", zero_line=True)
    sweep_figure(table, args.plan, "time_lost_vs_stop", "time lost vs stop signs (%)",
                 "sweep_time_lost_vs_stop", "Carma with learned bids: time lost per trip compared with stop "
                 "signs (negative = carma better)", zero_line=True)
    sweep_figure(table, args.plan, "eval_contested", "share of auctions contested",
                 "sweep_contested", "Share of auctions that were contested (evaluation week)")
    sweep_figure(table, args.plan, "cost_gini", "Gini of per-driver cost", "sweep_gini",
                 "Fairness: Gini of per-driver cost (evaluation week, carma with learned bids)")
    heatmap_figure(table, args.plan, "price_last_block", "heat_price",
                   "Price of priority with learned bids, by demand and trips per driver (% of starting carma)",
                   "price (% of starting carma)")
    heatmap_figure(table, args.plan, "time_lost_vs_actuated", "heat_time_lost_vs_actuated",
                   "Time lost compared with actuated lights, by demand and trips per driver (%)",
                   "time lost vs actuated (%)", diverging=True)
    fairness_figure(table, args.plan)
    print(f"Figures: analyze/figures/params/{args.plan}/")


if __name__ == "__main__":
    main()
