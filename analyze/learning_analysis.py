"""Learning study results (docs/learning_spec.md, S12): what drivers actually bid at every
intersection, how the price of priority evolves while they learn, whether and when the learners
converged, and what it does to travel times. Works on chains that are still running (it reads what
has been written so far).

Units: every amount is in carma. Every driver starts with 100 carma and the total never changes
(payments go to the losers), so 100 carma is always the average balance. The "price" of an
intersection is the winning bid in its contested auctions (two or more bidders); an uncontested
winner pays nothing whatever it bid.

Run from the repo root:
    python analyze/learning_analysis.py                       # every scenario found
    python analyze/learning_analysis.py --scenarios grid3     # one scenario
    python analyze/learning_analysis.py --no-gif              # skip the animations (faster)

Inputs: results/<results>/<scenario>/ written by controller/learning_study.py.
Outputs (figures as PNG and PDF in analyze/figures/learning/, tables in results/analysis/):

Per scenario (<scenario>_...):
  _bids_by_intersection  mean bid (number in every cell) by intersection and the bidder's balance,
                         one panel per learner, over the last 14 learning days
  _bid_distribution      box plots of the bids made at every intersection, per learner, with the
                         median price (winning bid) marked and written
  _learners              the price by day for every learner and seed, one panel per junction type,
                         final values written at the line ends
  _empirical_policy      mean bid (numbers in the cells) by balance and intersections left
Per learning chain (<scenario>_<strategy>_s<seed>_...):
  _prices_grid           one panel per intersection, laid out like the network: the daily median
                         price with its 25-75% and 10-90% bands, the median bid, and the last-14-day
                         numbers written in each panel
  _prices_weekly         the median price at every intersection, week by week, numbers in cells
  _prices_map            the network with the price written at every intersection, on day 1, 10,
                         30 and the last learning day
  _prices_map.gif        animation: the map with numbers, next to every intersection's price curve
                         up to that day
  _within_day            the price during the day (5-minute bins) for a few days
  _convergence           the section 7 metrics by day with their thresholds
  _policy                the learned bid (numbers in the cells) by balance and intersections left,
                         and, for learner A, the learned value of carma
  _belief                learner A: the rival bids it expects against the real ones
Across scenarios:
  equilibrium_prices     the price at every intersection over the last 14 days, by junction type
                         and against load
Tables (results/analysis/learning_*.csv):
  _bids_summary          per scenario, strategy and intersection: the bidders' balance, bids
                         (median, mean, 90%), share of zero bids, bid as a share of the bidder's
                         balance, price, share contested, payment per win, rebate per loss
  _convergence_summary   per chain: days run, converged or not and on which day
  _equilibrium_prices    per chain and intersection: price, its day-to-day spread, junction, load;
  _repeatability         spread of the price across seeds against its day-to-day spread
  _empirical_policy      mean bid by (strategy, balance bin, intersections left, losses here)
  _outcomes / _paired    evaluation block: time lost and per-driver fairness, paired by
                         (seed, evaluation day) against the baselines
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402

from paired import paired, show  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analyze", "figures", "learning")
TABLES = os.path.join(ROOT, "results", "analysis")
sys.path.insert(0, os.path.join(ROOT, "controller"))
from best_response import BestResponseLearner  # noqa: E402
from learning_study import LEARNERS, SCENARIOS  # noqa: E402

LABELS = {"learn_br": "A: best response", "learn_q": "B: Q-learning",
          "learn_hedge": "C: Hedge (myopic)", "front_wait": "front-car wait bid",
          "only_lane_urgency": "lane wait bid", "actuated_lights": "actuated lights",
          "stop_signs": "stop signs"}
COLORS = {"learn_br": "#1f77b4", "learn_q": "#d62728", "learn_hedge": "#2ca02c"}
# Days averaged for the "equilibrium" (the trailing convergence window).
LAST = 14
# The average balance in carma (every driver starts with it; the total never changes).
MEAN_BALANCE = 100.0
PRICE = "price = winning bid in contested auctions (carma; everyone starts with 100)"
# Balance bins in carma for the bid tables.
BALANCE_EDGES = [0, 25, 50, 75, 100, 150, 200, 300, 500, np.inf]
BALANCE_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-150", "150-200", "200-300",
                  "300-500", "500+"]


DPI = 300   # PNG resolution, and of any rasterized layer inside the PDF


def save(fig, name: str, dpi: int = DPI) -> None:
    """Save a figure as PNG and PDF."""
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=dpi)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), dpi=dpi)
    plt.close(fig)


def read_json(path: str) -> dict:
    """A JSON file, or {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(value) -> str:
    """A carma amount for a label: one decimal below 10, whole numbers above."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.1f}" if abs(value) < 10 else f"{value:.0f}"


def annotate_cells(ax, values: np.ndarray, image) -> None:
    """Write every cell's value on a heatmap, in black or white depending on the colour."""
    norm, cmap = image.norm, image.cmap
    for (i, j), v in np.ndenumerate(values):
        if pd.isna(v):
            continue
        r, g, b, _ = cmap(norm(v))
        color = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7, color=color)


def strategy_title(chain: dict) -> str:
    """'A: best response, seed 0' for a chain."""
    return f"{LABELS.get(chain['experiment'], chain['experiment'])}, seed {chain['seed']}"


# ---- Network geometry -------------------------------------------------------------------------

def network(scenario: str):
    """({junction: (x, y)}, [(from, to)]) of a scenario's network (internal parts left out)."""
    tree = ET.parse(os.path.join(ROOT, SCENARIOS[scenario]["net"]))
    junctions, edges = {}, []
    for j in tree.getroot().iter("junction"):
        if j.get("type") != "internal" and not j.get("id").startswith(":"):
            junctions[j.get("id")] = (float(j.get("x")), float(j.get("y")))
    for e in tree.getroot().iter("edge"):
        if e.get("function") is None and e.get("from") in junctions and e.get("to") in junctions:
            edges.append((e.get("from"), e.get("to")))
    return junctions, edges


def grid_layout(junctions: dict) -> tuple:
    """(rows, columns, {junction: (row, column)}) placing junctions as on the map."""
    xs = sorted({round(x) for x, _ in junctions.values()})
    ys = sorted({round(y) for _, y in junctions.values()}, reverse=True)
    pos = {j: (ys.index(round(y)), xs.index(round(x))) for j, (x, y) in junctions.items()}
    return len(ys), len(xs), pos


def junction_of(prices: pd.DataFrame) -> dict:
    """{tls: "4-way" | "T" | ...} from a price table."""
    if prices.empty:
        return {}
    return dict(prices.drop_duplicates("tls")[["tls", "junction"]].to_numpy())


# ---- Loading ------------------------------------------------------------------------------

def load_chains(results: str, scenario: str) -> dict:
    """{chain id: {experiment, seed, status, prices, conv, out}} for one scenario."""
    out = os.path.join(ROOT, "results", results, scenario)
    chains = {}
    for path in sorted(glob.glob(os.path.join(out, "*_status.json"))):
        status = read_json(path)
        cid = status.get("chain") or os.path.basename(path)[:-len("_status.json")]
        prices_path = os.path.join(out, f"{cid}_prices_by_day.csv")
        conv_path = os.path.join(out, f"{cid}_convergence.csv")
        chains[cid] = {
            "experiment": status.get("experiment"), "seed": status.get("seed"),
            "status": status, "out": out, "cid": cid,
            "prices": pd.read_csv(prices_path) if os.path.exists(prices_path) else pd.DataFrame(),
            "conv": pd.read_csv(conv_path) if os.path.exists(conv_path) else pd.DataFrame(),
        }
    return chains


def stop_day(chain: dict):
    """Day the chain converged on (None if it hasn't)."""
    return chain["status"].get("converged_on")


def active_tls(prices: pd.DataFrame, min_auctions: float = 50.0) -> list:
    """Intersections that average at least `min_auctions` auctions a day."""
    if prices.empty:
        return []
    mean = prices.groupby("tls")["n_auctions"].mean()
    return sorted(mean[mean >= min_auctions].index)


def last_learning_days(chain: dict) -> list:
    """The last LAST learning days of a chain."""
    if chain["prices"].empty:
        return []
    days = sorted(chain["prices"]["day"].unique())
    return days[-LAST:]


def load_bids(chain: dict, days=None) -> pd.DataFrame:
    """The decision bids of a chain on the given days (default: its last learning days)."""
    days = last_learning_days(chain) if days is None else days
    parts = []
    for day in days:
        path = os.path.join(chain["out"], f"{chain['cid']}_day{day}_bids.csv.gz")
        if os.path.exists(path):
            parts.append(pd.read_csv(path).assign(day=day, seed=chain["seed"],
                                                  strategy=chain["experiment"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tls_order(tls_list, geometry) -> list:
    """Intersections in map order (top row first, left to right)."""
    _, _, pos = grid_layout(geometry[0])
    return sorted(tls_list, key=lambda t: pos.get(t, (99, 99)))


# ---- What drivers bid (per scenario) ------------------------------------------------------

def bids_summary(scenario: str, chains: dict) -> pd.DataFrame:
    """Per strategy and intersection, over the last learning days (all seeds): the numbers that
    describe what drivers actually bid and pay."""
    rows = []
    by_strategy = {}
    for chain in chains.values():
        if chain["experiment"] in LEARNERS or chain["experiment"] in ("front_wait",
                                                                      "only_lane_urgency"):
            bids = load_bids(chain)
            if len(bids):
                by_strategy.setdefault(chain["experiment"], []).append(bids)
    junctions = {}
    for chain in chains.values():
        junctions.update(junction_of(chain["prices"]))
    for name, parts in by_strategy.items():
        bids = pd.concat(parts, ignore_index=True)
        for tls, g in bids.groupby("tls"):
            auctions = g.drop_duplicates(["seed", "day", "auction_id"])
            contested = auctions[auctions["n_rivals"] > 0]
            share = g["bid"] / g["balance"].where(g["balance"] > 0)
            rows.append({
                "scenario": scenario, "strategy": name, "tls": tls,
                "junction": junctions.get(tls, ""),
                "bidders_balance_median": g["balance"].median(),
                "bid_median": g["bid"].median(), "bid_mean": g["bid"].mean(),
                "bid_p90": g["bid"].quantile(0.9), "share_zero_bids": (g["bid"] <= 0).mean(),
                "bid_share_of_balance_median": share.median(),
                "price_median": contested["winning_bid"].median(),
                "share_contested": (auctions["n_rivals"] > 0).mean(),
                "paid_per_win": g[g["won"]]["payment"].mean(),
                "rebate_per_loss": g[~g["won"]]["rebate"].mean(),
                "decisions_per_day": len(g) / g[["seed", "day"]].drop_duplicates().shape[0]})
    return pd.DataFrame(rows)


def bids_by_intersection(scenario: str, chains: dict, geometry) -> None:
    """Mean bid by intersection and bidder's balance, numbers in every cell, one panel per
    learner."""
    parts = [load_bids(c) for c in chains.values() if c["experiment"] in LEARNERS]
    parts = [p for p in parts if len(p)]
    if not parts:
        return
    bids = pd.concat(parts, ignore_index=True)
    bids["balance_bin"] = pd.cut(bids["balance"], BALANCE_EDGES, labels=BALANCE_LABELS,
                                 right=False)
    names = [n for n in LEARNERS if n in bids["strategy"].unique()]
    junctions = {}
    for chain in chains.values():
        junctions.update(junction_of(chain["prices"]))
    order = tls_order(bids["tls"].unique(), geometry)
    vmax = bids.groupby(["strategy", "tls", "balance_bin"], observed=True)["bid"].mean()\
        .quantile(0.95)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 1.6 + 0.45 * len(order)),
                             squeeze=False, layout="constrained")
    image = None
    for ax, name in zip(axes[0], names):
        sub = bids[bids["strategy"] == name]
        heat = sub.pivot_table(index="tls", columns="balance_bin", values="bid", aggfunc="mean",
                               observed=False).reindex(order)
        heat = heat[[c for c in BALANCE_LABELS if c in heat.columns and heat[c].notna().any()]]
        image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0,
                          vmax=vmax)
        annotate_cells(ax, heat.to_numpy(dtype=float), image)
        overall = sub.groupby("tls")["bid"].median().reindex(order)
        ax.set_yticks(range(len(order)), [f"{t} ({junctions.get(t, '')}), median "
                                          f"{fmt(overall[t])}" for t in order], fontsize=7)
        ax.set_xticks(range(len(heat.columns)), heat.columns, fontsize=7, rotation=45)
        ax.set_xlabel("bidder's balance (carma)")
        ax.set_title(LABELS[name], fontsize=9)
    fig.colorbar(image, ax=axes[0].tolist(), label="mean bid (carma)", shrink=0.8)
    fig.suptitle(f"{scenario}: what drivers bid at each intersection, by how much carma they "
                 f"hold (mean bid in carma; last {LAST} learning days, all seeds)", fontsize=10)
    save(fig, f"{scenario}_bids_by_intersection")


def bid_distribution(scenario: str, chains: dict, geometry) -> None:
    """Box plots of the bids made at every intersection, per learner, with the median price."""
    parts = [load_bids(c) for c in chains.values() if c["experiment"] in LEARNERS]
    parts = [p for p in parts if len(p)]
    if not parts:
        return
    bids = pd.concat(parts, ignore_index=True)
    names = [n for n in LEARNERS if n in bids["strategy"].unique()]
    order = tls_order(bids["tls"].unique(), geometry)
    junctions = {}
    for chain in chains.values():
        junctions.update(junction_of(chain["prices"]))
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(order) * len(names) / 2), 4.5))
    width = 0.8 / len(names)
    top = bids["bid"].quantile(0.97)
    for k, name in enumerate(names):
        sub = bids[bids["strategy"] == name]
        data = [sub[sub["tls"] == t]["bid"].to_numpy() for t in order]
        positions = [i - 0.4 + width * (k + 0.5) for i in range(len(order))]
        box = ax.boxplot(data, positions=positions, widths=width * 0.85, showfliers=False,
                         patch_artist=True, whis=(10, 90))
        for patch in box["boxes"]:
            patch.set(facecolor=COLORS[name], alpha=0.35, edgecolor=COLORS[name])
        for median in box["medians"]:
            median.set(color=COLORS[name], lw=2)
        for i, t in enumerate(order):
            g = sub[sub["tls"] == t]
            contested = g[g["n_rivals"] > 0].drop_duplicates(["seed", "day", "auction_id"])
            price = contested["winning_bid"].median()
            ax.plot(positions[i], price, marker="D", color="k", ms=4, zorder=3)
            ax.text(positions[i], min(price, top) + top * 0.02, fmt(price), ha="center",
                    fontsize=7)
        ax.plot([], [], color=COLORS[name], lw=6, alpha=0.35, label=LABELS[name])
    ax.plot([], [], "kD", ms=4, label="median price (winning bid, contested; number)")
    ax.set_xticks(range(len(order)), [f"{t}\n{junctions.get(t, '')}" for t in order])
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("bid (carma)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{scenario}: bids made at each intersection (box 25-75%, whiskers 10-90%, "
                 f"line median; last {LAST} learning days, all seeds)", fontsize=10)
    fig.tight_layout()
    save(fig, f"{scenario}_bid_distribution")


def learners_compared(scenario: str, chains: dict) -> None:
    """The price by day for every learner and seed, one panel per junction type, final values
    written at the line ends."""
    tables = {cid: c for cid, c in chains.items()
              if c["experiment"] in LEARNERS and not c["prices"].empty}
    if not tables:
        return
    kinds = sorted({j for c in tables.values() for j in c["prices"]["junction"].unique()})
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.5 * len(kinds), 4.2), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for chain in tables.values():
            p = chain["prices"]
            p = p[(p["junction"] == kind) & p["tls"].isin(active_tls(p))]
            if p.empty:
                continue
            curve = p.groupby("day").apply(
                lambda g: np.average(g["win_mean"].fillna(0), weights=g["n_contested"] + 1e-9),
                include_groups=False)
            color = COLORS[chain["experiment"]]
            ax.plot(curve.index, curve.values, color=color, lw=1, alpha=0.8)
            end = curve.iloc[-LAST:].mean()
            ax.annotate(f"{fmt(end)} (s{chain['seed']})", (curve.index[-1], end),
                        xytext=(4, 0), textcoords="offset points", fontsize=7, color=color,
                        va="center")
        for name in LEARNERS:
            ax.plot([], [], color=COLORS[name], label=LABELS[name])
        ax.set_xlabel("day")
        ax.set_ylabel("price (carma)")
        ax.set_title(f"{kind} junctions", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        ax.margins(x=0.12)
    fig.suptitle(f"{scenario}: price of priority while learning, one line per seed (number at "
                 f"the end: mean of the last {LAST} days). {PRICE}", fontsize=9)
    fig.tight_layout()
    save(fig, f"{scenario}_learners")


def empirical_policy(scenario: str, chains: dict) -> pd.DataFrame:
    """Mean bid by bidder's balance and intersections left (no losses yet here), numbers in the
    cells, from the bid logs of the last learning days."""
    parts = [load_bids(c) for c in chains.values() if c["experiment"] in LEARNERS]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    bids = pd.concat(parts, ignore_index=True)
    bids["balance_bin"] = pd.cut(bids["balance"], BALANCE_EDGES, labels=BALANCE_LABELS,
                                 right=False)
    bids["k"] = bids["k"].clip(upper=6)
    table = bids.groupby(["strategy", "balance_bin", "k", "lost_here"], observed=True).agg(
        bid=("bid", "mean"), n=("bid", "size")).reset_index()
    names = [n for n in LEARNERS if n in table["strategy"].unique()]
    vmax = table[table.lost_here == 0]["bid"].quantile(0.95)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 3.8), squeeze=False,
                             layout="constrained")
    image = None
    for ax, name in zip(axes[0], names):
        sub = table[(table.strategy == name) & (table.lost_here == 0) & (table.n >= 20)]
        heat = sub.pivot_table(index="k", columns="balance_bin", values="bid", observed=False)
        heat = heat[[c for c in BALANCE_LABELS if c in heat.columns and heat[c].notna().any()]]
        image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", origin="lower",
                          cmap="viridis", vmin=0, vmax=vmax)
        annotate_cells(ax, heat.to_numpy(dtype=float), image)
        ax.set_xticks(range(len(heat.columns)), heat.columns, fontsize=7, rotation=45)
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_xlabel("bidder's balance (carma)")
        ax.set_ylabel("intersections left (incl. this one)")
        ax.set_title(LABELS[name], fontsize=9)
    fig.colorbar(image, ax=axes[0].tolist(), label="mean bid (carma)", shrink=0.8)
    fig.suptitle(f"{scenario}: mean bid by balance and intersections left (first bid at that "
                 f"intersection; last {LAST} learning days; cells with 20+ bids)", fontsize=10)
    save(fig, f"{scenario}_empirical_policy")
    table.insert(0, "scenario", scenario)
    return table


# ---- Price evolution (per chain) -----------------------------------------------------------

def prices_grid(scenario: str, chain: dict, geometry) -> None:
    """One panel per intersection in its place on the map: the daily price (median, 25-75% and
    10-90% of the day's contested auctions), the median bid, and the last-14-day numbers."""
    prices = chain["prices"]
    if prices.empty:
        return
    junctions, _ = geometry
    rows, cols, pos = grid_layout(junctions)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols + 0.5, 2.5 * rows + 1.2),
                             squeeze=False, sharex=True)
    top = np.nanmax([prices["win_q90"].quantile(0.98), prices["bid_q50"].max(), 1.0])
    for j, (r, c) in pos.items():
        ax = axes[r][c]
        if j not in set(prices["tls"]):
            # Hide this panel's ticks only (set_xticks would clear the shared axis everywhere).
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.text(0.5, 0.5, f"{j}\nno auctions", ha="center", va="center", color="0.6",
                    fontsize=8, transform=ax.transAxes)
            for spine in ax.spines.values():
                spine.set_color("0.85")
    last_days = last_learning_days(chain)
    for tls, g in prices.groupby("tls"):
        if tls not in pos:
            continue
        ax = axes[pos[tls]]
        g = g.sort_values("day")
        ax.fill_between(g["day"], g["win_q10"], g["win_q90"], color="C0", alpha=0.15, lw=0)
        ax.fill_between(g["day"], g["win_q25"], g["win_q75"], color="C0", alpha=0.3, lw=0)
        ax.plot(g["day"], g["win_q50"], color="C0", lw=1.3)
        ax.plot(g["day"], g["bid_q50"], color="0.3", lw=0.9, ls="--")
        if stop_day(chain):
            ax.axvline(stop_day(chain), color="k", ls=":", lw=0.8)
        recent = g[g["day"].isin(last_days)]
        ax.text(0.98, 0.97, f"last {LAST} days:\nprice {fmt(recent['win_q50'].mean())}"
                f"\nbid {fmt(recent['bid_q50'].mean())}"
                f"\ncontested {recent['share_contested'].mean():.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.85, "lw": 0.3})
        ax.set_title(f"{tls} ({g['junction'].iloc[0]}), {g['n_auctions'].mean():.0f} "
                     "auctions/day", fontsize=8)
        ax.set_ylim(0, top * 1.05)
        ax.set_ylabel("carma", fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7, labelbottom=True)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="C0", lw=1.3, label="price: median winning bid"),
               Patch(color="C0", alpha=0.45, label="25-75% of winning bids"),
               Patch(color="C0", alpha=0.15, label="10-90% of winning bids"),
               Line2D([], [], color="0.3", ls="--", lw=0.9, label="median bid (all bidders)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{scenario}, {strategy_title(chain)}: price of priority at each intersection "
                 f"by day\n{PRICE}", fontsize=10)
    fig.supxlabel("day", y=0.055)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, f"{scenario}_{chain['cid']}_prices_grid")


def prices_weekly(scenario: str, chain: dict, geometry) -> None:
    """The median price at every intersection, week by week, with the numbers in the cells."""
    prices = chain["prices"]
    if prices.empty:
        return
    p = prices.assign(week=(prices["day"] - 1) // 7 + 1)
    table = p.pivot_table(index="tls", columns="week", values="win_q50", aggfunc="mean")
    table = table.loc[tls_order(table.index, geometry)]
    junctions = junction_of(prices)
    fig, ax = plt.subplots(figsize=(1.6 + 0.62 * table.shape[1], 1.4 + 0.42 * len(table)),
                           layout="constrained")
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    annotate_cells(ax, table.to_numpy(), image)
    ax.set_yticks(range(len(table)), [f"{t} ({junctions.get(t, '')})" for t in table.index],
                  fontsize=8)
    weeks = table.columns
    ax.set_xticks(range(len(weeks)), [f"{w}\n({(w - 1) * 7 + 1}-{w * 7})" for w in weeks],
                  fontsize=7)
    ax.set_xlabel("week (days)")
    fig.colorbar(image, ax=ax, label="price (carma)")
    ax.set_title(f"{scenario}, {strategy_title(chain)}: median price at each intersection, "
                 "week by week (carma)", fontsize=10)
    save(fig, f"{scenario}_{chain['cid']}_prices_weekly")


def _draw_map(ax, geometry, values: dict, vmin: float, vmax: float, title: str,
              contested: dict | None = None):
    """The network with every auction-holding intersection coloured by its price and the price
    written next to it (grey squares: no auctions)."""
    junctions, edges = geometry
    for a, b in edges:
        (x1, y1), (x2, y2) = junctions[a], junctions[b]
        ax.plot([x1, x2], [y1, y2], color="0.82", lw=3, zorder=1, solid_capstyle="round")
    shown = None
    span = max(max(x for x, _ in junctions.values()) - min(x for x, _ in junctions.values()),
               max(y for _, y in junctions.values()) - min(y for _, y in junctions.values()), 1)
    for j, (x, y) in junctions.items():
        if j in values and not pd.isna(values[j]):
            shown = ax.scatter([x], [y], c=[values[j]], cmap="viridis", vmin=vmin, vmax=vmax,
                               s=520, edgecolors="k", zorder=2)
            extra = f"\n{contested[j]:.0%} contested" if contested and j in contested else ""
            ax.text(x, y + span * 0.07, f"{fmt(values[j])}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", zorder=3,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "lw": 0.4})
            ax.text(x, y - span * 0.07, f"{j}{extra}", ha="center", va="top", fontsize=7,
                    zorder=3)
        else:
            ax.scatter([x], [y], marker="s", s=60, color="0.75", zorder=2)
            ax.text(x, y - span * 0.05, j, ha="center", va="top", fontsize=6, color="0.5")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")
    return shown


def prices_map(scenario: str, chain: dict, geometry, gif: bool) -> None:
    """The network with the price written at every intersection on a few days, and an animation
    over all days next to every intersection's price curve."""
    prices = chain["prices"]
    if prices.empty:
        return
    days = sorted(prices["day"].unique())
    vmax = prices["win_q50"].quantile(0.98)
    by_day = {d: dict(g[["tls", "win_q50"]].to_numpy()) for d, g in prices.groupby("day")}
    contested_by_day = {d: dict(g[["tls", "share_contested"]].to_numpy())
                        for d, g in prices.groupby("day")}
    picks = sorted({d for d in (1, 10, 30, days[-1]) if d in days})
    fig, axes = plt.subplots(1, len(picks), figsize=(3.8 * len(picks), 4.2), squeeze=False,
                             layout="constrained")
    shown = None
    for ax, day in zip(axes[0], picks):
        shown = _draw_map(ax, geometry, by_day[day], 0, vmax, f"day {day}",
                          contested_by_day[day]) or shown
    fig.colorbar(shown, ax=axes[0].tolist(), label="price (carma)", shrink=0.8)
    fig.suptitle(f"{scenario}, {strategy_title(chain)}: number = median winning bid at the "
                 "intersection that day (carma; everyone starts with 100)", fontsize=10)
    save(fig, f"{scenario}_{chain['cid']}_prices_map")
    if not gif:
        return
    order = tls_order(prices["tls"].unique(), geometry)
    curves = prices.pivot_table(index="day", columns="tls", values="win_q50")
    junctions = junction_of(prices)
    styles = {"4-way": "-", "T": "--"}
    colors = {t: plt.get_cmap("tab10")(i % 10) for i, t in enumerate(order)}
    fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=(11, 4.8),
                                          gridspec_kw={"width_ratios": [1, 1.3]})

    def frame(day):
        ax_map.clear()
        ax_line.clear()
        _draw_map(ax_map, geometry, by_day.get(day, {}), 0, vmax, "price today (carma)",
                  contested_by_day.get(day, {}))
        upto = curves[curves.index <= day]
        for t in order:
            ax_line.plot(upto.index, upto[t], color=colors[t], lw=1.1,
                         ls=styles.get(junctions.get(t), "-"),
                         label=f"{t} ({junctions.get(t, '')})")
            if len(upto) and not pd.isna(upto[t].iloc[-1]):
                ax_line.plot(day, upto[t].iloc[-1], "o", color=colors[t], ms=4)
                ax_line.annotate(fmt(upto[t].iloc[-1]), (day, upto[t].iloc[-1]),
                                 xytext=(4, 0), textcoords="offset points", fontsize=7,
                                 color=colors[t], va="center")
        ax_line.set_xlim(0, days[-1] + 8)
        ax_line.set_ylim(0, vmax * 1.15)
        ax_line.set_xlabel("day")
        ax_line.set_ylabel("price: median winning bid (carma)")
        ax_line.grid(alpha=0.3)
        ax_line.legend(fontsize=7, loc="upper left", ncol=2)
        fig.suptitle(f"{scenario}, {strategy_title(chain)}: day {day} of {days[-1]}",
                     fontsize=11)
    movie = animation.FuncAnimation(fig, frame, frames=days, interval=250)
    movie.save(os.path.join(OUT, f"{scenario}_{chain['cid']}_prices_map.gif"),
               writer=animation.PillowWriter(fps=4))
    plt.close(fig)


def within_day(scenario: str, chain: dict) -> None:
    """Price in 5-minute bins over the day (contested auctions, all intersections), for a few
    days."""
    prices = chain["prices"]
    if prices.empty:
        return
    days = sorted(prices["day"].unique())
    picks = sorted({d for d in (1, 10, 30, days[-1]) if d in days})
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, day in enumerate(picks):
        path = os.path.join(chain["out"], f"{chain['cid']}_day{day}_prices.csv")
        if not os.path.exists(path):
            continue
        table = pd.read_csv(path)
        bins = table[table.scope == "bin"]
        curve = bins.groupby("bin_start").apply(
            lambda g: np.average(g["win_mean"].fillna(0), weights=g["n_contested"] + 1e-9),
            include_groups=False)
        ax.plot(curve.index / 60, curve.values, marker=".", label=f"day {day}",
                color=plt.get_cmap("viridis")(i / max(1, len(picks) - 1)))
    ax.set_xlabel("minutes into the day")
    ax.set_ylabel("mean winning bid (carma)")
    ax.set_title(f"{scenario}, {strategy_title(chain)}: price during the day (5-minute bins, "
                 "contested auctions, all intersections)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, f"{scenario}_{chain['cid']}_within_day")


def convergence_dashboard(scenario: str, chain: dict) -> None:
    """The section 7 metrics by day, with thresholds and the stop day."""
    conv = chain["conv"]
    if conv.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    axes[0].semilogy(conv["day"], (conv["m1"] * MEAN_BALANCE).clip(lower=1e-3), marker=".",
                     lw=1)
    axes[0].axhline(0.01 * MEAN_BALANCE, color="r", ls="--", lw=0.8)
    axes[0].set_ylabel("policy change\n(carma per day)")
    axes[1].plot(conv["day"], conv["m2_share"], marker=".", lw=1, label="no price drift")
    axes[1].plot(conv["day"], conv["m3_share"], marker=".", lw=1, label="no week-on-week change")
    axes[1].axhline(0.9, color="r", ls="--", lw=0.8)
    axes[1].set_ylabel("share of intersections\npassing")
    axes[1].legend(fontsize=7)
    axes[2].plot(conv["day"], conv["exploration"], marker=".", lw=1)
    axes[2].set_ylabel("exploration\n(temperature / epsilon)")
    axes[3].bar(conv["day"], conv["consecutive"], color=np.where(conv["passed"], "C2", "C3"))
    axes[3].axhline(7, color="r", ls="--", lw=0.8)
    axes[3].set_ylabel("passing days\nin a row")
    axes[3].set_xlabel("day")
    for ax in axes:
        ax.grid(alpha=0.3)
        if stop_day(chain):
            ax.axvline(stop_day(chain), color="k", ls="--", lw=0.8)
        ax.axvline(chain["status"].get("min_days", 60), color="0.5", ls=":", lw=0.8)
    state = (f"converged on day {stop_day(chain)}" if stop_day(chain) else
             f"not converged after {chain['status'].get('last_day')} days")
    fig.suptitle(f"{scenario}, {strategy_title(chain)}: {state} (red lines: thresholds)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, f"{scenario}_{chain['cid']}_convergence")


def policy_figure(scenario: str, chain: dict) -> None:
    """The learned policy on the last learning day: the bid (numbers in the cells) by balance and
    intersections left (no losses yet here, at a typical intersection), and for learner A the
    learned value against balance."""
    last = chain["status"].get("last_day")
    cid = chain["cid"]
    path = os.path.join(chain["out"], f"{cid}_day{last}_policy.csv")
    if not last or not os.path.exists(path):
        return
    table = pd.read_csv(path)
    if "k" not in table or "x" not in table:
        return
    table = table[table["lost_here"] == 0]
    if "urgency" in table:
        table = table[table["urgency"] == table["urgency"].min()]
    is_br = chain["experiment"] == "learn_br"
    learner = None
    if is_br:
        # Only the numbers of intersections left that the learner has actually met.
        learner = BestResponseLearner()
        learner.load_state(os.path.join(chain["out"], f"{cid}_day{last}_learner.json"))
        table = table[table["k"].isin({k for k, _, _ in learner.V})]
    table = table.assign(balance=table["x"] * MEAN_BALANCE, bid=table["bid_rel"] * MEAN_BALANCE)
    heat = table.pivot_table(index="k", columns="balance", values="bid")
    fig, axes = plt.subplots(1, 2 if is_br else 1, figsize=(13 if is_br else 7, 3.8),
                             squeeze=False, layout="constrained")
    image = axes[0][0].imshow(heat.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    annotate_cells(axes[0][0], heat.to_numpy(), image)
    axes[0][0].set_xticks(range(len(heat.columns)), [fmt(b) for b in heat.columns],
                          fontsize=7, rotation=90)
    axes[0][0].set_yticks(range(len(heat.index)), heat.index)
    axes[0][0].set_xlabel("balance (carma)")
    axes[0][0].set_ylabel("intersections left")
    fig.colorbar(image, ax=axes[0][0], label="bid (carma)")
    axes[0][0].set_title("learned bid (carma) at a typical intersection, before any loss there",
                         fontsize=9)
    if is_br:
        x = learner.grid[learner.grid <= 6]
        for k, lost, u in sorted(learner.V):
            if lost == 0 and u == min(key[2] for key in learner.V):
                axes[0][1].plot(x * MEAN_BALANCE, learner.value(x, k, lost, u), label=f"k={k}")
        axes[0][1].set_xlabel("balance (carma)")
        axes[0][1].set_ylabel("relative value (seconds)")
        axes[0][1].set_title("learned value of a balance (steeper = carma worth more seconds)",
                             fontsize=9)
        axes[0][1].legend(fontsize=7, title="intersections left", title_fontsize=7)
        axes[0][1].grid(alpha=0.3)
    fig.suptitle(f"{scenario}, {strategy_title(chain)}, policy on day {last}", fontsize=10)
    save(fig, f"{scenario}_{cid}_policy")


def belief_figure(scenario: str, chain: dict) -> None:
    """Learner A: the median and 90% highest rival bid it expects against the real ones (mean
    over intersections), by day."""
    rows = []
    cid = chain["cid"]
    for path in glob.glob(os.path.join(chain["out"], f"{cid}_day*_learner.json")):
        day = int(re.search(r"_day(\d+)_learner", path).group(1))
        bids_path = path.replace("_learner.json", "_bids.csv.gz")
        if not os.path.exists(bids_path):
            continue
        state = read_json(path)
        bids = pd.read_csv(bids_path, usecols=["tls", "top_rival", "n_rivals"])
        bids = bids[bids["n_rivals"] > 0]
        for tls, values in state.get("rivals", {}).items():
            real = bids[bids.tls == tls]["top_rival"]
            if len(values) and len(real):
                values = np.asarray(values) * MEAN_BALANCE
                rows.append({"day": day, "tls": tls,
                             "belief_q50": np.quantile(values, 0.5),
                             "belief_q90": np.quantile(values, 0.9),
                             "real_q50": real.quantile(0.5), "real_q90": real.quantile(0.9)})
    if not rows:
        return
    table = pd.DataFrame(rows).groupby("day").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(table.index, table["real_q50"], "k-", label="real median")
    ax.plot(table.index, table["belief_q50"], "C0--", label="expected median")
    ax.plot(table.index, table["real_q90"], "k-", alpha=0.4, label="real 90%")
    ax.plot(table.index, table["belief_q90"], "C0--", alpha=0.4, label="expected 90%")
    ax.set_xlabel("day")
    ax.set_ylabel("highest rival bid (carma)")
    ax.set_title(f"{scenario}, {strategy_title(chain)}: the rival bids it expects vs the real "
                 "ones (mean over intersections)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"{scenario}_{cid}_belief")


# ---- Across scenarios, outcomes, convergence ---------------------------------------------

def equilibrium_prices(scenario: str, chains: dict) -> pd.DataFrame:
    """Per chain and intersection: the price over the last learning days (carma), its
    day-to-day spread, junction type and load."""
    rows = []
    for chain in chains.values():
        if chain["experiment"] not in LEARNERS or chain["prices"].empty:
            continue
        days = last_learning_days(chain)
        window = chain["prices"][chain["prices"]["day"].isin(days)]
        for tls, g in window.groupby("tls"):
            rows.append({"scenario": scenario, "strategy": chain["experiment"],
                         "seed": chain["seed"], "tls": tls, "junction": g["junction"].iloc[0],
                         "price": g["win_q50"].mean(), "price_mean": g["win_mean"].mean(),
                         "price_sd_days": g["win_q50"].std(),
                         "bid_median": g["bid_q50"].mean(),
                         "auctions_per_day": g["n_auctions"].mean(),
                         "share_contested": g["share_contested"].mean(),
                         "cars_in_range": g["cars_in_range_mean"].mean(),
                         "days": f"{days[0]}-{days[-1]}",
                         "converged": bool(chain["status"].get("converged"))})
    return pd.DataFrame(rows)


def equilibrium_figure(table: pd.DataFrame) -> None:
    """The price at every intersection over the last learning days, by junction type (mean over
    seeds written) and against load."""
    if table.empty:
        return
    scenarios = [s for s in SCENARIOS if s in table["scenario"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    offsets = {name: i for i, name in enumerate(LEARNERS)}
    groups = sorted({(s, j) for s, j in table[["scenario", "junction"]].itertuples(index=False)},
                    key=lambda sj: (scenarios.index(sj[0]), sj[1]))
    ticks = []
    rng = np.random.default_rng(0)
    for i, (scenario, junction) in enumerate(groups):
        ticks.append(f"{scenario}\n{junction}")
        for name in LEARNERS:
            sub = table[(table.scenario == scenario) & (table.junction == junction)
                        & (table.strategy == name)]
            if len(sub):
                x0 = i + 0.25 * (offsets[name] - 1)
                axes[0].scatter(x0 + rng.uniform(-0.05, 0.05, len(sub)), sub["price"], s=14,
                                color=COLORS[name], alpha=0.7,
                                label=LABELS[name] if i == 0 else None)
                axes[0].text(x0, sub["price"].max() * 1.04 + 1, fmt(sub["price"].mean()),
                             ha="center", fontsize=7, color=COLORS[name])
    axes[0].set_xticks(range(len(ticks)), ticks, fontsize=8)
    axes[0].set_ylabel(f"price (carma, median winning bid, last {LAST} days)")
    axes[0].set_title("every intersection and seed is a dot; number = mean", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    markers = dict(zip(scenarios, "osD^"))
    for name in LEARNERS:
        for scenario in scenarios:
            sub = table[(table.strategy == name) & (table.scenario == scenario)]
            if len(sub):
                axes[1].scatter(sub["cars_in_range"], sub["price"], s=18, color=COLORS[name],
                                marker=markers[scenario], alpha=0.8,
                                label=f"{LABELS[name]}, {scenario}")
    axes[1].set_xlabel("cars within bidding range per auction (competition)")
    axes[1].set_ylabel("price (carma)")
    axes[1].set_title("price against competition", fontsize=9)
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Price of priority by junction type and competition ({PRICE})", fontsize=10)
    fig.tight_layout()
    save(fig, "equilibrium_prices")


def repeatability(table: pd.DataFrame) -> pd.DataFrame:
    """For every (scenario, learner, intersection): spread of the price across seeds against its
    typical day-to-day spread."""
    if table.empty:
        return pd.DataFrame()
    out = table.groupby(["scenario", "strategy", "tls"]).agg(
        seeds=("seed", "nunique"), price_mean=("price", "mean"), sd_across_seeds=("price", "std"),
        sd_days=("price_sd_days", "mean")).reset_index()
    out["repeatable"] = out["sd_across_seeds"] <= out["sd_days"]
    return out


def outcomes(scenario: str, results: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluation block: time lost per (strategy, seed, evaluation day), per-driver fairness, and
    paired differences against the baselines."""
    out = os.path.join(ROOT, "results", results, scenario)
    rows, drivers = [], []
    for path in glob.glob(os.path.join(out, "*_after*_eval*_cars.csv")):
        match = re.search(r"/(\w+)_s(\d+)_after(\d+)_eval(\d+)_cars\.csv$", path)
        if not match:
            continue
        name, seed, after, e = match.group(1), int(match.group(2)), int(match.group(3)), \
            int(match.group(4))
        cars = pd.read_csv(path)
        rows.append({"name": name, "seed": seed, "eval_day": e, "after_day": after,
                     "time_lost": cars["time_loss"].mean(),
                     "p95_time_lost": cars["time_loss"].quantile(0.95),
                     "travel_time": cars["travel_time"].mean(), "trips": len(cars)})
        cars["name"], cars["seed"] = name, seed
        drivers.append(cars[["name", "seed", "driver", "time_loss"]])
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    table = pd.DataFrame(rows)
    # Keep the latest evaluation block of every chain (a chain extended later is re-evaluated).
    latest = table.groupby(["name", "seed"])["after_day"].transform("max")
    table = table[table["after_day"] == latest]

    def gini(values):
        v = np.sort(np.asarray(values, dtype=float))
        if len(v) == 0 or v.sum() <= 0:
            return np.nan
        ranks = np.arange(1, len(v) + 1)
        return 2 * (ranks * v).sum() / (len(v) * v.sum()) - (len(v) + 1) / len(v)
    per_driver = pd.concat(drivers).groupby(["name", "seed", "driver"])["time_loss"].mean()
    fairness = per_driver.groupby(["name", "seed"]).apply(gini).rename("driver_gini")
    summary = table.groupby("name").agg(time_lost=("time_lost", "mean"),
                                        p95=("p95_time_lost", "mean"),
                                        seeds=("seed", "nunique"), days=("eval_day", "count"))
    summary = summary.join(fairness.groupby("name").mean())
    summary.insert(0, "scenario", scenario)
    pairs = []
    for reference in ("actuated_lights", "stop_signs", "front_wait", "only_lane_urgency"):
        if reference in table["name"].unique():
            result = paired(table, "time_lost", reference, units=("seed", "eval_day"))
            if len(result):
                result.insert(0, "scenario", scenario)
                pairs.append(result)
    return summary.reset_index(), (pd.concat(pairs, ignore_index=True) if pairs
                                   else pd.DataFrame())


def convergence_summary(scenario: str, chains: dict) -> pd.DataFrame:
    """Per learning chain: days run, converged or not, on which day."""
    rows = []
    for cid, chain in chains.items():
        if not chain["status"].get("learner"):
            continue
        s = chain["status"]
        rows.append({"scenario": scenario, "chain": cid, "strategy": chain["experiment"],
                     "seed": chain["seed"], "last_day": s.get("last_day"),
                     "converged": s.get("converged"), "converged_on": s.get("converged_on"),
                     "consecutive_passes": s.get("consecutive_passes"),
                     "evaluated_after": s.get("evaluated_after"),
                     "finished": s.get("finished", False),
                     "guest_trips_max": s.get("guest_trips_max")})
    return pd.DataFrame(rows)


def main() -> None:
    """Every figure and table for the scenarios found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--results", default="learning")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    os.makedirs(TABLES, exist_ok=True)
    collected = {k: [] for k in ("conv", "prices", "policy", "outcomes", "paired", "bids")}
    for scenario in args.scenarios:
        chains = load_chains(args.results, scenario)
        if not chains:
            continue
        geometry = network(scenario)
        print(f"######## {scenario}: {len(chains)} chains ########")
        for chain in chains.values():
            if chain["experiment"] in LEARNERS:
                prices_grid(scenario, chain, geometry)
                prices_weekly(scenario, chain, geometry)
                prices_map(scenario, chain, geometry, gif=not args.no_gif)
                within_day(scenario, chain)
                convergence_dashboard(scenario, chain)
                policy_figure(scenario, chain)
                if chain["experiment"] == "learn_br":
                    belief_figure(scenario, chain)
        learners_compared(scenario, chains)
        bids_by_intersection(scenario, chains, geometry)
        bid_distribution(scenario, chains, geometry)
        bids = bids_summary(scenario, chains)
        collected["bids"].append(bids)
        collected["conv"].append(convergence_summary(scenario, chains))
        collected["prices"].append(equilibrium_prices(scenario, chains))
        collected["policy"].append(empirical_policy(scenario, chains))
        summary, pairs = outcomes(scenario, args.results)
        collected["outcomes"].append(summary)
        collected["paired"].append(pairs)
        if len(bids):
            print(f"=== what drivers bid, last {LAST} learning days, all seeds (carma) ===")
            print(bids.drop(columns=["scenario"]).round(2).to_string(index=False))
        conv = collected["conv"][-1]
        if len(conv):
            print("=== convergence ===")
            print(conv.drop(columns=["chain"]).to_string(index=False))
        if len(summary):
            print("=== evaluation block: mean time lost (s), per-driver Gini ===")
            print(summary.round(3).to_string(index=False))
        if len(pairs):
            print("=== time lost vs actuated lights, paired by (seed, evaluation day) ===")
            print(show(pairs[pairs.reference == "actuated_lights"], by=("scenario",)))

    def write(parts, name):
        parts = [p for p in parts if p is not None and len(p)]
        if parts:
            table = pd.concat(parts, ignore_index=True)
            table.round(5).to_csv(os.path.join(TABLES, f"learning_{name}.csv"), index=False)
            return table
        return pd.DataFrame()
    write(collected["bids"], "bids_summary")
    write(collected["conv"], "convergence_summary")
    prices = write(collected["prices"], "equilibrium_prices")
    write([repeatability(prices)], "repeatability")
    write(collected["policy"], "empirical_policy")
    write(collected["outcomes"], "outcomes")
    write(collected["paired"], "paired")
    equilibrium_figure(prices)
    if len(prices):
        print(f"=== price (carma, median winning bid, last {LAST} days) by junction type ===")
        print(prices.groupby(["scenario", "strategy", "junction"])["price"]
              .agg(["mean", "std", "count"]).round(1).to_string())


if __name__ == "__main__":
    main()
