"""Pooled group bids against front-car bids, every car alone and actuated lights (plan "pooled").

Bidding units:
  vehicle    every car within 50 m bids alone for its own movement;
  lane       the front car of each lane bids for the whole lane (plan "fixed");
  pool       every car of a lane bids from its own balance by its own rule, the bids are summed;
  phase      the front car of each signal phase bids for the phase;
  phasepool  every car of a phase bids, the bids are summed.
In the pooled units a winning group pays only when contested, each car paying its own bid, and the
payment is shared among the cars of the losing groups.

All units replay the same evaluation-week trips (7 warm-up days first), so each run is paired with
actuated lights on the same trips and seed. Per network, lane count, unit, rule and seed (pooled
over the 7 evaluation days): time lost per trip and its change against actuated lights, the
urgency-weighted cost, how much less time urgent trips lose than calm ones, the slowest 1% of
trips, and from the bid logs the share of cars that bid, the winning group's size and amount, and
carma paid per trip.

Writes results/analysis/params_pooled.csv (per seed) and params_pooled_cells.csv (means and 95%
intervals over seeds).

    python analyze/pooled_analysis.py [workers | redraw]
"""
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fixed_rules_analysis as fra  # noqa: E402  (also sets the update style)
from fixed_rules_analysis import NAME, OUT, WIDTH, save  # noqa: E402
from parameter_analysis import INK, INK2, RESULTS, TABLES, interval  # noqa: E402

UNITS = ["vehicle", "lane", "pool", "phase", "phasepool"]
# Bar colours and legend names of the bidding units (light = front car, dark = pooled).
UNIT_STYLE = {"vehicle": ("every car bids alone", "#bdbcb3"),
              "lane": ("lane: front car bids", "#86b6ef"),
              "pool": ("lane: all cars bid, pooled", "#104281"),
              "phase": ("phase: front car bids", "#f7c0a0"),
              "phasepool": ("phase: all cars bid, pooled", "#b84a1c")}
for _unit, (_name, _colour) in UNIT_STYLE.items():
    fra.NAME[_unit], fra.COLOR[_unit] = _name, _colour
RULES = ["random", "fraction", "wait", "uwait", "ufrac", "mix"]
NETWORK_LIST = ("single", "grid3", "grid5")
EVAL_DAYS = range(1, 8)
URGENT = 6.0


def folder(network: str, lanes: int, unit: str, strategy: str) -> str:
    if strategy == "actuated":
        return os.path.join(RESULTS, "fixed", "baselines", f"{network}_L{lanes}_moderate_t6_u80-20")
    plan = "fixed" if unit == "lane" else "pooled"
    return os.path.join(RESULTS, plan, f"{network}_L{lanes}_moderate_t6_u80-20_{unit}_c80")


def week(path: str, strategy: str, seed: int, what: str = "cars") -> pd.DataFrame | None:
    frames = []
    for e in EVAL_DAYS:
        suffix = "cars.csv" if what == "cars" else "bids.csv.gz"
        f = os.path.join(path, f"{strategy}_s{seed}_eval{e}_{suffix}")
        if not os.path.exists(f):
            return None
        frames.append(pd.read_csv(f).assign(eval_day=e))
    return pd.concat(frames, ignore_index=True)


def per_seed(job: tuple) -> dict | None:
    network, lanes, unit, strategy, seed = job
    x = week(folder(network, lanes, unit, strategy), strategy, seed)
    a = week(folder(network, lanes, unit, "actuated"), "actuated", seed)
    if x is None or a is None:
        return None
    m = x.merge(a[["eval_day", "vid", "time_loss"]], on=["eval_day", "vid"], suffixes=("", "_act"))
    urgent = m["urgency"] >= URGENT
    row = {"network": network, "lanes": lanes, "unit": unit, "strategy": strategy, "seed": seed,
           "trips": len(x), "unpaired": len(x) + len(a) - 2 * len(m),
           "time_lost": x["time_loss"].mean(), "time_lost_act": a["time_loss"].mean(),
           "cost": (m["time_loss"] * m["urgency"]).mean(),
           "cost_act": (m["time_loss_act"] * m["urgency"]).mean(),
           "tl_urgent": m.loc[urgent, "time_loss"].mean(), "tl_calm": m.loc[~urgent, "time_loss"].mean(),
           "tl_q99": float(np.percentile(m["time_loss"], 99)),
           "tl_act_q99": float(np.percentile(m["time_loss_act"], 99))}
    row["vs_act"] = 100 * (row["time_lost"] / row["time_lost_act"] - 1)
    row["cost_vs_act"] = 100 * (row["cost"] / row["cost_act"] - 1)
    row["urgent_vs_calm"] = 100 * (row["tl_urgent"] / row["tl_calm"] - 1)
    row["q99_vs_act"] = 100 * (row["tl_q99"] / row["tl_act_q99"] - 1)
    b = week(folder(network, lanes, unit, strategy), strategy, seed, "bids")
    if b is not None and len(b):
        row["bid_rows"] = len(b)
        row["share_bidding"] = 100 * (b["bid"] > 0).mean()
        row["paid_per_trip"] = b["payment"].sum() / len(x)
        won = b[b["won"]]
        groups = won.groupby(["auction_id", "group_amount"]).size()
        row["win_group_cars"] = float(groups.mean()) if len(groups) else np.nan
        row["win_amount"] = float(won.groupby("auction_id")["group_amount"].max().mean()) if len(won) else np.nan
    return row


def jobs() -> list:
    return [(n, lanes, u, s, seed) for n in NETWORK_LIST for lanes in (1, 2, 3) for u in UNITS
            for s in RULES for seed in range(10)]


def summarise(seeds: pd.DataFrame) -> pd.DataFrame:
    values = [c for c in seeds.columns if c not in ("network", "lanes", "unit", "strategy", "seed")]
    out = []
    for (network, lanes, unit, strategy), g in seeds.groupby(["network", "lanes", "unit", "strategy"]):
        row = {"network": network, "lanes": lanes, "unit": unit, "strategy": strategy, "seeds": len(g)}
        for col in values:
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col].dropna())
        out.append(row)
    return pd.DataFrame(out)


def pooling_effect(seeds: pd.DataFrame) -> pd.DataFrame:
    """Pooled minus front-car (same rule, cell and seed): change in time lost, % of front-car."""
    rows = []
    for pooled, front in (("pool", "lane"), ("phasepool", "phase")):
        p = seeds[seeds["unit"] == pooled].set_index(["network", "lanes", "strategy", "seed"])
        f = seeds[seeds["unit"] == front].set_index(["network", "lanes", "strategy", "seed"])
        both = p.index.intersection(f.index)
        d = 100 * (p.loc[both, "time_lost"] / f.loc[both, "time_lost"] - 1)
        for (network, lanes, strategy), g in d.groupby(level=["network", "lanes", "strategy"]):
            mean, lo, hi = interval(g)
            rows.append({"network": network, "lanes": lanes, "grouping": front, "strategy": strategy,
                         "seeds": len(g), "pooled_vs_front": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def effect_figure(effect: pd.DataFrame, name: str) -> None:
    """Two stacked heatmaps (lane groups, phase groups): rows = rules, columns = network x lanes,
    cell = pooled minus front-car time lost (% of front-car); bold when the 95% interval over
    seeds excludes 0."""
    rules = ["random", "fraction", "wait", "uwait", "ufrac", "mix"]
    columns = [(n, lanes) for n in ("single", "grid3", "grid5") for lanes in (1, 2, 3)]
    cmap = LinearSegmentedColormap.from_list("pool", ["#2a78d6", "#f4f3ee", "#b84a1c"])
    limit = 8.0
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.4), gridspec_kw={"hspace": 0.55})
    for ax, (grouping, title) in zip(axes, (("lane", "Lane groups: all cars pooled vs front car"),
                                            ("phase", "Phase groups: all cars pooled vs front car"))):
        e = effect[effect["grouping"] == grouping].set_index(["strategy", "network", "lanes"])
        grid = np.array([[e.loc[(r, n, lanes), "pooled_vs_front"] for n, lanes in columns]
                         for r in rules])
        ax.imshow(grid, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        for i, r in enumerate(rules):
            for j, (n, lanes) in enumerate(columns):
                row = e.loc[(r, n, lanes)]
                significant = row["hi"] < 0 or row["lo"] > 0
                text = f"{row['pooled_vs_front']:+.1f}".replace("-", "−")
                ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                        color=INK if significant else "#8a897f",
                        fontweight="bold" if significant else "normal")
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels([NAME[r] for r in rules], fontsize=7.5)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([f"{lanes} lane" + ("s" if lanes > 1 else "") for _, lanes in columns],
                           fontsize=7, rotation=0)
        for j, label in ((1, "Single intersection"), (4, "3x3 grid"), (7, "5x5 grid")):
            ax.text(j, -0.85, label, ha="center", va="bottom", fontsize=8, color=INK2)
        for x in (2.5, 5.5):
            ax.axvline(x, color="white", lw=3)
        ax.tick_params(axis="both", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=30)
    fig.text(0.5, 0.0, "change in time lost per trip, pooled vs front-car bids on the same trips (%)\n"
             "below 0: pooling is faster; bold: the 95% interval excludes 0",
             ha="center", va="top", fontsize=8, color=INK2, linespacing=1.4)
    save(fig, name)


def latex(seeds: int) -> None:
    text = rf"""% Pooled group bids (analyze/pooled_analysis.py, plan "pooled").
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{pooled_units.pdf}}
    \caption{{Time lost per trip under each bidding unit, compared with actuated lights (0\%) on
    the same trips, for the even mix of the five fixed rules. Every car bids alone; the front car
    of each lane or signal phase bids for it; or every car of the lane or phase bids from its own
    balance by its own rule and the bids are summed (pooled; each winning car pays its own bid
    when contested, and the payment is shared among the cars of the losing groups). Reference
    setting, 7 warm-up days, then the same 7-day evaluation week; bars are means of {seeds}
    seeds, whiskers 95\% intervals; each panel has its own scale.}}
    \label{{fig:pooled_units}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{pooled_effect.pdf}}
    \caption{{What pooling changes, rule by rule: time lost per trip with every car of the group
    bidding (pooled) against only the front car bidding, same rule, trips and seed, as a
    percentage of the front-car time lost. Blue: pooling is faster. Bold: the 95\% interval over
    {seeds} seeds excludes 0. Settings as in Figure~\ref{{fig:pooled_units}}.}}
    \label{{fig:pooled_effect}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_pooled.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "8"
    if arg == "redraw":
        seeds = pd.read_csv(os.path.join(TABLES, "params_pooled.csv"))
    else:
        with Pool(int(arg)) as pool:
            seeds = pd.DataFrame([r for r in pool.map(per_seed, jobs()) if r is not None])
        seeds.to_csv(os.path.join(TABLES, "params_pooled.csv"), index=False)
    table = summarise(seeds)
    table.to_csv(os.path.join(TABLES, "params_pooled_cells.csv"), index=False)
    effect = pooling_effect(seeds)
    effect.to_csv(os.path.join(TABLES, "params_pooled_effect.csv"), index=False)
    mix = table[table["strategy"] == "mix"].drop(columns="strategy").rename(columns={"unit": "strategy"})
    fra.grouped_bars(mix, "vs_act", UNITS, "time lost per trip vs actuated lights (%)",
                     "pooled_units", "same as\nactuated")
    effect_figure(effect, "pooled_effect")
    latex(int(table["seeds"].min()))
    pd.set_option("display.width", 250)
    for value in ("vs_act", "urgent_vs_calm", "paid_per_trip"):
        print(f"\n{value} (mean of seeds)")
        print(table.pivot_table(index=["strategy", "unit"], columns=["network", "lanes"],
                                values=value).round(1).to_string())
    print("\npooled vs front-car, time lost (%)")
    print(effect.pivot_table(index=["grouping", "strategy"], columns=["network", "lanes"],
                             values="pooled_vs_front").round(1).to_string())
