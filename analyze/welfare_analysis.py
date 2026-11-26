"""Welfare of per-driver fill rates (formulation.tex, eq. 15) for every strategy of the fixed-rule
study, plus learned bids from the quick study, over the same 7-day evaluation week.

A trip's service is its share of free-flow speed, T_ff / T = (travel time - time lost) / travel
time. A driver's need d_i is their number of trips and their fill rate Z_i the average service
over those trips. Per strategy, cell and seed:
  efficiency  sum S_i / sum d_i  (mean share of free-flow speed over all trips)
  equity      Gini of the Z_i over drivers
  welfare     sum d_i log(1 + Z_i) / sum d_i  (per trip)

Writes results/analysis/params_welfare.csv and analyze/figures/update/welfare_efficiency_equity
.pdf/.png plus figures_welfare.tex.

    python analyze/welfare_analysis.py [workers | redraw]
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
import matplotlib.ticker  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fixed_rules_analysis import COLOR, NAME, NETWORKS, OUT, WIDTH, save  # noqa: E402
from parameter_analysis import INK, INK2, RESULTS, TABLES, gini, interval, settings_of  # noqa: E402

CARS_RE = re.compile(r"^(\w+?)_s(\d+)_eval(\d+)_cars\.csv$")
COLOR["actuated"], NAME["actuated"] = "#52514e", "actuated lights"
ORDER = ["random", "fraction", "wait", "uwait", "ufrac", "learn", "mix", "actuated", "stop",
         "maxpressure"]
MARKERS = {1: "o", 2: "s", 3: "^"}


def read_folder(job: tuple) -> list:
    """Per (strategy, seed) in one folder: efficiency, equity and welfare over the evaluation
    week. job = (folder, strategies to keep or None for all)."""
    folder, keep = job
    s = settings_of(folder)
    files = {}
    for name in os.listdir(folder):
        m = CARS_RE.match(name)
        if m and (keep is None or m[1] in keep):
            files.setdefault((m[1], int(m[2])), []).append(os.path.join(folder, name))
    rows = []
    for (strategy, seed), paths in files.items():
        cars = pd.concat([pd.read_csv(p, usecols=["driver", "travel_time", "time_loss"])
                          for p in paths])
        cars = cars[cars["travel_time"] > 0]
        cars["service"] = (cars["travel_time"] - cars["time_loss"]).clip(lower=0) / cars["travel_time"]
        per = cars.groupby("driver")["service"].agg(["sum", "count"])
        z = per["sum"] / per["count"]
        rows.append({"network": s["network"], "lanes": s["lanes"], "strategy": strategy,
                     "seed": seed, "days": len(paths), "drivers": len(per),
                     "efficiency": 100 * per["sum"].sum() / per["count"].sum(),
                     "equity_gini": gini(z),
                     "welfare": float((per["count"] * np.log1p(z)).sum() / per["count"].sum())})
    return rows


def jobs() -> list:
    out = [(f, None) for f in glob.glob(os.path.join(RESULTS, "fixed", "*_c80"))]
    out += [(f, None) for f in glob.glob(os.path.join(RESULTS, "fixed", "baselines", "*"))]
    out += [(f, {"learn"}) for f in glob.glob(os.path.join(RESULTS, "quick",
                                                           "*_moderate_t6_u80-20_lane_c80"))]
    return [j for j in out if settings_of(j[0])]


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (network, lanes, strategy), g in per_seed.groupby(["network", "lanes", "strategy"]):
        row = {"network": network, "lanes": lanes, "strategy": strategy, "seeds": len(g)}
        for col in ("efficiency", "equity_gini", "welfare"):
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col])
        out.append(row)
    return pd.DataFrame(out)


def figure(table: pd.DataFrame) -> None:
    """One stacked panel per network: efficiency (x) against inequality of fill rates (y), a
    point per strategy and lane count; right and low is better."""
    height = 2.3 * len(NETWORKS) + 0.9
    fig, axes = plt.subplots(len(NETWORKS), 1, figsize=(WIDTH, height), squeeze=False,
                             gridspec_kw={"hspace": 0.5})
    for ax, (network, title) in zip(axes[:, 0], NETWORKS):
        net = table[table["network"] == network]
        for lanes, g in net.groupby("lanes"):
            for _, r in g.iterrows():
                if r["strategy"] not in COLOR:
                    continue
                ax.errorbar(r["efficiency"], r["equity_gini"],
                            xerr=[[r["efficiency"] - r["efficiency_lo"]],
                                  [r["efficiency_hi"] - r["efficiency"]]],
                            yerr=[[r["equity_gini"] - r["equity_gini_lo"]],
                                  [r["equity_gini_hi"] - r["equity_gini"]]],
                            fmt=MARKERS[lanes], ms=5.5, color=COLOR[r["strategy"]], mec=INK2,
                            mew=0.4, ecolor="#b5b4ab", elinewidth=0.5, capsize=0, zorder=3)
        ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.08, y=0.25)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_xlabel("efficiency: mean share of free-flow speed over all trips (%)", labelpad=2)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=COLOR[k], mec=INK2, mew=0.4,
                          label=NAME[k]) for k in ORDER if k in set(table["strategy"])]
    handles += [plt.Line2D([], [], ls="", marker=MARKERS[n], ms=5.5, color="white", mec=INK2,
                           label=f"{n} lane" + ("s" if n > 1 else "")) for n in (1, 2, 3)]
    top = axes[0, 0].get_position()
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.3 / height), handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)
    fig.supylabel("inequality of fill rates across drivers (Gini)", x=0.02, fontsize=9,
                  color=INK2)
    save(fig, "welfare_efficiency_equity")


def latex(seeds: int) -> None:
    text = rf"""% Welfare of per-driver fill rates (analyze/welfare_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{welfare_efficiency_equity.pdf}}
    \caption{{Efficiency against equity of per-driver fill rates (eq.~\eqref{{eq:welfare}}): each
    point is a strategy at one lane count; right is more efficient (trips run at a larger share of
    free-flow speed), low is more equal (smaller Gini of drivers' average share). A driver's need
    is their number of trips. Reference setting, the same 7-day evaluation week for every
    strategy (fixed rules after 7 warm-up days, learned bids after 60 learning days); means of
    {seeds} seeds, whiskers 95\% intervals; each panel has its own scales.}}
    \label{{fig:welfare}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_welfare.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "8"
    if arg == "redraw":  # figure and caption only, from the saved table
        table = pd.read_csv(os.path.join(TABLES, "params_welfare.csv"))
    else:
        with Pool(int(arg)) as pool:
            per_seed = pd.DataFrame([r for rows in pool.map(read_folder, jobs()) for r in rows])
        table = summarise(per_seed)
        table.to_csv(os.path.join(TABLES, "params_welfare.csv"), index=False)
    figure(table)
    latex(int(table["seeds"].min()))
    pd.set_option("display.width", 200)
    for col in ("efficiency", "equity_gini", "welfare"):
        print(col)
        print(table.pivot_table(index=["network", "lanes"], columns="strategy", values=col)
              [[k for k in ORDER if k in set(table["strategy"])]].round(3).to_string())
