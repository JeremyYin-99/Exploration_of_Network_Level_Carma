"""Commuters (plan "commute", 2026-11-21): every driver drives the same trips every day, so each
driver has a lasting position (a long or a busy route) and "losers" are people, not trips.

Per policy, cell and seed, each driver's evaluation week is paired with the same driver's week
under actuated lights (the same trips; a few trips are swapped when their driver is still out):
  - the share of drivers whose time lost per trip is more than THRESHOLD seconds worse (better)
    than under actuated lights, and by how much;
  - how unequal the burden is: Gini of drivers' time lost per trip, and the slowest 10% of drivers
    against the slowest 10% under actuated lights;
  - persistence: rank correlation of a driver's time lost on evaluation days 1-3 and days 4-7
    (1 = the same people lose every day);
  - compensation: drivers grouped into fifths by their time lost under actuated lights on days
    1-3, and each fifth's change in time lost on days 4-7 (separate days, so day-to-day noise
    does not leak into the grouping). A change that grows from the fastest to the slowest fifth
    means the losers lose more; one that shrinks means the policy compensates them.

Writes results/analysis/params_commute.csv (per seed), params_commute_cells.csv and
params_commute_fifths.csv, the figures analyze/figures/update/commute_fifths.pdf/.png and
commute_drivers_tail.pdf/.png, and figures_commute.tex.

    python analyze/commute_analysis.py [workers | redraw]
"""
import os
import sys
from multiprocessing import Pool

import matplotlib
import matplotlib.ticker  # noqa: F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fixed_rules_analysis as fra  # noqa: E402  (also sets the update style)
from fixed_rules_analysis import COLOR, NAME, NETWORKS, OUT, RULES, WIDTH, save  # noqa: E402
from parameter_analysis import INK, INK2, RESULTS, TABLES, gini, interval  # noqa: E402

PLAN = os.path.join(RESULTS, "commute")
THRESHOLD = 5.0
KEYS = RULES + ["learn"]
NAME["learn"] = "learned bids (300 learning days)"
LANES = (1, 2)
FIFTHS = 5
EARLY, LATE = (1, 2, 3), (4, 5, 6, 7)


def cell_name(network: str, lanes: int) -> str:
    return f"{network}_L{lanes}_moderate_t6_u80-20_lane_c80"


def week(network: str, lanes: int, strategy: str, seed: int) -> pd.DataFrame | None:
    folder = (os.path.join(PLAN, "baselines", f"{network}_L{lanes}_moderate_t6_u80-20")
              if strategy == "actuated" else os.path.join(PLAN, cell_name(network, lanes)))
    frames = []
    for e in EARLY + LATE:
        path = os.path.join(folder, f"{strategy}_s{seed}_eval{e}_cars.csv")
        if not os.path.exists(path):
            return None
        frames.append(pd.read_csv(path, usecols=["vid", "driver", "time_loss", "urgency",
                                                 "final_carma"]).assign(eval_day=e))
    return pd.concat(frames, ignore_index=True)


def per_driver(d: pd.DataFrame, days) -> pd.Series:
    return d[d["eval_day"].isin(days)].groupby("driver")["time_loss"].mean()


def analyse(job: tuple) -> tuple[dict, list] | None:
    network, lanes, strategy, seed = job
    x, a = week(network, lanes, strategy, seed), week(network, lanes, "actuated", seed)
    if x is None or a is None:
        return None
    all_days = EARLY + LATE
    tx, ta = per_driver(x, all_days), per_driver(a, all_days)
    both = tx.index.intersection(ta.index)
    tx, ta = tx[both], ta[both]
    delta = tx - ta
    lose, win = delta > THRESHOLD, delta < -THRESHOLD
    k = max(1, len(both) // 10)
    row = {"network": network, "lanes": lanes, "strategy": strategy, "seed": seed,
           "drivers": len(both), "time_lost": tx.mean(), "time_lost_act": ta.mean(),
           "share_lose": 100 * lose.mean(), "share_win": 100 * win.mean(),
           "loss_mean": delta[lose].mean() if lose.any() else 0.0,
           "gain_mean": -delta[win].mean() if win.any() else 0.0,
           "gini": gini(tx), "gini_act": gini(ta),
           "worst10": tx.nlargest(k).mean(), "worst10_act": ta.nlargest(k).mean()}
    row["worst10_vs_act"] = 100 * (row["worst10"] / row["worst10_act"] - 1)
    row["gini_vs_act"] = row["gini"] - row["gini_act"]
    for q in (5, 25, 50, 75, 95):
        row[f"delta_q{q}"] = float(np.percentile(delta, q))
    early, late = per_driver(x, EARLY), per_driver(x, LATE)
    common = early.index.intersection(late.index)
    row["persistence"] = early[common].rank().corr(late[common].rank())
    early_a, late_a = per_driver(a, EARLY), per_driver(a, LATE)
    common_a = early_a.index.intersection(late_a.index)
    row["persistence_act"] = early_a[common_a].rank().corr(late_a[common_a].rank())
    # Compensation: fifths by actuated time lost on days 1-3, change on days 4-7.
    late_x = per_driver(x, LATE)
    ids = early_a.index.intersection(late_x.index).intersection(late_a.index)
    fifth = pd.qcut(early_a[ids].rank(method="first"), FIFTHS, labels=False)
    balance = x[x["eval_day"] == LATE[-1]].groupby("driver")["final_carma"].last()
    fifths = []
    for f in range(FIFTHS):
        members = fifth[fifth == f].index
        fifths.append({"network": network, "lanes": lanes, "strategy": strategy, "seed": seed,
                       "fifth": f + 1, "time_lost": late_x[members].mean(),
                       "time_lost_act": late_a[members].mean(),
                       "change": 100 * (late_x[members].mean() / late_a[members].mean() - 1),
                       "balance": balance.reindex(members).mean()})
    return row, fifths


def jobs() -> list:
    return [(n, lanes, s, seed) for n, _ in NETWORKS for lanes in LANES for s in KEYS
            for seed in range(10)]


def summarise(per_seed: pd.DataFrame, by: list) -> pd.DataFrame:
    values = [c for c in per_seed.columns if c not in by + ["seed"]]
    out = []
    for key, g in per_seed.groupby(by):
        row = dict(zip(by, key)) | {"seeds": len(g)}
        for col in values:
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col])
        out.append(row)
    return pd.DataFrame(out)


def fifths_figure(fifths: pd.DataFrame, name: str) -> None:
    """Rows = networks, columns = lane counts; a line per policy over the five groups of drivers
    (fastest to slowest commute under actuated lights) with its 95% band."""
    height = 2.0 * len(NETWORKS) + 0.9
    fig, axes = plt.subplots(len(NETWORKS), len(LANES), figsize=(WIDTH, height), squeeze=False,
                             gridspec_kw={"hspace": 0.6, "wspace": 0.28})
    x = np.arange(1, FIFTHS + 1)
    for i, (network, title) in enumerate(NETWORKS):
        for j, lanes in enumerate(LANES):
            ax = axes[i][j]
            t = fifths[(fifths["network"] == network) & (fifths["lanes"] == lanes)]
            for key in KEYS:
                r = t[t["strategy"] == key].set_index("fifth").reindex(x)
                if r["change"].notna().sum() == 0:
                    continue
                ax.plot(x, r["change"], color=COLOR[key], lw=1.5, marker="o", ms=3,
                        label=NAME[key])
                ax.fill_between(x, r["change_lo"], r["change_hi"], color=COLOR[key], alpha=0.12,
                                lw=0)
            ax.axhline(0, color="#8a897f", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(["fastest", "2nd", "3rd", "4th", "slowest"], fontsize=7.5)
            ax.set_xlim(0.7, FIFTHS + 0.3)
            ax.set_xlabel("fifth of drivers by time lost under actuated lights", fontsize=7.5,
                          labelpad=2)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:+g}%".replace("-", "−") if v else "0%"))
            ax.tick_params(axis="both", length=0, labelsize=7.5)
            ax.grid(axis="y", color="#e1e0d9", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{title}, {lanes} lane" + ("s" if lanes > 1 else ""), loc="left",
                         fontweight="bold", color=INK, fontsize=8.5)
    found = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            found.setdefault(lab, h)
    labels = [NAME[k] for k in KEYS if NAME[k] in found]
    top = axes[0, 0].get_position()
    fig.legend([found[lab] for lab in labels], labels, loc="lower left", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.35 / height),
               borderaxespad=0)
    fig.supylabel("change in the group's time lost per trip vs actuated lights (%)", x=0.015,
                  fontsize=9, color=INK2)
    save(fig, name)


def latex(seeds: int) -> None:
    text = rf"""% Commuters: who keeps losing (analyze/commute_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{commute_fifths.pdf}}
    \caption{{Commuters, who drive the same trips every day: drivers are split into fifths by
    their time lost per trip under actuated lights (evaluation days 1--3), and each fifth's time
    lost under every policy is compared with its time lost under actuated lights on days 4--7. A
    line rising to the right means the drivers who already lose most lose even more; a falling
    line means the policy compensates them. In seconds, at 1 lane the slowest fifth gains about
    twice as much as the fastest; at 2 lanes on the grids every fifth loses a similar number of
    seconds, so the percentage falls. Every driver uses the same rule; fixed rules after
    60 warm-up days, learned bids after 300 learning days; means of {seeds} seeds, bands 95\%
    intervals; each panel has its own scale.}}
    \label{{fig:commute_fifths}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{commute_drivers_tail.pdf}}
    \caption{{Commuters: average time lost per trip of the slowest 10\% of drivers over the
    evaluation week, compared with the slowest 10\% under actuated lights. Settings as in
    Figure~\ref{{fig:commute_fifths}}; bars are means of {seeds} seeds, whiskers 95\%
    intervals.}}
    \label{{fig:commute_drivers_tail}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_commute.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "4"
    if arg == "redraw":
        per_seed = pd.read_csv(os.path.join(TABLES, "params_commute.csv"))
        fifths_seed = pd.read_csv(os.path.join(TABLES, "params_commute_fifths_seed.csv"))
    else:
        with Pool(int(arg)) as pool:
            out = [r for r in pool.map(analyse, jobs()) if r is not None]
        per_seed = pd.DataFrame([r for r, _ in out])
        fifths_seed = pd.DataFrame([f for _, fs in out for f in fs])
        per_seed.to_csv(os.path.join(TABLES, "params_commute.csv"), index=False)
        fifths_seed.to_csv(os.path.join(TABLES, "params_commute_fifths_seed.csv"), index=False)
    table = summarise(per_seed, ["network", "lanes", "strategy"])
    table.to_csv(os.path.join(TABLES, "params_commute_cells.csv"), index=False)
    fifths = summarise(fifths_seed, ["network", "lanes", "strategy", "fifth"])
    fifths.to_csv(os.path.join(TABLES, "params_commute_fifths.csv"), index=False)
    fifths_figure(fifths, "commute_fifths")
    fra.grouped_bars(table, "worst10_vs_act", KEYS,
                     "time lost per trip of the slowest 10% of drivers vs actuated lights (%)",
                     "commute_drivers_tail", "same as actuated", lanes=LANES)
    latex(int(table["seeds"].min()))
    pd.set_option("display.width", 250)
    cols = ["network", "lanes", "strategy", "seeds", "drivers", "time_lost", "time_lost_act",
            "share_lose", "loss_mean", "share_win", "gain_mean", "gini", "gini_act",
            "worst10_vs_act", "persistence", "persistence_act"]
    print(table[cols].round(3).to_string(index=False))
    print(fifths.pivot_table(index=["network", "lanes", "strategy"], columns="fifth",
                             values="change").round(1).to_string())
    print(fifths.pivot_table(index=["network", "lanes", "strategy"], columns="fifth",
                             values="balance").round(1).to_string())
