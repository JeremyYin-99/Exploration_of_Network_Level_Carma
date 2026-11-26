"""Winners and losers of each policy against actuated lights, trip by trip.

Every plan replays the same evaluation-week trips (same vehicle ids, same scheduled departures),
so each trip under a carma policy can be paired with the very same trip under actuated lights.
The change in its time lost, delta = time lost under the policy - time lost under actuated, says
whether that trip won or lost by the switch. Per policy, cell and seed (pooled over the 7
evaluation days): the share of trips that lose or gain more than THRESHOLD seconds, how much they
lose or gain on average, quantiles of delta, and the tail of time lost itself.

All drivers use the same rule (fixed rules from plan "fixed", 7 warm-up days; learned bids from
plan "price300", 300 learning days); the even mix is shown for comparison.

Writes results/analysis/params_winners_losers.csv (per seed) and params_winners_losers_cells.csv
(means and 95% intervals over seeds), the figures analyze/figures/update/winners_losers_*.pdf/.png
and figures_winners_losers.tex.

    python analyze/winners_losers_analysis.py [workers | redraw]
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
from parameter_analysis import INK, INK2, RESULTS, TABLES, interval  # noqa: E402

THRESHOLD = 5.0   # seconds: smaller changes count as "about the same"
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
POLICIES = ["random", "fraction", "wait", "uwait", "ufrac", "mix", "learn"]
KEYS = RULES + ["learn", "mix"]   # bar order, as in Figure 2
NAME["learn"] = "learned bids (300 learning days)"   # this figure uses plan price300
URGENT = 6.0      # urgency level of urgent trips (calm trips have 1)
EVAL_DAYS = range(1, 8)


def cell_name(network: str, lanes: int) -> str:
    return f"{network}_L{lanes}_moderate_t6_u80-20_lane_c80"


def folder_of(network: str, lanes: int, strategy: str) -> str:
    if strategy == "actuated":
        return os.path.join(RESULTS, "fixed", "baselines", f"{network}_L{lanes}_moderate_t6_u80-20")
    plan = "price300" if strategy == "learn" else "fixed"
    return os.path.join(RESULTS, plan, cell_name(network, lanes))


def week(network: str, lanes: int, strategy: str, seed: int) -> pd.DataFrame | None:
    frames = []
    for e in EVAL_DAYS:
        path = os.path.join(folder_of(network, lanes, strategy), f"{strategy}_s{seed}_eval{e}_cars.csv")
        if not os.path.exists(path):
            return None
        d = pd.read_csv(path, usecols=["vid", "time_loss", "urgency", "driver"])
        frames.append(d.assign(eval_day=e))
    return pd.concat(frames, ignore_index=True)


def paired(job: tuple) -> dict | None:
    """One policy, cell and seed against actuated lights, trip by trip."""
    network, lanes, strategy, seed = job
    x, a = week(network, lanes, strategy, seed), week(network, lanes, "actuated", seed)
    if x is None or a is None:
        return None
    m = x.merge(a[["eval_day", "vid", "time_loss"]], on=["eval_day", "vid"], suffixes=("", "_act"))
    d = (m["time_loss"] - m["time_loss_act"]).to_numpy()
    lose, win = d > THRESHOLD, d < -THRESHOLD
    urgent = m["urgency"].to_numpy() >= URGENT
    row = {"network": network, "lanes": lanes, "strategy": strategy, "seed": seed,
           "trips": len(m), "unpaired": len(x) + len(a) - 2 * len(m),
           "time_lost": m["time_loss"].mean(), "time_lost_act": m["time_loss_act"].mean(),
           "mean_delta": d.mean(),
           "share_lose": 100 * lose.mean(), "share_win": 100 * win.mean(),
           "loss_mean": d[lose].mean() if lose.any() else 0.0,
           "gain_mean": -d[win].mean() if win.any() else 0.0,
           "share_lose_urgent": 100 * lose[urgent].mean(), "share_lose_calm": 100 * lose[~urgent].mean(),
           "share_win_urgent": 100 * win[urgent].mean(), "share_win_calm": 100 * win[~urgent].mean()}
    for q in QUANTILES:
        row[f"delta_q{q}"] = float(np.percentile(d, q))
    for q in (50, 90, 99):
        row[f"tl_q{q}"] = float(np.percentile(m["time_loss"], q))
        row[f"tl_act_q{q}"] = float(np.percentile(m["time_loss_act"], q))
    return row


def jobs() -> list:
    return [(n, lanes, s, seed) for n in ("single", "grid3", "grid5") for lanes in (1, 2, 3)
            for s in POLICIES for seed in range(10)]


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    per_seed = per_seed.assign(tl_q99_vs_act=100 * (per_seed["tl_q99"] / per_seed["tl_act_q99"] - 1),
                               tl_q90_vs_act=100 * (per_seed["tl_q90"] / per_seed["tl_act_q90"] - 1))
    values = [c for c in per_seed.columns if c not in ("network", "lanes", "strategy", "seed")]
    out = []
    for (network, lanes, strategy), g in per_seed.groupby(["network", "lanes", "strategy"]):
        row = {"network": network, "lanes": lanes, "strategy": strategy, "seeds": len(g)}
        for col in values:
            row[col], row[col + "_lo"], row[col + "_hi"] = interval(g[col])
        out.append(row)
    return pd.DataFrame(out)


def range_bars(table: pd.DataFrame, name: str) -> None:
    """One stacked panel per network; per lane count a group of range bars, one per policy, of the
    change in time lost per trip against actuated lights: box = middle half of trips (25th-75th
    percentile), line = 5th-95th percentile, white tick = median (means over seeds)."""
    height = 2.2 * len(NETWORKS) + 0.6
    fig, axes = plt.subplots(len(NETWORKS), 1, figsize=(WIDTH, height), squeeze=False,
                             gridspec_kw={"hspace": 0.55})
    width = 0.8 / len(KEYS)
    for ax, (network, title) in zip(axes[:, 0], NETWORKS):
        net = table[table["network"] == network]
        for i, key in enumerate(KEYS):
            r = net[net["strategy"] == key].sort_values("lanes")
            if r.empty:
                continue
            xs = r["lanes"] + (i - (len(KEYS) - 1) / 2) * width
            ax.vlines(xs, r["delta_q5"], r["delta_q95"], color=COLOR[key], lw=1.0, zorder=2)
            ax.bar(xs, r["delta_q75"] - r["delta_q25"], width * 0.8, bottom=r["delta_q25"],
                   color=COLOR[key], label=NAME[key], lw=0, zorder=3)
            ax.hlines(r["delta_q50"], xs - width * 0.3, xs + width * 0.3, color="white", lw=1.2,
                      zorder=4)
        ax.axhline(0, color="#8a897f", lw=0.8, zorder=1)
        ax.text(3.45, 0, "same as\nactuated", ha="left", va="center", fontsize=7.5, color=INK2,
                linespacing=1.1)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["1 lane", "2 lanes", "3 lanes"])
        ax.set_xlim(0.5, 3.5)
        ax.set_xlabel("lanes per road", labelpad=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:+g} s".replace("-", "\u2212") if v else "0 s"))
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
    found = {}
    for ax in axes[:, 0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    labels = [NAME[k] for k in KEYS if NAME[k] in found]
    top = axes[0, 0].get_position()
    fig.legend([found[lab] for lab in labels], labels, loc="lower left",
               bbox_to_anchor=(top.x0 - 0.01, top.y1 + 0.3 / height), ncol=2, frameon=False,
               fontsize=8, handlelength=1.2, columnspacing=1.5, borderaxespad=0)
    fig.supylabel("change in time lost per trip vs the same trip under actuated lights (s)",
                  x=0.02, fontsize=9, color=INK2)
    save(fig, name)


def latex(seeds: int) -> None:
    text = rf"""% Winners and losers against actuated lights, trip by trip (analyze/winners_losers_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{winners_losers_spread.pdf}}
    \caption{{Who wins and who loses when every driver uses the same rule: each trip of the
    evaluation week is paired with the very same trip under actuated lights, and the change in
    its time lost is spread over the trips. Box: the middle half of trips (25th to 75th
    percentile); line: 5th to 95th percentile; white tick: median. Below 0 the trip is faster than
    under actuated lights. Reference setting; fixed rules after 7 warm-up days, learned bids after
    300 learning days; percentiles are means of {seeds} seeds; each panel has its own scale.}}
    \label{{fig:winners_losers_spread}}
\end{{figure}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{winners_losers_tail.pdf}}
    \caption{{The worst-off trips: time lost by the slowest 1\% of trips (99th percentile),
    compared with the slowest 1\% under actuated lights on the same trips. Settings as in
    Figure~\ref{{fig:winners_losers_spread}}; bars are means of {seeds} seeds, whiskers 95\%
    intervals; each panel has its own scale.}}
    \label{{fig:winners_losers_tail}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_winners_losers.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "4"
    if arg == "redraw":   # figures from the saved per-seed table
        per_seed = pd.read_csv(os.path.join(TABLES, "params_winners_losers.csv"))
    else:
        with Pool(int(arg)) as pool:
            per_seed = pd.DataFrame([r for r in pool.map(paired, jobs()) if r is not None])
        per_seed.to_csv(os.path.join(TABLES, "params_winners_losers.csv"), index=False)
    table = summarise(per_seed)
    table.to_csv(os.path.join(TABLES, "params_winners_losers_cells.csv"), index=False)
    range_bars(table, "winners_losers_spread")
    fra.grouped_bars(table, "tl_q99_vs_act", KEYS,
                     "time lost by the slowest 1% of trips vs actuated lights (%)",
                     "winners_losers_tail", "same as actuated")
    latex(int(table["seeds"].min()))
    pd.set_option("display.width", 250)
    cols = ["network", "lanes", "strategy", "seeds", "unpaired", "time_lost", "time_lost_act",
            "mean_delta", "share_lose", "loss_mean", "share_win", "gain_mean", "delta_q5",
            "delta_q25", "delta_q50", "delta_q75", "delta_q95", "tl_q99", "tl_act_q99",
            "tl_q99_vs_act", "share_lose_urgent", "share_lose_calm"]
    print(table[cols].round(1).to_string(index=False))
