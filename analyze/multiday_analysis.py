"""Multi-day carma results: day-by-day delay and carma inequality, resetting carma every k days,
and fine-grained time series over the whole chain.

A chain runs the same drivers for several days (7 by default, 14 with --days 14); balances carry
over from one day to the next. Reset chains (controller/multiday.py --resets) put every driver
back at the starting balance every k days. Their days 1..k are identical to the no-reset chain's
(same trips, seeds and balances), so they are taken from it; later days come from the reset chain.
Every day of a reset chain sees the same trips as the same day of the no-reset chain, so a reset
is judged by the paired difference on the days after its first reset, day by day (see paired.py).
A chain without resets is shown as k = never.

Run from the repo root:
    python analyze/multiday_analysis.py                          # results/multiday, both scenarios
    python analyze/multiday_analysis.py --results multiday14 grid   # another results folder

Inputs: results/<results>/{single,grid} (written by controller/multiday.py). Outputs, named
<results>_<scenario>_... (e.g. multiday_grid_days.png):
  figures in analyze/figures/:
    _days.png              mean wait, mean time lost and carma Gini by day (no reset solid,
                           resets dashed)
    _carma_finegrain.png   carma Gini at every checkpoint (resets dashed)
    _timeseries.png        at every checkpoint, chains without resets: carma Gini, top-10% share
                           and largest balance, share of cars stopped, and the mean wait and
                           time lost of the trips that arrived in each checkpoint interval
  tables in results/analysis/:
    _by_age.csv            every metric by days since the last reset
    _by_reset.csv          mean over all days, per strategy, demand and reset interval
    _reset_paired.csv      reset vs no reset, paired by (seed, day), days after the first reset
    _paired.csv            every strategy vs stop signs, paired by (seed, day)
    _by_week.csv           chains longer than 7 days: week by week (wait, time lost, Gini, its
                           daily slope, and the paired differences to stop signs)

Two delay measures are reported. "wait" is SUMO's waiting time (seconds spent below 0.1 m/s);
"time lost" is the delay against free-flow travel (SUMO timeLoss), which also counts slowing down
and crawling. Stop signs make every car slow down and creep across, so their wait is small while
their time lost is not: compare strategies with stop signs on time lost.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paired import paired, show

# Folders, found relative to this file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analyze", "figures")
TABLES = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)
# Length of one simulated day in seconds (trip window + drain time), per scenario.
DAY_LEN = {"single": 3600 + 900, "grid": 1800 + 900}
# Label of the schedule without resets.
NEVER = "never"
# Line style per reset interval (0 = no reset).
STYLES = {0: "-", 3: "--", 5: ":", 7: "-."}


def _reset_every(run: dict) -> int:
    """The reset interval of a run (0 = no reset; runs written before resets existed have none)."""
    value = run.get("reset_every", 0)
    return 0 if pd.isna(value) else int(value)


def _time_days(day, t, scenario):
    """Time in days from the start of the chain: day-1 plus the position in the day."""
    return (day - 1) + t.clip(upper=DAY_LEN[scenario]) / DAY_LEN[scenario]


def load(results, scenario):
    """Read every finished day of results/<results>/<scenario>. Returns four tables:
    one row per day; the carma checkpoints (all drivers); the network checkpoints; and the mean
    wait and time lost of the trips that arrived in each checkpoint interval."""
    days, traces, network, waits = [], [], [], []
    base = os.path.join(ROOT, "results", results, scenario)
    for f in glob.glob(os.path.join(base, "*_run.csv")):
        run = pd.read_csv(f).iloc[0].to_dict()           # run-level numbers of this day
        cars = pd.read_csv(f.replace("_run.csv", "_cars.csv"))   # one row per car
        run["mean_wait"] = cars["wait_time"].mean()
        run["p95_wait"] = cars["wait_time"].quantile(0.95)
        run["mean_travel"] = cars["travel_time"].mean()
        run["mean_loss"] = cars["time_loss"].mean()
        run["p95_loss"] = cars["time_loss"].quantile(0.95)
        run["name"] = run["experiment_name"]
        run["seed"] = run["chain_seed"]
        run["reset_every"] = _reset_every(run)
        keys = {k: run[k] for k in ("name", "p", "seed", "day", "reset_every")}
        # The carma checkpoints of this day (only exist for strategies that spend carma).
        tf = f.replace("_run.csv", "_carma_trace.csv")
        gini = float("nan")
        if os.path.exists(tf):
            tr = pd.read_csv(tf)
            allrows = tr[tr["kind"] == "all"].assign(**keys)     # the summary over all drivers
            allrows["time_days"] = _time_days(run["day"], allrows["t"], scenario)
            traces.append(allrows)
            gini = allrows.iloc[-1]["gini"]               # inequality at the end of the day
        run["end_gini"] = gini
        # The network checkpoints (every run, written since the network trace was added).
        interval = DAY_LEN[scenario] / 16
        nf = f.replace("_run.csv", "_network_trace.csv")
        if os.path.exists(nf):
            nt = pd.read_csv(nf).assign(**keys)
            nt["time_days"] = _time_days(run["day"], nt["t"], scenario)
            network.append(nt)
            if len(nt) > 2:
                interval = float(nt["t"].diff().median())
        # Mean wait and time lost of the trips that arrived in each checkpoint interval (centre
        # of the bin).
        bins = (cars["arrive"] // interval) * interval + interval / 2
        w = cars.groupby(bins).agg(mean_wait=("wait_time", "mean"),
                                   mean_loss=("time_loss", "mean"),
                                   trips=("wait_time", "size")).reset_index()
        w = w.rename(columns={w.columns[0]: "t"})
        w = w.assign(**keys)
        w["time_days"] = _time_days(run["day"], w["t"], scenario)
        waits.append(w)
        days.append(run)

    def cat(parts):
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return pd.DataFrame(days), cat(traces), cat(network), cat(waits)


RUN = ["name", "p", "seed", "day", "reset_every"]


def align(data, interval, scenario, fill):
    """Put every run's checkpoints on one grid (0, interval, 2 * interval, ...), so that averaging
    over seeds compares the same moments. The last checkpoint of a day falls at a different time
    for every seed (a day ends when its network is empty); it moves to the next grid point. With
    fill, a run that ended early keeps its last values up to the latest end among the runs of that
    day (right for carma balances, which no longer change once every car has arrived)."""
    if data.empty:
        return data
    data = data.copy()
    data["t"] = np.ceil(data["t"] / interval - 1e-9) * interval
    data = data.drop_duplicates(RUN + ["t"], keep="last")
    if fill:
        ends = data.groupby(["name", "p", "day", "reset_every"])["t"].transform("max")
        parts = []
        for _, g in data.assign(_end=ends).groupby(RUN):
            grid = np.arange(0, g["_end"].iloc[0] + interval / 2, interval)
            parts.append(g.set_index("t").reindex(grid).ffill().rename_axis("t").reset_index())
        data = pd.concat(parts, ignore_index=True).drop(columns="_end")
    data["time_days"] = _time_days(data["day"], data["t"], scenario)
    return data


def schedules(d: pd.DataFrame) -> pd.DataFrame:
    """One row per day of every chain under every reset schedule. Column `k` is the reset interval
    (0 = never) and `age` the days since the last reset (day 1 = a fresh start).

    The no-reset chain is schedule 0. A reset-k schedule takes days 1..k from the no-reset chain
    (identical runs) and days k+1.. from its reset chain, for the chains (strategy, demand, seed)
    whose reset chain has at least one finished day."""
    chain = ["name", "p", "seed"]
    base = d[d["reset_every"] == 0]
    out = [base.assign(k=0, age=base["day"])]
    for k in sorted(set(d["reset_every"]) - {0}):
        reset = d[d["reset_every"] == k]
        early = base[base["day"] <= k].merge(reset[chain].drop_duplicates(), on=chain)
        both = pd.concat([early, reset], ignore_index=True)
        out.append(both.assign(k=k, age=(both["day"] - 1) % k + 1))
    s = pd.concat(out, ignore_index=True)
    s["schedule"] = s["k"].map(lambda k: NEVER if k == 0 else f"every {k}")
    return s


def _colors(names):
    """One fixed colour per strategy, so its reset schedules share its colour."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {n: ("black" if n.startswith("stop_signs") else cycle[i % len(cycle)])
            for i, n in enumerate(sorted(names))}


def _day_axis(ax, n_days):
    """Ticks at day boundaries (more sparse for long chains)."""
    step = 1 if n_days <= 7 else 2
    ax.set_xticks(range(0, n_days + 1, step))
    ax.set_xlim(0, n_days)
    ax.grid(alpha=.3)


def timeseries(results, scenario, tr, nt, waits, colors, ps, n_days):
    """The fine-grained figure: every checkpoint of the chains without resets, averaged over
    seeds. Rows: carma Gini, top-10% share, largest balance, share of cars stopped, mean wait and
    mean time lost of the trips that arrived in each interval. One column per demand level."""
    rows = [("carma Gini", tr, "gini"), ("share of carma held by the top 10%", tr, "top10_share"),
            ("largest balance", tr, "max"), ("share of cars stopped", nt, "stopped_share"),
            ("mean wait of arriving trips (s)", waits, "mean_wait"),
            ("mean time lost of arriving trips (s)", waits, "mean_loss")]
    rows = [r for r in rows if len(r[1])]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(ps), figsize=(9 * len(ps), 2.6 * len(rows)),
                             squeeze=False, sharex=True)
    for j, p in enumerate(ps):
        for i, (title, data, col) in enumerate(rows):
            ax = axes[i][j]
            sub = data[(data["p"] == p) & (data["reset_every"] == 0)]
            for name, g in sub.groupby("name"):
                m = g.groupby("time_days")[col].mean()
                ax.plot(m.index, m.values, lw=1.4 if name.startswith("stop_signs") else 1.0,
                        color=colors[name], label=name)
            ax.set_title(f"{scenario}, p={p}: {title}" if i == 0 else title, fontsize=9)
            _day_axis(ax, n_days)
        axes[-1][j].set_xlabel("days")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{results}_{scenario}_timeseries.png"), dpi=130)


def by_week(results, scenario, s, stop):
    """Chains longer than 7 days: week-by-week means, the daily slope of the end-of-day Gini
    within each week, and the paired difference to stop signs per week."""
    base = s[s["k"] == 0].assign(week=lambda x: (x["day"] - 1) // 7 + 1)
    weekly = base.groupby(["name", "p", "week"]).agg(
        wait=("mean_wait", "mean"), p95=("p95_wait", "mean"), loss=("mean_loss", "mean"),
        stopped=("mean_stopped_share", "mean"), gini=("end_gini", "mean"),
        days=("day", "nunique"))

    def slope(g):
        m = g.groupby("day")["end_gini"].mean().dropna()
        return np.polyfit(m.index, m.values, 1)[0] if len(m) > 1 else float("nan")
    weekly["gini_per_day"] = base.groupby(["name", "p", "week"]).apply(slope, include_groups=False)
    if stop is not None:
        for metric, short in (("mean_wait", "wait"), ("mean_loss", "loss")):
            vs = paired(base, metric, stop, units=("seed", "day"), by=("p", "week"))
            if len(vs):
                vs = vs.set_index(["name", "p", "week"])
                weekly[f"{short}_vs_stop_pct"] = vs["mean_diff"]
                weekly[f"{short}_vs_stop_ci"] = vs.apply(
                    lambda r: f"[{r.ci_low:+.1f}, {r.ci_high:+.1f}]", axis=1)
    weekly = weekly.round(4)
    weekly.to_csv(os.path.join(TABLES, f"{results}_{scenario}_by_week.csv"))
    print("=== week by week (no reset): wait and time lost (s), end-of-day carma Gini and its "
          f"slope per day, wait and time lost vs {stop} paired by (seed, day), % ===")
    print(weekly.to_string())


def one(results, scenario):
    """All figures and tables for one scenario ("single" or "grid") of results/<results>."""
    d, tr, nt, waits = load(results, scenario)
    if d.empty:
        print(f"{results}/{scenario}: no results")
        return
    s = schedules(d)
    stop = next((n for n in d["name"].unique() if n.startswith("stop_signs")), None)
    ps = sorted(d["p"].unique(), reverse=True)
    colors = _colors(d["name"].unique())
    n_days = int(d["day"].max())
    prefix = f"{results}_{scenario}"
    # Checkpoint spacing of these runs; put every run's checkpoints on that grid for plotting.
    interval = (float(nt["t"].diff().where(lambda x: x > 0).median()) if len(nt)
                else DAY_LEN[scenario] / 4)
    tr = align(tr, interval, scenario, fill=True)
    nt = align(nt, interval, scenario, fill=False)

    # --- day-by-day curves: mean wait (top row), mean time lost (middle) and carma Gini (bottom),
    # one column per p. No reset solid; reset every k days dashed / dotted / dash-dot.
    fig, axes = plt.subplots(3, len(ps), figsize=(6.5 * len(ps), 11), squeeze=False)
    for j, p in enumerate(ps):
        for (name, k), g in s[s["p"] == p].groupby(["name", "k"]):
            is_stop = name == stop
            label = name if k == 0 else f"{name}, reset every {k}"
            m = g.groupby("day")["mean_wait"].mean()      # average over seeds
            axes[0][j].plot(m.index, m.values, marker="o", ls=STYLES.get(k, "-."),
                            lw=2.6 if is_stop else 1.5, color=colors[name], label=label)
            m = g.groupby("day")["mean_loss"].mean()
            axes[1][j].plot(m.index, m.values, marker="o", ls=STYLES.get(k, "-."),
                            lw=2.6 if is_stop else 1.5, color=colors[name], label=label)
            # Stop signs have no carma, so no Gini line for them.
            if not is_stop:
                gm = g.groupby("day")["end_gini"].mean()
                axes[2][j].plot(gm.index, gm.values, marker="o", ls=STYLES.get(k, "-."),
                                color=colors[name], label=label)
        axes[0][j].set_title(f"{scenario}, p={p}: mean wait by day")
        axes[1][j].set_title("mean time lost by day")
        axes[2][j].set_title("carma Gini at end of day")
        for ax in axes[:, j]:
            ax.set_xlabel("day")
            ax.grid(alpha=.3)
        axes[0][j].set_ylabel("mean wait (s)")
        axes[1][j].set_ylabel("mean time lost (s)")
        axes[2][j].set_ylabel("Gini")
    axes[2][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{prefix}_days.png"), dpi=130)

    # --- carma Gini at every checkpoint; reset chains dashed (they start at their first reset,
    # so the drop in inequality shows)
    if len(tr):
        per_day = tr.groupby(["name", "p", "seed", "day", "reset_every"]).size().median()
        fig, axes = plt.subplots(1, len(ps), figsize=(9 * len(ps), 4), squeeze=False)
        for j, p in enumerate(ps):
            for (name, k), g in tr[tr["p"] == p].groupby(["name", "reset_every"]):
                m = g.groupby("time_days")["gini"].mean()
                axes[0][j].plot(m.index, m.values, lw=1.0, ls=STYLES.get(k, "-."),
                                color=colors[name],
                                label=name if k == 0 else f"{name}, reset every {k}")
            axes[0][j].set_title(f"{scenario}, p={p}: carma Gini, {per_day:.0f} checkpoints/day")
            axes[0][j].set_xlabel("days"); axes[0][j].set_ylabel("Gini")
            _day_axis(axes[0][j], n_days)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{prefix}_carma_finegrain.png"), dpi=130)

    # --- every checkpoint of the chains without resets: carma and network state, arrival waits
    timeseries(results, scenario, tr, nt, waits, colors, ps, n_days)

    # --- metrics by age (days since the last reset) and the mean over all days per reset interval
    t = s.groupby(["name", "p", "schedule", "age"]).agg(
        wait=("mean_wait", "mean"), p95=("p95_wait", "mean"), loss=("mean_loss", "mean"),
        gini=("end_gini", "mean"), stopped=("mean_stopped_share", "mean"),
        jammed=("jammed", "mean"), n=("day", "count")).round(3)
    t.to_csv(os.path.join(TABLES, f"{prefix}_by_age.csv"))
    summ = s.groupby(["name", "p", "schedule"]).agg(
        wait=("mean_wait", "mean"), loss=("mean_loss", "mean"), gini=("end_gini", "mean"),
        days=("day", "count")).round(3)
    summ.to_csv(os.path.join(TABLES, f"{prefix}_by_reset.csv"))

    # --- resets, paired: each reset schedule vs no reset on the same (seed, day), using only the
    # days after the first reset (before it the runs are identical)
    tables = []
    for k in sorted(set(s["k"]) - {0}):
        sub = s[s["k"].isin([0, k]) & (s["day"] > k) & (s["name"] != stop)]
        for metric in ("mean_wait", "mean_loss"):
            tables.append(paired(sub, metric, NEVER, units=("seed", "day"), by=("name", "p"),
                                 name="schedule"))
        tables.append(paired(sub, "end_gini", NEVER, units=("seed", "day"), by=("name", "p"),
                             name="schedule", relative=False))
    print(f"\n######## {results} / {scenario} ({n_days} days) ########")
    if tables:
        rp = pd.concat(tables, ignore_index=True)
        rp.round(4).to_csv(os.path.join(TABLES, f"{prefix}_reset_paired.csv"), index=False)
        print("=== reset vs no reset, paired by (seed, day) after the first reset: mean wait, % ===")
        print(show(rp[rp.metric == "mean_wait"], by=("name", "p"), name="schedule"))
        print("=== same: mean time lost, % ===")
        print(show(rp[rp.metric == "mean_loss"], by=("name", "p"), name="schedule"))
        print("=== same: end-of-day carma Gini, absolute difference ===")
        print(show(rp[rp.metric == "end_gini"], by=("name", "p"), name="schedule", digits=3))
    else:
        print("No reset chains found: run controller/multiday.py "
              f"{scenario} (resets every 3 and 5 days are simulated by default).")

    # --- strategies vs stop signs, paired by (seed, day), no resets
    if stop is not None:
        base = s[s["k"] == 0]
        vs = pd.concat([paired(base, m, stop, units=("seed", "day"), by=("p",))
                        for m in ("mean_wait", "mean_loss")], ignore_index=True)
        vs.round(4).to_csv(os.path.join(TABLES, f"{prefix}_paired.csv"), index=False)
        print(f"=== no reset: mean wait vs {stop}, paired by (seed, day), % "
              "(stop signs keep cars below 0.1 m/s only briefly: compare on time lost) ===")
        print(show(vs[vs.metric == "mean_wait"], by=("p",)))
        print(f"=== no reset: mean time lost vs {stop}, paired by (seed, day), % ===")
        print(show(vs[vs.metric == "mean_loss"], by=("p",)))

    if n_days > 7:
        by_week(results, scenario, s, stop)

    print(f"=== mean over the {n_days} days (wait and time lost in s, carma Gini at end of day) "
          "===")
    print(summ.unstack("schedule").to_string())
    print(d[d["reset_every"] == 0].groupby(["name", "p"]).agg(
        jam=("jammed", "mean"), guests=("guest_trips", "sum"),
        wait=("mean_wait", "mean"), loss=("mean_loss", "mean")).round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="multiday",
                        help="results folder to read (results/<name>/<scenario>); also the prefix "
                             "of the figure and table names")
    parser.add_argument("scenarios", nargs="*", default=["single", "grid"])
    args = parser.parse_args()
    for scenario in args.scenarios:
        one(args.results, scenario)
