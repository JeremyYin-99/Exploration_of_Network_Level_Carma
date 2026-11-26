"""Fixed rules after 60 warm-up days (plan "fixed60", run 2026-11-21) against the same rules after
7 warm-up days (plan "fixed"): does a longer warm-up, with balances drifting further, change how
the rules compare with actuated lights? Both plans share the evaluation week's trips, so every
comparison is paired by seed.

Writes results/analysis/params_fixed60_cells.csv, params_fixed60_vs_fixed.csv and
analyze/figures/update/fixed60_vs_fixed.pdf/.png plus figures_fixed60.tex.

    python analyze/fixed60_analysis.py [workers]
"""
import os
import sys

import pandas as pd

from fixed_rules_analysis import OUT, RULES, grouped_bars, strategy_table  # noqa: E402
from parameter_analysis import TABLES, interval, load, settings_of, week_per_seed  # noqa: E402

KEYS = RULES + ["mix"]


def weeks(plan: str, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, evals = load(plan, workers)
    week = week_per_seed(evals)
    week["network"] = [settings_of(k)["network"] for k in week["baseline_key"]]
    week["lanes"] = [settings_of(k)["lanes"] for k in week["baseline_key"]]
    return week, strategy_table(week, prices)


def paired(long: pd.DataFrame, short: pd.DataFrame) -> pd.DataFrame:
    """Per network, lanes and strategy: % change in time lost and cost per trip after 60 warm-up
    days compared with after 7, paired by seed; and the largest actuated-light difference between
    the plans (should be 0: same trips, no carried state)."""
    idx = ["network", "lanes", "strategy", "seed"]
    a = long.set_index(idx)[["time_lost", "cost", "cost_gini"]]
    b = short.set_index(idx)[["time_lost", "cost", "cost_gini"]]
    both = a.index.intersection(b.index)
    d = pd.DataFrame({"time_lost": 100 * (a.loc[both, "time_lost"] / b.loc[both, "time_lost"] - 1),
                      "cost": 100 * (a.loc[both, "cost"] / b.loc[both, "cost"] - 1),
                      "cost_gini": a.loc[both, "cost_gini"] - b.loc[both, "cost_gini"]}).reset_index()
    rows = []
    for (network, lanes, strategy), g in d.groupby(["network", "lanes", "strategy"]):
        row = {"network": network, "lanes": lanes, "strategy": strategy, "seeds": len(g)}
        for col in ("time_lost", "cost", "cost_gini"):
            m, lo, hi = interval(g[col])
            row[f"{col}_60_vs_7"], row[f"{col}_60_vs_7_lo"], row[f"{col}_60_vs_7_hi"] = m, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def latex(seeds: int) -> None:
    text = rf"""% Fixed rules after 60 warm-up days vs 7 (analyze/fixed60_analysis.py).
\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{fixed60_vs_fixed.pdf}}
    \caption{{Change in time lost per trip when the fixed rules run 60 warm-up days instead of 7
    before the same 7-day evaluation week (same trips, paired by seed). Reference setting; bars are
    means of {seeds} seeds, whiskers 95\% intervals; each panel has its own scale.}}
    \label{{fig:fixed60_vs_fixed}}
\end{{figure}}
"""
    with open(os.path.join(OUT, "figures_fixed60.tex"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    long_week, long_table = weeks("fixed60", workers)
    short_week, short_table = weeks("fixed", workers)
    long_table.to_csv(os.path.join(TABLES, "params_fixed60_cells.csv"), index=False)
    change = paired(long_week, short_week)
    change.to_csv(os.path.join(TABLES, "params_fixed60_vs_fixed.csv"), index=False)
    act = change[change["strategy"] == "actuated"]
    print("actuated, 60-day plan vs 7-day plan (should be 0):",
          act["time_lost_60_vs_7"].abs().max() if len(act) else "no actuated runs yet")
    grouped_bars(change, "time_lost_60_vs_7", KEYS,
                 "change in time lost per trip, 60 vs 7 warm-up days (%)", "fixed60_vs_fixed",
                 "same as 7 days")
    latex(int(change[change["strategy"].isin(KEYS)]["seeds"].min()))
    pd.set_option("display.width", 220)
    cols = ["network", "lanes", "strategy", "time_lost_vs_actuated", "time_lost_vs_actuated_lo",
            "time_lost_vs_actuated_hi", "cost_vs_actuated", "urgent_vs_calm"]
    merged = long_table[cols].merge(short_table[cols], on=["network", "lanes", "strategy"],
                                    suffixes=("_60", "_7"))
    print(merged.round(1).to_string(index=False))
    print(change.round(2).to_string(index=False))
