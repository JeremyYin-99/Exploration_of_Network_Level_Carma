"""Paired comparisons between strategies.

Every strategy in a study sees the same traffic: the same trips on the same day of a chain, and
the same trips (and urgency draws) for the same seed. So the fair way to compare two strategies is
unit by unit, where a unit is one seed (single-run studies) or one seed x day (multi-day studies):
take the difference on every unit, then summarise the differences. Day-to-day swings in traffic
(some days are simply busier for everyone) then cancel out, which they do not when two averages
are compared.

`paired()` returns, for every strategy and every group (e.g. demand level), the mean paired
difference to a reference strategy with its standard deviation, a 95% confidence interval and the
number of units on which the strategy beat the reference.

Caveat: days of the same chain are not fully independent (balances, and what learners have
learned, carry over), so with seed x day units the interval is somewhat optimistic. The count of
units won (a sign test) does not assume independence of the size of the differences, and is a
useful check on it.

Used by regret_analysis.py, multiday_analysis.py, urgency_analysis.py and lanes_analysis.py.
"""
import math

import pandas as pd

# Two-sided 95% Student-t quantiles for 1..30 degrees of freedom (avoids a scipy dependency);
# beyond 30 the normal value 1.96 is used.
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(df: int) -> float:
    """Two-sided 95% quantile of Student's t with `df` degrees of freedom."""
    return _T95[df - 1] if 1 <= df <= len(_T95) else 1.96


def paired(d: pd.DataFrame, metric: str, reference: str, units, by=(), name: str = "name",
           relative: bool = True) -> pd.DataFrame:
    """Paired differences of `metric` between every strategy and `reference`.

    d          one row per run (or per day of a chain), with columns `name`, `metric`, the unit
               columns and the `by` columns.
    units      columns that identify a unit of pairing, e.g. ("seed",) or ("seed", "day"). Rows
               with the same unit saw the same traffic. Several rows per unit are averaged.
    by         columns to compare within separately, e.g. ("p",).
    relative   True: differences in percent of the reference's value on the same unit;
               False: absolute differences (in the metric's own unit).

    name       the column holding what is compared (strategy name by default; e.g. "schedule"
               to compare reset schedules of the same strategy, with "name" in `by`).

    Returns one row per (by..., compared item) with: the `by` columns, the `name` column, the
    reference, the metric, n (paired units), mean_diff, sd, ci_low, ci_high
    (95%, Student t), better (units on which the strategy was strictly below the reference, i.e.
    better when lower is better), ref_mean and mean (the two averages over the paired units).
    Only units where both strategies have a value count.
    """
    by, units = list(by), list(units)
    rows = []
    groups = d.groupby(by) if by else [((), d)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        wide = g.pivot_table(index=units, columns=name, values=metric, aggfunc="mean")
        if reference not in wide.columns:
            continue
        for other in wide.columns:
            if other == reference:
                continue
            pair = wide[[other, reference]].dropna()
            if pair.empty:
                continue
            x, ref = pair[other], pair[reference]
            diff = 100.0 * (x - ref) / ref if relative else x - ref
            n = len(diff)
            mean = diff.mean()
            sd = diff.std(ddof=1) if n > 1 else float("nan")
            half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
            rows.append({**dict(zip(by, key)), name: other, "reference": reference,
                         "metric": metric, "scale": "percent" if relative else "absolute",
                         "n": n, "mean_diff": mean, "sd": sd,
                         "ci_low": mean - half, "ci_high": mean + half,
                         "better": int((x < ref).sum()), "ref_mean": ref.mean(), "mean": x.mean()})
    return pd.DataFrame(rows)


def show(table: pd.DataFrame, by=(), name: str = "name", digits: int = 1) -> str:
    """A compact text version of a `paired()` table: one line per compared item (column `name`)
    with both means, the mean difference, its 95% interval and how many units it won, sorted by
    the mean difference within each `by` group."""
    if table.empty:
        return "(no paired units)"
    t = table.copy()
    unit = "%" if t["scale"].iloc[0] == "percent" else ""
    t["diff"] = t.apply(lambda r: f"{r.mean_diff:+.{digits}f}{unit}", axis=1)
    t["95% CI"] = t.apply(lambda r: f"[{r.ci_low:+.{digits}f}, {r.ci_high:+.{digits}f}]"
                          if r.n > 1 else "", axis=1)
    t["won"] = t.apply(lambda r: f"{r.better}/{r.n}", axis=1)
    cols = list(dict.fromkeys(list(by) + [name])) + ["mean", "ref_mean", "diff", "95% CI", "won"]
    t = t.sort_values(list(by) + ["mean_diff"])
    return t[cols].round(digits).to_string(index=False)
