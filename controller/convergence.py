"""Convergence check of the learning study (docs/learning_spec.md, section 7).

Every day of a learning chain gets one row of metrics (written to <chain>_convergence.csv by
learning_study.py):

  M1  policy change: visit-weighted mean |bid today - bid yesterday| over states decided in at
      least `min_visits` times on both days, relative to the mean balance (policy_change()).
  M2  price drift: for each active intersection, the least-squares slope of its daily price over
      the trailing `window` days; it passes if the drift over the window is below `drift_max`
      of its mean price or if the slope's 95% interval includes 0.
  M3  week on week: the difference between the mean price in the second and the first half of the
      window (Welch t-test on the daily prices); passes if the 95% interval includes 0. The mean
      of these differences over intersections must also have an interval that includes 0.
  M4  exploration finished: the learner's exploration was at its floor on every day of the window.

The daily price of an intersection is its mean winning bid in contested auctions, relative to the
mean balance (prices.py, column win_rel_mean of the whole-day row). An intersection is active if
it averaged at least `active_auctions` auctions a day over the window; only those are tested.

A day passes when M1 < m1_max, M2 and M3 pass at a share >= `share` of the active intersections,
the network-wide M3 passes and M4 holds. A chain has converged on day d if d >= min_days and the
last `consecutive` days all passed.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 97.5% quantiles of Student's t (two-sided 95% intervals), by degrees of freedom.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 22: 2.074, 24: 2.064,
         26: 2.056, 28: 2.048, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: float) -> float:
    """97.5% quantile of Student's t with df degrees of freedom (interpolated; 1.96 beyond 120)."""
    if not df or df < 1 or math.isnan(df):
        return float("inf")
    keys = sorted(_T975)
    if df >= keys[-1]:
        return 1.96
    for lo, hi in zip(keys, keys[1:]):
        if lo <= df <= hi:
            return _T975[lo] + (_T975[hi] - _T975[lo]) * (df - lo) / (hi - lo)
    return _T975[1]


@dataclass
class Settings:
    """Thresholds of the convergence rule (defaults from the spec)."""
    window: int = 14
    active_auctions: float = 50.0
    m1_max: float = 0.01
    drift_max: float = 0.05
    share: float = 0.9
    consecutive: int = 7
    min_days: int = 60
    min_visits: int = 20


def policy_change(previous: dict | None, current: dict | None, min_visits: int = 20):
    """M1 from two consecutive policy snapshots ({"bids": {state: bid}, "visits": {...}}), or
    None if no state was decided in often enough on both days."""
    if not previous or not current:
        return None
    total, weight = 0.0, 0.0
    for state, bid in current["bids"].items():
        before = previous["bids"].get(state)
        n_now = current["visits"].get(state, 0)
        n_before = previous["visits"].get(state, 0)
        if before is None or n_now < min_visits or n_before < min_visits:
            continue
        if bid is None or before is None or math.isnan(bid) or math.isnan(before):
            continue
        total += n_now * abs(bid - before)
        weight += n_now
    return total / weight if weight > 0 else None


def daily_prices(prices_by_day: pd.DataFrame) -> pd.DataFrame:
    """(day, tls, price, n_auctions) from the chain's price history."""
    if prices_by_day.empty:
        return pd.DataFrame(columns=["day", "tls", "price", "n_auctions"])
    out = prices_by_day[["day", "tls", "win_rel_mean", "n_auctions"]].copy()
    return out.rename(columns={"win_rel_mean": "price"})


def _drift_test(days: np.ndarray, y: np.ndarray, window: int, drift_max: float) -> tuple:
    """(passes, slope, drift): M2 for one intersection."""
    n = len(y)
    if n < 3:
        return False, float("nan"), float("nan")
    slope, intercept = np.polyfit(days, y, 1)
    residuals = y - (slope * days + intercept)
    sxx = float(((days - days.mean()) ** 2).sum())
    se = math.sqrt(float((residuals ** 2).sum()) / (n - 2) / sxx) if sxx > 0 and n > 2 else 0.0
    half = t975(n - 2) * se
    mean = abs(float(y.mean()))
    drift = abs(slope) * window / mean if mean > 1e-9 else (0.0 if abs(slope) < 1e-9 else 1.0)
    return bool(drift < drift_max or (slope - half <= 0 <= slope + half)), float(slope), drift


def _welch(a: np.ndarray, b: np.ndarray) -> tuple:
    """(interval includes 0, difference b - a) for two samples of daily prices."""
    if len(a) < 2 or len(b) < 2:
        return False, float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = float(b.mean() - a.mean())
    se2 = va / len(a) + vb / len(b)
    if se2 <= 0:
        return diff == 0.0, diff
    df = se2 ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    half = t975(df) * math.sqrt(se2)
    return bool(diff - half <= 0 <= diff + half), diff


def price_tests(prices: pd.DataFrame, day: int, settings: Settings) -> dict:
    """M2 and M3 over the window ending on `day` (prices from daily_prices())."""
    start = day - settings.window + 1
    window = prices[(prices["day"] >= start) & (prices["day"] <= day)]
    result = {"n_active": 0, "m2_share": float("nan"), "m3_share": float("nan"),
              "m3_network": False, "m2_failing": "", "m3_failing": ""}
    if start < 1 or window.empty:
        return result
    half = settings.window // 2
    m2_pass, m3_pass, diffs, m2_fail, m3_fail = [], [], [], [], []
    for tls, g in window.groupby("tls"):
        if g["n_auctions"].sum() / settings.window < settings.active_auctions:
            continue
        g = g.dropna(subset=["price"]).sort_values("day")
        ok2, _, _ = _drift_test(g["day"].to_numpy(float), g["price"].to_numpy(float),
                                settings.window, settings.drift_max)
        first = g[g["day"] < start + half]["price"].to_numpy(float)
        second = g[g["day"] >= start + half]["price"].to_numpy(float)
        ok3, diff = _welch(first, second)
        m2_pass.append(ok2)
        m3_pass.append(ok3)
        if not math.isnan(diff):
            diffs.append(diff)
        if not ok2:
            m2_fail.append(str(tls))
        if not ok3:
            m3_fail.append(str(tls))
    n = len(m2_pass)
    result["n_active"] = n
    if n == 0:
        return result
    result["m2_share"] = sum(m2_pass) / n
    result["m3_share"] = sum(m3_pass) / n
    result["m2_failing"] = " ".join(m2_fail)
    result["m3_failing"] = " ".join(m3_fail)
    if len(diffs) >= 2:
        d = np.array(diffs)
        half_width = t975(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
        result["m3_network"] = bool(d.mean() - half_width <= 0 <= d.mean() + half_width)
    else:
        result["m3_network"] = bool(m3_pass[0]) if m3_pass else False
    return result


def convergence_table(prices_by_day: pd.DataFrame, learn_days: pd.DataFrame,
                      settings: Settings) -> pd.DataFrame:
    """One row per learning day: M1-M4, whether the day passed, the run of passing days and
    whether the chain has converged. learn_days has columns day, m1, exploration, floor."""
    prices = daily_prices(prices_by_day)
    rows = []
    run = 0
    learn_days = learn_days.sort_values("day")
    for _, r in learn_days.iterrows():
        day = int(r["day"])
        tests = price_tests(prices, day, settings)
        window = learn_days[(learn_days["day"] > day - settings.window) &
                            (learn_days["day"] <= day)]
        if pd.isna(r.get("floor")) or r.get("floor") is None:
            m4 = True
        else:
            m4 = bool(len(window) == settings.window and
                      (window["exploration"] <= window["floor"] + 1e-12).all())
        m1 = r.get("m1")
        m1_ok = m1 is not None and not pd.isna(m1) and m1 < settings.m1_max
        passed = bool(m1_ok and tests["n_active"] > 0
                      and tests["m2_share"] >= settings.share
                      and tests["m3_share"] >= settings.share
                      and tests["m3_network"] and m4)
        run = run + 1 if passed else 0
        rows.append({"day": day, "m1": m1, "m1_ok": m1_ok, **tests, "m4": m4,
                     "exploration": r.get("exploration"), "passed": passed,
                     "consecutive": run,
                     "converged": bool(day >= settings.min_days
                                       and run >= settings.consecutive)})
    return pd.DataFrame(rows)
