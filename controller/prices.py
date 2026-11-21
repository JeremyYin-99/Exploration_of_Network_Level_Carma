"""Per-intersection price summaries: what priority cost at each intersection, per time bin and for
the whole day (one row per (intersection, bin) plus one "day" row per intersection).

Input is the decision log of one run (history.AuctionHistory.decisions_frame(): one row per car
that chose a bid). carma.run writes the result as <label>_prices.csv when Experiment.price_bin is
set; the learning study joins the day rows of every day into <chain>_prices_by_day.csv, which the
convergence check (convergence.py) and the price-evolution figures (analyze/learning_analysis.py)
read.

Prices are given relative to the mean balance (the conserved carma supply, `mean_balance`) and
in carma. The "price" of an intersection is the winning bid in contested auctions (two or more
bids): in an uncontested auction the winner pays nothing whatever it bid.
"""
import numpy as np
import pandas as pd

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def junction_type(n_incoming_edges: int) -> str:
    """Name a junction by how many roads come into it."""
    return {4: "4-way", 3: "T", 2: "corner"}.get(n_incoming_edges, f"{n_incoming_edges}-arm")


def _stats(values: pd.Series, prefix: str) -> dict:
    """Mean, standard deviation and quantiles of `values`, with column names starting `prefix`."""
    out = {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    for q in QUANTILES:
        out[f"{prefix}_q{int(q * 100)}"] = np.nan
    if len(values):
        out[f"{prefix}_mean"] = values.mean()
        out[f"{prefix}_std"] = values.std(ddof=0)
        for q, v in zip(QUANTILES, values.quantile(list(QUANTILES))):
            out[f"{prefix}_q{int(q * 100)}"] = v
    return out


def _summary(dec: pd.DataFrame, auctions: pd.DataFrame, mean_balance: float) -> dict:
    """One summary row for a set of decisions and the auctions they belong to."""
    contested = auctions[auctions["contested"]]
    row = {
        "n_auctions": len(auctions),
        "n_contested": len(contested),
        "share_contested": len(contested) / len(auctions) if len(auctions) else np.nan,
        "n_decisions": len(dec),
    }
    row.update(_stats(dec["bid"] / mean_balance, "bid_rel"))
    row.update(_stats(dec["bid"], "bid"))
    row.update(_stats(contested["winning_bid"] / mean_balance, "win_rel"))
    row.update(_stats(contested["winning_bid"], "win"))
    losses = dec[~dec["won"]]
    row.update({
        "payment_mean": auctions["payment"].mean() if len(auctions) else np.nan,
        "rebate_mean": losses["rebate"].mean() if len(losses) else np.nan,
        "share_all_zero": auctions["all_zero"].mean() if len(auctions) else np.nan,
        "n_rivals_mean": dec["n_rivals"].mean() if len(dec) else np.nan,
        "share_broke": (dec["balance"] < 1.0).mean() if len(dec) else np.nan,
        "cars_in_range_mean": auctions["n_cars"].mean() if len(auctions) else np.nan,
        "balance_rel_mean": (dec["balance"] / mean_balance).mean() if len(dec) else np.nan,
    })
    return row


def summarize(decisions: pd.DataFrame, bin_seconds: float, mean_balance: float,
              junction_types: dict | None = None) -> pd.DataFrame:
    """The price table of one run: a row per (intersection, time bin of `bin_seconds`) with
    scope="bin", and a row per intersection for the whole run with scope="day"."""
    if decisions.empty:
        return pd.DataFrame()
    dec = decisions.copy()
    # One row per auction: when, where, the winning bid, what was paid, whether it was contested.
    per_auction = dec.groupby("auction_id").agg(
        t=("t", "first"), tls=("tls", "first"), winning_bid=("winning_bid", "first"),
        payment=("payment", "sum"), n_cars=("n_cars", "first"), n_rivals=("n_rivals", "max"),
        max_bid=("bid", "max")).reset_index()
    per_auction["contested"] = per_auction["n_rivals"] > 0
    if "contested" in dec.columns:
        # Each bid's own contest (auction.annotate_rivals): an auction is contested when some
        # winner beat a conflicting bid, and its price is the highest such winner's bid. With a
        # single winner this is the same as above; with several winners it leaves out winners
        # that displaced nobody (they pay nothing).
        paid = dec[dec["won"].astype(bool) & dec["contested"].astype(bool)] \
            .groupby("auction_id")["group_amount"].max()
        per_auction["contested"] = per_auction["auction_id"].isin(paid.index)
        per_auction["winning_bid"] = per_auction["auction_id"].map(paid) \
            .fillna(per_auction["winning_bid"])
    per_auction["all_zero"] = per_auction["max_bid"] <= 0
    dec["bin_start"] = (dec["t"] // bin_seconds) * bin_seconds
    per_auction["bin_start"] = (per_auction["t"] // bin_seconds) * bin_seconds
    junction_types = junction_types or {}
    rows = []
    for tls, d_tls in dec.groupby("tls"):
        a_tls = per_auction[per_auction["tls"] == tls]
        rows.append({"tls": tls, "scope": "day", "bin_start": np.nan,
                     "junction": junction_types.get(tls, ""),
                     **_summary(d_tls, a_tls, mean_balance)})
        for start, d_bin in d_tls.groupby("bin_start"):
            a_bin = a_tls[a_tls["bin_start"] == start]
            rows.append({"tls": tls, "scope": "bin", "bin_start": start,
                         "junction": junction_types.get(tls, ""),
                         **_summary(d_bin, a_bin, mean_balance)})
    return pd.DataFrame(rows)
