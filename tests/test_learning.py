"""Unit tests for the learning-study pieces that can be checked without SUMO (docs/learning_spec.md,
S13): front-car bidding and rebates, the tie-break, learner A's bid choice, and the stopping rule.

Run from the repo root:  python tests/test_learning.py      (or: python -m pytest tests)
"""
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "controller"))

from auction import FirstPrice, HighestBid  # noqa: E402
from best_response import FRACTIONS, BestResponseLearner  # noqa: E402
from bidders import make_bid  # noqa: E402
from convergence import Settings, convergence_table, policy_change  # noqa: E402


class Fixed:
    """A strategy that always bids the same amount."""

    def __init__(self, amount):
        self.amount_value = amount

    def amount(self, car, ctx):
        return self.amount_value


def car(vid, dist, wait, link=0, carma=100.0):
    """A stand-in for controller.car.Car with the fields bidding uses."""
    return SimpleNamespace(vid=vid, dist=dist, wait_time=wait, link=link, carma=carma)


def test_front_car_bidding_and_rebates():
    """Only the head of each lane bids; the longest wait wins a tie; only losing front cars get the
    rebate; carma is conserved."""
    a, b = car("a", 5.0, 5.0, link=0), car("b", 20.0, 2.0, link=1)
    c = car("c", 3.0, 8.0, link=2)
    lane1 = make_bid([b, a], "J", Fixed(10.0), sum, None, front_only=True)
    lane2 = make_bid([c], "J", Fixed(10.0), sum, None, front_only=True)
    assert lane1.member_bids == {"b": 0.0, "a": 10.0} and lane1.amount == 10.0
    assert lane1.bidders == {"a"} and lane1.links == {0, 1}
    winners = HighestBid(tie_break="wait").run([lane1, lane2])
    assert winners == [lane2], "equal bids: the longest-waiting group (c, 8 s) wins"
    deltas = FirstPrice(redistribute=True, free_if_uncontested=True,
                        rebate_to="bidders").settle(winners, [lane1, lane2])
    assert deltas == {"c": -10.0, "a": 10.0}, deltas
    members = FirstPrice(redistribute=True, free_if_uncontested=True).settle(winners,
                                                                            [lane1, lane2])
    assert members == {"c": -10.0, "a": 5.0, "b": 5.0}, members
    assert abs(sum(deltas.values())) < 1e-12


def test_whole_units():
    """Whole carma: bids round down and never exceed the balance; a payment of 10 shared by 3
    losers gives 4, 3, 3 with the extra unit to the longest wait; total carma is conserved."""
    a = car("a", 2.0, 1.0, link=0, carma=7.0)
    b, c, d = car("b", 3.0, 9.0, link=1), car("c", 4.0, 5.0, link=2), car("d", 5.0, 2.0, link=3)
    winner = make_bid([a], "J", Fixed(10.6), sum, None, front_only=True, integer=True)
    assert winner.amount == 7.0, "capped at the whole balance"
    small = make_bid([b], "J", Fixed(2.99), sum, None, front_only=True, integer=True)
    assert small.amount == 2.0, "rounded down"
    noise = make_bid([b], "J", Fixed(2.99999999999), sum, None, front_only=True, integer=True)
    assert noise.amount == 3.0, "floating-point error just below a whole number is not a lost unit"
    losers = [make_bid([x], "J", Fixed(1.0), sum, None, front_only=True, integer=True)
              for x in (b, c, d)]
    rule = FirstPrice(redistribute=True, free_if_uncontested=True, rebate_to="bidders",
                      integer=True)
    deltas = rule.settle([winner], [winner] + losers)
    assert deltas == {"a": -7.0, "b": 3.0, "c": 2.0, "d": 2.0}, deltas
    assert sum(deltas.values()) == 0
    assert all(float(v).is_integer() for v in deltas.values())
    from auction import split_whole
    assert split_whole(10, [("x", 1), ("y", 1), ("z", 1)]) == {"x": 4, "y": 3, "z": 3}
    assert split_whole(5, [("x", 3.0), ("y", 2.0)]) == {"x": 3, "y": 2}
    assert split_whole(0, [("x", 1)]) == {}


def test_best_response_whole_bids():
    """With whole units learner A only considers whole bids: every amount up to a small balance,
    rounded-down fractions of a large one."""
    learner = BestResponseLearner()
    learner.integer = True
    learner.base = 10.0
    assert list(learner.candidate_bids(7.0)) == [0, 1, 2, 3, 4, 5, 6, 7]
    big = learner.candidate_bids(640.0)
    assert all(float(v).is_integer() for v in big) and big.max() == 640 and big.min() == 0
    fraction, bid_rel = learner.greedy(0.7, 1, 0, 1.0)
    assert float(bid_rel * learner.base).is_integer()


def test_random_tie_break_unchanged():
    """The default tie-break still draws the same random numbers as before (old results repeat)."""
    bids = [make_bid([car(v, 1.0, 0.0)], "J", Fixed(1.0), sum, None) for v in "xyz"]
    random.seed(3)
    first = HighestBid().run(bids)[0].members[0].vid
    random.seed(3)
    keys = [(1.0, random.random()) for _ in bids]
    expected = bids[max(range(3), key=lambda i: keys[i])].members[0].vid
    assert first == expected


def test_best_response_bid():
    """With every rival bidding 0.2 (relative), a loss costing far more than a win, and a value of
    20 s per unit of balance, the best bid is the smallest candidate above 0.2."""
    learner = BestResponseLearner()
    for _ in range(100):
        learner.rivals["J"].append(0.2)
        learner.faced["J"].append(True)
    learner.times["J"] = {"win": [10.0, 100], "lose": [30.0, 100], "arrive": [10.0, 100],
                          "rebate": [0.0, 100]}
    grid = learner.grid
    for key, intercept in (((1, 0, 1.0), -50.0), ((2, 0, 1.0), -100.0), ((2, 1, 1.0), -120.0)):
        learner.V[key] = intercept + 20.0 * grid
        learner.visits[key] = np.ones(len(grid))
    fraction, bid_rel = learner.greedy(1.0, 2, 0, 1.0, "J")
    expected = float(FRACTIONS[FRACTIONS > 0.2].min())
    assert math.isclose(fraction, expected), (fraction, expected)
    # Q of that bid: win for sure, pay it: -10 + (-50 + 20 * (1 - b)).
    bids_rel, q = learner.q_values(1.0, 2, 0, 1.0, "J")
    assert math.isclose(q.max(), -10.0 - 50.0 + 20.0 * (1.0 - expected))
    # Bidding nothing loses: -30 + (-120 + 20).
    assert math.isclose(q[0], -130.0)


def test_unseen_balances_never_look_better():
    """A balance nobody has had yet takes the value of the nearest balance that has been seen."""
    learner = BestResponseLearner()
    learner._td((1.0, 1, 0, 1.0), -50.0)
    learner._td((1.0, 1, 0, 1.0), -60.0)
    value_far = float(learner.value(3.5, 1, 0, 1.0))
    value_near = float(learner.value(1.0, 1, 0, 1.0))
    assert math.isclose(value_far, value_near), (value_far, value_near)


def _prices(days, price_of_day, tls=("A", "B", "C")):
    rng = np.random.default_rng(0)
    rows = [{"day": d, "tls": t, "win_rel_mean": price_of_day(d) + rng.normal(0, 0.005),
             "n_auctions": 300} for d in days for t in tls]
    return pd.DataFrame(rows)


def _learn_days(days, m1=0.001, level=0.02, floor=0.02):
    return pd.DataFrame({"day": list(days), "m1": m1, "exploration": level, "floor": floor})


def test_stopping_rule():
    """Flat prices converge soon after the minimum day (not exactly on it: with few
    intersections a single false alarm of the 5% tests fails a day); trending prices never do;
    exploration that is still above its floor blocks convergence."""
    days = range(1, 91)
    settings = Settings()
    flat = convergence_table(_prices(days, lambda d: 0.3), _learn_days(days), settings)
    first = flat[flat["converged"]]["day"].min()
    assert 60 <= first <= 80, first
    assert not flat[flat["day"] < 60]["converged"].any()
    trend = convergence_table(_prices(days, lambda d: 0.3 + 0.01 * d), _learn_days(days),
                              settings)
    assert not trend["converged"].any()
    exploring = _learn_days(days)
    exploring.loc[exploring["day"] > 50, "exploration"] = 0.5
    blocked = convergence_table(_prices(days, lambda d: 0.3), exploring, settings)
    assert not blocked["converged"].any()


def test_policy_change():
    """M1: visit-weighted mean absolute change over states seen often enough on both days."""
    before = {"bids": {"s1": 0.10, "s2": 0.50, "s3": 0.2}, "visits": {"s1": 50, "s2": 50, "s3": 5}}
    after = {"bids": {"s1": 0.12, "s2": 0.50, "s3": 0.9}, "visits": {"s1": 30, "s2": 25, "s3": 50}}
    # s3 is left out: only 5 visits the day before.
    assert math.isclose(policy_change(before, after), (30 * 0.02 + 25 * 0.0) / 55)
    assert policy_change(None, after) is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
