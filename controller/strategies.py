"""BidStrategy: how much does a single car bid? (Pure functions of Car + Context.)

Strategies must never bid more than the car's carma. The payment rules rely on this to
keep balances non-negative.

A strategy is plugged into a BidderType (bidders.py) together with a grouper (who bids together)
and an aggregator (how the members' bids combine). Experiments in config.py are built from these.
The strategies here fall into families:
  * baselines that ignore the situation: RandomFraction, FixedFraction
  * wait-time based: WaitTimeUrgency, BudgetedUrgency
  * urgency aware (the trip's value of time, see Experiment.urgency_levels): UrgencyScaledWait,
    UrgencyFraction
  * benchmark only: OmniscientUrgency (breaks the balance rule, pair it with NoPayment)
  * history based: RecentPriceFollower
The regret-minimising learner lives in regret.py.
"""
import random
from typing import Protocol

from car import Car
from core import Context


class BidStrategy(Protocol):
    """The interface of a bidding strategy: `amount(car, ctx)` returns how much the car bids."""
    # The interface every strategy follows: given a car and the world, return its bid.
    # Any class with an `amount(car, ctx)` method counts as a strategy.
    def amount(self, car: Car, ctx: Context) -> float: ...


class RandomFraction:
    """Uniform random bid in [0, carma]."""

    def amount(self, car: Car, ctx: Context) -> float:
        """A uniformly random bid between 0 and the whole balance."""
        # Pick a random number between 0 and everything the car owns. Ignores the situation
        # entirely, which makes it a useful baseline.
        return random.uniform(0, car.carma)


class FixedFraction:
    """Always bid a fixed fraction of the current balance."""

    def __init__(self, fraction: float = 0.1):
        """Remember the share of the balance to bid."""
        # The share of the balance to bid each time (0.1 = 10%).
        self.fraction = fraction

    def amount(self, car: Car, ctx: Context) -> float:
        """The fixed share of the current balance."""
        # Always bid the same share of the current balance, so bids shrink as carma runs down.
        return self.fraction * car.carma


class RecentPriceFollower:
    """Example of a strategy that reads the bidding history (not a regret-minimising one).

    Bids `margin` times the average winning amount of the last `n` auctions at the car's next
    intersection, capped at the balance. With no history yet it bids `start_fraction` of its
    balance. A starting point for strategies that learn from `ctx.history` and `car.bid_history`.
    """

    def __init__(self, margin: float = 1.1, n: int = 5, start_fraction: float = 0.1):
        """Remember the margin, how many past auctions to average and the share to bid with no
        history.
        """
        # Bid this multiple of the recent winning amount (>1 aims to just beat the typical winner).
        self.margin = margin
        # How many of the most recent auctions at the intersection to average.
        self.n = n
        # Share of the balance to bid when there is no history yet.
        self.start_fraction = start_fraction

    def amount(self, car: Car, ctx: Context) -> float:
        """A multiple of the recent average winning amount (or a fixed share with no history), capped
        at the balance.
        """
        # No history available (or this car has no traffic light ahead): bid a fixed share.
        if ctx.history is None or car.tls is None:
            return self.start_fraction * car.carma
        prices = ctx.history.winning_amounts(car.tls, self.n)
        if not prices:
            return self.start_fraction * car.carma
        average = sum(prices) / len(prices)
        # Never more than the balance.
        return min(car.carma, self.margin * average)


class WaitTimeUrgency:
    """Bid grows with time spent waiting (k carma per second), capped at the balance."""

    def __init__(self, k: float = 1.0):
        """Remember how much carma is bid per second waited."""
        # How much carma the car is willing to bid per second it has waited.
        self.k = k

    def amount(self, car: Car, ctx: Context) -> float:
        """k carma per second waited, capped at the balance."""
        # The longer the car has waited, the more it bids, but never more than it owns
        # (min(...) enforces the "never bid more than carma" rule).
        return min(car.carma, self.k * car.wait_time)


class BudgetedUrgency:
    """Bid a share of the current balance that grows with waiting time: the share reaches 100%
    when the car has waited `full_wait` seconds. A car that has just arrived bids almost nothing
    and a car that has waited a long time bids everything it has, so the bid depends on how much
    carma the car holds (unlike WaitTimeUrgency, whose bid is capped by the balance but is
    otherwise independent of it). With carma that carries over between trips, a car that spent a
    lot recently bids less than one that has been saving."""

    def __init__(self, full_wait: float = 60.0):
        """Remember how many seconds of waiting lead to bidding the whole balance."""
        # Seconds of waiting after which the car bids its whole balance.
        self.full_wait = full_wait

    def amount(self, car: Car, ctx: Context) -> float:
        """A share of the balance that rises linearly with the waiting time."""
        # The share of the balance rises linearly from 0 to 1 over `full_wait` seconds.
        share = min(1.0, car.wait_time / self.full_wait)
        return share * car.carma


class UrgencyScaledWait:
    """Like WaitTimeUrgency but scaled by the trip's urgency: k * urgency * seconds waited,
    capped at the balance. An urgent car (high value of time) outbids a calm one that has waited
    equally long."""

    def __init__(self, k: float = 1.0):
        """Remember how much carma is bid per second waited and per unit of urgency."""
        self.k = k

    def amount(self, car: Car, ctx: Context) -> float:
        """k x urgency x seconds waited, capped at the balance."""
        return min(car.carma, self.k * car.urgency * car.wait_time)


class UrgencyFraction:
    """Bid a large share of the balance when urgent and a small share otherwise (a hand-made
    stand-in for the equilibrium policy: bid more when urgent, save carma when not)."""

    def __init__(self, high_fraction: float = 0.5, low_fraction: float = 0.05, threshold: float = 1.0):
        """Remember the share bid when urgent, the share bid otherwise and the urgency threshold."""
        # Share of the balance bid by an urgent car, and by a calm one.
        self.high_fraction = high_fraction
        self.low_fraction = low_fraction
        # Urgencies above this count as urgent.
        self.threshold = threshold

    def amount(self, car: Car, ctx: Context) -> float:
        """The high share of the balance for an urgent trip, the low share otherwise."""
        share = self.high_fraction if car.urgency > self.threshold else self.low_fraction
        return share * car.carma


class OmniscientUrgency:
    """BENCHMARK ONLY (the paper's DICT): bid the true urgency, plus a tiny wait-time tie-break,
    ignoring the balance. This deliberately breaks the 'never bid more than carma' rule, so pair
    it with NoPayment. It shows what perfect knowledge of urgency would achieve."""

    def amount(self, car: Car, ctx: Context) -> float:
        """The trip's true urgency (plus a tiny wait-time tie-break); ignores the balance."""
        # The 1e-3 * wait_time term only breaks ties between cars of equal urgency.
        return car.urgency + 1e-3 * car.wait_time
