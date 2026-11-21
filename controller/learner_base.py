"""Shared machinery for the learners of the learning study (best_response.py, qlearning.py).

A driver decides only when it chooses a bid (the front car of its lane under front-car bidding).
Its decisions form a chain: each decision leads to the driver's next decision, on the same trip
or on its next trip. The learner remembers every driver's last decision ("pending") and, when the
driver decides again, turns the two into one transition:

    state at the last decision  ->  reward  ->  state now

The reward is minus the trip's urgency times the seconds that passed. If the trip ended in
between, it is minus urgency times the seconds from the last decision to arrival: the time
between trips costs nothing, and the next trip's first decision is the next state (so leftover
carma keeps its value; the end of a trip is not terminal).

Objective. By default (gamma = 1) each driver minimises its long-run AVERAGE time per trip: the
values are relative ("differential") values, and every transition into a new trip is charged
minus rho, the average reward per trip (learned as a moving average of the trips' totals):

    target = reward - rho * [the transition starts a new trip] + V(next state)

This needs no discount factor and has no slowly converging overall level, so the comparison
"finish the trip now" against "lose and decide again" is right as soon as the relative values
are. With gamma < 1 the learner instead discounts per TRIP (nothing within a trip, gamma into the
next trip). Never per decision: every lost auction adds a decision, so the rest of the future
would be discounted once more and losing on purpose would look profitable; and never per second
of driving: a driver on the road always pays one second per second, so only the cost per trip
can be lowered. Learners build targets with `discount(same_trip)` and `crossing(same_trip)`.

State s = (x, k, l, u): balance / mean balance, auction-holding intersections left (counting
this one, capped at k_max), auctions already lost here (capped at l_max), trip urgency.

carma.run calls the hooks:
    amount(car, ctx)                    every decision (BidStrategy interface)
    observe_auction(t, tls, bids, winners, deltas, cars)    every finished auction
    on_trip_end(car, t)                 every arrival
    on_day_end(t)                       the end of a run (trips still driving are cut here)
and learning_study.py calls start_day(day), save_state/load_state, policy_snapshot and
exploration(). With `frozen` set the learner keeps bidding with what it knows but stops learning.

The pending decisions are kept by the learner, keyed by driver id, because a driver's own bid
history (Driver.bid_history) starts empty every simulated day.
"""
import json
import random
from collections import Counter


class DecisionLearner:
    """Base class: links decisions into transitions. Subclasses implement choose() and learn()
    (and their own state for save_state/load_state)."""

    def __init__(self, base_balance: float = 100.0, gamma: float = 1.0, k_max: int = 8,
                 l_max: int = 2, seed: int = 0, rho_rate: float = 0.002):
        """Common settings: the mean balance (prices are measured against it), gamma (1 = average
        cost per trip; below 1 = discount per trip), the caps on intersections left and auctions
        lost, the random seed, and the moving-average rate of rho."""
        self.base = base_balance
        self.gamma = gamma
        self.average = gamma >= 1.0
        # Average reward per trip (negative: minus urgency x seconds), and trips counted.
        self.rho: float | None = None
        self.rho_rate = rho_rate
        self.trips_seen = 0
        self.k_max = k_max
        self.l_max = l_max
        # Own random stream, separate from the simulation's, so every strategy sees the same
        # trips and the learner's choices are repeatable.
        self.rng = random.Random(seed)
        self.frozen = False
        # Day number (set by learning_study.py; drives the exploration schedule).
        self.day = 1
        # Runs (days) finished so far; part of every trip's identity, because SUMO vehicle ids
        # repeat from day to day.
        self.run_index = 0
        # driver -> its last decision, waiting for the next one to complete the transition.
        self.pending = {}
        # How often each (binned) state was decided in today, for the policy-change metric.
        self.visits_today = Counter()

    # ---- State ---------------------------------------------------------------------------

    def state(self, car) -> tuple:
        """The decision state (x, k, l, u) of a car right now."""
        x = max(0.0, car.carma) / self.base
        k = min(max(int(car.intersections_left), 1), self.k_max)
        lost = min(int(car.lost_here), self.l_max)
        return (x, k, lost, float(car.urgency))

    def discount(self, same_trip: bool) -> float:
        """Discount of a transition: none within a trip; gamma into the next trip (1 with the
        average-cost objective)."""
        return 1.0 if (same_trip or self.average) else self.gamma

    def crossing(self, same_trip: bool) -> float:
        """What a transition into a new trip is charged: -rho with the average-cost objective,
        0 otherwise (and 0 within a trip)."""
        if same_trip or not self.average or self.rho is None:
            return 0.0
        return -self.rho

    def _trip_finished(self, total: float) -> None:
        """Update rho with the total reward of a finished trip."""
        self.trips_seen += 1
        rate = max(self.rho_rate, 1.0 / self.trips_seen)
        self.rho = total if self.rho is None else self.rho + rate * (total - self.rho)

    @staticmethod
    def key(car) -> str:
        """Who is deciding: the persistent driver, or the vehicle if it has no driver."""
        return car.driver if car.driver is not None else car.vid

    def trip_id(self, car) -> tuple:
        """Identity of the car's current trip (unique across days)."""
        return (self.run_index, car.vid)

    @staticmethod
    def x_bin(x: float) -> float:
        """Balance rounded to a quarter of the mean below 4 times the mean, to whole multiples
        above (up to 50), for counting states."""
        if x < 4.0:
            return round(x * 4) / 4
        return float(min(round(x), 50))

    def state_key(self, s: tuple) -> str:
        """Text key of a binned state, used in snapshots and saved files."""
        return f"{self.x_bin(s[0])}|{s[1]}|{s[2]}|{s[3]}"

    # ---- BidStrategy interface -------------------------------------------------------------

    def amount(self, car, ctx) -> float:
        """The car's bid: first complete the transition from this driver's last decision, then
        choose a bid and remember this decision."""
        s = self.state(car)
        key = self.key(car)
        trip = self.trip_id(car)
        prev = self.pending.pop(key, None)
        # Reward collected so far on the current trip (for rho).
        trip_reward = 0.0
        if prev is not None and not self.frozen:
            dt, same_trip = None, False
            if prev["trip"] == trip:
                dt, same_trip = ctx.t - prev["t"], True
            elif prev["ended"] is not None:
                dt = prev["ended"] - prev["t"]
            if dt is not None:
                reward = -prev["s"][3] * dt
                if same_trip:
                    trip_reward = prev.get("acc", 0.0) + reward
                else:
                    self._trip_finished(prev.get("acc", 0.0) + reward)
                self.learn(prev, reward, s, same_trip, dt)
        bid, info = self.choose(s, car, ctx)
        bid = max(0.0, min(bid, car.carma))
        self.pending[key] = {"s": s, "t": ctx.t, "tls": car.tls, "trip": trip,
                             "won": None, "ended": None, "info": info, "acc": trip_reward}
        self.visits_today[self.state_key(s)] += 1
        return bid

    # ---- Hooks called by carma.run -----------------------------------------------------------

    def observe_auction(self, t, tls, bids, winners, deltas, cars) -> None:
        """Mark the outcome of the decisions made in this auction. Subclasses also learn from
        the public outcome (call this first)."""
        winner_ids = {id(w) for w in winners}
        for bid in bids:
            won = id(bid) in winner_ids
            for vid in bid.deciders():
                car = cars.get(vid)
                if car is None:
                    continue
                p = self.pending.get(self.key(car))
                if p is not None and p["t"] == t and p["trip"] == self.trip_id(car):
                    p["won"] = won

    def on_trip_end(self, car, t: float) -> None:
        """A trip ended: its last decision's transition ends at arrival time t."""
        p = self.pending.get(self.key(car))
        if p is not None and p["trip"] == self.trip_id(car):
            p["ended"] = t

    def on_day_end(self, t: float) -> None:
        """The run ended at time t: trips still driving are cut here."""
        for p in self.pending.values():
            if p["ended"] is None:
                p["ended"] = t
        self.run_index += 1

    def start_day(self, day: int) -> None:
        """A new day starts (sets the exploration schedule, clears today's state counts)."""
        self.day = day
        self.visits_today = Counter()

    # ---- To implement in subclasses ----------------------------------------------------------

    def choose(self, s: tuple, car, ctx) -> tuple[float, object]:
        """(bid in carma, anything the learner wants back in learn() as prev["info"])."""
        raise NotImplementedError

    def learn(self, prev: dict, reward: float, s_next: tuple, same_trip: bool, dt: float):
        """Learn from one transition: prev is the pending decision (s, t, tls, won, info)."""
        raise NotImplementedError

    def exploration(self) -> tuple[float, float]:
        """(current exploration level, its floor)."""
        raise NotImplementedError

    # ---- Saving ----------------------------------------------------------------------------

    def _base_state(self) -> dict:
        """The common part of save_state: day counters, pending decisions, random state."""
        pending = {}
        for key, p in self.pending.items():
            pending[key] = {**p, "s": list(p["s"]), "trip": list(p["trip"])}
        version, internal, gauss = self.rng.getstate()
        return {"day": self.day, "run_index": self.run_index, "pending": pending,
                "rng": [version, list(internal), gauss], "rho": self.rho,
                "trips_seen": self.trips_seen}

    def _load_base_state(self, state: dict) -> None:
        """Read back what _base_state wrote."""
        self.day = state["day"]
        self.run_index = state["run_index"]
        self.rho = state.get("rho")
        self.trips_seen = state.get("trips_seen", 0)
        self.pending = {}
        for key, p in state["pending"].items():
            self.pending[key] = {**p, "s": tuple(p["s"]), "trip": tuple(p["trip"])}
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))

    @staticmethod
    def _write(path: str, state: dict) -> None:
        """Write a state dict as JSON."""
        with open(path, "w") as f:
            json.dump(state, f)

    @staticmethod
    def _read(path: str) -> dict:
        """Read a state dict written by _write."""
        with open(path) as f:
            return json.load(f)
