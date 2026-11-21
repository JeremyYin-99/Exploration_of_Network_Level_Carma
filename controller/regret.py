"""Regret-minimising bidding: a bid policy learned from the outcomes of past auctions.

Every auction a car takes part in is stored with enough information (history.CarBidRecord) to ask
"what would have happened if I had bid x instead?": the car's own bid, the bid of its group, the
best competing bid, the car's balance and its urgency. RegretHedge uses this to run the Hedge
(exponential weights) algorithm with FULL-INFORMATION counterfactual payoffs: after every auction
the weight of each candidate bid is updated by the payoff that bid would have earned, not only the
one that was played. Hedge guarantees the learner's average payoff approaches that of the best
fixed bid in hindsight (no regret).

Candidate bids are fractions of the current balance (so a bid never exceeds the balance). The
learner keeps one set of weights per CONTEXT (urgency, how long the car has waited, how rich it
is), so it learns, e.g., to bid high when urgent and low when calm and poor. All cars share one
learner (a symmetric policy, like the stationary equilibrium policy in the CARMA paper), so it
sees many auctions per day.

Payoff of one past auction for candidate fraction f (bid x = f * balance):
    win(x)  = the group, with this car's own bid replaced by x, would have beaten the best rival
    u(x)    = urgency * win(x)  -  price(balance) * (payment(x) - (1 - win(x)) * rebate) / B0
where payment is x if it wins (0 if nobody competed), the rebate is what losers typically get
back, and price(balance) = lam0 * (B0 / max(balance, 5)) ** gamma is the shadow price of carma:
spending is worth more to a driver with little carma left. lam0 is the one knob: how many
low-urgency wins the whole starting balance B0 is worth.

"Value of winning" is therefore the trip's urgency (a modelling assumption: a win saves one unit
of time, worth `urgency`). The learner ignores the future: lam0 stands in for the value of saving
carma for later auctions.

Where it is used: config.REGRET_CONFIGS (all drivers learn), config.MIX_CONFIGS (a share of the
drivers learn), regret_study.py (the experiment), analyze/regret_analysis.py (the results).
"""
import json
import math
import random
from collections import defaultdict

import pandas as pd

# The candidate bids: each is a fraction of the current balance (0 = bid nothing, 1 = everything).
ACTIONS = (0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0)
# A car's context splits how long it has waited into <10 s, 10-30 s and >30 s ...
WAIT_EDGES = (10.0, 30.0)
# ... and its balance into poor (<0.4 x base), average, and rich (>1.5 x base).
BALANCE_EDGES = (0.4, 1.5)          # as multiples of the base balance
# The largest urgency; payoffs are divided by it so the learning rate doesn't depend on the scale.
MAX_URGENCY = 6.0                    # scales payoffs so the learning rate is comparable


def _bucket(value, edges):
    """Index of the interval `value` falls in: 0 if below edges[0], 1 if below edges[1], ...,
    len(edges) if above all of them."""
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


class RegretHedge:
    """BidStrategy: sample a fraction of the balance from the Hedge weights of the car's context
    and bid it. Learns from the car's own bid history (`car.bid_history`).

    Parameters:
        lam0          carma price (see the module docstring); bigger = spend more reluctantly.
        gamma         how fast the price rises as the balance falls (1 = inversely proportional).
        base_balance  B0, the "normal" balance; payments are measured in units of it.
        eta0          learning rate (decays as 1/sqrt(updates) per context).
        seed          seed for the random choice of which candidate bid to play.
    """

    def __init__(self, lam0: float = 1.0, gamma: float = 1.0, base_balance: float = 100.0,
                 eta0: float = 2.0, seed: int = 0):
        """Set the learner's parameters and start with equal weights for every candidate bid."""
        self.lam0 = lam0
        self.gamma = gamma
        self.base = base_balance
        self.eta0 = eta0
        # Own random stream, so the sampling is repeatable and doesn't disturb other randomness.
        self.rng = random.Random(seed)
        # context -> log-weight of every candidate bid, and how many updates the context has had.
        self.logw = defaultdict(lambda: [0.0] * len(ACTIONS))
        self.count = defaultdict(int)
        # Average carma a losing bid gets back (running mean of observed rebates).
        self.rebate = 0.0
        self.rebate_seen = 0
        # Frozen: keep playing the current weights but stop learning (evaluation runs).
        self.frozen = False
        # Contexts bid in today, for the day-to-day policy change (learning_study.py).
        self.visits_today = defaultdict(int)

    # ---- Policy ----------------------------------------------------------------------------

    def context(self, urgency: float, wait: float, balance: float):
        """The situation a bid is made in: (urgency, wait bucket, balance bucket). The weights
        are learned separately for every context."""
        return (round(urgency, 2), _bucket(wait, WAIT_EDGES),
                _bucket(balance / self.base, BALANCE_EDGES))

    def probabilities(self, ctx) -> list[float]:
        """Probability of playing each candidate bid in context `ctx`: a softmax of the log
        weights (subtracting the maximum first keeps the exponentials from overflowing)."""
        weights = self.logw[ctx]
        top = max(weights)
        exps = [math.exp(w - top) for w in weights]
        total = sum(exps)
        return [e / total for e in exps]

    def amount(self, car, ctx) -> float:
        """The bid of one car (called by bidders.make_bid for every car in every auction)."""
        # First absorb whatever this car's earlier auctions taught us.
        if not self.frozen:
            self._learn(car)
        key = self.context(car.urgency, car.wait_time, car.carma)
        self.visits_today[key] += 1
        # Draw one candidate fraction at random with the current probabilities (Hedge plays a
        # random choice, not the single best one) and bid that share of the balance.
        index = self.rng.choices(range(len(ACTIONS)), weights=self.probabilities(key))[0]
        return ACTIONS[index] * car.carma

    # ---- Learning --------------------------------------------------------------------------

    def _learn(self, car) -> None:
        """Update the weights with the auctions this car took part in since its last bid."""
        history = car.bid_history
        # First call of a trip: everything already in the (driver's) history was seen before.
        if getattr(car, "_regret_seen", None) is None:
            car._regret_seen = len(history)
            return
        # Every record added since the last call is an auction that has finished. Records of
        # auctions the car rode along in without choosing its bid (a follower under front-car
        # bidding) teach nothing about bidding.
        for record in history[car._regret_seen:]:
            if getattr(record, "decision", True):
                self._update(record)
        car._regret_seen = len(history)

    def _update(self, r) -> None:
        """Hedge update from one finished auction `r` (a history.CarBidRecord): add to every
        candidate bid's log-weight the payoff that bid would have earned."""
        if r.balance <= 0:
            return
        # Keep a running average of what losing bids got back, to value "losing and being paid".
        if not r.won and r.carma_change > 0:
            self.rebate_seen += 1
            self.rebate += (r.carma_change - self.rebate) / min(self.rebate_seen, 200)
        # Shadow price of carma for a driver with this balance (higher when poor).
        price = self.lam0 * (self.base / max(r.balance, 5.0)) ** self.gamma
        key = self.context(r.urgency, r.wait_time, r.balance)
        self.count[key] += 1
        # Learning rate: shrinks as the context gets more data, scaled by the largest payoff.
        step = self.eta0 / math.sqrt(self.count[key]) / MAX_URGENCY
        weights = self.logw[key]
        # If no rival bid at all the car won for free whatever it offered.
        contested = r.top_other_amount > 0
        for index, fraction in enumerate(ACTIONS):
            # What the car would have bid, and the group's total with that bid swapped in.
            own = fraction * r.balance
            group = r.group_amount - r.own_bid + own
            win = (not contested) or group > r.top_other_amount
            if win:
                # Value of the win (the trip's urgency) minus the carma paid, priced and
                # expressed in units of the base balance. An uncontested win costs nothing.
                payoff = r.urgency - price * (own if contested else 0.0) / self.base
            else:
                # A loss earns nothing but the typical rebate paid to losers.
                payoff = price * self.rebate / self.base
            weights[index] += step * payoff

    # ---- Learning study hooks (learning_study.py) ------------------------------------------

    def start_day(self, day: int) -> None:
        """A new simulated day starts (clears the count of contexts bid in today)."""
        self.visits_today = defaultdict(int)

    def exploration(self) -> tuple[float | None, float | None]:
        """(current exploration level, its floor). Hedge has no exploration schedule."""
        return None, None

    def policy_snapshot(self) -> dict:
        """{context: expected bid fraction} for every context seen, and how often each context
        was bid in today: {"bids": ..., "visits": ...}. learning_study.py compares consecutive
        days' snapshots to measure how much the policy still changes."""
        bids = {}
        for ctx in self.logw:
            probs = self.probabilities(ctx)
            bids["|".join(map(str, ctx))] = sum(p * f for p, f in zip(probs, ACTIONS))
        visits = {"|".join(map(str, ctx)): n for ctx, n in self.visits_today.items()}
        return {"bids": bids, "visits": visits}

    def save_state(self, path: str) -> None:
        """Write everything learned to a JSON file (to resume a chain or freeze the policy)."""
        state = {
            "logw": {"|".join(map(str, ctx)): w for ctx, w in self.logw.items()},
            "count": {"|".join(map(str, ctx)): n for ctx, n in self.count.items()},
            "rebate": self.rebate, "rebate_seen": self.rebate_seen,
            "snapshot": self.policy_snapshot(),
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load_state(self, path: str) -> None:
        """Read back what save_state wrote."""
        with open(path) as f:
            state = json.load(f)

        def key(text):
            urgency, wait, balance = text.split("|")
            return (float(urgency), int(wait), int(balance))
        self.logw = defaultdict(lambda: [0.0] * len(ACTIONS),
                                {key(k): v for k, v in state["logw"].items()})
        self.count = defaultdict(int, {key(k): n for k, n in state["count"].items()})
        self.rebate = state["rebate"]
        self.rebate_seen = state["rebate_seen"]

    # ---- Reporting -------------------------------------------------------------------------

    def report(self, path: str) -> None:
        """Save the learned policy: for each context, the probability of every candidate bid and
        the expected fraction of the balance that is bid."""
        rows = []
        for ctx in sorted(self.logw):
            probs = self.probabilities(ctx)
            row = {"urgency": ctx[0], "wait_bucket": ctx[1], "balance_bucket": ctx[2],
                   "updates": self.count[ctx],
                   "mean_fraction": sum(p * f for p, f in zip(probs, ACTIONS))}
            # One column per candidate bid, e.g. p_0.25 = probability of bidding 25% of balance.
            for fraction, p in zip(ACTIONS, probs):
                row[f"p_{fraction:g}"] = p
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
