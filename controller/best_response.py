"""Learner A of the learning study (docs/learning_spec.md, S5): a smoothed best response built from
what is measured at each intersection and a learned value of carma.

The auction rules are known (front-car bidding, FirstPrice with redistribution, free when
uncontested): a bid wins if it beats the highest rival bid, the winner pays its bid, losers get a
rebate. Only three things are unknown, and each is learned separately:

1. At each intersection j, the highest rival bid R a bidder faces (relative to the mean balance):
   P_unc,j = share of bidders with no rival at all, and F_j(b) = P(R < b) over the last `window`
   contested bids. Learned from every auction there (winning bids are public, like a price board).
   This distribution is the price of priority at j.
2. At each intersection, the seconds until a driver's next decision after a win (t_win), after a
   loss (t_lose) and, after a win at the last intersection, until arrival (t_arrive); and the
   mean rebate on a loss. Moving averages of what actually happened.
3. V(x, k, l, u): the value of being at a decision with balance x (relative), k intersections
   left, l auctions lost here, urgency u. Learned by TD(0) from the realised transitions
   (learner_base.py), on a balance grid with linear interpolation. Balances no driver has had yet
   take the value of the nearest balance that has been seen (and an unseen (k, l, u) row borrows
   the nearest seen row), so unexplored states never look better than explored ones. With
   `monotone`, the values read from a row never decrease with the balance (more carma can never
   hurt: you can always bid less); the stored values are left as learned.

The value of bidding b (relative units, at intersection j):

    Q(b) = P_unc * [ -u t_win + V(x, k-1, 0, u) ]                             (free win)
         + (1 - P_unc) * ( F_j(b)     * [ -u t_win  + V(x - b, k-1, 0, u) ]
                         + (1 - F_j(b)) * [ -u t_lose + V(x + rebate, k, l+1, u) ] )

At the last intersection (k = 1) a win leads to arrival and then to the first decision of the
driver's next trip, valued at the average V over the first-decision states seen so far, minus
rho (the average reward per trip) under the default average-cost objective, or times gamma when
discounting per trip (learner_base.py).

Equal bids go to the longest-waiting group (auction.HighestBid(tie_break="wait")), so the chance
of winning a tie grows with the auctions already lost there; it is learned per l from the ties
seen (it matters when many bid 0).

The bid is drawn from softmax(Q / T) over about 40 candidate bids (0 and 0.5%..100% of the
balance, geometric). The temperature T (seconds) anneals from t0 by t_decay per day to t_min.

Used by config.LEARNING_CONFIGS["learn_br"] and controller/learning_study.py.
"""
import math
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from learner_base import DecisionLearner

# Candidate bids, as fractions of the bidder's balance.
FRACTIONS = np.concatenate([[0.0], np.geomspace(0.005, 1.0, 39)])
# Defaults used at an intersection before anything has been measured there. The times are equal
# so that, before anything is known, neither winning nor losing looks faster.
DEFAULT_TIMES = {"win": 15.0, "lose": 15.0, "arrive": 15.0, "rebate": 0.0}


class BestResponseLearner(DecisionLearner):
    """BidStrategy: smoothed best response to the measured rival bids, with a learned value of
    carma. One object is shared by every driver that uses it."""

    def __init__(self, gamma: float = 1.0, window: int = 500, ema: float = 0.02,
                 t0: float = 0.5, t_decay: float = 0.9, t_min: float = 0.02,
                 x_max: float = 50.0, x_step: float = 0.1, k_max: int = 8, l_max: int = 2,
                 alpha_min: float = 0.01, alpha_power: float = 0.6, monotone: bool = True,
                 base_balance: float = 100.0, seed: int = 0):
        """Settings (see the module docstring and docs/learning_spec.md section 6)."""
        super().__init__(base_balance=base_balance, gamma=gamma, k_max=k_max, l_max=l_max,
                         seed=seed)
        self.window = window
        self.ema = ema
        self.t0, self.t_decay, self.t_min = t0, t_decay, t_min
        self.x_max, self.x_step = x_max, x_step
        self.alpha_min, self.alpha_power = alpha_min, alpha_power
        self.monotone = monotone
        # Whole units of carma: only whole bids are considered (learning_study --integer-carma).
        self.integer = False
        # Balance grid: steps of x_step up to 3 times the mean balance, coarser above, up to
        # x_max. It must reach far: carma can concentrate (a few drivers holding 10-20 times the
        # mean), and a driver whose balance is off the grid would value extra carma at nothing
        # and dump it into bids.
        coarse = [v for v in (3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)
                  if v < x_max]
        self.grid = np.unique(np.round(np.concatenate(
            [np.arange(0.0, 3.0 + x_step / 2, x_step), coarse, [x_max]]), 10))
        # V rows: (k, l, u) -> values on the balance grid, and the (weighted) visits per point.
        self.V: dict = {}
        self.visits: dict = {}
        # The rows as read (unseen balances filled in, made monotone), rebuilt after updates.
        self._effective_rows: dict = {}
        # Starting value before anything is learned: a running mean of the TD targets.
        self.v_init: float | None = None
        self._targets_seen = 0
        # Per intersection: contested top-rival bids (relative), whether each bidder faced a
        # rival, and the moving averages of times and rebates ({name: [mean, count]}).
        self.rivals = defaultdict(lambda: deque(maxlen=self.window))
        self.faced = defaultdict(lambda: deque(maxlen=self.window))
        self.times = defaultdict(dict)
        self._sorted = {}
        # (k, u) at the first decision of trips, to value "after arrival".
        self.first = Counter()
        # Ties: l -> [ties won, ties], for bids equal to the best rival bid.
        self.ties = defaultdict(lambda: [0, 0])

    # ---- Measured quantities -----------------------------------------------------------------

    def _ema(self, tls: str, name: str, value: float) -> None:
        """Update a moving average (a plain mean for the first 1/ema samples)."""
        mean, count = self.times[tls].get(name, (None, 0))
        count += 1
        rate = max(self.ema, 1.0 / count)
        mean = value if mean is None else mean + rate * (value - mean)
        self.times[tls][name] = [mean, count]

    def _time(self, tls, name: str) -> float:
        """A measured time (or rebate) at an intersection; pooled over intersections for
        tls=None; the default before anything was measured."""
        if tls is not None:
            entry = self.times.get(tls, {}).get(name)
            if entry is not None:
                return entry[0]
        entries = [t[name] for t in self.times.values() if name in t]
        if not entries:
            return DEFAULT_TIMES[name]
        total = sum(n for _, n in entries)
        return sum(m * n for m, n in entries) / total

    def _rival_values(self, tls) -> np.ndarray:
        """Sorted contested top-rival bids at an intersection (all intersections for None)."""
        if tls is None:
            parts = [np.asarray(v, dtype=float) for v in self.rivals.values() if len(v)]
            return np.sort(np.concatenate(parts)) if parts else np.array([])
        if tls not in self._sorted:
            self._sorted[tls] = np.sort(np.asarray(self.rivals.get(tls, ()), dtype=float))
        return self._sorted[tls]

    def tie_win(self, lost: int) -> float:
        """Chance of winning a tie after `lost` losses here (0.5 until 20 ties were seen)."""
        won, seen = self.ties.get(lost, (0, 0))
        return won / seen if seen >= 20 else 0.5

    def win_curve(self, tls, bids_rel: np.ndarray, lost: int = 0) -> tuple[float, np.ndarray]:
        """(P_unc, F(b) for every candidate bid): the chance of facing no rival, and the chance
        a bid beats the highest rival in a contested auction (a tie is won with the learned
        chance for `lost` losses here)."""
        if tls is None:
            faced = [f for d in self.faced.values() for f in d]
        else:
            faced = self.faced.get(tls, ())
        p_unc = 1.0 - (sum(faced) / len(faced)) if len(faced) else 0.5
        values = self._rival_values(tls)
        if len(values) == 0:
            return p_unc, np.where(bids_rel > 0, 1.0, 0.5)
        below = np.searchsorted(values, bids_rel, side="left")
        at_or_below = np.searchsorted(values, bids_rel, side="right")
        return p_unc, (below + self.tie_win(lost) * (at_or_below - below)) / len(values)

    # ---- Value function ----------------------------------------------------------------------

    def _nearest_row(self, k: int, lost: int, u: float):
        """The key of the row for (k, l, u), or of the nearest seen row (same urgency first)."""
        if (k, lost, u) in self.V:
            return (k, lost, u)
        best, best_distance = None, None
        for key in self.V:
            distance = abs(key[0] - k) + 0.5 * abs(key[1] - lost) + (0 if key[2] == u else 100)
            if best_distance is None or distance < best_distance:
                best, best_distance = key, distance
        return best

    def _effective(self, key) -> np.ndarray:
        """A row as it is read: balances never visited take the value of the nearest visited
        ones (linear in between, flat beyond), and with `monotone` the row is made
        non-decreasing (the mean of a running maximum from the left and a running minimum from
        the right, so neither end biases it)."""
        cached = self._effective_rows.get(key)
        if cached is not None:
            return cached
        row, visits = self.V[key], self.visits[key]
        seen = np.flatnonzero(visits > 0)
        if len(seen):
            row = np.interp(np.arange(len(row)), seen, row[seen])
        else:
            row = row.copy()
        if self.monotone:
            row = 0.5 * (np.maximum.accumulate(row) + np.minimum.accumulate(row[::-1])[::-1])
        self._effective_rows[key] = row
        return row

    def value(self, x, k: int, lost: int, u: float) -> np.ndarray:
        """V at balance(s) x (relative) for (k, l, u) (the nearest seen row if this one is new;
        the starting value if nothing has been learned yet)."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, self.x_max)
        key = self._nearest_row(k, lost, u)
        if key is None:
            return np.full(x.shape, self.v_init if self.v_init is not None else 0.0)
        return np.interp(x, self.grid, self._effective(key))

    def value_after_trip(self, x) -> np.ndarray:
        """Expected V at the first decision of the next trip, at balance(s) x."""
        x = np.asarray(x, dtype=float)
        total = sum(self.first.values())
        if total == 0:
            return np.full(x.shape, self.v_init if self.v_init is not None else 0.0)
        out = np.zeros(x.shape)
        for (k, u), n in self.first.items():
            out += n / total * self.value(x, k, 0, u)
        return out

    def _td(self, s: tuple, target: float) -> None:
        """One TD(0) step for state s towards `target`, spread over the two nearest grid points."""
        x, k, lost, u = s
        key = (k, lost, u)
        if key not in self.V:
            # A new row starts as a copy of the nearest seen row (or flat at the target).
            near = self._nearest_row(k, lost, u)
            start = self._effective(near).copy() if near is not None else \
                np.full(len(self.grid), target)
            self.V[key] = start
            self.visits[key] = np.zeros(len(self.grid))
        row, visits = self.V[key], self.visits[key]
        x = min(max(x, 0.0), self.x_max)
        i = min(int(np.searchsorted(self.grid, x, side="right")) - 1, len(self.grid) - 2)
        w1 = (x - self.grid[i]) / (self.grid[i + 1] - self.grid[i])
        w0 = 1.0 - w1
        # A grid point visited for the first time starts from the value it was read as.
        effective = self._effective(key)
        for index in (i, i + 1):
            if visits[index] == 0:
                row[index] = effective[index]
        error = target - (w0 * row[i] + w1 * row[i + 1])
        for index, weight in ((i, w0), (i + 1, w1)):
            if weight <= 0:
                continue
            visits[index] += weight
            alpha = max(self.alpha_min, visits[index] ** -self.alpha_power)
            row[index] += min(1.0, alpha) * weight * error
        self._effective_rows.pop(key, None)
        # Starting value before anything is learned: the running mean of all targets.
        self._targets_seen += 1
        rate = max(0.001, 1.0 / self._targets_seen)
        self.v_init = target if self.v_init is None else self.v_init + rate * (target - self.v_init)

    # ---- Decisions ---------------------------------------------------------------------------

    def temperature(self) -> float:
        """Softmax temperature in seconds: t0 * t_decay^(day-1), never below t_min."""
        if self.frozen:
            return self.t_min
        return max(self.t_min, self.t0 * self.t_decay ** (self.day - 1))

    def exploration(self) -> tuple[float, float]:
        """(temperature, its floor)."""
        return self.temperature(), self.t_min

    def candidate_bids(self, balance: float) -> np.ndarray:
        """The bids considered, in carma: the fractions FRACTIONS of the balance, or with whole
        units (`integer`) every whole amount up to the balance when it is at most 40 and the
        fractions rounded down otherwise."""
        if not self.integer:
            return FRACTIONS * balance
        whole = math.floor(balance + 1e-9)
        if whole <= 40:
            return np.arange(0.0, whole + 1.0)
        return np.unique(np.floor(FRACTIONS * whole))

    def q_values(self, x: float, k: int, lost: int, u: float, tls, bids=None) -> tuple:
        """(candidate bids relative to the mean balance, their Q values) at intersection tls
        (pooled over intersections for tls=None). `bids` are the candidate bids in carma
        (default: candidate_bids of the balance x * base)."""
        if bids is None:
            bids = self.candidate_bids(x * self.base)
        bids_rel = np.asarray(bids, dtype=float) / self.base
        p_unc, beat = self.win_curve(tls, bids_rel, lost)
        t_win, t_lose = self._time(tls, "win"), self._time(tls, "lose")
        rebate = self._time(tls, "rebate")
        # Within a trip nothing is discounted or charged; moving to the next trip is charged
        # -rho (average cost) or discounted by gamma (learner_base.py).
        lose = -u * t_lose + self.value(x + rebate, k, min(lost + 1, self.l_max), u)
        if k > 1:
            win_paid = -u * t_win + self.value(x - bids_rel, k - 1, 0, u)
            win_free = -u * t_win + self.value(x, k - 1, 0, u)
        else:
            t_arrive = self._time(tls, "arrive")
            cross, after = self.crossing(False), self.discount(False)
            win_paid = -u * t_arrive + cross + after * self.value_after_trip(x - bids_rel)
            win_free = -u * t_arrive + cross + after * self.value_after_trip(x)
        q = p_unc * win_free + (1.0 - p_unc) * (beat * win_paid + (1.0 - beat) * lose)
        return bids_rel, q

    def choose(self, s: tuple, car, ctx) -> tuple[float, int]:
        """Draw a candidate bid from softmax(Q / T)."""
        x, k, lost, u = s
        bids = self.candidate_bids(car.carma)
        _, q = self.q_values(x, k, lost, u, car.tls, bids)
        z = (q - q.max()) / self.temperature()
        weights = np.exp(np.maximum(z, -700.0))
        cumulative = np.cumsum(weights)
        index = int(np.searchsorted(cumulative, self.rng.random() * cumulative[-1]))
        index = min(index, len(bids) - 1)
        return float(bids[index]), index

    def greedy(self, x: float, k: int, lost: int, u: float, tls=None) -> tuple[float, float]:
        """(best fraction of the balance, best bid relative to the mean) for a state."""
        bids = self.candidate_bids(x * self.base)
        bids_rel, q = self.q_values(x, k, lost, u, tls, bids)
        index = int(np.argmax(q))
        balance = x * self.base
        fraction = float(bids[index] / balance) if balance > 0 else 0.0
        return fraction, float(bids_rel[index])

    # ---- Learning ----------------------------------------------------------------------------

    def learn(self, prev: dict, reward: float, s_next: tuple, same_trip: bool, dt: float):
        """Update the measured times and V from one transition."""
        tls, won, k_prev = prev["tls"], prev["won"], prev["s"][1]
        if tls is not None:
            if won is True:
                if same_trip:
                    self._ema(tls, "win", dt)
                elif k_prev == 1:
                    self._ema(tls, "arrive", dt)
            elif won is False and same_trip:
                self._ema(tls, "lose", dt)
        if not same_trip:
            self.first[(s_next[1], s_next[3])] += 1
        next_value = float(self.value(s_next[0], *s_next[1:]))
        self._td(prev["s"], reward + self.crossing(same_trip)
                 + self.discount(same_trip) * next_value)

    def observe_auction(self, t, tls, bids, winners, deltas, cars) -> None:
        """Learn the rival bids and rebates at this intersection from the public outcome."""
        super().observe_auction(t, tls, bids, winners, deltas, cars)
        if self.frozen:
            return
        annotated = all(b.contested is not None for b in bids)
        if len(bids) == 1 and not annotated:
            self.faced[tls].append(False)
        else:
            amounts = [b.amount for b in bids]
            winner_ids = {id(w) for w in winners}
            for i, bid in enumerate(bids):
                # The bid it had to beat (auction.annotate_rivals: with a multi-winner auction
                # only conflicting bids count, and a bid that displaced nobody was uncontested).
                if annotated:
                    if not bid.contested:
                        self.faced[tls].append(False)
                        continue
                    rival = bid.rival
                else:
                    rival = max(amounts[:i] + amounts[i + 1:])
                self.rivals[tls].append(rival / self.base)
                self.faced[tls].append(True)
                # A tie with the best rival: count it under the losses the decider had here.
                if bid.amount == rival:
                    for vid in bid.deciders():
                        car = cars.get(vid)
                        p = self.pending.get(self.key(car)) if car is not None else None
                        if p is not None and p["t"] == t:
                            entry = self.ties[p["s"][2]]
                            entry[0] += int(id(bid) in winner_ids)
                            entry[1] += 1
            for bid in bids:
                if id(bid) not in winner_ids:
                    for vid in bid.deciders():
                        self._ema(tls, "rebate", deltas.get(vid, 0.0) / self.base)
        self._sorted.pop(tls, None)

    # ---- Reporting ---------------------------------------------------------------------------

    def urgencies(self) -> list[float]:
        """Urgency levels seen so far (1 if none)."""
        seen = sorted({u for (_, u) in self.first} | {key[2] for key in self.V})
        return seen or [1.0]

    def policy_snapshot(self) -> dict:
        """The greedy bid (relative to the mean) at a typical intersection (pooled measurements)
        for every state decided in today, those states' counts, and each intersection's greedy
        bid for a driver with the mean balance and no losses at its last intersection."""
        bids = {}
        for key in self.visits_today:
            x, k, lost, u = key.split("|")
            bids[key] = self.greedy(float(x), int(k), int(lost), float(u))[1]
        per_tls = {}
        for tls in self.rivals:
            per_tls[tls] = {str(k): self.greedy(1.0, k, 0, 1.0, tls)[1] for k in (1, 2, 3)}
        return {"bids": bids, "visits": dict(self.visits_today), "per_tls": per_tls,
                "temperature": self.temperature()}

    def report(self, path: str) -> None:
        """Save the policy at a typical intersection: the greedy bid for every (x, k, l, u)."""
        rows = []
        for u in self.urgencies():
            for k in range(1, self.k_max + 1):
                for lost in range(self.l_max + 1):
                    for x in list(np.arange(0.0, 3.01, 0.25)) + [4.0, 6.0, 10.0, 20.0]:
                        fraction, bid_rel = self.greedy(float(x), k, lost, u)
                        rows.append({"x": x, "k": k, "lost_here": lost, "urgency": u,
                                     "fraction": fraction, "bid_rel": bid_rel,
                                     "value": float(self.value(x, k, lost, u))})
        pd.DataFrame(rows).to_csv(path, index=False)

    # ---- Saving ------------------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Write everything learned (and the pending decisions) to a JSON file."""
        state = self._base_state()
        state.update({
            "type": "best_response",
            "V": {f"{k}|{lost}|{u}": row.tolist() for (k, lost, u), row in self.V.items()},
            "visits": {f"{k}|{lost}|{u}": v.tolist() for (k, lost, u), v in self.visits.items()},
            "v_init": self.v_init, "targets_seen": self._targets_seen,
            "rivals": {tls: list(v) for tls, v in self.rivals.items()},
            "faced": {tls: [int(f) for f in v] for tls, v in self.faced.items()},
            "times": {tls: dict(v) for tls, v in self.times.items()},
            "first": {f"{k}|{u}": n for (k, u), n in self.first.items()},
            "ties": {str(lost): list(v) for lost, v in self.ties.items()},
            "snapshot": self.policy_snapshot(),
        })
        self._write(path, state)

    def load_state(self, path: str) -> None:
        """Read back what save_state wrote."""
        state = self._read(path)
        self._load_base_state(state)

        def key3(text):
            k, lost, u = text.split("|")
            return (int(k), int(lost), float(u))
        self.V = {key3(k): np.array(v) for k, v in state["V"].items()}
        self.visits = {key3(k): np.array(v) for k, v in state["visits"].items()}
        self._effective_rows = {}
        self.v_init = state["v_init"]
        self._targets_seen = state.get("targets_seen", 0)
        self.rivals = defaultdict(lambda: deque(maxlen=self.window))
        for tls, values in state["rivals"].items():
            self.rivals[tls].extend(values)
        self.faced = defaultdict(lambda: deque(maxlen=self.window))
        for tls, values in state["faced"].items():
            self.faced[tls].extend(bool(f) for f in values)
        self.times = defaultdict(dict, {tls: dict(v) for tls, v in state["times"].items()})
        self.first = Counter()
        for text, n in state["first"].items():
            k, u = text.split("|")
            self.first[(int(k), float(u))] = n
        self.ties = defaultdict(lambda: [0, 0],
                                {int(k): list(v) for k, v in state.get("ties", {}).items()})
        self._sorted = {}
