"""Learner B of the learning study (docs/learning_spec.md, S6): tabular Q-learning, the control for
learner A (best_response.py). Same states, rewards and transitions (learner_base.py), but no model
of the auction: one value per (state, bid).

State (x bucket, k, l, u); action = a share of the balance from regret.ACTIONS. Semi-Markov
update, since the time between decisions varies:

    Q(s, a) += alpha * (reward - rho * [new trip] + max_a' Q(s', a') - Q(s, a))

under the default average-cost objective (or with the next trip discounted by gamma < 1 instead;
see learner_base.py).

with alpha = max(alpha_min, visits^-alpha_power). Epsilon-greedy exploration falls linearly from
eps_start to eps_end over days 1..eps_days and then stays at eps_end (a fixed schedule, because
the length of a chain is not known in advance). One table is shared by every driver using it.

Used by config.LEARNING_CONFIGS["learn_q"] and controller/learning_study.py.
"""
import numpy as np
import pandas as pd

from learner_base import DecisionLearner
from regret import ACTIONS

# Balance buckets (relative to the mean balance) and a representative balance for each. They reach
# far because carma can concentrate in a few drivers (see best_response.py).
X_EDGES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
X_CENTERS = (0.05, 0.175, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 4.0, 7.5, 15.0, 30.0)


def x_bucket(x: float) -> int:
    """Index of the balance bucket of x."""
    for index, edge in enumerate(X_EDGES):
        if x < edge:
            return index
    return len(X_EDGES)


class QLearner(DecisionLearner):
    """BidStrategy: epsilon-greedy over a shared Q-table."""

    def __init__(self, gamma: float = 1.0, eps_start: float = 0.2, eps_end: float = 0.02,
                 eps_days: int = 40, alpha_min: float = 0.01, alpha_power: float = 0.6,
                 k_max: int = 8, l_max: int = 2, base_balance: float = 100.0, seed: int = 0):
        """Settings (see the module docstring and docs/learning_spec.md section 6)."""
        super().__init__(base_balance=base_balance, gamma=gamma, k_max=k_max, l_max=l_max,
                         seed=seed)
        self.eps_start, self.eps_end, self.eps_days = eps_start, eps_end, eps_days
        self.alpha_min, self.alpha_power = alpha_min, alpha_power
        # (x bucket, k, l, u) -> Q value of every action, and how often each action was updated.
        self.Q: dict = {}
        self.N: dict = {}
        # Starting value before anything is learned: a running mean of the targets.
        self.v_init: float | None = None
        self._targets_seen = 0

    def qkey(self, s: tuple) -> tuple:
        """Table key of a state."""
        x, k, lost, u = s
        return (x_bucket(x), k, lost, u)

    def _nearest(self, key: tuple):
        """The nearest state already in the table (same urgency first), or None."""
        best, best_distance = None, None
        for other in self.Q:
            distance = (abs(other[0] - key[0]) + abs(other[1] - key[1])
                        + 0.5 * abs(other[2] - key[2]) + (0 if other[3] == key[3] else 100))
            if best_distance is None or distance < best_distance:
                best, best_distance = other, distance
        return best

    def _row(self, key: tuple) -> np.ndarray:
        """The Q row of a state. A new state starts as a copy of the nearest known one, so an
        unexplored state never looks better than explored ones."""
        if key not in self.Q:
            near = self._nearest(key)
            if near is not None:
                self.Q[key] = self.Q[near].copy()
            else:
                self.Q[key] = np.full(len(ACTIONS), self.v_init or 0.0)
            self.N[key] = np.zeros(len(ACTIONS))
        return self.Q[key]

    def epsilon(self) -> float:
        """Exploration rate today."""
        if self.frozen:
            return self.eps_end
        day = min(max(self.day, 1), self.eps_days)
        return self.eps_start + (self.eps_end - self.eps_start) * (day - 1) / (self.eps_days - 1)

    def exploration(self) -> tuple[float, float]:
        """(epsilon, its floor)."""
        return self.epsilon(), self.eps_end

    def choose(self, s: tuple, car, ctx) -> tuple[float, int]:
        """Epsilon-greedy choice of a share of the balance."""
        row = self._row(self.qkey(s))
        if self.rng.random() < self.epsilon():
            action = self.rng.randrange(len(ACTIONS))
        else:
            best = np.flatnonzero(row == row.max())
            action = int(best[self.rng.randrange(len(best))])
        return ACTIONS[action] * car.carma, action

    def learn(self, prev: dict, reward: float, s_next: tuple, same_trip: bool, dt: float):
        """One semi-Markov Q-learning step for the previous decision."""
        key = self.qkey(prev["s"])
        row = self._row(key)
        action = prev["info"]
        next_value = float(self._row(self.qkey(s_next)).max())
        # Within a trip nothing is discounted or charged; moving to the next trip is charged
        # -rho (average cost) or discounted by gamma (learner_base.py).
        target = reward + self.crossing(same_trip) + self.discount(same_trip) * next_value
        self.N[key][action] += 1
        alpha = max(self.alpha_min, self.N[key][action] ** -self.alpha_power)
        row[action] += min(1.0, alpha) * (target - row[action])
        self._targets_seen += 1
        rate = max(0.001, 1.0 / self._targets_seen)
        self.v_init = target if self.v_init is None else self.v_init + rate * (target - self.v_init)

    # ---- Reporting ---------------------------------------------------------------------------

    def greedy(self, key: tuple) -> tuple[float, float]:
        """(best share of the balance, best bid relative to the mean) for a table key."""
        row = self.Q.get(key)
        if row is None:
            return float("nan"), float("nan")
        fraction = ACTIONS[int(np.argmax(row))]
        return fraction, fraction * X_CENTERS[key[0]]

    def state_key(self, s: tuple) -> str:
        """Text key of a state (its balance bucket, not a rounded balance)."""
        xb, k, lost, u = self.qkey(s)
        return f"{xb}|{k}|{lost}|{u}"

    def policy_snapshot(self) -> dict:
        """The greedy bid (relative to the mean) of every state decided in today, and those
        states' counts."""
        bids = {}
        for text in self.visits_today:
            xb, k, lost, u = text.split("|")
            bids[text] = self.greedy((int(xb), int(k), int(lost), float(u)))[1]
        return {"bids": bids, "visits": dict(self.visits_today), "epsilon": self.epsilon()}

    def report(self, path: str) -> None:
        """Save the greedy policy for every state in the table."""
        rows = []
        for key in sorted(self.Q):
            xb, k, lost, u = key
            fraction, bid_rel = self.greedy(key)
            rows.append({"x_bucket": xb, "x": X_CENTERS[xb], "k": k, "lost_here": lost,
                         "urgency": u, "fraction": fraction, "bid_rel": bid_rel,
                         "visits": int(self.N[key].sum()), "value": float(self.Q[key].max())})
        pd.DataFrame(rows).to_csv(path, index=False)

    # ---- Saving ------------------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Write the table (and the pending decisions) to a JSON file."""
        state = self._base_state()
        state.update({
            "type": "q_learning",
            "Q": {"|".join(map(str, k)): v.tolist() for k, v in self.Q.items()},
            "N": {"|".join(map(str, k)): v.tolist() for k, v in self.N.items()},
            "v_init": self.v_init, "targets_seen": self._targets_seen,
            "snapshot": self.policy_snapshot(),
        })
        self._write(path, state)

    def load_state(self, path: str) -> None:
        """Read back what save_state wrote."""
        state = self._read(path)
        self._load_base_state(state)

        def key4(text):
            xb, k, lost, u = text.split("|")
            return (int(xb), int(k), int(lost), float(u))
        self.Q = {key4(k): np.array(v) for k, v in state["Q"].items()}
        self.N = {key4(k): np.array(v) for k, v in state["N"].items()}
        self.v_init = state["v_init"]
        self._targets_seen = state.get("targets_seen", 0)
