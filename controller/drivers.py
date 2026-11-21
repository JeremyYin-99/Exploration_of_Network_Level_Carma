"""Persistent drivers: a fixed population whose carma balance carries over from trip to trip.

Without this, every vehicle is a fresh Car that starts each trip with the same balance and loses
whatever is left when it arrives. With it, a vehicle that departs is handed to a Driver. The Car
starts the trip with the driver's balance, and when the trip ends the balance goes back to the
driver. So carma spent (or earned) on one trip is still there on the next one, which is what
makes saving carma for later worth anything.

A driver keeps its bidder type for its whole life, and its bid history carries over too, so
strategies that learn from `car.bid_history` keep what they learned.

How it is used (see carma.run): at departure `pool.checkout()` picks an idle driver, whose balance
and bidder type the new Car takes over; when the trip ends `pool.checkin()` returns the balance.
The pool also produces the carma checkpoints (`snapshot`, `balance_rows`) and the per-driver
results table (`to_frame`, saved as *_drivers.csv). Multi-day chains read that table's
`final_carma` column to start the next day (see multiday.py).
"""
import random
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Driver:
    """One persistent driver: an id, a bidder type, a balance and running totals."""
    # Unique id, e.g. "d17".
    id: str
    # Name of the bidder type this driver always uses.
    kind: str
    # Current balance (only updated when a trip ends; the Car holds it during a trip).
    carma: float
    # Balance at the start of the simulation.
    start_carma: float
    # Whether the driver is currently on a trip.
    on_trip: bool = False
    # Totals over all finished trips, for the per-driver results.
    trips: int = 0
    wins: int = 0
    wait_sum: float = 0.0
    time_loss_sum: float = 0.0
    travel_sum: float = 0.0
    # The driver's own bids over all trips so far (a Car shares this list during a trip).
    bid_history: list = field(default_factory=list)


class DriverPool:
    """A fixed set of drivers. `checkout` hands one to a departing vehicle, `checkin` takes the
    balance back when the trip ends."""

    def __init__(self, n_drivers: int, types: list, default_carma: float, rng: random.Random):
        """Create `n_drivers` drivers, giving each a bidder type drawn in proportion to the types'
        share.
        """
        # The random stream used to give drivers their types and to pick who drives next.
        self.rng = rng
        self.drivers: dict[str, Driver] = {}
        # Idle drivers (not on a trip), so a free one can be picked quickly.
        self._idle: list[str] = []
        # Each driver's bidder type is drawn in proportion to the types' `share`.
        weights = []
        for btype in types:
            weights.append(btype.share)
        for number in range(n_drivers):
            btype = rng.choices(types, weights=weights)[0]
            # A type may have its own starting balance; otherwise use the experiment's.
            if btype.initial_carma is not None:
                start = btype.initial_carma
            else:
                start = default_carma
            driver = Driver(id=f"d{number}", kind=btype.name, carma=start, start_carma=start)
            self.drivers[driver.id] = driver
            self._idle.append(driver.id)
        # Departures that found nobody free (the vehicle then gets a throwaway balance).
        self.guest_trips = 0

    def checkout(self) -> Driver | None:
        """A random idle driver, now on a trip. None if every driver is already driving."""
        # More vehicles than drivers: the extra vehicle drives as a "guest" with a fresh balance.
        if not self._idle:
            self.guest_trips += 1
            return None
        index = self.rng.randrange(len(self._idle))
        # Swap-remove: O(1) removal of a random element.
        self._idle[index], self._idle[-1] = self._idle[-1], self._idle[index]
        driver = self.drivers[self._idle.pop()]
        driver.on_trip = True
        return driver

    def checkin(self, driver: Driver, car) -> None:
        """Take a finished trip's balance and results back into the driver."""
        # The balance the car ends with is the driver's balance for the next trip.
        driver.carma = car.carma
        driver.trips += 1
        driver.wins += car.wins
        driver.wait_sum += car.acc_wait
        driver.time_loss_sum += car.time_loss
        driver.on_trip = False
        self._idle.append(driver.id)

    def balances(self, cars: dict) -> dict[str, float]:
        """Every driver's current balance, including trips in progress. `cars` maps vehicle id
        to Car."""
        current = {}
        # Idle drivers hold their own balance ...
        for driver in self.drivers.values():
            current[driver.id] = driver.carma
        # ... but a driver on a trip is represented by the Car, whose balance is more recent.
        for car in cars.values():
            if getattr(car, "driver", None) is not None:
                current[car.driver] = car.carma
        return current

    def snapshot(self, t: float, cars: dict) -> list[dict]:
        """Summary of the carma distribution at time t: one row for everyone and one per
        bidder type."""
        current = self.balances(cars)
        groups = {"all": []}
        for driver in self.drivers.values():
            groups["all"].append(current[driver.id])
            groups.setdefault(driver.kind, []).append(current[driver.id])
        rows = []
        for name, values in groups.items():
            rows.append(carma_stats(t, name, values))
        return rows

    def balance_rows(self, t: float, cars: dict) -> list[dict]:
        """Every driver's balance at time t (for the per-driver checkpoints)."""
        current = self.balances(cars)
        return [{"t": t, "driver": d.id, "kind": d.kind, "carma": current[d.id]}
                for d in self.drivers.values()]

    def to_frame(self, cars: dict) -> pd.DataFrame:
        """One row per driver: who they are, how many trips, their average delay and balance."""
        current = self.balances(cars)
        rows = []
        for driver in self.drivers.values():
            # max(1, ...) avoids dividing by zero for a driver who never drove.
            trips = max(1, driver.trips)
            rows.append({
                "driver": driver.id,
                "kind": driver.kind,
                "trips": driver.trips,
                "wins": driver.wins,
                "start_carma": driver.start_carma,
                "final_carma": current[driver.id],      # read by multiday.py for the next day
                "mean_wait": driver.wait_sum / trips,
                "mean_time_loss": driver.time_loss_sum / trips,
            })
        return pd.DataFrame(rows)


def gini(values: list[float]) -> float:
    """Inequality of a list of balances: 0 = everyone equal, 1 = one holder has everything.

    Formula (values sorted ascending, ranks 1..n):  G = 2 * sum(rank * value) / (n * total) - (n + 1) / n
    """
    ordered = sorted(values)
    n = len(ordered)
    total = sum(ordered)
    # An empty or all-zero list has no inequality to measure.
    if n == 0 or total <= 0:
        return 0.0
    # Weight each value by its rank: richer holders count for more.
    weighted = 0.0
    for rank in range(n):
        weighted += (rank + 1) * ordered[rank]
    return (2 * weighted) / (n * total) - (n + 1) / n


def carma_stats(t: float, name: str, values: list[float]) -> dict:
    """The numbers recorded for a group of balances at one moment."""
    ordered = sorted(values)
    n = len(ordered)
    total = sum(ordered)
    # Share of all carma held by the richest 10% of the group.
    top_count = max(1, n // 10)
    top_sum = sum(ordered[n - top_count:])
    mean = total / n
    # Population variance, computed by hand to avoid importing numpy here.
    variance = 0.0
    for v in ordered:
        variance += (v - mean) ** 2
    return {
        "t": t, "kind": name, "drivers": n, "mean": mean,
        "std": (variance / n) ** 0.5, "min": ordered[0],
        "median": ordered[n // 2], "max": ordered[-1],
        "gini": gini(ordered),
        "top10_share": top_sum / total if total > 0 else 0.0,
    }
