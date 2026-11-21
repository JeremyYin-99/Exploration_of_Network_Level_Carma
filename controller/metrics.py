"""Per-car results and a per-bidder-type summary, so different bidding types in the same
simulation can be compared.

How these fit in: carma.py calls finish_car() once for every vehicle that leaves the network and
collects the CarRecords; at the end of a run records_to_frame() turns them into the *_cars.csv
file and summarize() into the *_summary.csv file. Every analysis script in analyze/ starts from
those per-car files.
"""
import pandas as pd

from car import Car
from core import CarRecord


def finish_car(car: Car, t: float) -> CarRecord:
    """Freeze a car's trip results at the moment it leaves the network at time `t`."""
    # Every field is copied from the Car (see car.py). The totals (acc_wait, time_loss) were
    # cached while the car was still driving, because SUMO can't be asked about a car that has
    # already arrived.
    return CarRecord(
        vid=car.vid,
        kind=car.kind,                       # which bidder type drove this trip
        depart=car.depart,                   # simulation time the car entered the network
        arrive=t,                            # simulation time it left
        travel_time=t - car.depart,          # door-to-door trip time in seconds
        wait_time=car.acc_wait,              # seconds spent stopped (speed < 0.1 m/s)
        time_loss=car.time_loss,             # seconds lost versus driving at full speed
        wins=car.wins,                       # auctions this car's group won
        final_carma=car.carma,               # balance when the trip ended
        start_carma=car.start_carma,         # balance when the trip began
        # getattr with a default, because cars that never had a driver/urgency set lack the field
        driver=getattr(car, "driver", None),     # persistent driver id (None if not persistent)
        urgency=getattr(car, "urgency", 1.0),    # value of time of this trip (1 = ordinary)
    )


def records_to_frame(records: list[CarRecord]) -> pd.DataFrame:
    """One row per car."""
    rows = []
    for record in records:
        # A dataclass keeps its fields in __dict__, which pandas turns into one table row.
        rows.append(record.__dict__)
    return pd.DataFrame(rows)


def summarize(cars: pd.DataFrame) -> pd.DataFrame:
    """Average results per bidder type: lower travel time, wait time and time loss is better.

    mean_wins and mean_final_carma show how often each type got green and what it had left.
    """
    # No finished cars (e.g. a run that jammed immediately): nothing to summarize.
    if cars.empty:
        return pd.DataFrame()
    # Group the cars by bidder type and average each result column.
    summary = cars.groupby("kind").agg(
        cars=("vid", "count"),
        mean_travel_time=("travel_time", "mean"),
        mean_wait_time=("wait_time", "mean"),
        mean_time_loss=("time_loss", "mean"),
        mean_wins=("wins", "mean"),
        mean_start_carma=("start_carma", "mean"),
        mean_final_carma=("final_carma", "mean"),
    )
    return summary.round(2)
