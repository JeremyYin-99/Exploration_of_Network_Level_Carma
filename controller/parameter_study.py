"""Parameter study (docs/parameter_spec.md): how the network, lanes per road, demand, trips per
driver, urgency mix, bidding unit and starting carma change what priority costs when drivers
learn their bids, and how carma then compares with actuated lights and stop signs.

A cell is one combination of settings. A chain is (cell, strategy, seed):
  learn     learner A (best response with a learned value of carma). It learns for --days days
            (resumable; --days can be raised later to extend), then plays a frozen evaluation
            week on reserved trips.
  wait      the wait rule (bid the seconds waited x urgency), same bidding unit; no learning, so
            it plays a 7-day warm-up then the evaluation week.
  actuated  actuated traffic lights; evaluation week only.
  stop      all-way stop signs; evaluation week only.
  maxpressure  max-pressure control (Varaiya 2013, baselines.MaxPressure, 10 s minimum green);
            evaluation week only.
  uwait / random / fraction   fixed bidding policies, played like the wait rule (warm-up, then
            evaluation): seconds waited x urgency (UrgencyScaledWait), a uniform random share of
            the balance (RandomFraction), 10% of the balance (FixedFraction).
Every strategy in a cell sees the same drivers, trips and urgencies on an evaluation day, so
everything is compared paired by (seed, evaluation day). The baselines do not depend on the
bidding unit or the starting carma, so they are run once per (network, lanes, demand, trips per
driver, urgency) and shared.

Rules for every carma chain: multi-winner auction over the network's real conflict table (after
the top bid, every bid that conflicts with no winner also wins), ties to the longest wait, first
price, a winner pays only if it displaced a conflicting bid, payments go to the losing deciders,
whole units of carma. Bidding units: vehicle (every car bids alone), lane (the lane's front car
bids for the lane), phase (the front car of each signal phase bids for the phase).

Runs on libsumo (SUMO inside the Python process): identical results to the TraCI socket and
about 40 times faster.

    python controller/parameter_study.py --plan stage1 --list
    python controller/parameter_study.py --plan smoke --workers 16
    python controller/parameter_study.py --plan calibrate --workers 16
    python controller/parameter_study.py --plan stage1 --workers 16
"""
import os

os.environ.setdefault("LIBSUMO_AS_TRACI", "quiet")   # before anything imports traci

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

CONTROLLER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONTROLLER_DIR.parent
if str(CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROLLER_DIR))

NET_DIR = ROOT_DIR / "network" / "net"
ROUTES_DIR = ROOT_DIR / "simulation" / "params"
RESULTS_DIR = ROOT_DIR / "results" / "params"

INSERTION, DRAIN = 1800, 900          # seconds of trip starts, then time to empty the network
URGENCY_LEVELS = (1.0, 6.0)           # low and high urgency (CARMA section 5.2)
EVAL_DAYS = 7
WARMUP_DAYS = 7                       # the wait rule's days before its evaluation week
EVAL_SEED_OFFSET = 100000
# Which movements may be green together (conflicts.ConflictTable): "program" = whatever the
# network's own signal program shows green together, with permissive movements shown as 'g'
# (as actuated lights run them); "foes" = no two crossing movements at once (the earlier studies).
CONFLICT_MODE = "program"


# The grids are built without U-turns (netgenerate --grid ... --no-turnarounds true). With them,
# 28% of random trips on the 3x3 (20% on the 5x5) made a U-turn, and U-turns caused collisions
# and junction deadlocks under carma (docs/parameter_spec.md, "U-turns"). The earlier networks
# (grid3_200_*, RandomGrid5_200_*) are kept so the earlier studies still reproduce.
GRID_VARIANT = "noU"
# Strategies that ignore the bidding unit and starting carma: run once per baseline key, shared.
BASELINES = ("actuated", "stop", "maxpressure")


def net_files(network: str, lanes: int) -> tuple[Path, Path]:
    """(traffic-light network, all-way-stop network) for a network and lane count."""
    if network == "single":
        return NET_DIR / f"single_{lanes}lane.net.xml", NET_DIR / f"single_{lanes}lane_stop.net.xml"
    base = f"{network}_200_L{lanes}_{GRID_VARIANT}"
    return NET_DIR / f"{base}_tls.net.xml", NET_DIR / f"{base}_stop.net.xml"


def routes_dir(network: str, lanes: int) -> Path:
    """Where a network's random trips are kept (the grids' own folder per network variant)."""
    if network == "single":
        return ROUTES_DIR / f"{network}_L{lanes}"
    return ROUTES_DIR / f"{network}_L{lanes}_{GRID_VARIANT}"


# Seconds between trip starts at the moderate level with one lane per road (the earlier studies'
# levels); the calibration scans multiples of it divided by the lane count.
DEMAND_BASE = {"single": 3.0, "grid3": 2.3, "grid5": 1.0}
# Seconds between trip starts per (network, lanes) and demand level, calibrated with --plan
# calibrate so that actuated lights stop the same share of cars on every network and lane count
# at each level (docs/parameter_spec.md, "Demand"); read from DEMAND_FILE when it exists.
DEMAND_FILE = ROOT_DIR / "results" / "params" / "calibrate" / "demand.json"
DEMAND_P = {(n, lanes): {"light": round(b * 1.25 / lanes, 4), "moderate": round(b / lanes, 4),
                         "heavy": round(b * 0.9 / lanes, 4)}
            for n, b in DEMAND_BASE.items() for lanes in (1, 2, 3)}
if DEMAND_FILE.exists():
    with open(DEMAND_FILE) as _f:
        for _key, _levels in json.load(_f).items():
            _n, _lanes = _key.split("_L")
            DEMAND_P[(_n, int(_lanes))] = _levels
RANDOMTRIPS = {"single": ["--fringe-factor", "1000", "--min-distance", "100"],
               "grid3": [], "grid5": []}

REFERENCE = {"network": "grid3", "lanes": 2, "demand": "moderate", "tpd": 6, "low": 80,
             "mode": "trip", "bidding": "lane", "carma": 80, "extend": False, "drivers": "lc"}
# Driver models: "lc" = SUMO's car with lcStrategic=4 and lcKeepRight=0 (every chain, all
# strategies; docs/parameter_spec.md, "Drivers"); "sumo" = SUMO's default car, run for each
# network's reference cell as a sensitivity check.
DRIVER_FILES = {"lc": str(ROOT_DIR / "network" / "drivers" / "lane_change.add.xml"), "sumo": None}


def p_of(cell: dict) -> float:
    """Seconds between trip starts in this cell (its demand level, times `dfactor` more trips
    when the cell has one: the capacity plan)."""
    return round(float(DEMAND_P[(cell["network"], cell["lanes"])][cell["demand"]])
                 / float(cell.get("dfactor", 1.0)), 4)


def drivers_of(cell: dict) -> int:
    """Size of the persistent driver pool: the day's trips / trips per driver."""
    return max(1, round(INSERTION / p_of(cell) / cell["tpd"]))


def cell_id(cell: dict) -> str:
    """Folder name of a cell."""
    mode = "d" if cell["mode"] == "driver" else ""
    ext = "_ext" if cell.get("extend") else ""
    drv = "_sumodrv" if cell.get("drivers") == "sumo" else ""
    dx = f"_x{cell['dfactor']:g}" if cell.get("dfactor", 1.0) != 1.0 else ""
    return (f"{cell['network']}_L{cell['lanes']}_{cell['demand']}_t{cell['tpd']}"
            f"_u{cell['low']}-{100 - cell['low']}{mode}_{cell['bidding']}{ext}_c{cell['carma']}{drv}{dx}")


def baseline_id(cell: dict) -> str:
    """Folder name of a cell's baselines (they ignore the bidding unit and starting carma)."""
    mode = "d" if cell["mode"] == "driver" else ""
    drv = "_sumodrv" if cell.get("drivers") == "sumo" else ""
    dx = f"_x{cell['dfactor']:g}" if cell.get("dfactor", 1.0) != 1.0 else ""
    return (f"{cell['network']}_L{cell['lanes']}_{cell['demand']}_t{cell['tpd']}"
            f"_u{cell['low']}-{100 - cell['low']}{mode}{drv}{dx}")


def chain_dir(plan: str, cell: dict, strategy: str) -> Path:
    """Where a chain writes its files."""
    if strategy in BASELINES:
        return RESULTS_DIR / plan / "baselines" / baseline_id(cell)
    return RESULTS_DIR / plan / cell_id(cell)


# ---- Plans ---------------------------------------------------------------------------------

def vary(**changes) -> dict:
    """The reference cell with some settings changed."""
    return {**REFERENCE, **changes}


def plan_cells(plan: str) -> list[dict]:
    """The cells of a plan (duplicates removed, reference first)."""
    cells = [vary()]
    if plan == "main":
        # Every network configuration (network x lanes), each with its own reference cell and
        # one setting varied at a time from it, plus the demand x trips-per-driver grid and the
        # bidding unit x green extension grid.
        cells = []
        for network in ("grid3", "single", "grid5"):
            for lanes in (1, 2, 3):
                ref = {"network": network, "lanes": lanes}
                cells.append(vary(**ref))
                cells += [vary(**ref, bidding=b, extend=e) for b in ("vehicle", "lane", "phase")
                          for e in (False, True)]
                cells += [vary(**ref, demand=d, tpd=t) for d in ("light", "moderate", "heavy")
                          for t in (2, 6, 10)]
                cells += [vary(**ref, low=low) for low in (20, 40, 50, 60, 80)]
                cells += [vary(**ref, low=low, mode="driver") for low in (20, 40, 50, 60, 80)]
                cells += [vary(**ref, carma=c) for c in (5, 10, 20, 40, 80, 160, 320)]
                if lanes > 1:   # driver-model sensitivity check (1 lane: nobody changes lanes)
                    cells.append(vary(**ref, drivers="sumo"))
    elif plan == "quick":
        # About an hour on 16 workers (decided 2026-11-20). Each cell carries its own runs:
        # "_runs" = [(strategies, seeds), ...]; learner chains are 60 days (--days 60).
        every = ("learn", "wait", "actuated", "stop", "maxpressure")
        wb = ("wait", "actuated", "stop", "maxpressure")
        s10, s5 = tuple(range(10)), tuple(range(5))
        configs = [(n, lanes) for n in ("single", "grid3", "grid5") for lanes in (1, 2, 3)]
        small = ("grid3", 1)
        cells = []
        for network, lanes in configs:
            ref = {"network": network, "lanes": lanes}
            # A: the reference cell, every strategy, 10 seeds.
            cells.append({**vary(**ref), "_runs": [(every, s10)]})
            # B: driver-model check at 2-3 lanes (wait rule and baselines).
            if lanes > 1:
                cells.append({**vary(**ref, drivers="sumo"), "_runs": [(wb, s10)]})
            if (network, lanes) == small:
                continue
            # D: bidding unit x green extension with the wait rule, 5 seeds;
            # F: and the learner at 2 lanes on the single intersection and the 5x5.
            learner_too = lanes == 2 and network in ("single", "grid5")
            for b in ("vehicle", "lane", "phase"):
                for e in (False, True):
                    if (b, e) != ("lane", False):
                        runs = [(("wait",), s5)] + ([(("learn",), s5)] if learner_too else [])
                        cells.append({**vary(**ref, bidding=b, extend=e), "_runs": runs})
            # E: light and heavy demand (6 trips per driver), wait rule and baselines, 10 seeds.
            for d in ("light", "heavy"):
                cells.append({**vary(**ref, demand=d), "_runs": [(wb, s10)]})
        # C: every sweep on the 3x3 with one lane, learner and wait rule, 5 seeds.
        ref = {"network": small[0], "lanes": small[1]}
        cells += [{**vary(**ref, carma=c), "_runs": [(("learn",), s5)]}
                  for c in (5, 10, 20, 40, 160, 320)]
        cells += [{**vary(**ref, low=low), "_runs": [(every, s5)]} for low in (20, 40, 50, 60)]
        cells += [{**vary(**ref, low=low, mode="driver"), "_runs": [(every, s5)]}
                  for low in (20, 40, 50, 60, 80)]
        cells += [{**vary(**ref, demand=d, tpd=t), "_runs": [(every, s5)]}
                  for d in ("light", "moderate", "heavy") for t in (2, 6, 10)
                  if (d, t) != ("moderate", 6)]
        cells += [{**vary(**ref, bidding=b, extend=e), "_runs": [(("learn", "wait"), s5)]}
                  for b in ("vehicle", "lane", "phase") for e in (False, True)
                  if (b, e) != ("lane", False)]
    elif plan == "price300":
        # Learned price over 300 days (decided 2026-11-20): each network configuration's reference
        # cell, learner A only, 10 seeds (--days 300; pick networks with --networks).
        cells = [{**vary(network=n, lanes=lanes), "_runs": [(("learn",), tuple(range(10)))]}
                 for n in ("single", "grid3", "grid5") for lanes in (1, 2, 3)]
    elif plan == "capacity":
        # Demand sweep to the jam (decided 2026-11-20): how much traffic each policy handles
        # before the network jams. Learner A learns for --days days at every level.
        every = ("learn", "wait", "uwait", "random", "fraction", "actuated", "stop", "maxpressure")
        cells = [{**vary(network=n, lanes=lanes, dfactor=f), "_runs": [(every, tuple(range(5)))]}
                 for n in ("single", "grid3", "grid5") for lanes in (1, 2)
                 for f in CAPACITY_FACTORS]
    elif plan == "stage1":
        cells += [vary(network=n) for n in ("single", "grid5")]
        cells += [vary(demand=d, tpd=t) for d in ("light", "moderate", "heavy")
                  for t in (2, 6, 10)]
        cells += [vary(low=low) for low in (20, 40, 50, 60, 80)]
        cells += [vary(low=low, mode="driver") for low in (20, 40, 50, 60, 80)]
        cells += [vary(bidding=b, lanes=lanes) for b in ("vehicle", "lane", "phase")
                  for lanes in (1, 2, 3)]
        cells += [vary(carma=c) for c in (5, 10, 20, 40, 80, 160, 320)]
    elif plan == "smoke":
        cells = [vary(bidding=b, lanes=lanes) for b in ("vehicle", "lane", "phase")
                 for lanes in (1, 2, 3)]
        cells += [vary(network="single"), vary(network="grid5", lanes=3),
                  vary(mode="driver", low=50), vary(extend=True), vary(carma=5),
                  vary(drivers="sumo"), vary(network="grid5", lanes=3, drivers="sumo")]
    elif plan != "reference":
        raise ValueError(f"unknown plan {plan}")
    unique, seen = [], {}
    for cell in cells:
        cid = cell_id(cell)
        if cid not in seen:
            seen[cid] = cell
            unique.append(cell)
        elif "_runs" in cell:   # the same cell listed twice: run the union
            seen[cid]["_runs"] = seen[cid].get("_runs", []) + cell["_runs"]
    return unique


# Demand multipliers of the capacity plan (trips per day, times moderate demand).
CAPACITY_FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
# Per-plan chain settings (default: 7-day warm-up and evaluation, no early stop at a jam).
PLAN_SETTINGS = {"capacity": {"warmup_days": 3, "eval_days": 3, "jam_stop": True}}

CALIBRATION_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5)
CALIBRATION_SEEDS = (999001, 999002, 999003)
# Share of cars stopped under actuated lights at each demand level: the earlier 5x5 one-lane
# network (with U-turns) at p = 1.25, 1.0 and 0.9, mean of the three calibration trip seeds, so
# the levels keep the congestion of the earlier grid studies.
CALIBRATION_TARGETS = {"light": 0.087573, "moderate": 0.110643, "heavy": 0.131042}


def demand_levels(table: pd.DataFrame) -> dict:
    """p per (network, lanes) and demand level: linear interpolation in p of the mean stopped
    share over the calibration trip seeds, at each level's target share."""
    import numpy as np
    levels = {}
    mean = table.groupby(["network", "lanes", "p"]).stopped_share.mean().reset_index()
    for (network, lanes), g in mean.groupby(["network", "lanes"]):
        g = g.sort_values("stopped_share")
        share, p = g.stopped_share.to_numpy(), g.p.to_numpy()
        for level, target in CALIBRATION_TARGETS.items():
            if not share[0] <= target <= share[-1]:
                raise ValueError(f"{network} L{lanes}: target {target} outside the calibrated "
                                 f"range {share[0]:.3f}-{share[-1]:.3f}; widen the factors")
        levels[f"{network}_L{lanes}"] = {level: round(float(np.interp(target, share, p)), 4)
                                         for level, target in CALIBRATION_TARGETS.items()}
    return levels


def calibration_runs() -> list[dict]:
    """Actuated lights for one day at a range of demands (factors of the 1-lane moderate p,
    divided by the lane count) on every network and lane count, on three trip seeds."""
    runs = []
    for network in ("single", "grid3", "grid5"):
        base = DEMAND_BASE[network]
        for lanes in (1, 2, 3):
            for factor in CALIBRATION_FACTORS:
                for route_seed in CALIBRATION_SEEDS:
                    cell = vary(network=network, lanes=lanes)
                    runs.append({"cell": cell, "p": round(base * factor / lanes, 4),
                                 "factor": factor, "route_seed": route_seed})
    return runs


# ---- Trips -----------------------------------------------------------------------------------

def make_routes(network: str, lanes: int, p: float, route_seed: int) -> Path:
    """The day's random trips (reused if they exist). Cars enter on the best lane."""
    folder = routes_dir(network, lanes)
    routes = folder / f"p_{p:.4f}_r{route_seed}.rou.xml"
    if routes.exists():
        return routes
    folder.mkdir(parents=True, exist_ok=True)
    tag = f".tmp{os.getpid()}"
    trips_tmp = folder / (routes.name.replace(".rou.xml", "") + tag + ".trips.xml")
    routes_tmp = folder / (routes.name.replace(".rou.xml", "") + tag + ".rou.xml")
    net, _ = net_files(network, lanes)
    command = [sys.executable, os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
               "-n", str(net), "-o", str(trips_tmp), "-r", str(routes_tmp), "-b", "0",
               "-e", str(INSERTION), "-p", f"{p:.4f}", "--seed", str(route_seed), "--validate",
               "--trip-attributes", 'departLane="best"'] + RANDOMTRIPS[network]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not routes_tmp.exists():
        raise RuntimeError(f"randomTrips failed:\n{result.stdout}\n{result.stderr}")
    os.replace(trips_tmp, routes.with_suffix(".trips.xml"))
    os.replace(routes_tmp, routes)
    for leftover in folder.glob(f"*{tag}*"):
        leftover.unlink(missing_ok=True)
    return routes


# ---- One chain ---------------------------------------------------------------------------------

def build_experiment(cell: dict, strategy: str):
    """The Experiment of one strategy in one cell (in a worker process)."""
    import config
    from auction import FirstPrice, NonConflictingGreedy
    from best_response import BestResponseLearner
    from bidders import BidderType, group_by_lane, group_by_phase, group_by_vehicle
    from conflicts import get_table
    from strategies import FixedFraction, RandomFraction, UrgencyScaledWait, WaitTimeUrgency

    net_tls, net_stop = net_files(cell["network"], cell["lanes"])
    if strategy == "actuated":
        cfg = config.signal_baseline("actuated", "actuated")
    elif strategy == "stop":
        cfg = config.stop_signs(str(net_stop), "stop")
    elif strategy == "maxpressure":
        cfg = config.signal_baseline("maxpressure", "max_pressure", min_green=10.0)
    else:
        grouper, front_only = {"vehicle": (group_by_vehicle, False),
                               "lane": (group_by_lane, True),
                               "phase": (group_by_phase, True)}[cell["bidding"]]
        rule = {"learn": BestResponseLearner, "wait": lambda: WaitTimeUrgency(1.0),
                "uwait": lambda: UrgencyScaledWait(1.0), "random": RandomFraction,
                "fraction": lambda: FixedFraction(0.1)}[strategy]()
        if hasattr(rule, "base"):
            rule.base = float(cell["carma"])
            rule.integer = True
        table = get_table(str(net_tls), CONFLICT_MODE)
        cfg = config.Experiment(
            name=strategy,
            population=[BidderType(strategy, grouper, rule, sum, front_only=front_only)],
            auction=NonConflictingGreedy(table.conflicts, tie_break="wait"),
            payment=FirstPrice(redistribute=True, free_if_uncontested=True, rebate_to="bidders",
                               integer=True, free_per_bid=True),
            conflicts=table.conflicts, integer_carma=True,
            yield_table=table if CONFLICT_MODE == "program" else None,
            extend_green=bool(cell.get("extend", False)))
    return replace(cfg, persistent_drivers=drivers_of(cell), initial_carma=float(cell["carma"]),
                   urgency_levels=URGENCY_LEVELS,
                   urgency_probs=(cell["low"] / 100.0, 1.0 - cell["low"] / 100.0),
                   urgency_mode=cell["mode"], trace_interval=300.0,
                   driver_file=DRIVER_FILES[cell.get("drivers", "lc")])


def read_json(path: Path) -> dict:
    """A JSON file, or {} if it doesn't exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def read_balances(out: Path, label: str) -> dict | None:
    """End-of-day balances of every driver (None if that day has no drivers file)."""
    path = out / f"{label}_drivers.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return dict(zip(table["driver"], table["final_carma"]))


def run_chain(chain: dict) -> dict:
    """All days of one chain (in a worker process): learning or warm-up days, then the frozen
    evaluation week. Resumable: finished days (their _run.csv exists) are skipped."""
    os.chdir(CONTROLLER_DIR)
    import carma

    time.sleep(chain["start_delay"])
    cell, strategy, seed = chain["cell"], chain["strategy"], chain["seed"]
    out = chain_dir(chain["plan"], cell, strategy)
    out.mkdir(parents=True, exist_ok=True)
    cid = f"{strategy}_s{seed}"
    base_cfg = copy.deepcopy(build_experiment(cell, strategy))
    learners = [bt.strategy for bt in base_cfg.bidder_types() if hasattr(bt.strategy, "save_state")]
    learner = learners[0] if learners else None
    bids = base_cfg.control == "carma"
    days = chain["days"] if learner else (chain.get("warmup_days", WARMUP_DAYS) if bids else 0)
    eval_days = chain.get("eval_days", EVAL_DAYS)
    p = p_of(cell)
    net_tls, net_stop = net_files(cell["network"], cell["lanes"])
    net = net_stop if strategy == "stop" else net_tls
    status_path = out / f"{cid}_status.json"
    status = read_json(status_path)
    if status.get("finished") and status.get("evaluated_after") == days:
        return status

    def run_one(label: str, route_seed: int, balances, full_log: bool, extra: dict) -> dict:
        cfg = replace(base_cfg, bid_log="full" if (bids and full_log) else "none",
                      price_bin=float(chain["price_bin"]) if bids else None)
        routes = make_routes(cell["network"], cell["lanes"], p, route_seed)
        return carma.run(
            cfg, gui=False, delay_ms=0, seed=route_seed, routes=str(routes), net=str(net),
            max_time=INSERTION + DRAIN, label=label, out_dir=str(out), save_history=False,
            quiet=True, start_balances=balances, pool_seed=seed,
            jam_stop=bool(chain.get("jam_stop", False)),
            extra_info={"cell": cell_id(cell), "strategy": strategy, "chain_seed": seed, "p": p,
                        "drivers": drivers_of(cell), **{k: v for k, v in cell.items()},
                        **extra})

    # Resume the learning / warm-up days.
    last = 0
    while (out / f"{cid}_day{last + 1}_run.csv").exists():
        last += 1
    # A kill between writing a day's _run.csv and saving its balances or learner state leaves
    # that day unusable: redo it from the day before (whose files are deleted only after this
    # day's are saved).
    while last and not ((out / f"{cid}_day{last}_drivers.csv").exists() and
                        (not learner or (out / f"{cid}_day{last}_learner.json").exists())):
        (out / f"{cid}_day{last}_run.csv").unlink()
        last -= 1
    balances = read_balances(out, f"{cid}_day{last}") if last else None
    if learner and last:
        learner.load_state(str(out / f"{cid}_day{last}_learner.json"))
    t0 = time.time()
    for day in range(last + 1, days + 1):
        label = f"{cid}_day{day}"
        if learner:
            learner.frozen = False
            learner.start_day(day)
        full = day % chain["log_every"] == 0 or day > days - 30
        info = run_one(label, 1000 * seed + day, balances, full, {"day": day})
        balances = read_balances(out, label)
        if learner:
            learner.save_state(str(out / f"{label}_learner.json"))
        status.update({"chain": cid, "cell": cell_id(cell), "strategy": strategy, "seed": seed,
                       "last_day": day, "days": days,
                       "guest_trips_max": max(status.get("guest_trips_max", 0),
                                              int(info.get("guest_trips", 0))),
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        write_json(status_path, status)
        # Save disk. Every 30th day and the last 30 keep every file; other days keep the small
        # summaries (run, summary, prices) and their bid log if one was written. Learner state
        # and balances are kept every 10th day and for the latest day (needed to resume).
        if not (day % 30 == 0 or day > days - 30):
            for suffix in (".csv", "_cars.csv", "_balance_trace.csv", "_carma_trace.csv",
                           "_network_trace.csv", "_policy.csv"):
                (out / f"{label}{suffix}").unlink(missing_ok=True)
        if day > 1 and (day - 1) % 10 != 0 and day - 1 <= days - 30:
            (out / f"{cid}_day{day - 1}_learner.json").unlink(missing_ok=True)
            (out / f"{cid}_day{day - 1}_drivers.csv").unlink(missing_ok=True)

    # Evaluation week: learning frozen, reserved trips (the same for every strategy).
    last_day = max(days, last)
    # A chain extended after its evaluation week: move the old week aside so it is replayed
    # after the new last day.
    prev = status.get("evaluated_after")
    if prev is not None and prev != last_day:
        old = out / f"after{prev}"
        old.mkdir(exist_ok=True)
        for f in out.glob(f"{cid}_eval*"):
            f.rename(old / f.name)
        status.pop("finished", None)
    if learner:
        learner.frozen = True
        learner.start_day(last_day)
    for e in range(1, eval_days + 1):
        label = f"{cid}_eval{e}"
        if not (out / f"{label}_run.csv").exists():
            info = run_one(label, EVAL_SEED_OFFSET + 1000 * seed + e, balances, True,
                           {"eval_day": e, "after_day": last_day})
            status["guest_trips_max"] = max(status.get("guest_trips_max", 0),
                                            int(info.get("guest_trips", 0)))
        if bids:
            balances = read_balances(out, label)
    status.update({"evaluated_after": last_day, "finished": True,
                   "wall_minutes": round((time.time() - t0) / 60, 2),
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(status_path, status)
    return status


def run_calibration(run: dict) -> dict:
    """One day of actuated lights (calibration); returns the stopped share and time lost."""
    os.chdir(CONTROLLER_DIR)
    import carma

    time.sleep(run["start_delay"])
    cell = run["cell"]
    cfg = build_experiment(cell, "actuated")
    out = RESULTS_DIR / "calibrate" / f"{cell['network']}_L{cell['lanes']}"
    out.mkdir(parents=True, exist_ok=True)
    label = f"actuated_p{run['p']:.4f}_r{run['route_seed']}"
    net, _ = net_files(cell["network"], cell["lanes"])
    if not (out / f"{label}_run.csv").exists():
        routes = make_routes(cell["network"], cell["lanes"], run["p"], run["route_seed"])
        carma.run(replace(cfg, persistent_drivers=None), gui=False, delay_ms=0,
                  seed=run["route_seed"],
                  routes=str(routes), net=str(net), max_time=INSERTION + DRAIN, label=label,
                  out_dir=str(out), save_history=False, quiet=True)
    info = pd.read_csv(out / f"{label}_run.csv").iloc[0]
    cars = pd.read_csv(out / f"{label}_cars.csv")
    return {"network": cell["network"], "lanes": cell["lanes"], "factor": run["factor"],
            "route_seed": run["route_seed"], "p": run["p"], "stopped_share": float(info["mean_stopped_share"]),
            "time_lost": float(cars["time_loss"].mean()), "trips": len(cars),
            "teleports": int(info["teleports"]), "jammed": bool(info["jammed"])}


# ---- Main ------------------------------------------------------------------------------------

def main() -> None:
    """Build the chains of a plan and run them in parallel."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--plan", default="smoke",
                        choices=["smoke", "calibrate", "reference", "stage1", "main", "quick",
                                 "capacity", "price300"])
    parser.add_argument("--networks", nargs="+", default=None,
                        help="run only the cells on these networks (e.g. grid3 single)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--days", type=int, default=300, help="learning days of learner A")
    parser.add_argument("--strategies", nargs="+", default=["learn", "actuated", "stop"],
                        choices=["learn", "wait", "uwait", "random", "fraction", "actuated",
                                 "stop", "maxpressure"])
    parser.add_argument("--log-every", type=int, default=10,
                        help="save the full bid log every N learning days (and the last 30)")
    parser.add_argument("--price-bin", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--list", action="store_true", help="print the cells and stop")
    args = parser.parse_args()
    os.environ.setdefault("SUMO_HOME", str(ROOT_DIR / "sumo"))

    if args.plan == "calibrate":
        runs = calibration_runs()
        for i, run in enumerate(runs):
            run["start_delay"] = 0.2 * (i % args.workers)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_calibration, r) for r in runs]):
                rows.append(future.result())
        table = pd.DataFrame(rows).sort_values(["network", "lanes", "p"], ascending=[1, 1, 0])
        (RESULTS_DIR / "calibrate").mkdir(parents=True, exist_ok=True)
        table.to_csv(RESULTS_DIR / "calibrate" / "calibration.csv", index=False)
        print(table.round(3).to_string(index=False))
        levels = demand_levels(table)
        write_json(DEMAND_FILE, levels)
        print(json.dumps(levels, indent=1))
        return

    cells = plan_cells(args.plan)
    if args.networks:
        cells = [c for c in cells if c["network"] in args.networks]
    if args.list:
        for cell in cells:
            print(f"{cell_id(cell):<48} p={p_of(cell):<7} drivers={drivers_of(cell)}")
        print(f"{len(cells)} cells")
        return
    days = 3 if args.plan == "smoke" else args.days
    chains, seen = [], set()
    for cell in cells:
        runs = cell.get("_runs") or [(args.strategies, args.seeds)]
        cell = {k: v for k, v in cell.items() if k != "_runs"}
        for strategies, seeds in runs:
            for strategy, seed in itertools.product(strategies, seeds):
                key = (str(chain_dir(args.plan, cell, strategy)), strategy, seed)
                if key in seen:
                    continue
                seen.add(key)
                chains.append({"plan": args.plan, "cell": cell, "strategy": strategy,
                               "seed": seed, "days": days, "log_every": args.log_every,
                               "price_bin": args.price_bin, **PLAN_SETTINGS.get(args.plan, {})})
    # Learner chains first, the slowest networks first (they are the longest).
    speed = {"grid5": 3, "grid3": 2, "single": 1}
    chains.sort(key=lambda c: (c["strategy"] != "learn", -speed[c["cell"]["network"]],
                               -c["cell"]["lanes"]))
    for i, chain in enumerate(chains):
        chain["start_delay"] = 0.2 * (i % args.workers)
    print(f"{len(cells)} cells, {len(chains)} chains on {args.workers} workers "
          f"(learner A: {days} days + "
          f"{PLAN_SETTINGS.get(args.plan, {}).get('eval_days', EVAL_DAYS)} evaluation days)",
          flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_chain, c): c for c in chains}
        for n, future in enumerate(as_completed(futures), 1):
            chain = futures[future]
            name = f"{cell_id(chain['cell'])} {chain['strategy']}_s{chain['seed']}"
            try:
                status = future.result()
                print(f"  [{n}/{len(chains)}] done {name} ({status.get('wall_minutes')} min, "
                      f"guest trips max {status.get('guest_trips_max')})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{n}/{len(chains)}] FAILED {name}: {exc!r}", flush=True)
    print(f"Done in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
