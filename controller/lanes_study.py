"""Lanes study: does having more lanes per approach change how carma, fixed lights and stop signs
compare?  A single 4-way intersection with 1..5 lanes on every incoming and outgoing road.

Lane layouts are SUMO's netconvert defaults (rightmost lane = right + straight, the lanes in
between = straight, leftmost lane = left + straight + U-turn). Signals use netconvert's own
fixed program (90 s cycle, NS then EW) as the "default_lights" baseline; "actuated_lights",
"delay_based_lights" and "max_pressure" switch the same phases adaptively (see baselines.py).

Demand is scaled with the lane count so every lane sees about the same load: the insertion
period is p = base / lanes seconds, with base in {3.0, 2.5, 2.0} (the p=3.0 and p=2.5 levels are
the ones used in the one-lane urgency study). Urgency is drawn as in that study (1 w.p. 0.8, 6
w.p. 0.2) and carma is persistent (one driver per 8 trips, balances start at 100).

Every carma configuration here uses a multi-winner auction (NonConflictingGreedy over the real
conflict table of the network): after the top bid wins, every lower bid that does not conflict
with the winners also wins, so e.g. two lanes of one approach can go green together. With a
single-winner auction a wide approach would be served one lane at a time, which would be unfair
to carma. (This also changes the one-lane numbers a bit compared with results/urgency.)

The same script also runs three "switching overhead" variants of the lane bidder
(config.LANE_VARIANTS): early release with a shorter yellow, a minimum green with queue
extension, and phase auctions with protected left turns.

Run from the repo root:  python controller/lanes_study.py --workers 16
Analyse with:            python analyze/lanes_analysis.py
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

# Folder layout, found relative to this file.
CONTROLLER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONTROLLER_DIR.parent
NET_DIR = ROOT_DIR / "network" / "net"              # generated .net.xml files go here
SRC_DIR = ROOT_DIR / "network" / "single"           # the node/edge description of the junction
ROUTES_DIR = ROOT_DIR / "simulation" / "lanes"      # generated trips
RESULTS_DIR = ROOT_DIR / "results" / "lanes"
# Trips are generated for 30 simulated minutes, then 15 more minutes let the network drain.
INSERTION_END = 1800
DRAIN_TIME = 900
TRIP_SEED = 42                                      # same trips for every strategy
# Trip urgency (value of time) is 1 with probability 0.8 and 6 with probability 0.2 (CARMA 5.2).
URGENCY_LEVELS = (1.0, 6.0)
URGENCY_PROBS = (0.8, 0.2)
# Persistent drivers: one driver per this many trips, so each driver makes about 8 trips.
TRIPS_PER_DRIVER = 8

LANES = [1, 2, 3, 4, 5]
# Load levels: p = base / lanes seconds between trips (each lane sees the same demand).
BASES = [3.0, 2.5, 2.0]
# Strategies that spend carma with persistent drivers (run as "carma" family).
CARMA_CONFIGS = ["only_vehicle_urgency", "only_lane_urgency", "urg_lane_wait",
                 "urg_vehicle_fraction"]
# Variants that change how long a won green is held (config.LANE_VARIANTS).
VARIANT_CONFIGS = ["lane_fast_release", "lane_min_green", "phase_urgency"]
# Baselines and benchmarks: no carma balances are spent.
FIXED_CONFIGS = ["default_lights", "actuated_lights", "delay_based_lights", "max_pressure",
                 "stop_signs", "urg_omniscient", "urg_coin"]
BENCHMARKS = ("urg_omniscient", "urg_coin")     # carma-controlled, but not carma-paid


def net_path(lanes: int, stop: bool = False) -> Path:
    """Where the network with `lanes` lanes per road lives (signal version or stop-sign version)."""
    return NET_DIR / f"single_{lanes}lane{'_stop' if stop else ''}.net.xml"


def build_nets(lanes: int) -> None:
    """Write the signal and the stop-sign network for `lanes` lanes per road (if missing)."""
    # Prefer the netconvert that ships with the repo's SUMO; otherwise rely on the PATH.
    netconvert = os.path.join(os.environ.get("SUMO_HOME", ""), "bin", "netconvert")
    if not os.path.exists(netconvert):
        netconvert = "netconvert"
    # The one-lane description is rewritten with the requested lane count.
    edges = (SRC_DIR / "single.edg.xml").read_text().replace('numLanes="1"', f'numLanes="{lanes}"')
    nodes = (SRC_DIR / "single.nod.xml").read_text()
    for stop in (False, True):
        out = net_path(lanes, stop)
        if out.exists():
            continue
        tag = f"_lanes{lanes}{'_stop' if stop else ''}"
        edge_file = SRC_DIR / f"single{tag}.edg.xml"
        node_file = SRC_DIR / f"single{tag}.nod.xml"
        edge_file.write_text(edges)
        # The stop-sign version turns the centre node from a traffic light into an all-way stop.
        node_file.write_text(nodes.replace('type="traffic_light"', 'type="allway_stop"')
                             if stop else nodes)
        # netconvert builds the junction geometry, lane connections and the default 90 s signal
        # program (phase lengths 42 / 3 / 42 / 3 seconds).
        result = subprocess.run(
            [netconvert, "-n", str(node_file), "-e", str(edge_file), "-o", str(out),
             "--tls.cycle.time", "90"], capture_output=True, text=True)
        if result.returncode != 0 or not out.exists():
            raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")


def make_routes(lanes: int, p: float) -> Path:
    """Random trips for this lane count, one every p seconds on average (reused if present)."""
    routes = ROUTES_DIR / f"lanes{lanes}_p{p:.3f}.rou.xml"
    if routes.exists():
        return routes
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    trips = routes.with_suffix(".trips.xml")
    # Same randomTrips settings as the other single-intersection studies (see persistence.py).
    command = [sys.executable, os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
               "-n", str(net_path(lanes)), "-o", str(trips), "-r", str(routes), "-b", "0",
               "-e", str(INSERTION_END), "-p", f"{p:.4f}", "--seed", str(TRIP_SEED),
               "--fringe-factor", "1000", "--min-distance", "100", "--validate"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not routes.exists():
        raise RuntimeError(f"randomTrips failed:\n{result.stdout}\n{result.stderr}")
    return routes


def run_task(task: dict) -> dict:
    """One simulation (runs in a worker process)."""
    # Workers are separate processes: set the working folder and import path, then import here.
    os.chdir(CONTROLLER_DIR)
    if str(CONTROLLER_DIR) not in sys.path:
        sys.path.insert(0, str(CONTROLLER_DIR))
    import carma
    import config
    from auction import NonConflictingGreedy
    from conflicts import get_table

    time.sleep(task["start_delay"])          # stagger SUMO start-ups
    name, lanes = task["experiment"], task["lanes"]
    lights_net = str(net_path(lanes))
    # Stop signs need a network whose junction is an all-way stop; everything else uses signals.
    cfg = config.get_experiment(name) if name != "stop_signs" else config.stop_signs(
        str(net_path(lanes, True)), "stop_signs")
    # Add the urgency draw and the starting balance, and checkpoint carma every 15 minutes.
    cfg = replace(cfg, urgency_levels=URGENCY_LEVELS, urgency_probs=URGENCY_PROBS,
                  initial_carma=100.0, trace_interval=900.0)
    net = str(net_path(lanes, True)) if name == "stop_signs" else lights_net
    if cfg.control == "carma":
        # Multi-winner auction over the network's real conflict table: lanes whose movements
        # don't cross can go green together (see the module docstring).
        table = get_table(lights_net)
        cfg = replace(cfg, auction=NonConflictingGreedy(table.conflicts),
                      conflicts=table.conflicts)
        # Strategies that spend carma use a fixed population of drivers (balances persist).
        if name in CARMA_CONFIGS + VARIANT_CONFIGS:
            cfg = replace(cfg, persistent_drivers=task["drivers"])
    return carma.run(
        cfg, gui=False, delay_ms=0, seed=task["seed"], routes=task["routes"], net=net,
        max_time=INSERTION_END + DRAIN_TIME, label=task["label"], out_dir=str(RESULTS_DIR),
        save_history=False, quiet=True,
        extra_info={"experiment_name": name, "lanes": lanes, "base": task["base"],
                    "p": task["p"], "family": task["family"]})


def main() -> None:
    """Build the lane-count x load x seed x strategy task list, run it in parallel and print
    progress.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lanes", type=int, nargs="+", default=LANES)
    parser.add_argument("--bases", type=float, nargs="+", default=BASES)
    parser.add_argument("--configs", nargs="+", default=CARMA_CONFIGS + FIXED_CONFIGS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Build the task list: lane count x load x seed x strategy.
    tasks = []
    for lanes in args.lanes:
        build_nets(lanes)
        for base in args.bases:
            p = base / lanes                 # more lanes = more trips, so each lane's load is equal
            routes = str(make_routes(lanes, p))
            # About TRIPS_PER_DRIVER trips per driver.
            drivers = max(20, round(INSERTION_END / p / TRIPS_PER_DRIVER))
            for seed in args.seeds:
                for name in args.configs:
                    tasks.append({
                        "experiment": name, "lanes": lanes, "base": base, "p": p, "seed": seed,
                        "routes": routes, "drivers": drivers,
                        "family": "carma" if name in CARMA_CONFIGS + VARIANT_CONFIGS else "fixed",
                        "label": f"{name}_L{lanes}_b{base:.1f}_seed{seed}"})
    # Skip finished runs so an interrupted batch can be restarted.
    if not args.rerun:
        tasks = [t for t in tasks if not (RESULTS_DIR / f"{t['label']}_run.csv").exists()]
    tasks.sort(key=lambda t: -t["lanes"] / t["base"])       # heaviest first
    print(f"{len(tasks)} runs on {args.workers} workers.", flush=True)
    for i, t in enumerate(tasks):
        t["start_delay"] = (i % args.workers) * 1.0
    # Run all tasks in parallel and print each one as it finishes.
    began = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for f in as_completed(futures):
            t = futures[f]
            try:
                info = f.result()
                print(f"  {t['label']}: arrived={info['cars_arrived']} "
                      f"not_in_network={info['not_in_network']} "
                      f"({(time.time() - began) / 60:.1f} min elapsed)", flush=True)
            except Exception as error:
                print(f"  FAILED {t['label']}: {error!r}", flush=True)
    print("Done. Results are in", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
