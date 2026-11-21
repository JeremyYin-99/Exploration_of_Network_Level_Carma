"""Persistent carma at a single intersection: does carry-over carma change anything?

For every configuration, demand level and seed this runs the single 4-way intersection
(network/net/single_4way.net.xml) for several simulated hours, in up to three modes:

  default   SUMO's own signal program (no carma). One run per demand level and seed.
  reset     carma strategies where every trip starts with a fresh balance (the old behaviour).
  persist   the same strategies, but a fixed population of drivers keeps its balance from trip
            to trip (see drivers.py).

Comparing reset with persist isolates the effect of persistence; comparing both with default
shows whether carma helps at all at one intersection.

Usage (from anywhere):

    python controller/persistence.py --seeds 0 1 2
    python controller/persistence.py --p-values 2.0 1.5 --configs only_lane_urgency --drivers 300

Results go to results/persistence/, one set of files per run.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

# Folder layout, found relative to this file so the script works from any directory.
CONTROLLER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONTROLLER_DIR.parent
NET_FILE = ROOT_DIR / "network" / "net" / "single_4way.net.xml"       # one 4-way signal junction
ROUTES_DIR = ROOT_DIR / "simulation" / "single"                       # generated trip files
RESULTS_DIR = ROOT_DIR / "results" / "persistence"                    # where outputs go

# Simulated seconds over which trips are generated (4 hours: many trips per driver).
INSERTION_END = 14400
# Extra time after the last insertion for the network to drain.
DRAIN_TIME = 1800
# Seed for the trip generator, so every strategy sees exactly the same trips.
TRIP_SEED = 42

# Demand levels: p = seconds between trip starts (smaller p = heavier traffic).
DEFAULT_P = [6.0, 4.0, 3.0, 2.5, 2.0]
# Strategies compared by default (names from config.STUDY_CONFIGS).
DEFAULT_CONFIGS = ["only_vehicle_random", "only_vehicle_urgency", "only_lane_urgency",
                   "only_link_fixed", "only_price_follower", "only_vehicle_budget",
                   "only_lane_budget"]


def make_routes(p: float) -> Path:
    """Trips on the single intersection, one every p seconds on average (unless they exist)."""
    routes = ROUTES_DIR / f"p_{p:.2f}.rou.xml"
    # Reuse a route file made earlier, so reruns see identical traffic.
    if routes.exists():
        return routes
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    trips = ROUTES_DIR / f"p_{p:.2f}.trips.xml"
    # SUMO's randomTrips.py makes random origin/destination trips; --validate also computes the
    # routes and drops trips that can't be driven. Fringe factor 1000 makes trips start and end
    # on the outer roads, so every trip crosses the junction.
    command = [sys.executable, os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
               "-n", str(NET_FILE), "-o", str(trips), "-r", str(routes),
               "-b", "0", "-e", str(INSERTION_END), "-p", str(p), "--seed", str(TRIP_SEED),
               "--fringe-factor", "1000", "--min-distance", "100", "--validate"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not routes.exists():
        raise RuntimeError(f"randomTrips failed for p={p}:\n{result.stdout}\n{result.stderr}")
    return routes


def run_task(task: dict) -> dict:
    """One simulation (runs in a worker process)."""
    # Each worker is a separate process: set the folder and import path for carma.py's relative
    # paths, then import the heavy modules here (not at the top) so only workers load them.
    os.chdir(CONTROLLER_DIR)
    if str(CONTROLLER_DIR) not in sys.path:
        sys.path.insert(0, str(CONTROLLER_DIR))
    import carma
    import config

    # Stagger the SUMO start-ups by a second each, so 16 workers don't all grab a port at once.
    time.sleep(task["start_delay"])
    cfg = config.get_experiment(task["experiment"])
    # The stop-sign baseline needs its own network (junctions are all-way stops).
    if task["experiment"] == "stop_signs":
        cfg = config.STOP_SIGNS_SINGLE
    # persist: a fixed pool of drivers keeps its balance between trips.
    if task["mode"] == "persist":
        cfg = replace(cfg, persistent_drivers=task["drivers"], initial_carma=task["initial_carma"])
    # reset: every trip starts fresh with the initial balance.
    elif task["mode"] == "reset":
        cfg = replace(cfg, initial_carma=task["initial_carma"])
    return carma.run(
        cfg, gui=False, delay_ms=0, seed=task["seed"], routes=task["routes"],
        net=cfg.net or str(NET_FILE), max_time=INSERTION_END + DRAIN_TIME, label=task["label"],
        out_dir=str(RESULTS_DIR), save_history=False, quiet=True,
        extra_info={"experiment_name": task["experiment"], "mode": task["mode"], "p": task["p"],
                    "demand_per_hour": 3600 / task["p"]},
    )


def finished(label: str) -> bool:
    """True if a run with this label already wrote its summary file (so it can be skipped)."""
    return (RESULTS_DIR / f"{label}_run.csv").exists()


def main() -> None:
    """Build the demand x seed x strategy x mode task list, run it in parallel and print progress.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--p-values", type=float, nargs="+", default=DEFAULT_P)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--modes", nargs="+", default=["reset", "persist"],
                        choices=["reset", "persist"])
    parser.add_argument("--drivers", type=int, default=300, help="drivers in the persist mode")
    parser.add_argument("--initial-carma", type=float, default=100.0)
    parser.add_argument("--no-default", action="store_true", help="skip default_lights")
    parser.add_argument("--stop-signs", action="store_true",
                        help="also run the all-way-stop baseline")
    parser.add_argument("--only-stop-signs", action="store_true",
                        help="run only the all-way-stop baseline (implies --stop-signs)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--rerun", action="store_true", help="redo runs that already finished")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Build the list of simulations: demand x seed x (baselines + strategy x mode).
    tasks = []
    for p in args.p_values:
        routes = str(make_routes(p))
        for seed in args.seeds:
            if args.stop_signs or args.only_stop_signs:
                tasks.append({"experiment": "stop_signs", "mode": "default", "p": p,
                              "seed": seed, "routes": routes,
                              "label": f"stop_signs_default_p{p:.2f}_seed{seed}"})
            if args.only_stop_signs:
                continue
            if not args.no_default:
                tasks.append({"experiment": "default_lights", "mode": "default", "p": p,
                              "seed": seed, "routes": routes,
                              "label": f"default_lights_default_p{p:.2f}_seed{seed}"})
            for name in args.configs:
                for mode in args.modes:
                    tasks.append({"experiment": name, "mode": mode, "p": p, "seed": seed,
                                  "routes": routes, "drivers": args.drivers,
                                  "initial_carma": args.initial_carma,
                                  "label": f"{name}_{mode}_p{p:.2f}_seed{seed}"})
    # Skip runs whose results already exist, so an interrupted batch can simply be restarted.
    if not args.rerun:
        tasks = [t for t in tasks if not finished(t["label"])]
    # Lightest traffic first, so early results appear quickly.
    tasks.sort(key=lambda t: -t["p"])
    print(f"{len(tasks)} runs on {args.workers} workers.", flush=True)
    for index, task in enumerate(tasks):
        task["start_delay"] = (index % args.workers) * 1.0

    # Run everything in parallel, one SUMO process per worker, and print each run as it finishes.
    began = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for task in tasks:
            futures[pool.submit(run_task, task)] = task
        for future in as_completed(futures):
            task = futures[future]
            minutes = (time.time() - began) / 60
            try:
                info = future.result()
            except Exception as error:
                # One failed run must not stop the batch.
                print(f"  FAILED {task['label']}: {error!r} ({minutes:.1f} min)", flush=True)
                continue
            print(f"  {task['label']}: arrived={info['cars_arrived']} "
                  f"in_network={info['cars_in_network']} not_in_network={info['not_in_network']} "
                  f"guests={info.get('guest_trips', 0)} ({info['wall_seconds'] / 60:.1f} min run, "
                  f"{minutes:.1f} min elapsed)", flush=True)
    print("Done. Results are in", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
