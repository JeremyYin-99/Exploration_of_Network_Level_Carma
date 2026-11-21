"""Urgency study on the single intersection: trips have a random urgency (value of time) and
the strategies, benchmarks and baselines are compared on urgency-weighted delay.

Urgency follows CARMA section 5.2: 1 with probability 0.8, 6 with probability 0.2, drawn
independently for every trip. Every run uses the same pool of 300 persistent drivers (carma is
persistent; for the baselines and benchmarks nothing is spent, but per-driver fairness becomes
comparable). Compared:
  urg_omniscient   benchmark: always serve the most urgent car (the paper's DICT)
  urg_coin         benchmark: random winner (COIN)
  only_vehicle_urgency / only_lane_urgency   urgency-blind carma strategies (existing)
  urg_vehicle_wait / urg_lane_wait / urg_vehicle_fraction   urgency-aware carma strategies
  default_lights, actuated_lights, delay_based_lights, max_pressure, stop_signs   baselines
Carma strategies run with two average balances (100 and 10), the others once.

Run from the repo root:  python controller/urgency_study.py --workers 16
Analyse with:            python analyze/urgency_analysis.py
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
NET_FILE = ROOT_DIR / "network" / "net" / "single_4way.net.xml"    # one 4-way signal junction
ROUTES_DIR = ROOT_DIR / "simulation" / "single_2h"                  # 2-hour trip files
RESULTS_DIR = ROOT_DIR / "results" / "urgency"
# Trips for 2 simulated hours, then 30 minutes for the network to drain.
INSERTION_END = 7200
DRAIN_TIME = 1800
TRIP_SEED = 42                      # same trips for every strategy
URGENCY_LEVELS = (1.0, 6.0)         # urgency 1 with probability 0.8, 6 with probability 0.2
URGENCY_PROBS = (0.8, 0.2)
DRIVERS = 300                       # size of the persistent driver pool

# Strategies that spend carma (run once per starting balance).
CARMA_CONFIGS = ["only_vehicle_urgency", "only_lane_urgency", "urg_vehicle_wait", "urg_lane_wait",
                 "urg_vehicle_fraction"]
# Baselines and benchmarks (no carma balances; run once).
FIXED_CONFIGS = ["default_lights", "actuated_lights", "delay_based_lights", "max_pressure",
                 "stop_signs", "urg_omniscient", "urg_coin"]


def make_routes(p: float) -> Path:
    """Random trips for the 2-hour window, one every p seconds on average (reused if present)."""
    routes = ROUTES_DIR / f"p_{p:.2f}.rou.xml"
    if routes.exists():
        return routes
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    trips = ROUTES_DIR / f"p_{p:.2f}.trips.xml"
    # SUMO's randomTrips.py; --validate also routes the trips and drops undrivable ones.
    command = [sys.executable, os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
               "-n", str(NET_FILE), "-o", str(trips), "-r", str(routes), "-b", "0",
               "-e", str(INSERTION_END), "-p", str(p), "--seed", str(TRIP_SEED),
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

    time.sleep(task["start_delay"])      # stagger SUMO start-ups
    name = task["experiment"]
    # Stop signs need their own network; everything else is looked up by name in config.py.
    cfg = config.STOP_SIGNS_SINGLE if name == "stop_signs" else config.get_experiment(name)
    # Add the urgency draw and this run's starting balance; checkpoint carma every 15 minutes.
    cfg = replace(cfg, urgency_levels=URGENCY_LEVELS, urgency_probs=URGENCY_PROBS,
                  initial_carma=task["initial_carma"], trace_interval=900.0)
    # Every run uses the fixed pool of persistent drivers (so per-driver results are comparable).
    cfg = replace(cfg, persistent_drivers=DRIVERS)
    return carma.run(
        cfg, gui=False, delay_ms=0, seed=task["seed"], routes=task["routes"],
        net=cfg.net or str(NET_FILE), max_time=INSERTION_END + DRAIN_TIME, label=task["label"],
        out_dir=str(RESULTS_DIR), save_history=False, quiet=True,
        extra_info={"experiment_name": name, "p": task["p"], "start_carma": task["initial_carma"],
                    "family": task["family"]})


def main() -> None:
    """Build the demand x seed x strategy x balance task list, run it in parallel and print progress.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--p-values", type=float, nargs="+", default=[3.0, 2.5])
    parser.add_argument("--carma", type=float, nargs="+", default=[100.0, 10.0])
    parser.add_argument("--configs", nargs="+", default=CARMA_CONFIGS + FIXED_CONFIGS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Task list: demand x seed x strategy x (starting balance, for carma strategies only).
    tasks = []
    for p in args.p_values:
        routes = str(make_routes(p))
        for seed in args.seeds:
            for name in args.configs:
                carmas = args.carma if name in CARMA_CONFIGS else [100.0]
                for k in carmas:
                    suffix = f"_k{k:g}" if name in CARMA_CONFIGS else ""
                    tasks.append({"experiment": name, "p": p, "seed": seed, "routes": routes,
                                  "initial_carma": k, "family": "carma" if name in CARMA_CONFIGS else "fixed",
                                  "label": f"{name}{suffix}_p{p:.2f}_seed{seed}"})
    # Skip finished runs so an interrupted batch can be restarted.
    if not args.rerun:
        tasks = [t for t in tasks if not (RESULTS_DIR / f"{t['label']}_run.csv").exists()]
    tasks.sort(key=lambda t: -t["p"])
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
                print(f"  {t['label']}: arrived={info['cars_arrived']} not_in_network={info['not_in_network']} "
                      f"({(time.time() - began) / 60:.1f} min elapsed)", flush=True)
            except Exception as error:
                print(f"  FAILED {t['label']}: {error!r}", flush=True)
    print("Done. Results are in", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
