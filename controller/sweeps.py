"""Parameter sweeps: many simulations in parallel, changing one setting.

Two sweeps:

  carma    starting carma balance of every car: 50, 100, ... 500 (one experiment, default MIXED).

  demand   traffic demand, as the randomTrips period -p: one vehicle is inserted every p seconds
           over 7200 s, so a smaller p means more traffic (p = 1.0 is 7,200 vehicles, p = 0.5 is
           14,400). The sweep starts at p = 1.0 and LOWERS p step by step for every configuration
           (the default traffic lights and each carma strategy on its own), until that
           configuration jams. A configuration stops descending at the first p where most of its
           seeds jammed, so the sweep ends by itself and the last p before the jam is how much
           traffic that configuration can handle. Trips are generated once per p value in
           simulation/sweep/.

Usage (from anywhere):

    python controller/sweeps.py carma --seeds 0 1 2
    python controller/sweeps.py demand --seeds 0 1 2
    python controller/sweeps.py demand --configs default_lights only_lane_urgency --p-step 0.1

Results go to results/sweeps/<sweep>/, one set of files per run, named for the value and seed.
analyze/sweep_analysis.py summarises and plots them.

A run counts as jammed when at least 100 cars are in the network and 90% or more of them are
stopped for 300 s in a row. Jammed runs are ended there, so they cost little.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

CONTROLLER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONTROLLER_DIR.parent
NET_FILE = ROOT_DIR / "network" / "net" / "RandomGrid5_200_tls.net.xml"
SWEEP_ROUTES_DIR = ROOT_DIR / "simulation" / "sweep"
SWEEP_RESULTS_DIR = ROOT_DIR / "results" / "sweeps"

# Simulated seconds over which randomTrips inserts vehicles (the same as the existing routes).
INSERTION_END = 7200
# Seed for randomTrips, so the trip files are the same every time they are generated.
TRIP_SEED = 42

CARMA_VALUES = [50 * k for k in range(1, 11)]                       # 50 ... 500


# ---- Trip generation --------------------------------------------------------------------------

def route_file_for(p: float) -> Path:
    """Path of the trip file for demand level p."""
    return SWEEP_ROUTES_DIR / f"p_{p:.2f}.rou.xml"


def make_routes(p: float) -> Path:
    """Generate the trips and routes for period p (unless they already exist)."""
    routes = route_file_for(p)
    if routes.exists():
        return routes
    SWEEP_ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    trips = SWEEP_ROUTES_DIR / f"p_{p:.2f}.trips.xml"
    sumo_home = os.environ["SUMO_HOME"]
    command = [sys.executable, os.path.join(sumo_home, "tools", "randomTrips.py"),
               "-n", str(NET_FILE), "-o", str(trips), "-r", str(routes),
               "-b", "0", "-e", str(INSERTION_END), "-p", str(p),
               "--seed", str(TRIP_SEED), "--validate"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not routes.exists():
        raise RuntimeError(f"randomTrips failed for p={p}:\n{result.stdout}\n{result.stderr}")
    return routes


# ---- One run (executed in a worker process) ---------------------------------------------------

def run_task(task: dict) -> dict:
    """Run a single simulation described by `task`. Imports happen here, in the worker."""
    # The paths in carma.py are relative to the controller folder.
    os.chdir(CONTROLLER_DIR)
    if str(CONTROLLER_DIR) not in sys.path:
        sys.path.insert(0, str(CONTROLLER_DIR))
    import carma
    import config

    # Start the workers a little apart, so two do not ask SUMO for the same network port.
    time.sleep(task["start_delay"])

    cfg = config.get_experiment(task["experiment"])
    if task["sweep"] == "initial_carma":
        cfg = replace(cfg, initial_carma=float(task["value"]))
    routes = task["routes"] or carma.ROUTES

    info = carma.run(
        cfg, gui=False, delay_ms=0, seed=task["seed"], routes=routes, net=cfg.net or carma.NET,
        max_time=task["max_time"], label=task["label"], out_dir=task["out_dir"],
        save_history=False, quiet=True, jam_stop=task["jam_stop"],
        extra_info={"sweep": task["sweep"], "value": task["value"]},
    )
    return info


def run_tasks(tasks: list[dict], workers: int) -> list[tuple[dict, dict | None]]:
    """Run the tasks in parallel. Returns (task, run info) pairs; the info is None if the run
    crashed (the others still finish)."""
    for index, task in enumerate(tasks):
        task["start_delay"] = (index % workers) * 1.0
    results = []
    began = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for task in tasks:
            futures[pool.submit(run_task, task)] = task
        for future in as_completed(futures):
            task = futures[future]
            elapsed = (time.time() - began) / 60
            try:
                info = future.result()
            except Exception as error:     # one failed run should not stop the rest
                print(f"  FAILED {task['label']}: {error!r} ({elapsed:.1f} min)", flush=True)
                results.append((task, None))
                continue
            jam = "JAMMED at %.0fs" % info["jam_time"] if info["jammed"] else "ok"
            print(f"  {task['label']}: {jam}; arrived={info['cars_arrived']} "
                  f"in_network={info['cars_in_network']} not_in_network={info['not_in_network']} "
                  f"stopped={info['mean_stopped_share'] * 100:.0f}% "
                  f"({info['wall_seconds'] / 60:.1f} min run, {elapsed:.1f} min elapsed)",
                  flush=True)
            results.append((task, info))
    return results


# ---- The two sweeps ---------------------------------------------------------------------------

def make_task(sweep: str, experiment: str, value: float, seed: int, label: str, routes,
              max_time, jam_stop: bool) -> dict:
    """A dict describing one simulation of a sweep (what run_task needs)."""
    return {
        "sweep": sweep, "experiment": experiment, "value": value, "seed": seed,
        "routes": routes, "max_time": max_time, "label": label, "jam_stop": jam_stop,
        "out_dir": str(SWEEP_RESULTS_DIR / sweep), "start_delay": 0.0,
    }


def carma_sweep(args) -> None:
    """Run one experiment for every starting carma value and seed."""
    values = args.values or CARMA_VALUES
    tasks = []
    for value in values:
        for seed in args.seeds:
            label = f"initial_carma_{int(value):03d}_seed{seed}"
            tasks.append(make_task("initial_carma", args.experiment, value, seed, label,
                                   None, None, False))
    print(f"Carma sweep: {len(tasks)} runs of {args.experiment} on {args.workers} workers.",
          flush=True)
    run_tasks(tasks, args.workers)


def demand_levels(args) -> list[float]:
    """The demand levels (p values) to run, lightest traffic first: exactly --p-values if given,
    otherwise a ladder from p_start down to p_min in steps of p_step."""
    if args.p_values:
        return sorted(args.p_values, reverse=True)
    levels = []
    value = args.p_start
    while value >= args.p_min - 1e-9:
        levels.append(round(value, 2))
        value = round(value - args.p_step, 2)
    return levels


def run_finished(task: dict) -> bool:
    """Whether this run's results are already on disk (so it can be skipped)."""
    return (Path(task["out_dir"]) / f"{task['label']}_run.csv").exists()


def report_boundaries(configs: list[str]) -> None:
    """Print where each configuration jammed, from the run summaries on disk.

    A demand level counts as jammed for a configuration when most of its seeds jammed.
    first_jam is the lightest traffic (largest p) that jammed; last_ok is the heaviest traffic
    (smallest p) that did not jam among the levels with lighter traffic than that.
    """
    frames = []
    for path in sorted((SWEEP_RESULTS_DIR / "demand").glob("*_run.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        print("No finished demand runs.")
        return
    runs = pd.concat(frames, ignore_index=True)
    print("\nWhere each configuration jammed (p = seconds between vehicles; smaller = more traffic):")
    for name in configs:
        at = runs[runs["experiment"] == name]
        jam_levels = []
        ok_levels = []
        for level, group in at.groupby("value"):
            if group["jammed"].sum() * 2 > len(group):
                jam_levels.append(level)
            else:
                ok_levels.append(level)
        first_jam = max(jam_levels) if jam_levels else None
        ok_above = []
        for level in ok_levels:
            if first_jam is None or level > first_jam:
                ok_above.append(level)
        last_ok = min(ok_above) if ok_above else None
        print(f"  {name}: last ok p={last_ok}, first jammed p={first_jam} "
              f"({len(at['value'].unique())} levels run)", flush=True)


def demand_sweep(args) -> None:
    """Run every demand level for every configuration at the same time.

    All levels go into one pool, so no core waits for the slowest run of a level before the next
    level can start. Jammed runs end early (jam_stop), so levels beyond a configuration's jam
    point are cheap. The jam point is worked out afterwards from the results. Runs whose results
    are already on disk are skipped (use --rerun to repeat them).
    """
    import config
    configs = args.configs or list(config.STUDY_CONFIGS.keys())
    tasks = []
    skipped = 0
    # Lightest traffic first: those runs are the longest, so they get started first.
    for p in demand_levels(args):
        routes = make_routes(p)
        for name in configs:
            for seed in args.seeds:
                label = f"{name}_p{p:.2f}_seed{seed}"
                task = make_task("demand", name, p, seed, label, str(routes), args.max_time, True)
                if run_finished(task) and not args.rerun:
                    skipped += 1
                else:
                    tasks.append(task)
    print(f"Demand sweep: {len(tasks)} runs to do on {args.workers} workers "
          f"({skipped} already finished and skipped).", flush=True)
    run_tasks(tasks, args.workers)
    report_boundaries(configs)


def demand_ladder(args) -> None:
    """The sequential version: one demand level at a time, stopping each configuration at its
    first jam. Slower than demand_sweep (idle cores while the slowest run of a level finishes)."""
    import config
    configs = args.configs or list(config.STUDY_CONFIGS.keys())
    # Configurations still descending (not yet jammed), and where each one stopped.
    active = list(configs)
    boundaries = {}
    levels = demand_levels(args)
    p = args.p_start
    for p in levels:
        if not active:
            break
        routes = make_routes(p)
        tasks = []
        for name in active:
            for seed in args.seeds:
                label = f"{name}_p{p:.2f}_seed{seed}"
                tasks.append(make_task("demand", name, p, seed, label, str(routes),
                                       args.max_time, True))
        print(f"\n== p = {p:.2f} ({int(INSERTION_END / p)} vehicles): {len(tasks)} runs "
              f"({', '.join(active)}) ==", flush=True)
        results = run_tasks(tasks, args.workers)

        # Count the jammed seeds for each configuration at this p.
        jammed_count = {}
        for task, info in results:
            name = task["experiment"]
            jammed_count[name] = jammed_count.get(name, 0)
            if info is not None and info["jammed"]:
                jammed_count[name] += 1
        # A configuration stops descending once most of its seeds jam at this p.
        still_active = []
        for name in active:
            if jammed_count.get(name, 0) * 2 > len(args.seeds):
                boundaries[name] = {"first_jammed_p": p, "last_ok_p": None}
                print(f"   {name}: JAMMED at p={p:.2f}", flush=True)
            else:
                still_active.append(name)
        active = still_active

    for name in active:
        boundaries[name] = {"first_jammed_p": None, "last_ok_p": p}
    # The full table (with the last ok p for every run) comes from analyze/sweep_analysis.py.
    print("\nFirst p at which each configuration jammed (None = never, down to p=%.2f):" % p,
          flush=True)
    for name, found in boundaries.items():
        print(f"  {name}: {found['first_jammed_p']}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    """Read the command line (which sweep, seeds, workers and the sweep-specific options)."""
    parser = argparse.ArgumentParser(description="Run parameter sweeps in parallel.")
    parser.add_argument("sweep", choices=["carma", "demand"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    # carma sweep
    parser.add_argument("--experiment", default="MIXED",
                        help="carma sweep: the experiment in config.py to run (default MIXED)")
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="carma sweep: starting carma values (default 50 to 500 by 50)")
    # demand sweep
    parser.add_argument("--configs", nargs="+", default=None,
                        help="demand sweep: names from config.STUDY_CONFIGS (default: all)")
    parser.add_argument("--p-start", type=float, default=1.0)
    parser.add_argument("--p-step", type=float, default=0.05)
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--sequential", action="store_true",
                        help="demand sweep: one level at a time, stopping each configuration at "
                             "its first jam (slower, but runs fewer simulations)")
    parser.add_argument("--rerun", action="store_true",
                        help="demand sweep: repeat runs whose results already exist")
    parser.add_argument("--p-values", nargs="+", type=float, default=None,
                        help="demand sweep: run exactly these p values (instead of a ladder)")
    parser.add_argument("--max-time", type=float, default=7800,
                        help="demand sweep: stop every run at this simulated time (s)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the chosen sweep."""
    args = parse_args(argv)
    if args.sweep == "carma":
        carma_sweep(args)
    elif args.sequential:
        demand_ladder(args)
    else:
        demand_sweep(args)
    print("Done. Results are in", SWEEP_RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
