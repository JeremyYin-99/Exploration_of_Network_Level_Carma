"""Multi-day carma: the same drivers repeat the same simulated "day" with new random trips, and
their carma balances carry over from one day to the next.

This is closer to the CARMA paper than a single day: a driver can save carma on one day and spend
it on another. A chain is one (scenario, strategy, demand, seed) run for DAYS days, one after the
other. Four checkpoints per day (quarters of the trip window, plus the start and the end of the
day) are saved as *_carma_trace.csv (summary) and *_balance_trace.csv (every driver).
Day d starts from the balances at the end of day d-1 (read from the previous day's
drivers file). The drivers (their bidder types) are fixed by the chain's seed; each day only
changes the trips (route seed) and the departure order.

"Reset carma every k days" is simulated: a reset chain (reset_every = k) puts every driver back at
the starting balance at the start of days 1, k+1, 2k+1, ... and carries balances over otherwise.
Its days 1..k are identical to the chain without resets (same trips, same seeds, same starting
balances), so only days k+1 onwards are run; the analysis takes days 1..k from the no-reset chain.
Every day of a reset chain sees the same trips as the same day of the no-reset chain, so the two
are compared day by day (analyze/multiday_analysis.py). Resets every 3 and 5 days are run by
default (--resets); a 7-day chain without resets is "reset every 7 days".

Scenarios: "single" (one 4-way intersection) and "grid" (the 5x5 network).
Run from the repo root:  python controller/multiday.py single --workers 16
                         python controller/multiday.py grid --workers 16
Analyse with:            python analyze/multiday_analysis.py

regret_study.py reuses SCENARIOS, make_routes and routes_for from this file.
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
# Days per chain. The chain without resets is the "reset every 7 days" case.
DAYS = 7
# Reset intervals (in days) simulated as separate chains next to the no-reset chain.
RESETS = [3, 5]

# One entry per scenario. insertion = seconds over which a day's trips start, drain = extra time
# for the network to empty, drivers = size of the persistent driver pool, p_values = demand
# levels (seconds between trip starts; smaller = heavier), randomtrips = extra generator options,
# stop_experiment = the stop-sign baseline's name in config.py.
SCENARIOS = {
    "single": {
        "net": ROOT_DIR / "network/net/single_4way.net.xml",
        "insertion": 3600, "drain": 900, "drivers": 120,
        "p_values": [3.0, 2.0],
        "randomtrips": ["--fringe-factor", "1000", "--min-distance", "100"],
        "stop_experiment": "stop_signs_single",
    },
    "grid": {
        "net": ROOT_DIR / "network/net/RandomGrid5_200_tls.net.xml",
        "insertion": 1800, "drain": 900, "drivers": 1500,
        "p_values": [1.0, 0.9],
        "randomtrips": [],
        "stop_experiment": "stop_signs",
    },
    # The 3x3 grid (learning study). p = 2.3 gives about the same share of stopped cars under
    # actuated lights as the 5x5 at p = 1.0 (calibrated; see learning_study.py).
    "grid3": {
        "net": ROOT_DIR / "network/net/grid3_200_tls.net.xml",
        "insertion": 1800, "drain": 900, "drivers": 130,
        "p_values": [2.3],
        "randomtrips": [],
        "stop_experiment": "stop_signs_grid3",
    },
}
# Strategies compared by default.
DEFAULT_CONFIGS = ["only_lane_urgency", "only_vehicle_random", "only_vehicle_budget"]


def routes_for(scenario: str, p: float, route_seed: int) -> Path:
    """Path of the trip file for one scenario, demand level and route seed."""
    return ROOT_DIR / "simulation" / "multiday" / scenario / f"p_{p:.2f}_r{route_seed}.rou.xml"


def make_routes(scenario: str, p: float, route_seed: int) -> Path:
    """Generate the day's random trips with SUMO's randomTrips.py (reused if they exist)."""
    routes = routes_for(scenario, p, route_seed)
    if routes.exists():
        return routes
    sc = SCENARIOS[scenario]
    routes.parent.mkdir(parents=True, exist_ok=True)
    # Write under temporary names and rename at the end, so two processes making the same file
    # at once (same seed, so the same content) never leave a half-written file behind.
    tag = f".tmp{os.getpid()}"
    trips_tmp = routes.parent / (routes.name.replace(".rou.xml", "") + tag + ".trips.xml")
    routes_tmp = routes.parent / (routes.name.replace(".rou.xml", "") + tag + ".rou.xml")
    # The route seed decides the trips, so each day has different traffic but every strategy
    # sees the same traffic on the same day.
    command = [sys.executable, os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
               "-n", str(sc["net"]), "-o", str(trips_tmp), "-r", str(routes_tmp),
               "-b", "0", "-e", str(sc["insertion"]), "-p", str(p), "--seed", str(route_seed),
               "--validate"] + sc["randomtrips"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not routes_tmp.exists():
        raise RuntimeError(f"randomTrips failed:\n{result.stdout}\n{result.stderr}")
    os.replace(trips_tmp, routes.with_suffix(".trips.xml"))
    os.replace(routes_tmp, routes)
    # randomTrips also leaves an .alt routes file next to the routes; drop the temporary one.
    for leftover in routes.parent.glob(f"*{tag}*"):
        leftover.unlink(missing_ok=True)
    return routes


def out_dir(scenario: str, results_name: str = "multiday") -> Path:
    """Folder the results of a scenario are written to (results/<results_name>/<scenario>)."""
    return ROOT_DIR / "results" / results_name / scenario


def label_for(chain: dict, day: int) -> str:
    """File-name prefix of one simulated day of one chain (reset chains add _r<k>)."""
    reset = f"_r{chain['reset_every']}" if chain.get("reset_every") else ""
    return f"{chain['experiment']}_p{chain['p']:.2f}_s{chain['seed']}{reset}_day{day}"


def run_chain(chain: dict) -> list[dict]:
    """All days of one chain, in order (runs in a worker process)."""
    # Workers are separate processes: set the working folder and import path, then import here.
    os.chdir(CONTROLLER_DIR)
    if str(CONTROLLER_DIR) not in sys.path:
        sys.path.insert(0, str(CONTROLLER_DIR))
    import carma
    import config

    time.sleep(chain["start_delay"])           # stagger SUMO start-ups
    sc = SCENARIOS[chain["scenario"]]
    name = chain["experiment"]
    # Stop signs have no bidding, so no balances to carry.
    is_stop = name.startswith("stop_signs")
    cfg = config.STOP_SIGNS_SINGLE if name == "stop_signs_single" else config.get_experiment(name)
    # Checkpoints every trace_interval seconds (default: every quarter of the trip-generation
    # window), plus the start and the end of the day: the network state for every strategy, and
    # the carma summary and every driver's balance for the ones that spend carma.
    cfg = replace(cfg, trace_interval=chain.get("trace_interval") or sc["insertion"] / 4)
    if not is_stop:
        cfg = replace(cfg, persistent_drivers=sc["drivers"], initial_carma=100.0)
    net = cfg.net or str(sc["net"])
    infos = []
    balances = None                    # day 1 starts everyone at the initial balance
    # A reset chain starts at its first reset day: the days before it are the no-reset chain's.
    reset_every = chain.get("reset_every") or 0
    first_day = reset_every + 1 if reset_every else 1
    for day in range(first_day, chain["days"] + 1):
        # Reset day: every driver starts again from the initial balance.
        if reset_every and (day - 1) % reset_every == 0:
            balances = None
        label = label_for(chain, day)
        out = out_dir(chain["scenario"], chain.get("results_name", "multiday"))
        # A different trip set every day (seed and day decide it), identical for all strategies.
        route_seed = 1000 * chain["seed"] + day
        routes = str(routes_for(chain["scenario"], chain["p"], route_seed))
        drivers_file = out / f"{label}_drivers.csv"
        # Skip days that already finished, so an interrupted chain resumes where it stopped.
        done = (out / f"{label}_run.csv").exists()
        if not done:
            info = carma.run(
                cfg, gui=False, delay_ms=0, seed=route_seed, routes=routes, net=net,
                max_time=sc["insertion"] + sc["drain"], label=label, out_dir=str(out),
                save_history=False, quiet=True, start_balances=balances,
                pool_seed=chain["seed"],
                extra_info={"scenario": chain["scenario"], "experiment_name": name,
                            "p": chain["p"], "chain_seed": chain["seed"], "day": day,
                            "reset_every": reset_every},
            )
            infos.append(info)
        # Carry the balances over to the next day (a finished day is read back from disk).
        if not is_stop and drivers_file.exists():
            table = pd.read_csv(drivers_file)
            balances = dict(zip(table["driver"], table["final_carma"]))
    return infos


def main() -> None:
    """Build one chain per (demand, seed, strategy), make the trips, and run the chains in parallel.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", choices=list(SCENARIOS))
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--p-values", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--resets", type=int, nargs="*", default=RESETS,
                        help="reset intervals (days) to simulate; give none to skip resets")
    parser.add_argument("--no-stop-signs", action="store_true")
    parser.add_argument("--trace-interval", type=float, default=None,
                        help="seconds between checkpoints (default: a quarter of the trip window)")
    parser.add_argument("--results-name", default="multiday",
                        help="results go to results/<name>/<scenario> (default: multiday)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    sc = SCENARIOS[args.scenario]
    p_values = args.p_values or sc["p_values"]
    out_dir(args.scenario, args.results_name).mkdir(parents=True, exist_ok=True)
    # Settings shared by every chain of this batch.
    common = {"scenario": args.scenario, "days": args.days, "results_name": args.results_name,
              "trace_interval": args.trace_interval}
    # The strategies to compare, plus the stop-sign baseline unless it was switched off.
    experiments = list(args.configs)
    if not args.no_stop_signs:
        experiments.append(sc["stop_experiment"])
    # One chain per (demand, seed, strategy); first make sure every day's trips exist.
    chains = []
    for p in p_values:
        for seed in args.seeds:
            for day in range(1, args.days + 1):
                make_routes(args.scenario, p, 1000 * seed + day)
            for name in experiments:
                chains.append({**common, "experiment": name, "p": p, "seed": seed})
                # Reset chains for the strategies that spend carma (stop signs have none).
                if not name.startswith("stop_signs"):
                    for k in args.resets:
                        if 0 < k < args.days:
                            chains.append({**common, "experiment": name, "p": p, "seed": seed,
                                           "reset_every": k})
    # Chains without resets first (stable sort), so their results arrive before the reset chains'.
    chains.sort(key=lambda c: bool(c.get("reset_every")))
    for index, chain in enumerate(chains):
        chain["start_delay"] = (index % args.workers) * 1.0
    print(f"{len(chains)} chains x {args.days} days on {args.workers} workers.", flush=True)
    # Run the chains in parallel (the days inside a chain run one after the other).
    began = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_chain, c): c for c in chains}
        for future in as_completed(futures):
            chain = futures[future]
            minutes = (time.time() - began) / 60
            try:
                future.result()
                print(f"  done {chain['experiment']} p={chain['p']} seed={chain['seed']} "
                      f"reset={chain.get('reset_every') or 'never'} "
                      f"({minutes:.1f} min elapsed)", flush=True)
            except Exception as error:
                # One failed chain must not stop the others.
                print(f"  FAILED {chain['experiment']} p={chain['p']} seed={chain['seed']} "
                      f"reset={chain.get('reset_every') or 'never'}: {error!r}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
