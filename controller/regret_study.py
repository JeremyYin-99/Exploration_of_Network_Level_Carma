"""Regret-minimisation study on the 5x5 grid: can drivers LEARN a better way to spend carma?

A chain is one (strategy, demand, seed) run for DAYS days with the same 1500 persistent drivers;
balances carry from day to day (as in multiday.py) and so does what the learning strategies have
learned (the strategy object lives for the whole chain). Trips have urgency 1 (prob. 0.8) or 6
(prob. 0.2). Compared: the regret-minimising bidders (carma price lam0 = 0.5, 1, 2), hand-made
heuristics (lane urgency, urgency-scaled lane wait), and the baselines in which nothing bids: fixed
lights, actuated and delay-based lights, max-pressure and stop signs. All strategies see identical
trips and urgencies on a given day. Every run, baselines included, uses the same pool of persistent
drivers, so per-driver fairness can be compared across all of them.

Other configurations (higher prices, non-learning fixed-share controls, mixed populations of
learners and heuristic drivers) are run with --configs, e.g.:
    python controller/regret_study.py --configs regret_lane_4.0 regret_lane_8.0 \\
        frac_lane_0.9_0.15 frac_lane_0.9_0.05 mix_learn20 mix_learn80

Run from the repo root:  python controller/regret_study.py --workers 16
Analyse with:            python analyze/regret_analysis.py
"""
import argparse
import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

CONTROLLER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONTROLLER_DIR.parent
# The 5x5 scenario (network, trip window, number of drivers) and the trip generator are shared
# with multiday.py.
sys.path.insert(0, str(CONTROLLER_DIR))
from multiday import SCENARIOS, make_routes, routes_for  # noqa: E402

RESULTS_DIR = ROOT_DIR / "results" / "regret"
DAYS = 7
# Trip urgency (value of time): 1 w.p. 0.8, 6 w.p. 0.2 (CARMA section 5.2).
URGENCY_LEVELS = (1.0, 6.0)
URGENCY_PROBS = (0.8, 0.2)
SC = SCENARIOS["grid"]
# The default set of experiments: learners, heuristics, and baselines with no carma.
LEARNERS = ["regret_lane_0.5", "regret_lane_1.0", "regret_lane_2.0"]
HEURISTICS = ["only_lane_urgency", "urg_lane_wait"]
# Baselines in which nothing bids: the signal controllers (config.SIGNAL_BASELINES) and stop signs.
FIXED = ["default_lights", "actuated_lights", "delay_based_lights", "max_pressure", "stop_signs"]


def label_for(chain: dict, day: int) -> str:
    """File-name prefix of one simulated day of one chain."""
    return f"{chain['experiment']}_p{chain['p']:.2f}_s{chain['seed']}_day{day}"


def run_chain(chain: dict) -> None:
    """All days of one chain, in order (runs in a worker process)."""
    os.chdir(CONTROLLER_DIR)
    import carma
    import config

    time.sleep(chain["start_delay"])           # stagger SUMO start-ups
    name = chain["experiment"]
    # A private copy, so what a learner learns never leaks into another chain in this process.
    cfg = copy.deepcopy(config.get_experiment(name))
    # Add the urgency draw and the starting balance; save carma checkpoints twice per day.
    cfg = replace(cfg, urgency_levels=URGENCY_LEVELS, urgency_probs=URGENCY_PROBS,
                  initial_carma=100.0, trace_interval=SC["insertion"] / 2)
    # Every run uses the fixed pool of persistent drivers. The baselines spend nothing, but the
    # driver ids make per-driver results (fairness) comparable with the carma runs.
    cfg = replace(cfg, persistent_drivers=SC["drivers"])
    net = cfg.net or str(SC["net"])
    balances = None                    # day 1 starts everyone at the initial balance
    for day in range(1, chain["days"] + 1):
        label = label_for(chain, day)
        # A different trip set every day, the same for every strategy (seed and day decide it).
        route_seed = 1000 * chain["seed"] + day
        drivers_file = RESULTS_DIR / f"{label}_drivers.csv"
        # Skip days that already finished, so an interrupted chain resumes where it stopped
        # (the balances are re-read from the last finished day below; a learner's weights are not).
        if not (RESULTS_DIR / f"{label}_run.csv").exists():
            carma.run(cfg, gui=False, delay_ms=0, seed=route_seed,
                      routes=str(routes_for("grid", chain["p"], route_seed)), net=net,
                      max_time=SC["insertion"] + SC["drain"], label=label,
                      out_dir=str(RESULTS_DIR), save_history=False, quiet=True,
                      start_balances=balances, pool_seed=chain["seed"],
                      extra_info={"experiment_name": name, "p": chain["p"],
                                  "chain_seed": chain["seed"], "day": day})
        # Carry the end-of-day balances over to the next day.
        if name not in FIXED and drivers_file.exists():
            table = pd.read_csv(drivers_file)
            balances = dict(zip(table["driver"], table["final_carma"]))


def main() -> None:
    """Build one chain per (strategy, demand, seed), make the trips, and run the chains in parallel.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="+", default=LEARNERS + HEURISTICS + FIXED)
    parser.add_argument("--p-values", type=float, nargs="+", default=[1.0, 0.9])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # One chain per (strategy, demand, seed); first make sure every day's trips exist.
    chains = []
    for p in args.p_values:
        for seed in args.seeds:
            for day in range(1, args.days + 1):
                make_routes("grid", p, 1000 * seed + day)
            for name in args.configs:
                chains.append({"experiment": name, "p": p, "seed": seed, "days": args.days})
    # Slowest chains (the carma ones) first.
    chains.sort(key=lambda c: c["experiment"] in FIXED)
    for index, chain in enumerate(chains):
        chain["start_delay"] = (index % args.workers) * 1.0
    print(f"{len(chains)} chains x {args.days} days on {args.workers} workers.", flush=True)
    # Run the chains in parallel (the days inside a chain run one after the other).
    began = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_chain, c): c for c in chains}
        for future in as_completed(futures):
            chain = futures[future]
            try:
                future.result()
                print(f"  done {chain['experiment']} p={chain['p']} seed={chain['seed']} "
                      f"({(time.time() - began) / 60:.1f} min)", flush=True)
            except Exception as error:
                print(f"  FAILED {chain['experiment']} p={chain['p']} seed={chain['seed']}: "
                      f"{error!r}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
