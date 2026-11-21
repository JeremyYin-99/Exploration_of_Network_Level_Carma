"""Learning study (docs/learning_spec.md): do self-interested drivers who learn from repeated play
converge on a bidding policy, and what does priority then cost at each intersection?

A chain is one (scenario, strategy, seed): the same persistent drivers play one simulated day
after another, with new trips every day; balances and what the learners learned carry over.

Run length (learning strategies): at least --min-days (60) and at most --max-days (90) days. From
day --min-days on, a chain stops as soon as the convergence rule holds (convergence.py; section 7
of the spec), unless --no-early-stop. A chain that reaches --max-days without converging stops
with converged = false; --extend-to D continues it (from its saved state) up to day D. Strategies
that don't learn run --baseline-days (7) days.

Evaluation block: after a chain stops, --eval-days (7) more days are run with learning frozen, on
reserved trips (route seeds 100000 + 1000 * seed + e) that every strategy sees, so outcomes can be
compared paired by (seed, evaluation day) even when chains stopped on different days.

Everything is resumable: finished days are skipped, the learner state and balances are re-read
from the last finished day.

Files, in results/<results-name>/<scenario>/ (chain = <strategy>_s<seed>):
  <chain>_day<d>*            the usual run outputs of each day (carma.py), plus
      _bids.csv.gz           every decision bid (bid log, --bid-log full)
      _prices.csv            per-intersection price summary (300 s bins and the whole day)
      _learner.json          the learner's state after the day (resume / freeze / analysis)
      _learnstats.json       policy change (M1) and exploration level of the day
      _policy.csv            the learner's policy table after the day
  <chain>_after<D>_eval<e>*  the evaluation days (after learning day D)
  <chain>_prices_by_day.csv  whole-day price rows of every day (what the evolution plots read)
  <chain>_convergence.csv    the convergence metrics of every day
  <chain>_status.json        where the chain is: last day, converged or not, and when

Exploitability test (--mutants, spec S10), run after the learning chains: the population's
learned policy is frozen, a share of the drivers (--mutant-share, 10%) are "mutants" who keep
learning against it for --mutant-days (14) days, and the same days are run again with the mutants
frozen too. Same drivers, same trips (reserved route seeds 200000 + 1000 * seed + d); only the
mutants' policy differs. If the mutants cannot lower their time lost (difference in differences
against the population, paired by day), the population's policy is an approximate equilibrium.
By default the mutants start from the population's learned state with exploration at its floor
(--mutant-start warm: can a driver who knows what the others know do better by deviating?);
--mutant-start fresh starts them from scratch with --mutant-config (e.g. learn_q).
Files: <chain>_mut_learn_day<d>*, <chain>_mut_frozen_day<d>*, <chain>_mutants.csv / .json.

Scenarios: single (1 intersection), grid3 (3x3), grid5 (5x5). Run from the repo root:
    python controller/learning_study.py --scenarios single --workers 8
    python controller/learning_study.py --scenarios grid3 grid5 --workers 8
    python controller/learning_study.py --scenarios grid5 --configs learn_br --extend-to 120
    python controller/learning_study.py --scenarios single --configs learn_br --mutants
Analyse with:  python analyze/learning_analysis.py
"""
import argparse
import copy
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

CONTROLLER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONTROLLER_DIR.parent
sys.path.insert(0, str(CONTROLLER_DIR))
from convergence import Settings, convergence_table, policy_change  # noqa: E402
from multiday import make_routes, routes_for  # noqa: E402

# One entry per scenario: the folder of its trip files (shared with multiday.py), the network,
# the stop-sign experiment, the trip window and drain time, the driver pool (about 6 trips per
# driver per day on the grids) and the demand (seconds between trip starts).
# The 3x3 demand was calibrated (phase P0) so that actuated lights stop about the same share of
# cars as on the 5x5 at p = 1.0: p = 2.3 gives 9-10% stopped and 23 s time lost (5x5: 10.6%,
# 35 s, with longer trips); p = 1.67 jammed and p = 2.0 stopped 15%.
SCENARIOS = {
    "single": {"routes": "single", "net": "network/net/single_4way.net.xml",
               "stop": "STOP_SIGNS_SINGLE", "insertion": 3600, "drain": 900,
               "drivers": 120, "p": 3.0},
    "grid3": {"routes": "grid3", "net": "network/net/grid3_200_tls.net.xml",
              "stop": "STOP_SIGNS_GRID3", "insertion": 1800, "drain": 900,
              "drivers": 130, "p": 2.3},
    "grid5": {"routes": "grid", "net": "network/net/RandomGrid5_200_tls.net.xml",
              "stop": "STOP_SIGNS", "insertion": 1800, "drain": 900,
              "drivers": 300, "p": 1.0},
}
LEARNERS = ["learn_br", "learn_q", "learn_hedge"]
BASELINES = ["front_wait", "only_lane_urgency", "actuated_lights", "stop_signs"]
EVAL_SEED_OFFSET = 100000
MUTANT_SEED_OFFSET = 200000


def route_seed(seed: int, day: int) -> int:
    """Trip seed of a learning day (days 1..999)."""
    return 1000 * seed + day


def eval_route_seed(seed: int, e: int) -> int:
    """Trip seed of evaluation day e (never used for learning)."""
    return EVAL_SEED_OFFSET + 1000 * seed + e


def out_dir(results_name: str, scenario: str) -> Path:
    """Folder of one scenario's results."""
    return ROOT_DIR / "results" / results_name / scenario


def chain_id(chain: dict) -> str:
    """File-name prefix of a chain."""
    return f"{chain['experiment']}_s{chain['seed']}"


def experiment_for(name: str, scenario: str, config):
    """The experiment of a strategy name (stop signs use the scenario's stop-sign network)."""
    if name == "stop_signs":
        return getattr(config, SCENARIOS[scenario]["stop"])
    return config.get_experiment(name)


def finished_days(out: Path, cid: str) -> list[int]:
    """Learning days of a chain that have finished (their _run.csv exists)."""
    days = []
    for path in glob.glob(str(out / f"{cid}_day*_run.csv")):
        match = re.search(r"_day(\d+)_run\.csv$", path)
        if match:
            days.append(int(match.group(1)))
    return sorted(days)


def read_balances(out: Path, label: str) -> dict | None:
    """End-of-day balances of every driver (None if the day has no drivers file)."""
    path = out / f"{label}_drivers.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return dict(zip(table["driver"], table["final_carma"]))


def read_json(path: Path) -> dict:
    """A JSON file, or {} if it doesn't exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (a crash never leaves a half-written status file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def price_history(out: Path, cid: str) -> pd.DataFrame:
    """The whole-day price rows of every learning day of a chain, with a day column."""
    parts = []
    for path in glob.glob(str(out / f"{cid}_day*_prices.csv")):
        match = re.search(r"_day(\d+)_prices\.csv$", path)
        if not match:
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        table = table[table["scope"] == "day"].copy()
        table.insert(0, "day", int(match.group(1)))
        parts.append(table)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["day", "tls"])


def learn_stats(out: Path, cid: str) -> pd.DataFrame:
    """Policy change (M1) and exploration of every learning day of a chain."""
    rows = []
    for path in glob.glob(str(out / f"{cid}_day*_learnstats.json")):
        rows.append(read_json(Path(path)))
    if not rows:
        return pd.DataFrame(columns=["day", "m1", "exploration", "floor"])
    return pd.DataFrame(rows).sort_values("day")


def run_day(carma, cfg, chain: dict, sc: dict, out: Path, label: str, seed_of_trips: int,
            balances: dict | None, extra: dict) -> dict:
    """Simulate one day of a chain."""
    p = chain["p"]
    make_routes(sc["routes"], p, seed_of_trips)
    net = cfg.net or str(ROOT_DIR / sc["net"])
    return carma.run(
        cfg, gui=False, delay_ms=0, seed=seed_of_trips,
        routes=str(routes_for(sc["routes"], p, seed_of_trips)), net=net,
        max_time=sc["insertion"] + sc["drain"], label=label, out_dir=str(out),
        save_history=False, quiet=True, start_balances=balances, pool_seed=chain["seed"],
        extra_info={"scenario": chain["scenario"], "experiment_name": chain["experiment"],
                    "p": p, "chain_seed": chain["seed"],
                    "initial_carma": chain.get("initial_carma", 100.0),
                    "integer_carma": chain.get("integer", False), **extra})


def apply_carma_settings(cfg, chain: dict):
    """The starting carma and whole units of a chain: sets the experiment's initial balance and,
    with whole units, rounds bids (make_bid) and makes the payment rule pay whole units; the
    learners measure balances against the starting amount (their `base`). Returns the new
    experiment (cfg is a private copy, so its strategies and payment rule may be changed)."""
    cfg = replace(cfg, initial_carma=float(chain["initial_carma"]),
                  integer_carma=bool(chain["integer"]))
    if chain["integer"] and hasattr(cfg.payment, "integer"):
        cfg.payment.integer = True
    for btype in cfg.bidder_types():
        strategy = btype.strategy
        if hasattr(strategy, "base"):
            strategy.base = float(chain["initial_carma"])
        if hasattr(strategy, "integer"):
            strategy.integer = bool(chain["integer"])
    return cfg


def run_chain(chain: dict) -> dict:
    """All days of one chain (runs in a worker process). Returns its final status."""
    os.chdir(CONTROLLER_DIR)
    import carma
    import config

    time.sleep(chain["start_delay"])            # stagger SUMO start-ups
    sc = SCENARIOS[chain["scenario"]]
    name = chain["experiment"]
    # A private copy, so what a learner learns never leaks into another chain in this process.
    cfg = copy.deepcopy(experiment_for(name, chain["scenario"], config))
    bids = cfg.control == "carma"
    cfg = replace(cfg, persistent_drivers=chain["drivers"],
                  trace_interval=chain["trace_interval"],
                  bid_log=chain["bid_log"] if bids else "none",
                  price_bin=chain["price_bin"] if bids else None,
                  urgency_levels=tuple(chain["urgency"]),
                  urgency_probs=tuple(chain["urgency_probs"]))
    cfg = apply_carma_settings(cfg, chain)
    learners = [bt.strategy for bt in cfg.bidder_types() if hasattr(bt.strategy, "save_state")]
    learner = learners[0] if learners else None
    out = out_dir(chain["results_name"], chain["scenario"])
    out.mkdir(parents=True, exist_ok=True)
    cid = chain_id(chain)
    settings = Settings(window=chain["window"], consecutive=chain["consecutive"],
                        min_days=chain["min_days"])
    status_path = out / f"{cid}_status.json"
    status = read_json(status_path)
    max_days = chain["max_days"] if learner else chain["baseline_days"]

    # Resume from the last finished learning day.
    done = finished_days(out, cid)
    last = done[-1] if done else 0
    balances = read_balances(out, f"{cid}_day{last}") if last else None
    previous_snapshot = None
    if learner and last:
        state_file = out / f"{cid}_day{last}_learner.json"
        learner.load_state(str(state_file))
        previous_snapshot = read_json(state_file).get("snapshot")
    converged = bool(status.get("converged")) and chain["early_stop"]

    day = last + 1
    while day <= max_days and not converged:
        label = f"{cid}_day{day}"
        if learner:
            learner.frozen = False
            learner.start_day(day)
        info = run_day(carma, cfg, chain, sc, out, label, route_seed(chain["seed"], day),
                       balances, {"day": day})
        balances = read_balances(out, label)
        status.setdefault("guest_trips_max", 0)
        status["guest_trips_max"] = max(status["guest_trips_max"], int(info.get("guest_trips", 0)))
        if learner:
            snapshot = learner.policy_snapshot()
            learner.save_state(str(out / f"{label}_learner.json"))
            level, floor = learner.exploration()
            write_json(out / f"{label}_learnstats.json", {
                "day": day, "m1": policy_change(previous_snapshot, snapshot, settings.min_visits),
                "exploration": level, "floor": floor, "wall_seconds": info.get("wall_seconds")})
            previous_snapshot = snapshot
        # The chain's price history and convergence table, rebuilt from the day files.
        prices = price_history(out, cid)
        if not prices.empty:
            prices.to_csv(out / f"{cid}_prices_by_day.csv", index=False)
        if learner:
            table = convergence_table(prices, learn_stats(out, cid), settings)
            table.to_csv(out / f"{cid}_convergence.csv", index=False)
            latest = table.iloc[-1]
            converged = bool(latest["converged"]) and chain["early_stop"]
            status.update({"converged": bool(latest["converged"]),
                           "converged_on": day if latest["converged"] else None,
                           "consecutive_passes": int(latest["consecutive"]),
                           "m2_failing": latest["m2_failing"], "m3_failing": latest["m3_failing"]})
        status.update({"chain": cid, "scenario": chain["scenario"], "experiment": name,
                       "seed": chain["seed"], "last_day": day, "max_days": max_days,
                       "min_days": chain["min_days"], "learner": bool(learner),
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        write_json(status_path, status)
        day += 1
    last_day = day - 1

    # Evaluation block: learning frozen, reserved trips.
    if chain["evaluate"] and last_day > 0:
        if learner:
            learner.frozen = True
            learner.start_day(last_day)
        for e in range(1, chain["eval_days"] + 1):
            label = f"{cid}_after{last_day}_eval{e}"
            if not (out / f"{label}_run.csv").exists():
                run_day(carma, cfg, chain, sc, out, label, eval_route_seed(chain["seed"], e),
                        balances, {"eval_day": e, "after_day": last_day})
            balances = read_balances(out, label)
        status["evaluated_after"] = last_day
    status["finished"] = True
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(status_path, status)
    return status


def mutant_experiment(base_cfg, population, mutant, share: float):
    """The base experiment with two bidder types: the frozen population and the mutants (front-car
    bidding, same auction and payment rule). The driver pool gives every driver one of them, the
    same way in every run with the same pool seed."""
    from bidders import BidderType, group_by_lane
    return replace(base_cfg, population=[
        BidderType("population", group_by_lane, population, sum, share=1.0 - share,
                   front_only=True),
        BidderType("mutant", group_by_lane, mutant, sum, share=share, front_only=True)])


def run_mutants(chain: dict) -> dict:
    """The exploitability test of one finished learning chain (runs in a worker process)."""
    os.chdir(CONTROLLER_DIR)
    import carma
    import config
    from convergence import t975

    time.sleep(chain["start_delay"])
    sc = SCENARIOS[chain["scenario"]]
    out = out_dir(chain["results_name"], chain["scenario"])
    cid = chain_id(chain)
    status = read_json(out / f"{cid}_status.json")
    last = status.get("last_day")
    if not last:
        raise RuntimeError(f"{cid}: no finished learning chain to test")
    state_file = str(out / f"{cid}_day{last}_learner.json")
    base = copy.deepcopy(config.get_experiment(chain["experiment"]))
    template = base.bidder_types()[0].strategy
    base = replace(base, persistent_drivers=chain["drivers"],
                   trace_interval=chain["trace_interval"], bid_log="none", price_bin=None,
                   urgency_levels=tuple(chain["urgency"]),
                   urgency_probs=tuple(chain["urgency_probs"]))
    base = apply_carma_settings(base, chain)

    def population_learner():
        learner = copy.deepcopy(template)
        learner.base = float(chain["initial_carma"])
        if hasattr(learner, "integer"):
            learner.integer = bool(chain["integer"])
        learner.load_state(state_file)
        learner.frozen = True
        learner.start_day(last)
        return learner

    def mutant_learner(learning: bool):
        if chain["mutant_start"] == "fresh" and learning:
            learner = copy.deepcopy(config.get_experiment(chain["mutant_config"])
                                    .bidder_types()[0].strategy)
            learner.base = float(chain["initial_carma"])
            if hasattr(learner, "integer"):
                learner.integer = bool(chain["integer"])
        else:
            learner = population_learner()
        learner.frozen = not learning
        return learner

    rows = []
    for arm in ("learn", "frozen"):
        mutant = mutant_learner(arm == "learn")
        cfg = mutant_experiment(base, population_learner(), mutant, chain["mutant_share"])
        balances = read_balances(out, f"{cid}_day{last}")
        for d in range(1, chain["mutant_days"] + 1):
            label = f"{cid}_mut_{arm}_day{d}"
            mutant_state = out / f"{label}_mutant.json"
            if (out / f"{label}_run.csv").exists() and mutant_state.exists():
                mutant.load_state(str(mutant_state))
            else:
                # Fresh mutants follow their own schedule from day 1; warm ones stay at the floor.
                fresh = chain["mutant_start"] == "fresh" and arm == "learn"
                mutant.start_day(d if fresh else last + d)
                run_day(carma, cfg, chain, sc, out, label,
                        MUTANT_SEED_OFFSET + 1000 * chain["seed"] + d, balances,
                        {"mutant_arm": arm, "mutant_day": d, "after_day": last})
                mutant.save_state(str(mutant_state))
            balances = read_balances(out, label)
            cars = pd.read_csv(out / f"{label}_cars.csv")
            for kind in ("mutant", "population"):
                sub = cars[cars["kind"] == kind]
                rows.append({"arm": arm, "day": d, "kind": kind, "trips": len(sub),
                             "time_lost": sub["time_loss"].mean() if len(sub) else float("nan")})
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{cid}_mutants.csv", index=False)
    # Difference in differences per day: (mutant - population) when the mutants learn, minus the
    # same gap when they play the population's policy, relative to the mutants' frozen time lost.
    wide = table.pivot_table(index="day", columns=["arm", "kind"], values="time_lost")
    did = ((wide[("learn", "mutant")] - wide[("learn", "population")])
           - (wide[("frozen", "mutant")] - wide[("frozen", "population")]))
    pct = 100.0 * did / wide[("frozen", "mutant")]
    direct = 100.0 * (wide[("learn", "mutant")] - wide[("frozen", "mutant")]) \
        / wide[("frozen", "mutant")]

    def interval(series):
        values = series.dropna()
        n = len(values)
        if n < 2:
            return float(values.mean()) if n else float("nan"), float("nan"), float("nan")
        half = t975(n - 1) * values.std(ddof=1) / n ** 0.5
        return float(values.mean()), float(values.mean() - half), float(values.mean() + half)
    gain, low, high = interval(pct)
    direct_gain, direct_low, direct_high = interval(direct)
    summary = {"chain": cid, "scenario": chain["scenario"], "experiment": chain["experiment"],
               "seed": chain["seed"], "after_day": last, "mutant_start": chain["mutant_start"],
               "mutant_config": chain["mutant_config"] if chain["mutant_start"] == "fresh"
               else chain["experiment"],
               "mutant_share": chain["mutant_share"], "days": chain["mutant_days"],
               "mutant_trips_per_day": float(table[(table.kind == "mutant")]["trips"].mean()),
               "gain_pct": gain, "gain_ci": [low, high],
               "direct_gain_pct": direct_gain, "direct_gain_ci": [direct_low, direct_high],
               # Negative = the mutants lose less time than by following the population.
               "exploitable": bool(high < 0 and gain < -2.0)}
    write_json(out / f"{cid}_mutants.json", summary)
    return summary


def main() -> None:
    """Build one chain per (scenario, strategy, seed) and run them in parallel."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--scenarios", nargs="+", default=["single"], choices=list(SCENARIOS))
    parser.add_argument("--configs", nargs="+", default=LEARNERS + BASELINES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-days", type=int, default=60)
    parser.add_argument("--max-days", type=int, default=90)
    parser.add_argument("--extend-to", type=int, default=None,
                        help="raise the day limit of chains that have not converged")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--consecutive", type=int, default=7)
    parser.add_argument("--urgency", type=float, nargs="*", default=[],
                        help="urgency levels, e.g. 1 6 (default: every trip has urgency 1)")
    parser.add_argument("--urgency-probs", type=float, nargs="*", default=[])
    parser.add_argument("--bid-log", choices=["full", "summary"], default="full")
    parser.add_argument("--price-bin", type=float, default=300.0)
    parser.add_argument("--trace-interval", type=float, default=60.0)
    parser.add_argument("--p", type=float, default=None, help="override the scenario's demand")
    parser.add_argument("--drivers", type=int, default=None, help="override the driver pool")
    parser.add_argument("--results-name", default="learning")
    parser.add_argument("--initial-carma", type=float, default=100.0,
                        help="starting balance of every driver (the average balance)")
    parser.add_argument("--carma-levels", type=int, nargs="*", default=None,
                        help="sweep of starting balances, e.g. 10 20 40 80 160 320 640; each "
                             "level goes to results/<results-name>/c<level>/<scenario>")
    parser.add_argument("--integer-carma", action="store_true",
                        help="whole units of carma: whole bids, payments and rebates")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--mutants", action="store_true",
                        help="run the exploitability test on finished learning chains")
    parser.add_argument("--mutant-share", type=float, default=0.1)
    parser.add_argument("--mutant-days", type=int, default=14)
    parser.add_argument("--mutant-start", choices=["warm", "fresh"], default="warm")
    parser.add_argument("--mutant-config", default="learn_br",
                        help="learner of fresh mutants (--mutant-start fresh)")
    args = parser.parse_args()
    if len(args.urgency) != len(args.urgency_probs):
        parser.error("--urgency and --urgency-probs need the same number of values")
    if args.mutants:
        run_mutant_tests(args)
        return

    max_days = args.extend_to if args.extend_to else args.max_days
    chains = []
    for scenario, level, results_name in levels(args):
        sc = SCENARIOS[scenario]
        out_dir(results_name, scenario).mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            for name in args.configs:
                chains.append({
                    "initial_carma": level, "integer": args.integer_carma,
                    "scenario": scenario, "experiment": name, "seed": seed,
                    "p": args.p or sc["p"], "drivers": args.drivers or sc["drivers"],
                    "min_days": args.min_days, "max_days": max_days,
                    "early_stop": not args.no_early_stop, "baseline_days": args.baseline_days,
                    "evaluate": not args.no_evaluate, "eval_days": args.eval_days,
                    "window": args.window, "consecutive": args.consecutive,
                    "urgency": args.urgency, "urgency_probs": args.urgency_probs,
                    "bid_log": "full" if args.bid_log == "full" else "none",
                    "price_bin": args.price_bin, "trace_interval": args.trace_interval,
                    "results_name": results_name})
    # Learners first (they run much longer), bigger networks first.
    order = {"grid5": 0, "grid3": 1, "single": 2}
    chains.sort(key=lambda c: (c["experiment"] not in LEARNERS, order[c["scenario"]]))
    for index, chain in enumerate(chains):
        chain["start_delay"] = (index % args.workers) * 1.0
    print(f"{len(chains)} chains on {args.workers} workers "
          f"(learners: {args.min_days}-{max_days} days).", flush=True)
    began = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_chain, c): c for c in chains}
        for future in as_completed(futures):
            chain = futures[future]
            minutes = (time.time() - began) / 60
            try:
                status = future.result()
                print(f"  done {chain['scenario']} {chain_id(chain)}: last day "
                      f"{status.get('last_day')}, converged={status.get('converged')} "
                      f"(on day {status.get('converged_on')}) ({minutes:.1f} min)", flush=True)
            except Exception as error:      # one failed chain must not stop the others
                print(f"  FAILED {chain['scenario']} {chain_id(chain)}: {error!r}", flush=True)
    print("Done.", flush=True)


def levels(args) -> list[tuple[str, float, str]]:
    """(scenario, starting carma, results name) for every scenario and starting-carma level."""
    out = []
    for scenario in args.scenarios:
        if args.carma_levels:
            for level in args.carma_levels:
                out.append((scenario, float(level), f"{args.results_name}/c{level}"))
        else:
            out.append((scenario, args.initial_carma, args.results_name))
    return out


def run_mutant_tests(args) -> None:
    """Run the exploitability test for every (scenario, learner, seed) asked for."""
    chains = []
    for scenario, level, results_name in levels(args):
        sc = SCENARIOS[scenario]
        for seed in args.seeds:
            for name in args.configs:
                if name not in LEARNERS:
                    continue
                chains.append({
                    "scenario": scenario, "experiment": name, "seed": seed,
                    "initial_carma": level, "integer": args.integer_carma,
                    "p": args.p or sc["p"], "drivers": args.drivers or sc["drivers"],
                    "urgency": args.urgency, "urgency_probs": args.urgency_probs,
                    "trace_interval": args.trace_interval, "results_name": results_name,
                    "mutant_share": args.mutant_share, "mutant_days": args.mutant_days,
                    "mutant_start": args.mutant_start, "mutant_config": args.mutant_config})
    for index, chain in enumerate(chains):
        chain["start_delay"] = (index % args.workers) * 1.0
    print(f"{len(chains)} exploitability tests on {args.workers} workers.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_mutants, c): c for c in chains}
        for future in as_completed(futures):
            chain = futures[future]
            try:
                s = future.result()
                print(f"  {chain['scenario']} {chain_id(chain)}: mutants' time lost "
                      f"{s['gain_pct']:+.1f}% [{s['gain_ci'][0]:+.1f}, {s['gain_ci'][1]:+.1f}] "
                      f"vs following the population (exploitable: {s['exploitable']})",
                      flush=True)
            except Exception as error:      # one failed test must not stop the others
                print(f"  FAILED {chain['scenario']} {chain_id(chain)}: {error!r}", flush=True)


if __name__ == "__main__":
    main()
