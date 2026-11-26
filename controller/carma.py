# Controller for traffic lights using SUMO's TraCI interface.
# Run from the Controller/ directory (paths below are relative to it).

# General imports.
import os
import sys
import datetime
import random
import time

# Repo imports.
from baselines import SUMO_CONTROLS, MaxPressure, activate_programs, program_file
from auction import NonConflictingGreedy, annotate_rivals
from bidders import make_bid
from blocking import build_out_lanes, find_blocked_links
from car import Car
from config import BASELINE, Experiment
from core import AuctionRecord, CarRecord, Context, Grant
from drivers import DriverPool
from history import AuctionHistory
from metrics import finish_car, records_to_frame, summarize
from prices import junction_type
from prices import summarize as summarize_prices
from signals import SignalController

import pandas as pd
import traci

# Make SUMO's own Python tools importable if the SUMO_HOME environment variable is set.
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


# The road network and the vehicle routes (trips) to simulate.
NET = "../network/net/RandomGrid5_200_tls.net.xml"
ROUTES = "../simulation/5_200grid_tls_routes.rou.xml"


def controllable_tls():
    """Skip junctions whose only phase is all-green (nothing to arbitrate)."""
    # Ids of the intersections we will control.
    ids = []
    # Look at every traffic light in the network.
    for tls in traci.trafficlight.getIDList():
        # The phases of its built-in program, e.g. "GggrrrGGg", "yyyrrrGyy", ...
        phases = traci.trafficlight.getAllProgramLogics(tls)[0].getPhases()
        # Assume the junction is all-green until we find a phase that shows a red.
        has_red = False
        for p in phases:
            if "r" in p.state.lower():
                has_red = True
        # If any phase ever shows a red, the junction has conflicting movements that need
        # arbitrating. All-green junctions (e.g. state "GG") have nothing to decide.
        if has_red:
            ids.append(tls)
    return ids


def build_context(t, cars, controlled, bid_range, blocked, history=None):
    """Snapshot of the world for this step: which cars are close enough to each controlled
    intersection to bid.
    """
    # Start with an empty list of bidders for each controlled intersection.
    approaching = {}
    for tls in controlled:
        approaching[tls] = []
    # Check every car in the simulation.
    for car in cars.values():
        # A car may bid at an intersection only if (1) its next traffic light is one we control
        # and (2) it is within `bid_range` metres of the stop line.
        if car.tls in approaching and car.dist is not None and car.dist <= bid_range:
            approaching[car.tls].append(car)
    # Package the time and the approach lists into one read-only snapshot for this step.
    return Context(t=t, approaching=approaching, blocked=blocked, history=history)


def network_row(t, cars, arrived):
    """The state of the network at time t (one row of *_network_trace.csv): cars driving, the
    share of them stopped, their mean speed and mean waiting time accumulated so far on their
    trip, and how many trips have finished since the start of the run."""
    n = len(cars)
    if n == 0:
        return {"t": t, "in_network": 0, "stopped_share": 0.0, "mean_speed": 0.0,
                "mean_acc_wait": 0.0, "arrived": arrived}
    stopped = sum(1 for car in cars.values() if car.speed < 0.1)
    return {"t": t, "in_network": n, "stopped_share": stopped / n,
            "mean_speed": sum(car.speed for car in cars.values()) / n,
            "mean_acc_wait": sum(car.acc_wait for car in cars.values()) / n,
            "arrived": arrived}


def strategy_hooks(cfg, name):
    """The distinct strategy objects of an experiment that have a method `name` (learners that
    want to see auctions, trip ends or day ends; see learner_base.py)."""
    found = []
    seen = set()
    for btype in cfg.bidder_types():
        strategy = btype.strategy
        if id(strategy) not in seen and hasattr(strategy, name):
            seen.add(id(strategy))
            found.append(strategy)
    return found


def junction_types(controlled):
    """{tls: "4-way" | "T" | "corner" | ...}: how many roads come into each junction."""
    types = {}
    for tls in controlled:
        edges = set()
        for lane in traci.trafficlight.getControlledLanes(tls):
            edges.add(lane.rsplit("_", 1)[0])
        types[tls] = junction_type(len(edges))
    return types


def step_intersection(tls, ctx, cfg, cars, grants, signals, log, observers=()):
    """One step at one intersection: drop finished grants; if none remain hold an auction (group,
    bid, pick winners, settle payments, record history); activate grants whose front car is near;
    and send the resulting green set to the signal controller.

    observers are strategies with an observe_auction method; each sees every finished auction
    (all bids, the winners and the carma that moved), like a public price board.
    """
    # Drop finished grants. While any grant (pending or active) is still around, no new
    # auction runs at this intersection.
    # Signal indices at this intersection whose outgoing lane is currently full.
    blocked = ctx.blocked.get(tls, set())
    still_active = []
    for g in grants[tls]:
        # Skip a grant whose links are ALL blocked: its cars couldn't leave the intersection,
        # so holding green for it would waste the green.
        all_blocked = len(g.bid.links) > 0
        for link in g.bid.links:
            if link not in blocked:
                all_blocked = False
        # Keep the grant only if its group hasn't finished crossing (and it hasn't timed out)
        # and it isn't stuck behind a full lane.
        if not g.done(ctx) and not all_blocked:
            still_active.append(g)
    grants[tls] = still_active

    # No active grant means the intersection is free: hold a new auction.
    if not grants[tls]:
        # Only cars whose link leads into a lane with room are allowed to bid.
        bidders = []
        for car in ctx.approaching[tls]:
            if car.link not in blocked:
                bidders.append(car)
        # Sort the bidding cars by bidder type, because each type groups, bids and aggregates in
        # its own way.
        cars_by_kind = {}
        for car in bidders:
            if car.kind not in cars_by_kind:
                cars_by_kind[car.kind] = []
            cars_by_kind[car.kind].append(car)
        # For each type: split its cars into groups (one car each, a lane each, ...), then turn
        # each group into a Bid (ask the strategy for each car's bid, then aggregate). All the
        # types' bids go into the same auction.
        bids = []
        if cfg.group_across_types:
            # Mixed population with group bidding: the cars of a lane (or phase) form one group
            # whatever their types. With front-car bidding the car at the head of the queue bids
            # by its own type's rule; with pooled bidding (front_only off) every member bids by
            # its own type's rule and the bids are summed. The bid is recorded under the front
            # car's type.
            types = {t.name: t for t in cfg.bidder_types()}
            rules = {name: t.strategy for name, t in types.items()}
            for group in cfg.bidder_types()[0].grouper(bidders):
                front = min(group, key=lambda c: c.dist if c.dist is not None else float("inf"))
                btype = types[front.kind]
                bids.append(make_bid(group, tls, btype.strategy, btype.aggregate, ctx, btype.name,
                                     btype.front_only, cfg.integer_carma,
                                     None if btype.front_only else rules))
        else:
            for btype in cfg.bidder_types():
                typed_cars = cars_by_kind.get(btype.name, [])
                for group in btype.grouper(typed_cars):
                    bids.append(make_bid(group, tls, btype.strategy, btype.aggregate, ctx,
                                         btype.name, btype.front_only, cfg.integer_carma))
        # The auction picks the winning bid(s). With no bids there are no winners.
        winners = cfg.auction.run(bids)
        # What each bid had to beat (bid.rival / bid.contested), read by the payment rule, the
        # bid log and the learners. A multi-winner auction compares only conflicting bids.
        annotate_rivals(bids, winners, cfg.auction.conflicts
                        if isinstance(cfg.auction, NonConflictingGreedy) else None)

        if winners:
            # Ask the payment rule how each car's balance changes and apply it. This is the
            # only place carma balances are modified.
            deltas = cfg.payment.settle(winners, bids)
            for vid, delta in deltas.items():
                cars[vid].carma += delta          # the only place balances change
            # Add this auction to the bidding history (every bid, who won, what moved), so later
            # auctions' strategies can learn from it. It is stored AFTER the bids were made, so a
            # strategy never sees the auction it is bidding in.
            if ctx.history is not None:
                ctx.history.record(ctx.t, tls, bids, winners, deltas)
            # Count the loss for every car that chose a losing bid (Car.lost_here), after the
            # history has stored the count the car bid with.
            for bid in bids:
                if bid not in winners:
                    for vid in bid.deciders():
                        car = cars[vid]
                        car.lost_at[tls] = car.lost_at.get(tls, 0) + 1
            # Learners that watch every auction (the public price board).
            for observer in observers:
                observer.observe_auction(ctx.t, tls, bids, winners, deltas, cars)
            # Each winner gets a Grant: it remembers who was in the group, so green is held
            # until they have all crossed (or max_green seconds pass).
            new_grants = []
            # For the log: the member vehicle ids and the bid amount of each winner.
            winner_vids = []
            winner_amounts = []
            winner_kinds = []
            for w in winners:
                # Count this win for every car in the winning group (shown in the results).
                for c in w.members:
                    c.wins += 1
                winner_kinds.append(w.kind)
                # The ids of every car in the winning group (a set for the grant, a list for the log).
                member_set = set()
                member_list = []
                # The lanes the group is queued on (used by the lane-front fix).
                lane_set = set()
                for c in w.members:
                    member_set.add(c.vid)
                    member_list.append(c.vid)
                    if c.lane:
                        lane_set.add(c.lane)
                new_grants.append(Grant(w, member_set, ctx.t, cfg.max_green, lanes=lane_set,
                                        min_duration=cfg.min_green,
                                        release_fraction=cfg.release_fraction,
                                        extend=cfg.extend_green))
                winner_vids.append(member_list)
                winner_amounts.append(w.amount)
            grants[tls] = new_grants
            # Record the auction for later analysis.
            log.append(AuctionRecord(
                t=ctx.t, tls=tls, n_bids=len(bids),
                winners=winner_vids,
                amounts=winner_amounts,
                kinds=winner_kinds,
            ))

    # Turn on any pending grant whose front car is now within activation_dist of the stop line
    # (this includes a grant that was just won this step).
    for g in grants[tls]:
        g.try_activate(ctx, cfg.activation_dist, cfg.serve_lane_front)

    # The green set is every signal index wanted by an ACTIVATED grant: its bid's links plus,
    # with the lane-front fix, the link of the car currently at the head of each served lane.
    # Pending grants don't count yet. With none activated it is empty, and the signal
    # controller then keeps the last green showing (never all red).
    green = set()
    for g in grants[tls]:
        if g.active:
            green.update(g.bid.links)
    # Lane-front fix: also serve the head car of each served lane. When the experiment knows
    # which movements conflict (cfg.conflicts), a head car whose movement would clash with a
    # link that is already green keeps waiting instead.
    if cfg.serve_lane_front:
        for g in grants[tls]:
            if g.active:
                for link in g.front_links(ctx, blocked):
                    clash = False
                    if cfg.conflicts is not None:
                        for other in green:
                            if cfg.conflicts(tls, link, other):
                                clash = True
                    if not clash:
                        green.add(link)
    # Hand the desired green set to the signal controller, which handles yellow/all-red timing
    # and sends the state string to SUMO.
    signals.apply(tls, green, ctx.t)


def pick_bidder_type(types, rng):
    """Choose a bidder type at random in proportion to the types' share."""
    # Choose a bidder type at random, in proportion to each type's `share`.
    weights = []
    for btype in types:
        weights.append(btype.share)
    return rng.choices(types, weights=weights)[0]


def fixed_urgencies(driver_ids: list, levels: tuple, probs: tuple, seed: int) -> dict:
    """Per-driver urgency: driver id -> one of `levels`, with exactly round(share * n) drivers at
    each level (the last level takes the remainder). Who gets which level is a shuffle seeded
    by `seed`, so the same pool seed gives the same assignment every day."""
    order = list(driver_ids)
    random.Random(f"urgency-{seed}").shuffle(order)
    total = float(sum(probs))
    counts = [round(len(order) * p / total) for p in probs[:-1]]
    counts.append(len(order) - sum(counts))
    out, start = {}, 0
    for level, count in zip(levels, counts):
        for driver_id in order[start:start + count]:
            out[driver_id] = level
        start += count
    return out


def run(cfg: Experiment = BASELINE, gui: bool = True, delay_ms: int = 100, seed: int = 0,
        routes: str = ROUTES, net: str = NET, max_time: float | None = None, label: str | None = None,
        out_dir: str = "../results", save_history: bool = True, quiet: bool = False,
        extra_info: dict | None = None, jam_stop: bool = False, jam_share: float = 0.9,
        jam_seconds: float = 300, jam_min_cars: int = 100,
        start_balances: dict | None = None, pool_seed: int | None = None,
        trip_drivers: dict | None = None) -> dict:
    """Run one simulation and save its results. Returns a dict describing the run.

    routes        route file to simulate (default: the one in ROUTES).
    net           network file (default: the 5x5 grid in NET).
    max_time      stop at this simulation time (seconds) even if cars are still driving. Cars
                  still in the network then are not in the per-car results. None = run until
                  every car has finished.
    label         file name for the outputs (<out_dir>/<label>*.csv). Default: the experiment
                  name plus the date and time.
    out_dir       folder the results are written to.
    save_history  also save the full bid history (large for long or busy runs).
    quiet         hide SUMO's step log and warnings (useful when running many in parallel).
    extra_info    extra columns to add to the run summary row (e.g. the sweep value).
    jam_*         a run counts as jammed when at least jam_min_cars cars are in the network
                  and a share jam_share or more of them are stopped, for jam_seconds in a row.
                  The time it happens is reported as jam_time. With jam_stop the run ends
                  there (a gridlocked run is not worth continuing).

    trip_drivers  vehicle id -> the driver who owns that trip (commuters, who drive the same
                  trips every day); None hands each trip to a random idle driver.

    If cfg.control is "default", the network's own traffic light programs run untouched and
    nothing bids. This is the control group for judging whether carma helps. "actuated" and
    "delay_based" run SUMO's actuated controllers over the same phases, and "max_pressure" runs
    max-pressure control from Python (see baselines.py); nothing bids in those either.
    """
    if cfg.control not in ("carma", "default", "max_pressure") + SUMO_CONTROLS:
        raise ValueError(f"Unknown control {cfg.control!r}")
    # When the run started, to report how long it took.
    started = time.time()
    # Seed Python's random numbers (auction tie-breaks, random bids) so a run can be repeated.
    random.seed(seed)
    # A separate random stream for assigning bidder types, so the assignment depends only on the
    # seed and the order vehicles depart in.
    type_rng = random.Random(seed)
    # Urgency draws use their own generator, so the same vehicle gets the same urgency whatever
    # strategy is being run (departures happen in the same order).
    urgency_rng = random.Random(seed + 7919)
    # Launch SUMO (with or without the GUI) and connect to it through TraCI.
    # --start begins the simulation immediately; --delay slows GUI playback (ms per step).
    if gui:
        sumo_binary = "sumo-gui"
    else:
        sumo_binary = "sumo"
    sumo_command = [sumo_binary, "-n", net, "-r", routes,
                    "--start", "--delay", str(delay_ms), "--seed", str(seed)]
    if quiet:
        sumo_command += ["--no-step-log", "--no-warnings"]
    # SUMO's actuated controllers come as an extra program for every junction.
    additional = []
    # Driver settings (a redefined default car type) must load before the routes; SUMO reads
    # additional files first.
    if cfg.driver_file:
        additional.append(cfg.driver_file)
    if cfg.control in SUMO_CONTROLS:
        additional.append(program_file(net, cfg.control))
    if additional:
        sumo_command += ["-a", ",".join(additional)]
    traci.start(sumo_command)
    # Filled in during the run and returned at the end.
    run_info = {}
    # try/finally guarantees the SUMO connection is closed even if the loop crashes.
    try:
        # Without carma nothing bids: SUMO runs its own signal programs (fixed or actuated), or
        # max-pressure picks the phases. Cars then only need the cheap per-step update.
        default_lights = cfg.control != "carma"
        pressure = None
        if default_lights:
            controlled = []
            signals = None
            out_lanes = {}
            if cfg.control in SUMO_CONTROLS:
                activate_programs(cfg.control)
            elif cfg.control == "max_pressure":
                pressure = MaxPressure(controllable_tls(), min_green=cfg.min_green,
                                       yellow=cfg.yellow)
        else:
            # Only the intersections that actually have something to arbitrate.
            controlled = controllable_tls()
            # Object that translates "these links are green" into safe signal state strings.
            signals = SignalController(controlled, yellow=cfg.yellow, all_red=cfg.all_red,
                                       yield_table=cfg.yield_table)
            # For every signal link, the lane it leads into (used to detect full outgoing lanes).
            out_lanes = build_out_lanes(controlled)
        # Only intersections that hold auctions count as decisions left on a route.
        Car.decision_tls = frozenset(controlled)
        # Learners that watch every auction, every trip end and the end of the day.
        observers = strategy_hooks(cfg, "observe_auction")
        trip_watchers = strategy_hooks(cfg, "on_trip_end")
        day_watchers = strategy_hooks(cfg, "on_day_end")
        # Decision log and price summary (see Experiment.bid_log / price_bin).
        keep_decisions = cfg.bid_log != "none" or cfg.price_bin is not None
        jtypes = junction_types(controlled) if keep_decisions else {}
        # All vehicles currently in the simulation, keyed by SUMO vehicle id.
        cars: dict[str, Car] = {}
        # Currently active (won) grants at each intersection. Empty list = intersection free.
        grants = {}
        for tls in controlled:
            grants[tls] = []
        # One record per auction, written to CSV at the end.
        log: list[AuctionRecord] = []
        # One record per finished trip (per car), with its bidder type and trip results.
        car_records: list[CarRecord] = []
        # The bidder types cars can be assigned (just one unless the experiment mixes types).
        types = cfg.bidder_types()
        # Persistent carma: a fixed population of drivers whose balance carries across trips.
        pool = None
        # Per-driver urgency (cfg.urgency_mode == "driver"): driver id -> urgency.
        driver_urgency = None
        carma_trace = []
        balance_trace = []
        # State of the network at every checkpoint (every run, persistent drivers or not).
        network_trace = []
        next_trace = 0.0
        if cfg.persistent_drivers:
            # pool_seed fixes who the drivers are (their bidder types) across days; the day's
            # seed then only decides departures. start_balances carries balances over from the
            # previous day (driver id -> carma).
            pool = DriverPool(cfg.persistent_drivers, types, cfg.initial_carma,
                              random.Random(pool_seed) if pool_seed is not None else type_rng,
                              exact=cfg.exact_type_shares)
            pool.rng = type_rng
            if cfg.urgency_mode == "driver" and cfg.urgency_levels:
                driver_urgency = fixed_urgencies(sorted(pool.drivers, key=lambda d: int(d[1:])),
                                                 cfg.urgency_levels, cfg.urgency_probs,
                                                 pool_seed if pool_seed is not None else seed)
            if start_balances:
                for driver_id, balance in start_balances.items():
                    pool.drivers[driver_id].carma = balance
                    pool.drivers[driver_id].start_carma = balance
        # Past auctions at every intersection, for strategies that learn from earlier outcomes.
        history = AuctionHistory(cfg.history_length, keep_decisions=keep_decisions)
        # How many vehicles SUMO had to teleport out of a jam (a sign of gridlock).
        teleports = 0
        t = 0.0
        # Jam detection: how many seconds in a row the network has been (nearly) all stopped,
        # when that first reached jam_seconds, the running total of the stopped share (for its
        # average) and the most cars in the network at once.
        jam_run = 0
        jam_time = None
        stopped_share_sum = 0.0
        steps = 0
        max_in_network = 0

        # Keep stepping while any vehicle is still running or waiting to depart.
        while traci.simulation.getMinExpectedNumber() > 0:
            # Advance the simulation by one step (one second by default).
            traci.simulationStep()
            # Current simulation time in seconds.
            t = traci.simulation.getTime()
            teleports += traci.simulation.getStartingTeleportNumber()

            # Create a Car for every vehicle that just entered: the starting carma balance and a
            # randomly chosen bidder type.
            for vid in traci.simulation.getDepartedIDList():
                # With persistent carma the vehicle is driven by an idle driver, who brings their
                # own type, balance and bid history. (If every driver is out, it is a guest.)
                driver = None
                if pool is not None:
                    driver = pool.checkout(trip_drivers.get(vid) if trip_drivers else None)
                if driver is not None:
                    car = Car(vid, driver.carma, driver.kind)
                    car.driver = driver.id
                    car.bid_history = driver.bid_history
                    if driver_urgency is not None:
                        car.urgency = driver_urgency[driver.id]
                    elif cfg.urgency_levels:
                        car.urgency = urgency_rng.choices(cfg.urgency_levels, weights=cfg.urgency_probs)[0]
                    cars[vid] = car
                    continue
                btype = pick_bidder_type(types, type_rng)
                # Start with the type's own balance if it has one, otherwise the experiment's.
                if btype.initial_carma is not None:
                    start_carma = btype.initial_carma
                else:
                    start_carma = cfg.initial_carma
                cars[vid] = Car(vid, start_carma, btype.name)
                if cfg.urgency_levels:
                    cars[vid].urgency = urgency_rng.choices(cfg.urgency_levels, weights=cfg.urgency_probs)[0]
            # Remove vehicles that just reached their destination, saving their trip results
            # first. pop(..., None) is safe even if the car is already gone.
            for vid in traci.simulation.getArrivedIDList():
                finished = cars.pop(vid, None)
                if finished is not None:
                    for watcher in trip_watchers:
                        watcher.on_trip_end(finished, t)
                    # Give the trip's balance and results back to the driver.
                    if pool is not None and finished.driver is not None:
                        pool.checkin(pool.drivers[finished.driver], finished)
                    car_records.append(finish_car(finished, t))
            # Refresh every remaining car's position, speed, waiting time and next traffic light.
            for car in cars.values():
                if default_lights:
                    car.update_light()
                else:
                    car.update()

            # Every trace_interval seconds record the state of the network (every run) and how
            # carma is spread over the drivers (persistent runs).
            if t >= next_trace:
                network_trace.append(network_row(t, cars, len(car_records)))
                if pool is not None:
                    carma_trace.extend(pool.snapshot(t, cars))
                    balance_trace.extend(pool.balance_rows(t, cars))
                next_trace = t + cfg.trace_interval

            # Work out which cars are close enough to each intersection to bid this step.
            # Find the links whose outgoing lane is full, so they can be skipped this step.
            blocked = find_blocked_links(controlled, out_lanes, cars, cfg.blocked_entrance_dist)
            ctx = build_context(t, cars, controlled, cfg.bid_range, blocked, history)
            # Run the grant / auction / signal logic once for every intersection.
            for tls in controlled:
                step_intersection(tls, ctx, cfg, cars, grants, signals, log, observers)
            if pressure is not None:
                pressure.step(t)

            # Count the grants that are active across all intersections.
            active_grants = 0
            for tls_grants in grants.values():
                active_grants += len(tls_grants)

            # Jam check: what share of the cars in the network are stopped right now?
            stopped = 0
            for car in cars.values():
                if car.speed < 0.1:
                    stopped += 1
            stopped_share = 0.0
            if cars:
                stopped_share = stopped / len(cars)
            stopped_share_sum += stopped_share
            steps += 1
            if len(cars) > max_in_network:
                max_in_network = len(cars)
            if len(cars) >= jam_min_cars and stopped_share >= jam_share:
                jam_run += 1
            else:
                jam_run = 0
            if jam_run >= jam_seconds and jam_time is None:
                jam_time = t
                if jam_stop:
                    break

            # Stop early if a time limit was given (the cars still driving are not recorded).
            if max_time is not None and t >= max_time:
                break

        # How the run ended. "not_in_network" counts vehicles that had not entered the network
        # yet (waiting to be inserted, because the roads were full, or not due to depart).
        run_info = {
            "end_time": t,
            "cars_arrived": len(car_records),
            "cars_in_network": len(cars),
            "not_in_network": traci.simulation.getMinExpectedNumber() - len(cars),
            "teleports": teleports,
            "auctions": len(log),
            "jammed": jam_time is not None,
            "jam_time": jam_time,
            "mean_stopped_share": stopped_share_sum / max(1, steps),
            "max_in_network": max_in_network,
            "wall_seconds": time.time() - started,
            "guest_trips": pool.guest_trips if pool is not None else 0,
            "swapped_trips": pool.swapped_trips if pool is not None else 0,
        }
        # Learners close the day (trips still running are cut here).
        for watcher in day_watchers:
            watcher.on_day_end(t)
        # Final network state and carma picture (before the SUMO connection closes the cars are
        # still known).
        network_trace.append(network_row(t, cars, len(car_records)))
        if pool is not None:
            carma_trace.extend(pool.snapshot(t, cars))
            balance_trace.extend(pool.balance_rows(t, cars))
            driver_table = pool.to_frame(cars)
    finally:
        # Always disconnect from SUMO, even after an error.
        traci.close()

    # Save the auction log: one row per auction, in <out_dir>/<name>.csv.
    os.makedirs(out_dir, exist_ok=True)
    # Turn each AuctionRecord into a plain dict (one per CSV row).
    rows = []
    for record in log:
        rows.append(record.__dict__)
    if label is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = f"{cfg.name}_{current_time}"
    base_name = f"{out_dir}/{label}"
    pd.DataFrame(rows).to_csv(f"{base_name}.csv", index=False)

    # Save one row per car, and the average results per bidder type, so the types can be compared.
    car_table = records_to_frame(car_records)
    car_table.to_csv(f"{base_name}_cars.csv", index=False)
    # The network state at every checkpoint (every trace_interval seconds).
    pd.DataFrame(network_trace).to_csv(f"{base_name}_network_trace.csv", index=False)
    # Save every bid in every auction (one row per bid), for analysing bidding behaviour.
    if pool is not None:
        pd.DataFrame(carma_trace).to_csv(f"{base_name}_carma_trace.csv", index=False)
        pd.DataFrame(balance_trace).to_csv(f"{base_name}_balance_trace.csv", index=False)
        driver_table.to_csv(f"{base_name}_drivers.csv", index=False)
    # Strategies that learn (see regret.py) can save what they learned.
    for btype in cfg.bidder_types():
        if hasattr(btype.strategy, "report"):
            btype.strategy.report(f"{base_name}_policy.csv")
    if save_history:
        history.to_frame().to_csv(f"{base_name}_history.csv", index=False)
    # Every decision (bid log) and the per-intersection price summary.
    if keep_decisions:
        decisions = history.decisions_frame()
        if not decisions.empty:
            decisions["x"] = decisions["balance"] / cfg.initial_carma
            decisions["bid_rel"] = decisions["bid"] / cfg.initial_carma
        if cfg.bid_log == "full":
            decisions.to_csv(f"{base_name}_bids.csv.gz", index=False, compression="gzip")
        if cfg.price_bin is not None:
            summarize_prices(decisions, cfg.price_bin, cfg.initial_carma, jtypes).to_csv(
                f"{base_name}_prices.csv", index=False)
    summary = summarize(car_table)
    summary.to_csv(f"{base_name}_summary.csv")
    # Save a one-row description of the run (how it ended, plus any extra columns given).
    run_info["label"] = label
    run_info["experiment"] = cfg.name
    run_info["seed"] = seed
    if extra_info:
        run_info.update(extra_info)
    pd.DataFrame([run_info]).to_csv(f"{base_name}_run.csv", index=False)
    if not quiet:
        print(f"\nResults by bidder type ({cfg.name}):")
        print(summary.to_string())
    return run_info


# Only start a run when this file is executed directly (not when it is imported).
if __name__ == "__main__":
    run()
