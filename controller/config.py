"""An Experiment is a bundle of choices, one per layer. A new bidding structure is a new
Experiment, not a new simulation loop."""
from dataclasses import dataclass, replace
from typing import Callable

from auction import Auction, FirstPrice, HighestBid, NoPayment, PaymentRule, SecondPrice
from best_response import BestResponseLearner
from bidders import (Aggregator, BidderType, Grouper, group_by_compatible, group_by_lane,
                     group_by_link, group_by_phase, group_by_vehicle, mean)
from conflicts import network_conflicts
from qlearning import QLearner
from regret import RegretHedge
from strategies import (BidStrategy, BudgetedUrgency, FixedFraction, OmniscientUrgency, RandomFraction,
                        RecentPriceFollower, UrgencyFraction, UrgencyScaledWait, WaitTimeUrgency)


@dataclass
class Experiment:
    """One experiment: the auction, payment rule, bidder types and every setting that defines a run.
    """
    # Label used in the output file names.
    name: str
    # Who wins (single highest bid, several non-conflicting winners, ...). Shared by all bidders.
    auction: Auction
    # Who pays what, and whether payments are redistributed. Shared by all bidders.
    payment: PaymentRule
    # Bidding behaviour, in one of two ways:
    #  (a) every car bids the same way: give grouper, strategy and aggregate; or
    #  (b) a mix of behaviours in the same simulation: give `population`, a list of BidderType.
    #      Each departing car is assigned a type at random, in proportion to the types' `share`.
    # Who bids together (vehicle, lane, ...).
    grouper: Grouper | None = None
    # How much each car bids.
    strategy: BidStrategy | None = None
    # How member bids combine into the group's bid (sum, max, mean).
    aggregate: Aggregator | None = None
    # A mix of bidder types competing in the same simulation (overrides the three fields above).
    population: list[BidderType] | None = None
    # Metres from the stop line within which a car is allowed to bid.
    bid_range: float = 50.0       # m from the stop line within which cars may bid
    # Lane-front fix: while a grant is green, also give green to the link of the car at the head
    # of each lane the grant serves. Stops a head car that needs a different movement from
    # blocking the winners behind it. Also makes activation_dist measure the lane's head car.
    serve_lane_front: bool = True
    # A link counts as blocked when a stopped car sits within this many metres of the start of
    # the lane it leads into (a queue has spilled back). Blocked links are skipped: their cars
    # don't bid and their grants are dropped. None turns the check off.
    blocked_entrance_dist: float | None = 10.0
    # A won grant only turns green once its front car (the winning group's member closest to
    # the stop line) is within this many metres of the intersection. Until then the grant is
    # pending and the intersection keeps its current green. Set it >= bid_range to activate
    # grants immediately when they are won.
    activation_dist: float = 50.0
    # Longest a winning group may hold green, in seconds.
    max_green: float = 30.0       # s cap on a Grant
    # Seconds of yellow before a link turns red.
    yellow: float = 3.0
    # Seconds of all-red between one green and the next. Keep at 0 so there is always a
    # green or yellow showing at every intersection (the yellow already provides clearance).
    all_red: float = 0.0
    # How many of the most recent auctions at each intersection strategies can read through
    # ctx.history (None = all of them). The full history is always saved to the results.
    history_length: int | None = 100
    # Who controls the traffic lights: "carma" (the auctions in this project), or one of the
    # baselines in which nothing bids: "default" (SUMO's own fixed programs from the network
    # file; also used for stop signs), "actuated" / "delay_based" (SUMO's actuated controllers
    # over the same phases) or "max_pressure" (max-pressure control; see baselines.py).
    control: str = "carma"
    # Carma balance every car starts with. A BidderType can override this with its own
    # `initial_carma`, so types in a mixed population can start with different balances.
    initial_carma: float = 100.0
    # Optional test for "can these two signal indices at an intersection be green together?",
    # as conflicts(tls, a, b) -> bool (see conflicts.py). When given, the lane-front fix never
    # turns on a link that clashes with one already green. Needed whenever one grant can hold
    # several movements (paired auctions). None = old behaviour.
    conflicts: Callable[[str, int, int], bool] | None = None
    # Optional conflicts.ConflictTable used by the signal controller to show permissive
    # movements as 'g' (green, yield to foes) when a foe is green too. None = every green is 'G'.
    yield_table: object | None = None
    # Grant timing (see core.Grant): fewest seconds of green once active, share of the winning
    # group that must cross before the grant can end, and whether green is extended while cars
    # are still queued on the served lanes. With control="max_pressure", min_green is the
    # shortest time a phase stays green before the controller may switch.
    min_green: float = 0.0
    release_fraction: float = 1.0
    extend_green: bool = False
    # Mixed populations with front-car bidding: group the cars of all types together (with the
    # first type's grouper), and let the car at the head of each group bid by its own type's
    # rule. False = each type groups only its own cars, so a lane holding two types makes two
    # bids for the same green.
    group_across_types: bool = False
    # Persistent drivers only: give the types exactly their shares of the drivers (as closely as
    # whole drivers allow) instead of drawing each driver's type at random.
    exact_type_shares: bool = False
    # Persistent carma: the number of drivers in a fixed population whose balance carries over from
    # trip to trip (see drivers.py). Each departing vehicle is handed a random idle driver, who
    # keeps its bidder type. None = every vehicle starts its trip with a fresh balance.
    persistent_drivers: int | None = None
    # Network file to simulate, relative to controller/ (None = the run() default, the 5x5 grid).
    # Used by the stop-sign baseline, which runs on a copy of the network with no traffic lights.
    net: str | None = None
    # Urgency (value of time), drawn for every trip when the car departs: urgency_levels[i] with
    # probability urgency_probs[i] (independent per trip, like CARMA section 5.2). Empty = no urgency
    # (every trip has urgency 1). Only the car's strategy sees it; the auction sees only bids.
    urgency_levels: tuple = ()
    urgency_probs: tuple = ()
    # "trip": a new urgency is drawn for every trip (above). "driver": with persistent drivers,
    # each driver keeps one urgency for good, and the shares of drivers at each level are exactly
    # urgency_probs (rounded; who gets which is fixed by the pool seed, so it is the same every
    # day). Guest trips (no idle driver) still draw per trip.
    urgency_mode: str = "trip"
    # Optional SUMO additional file with driver settings (e.g. a redefined DEFAULT_VEHTYPE with
    # lane-change parameters), loaded for every strategy. None = SUMO's default drivers.
    driver_file: str | None = None
    # Checkpoint interval in simulated seconds: the network state (every run, *_network_trace.csv)
    # and, with persistent drivers, the carma distribution and every driver's balance.
    trace_interval: float = 600.0
    # Bid log: "full" saves every decision (one row per car that chose a bid, with its state and
    # the auction outcome) as <label>_bids.csv.gz; "none" saves nothing extra.
    bid_log: str = "none"
    # Per-intersection price summary (<label>_prices.csv, see prices.py) in time bins of this many
    # seconds, plus a whole-day row per intersection. None = not written.
    price_bin: float | None = None
    # Whole units of carma: every bid is rounded down to a whole number (bidders.make_bid). Use it
    # with a payment rule whose `integer` is set (auction.FirstPrice), so that payments and rebates
    # are whole too; learning_study.py --integer-carma sets both.
    integer_carma: bool = False

    def __post_init__(self):
        """Build the list of bidder types from `population` (or from grouper/strategy/aggregate) and
        check the names are unique.
        """
        # Work out the list of bidder types in use: the population if given, otherwise a single
        # type built from grouper / strategy / aggregate.
        if self.population:
            names = []
            for btype in self.population:
                if btype.name in names:
                    raise ValueError(f"Duplicate bidder type name: {btype.name}")
                names.append(btype.name)
            self._types = list(self.population)
        elif (self.grouper is not None and self.strategy is not None
              and self.aggregate is not None):
            self._types = [BidderType("default", self.grouper, self.strategy, self.aggregate)]
        else:
            raise ValueError("Give either `population`, or all of grouper, strategy and aggregate.")

    def bidder_types(self) -> list[BidderType]:
        """The bidder types cars can be assigned (a single "default" type unless a population
        was given)."""
        return self._types


# Baseline: every vehicle bids alone, highest random bid wins, winner pays its bid.
BASELINE = Experiment(
    name="vehicle_first_price",
    grouper=group_by_vehicle,
    strategy=RandomFraction(),
    aggregate=sum,
    auction=HighestBid(),
    payment=FirstPrice(),
)

# Lane groups bid together; the group bid is the sum of waiting-time-based member bids.
# Payments go back to the losing cars, so total carma stays constant.
LANE_SUM = Experiment(
    name="lane_sum_first_price",
    grouper=group_by_lane,
    strategy=WaitTimeUrgency(k=1.0),
    aggregate=sum,
    auction=HighestBid(),
    payment=FirstPrice(redistribute=True, free_if_uncontested=True),
)

# Link groups: all cars on a lane that need the same movement (e.g. "straight") bid together,
# and a win gives green to exactly that one link.
LINK_SUM = Experiment(
    name="link_sum_first_price",
    grouper=group_by_link,
    strategy=WaitTimeUrgency(k=1.0),
    aggregate=sum,
    auction=HighestBid(),
    payment=FirstPrice(redistribute=True, free_if_uncontested=True),
)

# Same, but the group bid is the mean.
LINK_MEAN = Experiment(
    name="link_mean_second_price",
    grouper=group_by_link,
    strategy=WaitTimeUrgency(k=1.0),
    aggregate=mean,
    auction=HighestBid(),
    payment=SecondPrice(redistribute=True),
)

# Three bidding behaviours competing in the same simulation, in equal shares. Compare them in
# the per-type summary written at the end of the run.
MIXED = Experiment(
    name="mixed",
    population=[
        # Each car bids alone, a random amount.
        BidderType("vehicle_random", group_by_vehicle, RandomFraction(), sum),
        # Cars in a lane bid together, urgency grows with waiting time.
        BidderType("lane_urgency", group_by_lane, WaitTimeUrgency(k=1.0), sum),
        # Cars needing the same movement bid together, a fixed 10% of their balance each.
        BidderType("link_fixed", group_by_link, FixedFraction(0.1), sum),
    ],
    auction=HighestBid(),
    payment=FirstPrice(redistribute=True, free_if_uncontested=True),
)

# Same, but the group bid is the mean (stops big groups winning just because they're big),
# and the winner pays the second-highest bid instead of its own.
LANE_MEAN = Experiment(
    name="lane_mean_second_price",
    grouper=group_by_lane,
    strategy=WaitTimeUrgency(k=1.0),
    aggregate=mean,
    auction=HighestBid(),
    payment=SecondPrice(redistribute=True),
)


# ---- Experiments for the "does carma beat the default lights, and when?" study ----------------

def only(btype: BidderType) -> Experiment:
    """An experiment in which EVERY car bids with one behaviour (no mixing), using the same
    auction and payment rule as MIXED. Comparing several of these shows what each behaviour does
    on its own, without the other types competing with it."""
    return Experiment(
        name=f"only_{btype.name}",
        population=[replace(btype, share=1.0)],
        auction=HighestBid(),
        payment=FirstPrice(redistribute=True, free_if_uncontested=True),
    )


def signal_baseline(name: str, control: str, **settings) -> Experiment:
    """A baseline in which nothing bids and the lights are run by `control` (see
    Experiment.control). The bidder type is a placeholder; the cars are recorded under `name`."""
    return Experiment(
        name=name,
        population=[BidderType(name, group_by_vehicle, FixedFraction(0.0), sum)],
        auction=HighestBid(),
        payment=FirstPrice(),
        control=control,
        **settings,
    )


# The control group: SUMO's own fixed-time programs (90 s cycle).
DEFAULT_LIGHTS = signal_baseline("default_lights", "default")
# Stronger signal baselines over the same phases (baselines.py): SUMO's gap-based and delay-based
# actuated control (green phases 5-50 s), and max-pressure with a 10 s minimum green.
ACTUATED_LIGHTS = signal_baseline("actuated_lights", "actuated")
DELAY_BASED_LIGHTS = signal_baseline("delay_based_lights", "delay_based")
MAX_PRESSURE = signal_baseline("max_pressure", "max_pressure", min_green=10.0)
# Every signal baseline by name, weakest first (the studies run all of them).
SIGNAL_BASELINES = ["default_lights", "actuated_lights", "delay_based_lights", "max_pressure"]

# Stop-sign baseline: the same networks with every traffic light replaced by an all-way stop.
# Nothing is controlled or bid on; SUMO's own right-of-way rules run the junctions.
def stop_signs(net: str, name: str = "stop_signs") -> Experiment:
    """A baseline experiment: the same network with all-way stops; nothing is controlled or bid on.
    """
    return Experiment(
        name=name,
        population=[BidderType(name, group_by_vehicle, FixedFraction(0.0), sum)],
        auction=HighestBid(),
        payment=FirstPrice(),
        control="default",
        net=net,
    )


STOP_SIGNS = stop_signs("../network/net/RandomGrid5_200_stop.net.xml")
STOP_SIGNS_SINGLE = stop_signs("../network/net/single_4way_stop.net.xml", "stop_signs")
STOP_SIGNS_GRID3 = stop_signs("../network/net/grid3_200_stop.net.xml", "stop_signs")

# Paired auctions: cars whose movements don't conflict bid together as one group, and the winning
# pair (or larger set) gets green at the same time. `paired_*` differ in how the cars' bids are
# combined: summed (a bigger set bids more) or averaged (set size doesn't help).
PAIRED_GROUPER = group_by_compatible(network_conflicts)


def paired(name: str, strategy: BidStrategy, aggregate: Aggregator) -> Experiment:
    """An experiment in which every car bids in compatible sets (paired auctions)."""
    return Experiment(
        name=name,
        population=[BidderType(name, PAIRED_GROUPER, strategy, aggregate)],
        auction=HighestBid(),
        payment=FirstPrice(redistribute=True, free_if_uncontested=True),
        conflicts=network_conflicts,
    )


# Urgency study: benchmarks and urgency-aware strategies (the experiment adds the urgency draw).
def _bench(name, strategy):
    """A benchmark experiment: all cars use one strategy and nobody pays (carma balances never
    change).
    """
    return Experiment(name=name, population=[BidderType(name, group_by_vehicle, strategy, sum)],
                      auction=HighestBid(), payment=NoPayment())


URGENCY_CONFIGS = {
    # Benchmarks: perfect knowledge of urgency (efficiency ceiling) and a coin toss (floor).
    "urg_omniscient": _bench("urg_omniscient", OmniscientUrgency()),
    "urg_coin": _bench("urg_coin", RandomFraction()),
    # Urgency-aware carma strategies.
    "urg_vehicle_wait": only(BidderType("urg_vehicle_wait", group_by_vehicle, UrgencyScaledWait(1.0), sum)),
    "urg_lane_wait": only(BidderType("urg_lane_wait", group_by_lane, UrgencyScaledWait(1.0), sum)),
    "urg_vehicle_fraction": only(BidderType("urg_vehicle_fraction", group_by_vehicle,
                                            UrgencyFraction(0.5, 0.05), sum)),
}

# Switching-overhead variants of the lane-urgency bidder (lanes study).
_LANE = BidderType("lane_urgency", group_by_lane, WaitTimeUrgency(1.0), sum)
LANE_VARIANTS = {
    # Shorter yellow, and a grant ends once half of its winners have crossed.
    "lane_fast_release": replace(only(_LANE), name="lane_fast_release", yellow=2.0,
                                 release_fraction=0.5),
    # Minimum green 10 s, extended while the served lanes still have cars (cap: max_green 30 s).
    "lane_min_green": replace(only(_LANE), name="lane_min_green", min_green=10.0,
                              extend_green=True),
    # Phase auctions: cars bid as NS-through, EW-through, NS-left or EW-left (protected lefts).
    "phase_urgency": only(BidderType("phase_urgency", group_by_phase, WaitTimeUrgency(1.0), sum)),
}

# Regret-minimising bidders (lane groups), for three values of the carma price lam0.
def _regret(name: str, lam0: float) -> Experiment:
    """An experiment in which every driver uses a RegretHedge learner with carma price `lam0`."""
    return only(BidderType(name, group_by_lane, RegretHedge(lam0=lam0), sum))


REGRET_CONFIGS = {f"regret_lane_{lam}": _regret(f"regret_lane_{lam}", lam)
                  for lam in (0.5, 1.0, 2.0, 4.0, 8.0)}

# Non-learning controls with the same shape as the learned policy: lane groups, each car bids a
# fixed share of its balance, a large one when urgent and a small one otherwise.
FIXED_RULE_CONFIGS = {
    "frac_lane_0.9_0.15": only(BidderType("frac_lane_0.9_0.15", group_by_lane,
                                          UrgencyFraction(0.9, 0.15), sum)),
    "frac_lane_0.9_0.05": only(BidderType("frac_lane_0.9_0.05", group_by_lane,
                                          UrgencyFraction(0.9, 0.05), sum)),
}


def _mix(name: str, learner_share: float, lam0: float = 2.0) -> Experiment:
    """A population in which a share of the drivers learn (RegretHedge) and the rest bid by the
    lane-urgency heuristic, competing in the same auctions."""
    learner = BidderType(f"regret_lane_{lam0}", group_by_lane, RegretHedge(lam0=lam0), sum,
                         share=learner_share)
    heuristic = BidderType("lane_urgency", group_by_lane, WaitTimeUrgency(1.0), sum,
                           share=1.0 - learner_share)
    return Experiment(name=name, population=[learner, heuristic], auction=HighestBid(),
                      payment=FirstPrice(redistribute=True, free_if_uncontested=True))


MIX_CONFIGS = {"mix_learn20": _mix("mix_learn20", 0.2), "mix_learn80": _mix("mix_learn80", 0.8)}


# ---- Learning study (docs/learning_spec.md, controller/learning_study.py) ------------------------

def front(name: str, strategy: BidStrategy) -> Experiment:
    """Front-car bidding: lanes bid as groups but only the car at the head of each lane chooses
    the amount (the others bid 0 and are served with it). Equal bids go to the longest wait, and
    payments are shared only among the losing front cars, so every rebate goes to a decision."""
    return Experiment(
        name=name,
        population=[BidderType(name, group_by_lane, strategy, sum, front_only=True)],
        auction=HighestBid(tie_break="wait"),
        payment=FirstPrice(redistribute=True, free_if_uncontested=True, rebate_to="bidders"),
    )


LEARNING_CONFIGS = {
    # A: smoothed best response to the measured rival bids, with a learned value of carma.
    "learn_br": front("learn_br", BestResponseLearner()),
    # B: tabular Q-learning (the control for A).
    "learn_q": front("learn_q", QLearner()),
    # C: the myopic regret-minimising learner (carma price lam0 = 2).
    "learn_hedge": front("learn_hedge", RegretHedge(lam0=2.0)),
    # The lane-urgency rule (bid the seconds waited) with front-car bidding.
    "front_wait": front("front_wait", WaitTimeUrgency(1.0)),
}

# Everything the study compares, by name (sweeps.py picks experiments from here).
STUDY_CONFIGS = {
    "default_lights": DEFAULT_LIGHTS,
    "actuated_lights": ACTUATED_LIGHTS,
    "delay_based_lights": DELAY_BASED_LIGHTS,
    "max_pressure": MAX_PRESSURE,
    "only_vehicle_random": only(BidderType("vehicle_random", group_by_vehicle, RandomFraction(), sum)),
    "only_vehicle_urgency": only(BidderType("vehicle_urgency", group_by_vehicle, WaitTimeUrgency(1.0), sum)),
    "only_lane_urgency": only(BidderType("lane_urgency", group_by_lane, WaitTimeUrgency(1.0), sum)),
    "only_link_fixed": only(BidderType("link_fixed", group_by_link, FixedFraction(0.1), sum)),
    "only_price_follower": only(BidderType("price_follower", group_by_vehicle, RecentPriceFollower(), sum)),
    "only_vehicle_budget": only(BidderType("vehicle_budget", group_by_vehicle, BudgetedUrgency(60.0), sum)),
    "only_lane_budget": only(BidderType("lane_budget", group_by_lane, BudgetedUrgency(60.0), sum)),
    "stop_signs": STOP_SIGNS,
    "stop_signs_grid3": STOP_SIGNS_GRID3,
    **URGENCY_CONFIGS,
    **LANE_VARIANTS,
    **REGRET_CONFIGS,
    **FIXED_RULE_CONFIGS,
    **MIX_CONFIGS,
    **LEARNING_CONFIGS,
    "paired_urgency_sum": paired("paired_urgency_sum", WaitTimeUrgency(1.0), sum),
    "paired_urgency_mean": paired("paired_urgency_mean", WaitTimeUrgency(1.0), mean),
    "paired_fixed_sum": paired("paired_fixed_sum", FixedFraction(0.1), sum),
}


def get_experiment(name: str) -> Experiment:
    """Find an experiment by name: one of the study configurations, or a preset in this file."""
    if name in STUDY_CONFIGS:
        return STUDY_CONFIGS[name]
    return globals()[name]
