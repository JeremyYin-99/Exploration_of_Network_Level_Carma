"""Groupers (who bids together), aggregators (how member bids combine) and bid construction.

Single-vehicle bidding is simply a grouper that makes groups of size 1.
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from car import Car
from core import Bid, Context
from strategies import BidStrategy

# A grouper takes all the cars at one intersection and returns a list of groups of cars.
Grouper = Callable[[list[Car]], list[list[Car]]]
# An aggregator takes the members' individual bids and returns one number for the group.
Aggregator = Callable[[Iterable[float]], float]


@dataclass
class BidderType:
    """One bidding behaviour. Cars are assigned a type when they enter the simulation, so several
    types can compete in the same run (see Experiment.population in config.py)."""
    # Label stored on each car and used to split the results (e.g. "lane_urgency").
    name: str
    # How cars of this type group up when bidding (alone, per lane, per link, ...).
    grouper: Grouper
    # How much each car of this type bids.
    strategy: BidStrategy
    # How a group's member bids combine into the group's bid.
    aggregate: Aggregator
    # Relative share of cars given this type. Shares are weights, so 2 and 1 means two thirds
    # and one third.
    share: float = 1.0
    # Carma balance cars of this type start with. None means use the experiment's
    # `initial_carma`.
    initial_carma: float | None = None
    # Front-car bidding: in each group only the car closest to the stop line decides how much to
    # bid; the other members bid 0, pay nothing and are served by the grant if the group wins.
    # Use with group_by_lane and the sum (or max) aggregator. Removes the free-riding that lane
    # groups invite (a member who bids 0 rides on its lane-mates' bids).
    front_only: bool = False


# ---- Groupers: cars approaching ONE intersection -> list of bidding groups -----------------

def group_by_vehicle(cars: list[Car]) -> list[list[Car]]:
    """One group per car: every car bids on its own."""
    # Every car is its own group of one, so each car bids alone (single-vehicle bidding).
    groups = []
    for c in cars:
        groups.append([c])
    return groups


def group_by_lane(cars: list[Car]) -> list[list[Car]]:
    """One group per incoming lane. A lane may serve several links (e.g. left + straight),
    so the group's bid asks for the union of its members' links."""
    # Dictionary that creates an empty list the first time a new lane is seen.
    groups = defaultdict(list)
    # Put each car into the bucket for the lane it is queued on.
    for c in cars:
        groups[c.lane].append(c)
    # Each bucket (all cars in one lane) becomes one bidding group.
    return list(groups.values())


def group_by_link(cars: list[Car]) -> list[list[Car]]:
    """One group per (lane, signal index): all cars that need exactly the same movement."""
    groups = defaultdict(list)
    # Key on both lane and link, so a lane's left-turners and through-traffic are separate groups.
    for c in cars:
        groups[(c.lane, c.link)].append(c)
    return list(groups.values())


_phase_tables: dict = {}


def _phase_table(net_file: str) -> dict:
    """{tls: {signal index: (axis, movement class)}} for a network: axis is "NS" or "EW" by the
    direction of the incoming road, movement class is "left" (left turns and U-turns) or
    "through" (straight and right)."""
    if net_file not in _phase_tables:
        import sumolib
        net = sumolib.net.readNet(net_file, withPrograms=False)
        table = {}
        for tls in net.getTrafficLights():
            links = {}
            for conn in net.getNode(tls.getID()).getConnections():
                index = conn.getTLLinkIndex()
                if index is None or index < 0:
                    continue
                edge = conn.getFrom()
                (x1, y1), (x2, y2) = edge.getShape()[-2], edge.getShape()[-1]
                axis = "EW" if abs(x2 - x1) > abs(y2 - y1) else "NS"
                links[index] = (axis, "left" if conn.getDirection() in ("l", "L", "t") else "through")
            table[tls.getID()] = links
        _phase_tables[net_file] = table
    return _phase_tables[net_file]


def group_by_phase(cars: list[Car]) -> list[list[Car]]:
    """One group per signal phase: (NS or EW) x (through+right, or left). Cars bid for a whole
    phase, so the winner turns every compatible movement of that phase green at once."""
    if not cars:
        return []
    import traci
    table = _phase_table(traci.simulation.getOption("net-file"))
    groups = defaultdict(list)
    for c in cars:
        groups[table.get(c.tls, {}).get(c.link, ("?", "?"))].append(c)
    return list(groups.values())


def group_by_compatible(conflicts: Callable[[str, int, int], bool]) -> Grouper:
    """Paired auctions: build a grouper whose groups are sets of movements that can be green
    together, so cars that don't conflict bid (and pass) together.

    Cars are first split into link groups (same lane, same movement). Two link groups are
    compatible when their signal indices don't conflict (`conflicts(tls, a, b)`, see
    conflicts.py). Every MAXIMAL set of mutually compatible link groups becomes one group: no
    group can take in one more waiting movement without clashing with something already in it.

    A car can therefore belong to several groups (one per way of pairing its movement with
    others), so one car's bid appears in several competing bids. Only one group wins, and the
    payment rules only charge or reward cars once (see auction.py).
    """

    def grouper(cars: list[Car]) -> list[list[Car]]:
        """Turn the waiting cars into groups: one group per maximal set of mutually compatible
        movements.
        """
        # Waiting cars split by movement: one link group per (lane, signal index).
        link_groups = group_by_link(cars)
        n = len(link_groups)
        if n == 0:
            return []
        tls = link_groups[0][0].tls
        # compatible[i] = indices of link groups that can be green together with group i.
        compatible = []
        for i in range(n):
            partners = set()
            for j in range(n):
                if i != j:
                    if not conflicts(tls, link_groups[i][0].link, link_groups[j][0].link):
                        partners.add(j)
            compatible.append(partners)

        # Enumerate maximal sets of mutually compatible groups (Bron-Kerbosch).
        cliques = []

        def extend(current, candidates, excluded):
            """Bron-Kerbosch step: grow the set `current`; when nothing can be added it is a maximal
            compatible set.
            """
            # Nothing can be added and nothing earlier could have been: `current` is maximal.
            if not candidates and not excluded:
                cliques.append(current)
                return
            for v in list(candidates):
                extend(current + [v], candidates & compatible[v], excluded & compatible[v])
                candidates = candidates - {v}
                excluded = excluded | {v}

        extend([], set(range(n)), set())

        # Flatten each set of link groups into one list of cars.
        groups = []
        for clique in cliques:
            members = []
            for index in sorted(clique):
                for car in link_groups[index]:
                    members.append(car)
            groups.append(members)
        return groups

    return grouper


def group_by_intersection(cars: list[Car]) -> list[list[Car]]:
    """Everyone at the intersection forms one group (none if there are no cars)."""
    # Everyone at the intersection forms one group; return no groups if the intersection is empty.
    if cars:
        return [cars]
    return []


# ---- Aggregators: member bids -> one group bid ----------------------------------------------
# `sum` and `max` are builtins and work directly.

def mean(values: Iterable[float]) -> float:
    """Average of the member bids (0 for an empty group); an Aggregator like sum and max."""
    # Convert to a list so it can be measured and iterated more than once.
    values = list(values)
    # Average bid per member; return 0 for an empty group to avoid dividing by zero.
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---- Bid construction -----------------------------------------------------------------------

def whole(amount: float, balance: float) -> float:
    """A bid in whole units of carma: rounded down, never above the (whole) balance, never < 0."""
    return float(max(0, min(math.floor(amount + 1e-9), math.floor(balance + 1e-9))))


def make_bid(members: list[Car], tls: str, strategy: BidStrategy,
             aggregate: Aggregator, ctx: Context, kind: str = "default",
             front_only: bool = False, integer: bool = False,
             member_strategies: dict | None = None) -> Bid:
    """Build one Bid from a group of cars: ask the strategy for each member's bid, collect the links
    the members need, and combine the member bids into the group amount.

    With front_only, only the member closest to the stop line is asked; the others bid 0. With
    integer (whole units of carma), every bid is rounded down to a whole number. With
    member_strategies (bidder type name -> strategy; pooled groups in a mixed population), every
    member bids by its own type's rule instead of `strategy`.
    """
    # Ask the strategy how much each member bids, keyed by vehicle id.
    member_bids = {}
    # The group needs green for every movement any of its members wants (a set, so
    # duplicates collapse).
    links = set()
    bidders = None
    if front_only and members:
        # The head of the queue decides for the whole lane.
        front = min(members, key=lambda c: c.dist if c.dist is not None else float("inf"))
        for c in members:
            member_bids[c.vid] = 0.0
            links.add(c.link)
        member_bids[front.vid] = strategy.amount(front, ctx)
        if integer:
            member_bids[front.vid] = whole(member_bids[front.vid], front.carma)
        bidders = {front.vid}
    else:
        for c in members:
            rule = member_strategies.get(c.kind, strategy) if member_strategies else strategy
            member_bids[c.vid] = rule.amount(c, ctx)
            if integer:
                member_bids[c.vid] = whole(member_bids[c.vid], c.carma)
            links.add(c.link)
    return Bid(
        tls=tls,
        members=members,
        links=links,
        member_bids=member_bids,
        # The bidder type this bid came from (all members share it).
        kind=kind,
        # Collapse the individual bids into the single number the auction ranks on.
        amount=aggregate(member_bids.values()),
        bidders=bidders,
    )
