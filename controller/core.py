"""Small shared data types passed between layers (no TraCI calls and no logic beyond helpers)."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from car import Car

if TYPE_CHECKING:
    # Only for type hints; history.py imports from this module, so importing it here at run time
    # would be circular.
    from history import AuctionHistory


@dataclass
class Context:
    """Read-only snapshot of the world for one simulation step."""
    # Current simulation time in seconds.
    t: float
    # For each controlled intersection, the cars close enough to it to be allowed to bid.
    approaching: dict[str, list[Car]]   # tls id -> cars within bid range heading for it
    # For each intersection, the signal indices whose outgoing lane is full (see blocking.py).
    blocked: dict[str, set[int]] = field(default_factory=dict)
    # Past auctions at every intersection (see history.py). A BidStrategy can read it to learn
    # from earlier outcomes, e.g. ctx.history.last(car.tls, 10). It only contains auctions that
    # have already finished.
    history: "AuctionHistory | None" = None

    def front_car(self, tls: str, lane: str) -> Car | None:
        """The car closest to the stop line on `lane` (the head of that lane's queue), among the
        cars approaching `tls`. None if the lane has no approaching cars."""
        front = None
        for car in self.approaching.get(tls, []):
            if car.lane == lane:
                if front is None or car.dist < front.dist:
                    front = car
        return front

    def on_approach(self, tls: str) -> set[str]:
        """Ids of the cars currently approaching `tls`."""
        # The ids of cars currently approaching `tls`. Used to check whether a granted group
        # has finished crossing: once a car passes the stop line it drops out of this set.
        vids = set()
        for c in self.approaching.get(tls, []):
            vids.add(c.vid)
        return vids


@dataclass
class Bid:
    """One bidding entity's offer for green on a set of signal links at an intersection.

    A single vehicle and a lane group are both just a Bid with different members.
    """
    # Which intersection the bid is for.
    tls: str
    # The cars bidding together. One car for single-vehicle bidding, several for a group.
    members: list[Car]
    # Signal indices (movements) the group needs green. The winner gets exactly these links green.
    links: set[int]                 # signal indices this bid needs green
    # What each individual car offered. Kept so payments can be split fairly among members.
    member_bids: dict[str, float]   # vid -> that member's individual bid
    # The single number the auction compares: member bids combined by the aggregator.
    amount: float                   # aggregated group bid used by the auction
    # Name of the bidder type that made this bid (see BidderType in bidders.py).
    kind: str = "default"
    # Vehicle ids of the members that chose an amount. None = every member did. With front-car
    # bidding (BidderType.front_only) only the car at the head of the lane decides; the others
    # bid 0 and ride along. Payment rules can pay rebates to deciders only (FirstPrice.rebate_to).
    bidders: set[str] | None = None
    # Filled in right after the auction (auction.annotate_rivals): `rival` is the amount this
    # bid had to beat, the highest bid that would have won a movement conflicting with it had
    # this bid not been made; `contested` says whether there was such a bid at all (False = it
    # displaced nobody). With a single-winner auction these are the highest other bid and
    # "there was another bid". None = not computed.
    rival: float | None = None
    contested: bool | None = None

    def deciders(self) -> list[str]:
        """Ids of the members that chose an amount (every member unless `bidders` is set)."""
        if self.bidders is None:
            return list(self.member_bids)
        return [vid for vid in self.member_bids if vid in self.bidders]


@dataclass
class Grant:
    """A won bid. Holds green until every member has cleared the intersection (or a timeout)."""
    # The bid that won. Its `links` tell the signal controller what to keep green.
    bid: Bid
    # Vehicle ids of the group at the moment of winning. Cars arriving later are NOT included,
    # so a steady stream of arrivals can't keep the green forever.
    members: set[str]               # vehicle ids snapshotted at award time
    # Simulation time at which the grant was awarded.
    started: float
    # Longest the grant may last, even if some member is stuck.
    max_duration: float             # safety cap so one stalled car can't hold green forever
    # Time at which the grant became active (its links actually turned green).
    # None means the grant is still pending: it has won, but its front car is too far away.
    activated_at: float | None = None
    # The lanes the members were on when the grant was won. With the lane-front fix, the head
    # car of each of these lanes is also served, whatever movement it needs.
    lanes: set[str] = field(default_factory=set)
    # Fewest seconds the grant stays green once active, even if its members have crossed (0 = none).
    min_duration: float = 0.0
    # Share of the members that must have crossed before the grant may end (1.0 = all of them).
    release_fraction: float = 1.0
    # If True, the grant keeps its green past the members while cars are still approaching on the
    # lanes it serves (up to max_duration): a queue-clearing extension.
    extend: bool = False

    def front_distance(self, ctx: Context, serve_lane_front: bool = False) -> float | None:
        """Distance to the stop line of the grant's "front car": the closest member, or, with
        serve_lane_front, the closest car of any kind on the lanes the grant serves.
        None if there is no such car on the approach any more."""
        closest = None
        for car in ctx.approaching.get(self.bid.tls, []):
            is_member = car.vid in self.members
            on_served_lane = serve_lane_front and car.lane in self.lanes
            if is_member or on_served_lane:
                if closest is None or car.dist < closest:
                    closest = car.dist
        return closest

    def green_links(self, ctx: Context, serve_lane_front: bool, blocked: set[int]) -> set[int]:
        """Signal indices this grant wants green right now.

        Always the links the bid asked for. With serve_lane_front, also the link of the car at
        the head of each served lane (recomputed every step, since the head car changes as cars
        cross). This stops a head car that needs a different movement from blocking the group
        behind it. Links leading into a full lane are left out.
        """
        links = set(self.bid.links)
        if serve_lane_front:
            for lane in self.lanes:
                front = ctx.front_car(self.bid.tls, lane)
                if front is not None and front.link not in blocked:
                    links.add(front.link)
        return links

    def front_links(self, ctx: Context, blocked: set[int]) -> set[int]:
        """Signal indices of the cars at the head of each lane this grant serves (lane-front
        fix), leaving out links into a full lane. carma.py adds these to the green set, skipping
        any that would conflict with a link that is already green."""
        links = set()
        for lane in self.lanes:
            front = ctx.front_car(self.bid.tls, lane)
            if front is not None and front.link not in blocked:
                links.add(front.link)
        return links

    def try_activate(self, ctx: Context, activation_dist: float,
                     serve_lane_front: bool = False) -> None:
        """Activate the grant once its front car is within `activation_dist` metres of the
        intersection. Until then the grant is pending and its links are not given green."""
        # Already active: nothing to do.
        if self.activated_at is not None:
            return
        distance = self.front_distance(ctx, serve_lane_front)
        # Activate only if a member is on the approach and close enough.
        if distance is not None and distance <= activation_dist:
            self.activated_at = ctx.t

    @property
    def active(self) -> bool:
        """True once the grant has turned green (its front car got close enough)."""
        return self.activated_at is not None

    def done(self, ctx: Context) -> bool:
        """True when the grant should end: it timed out, its members have crossed (see min_duration,
        release_fraction and extend), or a pending grant never became active in time.
        """
        tls = self.bid.tls
        if self.activated_at is not None:
            held = ctx.t - self.activated_at
            # An active grant times out `max_duration` after it turned green.
            if held > self.max_duration:
                return True
            # Minimum green: never end early.
            if held < self.min_duration:
                return False
            # Extension: keep green while cars are still queued on the lanes this grant serves.
            if self.extend:
                for car in ctx.approaching.get(tls, []):
                    if car.lane in self.lanes:
                        return False
        # Done when enough of the original members have crossed the stop line (or left).
        remaining = self.members & set(ctx.on_approach(tls))
        if len(remaining) <= len(self.members) * (1.0 - self.release_fraction):
            return True
        if self.activated_at is not None:
            return False
        # A pending grant whose front car never got close enough also expires, measured from
        # when it was awarded, so a stuck group can't block the intersection forever.
        return ctx.t - self.started > self.max_duration


@dataclass
class AuctionRecord:
    """One row of the experiment log."""
    # When the auction was run.
    t: float
    # Which intersection it was run at.
    tls: str
    # How many bids competed.
    n_bids: int
    # For each winning bid, the vehicle ids in that group.
    winners: list[list[str]] = field(default_factory=list)   # member vids of each winning bid
    # The aggregated amount of each winning bid (same order as `winners`).
    amounts: list[float] = field(default_factory=list)
    # The bidder type of each winning bid (same order as `winners`).
    kinds: list[str] = field(default_factory=list)


@dataclass
class CarRecord:
    """One row of the per-car results: how a single vehicle fared over its whole trip."""
    vid: str
    # Bidder type the car was assigned.
    kind: str
    # Simulation time the car entered and left the network, and the difference.
    depart: float
    arrive: float
    travel_time: float
    # Seconds spent stopped or crawling, summed over the trip.
    wait_time: float
    # Seconds lost compared with driving the whole route at full speed (SUMO's "time loss").
    time_loss: float
    # Number of auctions this car was part of the winning group in.
    wins: int
    # Carma balance when the car left, and the balance it started with.
    final_carma: float
    start_carma: float
    # Persistent-driver runs: which driver made this trip (None when carma is not persistent).
    driver: str | None = None
    # Urgency (value of time) of this trip; 1.0 when the experiment has no urgency.
    urgency: float = 1.0
