"""Bidding history, for strategies that learn from past auctions (e.g. regret minimisation).

Two views of the same auctions:

* Per intersection: every past auction there, with all the bids, who won and the carma that moved.
  Read it through `ctx.history` (see Context in core.py) inside a BidStrategy.
* Per car: `car.bid_history`, the car's own bid in each auction it took part in, with its outcome
  and the numbers needed to ask "what if I had bid differently?".

History only ever contains auctions that have already finished. A strategy asked for a bid at
step t sees the auctions up to the previous one at that intersection, never the current one.
"""
from collections import deque
from dataclasses import dataclass

import pandas as pd

from core import Bid


@dataclass
class BidRecord:
    """One bid in one past auction (as every bidder at the intersection could have seen it)."""
    # Bidder type that made the bid.
    kind: str
    # Vehicle ids of the cars in the bidding group.
    members: list[str]
    # Signal indices the group asked for.
    links: list[int]
    # What each member offered, and the aggregated amount the auction ranked on.
    member_bids: dict[str, float]
    amount: float
    # Whether this bid won the auction.
    won: bool
    # Total carma change for the group's members (negative = paid, positive = received).
    net_carma: float


@dataclass
class AuctionOutcome:
    """One past auction at one intersection."""
    t: float
    tls: str
    bids: list[BidRecord]

    @property
    def winning_amount(self) -> float:
        """Highest amount among the winning bids (0 if there is none)."""
        best = 0.0
        for bid in self.bids:
            if bid.won and bid.amount > best:
                best = bid.amount
        return best

    @property
    def highest_losing_amount(self) -> float:
        """Highest amount among the losing bids (0 if everyone won or there were no losers)."""
        best = 0.0
        for bid in self.bids:
            if not bid.won and bid.amount > best:
                best = bid.amount
        return best


@dataclass
class CarBidRecord:
    """One car's bid in one auction, and how it turned out (kept in `car.bid_history`).

    To ask "would a different bid have won?", compare against `top_other_amount`: the highest
    amount any OTHER bid offered. For a car bidding alone, a bid above that would have won. For a
    car in a group, its bid only changes the group's amount (`group_amount`), so the answer
    depends on how the group's bid is aggregated.
    """
    t: float
    tls: str
    # What this car offered, and the amount of the group it bid in.
    own_bid: float
    group_amount: float
    # Whether the group won, and this car's carma change (negative = paid, positive = received).
    won: bool
    carma_change: float
    # How long the car had been waiting when it bid (a proxy for how much it needed the green).
    wait_time: float
    # Highest amount among the other bids in that auction, and the winning amount.
    top_other_amount: float
    winning_amount: float
    # The car's balance when it bid (before this auction's payment) and its trip urgency; what a
    # counterfactual "what if I had bid x?" needs (see regret.py).
    balance: float = 0.0
    urgency: float = 1.0
    # Where the car was on its trip: auction-holding intersections left (counting this one) and
    # auctions it had already lost here.
    intersections_left: int = 0
    lost_here: int = 0
    # Which auction this was (numbered from 0 within a run), how many other bids competed, and
    # the rebate the car received if it lost.
    auction_id: int = -1
    n_rivals: int = 0
    rebate: float = 0.0
    # False for a member that did not choose its bid (a follower under front-car bidding).
    # Learners must skip those records.
    decision: bool = True


class AuctionHistory:
    """Past auctions, kept per intersection.

    max_per_tls limits how many of the most recent auctions are kept for each intersection for
    strategies to read (None = keep them all). The full record is always kept separately for
    `to_frame()`, so analysis after a run isn't limited by it.

    With keep_decisions, every decision (one row per car that chose a bid, with its state and the
    outcome) is also kept, for the bid log and the per-intersection price summary
    (`decisions_frame()`, see prices.py).
    """

    def __init__(self, max_per_tls: int | None = 100, keep_decisions: bool = False):
        """Start empty; keep at most `max_per_tls` recent auctions per intersection for strategies to
        read.
        """
        self.max_per_tls = max_per_tls
        # intersection id -> the most recent auctions there, oldest first.
        self._by_tls: dict[str, deque] = {}
        # One flat row per bid per auction, for export.
        self._rows: list[dict] = []
        # Number of auctions recorded so far (the next auction's id).
        self.n_auctions = 0
        # One row per decision, if asked for.
        self.keep_decisions = keep_decisions
        self._decisions: list[dict] = []

    # ---- Writing (called once per auction by carma.step_intersection) ------------------------

    def record(self, t: float, tls: str, bids: list[Bid], winners: list[Bid],
               deltas: dict[str, float]) -> None:
        """Store a finished auction.

        bids are all the bids that competed, winners the ones that won, and deltas the carma
        change per vehicle id returned by the payment rule.
        """
        # Winning bids are matched by identity, not equality.
        winner_ids = set()
        for w in winners:
            winner_ids.add(id(w))

        # Build the public record of each bid.
        records = []
        for bid in bids:
            member_ids = []
            net = 0.0
            for car in bid.members:
                member_ids.append(car.vid)
                net += deltas.get(car.vid, 0.0)
            records.append(BidRecord(
                kind=bid.kind,
                members=member_ids,
                links=sorted(bid.links),
                member_bids=dict(bid.member_bids),
                amount=bid.amount,
                won=id(bid) in winner_ids,
                net_carma=net,
            ))
        outcome = AuctionOutcome(t=t, tls=tls, bids=records)

        # Keep it for strategies to read (oldest entries drop off once max_per_tls is reached).
        if tls not in self._by_tls:
            self._by_tls[tls] = deque(maxlen=self.max_per_tls)
        self._by_tls[tls].append(outcome)

        # Give every car its own record, and add the flat export rows.
        winning = outcome.winning_amount
        auction_id = self.n_auctions
        self.n_auctions += 1
        n_cars = sum(len(bid.members) for bid in bids)
        for bid, rec in zip(bids, records):
            # The best offer from any other bid in this auction, or, when the auction worked it
            # out (auction.annotate_rivals), the bid this one had to beat: with a multi-winner
            # auction only conflicting bids count.
            top_other = 0.0
            for other in records:
                if other is not rec and other.amount > top_other:
                    top_other = other.amount
            contested = len(bids) > 1
            if bid.contested is not None:
                top_other, contested = (bid.rival or 0.0), bid.contested
            deciders = set(bid.deciders())
            for car in bid.members:
                delta = deltas.get(car.vid, 0.0)
                record = CarBidRecord(
                    t=t,
                    tls=tls,
                    own_bid=bid.member_bids[car.vid],
                    group_amount=bid.amount,
                    won=rec.won,
                    carma_change=delta,
                    wait_time=car.wait_time,
                    top_other_amount=top_other,
                    winning_amount=winning,
                    balance=car.carma - delta,
                    urgency=getattr(car, "urgency", 1.0),
                    intersections_left=getattr(car, "intersections_left", 0),
                    lost_here=getattr(car, "lost_here", 0),
                    auction_id=auction_id,
                    n_rivals=len(bids) - 1,
                    rebate=delta if (not rec.won and delta > 0) else 0.0,
                    decision=car.vid in deciders,
                )
                car.bid_history.append(record)
                if self.keep_decisions and record.decision:
                    self._decisions.append({
                        "t": t, "tls": tls, "auction_id": auction_id, "vid": car.vid,
                        # The car's own rule (in a pooled mixed group, members differ).
                        "driver": getattr(car, "driver", None),
                        "kind": getattr(car, "kind", bid.kind),
                        "balance": record.balance, "k": record.intersections_left,
                        "lost_here": record.lost_here, "urgency": record.urgency,
                        "wait": record.wait_time, "bid": record.own_bid,
                        "group_amount": bid.amount, "won": rec.won,
                        "top_rival": top_other, "winning_bid": winning,
                        "payment": -delta if rec.won else 0.0, "rebate": record.rebate,
                        "n_rivals": record.n_rivals, "n_cars": n_cars,
                        "contested": contested,
                    })
            # Links as text for the CSV, e.g. "6 7".
            link_names = []
            for link in rec.links:
                link_names.append(str(link))
            self._rows.append({
                "t": t,
                "tls": tls,
                "kind": rec.kind,
                "members": " ".join(rec.members),
                "links": " ".join(link_names),
                "amount": rec.amount,
                "won": rec.won,
                "net_carma": rec.net_carma,
                "winning_amount": winning,
                "top_other_amount": top_other,
            })

    # ---- Reading (for strategies) -----------------------------------------------------------

    def last(self, tls: str, n: int | None = None) -> list[AuctionOutcome]:
        """The most recent `n` auctions at an intersection, oldest first (all kept if n is None)."""
        past = list(self._by_tls.get(tls, []))
        if n is None:
            return past
        if n <= 0:
            return []
        return past[-n:]

    def winning_amounts(self, tls: str, n: int | None = None) -> list[float]:
        """The winning amount of each of the most recent `n` auctions at an intersection."""
        amounts = []
        for outcome in self.last(tls, n):
            amounts.append(outcome.winning_amount)
        return amounts

    # ---- Export -----------------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Every bid in every auction, one row per bid (not limited by max_per_tls)."""
        return pd.DataFrame(self._rows)

    def decisions_frame(self) -> pd.DataFrame:
        """Every decision (one row per car that chose a bid), if keep_decisions was set."""
        return pd.DataFrame(self._decisions)
