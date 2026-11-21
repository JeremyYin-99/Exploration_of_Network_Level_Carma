"""Auction mechanisms (who wins) and payment rules (who pays / receives what).

Payment rules return {vid: signed carma delta}. This is the ONLY place balances should change,
so total carma conservation can be asserted in one spot.
"""
import random
from typing import Callable, Protocol

from core import Bid


# ---- Auctions -------------------------------------------------------------------------------

class Auction(Protocol):
    """An auction rule: takes every bid at one intersection and returns the winning bid(s)."""
    # The interface every auction follows: take all bids at an intersection, return the winners.
    def run(self, bids: list[Bid]) -> list[Bid]: ...


TIE_BREAKS = ("random", "wait")


def _longest_wait(bid: Bid) -> float:
    """The longest current waiting time among a bid's members (0 for an empty bid)."""
    return max((c.wait_time for c in bid.members), default=0.0)


def _rank(bids: list[Bid], tie_break: str = "random") -> list[Bid]:
    """Highest bid first. Ties are broken at random, or with tie_break="wait" by the longest
    waiting member first (then at random)."""
    # Sort by bid amount; the random second key only matters when amounts are equal, so ties
    # aren't always won by whichever bid happened to come first.
    def sort_key(bid: Bid):
        """Sort key: the bid amount first, then a random number to break exact ties."""
        return (bid.amount, random.random())

    def sort_key_wait(bid: Bid):
        """Sort key: the bid amount, then the longest wait, then a random number."""
        return (bid.amount, _longest_wait(bid), random.random())

    return sorted(bids, key=sort_key_wait if tie_break == "wait" else sort_key, reverse=True)


class HighestBid:
    """Single winner: the highest aggregated bid.

    tie_break: "random" (default) or "wait" (equal bids go to the longest-waiting group, which
    matters when everyone is broke and every bid is 0)."""

    def __init__(self, tie_break: str = "random"):
        """Choose how equal bids are ranked."""
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
        self.tie_break = tie_break

    def run(self, bids: list[Bid]) -> list[Bid]:
        """The single bid with the highest amount wins (ties broken as `tie_break` says)."""
        # Take only the top-ranked bid. Slicing an empty list gives [], so no bids = no winner.
        return _rank(bids, self.tie_break)[:1]


def _every_distinct_pair_conflicts(tls: str, a: int, b: int) -> bool:
    """Default conflict test: any two different signal indices clash, so only one bid can win."""
    # Placeholder conflict rule: two different signal indices are assumed to conflict.
    return a != b


class NonConflictingGreedy:
    """Several winners: take bids in descending order, skipping any whose links conflict with
    an already-accepted bid.

    conflicts(tls, a, b) -> True if signal indices a and b can't be green together.
    TODO: build this from the network's connection `foes` data (see the <junction>/<connection>
    info in the .net.xml). Until then use the default, which treats every distinct pair as a
    conflict (equivalent to a single winner per link set).
    """

    def __init__(self, conflicts: Callable[[str, int, int], bool] = _every_distinct_pair_conflicts,
                 tie_break: str = "random"):
        """Remember the function that says which signal indices can't be green together, and how
        equal bids are ranked ("random", or "wait": longest-waiting group first)."""
        # Function that says whether two signal indices at an intersection clash.
        self.conflicts = conflicts
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
        self.tie_break = tie_break

    def run(self, bids: list[Bid]) -> list[Bid]:
        """Accept bids from highest to lowest, skipping any that conflict with one already accepted.
        """
        # Bids accepted so far; they will all get green at the same time.
        winners: list[Bid] = []
        # Walk the bids from highest to lowest.
        for bid in _rank(bids, self.tie_break):
            # Accept this bid only if none of its links conflict with any link of any bid
            # already accepted (so higher bidders always take priority).
            has_conflict = False
            for w in winners:
                for a in w.links:
                    for b in bid.links:
                        if self.conflicts(bid.tls, a, b):
                            has_conflict = True
            if not has_conflict:
                winners.append(bid)
        return winners


def annotate_rivals(bids: list[Bid], winners: list[Bid],
                    conflicts: Callable[[str, int, int], bool] | None = None) -> None:
    """Set bid.rival and bid.contested for every bid of one auction (see core.Bid).

    A bid's rival is the highest bid that the greedy auction would have accepted for a movement
    conflicting with it, had the bid not been made: the amount it has to beat to win. conflicts
    None means every pair of bids conflicts (single winner), so the rival is simply the highest
    other bid. Bids are taken in the order the auction used (amount, then longest wait, winners
    first among equals); nothing here draws random numbers, so the simulation is unchanged.
    """
    if not bids:
        return
    winner_ids = {id(w) for w in winners}

    def clash(a: Bid, b: Bid) -> bool:
        if conflicts is None:
            return True
        return any(conflicts(a.tls, x, y) for x in a.links for y in b.links)

    order = sorted(bids, key=lambda b: (b.amount, _longest_wait(b), id(b) in winner_ids),
                   reverse=True)
    for i, bid in enumerate(order):
        above = [b for b in order[:i] if id(b) in winner_ids]
        hit = [b.amount for b in above if clash(bid, b)]
        if hit:
            # Blocked (or would be) by a conflicting winner ranked above it.
            bid.rival, bid.contested = max(hit), True
            continue
        # Nothing conflicting above: continue the greedy run below it without it; the first
        # conflicting bid that gets accepted is the one it displaced.
        accepted = list(above)
        bid.rival, bid.contested = 0.0, False
        for other in order[i + 1:]:
            if any(clash(other, a) for a in accepted):
                continue
            accepted.append(other)
            if clash(bid, other):
                bid.rival, bid.contested = other.amount, True
                break


# ---- Payment rules --------------------------------------------------------------------------

class PaymentRule(Protocol):
    """A payment rule: decides how every car's carma balance changes after an auction."""
    # The interface every payment rule follows: given the winners and all bids, return how much
    # each car's balance changes (negative = pays, positive = receives), keyed by vehicle id.
    def settle(self, winners: list[Bid], all_bids: list[Bid]) -> dict[str, float]: ...


def _losing_cars(winners: list[Bid], all_bids: list[Bid],
                 bidders_only: bool = False) -> list[str]:
    """Ids of the cars that lost: members of a non-winning bid who are not also in a winning
    bid. (With paired auctions one car sits in several bids, so a car in a losing bid can still
    have won through another one.) Each car is listed once. With bidders_only, only the members
    that chose the bid count (the front car under front-car bidding, see Bid.bidders)."""
    winning_vids = set()
    for w in winners:
        for vid in w.member_bids:
            winning_vids.add(vid)
    losers = []
    seen = set()
    for b in all_bids:
        if b in winners:
            continue
        candidates = b.deciders() if bidders_only else b.member_bids
        for vid in candidates:
            if vid not in winning_vids and vid not in seen:
                seen.add(vid)
                losers.append(vid)
    return losers


class NoPayment:
    """Winners pay nothing and balances never change. Used by the omniscient and coin-toss
    benchmarks, where the bids are not carma."""

    def settle(self, winners: list[Bid], all_bids: list[Bid]) -> dict[str, float]:
        """Nobody pays or receives anything."""
        return {}


class FirstPrice:
    """Each winning group pays its own aggregated bid.

    The group's payment is split among its members in proportion to their individual bids,
    so no member ever pays more than they offered (and thus never more than their balance).

    If redistribute is True, everything collected is shared equally among the members of the
    losing bids (so total carma is conserved); otherwise it leaves the system.

    If free_if_uncontested is True, an auction with no losing bid costs the winner nothing.
    Without it the winner pays its own bid even with no competition; with redistribute there is
    nobody to receive that payment, so it leaves the system. Turn it on to keep total carma
    exactly constant. (It makes no difference to SecondPrice, which is already free then.)

    rebate_to decides who shares the redistributed carma: "members" (every car in a losing bid)
    or "bidders" (only the cars that chose a losing bid, i.e. the front car of each losing lane
    under front-car bidding), so income goes only to those who made a decision.

    integer: carma comes in whole units (as in the karma papers). Bids are whole numbers already
    (see bidders.make_bid); a group's payment is split among its members in whole units (largest
    remainders), and the collected carma is shared among the losers as evenly as possible, the
    leftover units going one each to the longest-waiting losers. Total carma stays exactly
    constant.
    """

    def __init__(self, redistribute: bool = False, free_if_uncontested: bool = False,
                 rebate_to: str = "members", integer: bool = False, free_per_bid: bool = False):
        """Choose whether collected carma is redistributed to the losers, whether uncontested wins
        are free, who among the losers receives the rebate, and whether carma is whole units.
        """
        if rebate_to not in ("members", "bidders"):
            raise ValueError('rebate_to must be "members" or "bidders"')
        # Whether collected carma is paid back out to the losers (True) or removed (False).
        self.redistribute = redistribute
        # Whether a winner with no competition pays nothing instead of its own bid.
        self.free_if_uncontested = free_if_uncontested
        # Who among the losers shares the rebate.
        self.rebate_to = rebate_to
        # Whole units of carma (payments and rebates are whole numbers).
        self.integer = integer
        # With free_if_uncontested: judge "uncontested" per winning bid (it displaced no
        # conflicting bid, bid.contested is False; see annotate_rivals) instead of per auction
        # (no losing bid at all). Matters only with a multi-winner auction.
        self.free_per_bid = free_per_bid

    def price(self, winner: Bid, winners: list[Bid], all_bids: list[Bid]) -> float:
        """What a winning group pays: its own bid (free when uncontested, if that option is on)."""
        # Uncontested = every bid in this auction won, so there is no losing bid. The winner
        # then pays nothing, if that option is on.
        if self.free_if_uncontested:
            if getattr(self, "free_per_bid", False) and winner.contested is not None:
                if not winner.contested:
                    return 0.0
            elif len(_losing_cars(winners, all_bids, self.rebate_to == "bidders")) == 0:
                return 0.0
        # First-price rule: the winner pays exactly what it bid. SecondPrice overrides this
        # one method; everything else in settle() is shared.
        return winner.amount

    def settle(self, winners: list[Bid], all_bids: list[Bid]) -> dict[str, float]:
        """Charge every winning group its price (split among its members by their bids) and
        optionally pay the total out to the losing cars.
        """
        if getattr(self, "integer", False):
            return self._settle_integer(winners, all_bids)
        # Running tally of each car's balance change.
        deltas: dict[str, float] = {}
        # Total carma taken from winners; may be handed to the losers afterwards.
        collected = 0.0
        for w in winners:
            # What this winning group owes, according to the rule (first or second price).
            price = self.price(w, winners, all_bids)
            # Sum of the members' own bids, used to work out each member's share.
            total = sum(w.member_bids.values())
            for vid, b in w.member_bids.items():
                # Each member pays a share of the price proportional to what they offered.
                # If everyone bid 0, nobody pays (avoids dividing by zero).
                if total > 0:
                    share = price * b / total
                else:
                    share = 0.0
                # Subtract the share from that car's change (they are paying).
                deltas[vid] = deltas.get(vid, 0.0) - share
            collected += price

        # Optionally pay the collected carma back out to the cars that lost.
        if self.redistribute and collected > 0:
            # Every member (or, with rebate_to="bidders", every decider) of every losing bid (a
            # bid is a loser if it isn't among the winners).
            losers = _losing_cars(winners, all_bids, self.rebate_to == "bidders")
            # Split the pot equally among them. If there are no losers, nothing is paid out
            # (the carma leaves the system).
            for vid in losers:
                deltas[vid] = deltas.get(vid, 0.0) + collected / len(losers)
        return deltas


    def _settle_integer(self, winners: list[Bid], all_bids: list[Bid]) -> dict[str, float]:
        """settle() with whole units of carma (see the class docstring)."""
        deltas: dict[str, int] = {}
        collected = 0
        for w in winners:
            price = int(round(self.price(w, winners, all_bids)))
            for vid, share in split_whole(price, list(w.member_bids.items())).items():
                deltas[vid] = deltas.get(vid, 0) - share
            collected += price
        if self.redistribute and collected > 0:
            losers = _losing_cars(winners, all_bids, self.rebate_to == "bidders")
            if losers:
                cars = {c.vid: c for b in all_bids for c in b.members}
                # Longest-waiting losers first get the leftover units.
                order = sorted(losers, key=lambda v: (-cars[v].wait_time, v))
                each, extra = divmod(collected, len(order))
                for i, vid in enumerate(order):
                    deltas[vid] = deltas.get(vid, 0) + each + (1 if i < extra else 0)
        return {vid: float(d) for vid, d in deltas.items() if d != 0}


def split_whole(total: int, weights: list[tuple[str, float]]) -> dict[str, int]:
    """Split a whole number `total` among (id, weight) pairs in proportion to the weights, in whole
    units: everyone gets the floor of their share and the leftover units go to the largest
    remainders (ties in the order given). Nobody gets more than their weight if the weights are
    whole numbers summing to at least `total`. Zero total or zero weights give nothing."""
    weight_sum = sum(w for _, w in weights)
    if total <= 0 or weight_sum <= 0:
        return {}
    exact = [(vid, total * w / weight_sum) for vid, w in weights]
    shares = {vid: int(x // 1) for vid, x in exact}
    left = total - sum(shares.values())
    by_remainder = sorted(range(len(exact)), key=lambda i: -(exact[i][1] - shares[exact[i][0]]))
    for i in by_remainder[:left]:
        shares[exact[i][0]] += 1
    return {vid: s for vid, s in shares.items() if s}


class SecondPrice(FirstPrice):
    """Each winner pays the highest losing bid (or 0 if uncontested), capped at its own bid."""

    def price(self, winner: Bid, winners: list[Bid], all_bids: list[Bid]) -> float:
        """What a winning group pays: the best competing bid, but never more than its own bid."""
        # Amounts of every bid that didn't win.
        # Only bids that share no car with a winner count as competition (with paired auctions
        # an overlapping bid is just another way of grouping the same cars).
        winning_vids = set()
        for w in winners:
            for vid in w.member_bids:
                winning_vids.add(vid)
        losing = []
        for b in all_bids:
            if b in winners:
                continue
            shares_car = False
            for vid in b.member_bids:
                if vid in winning_vids:
                    shares_car = True
            if not shares_car:
                losing.append(b.amount)
        # Find the best losing bid; stays 0 if there were no competitors.
        best_losing = 0.0
        for amount in losing:
            if amount > best_losing:
                best_losing = amount
        # Pay the best losing bid, but never more than the winner's own bid.
        return min(winner.amount, best_losing)
