"""Which movements at an intersection can NOT be green at the same time.

SUMO's network file records, for every connection through a junction, which other connections
it crosses or merges with (its "foes"). Two signal indices that are not foes of each other can
safely be green together, for example the two opposing straight movements, or a right turn and
the straight movement beside it. Those are the pairs a "paired auction" lets bid together.

The table is built once from the network file with sumolib and cached per process, so every
simulation (and every worker of a parallel sweep) reads the file at most once.

Used by: auction.NonConflictingGreedy (several winners per auction), bidders.group_by_compatible
(paired auctions), and carma.step_intersection (the lane-front fix never turns on a link that
clashes with one already green). lanes_study.py builds a table for each lane-count network.
"""
from pathlib import Path

import sumolib

# The network the study uses, found relative to this file so it works from any directory.
DEFAULT_NET = str(Path(__file__).resolve().parent.parent / "network" / "net"
                  / "RandomGrid5_200_tls.net.xml")


class ConflictTable:
    """Answers "can signal index a and signal index b at intersection `tls` be green together?"

    Attributes:
        pairs (dict): tls id -> set of (a, b) signal-index pairs that conflict (both orders).
    """

    def __init__(self, net_file: str = DEFAULT_NET, mode: str = "foes"):
        """Read the network file and work out, for every traffic light, which pairs of signal indices
        conflict.

        mode "foes": two signal indices conflict if any of their connections are foes in SUMO's
        junction logic (crossing or merging), so a left turn never runs with the opposing
        straight. mode "program": they conflict unless the network's own signal program shows
        them green together in some phase (G or g), so permissive movements (a left turn that
        yields to the opposing straight) can run together, as under SUMO's own signals. In both
        modes `foes` keeps the foe pairs and `minor` the indices the program ever shows as 'g'
        (green but must yield), which the signal controller uses to show 'g' (see signals.py).
        """
        if mode not in ("foes", "program"):
            raise ValueError('mode must be "foes" or "program"')
        self.mode = mode
        # Read the network, with the signal programs (needed for mode "program" and `minor`).
        net = sumolib.net.readNet(net_file, withPrograms=True)
        # tls id -> set of conflicting (a, b) pairs of signal indices.
        self.pairs = {}
        # tls id -> foe pairs (both orders), and the indices shown as 'g' by the program.
        self.foes = {}
        self.minor = {}
        # tls id -> pairs that the program shows green together (both orders).
        self.cogreen = {}
        for tls in net.getTrafficLights():
            tls_id = tls.getID()
            node = net.getNode(tls_id)
            # Every connection through this junction: remember its signal index and its place in
            # the junction's foe table, and which lane it leads to.
            links = []
            for connection in node.getConnections():
                signal_index = connection.getTLLinkIndex()
                # Connections without a signal (e.g. uncontrolled right turns) have index -1.
                if signal_index is not None and signal_index >= 0:
                    links.append((signal_index, connection.getJunctionIndex(),
                                  connection.getToLane().getID()))
            # Compare every ordered pair of connections.
            conflicting = set()
            for a in range(len(links)):
                for b in range(len(links)):
                    if a == b:
                        continue
                    sig_a, junction_a, out_a = links[a]
                    sig_b, junction_b, out_b = links[b]
                    # Connections that share a signal index are always green together.
                    if sig_a == sig_b:
                        continue
                    # They conflict if SUMO lists them as foes (crossing or merging), or if they
                    # leave onto the same lane from different signals (a merge).
                    if node.areFoes(junction_a, junction_b) or out_a == out_b:
                        conflicting.add((sig_a, sig_b))
            self.foes[tls_id] = conflicting
            # Pairs green together in some phase of the program, and the 'g' indices.
            together, minor = set(), set()
            for program in tls.getPrograms().values():
                for phase in program.getPhases():
                    green = [i for i, ch in enumerate(phase.state) if ch in "Gg"]
                    minor.update(i for i, ch in enumerate(phase.state) if ch == "g")
                    together.update((a, b) for a in green for b in green if a != b)
            self.cogreen[tls_id] = together
            self.minor[tls_id] = minor
            if mode == "foes":
                self.pairs[tls_id] = conflicting
            else:
                indices = {sig for sig, _, _ in links}
                self.pairs[tls_id] = {(a, b) for a in indices for b in indices
                                      if a != b and (a, b) not in together}

    def yields(self, tls: str, link: int, green: set[int]) -> bool:
        """True if `link` should be shown as 'g' (green, must yield) with the links `green`: the
        program shows it as 'g' somewhere and one of its foes is green too."""
        if link not in self.minor.get(tls, ()):
            return False
        foes = self.foes.get(tls, ())
        return any((link, other) in foes for other in green if other != link)

    def conflicts(self, tls: str, a: int, b: int) -> bool:
        """True if signal indices a and b can't be green together. A signal never conflicts
        with itself."""
        if a == b:
            return False
        return (a, b) in self.pairs.get(tls, ())


# One table per network file, built the first time it is asked for.
_tables: dict[tuple, ConflictTable] = {}


def get_table(net_file: str = DEFAULT_NET, mode: str = "foes") -> ConflictTable:
    """The (cached) conflict table for a network file (see ConflictTable for `mode`)."""
    key = (net_file, mode)
    if key not in _tables:
        _tables[key] = ConflictTable(net_file, mode)
    return _tables[key]


def network_conflicts(tls: str, a: int, b: int) -> bool:
    """Conflict test for the study's network, in the (tls, a, b) form the auctions and groupers
    expect. Always uses the 5x5 grid (DEFAULT_NET); for other networks call
    get_table(net_file).conflicts instead."""
    return get_table().conflicts(tls, a, b)
