"""SignalController: turns "these links should be green" into safe TLS state strings.

The auction layer decides who deserves green. This layer decides how to get there safely:
every link not in the requested green set is red, and links that lose green go
yellow -> all-red before any new green starts. (How long a winner KEEPS green is the Grant's
job, not this class's.)
"""
import traci


class _State:
    """Bookkeeping for one intersection's signal: what is green, what is in yellow, and when the
    clearance ends.
    """
    # Bookkeeping for one intersection's signal.
    def __init__(self):
        """Start with nothing green or yellow and no clearance in progress."""
        # Signal indices currently shown green.
        self.green: set[int] = set()
        # Signal indices currently in their yellow phase (about to turn red).
        self.yellow: set[int] = set()
        # Time at which the yellow phase ends and those links turn red.
        self.yellow_end = 0.0
        # Time at which the whole clearance (yellow + all-red) ends. None = not clearing.
        self.clear_end: float | None = None   # None = not clearing
        # The last state string sent to SUMO, so unchanged states aren't re-sent every step.
        self.last_written: str | None = None
        # The most recent non-empty green request. Used as the "idle" green when nobody is
        # bidding, so the intersection never falls back to all red.
        self.target: set[int] = set()


def _default_green(tls: str) -> set[int]:
    """Links that the network's own program gives priority ('G') in its first phase that has
    any. Used as the idle green before any auction has been won at this intersection."""
    for phase in traci.trafficlight.getAllProgramLogics(tls)[0].getPhases():
        links = set()
        for i, ch in enumerate(phase.state):
            if ch == "G":
                links.add(i)
        if links:
            return links
    return {0}


class SignalController:
    """Guarantees at least one green or yellow link at every controlled intersection, as long as
    all_red is 0 (the default). A non-zero all_red adds a moment with no green or yellow."""

    def __init__(self, tls_ids, yellow: float = 3, all_red: float = 0, yield_table=None):
        """Create the state for every controlled intersection and choose its first idle green.

        yield_table: a conflicts.ConflictTable. With it, a green link that the network's program
        shows as permissive ('g') is sent as 'g' (must yield) whenever one of its foes is green
        too, as SUMO's own signals do; without it every green link is 'G'."""
        self.yield_table = yield_table
        # Seconds of yellow shown to a link that is losing green.
        self.yellow = yellow
        # Seconds of all-red after the yellow, before any new green may start. Keep this at 0
        # if there must always be a green or yellow showing.
        self.all_red = all_red
        # How many signal indices (characters in the state string) each intersection has.
        self.n_links = {}
        # One bookkeeping record per intersection.
        self.state = {}
        for t in tls_ids:
            self.n_links[t] = len(traci.trafficlight.getControlledLinks(t))
            self.state[t] = _State()
            # Start each intersection with the network program's own priority links as the
            # idle green.
            self.state[t].target = _default_green(t)

    def apply(self, tls: str, desired: set[int], t: float) -> None:
        """Call once per step per intersection with the full set of links that should be green.

        An empty `desired` means "nobody has won": keep showing the last green instead of
        going all red. A bid from another approach still switches it, via yellow.
        """
        s = self.state[tls]

        # Remember a real request as the new idle green; fall back to it when nothing is requested.
        if desired:
            s.target = set(desired)
        else:
            desired = s.target

        # Case 1: we are partway through a yellow / all-red clearance that was started earlier.
        if s.clear_end is not None:
            # Still clearing: keep showing the transition and ignore `desired` until it ends.
            if t < s.clear_end:                       # still in yellow / all-red
                # Show yellow until yellow_end, then red (all-red) for the remainder. Links that
                # are staying green stay green.
                if t < s.yellow_end:
                    self._write(tls, s.green, s.yellow)
                else:
                    self._write(tls, s.green, set())
                return
            # Clearance is over: forget it and fall through to apply the desired greens.
            s.clear_end, s.yellow = None, set()       # clearance done

        # Links that are green now but are not wanted any more.
        leaving = s.green - desired
        # Case 2: some links must stop. Don't switch straight to red: begin a clearance.
        if leaving:                                   # start clearing the links losing green
            # These links go yellow now.
            s.yellow = leaving
            # Links that are still wanted keep their green.
            s.green = s.green & desired
            # Yellow ends `yellow` seconds from now; the whole clearance adds the all-red time.
            s.yellow_end = t + self.yellow
            s.clear_end = s.yellow_end + self.all_red
            self._write(tls, s.green, s.yellow)
            return

        # Case 3: nothing needs to stop (links are only being added or kept), so go to the
        # desired set directly. `desired` is never empty here (see the idle fallback above).
        s.green = set(desired)
        self._write(tls, s.green, set())

    def _write(self, tls: str, green: set[int], yellow: set[int]) -> None:
        """Send the state string (green 'G', yellow 'y', everything else red) to SUMO, only when it
        changed.
        """
        # Start with every link red.
        chars = ["r"] * self.n_links[tls]
        # Mark yellow links ('y').
        for i in yellow:
            chars[i] = "y"
        # Mark green links ('G' = green with right of way; 'g' = green but yield to foes).
        for i in green:
            yields = self.yield_table is not None and self.yield_table.yields(tls, i, green)
            chars[i] = "g" if yields else "G"
        # Join the characters into the string SUMO expects, e.g. "rrGGrrrry".
        state = "".join(chars)
        # Only talk to SUMO if the state actually changed since the last step.
        if state != self.state[tls].last_written:     # avoid redundant TraCI calls
            traci.trafficlight.setRedYellowGreenState(tls, state)
            self.state[tls].last_written = state
