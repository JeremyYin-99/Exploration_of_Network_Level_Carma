"""Stronger signal baselines than SUMO's fixed 90 s program (the `default_lights` control group).

Three controllers, chosen with `Experiment.control`:

  "actuated"      SUMO's gap-based actuated control: a green phase is extended while the
                  (automatically placed) detectors keep seeing vehicles, between minDur and maxDur.
  "delay_based"   SUMO's delay-based actuated control (Oertel & Wagner 2011): a green phase is
                  extended while vehicles within detectorRange are accumulating time loss.
  "max_pressure"  Max-pressure control (Varaiya 2013), run from Python through TraCI: once a phase
                  has had its minimum green, switch to the phase with the largest pressure.

All three use the phases of the network's own static program: the same movements, with the same
priorities ('G' = priority, 'g' = permissive left that yields). So the only thing that differs from
default_lights is WHEN the phases switch, not WHICH movements may be green together.

The two SUMO controllers are loaded as an additional file written next to the networks
(network/tls/<net>_<type>.add.xml, generated on first use by `program_file`). Loading an additional
program makes it the active one; carma.run also selects it explicitly.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import traci

# Where the generated actuated / delay-based programs are written.
TLS_DIR = Path(__file__).resolve().parent.parent / "network" / "tls"
# Bounds of every green phase of an actuated program, in seconds (netconvert's defaults for
# --tls.min-dur / --tls.max-dur).
MIN_DUR = 5
MAX_DUR = 50
# Controller parameters written into each program (SUMO's documented defaults).
PARAMS = {
    "actuated": {"max-gap": "3.0", "detector-gap": "2.0", "passing-time": "2.0"},
    "delay_based": {"detectorRange": "100", "minTimeLoss": "1"},
}
# The SUMO-side controllers this module can write programs for.
SUMO_CONTROLS = tuple(PARAMS)


def is_green_phase(state: str) -> bool:
    """A phase that serves traffic: at least one green ('G' or 'g') and no yellow."""
    return "y" not in state.lower() and any(ch in "Gg" for ch in state)


def needs_control(states: list[str]) -> bool:
    """Whether a signal program has anything to arbitrate: some phase shows a red. Junctions whose
    only phase is all-green (the grid's corners) are left on their own program, as in carma.py."""
    return any("r" in s.lower() for s in states)


def program_file(net: str, control: str) -> str:
    """Path of the additional file holding a `control` ("actuated" or "delay_based") program for
    every signalised junction of `net`, writing it if it is missing or older than the network.

    Each program copies the network's static phases. Green phases get minDur = MIN_DUR and
    maxDur = MAX_DUR (their static duration is kept as the initial duration); yellow phases are
    copied unchanged. The programID is the control type, e.g. "actuated".
    """
    if control not in PARAMS:
        raise ValueError(f"No SUMO program type for control={control!r}; use one of {SUMO_CONTROLS}")
    net_path = Path(net).resolve()
    stem = net_path.name.removesuffix(".xml").removesuffix(".net")
    out = TLS_DIR / f"{stem}_{control}.add.xml"
    if out.exists() and out.stat().st_mtime >= net_path.stat().st_mtime:
        return str(out)

    root = ET.Element("additional")
    for logic in ET.parse(net_path).getroot().iter("tlLogic"):
        phases = logic.findall("phase")
        if not needs_control([ph.get("state") for ph in phases]):
            continue
        program = ET.SubElement(root, "tlLogic", id=logic.get("id"), type=control,
                                programID=control, offset=logic.get("offset", "0"))
        for key, value in PARAMS[control].items():
            ET.SubElement(program, "param", key=key, value=value)
        for ph in phases:
            attrs = {"duration": ph.get("duration"), "state": ph.get("state")}
            if is_green_phase(ph.get("state")):
                attrs["minDur"] = str(MIN_DUR)
                attrs["maxDur"] = str(MAX_DUR)
            ET.SubElement(program, "phase", **attrs)
    ET.indent(root)
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    # Several workers may ask for the same file at once: write a private temporary file, then
    # rename it into place (atomic), so nobody ever reads a half-written file.
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    ET.ElementTree(root).write(tmp, encoding="UTF-8", xml_declaration=True)
    os.replace(tmp, out)
    return str(out)


def activate_programs(control: str) -> int:
    """After traci.start: switch every junction that has a `control` program to it. Returns how
    many junctions were switched."""
    switched = 0
    for tls in traci.trafficlight.getIDList():
        programs = [logic.programID for logic in traci.trafficlight.getAllProgramLogics(tls)]
        if control in programs:
            traci.trafficlight.setProgram(tls, control)
            switched += 1
    return switched


class _Junction:
    """Max-pressure bookkeeping for one junction: its green phases and what is showing."""

    def __init__(self, phases: list[str], served: list[dict[str, set[str]]]):
        """`phases` are the green phase states; `served[j]` maps each incoming lane that phase j
        serves to the outgoing lanes it serves from that lane."""
        self.phases = phases
        self.served = served
        # Index of the phase that is green (or that the junction is changing to).
        self.current = 0
        # Time the current green started (after its yellow).
        self.green_since = 0.0
        # While changing phase: the time the yellow ends, and the phase that comes next.
        self.yellow_until: float | None = None
        self.next = 0


class MaxPressure:
    """Max-pressure signal control (Varaiya 2013) over the network's own green phases.

    The pressure of a phase is the sum, over the incoming lanes it serves, of
        (vehicles on the incoming lane) - (mean vehicles on the outgoing lanes it serves from it),
    so a phase scores high when it would move many waiting vehicles into roads that have room.
    Using lanes rather than individual movements avoids counting a shared lane (e.g. straight +
    right) twice.

    Once the current phase has been green for `min_green` seconds, the junction switches to the
    phase with the highest pressure if that is strictly higher than the current phase's; ties keep
    the current phase. A switch shows `yellow` seconds of yellow on every link that loses green
    (links green in both phases stay green), then the new phase. There is no maximum green: plain
    max-pressure needs none to be stable.
    """

    def __init__(self, tls_ids: list[str], min_green: float = 10.0, yellow: float = 3.0):
        """Read the green phases and the lanes they serve for every junction in `tls_ids`, and show
        each junction's first green phase."""
        self.min_green = min_green
        self.yellow = yellow
        self.junctions: dict[str, _Junction] = {}
        # Every lane whose vehicle count enters some pressure.
        self.lanes: set[str] = set()
        t = traci.simulation.getTime()
        for tls in tls_ids:
            links = traci.trafficlight.getControlledLinks(tls)
            phases, served = [], []
            for phase in traci.trafficlight.getAllProgramLogics(tls)[0].getPhases():
                state = phase.state
                if not is_green_phase(state) or state in phases:
                    continue
                lanes: dict[str, set[str]] = {}
                for index, ch in enumerate(state):
                    if ch in "Gg":
                        for in_lane, out_lane, _via in links[index]:
                            lanes.setdefault(in_lane, set()).add(out_lane)
                            self.lanes.update((in_lane, out_lane))
                phases.append(state)
                served.append(lanes)
            if not phases:
                continue
            junction = _Junction(phases, served)
            junction.green_since = t
            self.junctions[tls] = junction
            traci.trafficlight.setRedYellowGreenState(tls, phases[0])

    def _pressure(self, served: dict[str, set[str]], count: dict[str, int]) -> float:
        """Pressure of one phase given the vehicle count of every lane."""
        total = 0.0
        for in_lane, out_lanes in served.items():
            downstream = sum(count[lane] for lane in out_lanes) / len(out_lanes)
            total += count[in_lane] - downstream
        return total

    @staticmethod
    def _transition(current: str, nxt: str) -> str:
        """The yellow state between two phases: links losing green show 'y', links green in both
        keep their current character, everything else is red."""
        chars = []
        for cur, new in zip(current, nxt):
            if cur in "Gg" and new not in "Gg":
                chars.append("y")
            elif cur in "Gg":
                chars.append(cur)
            else:
                chars.append("r")
        return "".join(chars)

    def step(self, t: float) -> None:
        """Advance every junction by one simulation step (call once per step)."""
        count = None
        for tls, j in self.junctions.items():
            # Finishing a yellow: show the next phase once it ends.
            if j.yellow_until is not None:
                if t >= j.yellow_until:
                    j.current, j.yellow_until, j.green_since = j.next, None, t
                    traci.trafficlight.setRedYellowGreenState(tls, j.phases[j.current])
                continue
            if t - j.green_since < self.min_green or len(j.phases) < 2:
                continue
            # Read the lane counts once per step, and only if some junction is deciding.
            if count is None:
                count = {lane: traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes}
            pressures = [self._pressure(s, count) for s in j.served]
            best = max(range(len(pressures)), key=lambda k: pressures[k])
            if pressures[best] > pressures[j.current]:
                j.next, j.yellow_until = best, t + self.yellow
                traci.trafficlight.setRedYellowGreenState(
                    tls, self._transition(j.phases[j.current], j.phases[best]))
