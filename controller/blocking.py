"""Detect signal links whose outgoing lane has no room, so they can be skipped.

A link is "blocked" when a queue has spilled back to the entrance of the lane it leads into:
a car that crossed the stop line would have nowhere to go and would just sit in the junction.
Giving green to a blocked link wastes the green and makes the jam worse.

Everything here uses the Car state that is already refreshed each step (lane, pos, speed), so it
makes no extra TraCI calls per step.
"""
import traci

# A car slower than this (m/s) counts as stopped when judging whether a lane entrance is blocked.
STOPPED_SPEED = 0.5


def build_out_lanes(tls_ids):
    """For each intersection, map signal index -> the lane a car on that link ends up on.

    Returns {tls_id: {link_index: out_lane_id}}. Computed once at the start of the run.
    """
    out_lanes = {}
    for tls in tls_ids:
        out_lanes[tls] = {}
        # getControlledLinks gives, for each signal index, a list of (inLane, outLane, viaLane).
        controlled_links = traci.trafficlight.getControlledLinks(tls)
        for index in range(len(controlled_links)):
            connections = controlled_links[index]
            if connections:
                out_lanes[tls][index] = connections[0][1]
    return out_lanes


def find_blocked_links(controlled, out_lanes, cars, entrance_dist):
    """Return {tls_id: set of blocked signal indices} for this step.

    A lane's entrance is blocked if some stopped car is within `entrance_dist` metres of the lane
    start (the queue has backed up to the junction). `entrance_dist` of None turns the check off.
    """
    blocked = {}
    for tls in controlled:
        blocked[tls] = set()
    # Check disabled: nothing is ever blocked.
    if entrance_dist is None:
        return blocked

    # Collect the lanes that currently have a stopped car close to their start.
    blocked_lanes = set()
    for car in cars.values():
        # Ignore cars inside a junction (their lane id is empty).
        if car.lane and car.pos <= entrance_dist and car.speed < STOPPED_SPEED:
            blocked_lanes.add(car.lane)

    # A link is blocked if the lane it leads into is one of those lanes.
    for tls in controlled:
        for index, out_lane in out_lanes[tls].items():
            if out_lane in blocked_lanes:
                blocked[tls].add(index)
    return blocked
