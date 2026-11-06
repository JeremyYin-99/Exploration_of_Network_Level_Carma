import os, sys
import traci

SUMO_HOME = os.environ["SUMO_HOME"]
sys.path.append(os.path.join(SUMO_HOME, "tools"))

NET = "Network/net/RandomGrid10_500_tls.net.xml"
ROUTES = "Simulation/grid_tls_routes.rou.xml"
MIN_GREEN, MAX_GREEN = 10, 45     # seconds
SWITCH_MARGIN = 2                  # other phase needs this many more halted cars

def phase_lanes(tls):
    """For each phase index, the set of incoming lanes that have green."""
    links = traci.trafficlight.getControlledLinks(tls)   # one entry per signal index
    logic = traci.trafficlight.getAllProgramLogics(tls)[0]
    result = {}
    for i, phase in enumerate(logic.phases):
        lanes = set()
        for idx, ch in enumerate(phase.state):
            if ch in "Gg" and links[idx]:
                lanes.add(links[idx][0][0])               # incoming lane id
        result[i] = lanes
    return logic, result

def is_green(state):
    return "y" not in state.lower()   # yellow phases contain y/Y

def main():
    traci.start(["sumo-gui", "-n", NET, "-r", ROUTES, "--start", "--delay", "100"])
    tls_ids = traci.trafficlight.getIDList()
    info = {t: phase_lanes(t) for t in tls_ids}
    last_switch = {t: 0.0 for t in tls_ids}

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        now = traci.simulation.getTime()
        for tls in tls_ids:
            logic, lanes_by_phase = info[tls]
            green_phases = [i for i, p in enumerate(logic.phases) if is_green(p.state) and lanes_by_phase[i]]
            if len(green_phases) < 2:
                continue                                    # nothing to decide
            cur = traci.trafficlight.getPhase(tls)
            if cur not in green_phases:
                continue                                    # in yellow: let it run
            held = now - last_switch[tls]
            if held < MIN_GREEN:
                continue
            queue = {i: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes_by_phase[i])
                     for i in green_phases}
            best = max(queue, key=queue.get)
            if held >= MAX_GREEN or (best != cur and queue[best] >= queue[cur] + SWITCH_MARGIN):
                traci.trafficlight.setPhase(tls, (cur + 1) % len(logic.phases))  # next phase = yellow
                last_switch[tls] = now
    traci.close()

if __name__ == "__main__":
    main()
