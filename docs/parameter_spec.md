# Parameter study: one protocol for every comparison

Status: ready to restart, 2026-11-20. The first `main` run (started 04:47) was stopped after the
U-turn problem was found (see "U-turns" below) and archived in `results/params/archive_uturn/`;
the study restarts from scratch on grids without U-turns, with the driver settings of "Drivers",
recalibrated demand and a passing smoke test. Start: `results/params/main/launch.sh 16`. Written before any results. Runner:
`controller/parameter_study.py`; results: `results/params/main/`.

## Why

The earlier studies each changed several things at once (network, load, number of drivers, days,
urgency, who bids, auction, payment rule, starting carma, and the baseline), so their results
cannot be put side by side. This study fixes one set of rules and one yardstick, and varies the
settings of interest from a reference cell.

## Questions

1. How do the network, lanes per road, demand, trips per driver, urgency mix, bidding unit and
   starting carma change what priority costs when drivers learn their bids (learner A)?
2. In each setting, how does carma compare with actuated lights and stop signs on delay, waiting
   time and fairness?

## Fixed rules (every carma chain)

- **Auction:** several winners per auction: bids are taken from highest to lowest and every bid
  compatible with all accepted bids also wins. Ties go to the longest-waiting group.
- **Which movements are compatible:** exactly those the network's own signal program shows green
  together (`conflicts.ConflictTable(mode="program")`). A permissive movement (a left turn that
  yields to the opposing straight, 'g' in the program) is shown as 'g' whenever one of its foes
  is green, so SUMO makes it yield, as under SUMO's own signals. See "Controller fix" below.
- **Payment:** first price in whole units of carma. A winner pays its bid only if it displaced a
  conflicting bid, i.e. some conflicting bid would have won had it not bid; otherwise it is free.
  Everything paid is shared among the losing deciders (ties of whole units to the longest wait).
  Total carma never changes.
  - Consequence: if two compatible winners each block the same loser, and removing either alone
    would not let the loser win, neither displaced it, so neither pays and the loser gets nothing.
- **What a bid has to beat** (used by the payment rule, the bid log and learner A): the highest
  bid that would have won a movement conflicting with it had it not bid
  (`auction.annotate_rivals`). With a single-winner auction this is the highest other bid, so the
  earlier studies are unchanged (checked: the 5x5 sweep's days 1-3 reproduce exactly).
- **Bidding units:** one decision per group, so nobody rides on someone else's bid.
  - vehicle: every car bids alone;
  - lane: the front car of each lane bids for the lane;
  - phase: the front car of each signal phase (NS or EW, through+right or left) bids for the phase.
- **Learner A** as in the learning study (docs/learning_spec.md), with whole units.
- **Persistent drivers:** a fixed pool; balances carry over from trip to trip and day to day; no
  resets.
- **Urgency:** low = 1, high = 6.
- **Trips:** randomTrips on each network, cars enter on the best lane; trip seed
  1000 x seed + day for learning days and 100000 + 1000 x seed + e for evaluation day e.
- **Day:** 1800 s of trip starts, then up to 900 s for the network to empty.
- **Simulator:** SUMO through libsumo (identical results to the TraCI socket, about 40x faster).
- **Drivers:** SUMO's default car with two lane-change settings, `lcStrategic="4"` and
  `lcKeepRight="0"` (`network/drivers/lane_change.add.xml`), for every strategy; see "Drivers".

## Settings varied

| Setting | Levels | Reference |
|---|---|---|
| Network | single intersection, 3x3 grid, 5x5 grid (200 m blocks, no U-turns) | 3x3 |
| Lanes per road | 1, 2, 3 | 2 |
| Demand | light, moderate, heavy (calibrated, below) | moderate |
| Trips per driver per day | 2, 6, 10 (drivers = trips per day / this) | 6 |
| Urgency mix (low-high share) | 20-80, 40-60, 50-50, 60-40, 80-20 | 80-20 |
| Urgency belongs to | each trip (drawn per trip) or each driver (fixed for good, exact shares) | trip |
| Bidding unit | vehicle, lane, phase | lane |
| Starting carma | 5, 10, 20, 40, 80, 160, 320 | 80 |
| Green extension | off, on (keep green while cars approach on served lanes, up to 30 s) | off |
| Driver model (check only) | study drivers (lcStrategic 4, lcKeepRight 0), SUMO default; reference cell at 2-3 lanes | study drivers |

640 is left out: in the 5x5 sweep it gave the same price (% of starting carma) as 320.

**Demand.** Calibrated per network and lane count (`--plan calibrate`: actuated lights, one day,
11 demand levels x 3 trip seeds; `demand_levels()` writes `results/params/calibrate/demand.json`)
so that actuated lights stop the same share of cars at each level: 8.76%, 11.06% and 13.10%
(`CALIBRATION_TARGETS`), the shares of the earlier 5x5 one-lane network (with U-turns) at
p = 1.25, 1.0 and 0.9, so moderate and heavy keep the congestion of the earlier grid studies.
Dividing by the lane count does not keep congestion equal: at 3 lanes it doubles the stopped
share, because turning movements don't gain lanes; 3 lanes carry about the same traffic as 2 at
equal congestion. Levels on the no-U-turn grids with the study drivers (recalibrated
2026-11-20); no teleports at any of these levels:

| Network | Lanes | light p (trips) | moderate p (trips) | heavy p (trips) | actuated time lost, near moderate |
|---|---|---|---|---|---|
| single | 1 | 3.52 (512) | 3.18 (567) | 2.99 (601) | 11.7 s |
| single | 2 | 2.06 (873) | 1.79 (1004) | 1.66 (1086) | 11.1 s |
| single | 3 | 2.06 (872) | 1.75 (1027) | 1.61 (1115) | 11.0 s |
| 3x3 | 1 | 2.66 (677) | 2.29 (787) | 2.18 (825) | 30.3 s |
| 3x3 | 2 | 1.39 (1294) | 1.19 (1512) | 1.10 (1643) | 30.9 s |
| 3x3 | 3 | 1.42 (1264) | 1.21 (1486) | 1.09 (1654) | 27.5 s |
| 5x5 | 1 | 1.29 (1400) | 1.05 (1709) | 0.96 (1878) | 36.4 s |
| 5x5 | 2 | 0.79 (2287) | 0.62 (2900) | 0.55 (3286) | 36.5 s |
| 5x5 | 3 | 0.82 (2208) | 0.62 (2907) | 0.54 (3348) | 36.0 s |

## Strategies and the yardstick

- learner A, learning then frozen;
- baselines: actuated lights and all-way stop signs (fixed-time lights are dropped);
- optional: the wait rule (bid 1 carma per second waited, capped at the balance; it ignores
  urgency: `strategies.WaitTimeUrgency(1.0)`) with the same bidding unit, to show what
  learning adds.

After learning, every strategy plays the same **evaluation week** (7 days, reserved trips, same
drivers and urgencies), and is compared paired by (seed, evaluation day) with actuated lights.
The baselines ignore the bidding unit and the starting carma, so they are run once per
(network, lanes, demand, trips per driver, urgency) and shared.

## Seeds and learning length

10 seeds per cell (decided 2026-11-20).


Learner A did not settle in 240 days with urgency off (learning_spec 8e). Here every learner
chain runs **300 days** and the settling day is found afterwards, not by a stopping rule
(per-chain block means are too noisy to stop on):

- price = median winning bid of contested winners, % of starting carma, mean over intersections;
- 30-day block means per chain, change from the previous block paired within chain, 95% interval
  across seeds;
- **settled** at block k if the changes into blocks k and k+1 both have intervals that include 0
  and means within 3% of the price. Prices and delays are then reported from the settled blocks;
- a cell not settled by day 300 is extended in steps of 60 days (the runner resumes) up to 480,
  and reported as not settled if it is still moving then.

## Measured

Per cell (learning blocks and evaluation week):

- time lost and waiting time per trip, and urgency-weighted cost;
- for urgent and calm trips separately;
- price per junction type and per intersection;
- share of contested auctions;
- Gini of per-driver cost over the week, and of balances;
- guest trips (departures with no idle driver; should be 0);
- teleports per day (cars stuck 300 s or in a collision; should be about 0).

## Plans

- `calibrate`: above.
- `smoke`: every bidding unit at 1-3 lanes on the 3x3, the single intersection, the 5x5 with 3
  lanes, per-driver urgency, green extension and 5 carma; 3 days + the evaluation week, seed 0.
- `main` (what is run; decided 2026-11-20 after libsumo made runs about 40x faster): for every
  network configuration (single, 3x3, 5x5 x 1-3 lanes), its own reference cell (moderate demand,
  6 trips per driver, 80-20 urgency per trip, lane bids, no extension, 80 carma) and, from it:
  bidding unit x green extension (6 cells), demand x trips per driver (9), urgency mix per trip
  (5) and per driver (5), starting carma (7), and at 2-3 lanes the reference cell with SUMO's
  default drivers (1). 29-30 cells per configuration, 267 in all, 10 seeds each. Strategies: learner A (300 days + evaluation week), the wait rule (7 days + evaluation
  week), actuated lights and stop signs (evaluation week). Run in three phases (3x3, single,
  5x5) by `results/params/main/run_main.sh`; logs in `results/params/main/logs/`.
- `quick` (run 2026-11-20 16:30, about an hour on 16 workers, when results were needed fast;
  `results/params/quick/`). Learner A learns for **60 days** (+ the evaluation week): in the
  archived 300-day 3x3 run, time lost over days 31-60 is within 1% of days 271-300 in every
  cell (contested share unchanged), while the price is still about 20% above its day-300 level
  (median over 87 cells; 7-33%), with the same pattern across settings. So delays are final at
  60 days and prices are an upper bound; the archive gives the long run. Cells (99, 1,690 chains):
  - A: every network configuration's reference cell, all four strategies, 10 seeds;
  - B: the reference cell with SUMO's default drivers at 2-3 lanes, wait rule and baselines,
    10 seeds;
  - C: every sweep on the 3x3 with 1 lane (starting carma, urgency per trip and per driver,
    demand x trips per driver, bidding unit x green extension), learner and wait rule (and
    baselines where they differ), 5 seeds;
  - D: bidding unit x green extension on the other 8 configurations, wait rule, 5 seeds;
  - E: light and heavy demand (6 trips per driver) on the other 8, wait rule and baselines,
    10 seeds;
  - F: the learner too for D at 2 lanes on the single intersection and the 5x5, 5 seeds.
- Salvaged from the archived run (`analyze/salvage_archive.py`, `analyze/archive_figures.py`):
  learner A's price and time lost over 300 days in all 87 3x3 cells (U-turn grids, SUMO default
  drivers, no baselines), `results/analysis/salvage/`, `analyze/figures/params/archive/`.
- Disk: ordinary learning days keep only run, summary, prices and bid log; every 30th day, the
  last 30 days and the evaluation week keep every file; learner state and balances every 10th
  day.

## Controller fix (2026-11-20, before any `main` run)

The first smoke test showed carma breaking down from 2 lanes up and with phase bids (3x3,
moderate: 15 s lost per trip at 1 lane, 58-100 s at 2 lanes, 150-260 s at 3 lanes with
teleports; phase bids jammed even at 1 lane). The earlier lanes study had seen the same on the
single intersection. Causes found:

1. The conflict table forbade any two crossing movements, including a left turn with the
   opposing straight. SUMO's own programs run both opposing approaches together with permissive
   lefts ('g'), so actuated lights had about twice carma's usable green on multi-lane roads.
2. A phase bid contains both opposing left turns. Shown as 'G' (priority) together, foes could
   block each other inside the junction: the phase-bid jams.

Fix: compatible = green together in the network's program, and permissive movements shown as
'g' when a foe is green (both off by default, so earlier studies are unchanged). One day, wait
rule, time lost per trip, old table -> new table (actuated lights):

| | single 2 lanes | single 3 lanes | 3x3 1 lane | 3x3 2 lanes | 3x3 3 lanes |
|---|---|---|---|---|---|
| vehicle bids | 112 -> 26 | 86 -> 47 | 15 -> 16 | 226 -> 43 | 257 -> 71 |
| lane bids | 48 -> 28 | 61 -> 58 | 16 -> 15 | 53 -> 44 | 130 -> 79 |
| phase bids | 156 -> 17 | 57 -> 28 | 303 -> 15 | 200 -> 31 | 110 -> 51 |
| actuated lights | 15 | 43 | 24 | 25 | 51 |

No jams or teleports remain. Lane and vehicle bids still lose capacity on multi-lane roads,
because a grant serves only the cars present when it was won. Keeping green while cars keep
arriving (green extension) recovers part of it (e.g. lane bids, single 3 lanes 58 -> 33 s) but
costs a little at 1 lane and for phase bids, so it is a setting of the study, not part of the
fix. Lane changes after winning also make about 1-2% of grants run to the 30 s cap.

## U-turns (2026-11-20, restart)

The first `main` run had teleports on the grids: SUMO removes a car that has been stuck for 300 s
or has collided, and puts it back further along its route. A scan of the 264,652 finished days
(3x3 only) found two kinds:

- **Collisions**, about one a day: on 11% of days at 2 lanes, 5% at 3 lanes, under 1% at 1 lane.
- **Gridlocks**: 15 days with 20-72 teleports and 100-580 cars still in the network at the end
  (11 at 1 lane), one of them in an evaluation week.

Replaying recorded days exactly with SUMO's warnings on (they were off in the runs) showed both
start with a **U-turn**. `netgenerate --grid` builds a U-turn at every junction, and 28% of
random trips on the 3x3 (20% on the 5x5, none on the single intersection) used one, because a
trip that starts facing away from its destination is fastest with a U-turn at the next junction.

- Collision: a U-turning car on a permissive green ('g') pulls into the same exit lane as an
  oncoming straight car on a full green, 2 s ahead of it, and is hit. The network's priority
  rules say the U-turn must yield; SUMO's U-turn waiting point does not cover the merge.
- Gridlock: a U-turning car and a left-turning car stop inside a junction, both waiting to merge
  into the same exit. The controller switches back to the crossing direction within about 10 s,
  a straight car enters on a full green, and the three block each other. Queues then spread to
  the neighbouring junctions and cars waiting at permissive turns time out one after another.

Under actuated lights the calibration had no teleports at the demand levels used.

**Fix:** the grids are rebuilt with the same netgenerate options plus `--no-turnarounds true`
(`network/net/{grid3,grid5}_200_L{1,2,3}_noU_{tls,stop}.net.xml`; `GRID_VARIANT` in the runner).
Cars that would have U-turned now go round a block. The earlier networks are kept, so the
earlier studies still reproduce (they share the U-turn issue: 20% of 5x5 trips).

A/B on the 3x3 (1-3 lanes, moderate and heavy demand, 30 trip seeds, one day each, fresh
drivers): with U-turns vs without.

- Carma (wait rule, lane and phase bids): collisions 22 -> 5 in 240 days; no jams, no cars left
  at the end. Emergency braking about halves at 2 lanes, unchanged at 3 lanes with phase bids.
- Actuated lights and stop signs: no teleports either way.
- Delay rises for every strategy (detours; e.g. actuated, 1 lane, moderate: 25.3 -> 34.6 s), so
  demand was recalibrated. Carma's ratio to actuated lights barely changes (1 lane 0.64 -> 0.60,
  2 lanes 1.56 -> 1.58, wait rule with lane bids).

**What remained** after removing U-turns: rear-end collisions on about 5% of multi-lane carma days
in the smoke test (5 of 24 days on the 5x5 with 3 lanes). Every replayed case (5, with SUMO's
lane-change log) was the same: carma showed one lane's movement green and the next lane's red, and
SUMO's lane-change model moved a car sideways from the green lane into the red one 6-15 m before
the stop line at 11-16 m/s ("strategic|urgent" or "keepRight"); it made an emergency stop and the
car behind hit it. SUMO's own programs always show both lanes of a movement the same colour, so
this needs lane or vehicle bids. Fixed by the driver settings below.

## Drivers (2026-11-20)

Two SUMO lane-change settings, for every strategy alike (`network/drivers/lane_change.add.xml`,
loaded with `-a` before the routes; `Experiment.driver_file`):

- `lcStrategic="4"` (default 1): move to the lane the route needs earlier, not at the stop line.
- `lcKeepRight="0"` (default 1): no keep-right lane changes (a European motorway habit; it caused
  one of the replayed collisions).

A/B on the no-U-turn grids (30 trip seeds, one day each, fresh drivers, moderate demand):

| | carma, wait rule, lane bids | actuated lights | stop signs |
|---|---|---|---|
| 3x3, 2 lanes: time lost, default -> new | 45.6 -> 42.8 s | 28.4 -> 29.8 s | 49.9 -> 51.3 s |
| 3x3, 3 lanes | 47.4 -> 36.7 s | 27.0 -> 28.1 s | 45.9 -> 47.1 s |
| 5x5, 3 lanes | 68.9 -> 49.2 s | 35.2 -> 37.2 s | 70.3 -> 72.1 s |
| collisions / red-light emergency stops (all three) | 2 / 52 -> 0 / 2 | 0 / 0 both | 0 / 0 both |

Either setting alone left most of the emergency stops. The settings help carma a lot and cost the
baselines 1-2 s, so carma's multi-lane gap to actuated lights shrinks (5x5 3 lanes: 1.96x ->
1.32x): with the defaults, part of carma's multi-lane loss was SUMO drivers swerving into red
lanes. Because this choice moves a headline comparison, each network's reference cell at 2-3
lanes is also run with SUMO's default drivers (`drivers="sumo"`, cell names ending `_sumodrv`,
same demand), and the analysis reports both (`params_main_drivers.csv`). Demand was recalibrated
with the new drivers. Teleports are counted per day in `_run.csv` and reported.

**Residual:** in the smoke test with these drivers, 4 of about 260 multi-lane carma days (1 in 65)
had one rear-end collision (3 with SUMO's defaults on 48 days). Replayed: a car changes lanes
6-15 m before the stop line at 6-8 m/s ("speedGain" or "strategic|urgent"), from a lane stopped
at red into the next lane flowing at green, in front of a car that cannot stop. Removing it
would mean switching off speed-gain lane changes (distorting normal multi-lane driving) or
halving SUMO's 1 s step (twice the run time), so it is left and reported: one car teleported a
few hundred metres, out of 1,500-2,900 a day.

## Analysis and figures

One script, `analyze/parameter_analysis.py`, reads one plan's folder and makes every figure with
one colour per strategy group and the same strategy order everywhere:

1. price per setting (one panel per setting, levels on x), learned bids;
2. % difference from actuated lights in time lost and cost, per setting, with 95% intervals;
3. time lost for urgent and calm trips;
4. Gini against mean cost;
5. demand x trips per driver and bidding unit x lanes as heat maps;
6. price over days for the reference cell (settling).
