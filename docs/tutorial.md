# Tutorial: running experiments

A hands-on guide in the order you would use the project. Every command is run from the
**repository root** unless it says otherwise. For the details of any function or option see the
[API reference](api.md).

| # | Tutorial | Time |
|---|---|---|
| 0 | [Set up and check your environment](#0-set-up-and-check-your-environment) | 5 min |
| 1 | [Your first simulation and its output files](#1-your-first-simulation) | 5 min |
| 2 | [Compare carma with fixed lights and stop signs](#2-compare-carma-with-fixed-lights-and-stop-signs) | 5 min |
| 3 | [Write your own bidding strategy](#3-write-your-own-bidding-strategy) | 10 min |
| 4 | [Persistent drivers and multi-day runs](#4-persistent-drivers-and-multi-day-runs) | 15 min + run |
| 5 | [Trips with different urgency](#5-trips-with-different-urgency) | 10 min + run |
| 6 | [More lanes per road](#6-more-lanes-per-road) | 10 min + run |
| 7 | [Learning to bid with regret minimisation](#7-learning-to-bid-with-regret-minimisation) | 20 min + run |
| 7b | [Learning to bid from repeated play (the learning study)](#7b-learning-to-bid-from-repeated-play) | 20 min + hours of runs |
| 8 | [Finding how much traffic a strategy can handle (demand sweep)](#8-demand-sweep) | 10 min + run |
| 9 | [Write your own study script](#9-write-your-own-study-script) | 15 min |
| 10 | [Troubleshooting](#10-troubleshooting) | |

Run times below were measured on an 18-core Mac with 16 workers.

---

## 0. Set up and check your environment

You need SUMO, the Python packages, and two environment variables' worth of setup (see
[Setup](../README.md#setup)):

```bash
export SUMO_HOME="$PWD/sumo"            # or wherever SUMO is installed
export PATH="$SUMO_HOME/bin:$PATH"
source .venv/bin/activate               # your virtual environment
pip install -r requirements.txt

sumo --version                          # prints the SUMO version
python -c "import traci, pandas, matplotlib; print('python ok')"
ls $SUMO_HOME/tools/randomTrips.py      # the trip generator must exist
```

Then run the smallest experiment: one fixed-light simulation of a 30-minute demand on the
single junction (about 15 seconds). It also generates the network and trips it needs:

```bash
python controller/lanes_study.py --lanes 1 --bases 3.0 --seeds 0 --configs default_lights --workers 1
```

You should see `1 runs on 1 workers.`, one line starting `default_lights_L1_b3.0_seed0:` with
`arrived=600`, and `Done.`. (If it says `0 runs`, the result already exists.) Lines reading
` Retrying in 1 seconds` are SUMO starting up and are harmless.

---

## 1. Your first simulation

Work from `controller/` for single runs, because the default paths are relative to it.

```python
# cd controller; python
import carma, config

info = carma.run(
    config.get_experiment("only_lane_urgency"),      # every car bids its waiting time, per lane
    gui=False, delay_ms=0, seed=0,
    net="../network/net/single_4way.net.xml",        # one 4-way junction, 1 lane per road
    routes="../simulation/single_2h/p_3.00.rou.xml", # a car every 3 s on average
    max_time=600,                                    # simulate 10 minutes
    label="demo", out_dir="/tmp/demo", quiet=True, save_history=False,
)
print(info["cars_arrived"], info["auctions"], info["jammed"])      # 179 59 False
```

(Use `gui=True` and `delay_ms=100` to watch it in `sumo-gui`.) Look at what was written:

```python
import pandas as pd
print(pd.read_csv("/tmp/demo/demo_summary.csv"))   # per bidder type: mean wait, time loss, wins...
cars = pd.read_csv("/tmp/demo/demo_cars.csv")      # one row per finished trip
print(cars[["vid", "travel_time", "wait_time", "time_loss", "final_carma"]].head())
print(cars["time_loss"].mean())                    # the headline delay measure
```

| File | What it is |
|---|---|
| `demo.csv` | One row per auction: who won and for how much. |
| `demo_cars.csv` | One row per finished trip. **The input of every analysis.** |
| `demo_summary.csv` | The per-bidder-type averages of the cars file. |
| `demo_run.csv` | How the run ended: cars arrived, backlog (`not_in_network`), jam, wall time. |

**The one number to focus on is `time_loss`**: seconds lost compared with driving the trip at full
speed. `wait_time` (seconds stopped) is also reported, but it flatters stop signs, where cars roll
slowly instead of stopping. Also always look at `not_in_network`, the cars that never finished:
delay is only measured over cars that arrived.

---

## 2. Compare carma with fixed lights and stop signs

The pattern for every comparison: run each strategy on the **same trips and seed**, then compare
`time_loss`. The lanes study script does exactly this and also builds the networks for you:

```bash
python controller/lanes_study.py --lanes 1 --bases 3.0 --seeds 0 1 \
    --configs default_lights stop_signs only_lane_urgency urg_coin --workers 8
python analyze/lanes_analysis.py
```

`--lanes 1` is a single lane per road, `--bases 3.0` is a car every 3 seconds (moderate load).
The analysis prints, for each load, a table of **urgency-weighted time lost** (lower is better)
with one row per strategy and one column per lane count. With two seeds we got:

```
=== urgency-weighted time lost, load base=3.0 (p = base/lanes) ===
lanes                    1
name
default_lights        97.6
only_lane_urgency     27.6
stop_signs            24.3
urg_coin              74.9
```

How to read it: the fixed 90 s light loses a lot at this load, carma (lane-level bidding) loses
about a quarter as much, stop signs are slightly better still, and a coin-flip auction (`urg_coin`)
is not much better than the lights, so the bidding is doing real work. (Urgency-weighted means a
second lost by an urgent trip counts six times; see [tutorial 5](#5-trips-with-different-urgency).)

---

## 3. Write your own bidding strategy

A strategy is any class with a method `amount(car, ctx)` that returns how much a car bids (never
more than `car.carma`). Save this as `controller/my_experiment.py`:

```python
import math
import pandas as pd
import carma, config
from auction import FirstPrice, HighestBid
from bidders import BidderType, group_by_lane
from config import Experiment


class SqrtWait:
    """Bid the square root of the waiting time (in seconds), capped at the balance."""

    def amount(self, car, ctx):
        return min(car.carma, math.sqrt(car.wait_time))


mine = Experiment(
    name="sqrt_wait",
    population=[BidderType("sqrt_wait", group_by_lane, SqrtWait(), sum)],   # who bids / how much / combine
    auction=HighestBid(),                                                    # highest group bid wins
    payment=FirstPrice(redistribute=True, free_if_uncontested=True),        # winner pays, losers share it
)

common = dict(gui=False, delay_ms=0, seed=0, net="../network/net/single_4way.net.xml",
              routes="../simulation/single_2h/p_3.00.rou.xml", max_time=1200,
              out_dir="/tmp/tut3", quiet=True, save_history=False)

for cfg in (mine, config.DEFAULT_LIGHTS, config.get_experiment("only_lane_urgency")):
    carma.run(cfg, label=cfg.name, **common)
    cars = pd.read_csv(f"/tmp/tut3/{cfg.name}_cars.csv")
    print(f"{cfg.name:20s} cars={len(cars):4d}  mean time lost = {cars['time_loss'].mean():5.1f} s")
```

```bash
cd controller && mkdir -p /tmp/tut3 && python my_experiment.py
```

```
sqrt_wait            cars= 380  mean time lost =  30.2 s
default_lights       cars= 370  mean time lost =  47.5 s
only_lane_urgency    cars= 379  mean time lost =  29.0 s
```

(20 simulated minutes; the seed makes the output reproducible.) The three pieces you can swap:

| Change this | To try... | Where |
|---|---|---|
| **Who bids together** (`group_by_lane`) | `group_by_vehicle`, `group_by_link`, `group_by_phase` | [`bidders`](api.md#groupers) |
| **How much** (`SqrtWait`) | any class with `amount(car, ctx)`; or `WaitTimeUrgency(k=2)`, `FixedFraction(0.1)` | [`strategies`](api.md#module-strategies) |
| **How group bids combine** (`sum`) | `max`, or `bidders.mean` | |
| **Who wins / pays** | `NonConflictingGreedy`, `SecondPrice`, `FirstPrice(redistribute=False)` | [`auction`](api.md#module-auction) |

To make it available to the study scripts by name, add it to `STUDY_CONFIGS` in
[config.py](../controller/config.py); then `--configs sqrt_wait` works everywhere.

---

## 4. Persistent drivers and multi-day runs

By default every trip starts with a fresh balance, so saving carma means nothing. With
`persistent_drivers=N`, N drivers keep their balance (and their bid history) from trip to trip.
A **multi-day chain** goes further: the same drivers repeat a simulated "day" with new random
trips, carrying their balances from one day to the next.

```python
from dataclasses import replace
cfg = replace(config.get_experiment("only_lane_urgency"), persistent_drivers=300)
carma.run(cfg, ..., out_dir="/tmp/persist")            # extra files: *_drivers.csv, *_carma_trace.csv
```

For the full multi-day experiment use the script:

```bash
python controller/multiday.py single --configs only_lane_urgency only_vehicle_budget \
    --p-values 3.0 --seeds 0 1 --days 7 --workers 8
python analyze/multiday_analysis.py
```

- One **chain** is (strategy, demand, seed) for 7 days; chains run in parallel, days run in order.
- **"Reset carma every k days"** is simulated: for every strategy the script also runs reset chains
  (`--resets`, default `3 5`) in which every driver goes back to the starting balance at the start
  of days 1, k+1, 2k+1, ... Days 1..k of a reset chain would be identical to the chain without
  resets, so only days k+1..7 are run (files `..._r<k>_day<d>_*`). A 7-day chain without resets
  is "reset every 7 days". Pass `--resets` with no values to skip them.
- Because a reset chain sees the same trips as the no-reset chain on every day, the analysis
  compares them **day by day** (paired by seed and day, after the first reset) rather than by
  averages.
- Checkpoints are written four times a day by default (`--trace-interval` sets the seconds
  between them): `*_carma_trace.csv` (mean, Gini, ... over all drivers), `*_balance_trace.csv`
  (every driver) and `*_network_trace.csv` (cars driving, share stopped, ...; every strategy,
  stop signs included).
- The analysis writes `multiday_single_days.png` (mean wait, mean time lost and carma Gini by
  day; reset schedules dashed), `multiday_single_carma_finegrain.png` (Gini over time) and the tables
  `results/analysis/multiday_single_by_age.csv` / `_by_reset.csv` / `_reset_paired.csv` (reset vs
  no reset, day by day) / `_paired.csv` (each strategy vs stop signs, day by day).
- The 5x5 version is `python controller/multiday.py grid ...` (slower: about 5 minutes per day).
- **Longer chains with fine checkpoints.** Two weeks, resets every 3, 5 and 7 days, a checkpoint
  every simulated minute, written to `results/multiday14/` so the 7-day results stay as they are:

  ```bash
  python controller/multiday.py grid --configs only_lane_urgency only_vehicle_budget --seeds 0 1 \
      --days 14 --resets 3 5 7 --trace-interval 60 --results-name multiday14 --workers 8
  python analyze/multiday_analysis.py --results multiday14 grid
  ```

  Besides the usual figures and tables (named `multiday14_grid_...`) this adds
  `multiday14_grid_timeseries.png` (carma Gini, top-10% share and largest balance, share of cars
  stopped and the mean wait and time lost of arriving trips, at every checkpoint over the two
  weeks) and `results/analysis/multiday14_grid_by_week.csv` (week 1 against week 2).
- Compare with stop signs on **time lost**, not wait. Wait only counts seconds below 0.1 m/s, and
  at a stop sign cars barely stop: on the grid stop signs wait 4-5 s per trip against 12-29 s for
  carma, yet `only_lane_urgency` loses *less* time than stop signs at p=1.0. The analysis prints
  both.

Check `guest_trips` in `*_run.csv`: it counts vehicles that found every driver busy. It should be
near 0; if not, raise the number of drivers (`SCENARIOS[...]["drivers"]` in
[multiday.py](../controller/multiday.py)).

---

## 5. Trips with different urgency

Real drivers differ in how much a second matters. Give the experiment an urgency draw:

```python
cfg = replace(cfg, urgency_levels=(1.0, 6.0), urgency_probs=(0.8, 0.2))   # 80% calm, 20% urgent
```

Every trip then gets `car.urgency` of 1 or 6. Urgency gives **no advantage in the auction**; it is
used by urgency-aware strategies (`UrgencyScaledWait`, `UrgencyFraction`, `RegretHedge`) and by
the **metric**: *cost = urgency x time lost*. Handing green to urgent trips lowers the cost.

The study compares urgency-aware and urgency-blind strategies with two benchmarks (`urg_omniscient`
always serves the most urgent car, `urg_coin` picks at random) and the baselines:

```bash
python controller/urgency_study.py --p-values 3.0 --seeds 0 1 2 --carma 100 --workers 12   # about 25 min
python analyze/urgency_analysis.py
```

The output has one table per demand level. Columns: `cost` (urgency x time lost),
`loss_urgent` / `loss_calm` (time lost by urgent and calm trips), `backlog`, and
`driver_cost_gini` (inequality of the drivers' average cost). The figure
`analyze/figures/urgency_time_lost.png` puts urgent and calm delay side by side per strategy.
Look for strategies where `loss_urgent` is clearly below `loss_calm`: those are the ones that
discriminate by urgency.

---

## 6. More lanes per road

[lanes_study.py](../controller/lanes_study.py) builds a single junction with 1 to 5 lanes per road
(`network/net/single_<N>lane.net.xml` plus a stop-sign twin), scales the demand with the lane count
(`p = base / lanes`), and uses an auction in which non-conflicting lanes win together.

```bash
python controller/lanes_study.py --workers 16                  # all lanes x 3 loads x 2 seeds x 8 strategies (238 runs, ~70 min)
python controller/lanes_study.py --lanes 3 --bases 2.5 --seeds 0 1 --workers 8   # one cell
python analyze/lanes_analysis.py                               # tables + analyze/figures/lanes_cost.png
```

Three variants of the lane bidder reduce the cost of switching the lights:

```bash
python controller/lanes_study.py --lanes 2 3 5 \
    --configs lane_fast_release lane_min_green phase_urgency --workers 16
python analyze/lanes_analysis.py
```

| Variant | What changes |
|---|---|
| `lane_fast_release` | Yellow 2 s; a grant ends once half of its winners have crossed. |
| `lane_min_green` | A grant lasts at least 10 s and is extended while its lanes are still queued (cap 30 s). |
| `phase_urgency` | Cars bid for whole phases (NS or EW, through or left) instead of lanes. |

Read the table: columns are lane counts, rows are strategies; compare each carma row with
`default_lights`, the adaptive signal baselines (`actuated_lights`, `delay_based_lights`,
`max_pressure`) and `stop_signs`. Backlog tables follow the cost tables; a strategy with a large
backlog is falling behind even if its delay looks small. The last tables pair every strategy with
`default_lights` and with `actuated_lights` seed by seed (`-35% 2/2` = 35% lower cost on average,
better on both seeds).

---

## 7. Learning to bid with regret minimisation

`RegretHedge` ([API](api.md#regretregrethedge)) learns a bid policy from the outcomes of past
auctions. Every driver bids a *fraction of their balance*; after each auction the learner scores
every alternative fraction on what it **would have** earned and shifts towards the ones that would
have done better. One number needs choosing: the **carma price** `lam0` (larger = spend more
reluctantly).

**Run the study** (5x5 grid, 1500 drivers, 7 days, urgency 1 / 6; about 40 min for the first
command and 35 min for the second on 16 workers):

```bash
python controller/regret_study.py --workers 16
python controller/regret_study.py --workers 16 \
    --configs regret_lane_4.0 regret_lane_8.0 frac_lane_0.9_0.15 frac_lane_0.9_0.05 \
              mix_learn20 mix_learn80
python analyze/regret_analysis.py
```

**What each configuration is for**

| Name | Question it answers |
|---|---|
| `regret_lane_<price>` | Does a learned policy beat hand-made ones, and which price is best? |
| `only_lane_urgency`, `urg_lane_wait` | The best hand-made strategies to beat. |
| `frac_lane_0.9_0.15`, `frac_lane_0.9_0.05` | **Control without learning**: urgent drivers bid 90% of their balance, calm 15% / 5%. Shows how much of the gain is just that policy shape. |
| `mix_learn20`, `mix_learn80` | 20% / 80% of drivers learn; the rest use lane urgency. Does a learner do better than the heuristic drivers it shares roads with? |
| `default_lights`, `actuated_lights`, `delay_based_lights`, `max_pressure`, `stop_signs` | Baselines in which nothing bids (fixed, actuated and delay-based lights, max-pressure, stop signs). They also use the persistent drivers, so their per-driver fairness is comparable. |

**Reading the output.** The script prints, for each demand level, the cost by day, the days 5-7
average (lower is better), and for the mixed runs the cost of each driver type. Then come the
**paired comparisons**: every strategy against `only_lane_urgency` and against `default_lights` on
the same seed and day (same trips), as a mean % difference with a 95% interval and the number of
days on which the strategy was better (e.g. `14/14`). Read these rather than the gaps between
averages: busy days are busy for every strategy, and pairing removes that. Then it draws
`analyze/figures/regret_days.png`.

**Look at what was learned.** Each learning run saves its policy:

```python
import pandas as pd
pol = pd.read_csv("results/regret/regret_lane_2.0_p1.00_s0_day7_policy.csv")
print(pol.pivot_table(index=["urgency", "wait_bucket"], columns="balance_bucket",
                      values="mean_fraction").round(2))
```

`mean_fraction` is the expected share of the balance that a driver in that situation bids.
Rows are urgency (1 calm, 6 urgent) and waiting-time bucket (0: under 10 s, 1: 10-30 s,
2: over 30 s); columns are the balance bucket (0 poor, 1 average, 2 rich). A learner that "gets
it" bids a high share when urgent and a low share when calm.

**Try your own variation.** Add an experiment at the bottom of [config.py](../controller/config.py):

```python
REGRET_CONFIGS["regret_lane_16.0"] = _regret("regret_lane_16.0", 16.0)   # a higher price
STUDY_CONFIGS["regret_lane_16.0"] = REGRET_CONFIGS["regret_lane_16.0"]
```

or change the learner itself (`gamma`, `eta0`, the candidate fractions `ACTIONS` in
[regret.py](../controller/regret.py)), then rerun with `--configs regret_lane_16.0 --seeds 0 1`.

**Two cautions.** The learner is a heuristic, not an equilibrium solution: the study does not test
that a driver cannot gain by deviating. And give each chain its own copy of the experiment
(`copy.deepcopy`), as `regret_study.py` does, or learners will share what they learn across runs.

---

## 7b. Learning to bid from repeated play

The learning study ([learning_study.py](../controller/learning_study.py), spec in
[learning_spec.md](learning_spec.md)) lets drivers who each minimise their own travel time learn
how to bid over many simulated days, and records what priority costs at every intersection while
they learn. Three learners are compared: `learn_br` (best response with a learned value of carma),
`learn_q` (Q-learning) and `learn_hedge` (the regret learner of tutorial 7). All use front-car
bidding: only the car at the head of each lane bids for it.

**Run it.** Phase by phase (the single intersection takes about an hour on 8 workers; the 3x3 a
few hours; the 5x5 several hours):

```bash
python controller/learning_study.py --scenarios single --seeds 0 1 2 --workers 8
python controller/learning_study.py --scenarios grid3 grid5 --seeds 0 1 2 --workers 8
```

Learning chains run at least 60 and at most 90 days. From day 60 a chain stops as soon as the
convergence rule holds (policy stable, no price drift and no week-on-week change at 90% of the
intersections, exploration finished, on 7 days in a row). Then it runs a 7-day evaluation block
with learning frozen, on trips every strategy shares. The baselines (`front_wait`,
`only_lane_urgency`, `actuated_lights`, `stop_signs`) run 7 days plus the evaluation block.

**Watch it while it runs.** Each chain keeps a status file and rebuilds its tables every day:

```bash
cat results/learning/grid5/learn_br_s0_status.json        # last day, converged?, failing tests
python analyze/learning_analysis.py --scenarios grid5 --no-gif
```

**Not converged at day 90?** Extend it; it continues from its saved state:

```bash
python controller/learning_study.py --scenarios grid5 --configs learn_br --seeds 1 --extend-to 120
```

**Is the result an equilibrium?** Freeze the population and let 10% of the drivers keep learning
against it for 14 days:

```bash
python controller/learning_study.py --scenarios grid5 --configs learn_br --mutants
python controller/learning_study.py --scenarios grid5 --configs learn_br --mutants \
    --mutant-start fresh --mutant-config learn_q            # newcomers with another algorithm
```

It prints the mutants' change in time lost against following the population (difference in
differences, paired by day). A gain that is not significantly below 0% (or within 2%) means the
learned policy is an approximate equilibrium.

**Reading the results.** `python analyze/learning_analysis.py` prints what drivers actually bid at
every intersection (bidders' balance, median / mean / 90% bid, share of zero bids, bid as a share
of the bidder's balance, the price, how often auctions are contested, payment per win, rebate per
loss) and writes figures to `analyze/figures/learning/` (PNG and PDF). Every amount is in carma:
everyone starts with 100 and the total never changes, so 100 is always the average balance. The
*price* of an intersection is the winning bid in its contested auctions (an uncontested winner
pays nothing).

Per scenario:
- `<scenario>_bids_by_intersection`: the mean bid at every intersection by how much carma the
  bidder holds, with the number in every cell, one panel per learner. Start here.
- `<scenario>_bid_distribution`: box plots of the bids made at every intersection, with the median
  price written above each box.
- `<scenario>_learners`: the price by day for every learner and seed, one panel per junction type,
  with the final values written at the line ends.
- `<scenario>_empirical_policy`: the mean bid by balance and intersections left.

Per learning chain (`<scenario>_<strategy>_s<seed>_...`):
- `_prices_grid`: one panel per intersection laid out like the network: the daily price (median
  winning bid, with 25-75% and 10-90% bands), the median bid (dashed), and the last-14-day numbers
  in a box in each panel. The figure for presenting how prices evolve.
- `_prices_weekly`: the median price at every intersection week by week, numbers in the cells.
- `_prices_map` (the network with the price written at every intersection on day 1, 10, 30 and the
  last day) and `_prices_map.gif` (every day: the map with numbers next to the price curves up to
  that day).
- `_within_day`, `_convergence` (the stopping-rule metrics), `_policy` (the learned bid by balance
  and intersections left, numbers in the cells, and the learned value of carma), `_belief`
  (learner A's expected rival bids against the real ones).

Plus `equilibrium_prices` across networks. The tables in `results/analysis/learning_*.csv` hold
the bids summary, the convergence summary, the price per intersection (with junction type and
load), the spread across seeds, the empirical policy and the outcomes of the evaluation block
paired against the baselines.

Every decision bid is in `*_bids.csv.gz` (kept out of git); the daily per-intersection summaries
in `*_prices.csv` and `*_prices_by_day.csv` are small.

**Trips with different urgency** (phase 2): add `--urgency 1 6 --urgency-probs 0.8 0.2
--results-name learning_urgency`.

**How much carma should drivers start with?** With fractional carma the starting amount only sets
the scale (every bid and payment scales with it). With whole units it matters: starting with 10, a
driver can only bid 0, 1, ..., 10. The starting-carma sweep on the 5x5 (about 5 hours on 16
workers):

```bash
python controller/learning_study.py --scenarios grid5 --seeds 0 1 2 \
    --configs learn_br front_wait only_lane_urgency \
    --carma-levels 10 20 40 80 160 320 640 --integer-carma --no-early-stop \
    --results-name learning_carma --workers 14
python controller/learning_study.py --scenarios grid5 --seeds 0 1 2 \
    --configs actuated_lights stop_signs --results-name learning_carma/baselines --workers 2
python analyze/carma_sweep_analysis.py                       # compare the levels
python analyze/learning_analysis.py --results learning_carma/c40 --scenarios grid5   # one level
```

`carma_sweep_prices` shows the price against starting carma, in carma and as a share of the
starting amount (a flat line there means only the scale changed); `carma_sweep_bids` the bids,
zero bids, ties and broke bidders; `carma_sweep_time_lost` travel times against actuated lights
and stop signs; `carma_sweep_by_intersection` the price at every intersection and level.

---

## 8. Demand sweep

How much traffic can a strategy handle before the network jams? Demand is the `-p` period of SUMO's
`randomTrips.py`: one vehicle every `p` seconds (smaller `p` = more traffic).

```bash
python controller/sweeps.py demand --configs default_lights only_lane_urgency --seeds 0 1 2 --workers 16
python controller/sweeps.py demand --p-values 0.65 0.55 --seeds 0 1 2     # refine around a jam point
python analyze/sweep_analysis.py demand
```

By default the sweep starts at `p = 1.0` and lowers `p` by 0.05 for every configuration (the
default lights and each strategy on its own) until it jams; a run counts as jammed when at least
100 cars are in the network and 90% of them are stopped for 300 s in a row, and jammed runs are
ended there, so they cost little. The analysis reports the heaviest demand each configuration
handled, delay and throughput against demand, and a heat map of the improvement over the default
lights (`D` = lights jammed but the strategy did not, `S` = the reverse, `J` = both, `*` = every
seed agrees). The starting-balance sweep is `python controller/sweeps.py carma --seeds 0 1 2`.

---

## 9. Write your own study script

All study scripts share one skeleton; copy
[urgency_study.py](../controller/urgency_study.py) (the shortest) and change three things.

```python
# 1. The trips: generate once with SUMO's randomTrips.py and reuse (make_routes).
# 2. run_task(task): build the Experiment and call carma.run in a worker process.
def run_task(task):
    os.chdir(CONTROLLER_DIR)                         # carma.py uses paths relative to controller/
    sys.path.insert(0, str(CONTROLLER_DIR))
    import carma, config                             # import inside the worker
    time.sleep(task["start_delay"])                  # stagger SUMO start-ups
    cfg = replace(config.get_experiment(task["experiment"]),
                  urgency_levels=(1.0, 6.0), urgency_probs=(0.8, 0.2),
                  persistent_drivers=300)
    return carma.run(cfg, gui=False, delay_ms=0, seed=task["seed"], routes=task["routes"],
                     net=NET_FILE, max_time=3600, label=task["label"], out_dir=str(RESULTS_DIR),
                     quiet=True, save_history=False,
                     extra_info={"experiment_name": task["experiment"], "p": task["p"]})

# 3. main(): one task per combination; skip finished ones; run in a process pool.
tasks = [{"experiment": n, "p": p, "seed": s, ...} for n in configs for p in ps for s in seeds]
tasks = [t for t in tasks if not (RESULTS_DIR / f"{t['label']}_run.csv").exists()]
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(run_task, t): t for t in tasks}
    for f in as_completed(futures): ...
```

Rules that keep a study correct:

- **Same trips for every strategy**, generated with a fixed seed.
- Put everything you vary in `extra_info`: it becomes a column of `_run.csv`, which the analysis
  scripts use to group runs.
- **Label** every run uniquely (strategy, demand, seed); the label is the file prefix and also how
  "already finished" is detected.
- Learning strategies and anything with state: `copy.deepcopy(cfg)` per run or chain.
- Use `--workers` no larger than the number of cores.

For the analysis side, read `*_run.csv` (one row per run) and `*_cars.csv` (one row per trip) with
pandas; [lanes_analysis.py](../analyze/lanes_analysis.py) is a 60-line template that groups by
the study's parameters, prints tables and saves a figure.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `KeyError: 'SUMO_HOME'` or `randomTrips failed` | `SUMO_HOME` is not set in this shell. `export SUMO_HOME="$PWD/sumo"`. |
| `FileNotFoundError: sumo` / `traci` cannot start SUMO | `sumo` is not on the `PATH`: `export PATH="$SUMO_HOME/bin:$PATH"`. |
| ` Retrying in 1 seconds` lines in the output | Normal: TraCI waiting for SUMO to open its port. |
| A batch printed `FAILED <label>: ...` | That run crashed; the others continued. Fix the cause and rerun the same command: only the missing runs execute. |
| Every run reports `0 runs` | All results already exist. Delete the `*_run.csv` files of the runs to redo (or pass `--rerun` where available). |
| Runs hang or use huge memory | Too many workers for the machine, or `save_history=True` on a long run. Lower `--workers`; keep `save_history=False` in studies. |
| Stray `sumo` processes after interrupting | `pkill sumo` (they hold ports). |
| `guest_trips` is large | Persistent drivers: fewer drivers than simultaneous vehicles. Raise the driver count. |
| Delay looks good but `not_in_network` is large | The strategy is falling behind; delay is only measured over finished cars. Judge it together with the backlog. |
| Results differ slightly between machines | SUMO versions differ. Within one machine a seed reproduces a run exactly. |
| A learning strategy gives the same result for every seed | Check the experiment is deep-copied per chain and that `seed` reaches `carma.run`. |

**Practical tips for long batches.** Keep the machine awake (`caffeinate -i`), start with
`nohup python controller/<study>.py ... > run.log 2>&1 &`, and watch progress with
`grep -c "_run.csv" <(ls results/<study>)` or `tail run.log`. Every study can be stopped at any
time and resumed by repeating the command.
