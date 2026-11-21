# Exploration_of_Network_Level_Carma

A [SUMO](https://eclipse.dev/sumo/) simulation in which vehicles bid **carma** (a non-monetary
credit) for green time at signalised intersections. The traffic lights are controlled from Python
through TraCI: when a bid wins, its movement is turned green and every other signal at that
intersection stays red. The project asks: *does bidding with carma beat ordinary traffic lights
and stop signs, when does it, and can drivers learn to spend their carma better?* It is modelled
on the CARMA / karma papers (Elokda et al., "A self-contained karma economy for the dynamic
allocation of common resources"; "Dynamic population game" formulation).

The code is organised so that **who bids**, **how much they bid**, **who wins**, **who pays**
and **how the lights respond** are separate, swappable pieces. Trying a new bidding structure
means building a new `Experiment` (see [config.py](controller/config.py)), not editing the
simulation loop.

## Myopic Behaviors

CAV agents will need to repeatedly face each other and make decision on the next optimal bid based on their past choices.

---

## Documentation

| | |
|---|---|
| **[Tutorial](docs/tutorial.md)** | Step-by-step guide: set up, run a first simulation, compare strategies, write your own strategy, run every study (persistent drivers and multi-day runs, urgency, lanes, regret minimisation, demand sweeps), write your own study, troubleshooting. |
| **[API reference](docs/api.md)** | Every public function and class in the style of the NumPy / pandas docs: signature, parameters, returns, notes and runnable examples. Also the command-line reference of every study and analysis script. |
| This README | Overview, setup, the experiment map, results so far, and the concepts behind the code. |

## Contents

- [Documentation](#documentation)
- [Setup](#setup)
- [Quick start: one simulation](#quick-start-one-simulation)
- [Running the experiments at a glance](#running-the-experiments-at-a-glance)
- [What the studies have found so far](#what-the-studies-have-found-so-far)
- [Project structure](#project-structure)
- [How one simulation step works](#how-one-simulation-step-works)
- [Named experiments](#named-experiments)
- [Persistent drivers, urgency and learning](#persistent-drivers-urgency-and-learning)
- [Analysing results](#analysing-results)
- [Mixing bidding types](#mixing-bidding-types)
- [Extending the project](#extending-the-project)
- [Output files](#output-files)
- [Known limitations and to-do](#known-limitations-and-to-do)

---

## Setup

1. **SUMO.** Install SUMO, or use the copy in [sumo/](sumo/). Two things must be true in the shell
   you run from:
   - `SUMO_HOME` points at the SUMO folder (the trip generator `randomTrips.py` is found through
     it), and
   - the `sumo` and `netconvert` programs are on the `PATH`.

   ```bash
   export SUMO_HOME="$PWD/sumo"          # from the repo root; adjust if SUMO is installed elsewhere
   export PATH="$SUMO_HOME/bin:$PATH"
   sumo --version                        # should print the version
   ```
2. **Python packages** (a virtual environment is recommended):

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt       # traci, pandas, numpy, matplotlib, ...
   ```
3. **Where to run from.** Every study script and analysis script can be started from the repo root
   (`python controller/lanes_study.py ...`); they find their folders relative to their own file.
   The single-run example below is the exception: it uses relative network paths, so run it from
   `controller/`.

---

## Quick start: one simulation

```bash
cd controller

# Baseline experiment with the SUMO GUI
python3 carma.py

# Any named experiment, headless and fast
python3 -c "import carma, config; carma.run(config.LANE_SUM, gui=False, delay_ms=0)"
```

`run(cfg, gui, delay_ms, seed, ...)` takes an `Experiment`, whether to open the GUI, the GUI delay
per step in milliseconds, and the SUMO random seed. Optional arguments:

| Argument | Meaning |
|---|---|
| `routes`, `net` | Another trip file / network file (defaults: the 5x5 grid and its 7,200 s of trips). |
| `max_time` | Stop at this simulated time (cars still driving are not in the results). |
| `label`, `out_dir` | File-name prefix and folder for the results. |
| `quiet` | Hide SUMO's step log (use when running many in parallel). |
| `save_history` | Also save the full bid history (large). |
| `jam_stop` | End the run if the network gridlocks. |
| `start_balances`, `pool_seed` | Persistent-driver runs: balances carried from a previous day, and the seed that fixes who the drivers are (see [multiday.py](controller/multiday.py)). |
| `extra_info` | Extra columns for the one-row `_run.csv` summary. |

`run` returns a dict describing how the run ended. The default full run (5x5 grid, 7,200 vehicles)
takes a few minutes headless. Results are written to `results/` (see [Output files](#output-files)).

---

## Running the experiments at a glance

### The common recipe

Every study is a script in [controller/](controller/) that builds a list of simulations, runs them
in parallel (one SUMO process per worker) and writes one set of files per simulation. A matching
script in [analyze/](analyze/) reads those files and prints tables and saves figures. The
workflow is always:

```bash
export SUMO_HOME="$PWD/sumo"; export PATH="$SUMO_HOME/bin:$PATH"   # once per shell
python controller/<study>.py --workers 16                          # 1. run  (long)
python analyze/<study>_analysis.py                                 # 2. analyse (seconds)
```

Things that apply to all of them:

- **`--workers N`** is the number of simulations running at once. The default is the number of CPU
  cores minus two. SUMO is single-threaded, so more workers than cores does not help.
- **Finished runs are skipped.** A run is finished when its `*_run.csv` file exists, so an
  interrupted batch is resumed by running the same command again. To redo runs, delete their files
  (or pass `--rerun` where offered).
- **Every strategy sees the same trips.** The trip files are generated once (with a fixed seed) and
  reused, and trip urgencies are drawn from a generator seeded by the run, so differences between
  strategies are not caused by different traffic.
- **Seeds.** `--seeds 0 1 2` repeats everything with different seeds (different random drivers,
  different tie-breaks). Report averages over seeds; a single run is one sample.
- **Long runs on a laptop.** Keep the machine awake (`caffeinate -i` on a Mac) and start the batch
  with `nohup python ... > run.log 2>&1 &` so it survives closing the terminal. The log prints one
  line per finished run.
- **Load.** Demand is set by `p`, the average seconds between inserted vehicles: *smaller `p` means
  more traffic*.

### Which study answers which question

The [tutorial](docs/tutorial.md) walks through each of them with commands and how to read the output; the options of every script are in the [API reference](docs/api.md#command-line-scripts).

| Study (script) | Question | Network | Runs (default) | Results folder | Analysis |
|---|---|---|---|---|---|
| [sweeps.py](controller/sweeps.py) `demand` | How much traffic does each strategy handle before the network jams, compared with SUMO's default lights? | 5x5 grid | one per strategy x demand x seed | `results/sweeps/demand/` | `sweep_analysis.py` |
| [sweeps.py](controller/sweeps.py) `carma` | Does the starting carma balance matter? | 5x5 grid | one per balance x seed | `results/sweeps/initial_carma/` | `sweep_analysis.py` |
| [persistence.py](controller/persistence.py) | Does carrying carma from trip to trip (persistent drivers) change anything at one intersection? | 1 intersection, 4 h | strategy x mode x demand x seed | `results/persistence/` | `stop_sign_analysis.py` |
| [multiday.py](controller/multiday.py) | Does carma carry over days, and what happens if carma is reset every 3 or 5 days (simulated as separate chains, compared day by day with the no-reset chain)? | 1 intersection or 5x5 | chains of 7 days (14 with `--days 14`) | `results/multiday/{single,grid}/` (`results/<name>/` with `--results-name`) | `multiday_analysis.py` |
| [urgency_study.py](controller/urgency_study.py) | When trips differ in urgency (value of time), do urgency-aware bidders beat urgency-blind ones? | 1 intersection, 2 h | strategy x demand x seed x starting balance | `results/urgency/` | `urgency_analysis.py` |
| [lanes_study.py](controller/lanes_study.py) | Does the comparison with lights and stop signs change with 1 to 5 lanes per road, and can switching overhead be cut? | 1 intersection, 1-5 lanes | lanes x load x seed x strategy | `results/lanes/` | `lanes_analysis.py` |
| [regret_study.py](controller/regret_study.py) | Can drivers *learn* a better way to spend carma (regret minimisation)? | 5x5 grid, 7-day chains | strategy x demand x seed chains | `results/regret/` | `regret_analysis.py` |
| [learning_study.py](controller/learning_study.py) | Do self-interested drivers who learn from repeated play converge on a bidding policy, and what does priority then cost at each intersection? Is it an equilibrium? ([spec](docs/learning_spec.md)) | 1 intersection, 3x3, 5x5 | chains of 60-90 days (stop on convergence, extendable), then a frozen evaluation week; exploitability test with `--mutants` | `results/learning/{single,grid3,grid5}/` | `learning_analysis.py` |

---

## What the studies have found so far

"Cost" is urgency x time lost (lower is better); "time lost" is unweighted. Differences are
**paired**: each strategy is compared with a reference on the same seed (and day), and given as
the mean % difference, its 95% interval and the number of seeds or days on which it was better
(see [Analysing results](#analysing-results)). The multi-day studies have 14-24 paired days per
comparison and detect differences of a few percent; the single-run studies (urgency, lanes) have
only 2-3 paired seeds. Tables: `results/analysis/*_paired.csv`.

**Signal baselines.** Fixed lights (`default_lights`, SUMO's 90 s program) are a weak baseline.
SUMO's actuated control over the same phases (`actuated_lights`) cuts time lost by 58% on the 5x5
grid and by 70% at the single intersection (p=3.0). It is the strongest signal baseline on the
grid and at moderate single-intersection load; at heavier single-intersection loads max-pressure
(`max_pressure`, our implementation) or delay-based control is sometimes a few percent better. Results that say carma "beats the lights" held only against fixed lights.

| Study | Finding |
|---|---|
| Demand sweep (5x5) | Several carma strategies handle more traffic than fixed 90 s lights before jamming; stop signs handle moderate load best at one lane per road. *Not yet re-run against the adaptive signal baselines.* |
| Persistence (1 intersection) | Carrying carma between trips changes little for the strategies tried: the bids are capped by waiting time, so the balance rarely binds. |
| Urgency (1 intersection, 3 seeds) | Lane-level carma beats fixed lights at p=3.0 (cost 62-66 vs 119, -45%, 3/3 seeds) but loses to every adaptive signal: actuated lights 36, max-pressure 41, delay-based 46 (carma worse than actuated on 3/3 seeds), and to stop signs (24). At p=2.5 every carma strategy is at least 58% worse than fixed lights. Urgency awareness helps only a little at lane level (about 6%). Vehicle-level auctions are worse than fixed lights; the omniscient benchmark at vehicle level is not a true ceiling. Carma spreads cost more evenly across drivers (per-driver cost Gini 0.20-0.24 vs 0.29-0.35 for the signal baselines). |
| Lanes (1 intersection, 1-5 lanes, 2 seeds) | Stop signs collapse as lanes are added (they serve one vehicle at a time). At one lane and light load (base 3.0) lane-level carma ties actuated lights (cost 25-28 vs 29); from two lanes up, or at heavier load, the adaptive signals match or beat every carma variant. A minimum-green rule with queue extension recovers part of the gap; early release and phase auctions do not help. At 4-5 lanes and heavy load every controller saturates. |
| Regret minimisation (5x5, 7-day chains, 14 paired days per demand) | The learned policy (Hedge over bid fractions) beats the best hand-made bidder (`only_lane_urgency`) on every day: -7.5% / -8.0% cost at carma price 2 and -12.7% / -13.4% at price 8 (p=0.9 / p=1.0; 14/14 days; 95% intervals within about +-1.5 points). Its total time lost is the same as every other carma strategy (all within ~1 s), so the gain is redistribution: urgent drivers lose 26% less time, calm drivers 6-8% more. Against the signal baselines it depends on load: at p=1.0 it beats actuated lights by 10% (-13.9 to -6.1%, 14/14 days) with statistically equal time lost (+3%, interval includes 0) and beats stop signs by 20%; at p=0.9 actuated lights beat it by 12% (0/14 days; the learner loses 29% more time) and stop signs by 5% (5/14 days). It has the most even per-driver cost of all strategies (Gini 0.43-0.44 vs 0.49 for fixed lights). |
| Regret controls | A fixed rule with the same shape (urgent drivers bid 90% of their balance, calm ones 5%) matches the learner at price 2 (+0.7% / +0.2%, intervals include 0), so most of the gain comes from the *shape* of the policy; higher prices add about 5 more points. In mixed populations a learner does about 7% better than the heuristic drivers it shares roads with (20% learners), and a heuristic driver among learners does about 30% worse (80% learners). |
| Multi-day resets (7-day chains; 1 intersection 3 seeds, 5x5 2 seeds) | Resetting every driver's carma every 3 or 5 days (simulated, compared day by day with no reset on the same trips) leaves waiting time unchanged (within about +-1%, intervals include 0, except +3.6% for `only_vehicle_budget` on the grid at p=1.0, 4 days). It does cut balance inequality for `only_lane_urgency`, whose Gini otherwise drifts upward over days without affecting delay (end-of-day Gini -0.05 to -0.17, every day), but not for the strategies that bid a share of their balance (`only_vehicle_budget`, `only_vehicle_random`), whose Gini settles within a day. |
| Multi-day, 14 days (60 s checkpoints; resets every 3, 5, 7 days; `results/multiday14`) | Two weeks change nothing about delay: there is no trend in wait or time lost for any strategy (week 2 is within week 1's day-to-day spread), and the gap to stop signs is the same in both weeks (days 1-7 are identical, car for car, to the 7-day runs). The vehicle-level bidders' Gini is flat after day 1 (single) or day 3-5 (grid). `only_lane_urgency`'s Gini keeps rising but slows down: at the single intersection it levels off near 0.44 at p=2.0 (+0.037 per day in week 1, +0.004 in week 2), at p=3.0 it is still rising (0.22 -> 0.36 week means); on the grid it goes 0.12 -> 0.22 (p=0.9) and 0.08 -> 0.14 (p=1.0) with the daily rise halving. This drift is inert: on the grid at p=1.0 no `only_lane_urgency` driver ever runs out (lowest balance 3.4), so its waiting-time bids are never capped and resets change *nothing* (0.0% on all 54 paired days); at p=0.9 at most 7 of 1500 drivers hit zero (resets: within 0.3%). Resets lower `only_lane_urgency`'s Gini (-0.07 to -0.18, every day) and leave the vehicle-level bidders' Gini unchanged. At the single intersection no reset schedule moves wait or time lost by more than about 1% (most intervals include 0); the largest effect anywhere is `only_vehicle_budget` losing 3% more time with resets on the grid at p=1.0 (intervals exclude 0). On time lost, `only_lane_urgency` beats stop signs on the grid at p=1.0 (-8.4%, 26/28 days) and loses at p=0.9 (+19.5%); at the single intersection stop signs beat every carma strategy (+133% to +1450% time lost). |

---

## Project structure

```
Exploration_of_Network_Level_Carma/
├── controller/              Python controller (all of the logic lives here)
│   ├── carma.py             Simulation loop, run() entry point
│   ├── config.py            Experiment settings and all named experiments
│   ├── core.py              Shared data types: Context, Bid, Grant, AuctionRecord, CarRecord
│   ├── car.py               Car: vehicle state + carma balance
│   ├── strategies.py        How much a single car bids (hand-made strategies)
│   ├── regret.py            RegretHedge: a bid policy learned by regret minimisation
│   ├── bidders.py           Grouping cars into bidders, aggregating bids, building Bids
│   ├── auction.py           Auction mechanisms and payment rules
│   ├── conflicts.py         Which signal movements can be green together (from the network file)
│   ├── blocking.py          Detects links whose outgoing lane is full
│   ├── drivers.py           Persistent drivers whose carma carries over between trips
│   ├── history.py           Bidding history per intersection and per car
│   ├── metrics.py           Per-car trip results and the per-bidder-type summary
│   ├── signals.py           Turns "these links are green" into safe light states
│   ├── baselines.py         Signal baselines: SUMO actuated / delay-based programs, max-pressure
│   ├── sweeps.py            Study: parallel sweeps (demand; starting carma)
│   ├── persistence.py       Study: persistent carma at one intersection
│   ├── multiday.py          Study: multi-day chains, carma reset every k days
│   ├── urgency_study.py     Study: urgency-aware bidding, one intersection
│   ├── lanes_study.py       Study: 1-5 lanes per road, switching-overhead variants
│   ├── regret_study.py      Study: regret-minimising bidders, 5x5 grid
│   ├── learner_base.py      Learning study: decisions -> transitions (average cost per trip)
│   ├── best_response.py     Learning study: learner A, best response + learned value of carma
│   ├── qlearning.py         Learning study: learner B, tabular Q-learning
│   ├── prices.py            Per-intersection price summaries from the decision log
│   ├── convergence.py       Learning study: the convergence / stopping rule
│   └── learning_study.py    Study: learning to bid over 60-90 days; exploitability test
├── network/
│   ├── net/                 SUMO networks: the 5x5 grid (RandomGrid5_200_tls / _stop), the 3x3
│   │                        grid (grid3_200_tls / _stop), single_4way, single_{1..5}lane, ...
│   ├── tls/                 Actuated / delay-based signal programs (generated on first use)
│   ├── single/              Node/edge descriptions the single-intersection networks are built from
│   ├── osm/, shp/           Source map data
├── simulation/              Generated trip/route files (one folder per study)
├── docs/
│   ├── tutorial.md          Step-by-step guide to running the experiments
│   ├── api.md               API reference (NumPy/pandas style) and command-line reference
│   └── learning_spec.md     Specification of the learning study
├── scripts/run_sim.sh       Commands to generate the network/routes and view them in SUMO
├── tests/                   Unit tests (python tests/test_learning.py)
├── analyze/                 Analysis scripts (one per study, plus delay_metrics.py and plot_delay.py),
│                            and paired.py (paired comparisons, shared by the study analyses);
│                            figures go to analyze/figures/, tables to results/analysis/ and analyze/tables/
├── results/                 CSV files written by each run (one folder per study)
├── requirements.txt
└── README.md
```

The 5x5 grid has 200 m edges and a traffic light at every junction. The 4 corner junctions have a
single all-green phase, so there is nothing to arbitrate and they are left on their own program.
The other 21 intersections are controlled. The 3x3 grid is built the same way (`netgenerate --grid
--grid.number 3 --grid.length 200`): one 4-way junction in the centre and four T-junctions are
controlled, the four corners are not. The single-intersection networks are one 4-way
junction with 200 m arms; `network/net/single_{N}lane.net.xml` has N lanes on every road.

---

## How one simulation step works

[carma.py](controller/carma.py) repeats the following until all vehicles have finished.

```
SUMO step
   |
   v
Create Cars for departed vehicles (hand each to a persistent Driver if enabled, draw its urgency),
drop arrived ones (return the balance to the Driver), Car.update() for the rest
   |
   v
blocking.find_blocked_links()      which links lead into a full lane?
   |
   v
carma.build_context()              Context: cars within bid_range of each intersection
   |
   v
For each controlled intersection -> carma.step_intersection():
   |
   |  1. Drop finished grants (members crossed, timed out, or all links blocked)
   |  2. If no grant is left at this intersection -> hold an auction:
   |        cars on blocked links are removed
   |        for each bidder type (cars are assigned one when they depart):
   |          type.grouper         that type's cars -> groups
   |          bidders.make_bid     type.strategy per car, type.aggregate -> Bid
   |        (all types' Bids compete in the same auction)
   |        cfg.auction.run        Bids -> winning Bids
   |        cfg.payment.settle     winners/all bids -> carma changes (applied to Cars)
   |        history.record         the finished auction is stored (after the bids were made)
   |        Grant(...)             one pending Grant per winner; AuctionRecord logged
   |  3. Grant.try_activate        pending grants turn on once the front car is close enough
   |  4. green set = bid links of every active grant, plus the lane-front links
   |  5. SignalController.apply    green set -> yellow/red/green state string -> SUMO
```

Key ideas:

- **A win is a commitment.** An auction only runs at an intersection that has no grant. The
  winner keeps its green until every member car has crossed the stop line (or `max_green`
  seconds pass), so the light does not flip while the group is still crossing. Three optional
  settings change this: `min_green` (never end earlier), `extend_green` (keep green while cars
  are still queued on the served lanes) and `release_fraction` (end once that share of the winners
  has crossed).
- **A grant is pending until its front car is near.** The grant is awarded (and paid for) at
  once, but the lights only change when the front car is within `activation_dist` of the
  intersection.
- **The lane-front fix.** The bid's own links are not enough on a single-lane approach: a car at
  the head of the queue may need a different movement from the winner behind it and would block it.
  While a grant is green, the link of the car currently at the head of each served lane is also
  green (unless it conflicts with a link already green, when `cfg.conflicts` is set).
- **Several winners.** With `NonConflictingGreedy` and a conflict table, bids that do not cross
  each other's movements are all granted, so e.g. two lanes of one approach go green together.
  With the default `HighestBid` only one group wins per auction.
- **Intersections are never all red.** With no active grant the last green stays showing, and with
  `all_red = 0` a link losing green shows yellow first. So there is always at least one green or
  yellow link.
- **Carma balances change in one place only:** the `PaymentRule.settle` result applied in
  `step_intersection`.

---

## Named experiments

Looked up with `config.get_experiment(name)` (the `STUDY_CONFIGS` dictionary). Unless noted all
use `HighestBid` and `FirstPrice(redistribute=True, free_if_uncontested=True)`.

| Name | What it is |
|---|---|
| `default_lights` | Control: SUMO's own fixed-time signal programs (90 s cycle), nothing bids. |
| `actuated_lights` | Baseline: SUMO's gap-based actuated control over the same phases (green 5-50 s), nothing bids. |
| `delay_based_lights` | Baseline: SUMO's delay-based actuated control over the same phases (green 5-50 s), nothing bids. |
| `max_pressure` | Baseline: max-pressure control over the same phases, 10 s minimum green ([baselines.py](controller/baselines.py)), nothing bids. |
| `stop_signs` | Control: the same network with all-way stops (`config.STOP_SIGNS`, `STOP_SIGNS_SINGLE`). |
| `only_vehicle_random` | Every car bids a random fraction of its balance. |
| `only_vehicle_urgency` / `only_lane_urgency` | Wait-time urgency, per vehicle / per lane group (urgency-blind). |
| `only_link_fixed` | Per (lane, link) group, 10% of the balance. |
| `only_price_follower` | Bids a margin above the recent winning amount (uses history). |
| `only_vehicle_budget` / `only_lane_budget` | A share of the balance that grows with waiting time. |
| `urg_vehicle_wait` / `urg_lane_wait` | Urgency x waiting time, per vehicle / per lane group. |
| `urg_vehicle_fraction` | Large share of balance if urgent, small otherwise, per vehicle. |
| `urg_omniscient` / `urg_coin` | Benchmarks (`NoPayment`): serve the most urgent car / a random car. |
| `lane_fast_release` | Lane bidder; yellow 2 s and grants end once half the winners have crossed. |
| `lane_min_green` | Lane bidder; minimum green 10 s, extended while the served lanes are queued. |
| `phase_urgency` | Cars bid for whole signal phases (NS/EW x through/left). |
| `regret_lane_0.5` ... `regret_lane_8.0` | Every driver uses `RegretHedge` with carma price 0.5, 1, 2, 4 or 8, lane groups. |
| `frac_lane_0.9_0.15`, `frac_lane_0.9_0.05` | Non-learning controls: bid 90% of balance when urgent, 15% / 5% otherwise, lane groups. |
| `mix_learn20`, `mix_learn80` | 20% / 80% of drivers learn (`RegretHedge`, price 2), the rest bid by lane urgency. |
| `paired_urgency_sum`, `paired_urgency_mean`, `paired_fixed_sum` | Paired auctions: compatible movements bid together. |

Older presets for single runs: `BASELINE`, `LANE_SUM`, `LANE_MEAN`, `LINK_SUM`, `LINK_MEAN`, `MIXED`
(one bidding structure each; see the file).

The lanes study replaces the auction of every carma experiment by
`NonConflictingGreedy(conflicts)` over the network's real conflict table (see
[lanes_study.py](controller/lanes_study.py)).

---

## Persistent drivers, urgency and learning

**Persistent drivers** (`persistent_drivers=N`). Without them every vehicle starts each trip with
the same balance and loses what is left. With them, N drivers keep their balance (and their bid
history) from trip to trip, so saving carma now to spend it later has a meaning. A multi-day
chain ([multiday.py](controller/multiday.py)) repeats the same drivers over several days with new
random trips, carrying balances from day to day. "Reset every k days" is simulated: a reset chain
puts every driver back at the starting balance on days 1, k+1, 2k+1, ... and is compared day by
day with the chain without resets (both see the same trips each day). The baselines in the
urgency and regret studies use the same persistent drivers (nothing is spent), so per-driver
fairness can be compared with them.

**Urgency** (`urgency_levels`, `urgency_probs`). Each trip draws a value of time: 1 with
probability 0.8 and 6 with probability 0.2 in the studies (CARMA section 5.2). It does not give
anyone an advantage in the auction. It is used (a) by urgency-aware strategies to decide how much
to bid, and (b) by the metrics: *cost = urgency x time lost*, so giving green to urgent trips
lowers the cost.

**Regret minimisation** ([regret.py](controller/regret.py)). `RegretHedge` runs the Hedge
(exponential weights) algorithm with full-information counterfactual payoffs:

- A bid is one of 10 fractions of the current balance (0, 1%, ..., 100%).
- After every auction the car took part in, each candidate fraction f is scored on what it would
  have earned. With x = f x balance, the group's total with the car's own bid replaced by x
  decides whether it would have won (`win(x)`), and

  `u(x) = urgency x win(x) - price(balance) x (payment(x) - (1 - win(x)) x rebate) / B0`

  where `payment(x)` is x if it wins and someone competed (0 if nobody did), `rebate` is the
  average carma a losing bid gets back, `B0` is the starting balance, and
  `price(balance) = lam0 x (B0 / max(balance, 5))^gamma` is the shadow price of carma (higher
  when the driver is poor).
- Each context (urgency, waiting-time bucket, balance bucket) keeps one log-weight per candidate
  fraction; after each auction `weight[f] += eta0 / sqrt(n) x u(f) / 6`. The car then bids a
  fraction sampled from the softmax of the weights.
- All drivers share one learner (a symmetric policy), and it persists across the days of a chain.

The "value of winning" is a modelling assumption: a win saves one unit of time, worth the trip's
urgency. The learner ignores the future; `lam0` stands in for the value of saving carma for later,
and is the one number to tune (a sweep over 0.5-8 is in the regret study).

**Learning to bid from repeated play** ([learning_study.py](controller/learning_study.py), spec in
[docs/learning_spec.md](docs/learning_spec.md)). Drivers who each minimise their own travel time
play the auction game every day and learn how to bid; the study watches whether the bids settle and
what priority then costs at each intersection.
- *Front-car bidding* (`BidderType(front_only=True)`, config `front()`): lanes bid as groups but only
  the car at the head of each lane chooses the amount, so nobody can ride free on lane-mates' bids.
  Ties go to the longest wait (`HighestBid(tie_break="wait")`) and payments are shared among the
  losing front cars only (`FirstPrice(rebate_to="bidders")`).
- *Learner A* ([best_response.py](controller/best_response.py), `learn_br`): learns the highest rival
  bid at each intersection (from the public auction outcomes; this distribution is the price there),
  the time a win and a loss cost, and the value of carma (TD learning), then bids the smoothed best
  response. *Learner B* ([qlearning.py](controller/qlearning.py), `learn_q`): tabular Q-learning
  with the same states and no auction model, as a check. *Learner C*: `RegretHedge` (`learn_hedge`).
- Both A and B minimise the average time per trip ([learner_base.py](controller/learner_base.py)):
  no discount factor; the state is (balance, intersections left, auctions lost here, urgency).
- Chains run 60-90 days and stop as soon as the convergence rule holds
  ([convergence.py](controller/convergence.py)); unconverged chains can be extended (`--extend-to`).
  Every decision bid is logged (`*_bids.csv.gz`) and every intersection's price summarised per day
  (`*_prices.csv`, `*_prices_by_day.csv`) for the price-evolution figures of
  [learning_analysis.py](analyze/learning_analysis.py). `--mutants` tests whether the learned policy
  is an equilibrium.

---

## Analysing results

The scripts in [analyze/](analyze/) read the files in `results/`. They need `matplotlib` (in
`requirements.txt`). The per-study scripts are described in
the [tutorial](docs/tutorial.md); `delay_metrics.py` and `plot_delay.py` work on any `*_cars.csv`
files (by default the most recent run; several files are pooled):

```bash
python analyze/delay_metrics.py                 # print the tables, save CSVs to analyze/tables/
python analyze/plot_delay.py                    # save PNGs to analyze/figures/ (--show to open them)
python analyze/plot_delay.py a_cars.csv b_cars.csv --thresholds 30 60 120
```

| Measure (per bidder type) | Where |
|---|---|
| Mean, median, 90th and 95th percentile and max of wait time, travel time and time loss | `delay_metrics.py` tables; `percentiles` and `distributions` figures |
| Delay ratio = travel time / free-flow time (free-flow time = travel time - time loss) | `delay_ratio` table; `distributions` and `cdfs` figures |
| Spread: standard deviation, IQR of wait time, share of cars waiting longer than 30 / 60 / 120 s | `spread` table; `thresholds` figure |
| Carma efficiency: seconds of wait saved (against the average car) per carma spent | `efficiency` table; `carma` figure |
| Urgency-weighted cost, urgent vs calm delay, fairness across drivers (Gini of per-driver cost) | `urgency_analysis.py`, `lanes_analysis.py`, `regret_analysis.py` |

**Gini coefficient.** 0 = everyone equal, 1 = one holder has everything. With values sorted
ascending, ranks 1..n: `G = 2 x sum(rank x value) / (n x total) - (n + 1) / n`. Used for the
carma balances (`*_carma_trace.csv`) and for the per-driver average cost.

**Paired comparisons** ([paired.py](analyze/paired.py)). Every strategy in a study sees the
same traffic for the same seed (and, in multi-day chains, the same day). The regret, multiday,
urgency and lanes analyses therefore compare each strategy with a reference unit by unit (seed,
or seed x day) and report the mean difference, a 95% interval and the number of units on which
the strategy was better (`results/analysis/*_paired.csv`). This removes the day-to-day swings in
traffic that affect every strategy alike, so it detects much smaller effects than comparing
averages. Days of one chain are not fully independent (balances and learners carry over), so the
interval is somewhat optimistic; the count of units won is a check on it.

Notes on reading the results:

- "Carma spent" is net (start minus final balance). Redistribution pays carma back to losing
  bidders, so it is negative for many cars.
- "Wait saved" is measured against the average car in the same data.
- One run is one sample; pool several seeds before reading small differences.
- Delay is only measured over cars that finished; read it next to the backlog.

---

## Mixing bidding types

To find out which bidding behaviour does best, run them against each other in one simulation:

```python
Experiment(
    name="mixed",
    population=[
        BidderType("vehicle_random", group_by_vehicle, RandomFraction(), sum),
        BidderType("lane_urgency", group_by_lane, WaitTimeUrgency(k=1.0), sum, share=2,
                   initial_carma=200),   # optional: this type starts with more carma
    ],
    auction=HighestBid(),
    payment=FirstPrice(redistribute=True),
)
```

- Each departing car (or persistent driver) is given a type at random, in proportion to `share`
  (here one third and two thirds). The draw uses the run's `seed`, so the same seed gives the same
  assignment.
- At each intersection the cars of every type are grouped, priced and aggregated by their own
  type, and **all the resulting bids go into the same auction**. The auction and payment rule
  are shared, so the types genuinely compete for the same green time and the same carma pool.
- The per-type averages appear at the end of the run and in the `_summary.csv` file; the `kind`
  column of `_cars.csv` identifies the type of every trip (`mix_learn20` / `mix_learn80` are
  analysed this way by `regret_analysis.py`).

Things to keep in mind when reading the comparison: one run is one sample; cars in the same lane
share green (with `serve_lane_front` the head car is served whichever type it is); and carma is
not replenished, so read `mean_final_carma` next to the delay columns.

---

## Extending the project

| To add... | Do this |
|---|---|
| A new bidding behaviour | Add a class with `amount(car, ctx)` in [strategies.py](controller/strategies.py); use it as `strategy`. |
| A new way to group cars | Add a function `cars -> list of groups` in [bidders.py](controller/bidders.py); use it as `grouper`. |
| A mix of bidding types | Make a `BidderType(name, grouper, strategy, aggregate, share)` for each and pass them as `population`. |
| A new aggregation | Any function from an iterable of floats to a float; use it as `aggregate`. |
| A strategy that learns | Write a `BidStrategy` that reads `ctx.history` and `car.bid_history` (see `RegretHedge`); give it a `report(path)` method to save what it learned. In a study, `copy.deepcopy` the experiment per chain so learners do not share state. |
| A new auction | Add a class with `run(bids)` in [auction.py](controller/auction.py). |
| A new payment rule | Add a class with `settle(winners, all_bids)` (subclass `FirstPrice` and override `price()` for a new price). |
| A named experiment | Make an `Experiment(...)` in [config.py](controller/config.py) and add it to `STUDY_CONFIGS`; then every study script can run it by name. |
| A new study | See [Tutorial 9](docs/tutorial.md#9-write-your-own-study-script). |

The simulation loop only calls these through the interfaces above, so none of them needs changes in
[carma.py](controller/carma.py).

---

## Output files

Each simulation writes these files (the name is `<label>` plus a suffix; study scripts choose the
label, a single run uses `<experiment name>_<date time>`):

| File | Contents |
|---|---|
| `<label>.csv` | The auction log, one row per auction (`t`, `tls`, `n_bids`, `winners`, `amounts`, `kinds`). |
| `<label>_cars.csv` | One row per finished trip: `vid`, `kind`, `depart`, `arrive`, `travel_time`, `wait_time`, `time_loss`, `wins`, `final_carma`, `start_carma`, `driver`, `urgency`. **The main input of every analysis.** |
| `<label>_summary.csv` | Per bidder type: `cars`, `mean_travel_time`, `mean_wait_time`, `mean_time_loss`, `mean_wins`, `mean_start_carma`, `mean_final_carma`. |
| `<label>_run.csv` | One row describing the run: `end_time`, `cars_arrived`, `cars_in_network`, `not_in_network` (backlog), `teleports`, `auctions`, `jammed`, `jam_time`, `mean_stopped_share`, `max_in_network`, `wall_seconds`, `guest_trips`, plus the study's `extra_info` (e.g. `experiment_name`, `p`, `lanes`, `day`). A run is "finished" when this file exists. |
| `<label>_network_trace.csv` | Every run: the network state at every checkpoint (every `trace_interval` seconds, and at the end): `t`, `in_network` (cars driving), `stopped_share`, `mean_speed`, `mean_acc_wait` (mean waiting so far on their trip), `arrived` (trips finished so far). |
| `<label>_carma_trace.csv` | Persistent runs: the carma distribution (mean, std, min, median, max, Gini, top-10% share) over all drivers and per type at every checkpoint. |
| `<label>_balance_trace.csv` | Persistent runs: every driver's balance at every checkpoint. |
| `<label>_drivers.csv` | Persistent runs: one row per driver (`trips`, `wins`, `start_carma`, `final_carma`, `mean_wait`, `mean_time_loss`). `final_carma` starts the next day in multi-day chains. |
| `<label>_policy.csv` | Learning strategies: for each context the probability of every candidate bid and the mean fraction of the balance bid (`RegretHedge`), or the greedy bid for every state (learners A and B). |
| `<label>_history.csv` | Only with `save_history=True`: one row per bid per auction (large). |
| `<label>_bids.csv.gz` | With `bid_log="full"` (learning study): one row per decision (a car that chose a bid): its state (`balance`, `x`, `k`, `lost_here`, `urgency`, `wait`), `bid`, `bid_rel`, `won`, `top_rival`, `winning_bid`, `payment`, `rebate`, `n_rivals`, `n_cars`. Kept out of git. |
| `<label>_prices.csv` | With `price_bin` set: per intersection, per time bin and for the whole day (`scope`), the number of auctions, mean / std / quantiles of bids and of winning bids in contested auctions (relative to the mean balance, `*_rel_*`, and in carma), payments, rebates, share of all-zero auctions, share of broke bidders, load, junction type ([prices.py](controller/prices.py)). |

`guest_trips` counts departures that found every persistent driver busy (the vehicle then drives
with a throw-away balance); it should be 0 or small.

---

## Known limitations and to-do

- **The regret learner ignores the future.** Its payoff is the value of winning now minus a
  hand-set carma price; it is not the equilibrium of the dynamic game in the CARMA paper, and the
  results have not been checked for equilibrium (no test that a driver cannot gain by deviating).
- **The value of a win is fixed at one unit of time** and does not depend on how long the car has
  waited or how far it still has to travel.
- **Lane layouts are SUMO defaults** (the leftmost lane shares left and straight); there are no
  dedicated left-turn pockets.
- **Vehicle-level omniscient benchmark is not a true ceiling.** A lane-level omniscient benchmark
  is not implemented.
- **Signal baselines are untuned.** `actuated_lights` and `delay_based_lights` use SUMO's
  default parameters (green 5-50 s), and `max_pressure` uses whole-lane vehicle counts, a 10 s
  minimum green and no maximum green. There is no coordinated green wave, so even these are not
  the best a tuned signal could do. `default_lights` (SUMO's fixed 90 s program) is the weakest
  baseline; the sweeps, persistence and multiday studies have not been re-run against the new
  baselines yet.
- **One auction at a time per intersection.** A new auction runs only when all grants at the
  intersection have ended; grants per link would let a freed link be re-auctioned meanwhile.
  Winners of a paired or multi-winner auction are re-auctioned together.
- **Carma leaks in some setups.** With `BASELINE` (no redistribution), or a first-price rule
  without `free_if_uncontested`, the winner's payment can leave the system. The experiments that
  redistribute have `free_if_uncontested` on, so total carma stays exactly constant.
- **Pay on award.** Members pay when a bid is won, even if the grant later expires without going
  green.
- **Few seeds.** Most studies use 2-3 seeds. Paired comparisons (by seed and day) make the
  multi-day studies much more precise, but the single-run studies (urgency, lanes) still have
  only 2-3 paired units per comparison.
- **No tests** and no collision/teleport monitoring beyond the `teleports` count. `libsumo` is
  listed in the requirements but not used.
- **Results are not committed.** `results/saved` (v1 copies) is large and the repository's
  `.gitignore` / LFS settings still need a decision; `controller/Carma.py` and `controller/carma.py`
  differ only by case in the git index (a rename in progress).
