# API reference

The reference follows the layout used by NumPy and pandas: every public object has a signature,
a one-line summary, **Parameters**, **Returns**, **See Also**, **Notes** and **Examples**.
Examples marked `>>>` were run against the code in this repository; examples that need SUMO say
so. Modules are listed in the order a simulation uses them.

All modules live in [controller/](../controller/). Import them from inside that folder (or put it
on `PYTHONPATH`):

```bash
cd controller
python
>>> import carma, config
```

**Contents**

| Module | What is in it |
|---|---|
| [`carma`](#module-carma) | `run`, the simulation loop |
| [`config`](#module-config) | `Experiment` and the named experiments |
| [`car`](#module-car) | `Car` |
| [`core`](#module-core) | `Context`, `Bid`, `Grant`, `AuctionRecord`, `CarRecord` |
| [`strategies`](#module-strategies) | hand-made bid strategies |
| [`regret`](#module-regret) | `RegretHedge`, the learned strategy |
| [`bidders`](#module-bidders) | `BidderType`, groupers, `make_bid` |
| [`auction`](#module-auction) | auctions and payment rules |
| [`conflicts`](#module-conflicts) | `ConflictTable` |
| [`drivers`](#module-drivers) | `DriverPool`, `gini` |
| [`history`](#module-history) | `AuctionHistory` and its records |
| [`metrics`](#module-metrics) | per-car results |
| [`blocking`](#module-blocking) | full outgoing lanes |
| [`signals`](#module-signals) | `SignalController` |
| [`baselines`](#module-baselines) | actuated / delay-based programs, `MaxPressure` |
| [`learner_base`](#module-learner_base) | `DecisionLearner`: decisions linked into transitions (learning study) |
| [`best_response`](#module-best_response) | `BestResponseLearner` (learner A) |
| [`qlearning`](#module-qlearning) | `QLearner` (learner B) |
| [`prices`](#module-prices) | per-intersection price summaries |
| [`convergence`](#module-convergence) | the convergence / stopping rule |
| [Command-line scripts](#command-line-scripts) | the study scripts and analysis scripts |

The examples below build cars with `types.SimpleNamespace` so that they run without SUMO; a real
`Car` has the same attributes.

---

## Module `carma`

### `carma.run`

```python
carma.run(cfg=BASELINE, gui=True, delay_ms=100, seed=0, routes=ROUTES, net=NET,
          max_time=None, label=None, out_dir='../results', save_history=True, quiet=False,
          extra_info=None, jam_stop=False, jam_share=0.9, jam_seconds=300, jam_min_cars=100,
          start_balances=None, pool_seed=None) -> dict
```

Run one simulation and save its results.

**Parameters**

| Name | Type, default | Description |
|---|---|---|
| `cfg` | `Experiment`, `BASELINE` | The experiment: who bids, who wins, who pays, and every setting (see [`config.Experiment`](#configexperiment)). With `cfg.control == "default"` SUMO's own signal programs run and nothing bids. |
| `gui` | `bool`, `True` | Open the SUMO GUI. Use `False` for batch runs. |
| `delay_ms` | `int`, `100` | GUI delay per step in milliseconds (ignored without the GUI). |
| `seed` | `int`, `0` | SUMO's random seed. Also seeds the tie-breaks, the assignment of bidder types and the urgency draws, so a run is repeatable. |
| `routes` | `str` | Route file (trips) to simulate. Default: the 5x5 grid's 7,200 s of trips. |
| `net` | `str` | Network file. Default: the 5x5 grid (`../network/net/RandomGrid5_200_tls.net.xml`). Paths are relative to the working directory. |
| `max_time` | `float` or `None`, `None` | Stop at this simulated time (s). Cars still driving then are not in the per-car results. `None` runs until every car has finished. |
| `label` | `str` or `None` | File-name prefix of the outputs. Default: `<cfg.name>_<date time>`. |
| `out_dir` | `str`, `'../results'` | Folder the outputs are written to (created if missing). |
| `save_history` | `bool`, `True` | Also write the full bid history (`_history.csv`, large on long runs). |
| `quiet` | `bool`, `False` | Hide SUMO's step log and warnings. Use when running many in parallel. |
| `extra_info` | `dict` or `None` | Extra columns added to the one-row `_run.csv` (e.g. `{"p": 3.0}`). The analysis scripts read the study parameters from here. |
| `jam_stop` | `bool`, `False` | End the run when it gridlocks. |
| `jam_share`, `jam_seconds`, `jam_min_cars` | `float`, `float`, `int` | A run counts as jammed when at least `jam_min_cars` cars are in the network and at least `jam_share` of them are stopped for `jam_seconds` in a row. |
| `start_balances` | `dict` or `None` | Persistent drivers only: `{driver id: balance}` to start from (the previous day's `final_carma`). |
| `pool_seed` | `int` or `None` | Persistent drivers only: seed that fixes who the drivers are (their bidder types), so every day of a chain has the same population. |

**Returns**

`info` : `dict`
: One entry per column of `_run.csv`: `end_time`, `cars_arrived`, `cars_in_network`,
  `not_in_network` (the backlog), `teleports`, `auctions`, `jammed`, `jam_time`,
  `mean_stopped_share`, `max_in_network`, `wall_seconds`, `guest_trips`, `label`, `experiment`,
  `seed`, plus everything in `extra_info`.

**Notes**

- Files written, all named `<out_dir>/<label>...`: `.csv` (auction log), `_cars.csv`,
  `_summary.csv`, `_run.csv`, `_network_trace.csv` (network state at every checkpoint), and,
  for persistent drivers, `_carma_trace.csv`,
  `_balance_trace.csv` and `_drivers.csv`; for learning strategies `_policy.csv`; with
  `save_history`, `_history.csv`. See [Output files](../README.md#output-files).
- The per-car results contain only cars that **arrived**. Read delay together with
  `not_in_network` (the backlog).
- TraCI prints ` Retrying in 1 seconds` while SUMO starts; this is harmless.

**See Also**: [`config.Experiment`](#configexperiment), [`metrics.summarize`](#module-metrics)

**Examples** (needs SUMO; run from `controller/`)

```python
>>> import carma, config
>>> info = carma.run(config.get_experiment("only_lane_urgency"), gui=False, delay_ms=0, seed=0,
...                  net="../network/net/single_4way.net.xml",
...                  routes="../simulation/single_2h/p_3.00.rou.xml",
...                  max_time=600, label="demo", out_dir="/tmp/demo", quiet=True,
...                  save_history=False)
>>> info["cars_arrived"], info["auctions"], info["jammed"]
(179, 59, False)
```

### `carma.controllable_tls`

```python
carma.controllable_tls() -> list[str]
```

Ids of the intersections the controller arbitrates: those whose built-in program ever shows a
red. Junctions whose only phase is all-green (the corners of the 5x5 grid) have nothing to
decide and are left alone. Needs a live TraCI connection (called inside `run`).

### `carma.build_context`

```python
carma.build_context(t, cars, controlled, bid_range, blocked, history=None) -> Context
```

Snapshot of the world for one step: for each controlled intersection, the cars whose next traffic
light it is and that are within `bid_range` metres of the stop line.

**Parameters**: `t` (`float`) simulation time; `cars` (`dict[str, Car]`); `controlled` (`list[str]`)
intersection ids; `bid_range` (`float`) metres; `blocked` (`dict[str, set[int]]`) blocked signal
indices per intersection; `history` (`AuctionHistory` or `None`).

**Returns**: [`core.Context`](#corecontext).

### `carma.step_intersection`

```python
carma.step_intersection(tls, ctx, cfg, cars, grants, signals, log) -> None
```

One step at one intersection: drop finished grants; if none remain hold an auction (group, bid,
pick winners, settle payments, record history); activate grants whose front car is near; send the
resulting green set to the signal controller. This is the only place carma balances change.

### `carma.pick_bidder_type`

```python
carma.pick_bidder_type(types, rng) -> BidderType
```

Draw a bidder type at random with probability proportional to `BidderType.share`.

---

## Module `config`

### `config.Experiment`

```python
config.Experiment(name, auction, payment, grouper=None, strategy=None, aggregate=None,
                  population=None, bid_range=50.0, serve_lane_front=True,
                  blocked_entrance_dist=10.0, activation_dist=50.0, max_green=30.0,
                  yellow=3.0, all_red=0.0, history_length=100, control='carma',
                  initial_carma=100.0, conflicts=None, min_green=0.0, release_fraction=1.0,
                  extend_green=False, persistent_drivers=None, net=None, urgency_levels=(),
                  urgency_probs=(), trace_interval=600.0)
```

A dataclass bundling one choice per layer plus the numeric settings of a run. Give either
`population`, or all three of `grouper`, `strategy` and `aggregate`.

**Parameters**

| Name | Type, default | Description |
|---|---|---|
| `name` | `str` | Label of the experiment (also the default output name). |
| `auction` | `Auction` | Who wins: [`HighestBid`](#auctionhighestbid), [`NonConflictingGreedy`](#auctionnonconflictinggreedy). |
| `payment` | `PaymentRule` | Who pays what: [`FirstPrice`](#auctionfirstprice), [`SecondPrice`](#auctionsecondprice), [`NoPayment`](#auctionnopayment). |
| `grouper`, `strategy`, `aggregate` | callable, `None` | How cars group into bidders, how much each bids, how member bids combine. Used when every car bids the same way. |
| `population` | `list[BidderType]`, `None` | Several bidder types in one simulation (overrides the three above). |
| `bid_range` | `float`, `50.0` | A car may bid only within this many metres of the stop line. |
| `serve_lane_front` | `bool`, `True` | While a grant is green also serve the head car of each lane it covers (prevents a head car with another movement from blocking the group). |
| `blocked_entrance_dist` | `float` or `None`, `10.0` | A link is blocked when a stopped car is this close to the start of the lane it leads into. `None` disables the check. |
| `activation_dist` | `float`, `50.0` | A won grant turns green only when its front car is this close. At or above `bid_range` it activates at once. |
| `max_green` | `float`, `30.0` | Longest a grant may last (s). |
| `yellow`, `all_red` | `float`, `3.0`, `0.0` | Yellow time for a link losing green, and the all-red gap after it. Keep `all_red` at 0 so there is always a green or yellow link. |
| `history_length` | `int` or `None`, `100` | How many recent auctions per intersection strategies can read. |
| `control` | `str`, `'carma'` | `'carma'` runs the auctions. The baselines, in which nothing bids: `'default'` leaves SUMO's own fixed programs untouched (also used for stop signs); `'actuated'` and `'delay_based'` run SUMO's actuated controllers over the same phases; `'max_pressure'` runs max-pressure control (see [`baselines`](#module-baselines)). |
| `initial_carma` | `float`, `100.0` | Starting balance. A `BidderType` can override it. |
| `conflicts` | callable or `None` | `conflicts(tls, a, b) -> bool`; stops the lane-front fix from turning on a link that clashes with one already green. |
| `min_green`, `extend_green`, `release_fraction` | `float`, `bool`, `float` | Grant timing: fewest seconds of green once active; keep green while cars are still queued on the served lanes; share of winners that must cross before the grant may end. With `control='max_pressure'`, `min_green` is the shortest time a phase stays green before it may switch. |
| `persistent_drivers` | `int` or `None` | Size of a fixed driver population whose balances carry from trip to trip. |
| `net` | `str` or `None` | Network for this experiment (the stop-sign experiments carry their own). |
| `urgency_levels`, `urgency_probs` | `tuple`, `()` | Each trip draws an urgency from these levels with these probabilities. |
| `trace_interval` | `float`, `600.0` | Seconds between checkpoints: the network state (every run, `_network_trace.csv`) and, for persistent runs, the carma distribution and every driver's balance. |
| `bid_log` | `str`, `"none"` | `"full"` writes every decision (a car that chose a bid, with its state and the outcome) to `_bids.csv.gz`. |
| `price_bin` | `float` or `None` | Write the per-intersection price summary (`_prices.csv`, see [prices](#module-prices)) in bins of this many seconds, plus a whole-day row. |
| `integer_carma` | `bool`, `False` | Whole units of carma: every bid is rounded down to a whole number (`bidders.make_bid`). Use with `FirstPrice(integer=True)` so payments and rebates are whole too. |

**Methods**: `bidder_types() -> list[BidderType]` returns the types in use.

**Notes**: an `Experiment` is immutable in practice; derive variants with
`dataclasses.replace(cfg, field=value)`.

**Examples**

```python
>>> import config
>>> from dataclasses import replace
>>> cfg = config.get_experiment("regret_lane_2.0")
>>> cfg.name, cfg.persistent_drivers, cfg.bidder_types()[0].name
('only_regret_lane_2.0', None, 'regret_lane_2.0')
>>> replace(cfg, persistent_drivers=300).persistent_drivers
300
```

### `config.get_experiment`

```python
config.get_experiment(name: str) -> Experiment
```

Look up a named experiment (the keys of `config.STUDY_CONFIGS`, or a preset such as `LANE_SUM`).
This is how the study scripts turn `--configs` names into experiments. See
[Named experiments](../README.md#named-experiments) for the list.

**Raises**: `KeyError` if the name is unknown.

### `config.only`

```python
config.only(btype: BidderType) -> Experiment
```

An experiment in which every car uses `btype`, with `HighestBid` and
`FirstPrice(redistribute=True, free_if_uncontested=True)`. The experiment is named
`"only_" + btype.name`.

### `config.stop_signs`

```python
config.stop_signs(net: str, name: str = 'stop_signs') -> Experiment
```

A baseline: the given network (whose junctions are all-way stops) with `control='default'`;
nothing is controlled or bid on. `config.STOP_SIGNS` (5x5 grid) and `config.STOP_SIGNS_SINGLE` are
ready-made.

### `config.signal_baseline`

```python
config.signal_baseline(name: str, control: str, **settings) -> Experiment
```

A baseline in which nothing bids and the lights are run by `control` (`'default'`, `'actuated'`,
`'delay_based'` or `'max_pressure'`); extra keyword arguments are `Experiment` settings. The cars
are recorded under `name`. Ready-made, by name in `STUDY_CONFIGS` (and listed in
`config.SIGNAL_BASELINES`):

| Name | Control |
|---|---|
| `default_lights` | SUMO's fixed 90 s program from the network file. |
| `actuated_lights` | SUMO gap-based actuated control over the same phases; green phases 5-50 s. |
| `delay_based_lights` | SUMO delay-based actuated control over the same phases; green phases 5-50 s. |
| `max_pressure` | Max-pressure over the same phases, minimum green 10 s. |

### `config.paired`

```python
config.paired(name: str, strategy: BidStrategy, aggregate: Aggregator) -> Experiment
```

An experiment with **paired auctions**: cars whose movements do not conflict bid together as one
group (see [`group_by_compatible`](#groupers)). Uses the 5x5 grid's
conflict table.

---

## Module `car`

### `car.Car`

```python
car.Car(vid, carma=100, kind='default')
```

A vehicle's cached SUMO state and carma balance. Makes no decisions. Created by `carma.run` when a
vehicle departs; `update()` is the only place cars read TraCI.

**Attributes**

| Name | Description |
|---|---|
| `vid` | SUMO vehicle id. |
| `carma`, `start_carma` | Current balance (changed only by the payment step) and the balance at the start of the trip. |
| `kind` | Name of the `BidderType` the car was assigned. |
| `driver` | Id of the persistent driver (`None` if carma is not persistent). |
| `urgency` | The trip's value of time (1.0 unless the experiment draws urgencies). |
| `bid_history` | `list[CarBidRecord]`: the car's own bids and outcomes (shared with its driver across trips). |
| `depart`, `wins` | Entry time and number of winning groups the car was part of. |
| `lane`, `pos`, `speed`, `length`, `accel`, `wait_time` | Where the car is and how long it has been stopped. |
| `acc_wait`, `time_loss` | Total waiting time and SUMO time loss over the trip so far. |
| `tls`, `link`, `dist` | Next traffic light, the signal index (movement) the car needs, and metres to the stop line. `None` when no light remains. |
| `intersections_left`, `route_len`, `route_index` | Auction-holding intersections still ahead (counting the next one), at the start of the trip, and the position of the next one (lights with nothing to decide are not counted; `Car.decision_tls` is set by `carma.run`). |
| `lost_at`, `lost_here` | Auctions lost at each intersection on this trip, and at the next one. |

**Methods**: `update()` refreshes everything from TraCI (once per step); `update_light()` is a
cheaper refresh for runs where nothing bids.

---

## Module `core`

Plain data types passed between the layers (no TraCI calls).

### `core.Context`

```python
core.Context(t, approaching, blocked={}, history=None)
```

Read-only snapshot of one step. `approaching` maps an intersection id to the cars within bid range;
`blocked` maps it to the blocked signal indices; `history` is the `AuctionHistory`.

**Methods**

- `front_car(tls, lane) -> Car | None`: the approaching car closest to the stop line on `lane`.
- `on_approach(tls) -> set[str]`: ids of the cars currently approaching `tls`. A car that crosses
  the stop line drops out.

### `core.Bid`

```python
core.Bid(tls, members, links, member_bids, amount, kind='default', bidders=None)
```

One bidder's offer: `members` (cars) ask for green on `links` (signal indices) and offer
`member_bids` (`{vid: amount}`); `amount` is the aggregated number the auction compares. A single
car and a lane group are both just a `Bid`. `bidders` holds the ids of the members that chose the
amount (`None` = all of them; the front car under front-car bidding); `deciders()` lists them.

### `core.Grant`

```python
core.Grant(bid, members, started, max_duration, activated_at=None, lanes=set(),
           min_duration=0.0, release_fraction=1.0, extend=False)
```

A won bid. Holds green until its members have crossed (see `done`) or `max_duration` passes.

**Methods**

| Method | Description |
|---|---|
| `try_activate(ctx, activation_dist, serve_lane_front=False)` | Activate once the front car is within `activation_dist` metres. |
| `front_distance(ctx, serve_lane_front=False)` | Distance of the front car (closest member, or closest car of any kind on the served lanes). |
| `front_links(ctx, blocked)` | Signal indices of the head car of each served lane (lane-front fix), without blocked links. |
| `green_links(ctx, serve_lane_front, blocked)` | The bid's links plus the lane-front links. |
| `active` (property) | `True` once activated. |
| `done(ctx)` | `True` when the grant should end: timed out; or, after `min_duration` and (if `extend`) with no car still queued on the served lanes, once enough members (`release_fraction`) have crossed. |

### `core.AuctionRecord`, `core.CarRecord`

`AuctionRecord(t, tls, n_bids, winners, amounts, kinds)` is one row of the auction log.
`CarRecord(vid, kind, depart, arrive, travel_time, wait_time, time_loss, wins, final_carma,
start_carma, driver=None, urgency=1.0)` is one row of `_cars.csv`.

---

## Module `strategies`

A **strategy** answers "how much does this car bid?". Any object with a method
`amount(car, ctx) -> float` is a strategy ([`strategies.BidStrategy`](#strategiesbidstrategy)).
A strategy must never bid more than `car.carma`, because the payment rules rely on it to keep
balances non-negative (the omniscient benchmark is the one deliberate exception).

### `strategies.BidStrategy`

The interface (a `typing.Protocol`): `amount(car: Car, ctx: Context) -> float`.

### Built-in strategies

| Class and signature | Bid |
|---|---|
| `RandomFraction()` | Uniform random in `[0, carma]`. |
| `FixedFraction(fraction=0.1)` | `fraction * carma`. |
| `WaitTimeUrgency(k=1.0)` | `min(carma, k * wait_time)`. |
| `BudgetedUrgency(full_wait=60.0)` | `min(1, wait_time / full_wait) * carma`. |
| `UrgencyScaledWait(k=1.0)` | `min(carma, k * urgency * wait_time)`. |
| `UrgencyFraction(high_fraction=0.5, low_fraction=0.05, threshold=1.0)` | `high_fraction * carma` if `urgency > threshold`, else `low_fraction * carma`. |
| `RecentPriceFollower(margin=1.1, n=5, start_fraction=0.1)` | `margin` times the mean winning amount of the last `n` auctions at the car's next intersection, capped at the balance; `start_fraction * carma` without history. Reads `ctx.history`. |
| `OmniscientUrgency()` | **Benchmark only.** `urgency + 1e-3 * wait_time`, ignoring the balance. Pair with `NoPayment`. |

**Examples**

```python
>>> from types import SimpleNamespace as NS
>>> from strategies import WaitTimeUrgency, UrgencyFraction, UrgencyScaledWait, BudgetedUrgency
>>> car = NS(carma=100.0, wait_time=12.0, urgency=6.0)
>>> WaitTimeUrgency(k=2.0).amount(car, None)
24.0
>>> UrgencyScaledWait().amount(car, None)        # 1 * 6 * 12, below the balance
72.0
>>> UrgencyFraction(0.9, 0.05).amount(car, None) # urgent: 90% of the balance
90.0
>>> BudgetedUrgency(60).amount(car, None)        # 12/60 of the balance
20.0
```

A custom strategy is a class with an `amount` method:

```python
>>> import math
>>> class SqrtWait:
...     def amount(self, car, ctx):
...         return min(car.carma, math.sqrt(car.wait_time))
>>> SqrtWait().amount(NS(carma=100.0, wait_time=16.0), None)
4.0
```

---

## Module `regret`

### `regret.RegretHedge`

```python
regret.RegretHedge(lam0=1.0, gamma=1.0, base_balance=100.0, eta0=2.0, seed=0)
```

A strategy that **learns** how much to bid by regret minimisation (the Hedge / exponential
weights algorithm with full-information counterfactual payoffs). Each bid is one of ten
fractions of the balance: `0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0`.

After every auction a car took part in, each candidate fraction is scored on what it **would**
have earned (using the car's `CarBidRecord`: its own bid, its group's amount, the best rival bid,
its balance and its urgency). With bid `x = fraction * balance`:

```
win(x) = (group_amount - own_bid + x) > top_other_amount      (or nobody competed)
u(x)   = urgency * win(x) - price(balance) * (payment(x) - (1 - win(x)) * rebate) / B0
price(balance) = lam0 * (B0 / max(balance, 5)) ** gamma
log_weight[context][fraction] += eta0 / sqrt(n_context) * u(x) / 6
```

and the car then bids a fraction drawn from `softmax(log_weight[context])`. The *context* is
`(urgency, waiting-time bucket, balance bucket)`: waiting `<10 s`, `10-30 s`, `>30 s`; balance
`<0.4 x B0`, `0.4-1.5 x B0`, `>1.5 x B0`. All cars share one learner, so it sees many auctions.

**Parameters**

| Name | Type, default | Description |
|---|---|---|
| `lam0` | `float`, `1.0` | Carma price: how many low-urgency wins the whole starting balance is worth. Larger = spend more reluctantly. The one number to tune (0.5 to 8 in the regret study). |
| `gamma` | `float`, `1.0` | How fast the price rises as the balance falls. |
| `base_balance` | `float`, `100.0` | `B0`, the normal balance; payments are measured in units of it. |
| `eta0` | `float`, `2.0` | Learning rate (decays as `1/sqrt(n)` per context). |
| `seed` | `int`, `0` | Seed of the random choice of which candidate to play. |

**Methods**

| Method | Description |
|---|---|
| `amount(car, ctx) -> float` | Learn from the car's new auction records, then draw a fraction and return `fraction * car.carma`. |
| `context(urgency, wait, balance)` | The context tuple used as the key of the weights. |
| `probabilities(ctx) -> list[float]` | The current probability of each candidate fraction in a context. |
| `report(path)` | Write the learned policy (probability of each fraction per context) to a CSV. Called by `carma.run` after the simulation (`*_policy.csv`). |

**Notes**

- The learner needs `car.bid_history`, which `carma.run` fills (via `history.record`). The first
  call for a car in a trip only records where its history stands; learning starts from the next
  auction.
- The learner persists as long as the object does: in a multi-day chain the same `Experiment`
  object is reused, so what is learned on day 1 is used on day 2. Give every chain its **own**
  copy (`copy.deepcopy(cfg)`), as `regret_study.py` does.
- The value of winning is a modelling assumption (the trip's urgency); the learner ignores the
  future, and `lam0` stands in for the value of saving carma.

**Examples**

```python
>>> from types import SimpleNamespace as NS
>>> from regret import RegretHedge
>>> from history import CarBidRecord
>>> learner = RegretHedge(lam0=2.0, seed=1)
>>> car = NS(carma=100.0, wait_time=5.0, urgency=6.0, bid_history=[])
>>> learner.amount(car, None)            # nothing learned yet; first call only marks the history
1.0
>>> car.bid_history.append(CarBidRecord(      # an auction the car lost with a bid of 0
...     t=1.0, tls="C", own_bid=0.0, group_amount=0.0, won=False, carma_change=0.0,
...     wait_time=5.0, top_other_amount=10.0, winning_amount=10.0, balance=100.0, urgency=6.0))
>>> _ = learner.amount(car, None)        # learns from that auction
>>> key = learner.context(6.0, 5.0, 100.0)
>>> [round(p, 3) for p in learner.probabilities(key)]   # shifts towards bids that would have won
[0.031, 0.031, 0.031, 0.031, 0.031, 0.207, 0.193, 0.175, 0.153, 0.117]
```

---

## Module `bidders`

### `bidders.BidderType`

```python
bidders.BidderType(name, grouper, strategy, aggregate, share=1.0, initial_carma=None,
                   front_only=False)
```

One bidding behaviour. Cars (or persistent drivers) are assigned a type when they enter; several
types can compete in one run (see `Experiment.population`).

| Name | Description |
|---|---|
| `name` | Label stored on each car and used to split the results. |
| `grouper` | Function `cars -> list of groups`: who bids together. |
| `strategy` | A [strategy](#module-strategies): how much each car bids. |
| `aggregate` | Function `iterable of floats -> float` (`sum`, `max`, `bidders.mean`, ...): how the members' bids combine. |
| `share` | Relative weight of this type when types are drawn (2 and 1 means two thirds and one third). |
| `initial_carma` | Starting balance of this type; `None` uses the experiment's. |
| `front_only` | Front-car bidding: in each group only the car closest to the stop line chooses the amount; the others bid 0, pay nothing and are served with it. Use with `group_by_lane` and `sum`. `config.front(name, strategy)` builds such an experiment (with `HighestBid(tie_break="wait")` and `FirstPrice(..., rebate_to="bidders")`). |

### Groupers

A grouper takes the cars approaching **one** intersection and returns a list of groups; each group
becomes one `Bid`.

| Function | Groups |
|---|---|
| `group_by_vehicle(cars)` | One group per car (single-vehicle bidding). |
| `group_by_lane(cars)` | One group per incoming lane; the group asks for the union of its members' links. |
| `group_by_link(cars)` | One group per (lane, signal index): cars needing exactly the same movement. |
| `group_by_phase(cars)` | One group per signal phase: (north-south or east-west) x (through+right, or left). Designed for the single intersection (the phase table is read from the network file). |
| `group_by_compatible(conflicts)` | **Returns a grouper** for paired auctions: every maximal set of mutually compatible movements is one group, so a car can belong to several groups. |
| `group_by_intersection(cars)` | Everyone in one group. |

### `bidders.make_bid`

```python
bidders.make_bid(members, tls, strategy, aggregate, ctx, kind='default', front_only=False,
                 integer=False) -> Bid
```

Ask `strategy` for each member's bid, collect the links the members need, and combine the member
bids with `aggregate` into the group's `amount`. With `front_only`, only the member closest to the
stop line is asked and `Bid.bidders` is set to it. With `integer`, every bid is rounded down to
a whole number of carma, never above the balance (`bidders.whole`).

**Examples**

```python
>>> from types import SimpleNamespace as NS
>>> from bidders import group_by_vehicle, group_by_lane, group_by_link, mean
>>> cars = [NS(vid="a", lane="NC_0", link=1, tls="C"),
...         NS(vid="b", lane="NC_0", link=0, tls="C"),
...         NS(vid="c", lane="SC_0", link=5, tls="C")]
>>> [[c.vid for c in g] for g in group_by_vehicle(cars)]
[['a'], ['b'], ['c']]
>>> [[c.vid for c in g] for g in group_by_lane(cars)]     # a and b share lane NC_0
[['a', 'b'], ['c']]
>>> mean([1, 2, 3])
2.0
```

---

## Module `auction`

### Auctions

An auction has `run(bids: list[Bid]) -> list[Bid]` and returns the winning bids.

#### `auction.HighestBid`

`HighestBid(tie_break="random")`. One winner: the highest `amount`. Exact ties are broken at
random, or with `tie_break="wait"` by the longest-waiting member first (matters when everyone is
broke and bids 0).

#### `auction.NonConflictingGreedy`

```python
auction.NonConflictingGreedy(conflicts=<every distinct pair conflicts>)
```

Several winners: bids are taken from highest to lowest and a bid is accepted unless one of its
links conflicts with a link of an already accepted bid. `conflicts(tls, a, b) -> bool` should be
`conflicts.get_table(net_file).conflicts` so that real crossing movements are detected; the
default treats every pair of different signal indices as conflicting (equivalent to one winner).

### Payment rules

A payment rule has `settle(winners, all_bids) -> dict[str, float]` and returns the change of each
car's balance (negative = pays, positive = receives). This is the only code that decides balance
changes.

#### `auction.FirstPrice`

```python
auction.FirstPrice(redistribute=False, free_if_uncontested=False, rebate_to="members",
                   integer=False)
```

Each winning group pays its own `amount`, split among its members in proportion to their
individual bids (nobody pays more than they offered).

| Name | Description |
|---|---|
| `redistribute` | Share everything collected equally among the cars of the losing bids (total carma is conserved); otherwise it leaves the system. |
| `free_if_uncontested` | An auction with no losing bid costs the winner nothing. Use with `redistribute=True` to keep total carma exactly constant. |
| `rebate_to` | `"members"`: every car of a losing bid shares the rebate; `"bidders"`: only the cars that chose a losing bid (`Bid.bidders`, the front car under front-car bidding). |
| `integer` | Whole units of carma: a group's payment is split among its members in whole units (largest remainders, `auction.split_whole`), and the collected carma is shared among the losers as evenly as possible, the leftover units going one each to the longest-waiting losers. Total carma stays exactly constant. |

#### `auction.SecondPrice`

`SecondPrice(redistribute=False, free_if_uncontested=False)`. As `FirstPrice`, but each winner pays
the highest losing bid (0 if uncontested), capped at its own bid.

#### `auction.NoPayment`

`NoPayment()`. Nobody pays or receives anything; used by the omniscient and coin benchmarks.

**Examples**

```python
>>> from types import SimpleNamespace as NS
>>> from core import Bid
>>> from auction import HighestBid, FirstPrice, SecondPrice, NoPayment
>>> a, b, c = NS(vid="a"), NS(vid="b"), NS(vid="c")
>>> lane1 = Bid(tls="C", members=[a, b], links={0, 1}, member_bids={"a": 10.0, "b": 5.0}, amount=15.0)
>>> lane2 = Bid(tls="C", members=[c], links={5}, member_bids={"c": 8.0}, amount=8.0)
>>> winners = HighestBid().run([lane1, lane2])
>>> [w.amount for w in winners]
[15.0]
>>> FirstPrice(redistribute=True).settle(winners, [lane1, lane2])   # a and b pay 15; c, the loser, gets it
{'a': -10.0, 'b': -5.0, 'c': 15.0}
>>> SecondPrice().settle(winners, [lane1, lane2])                   # winner pays the best losing bid, 8
{'a': -5.333333333333333, 'b': -2.6666666666666665}
>>> NoPayment().settle(winners, [lane1, lane2])
{}
```

---

## Module `conflicts`

### `conflicts.ConflictTable`

```python
conflicts.ConflictTable(net_file=DEFAULT_NET)
```

For every traffic light in a network file, the set of pairs of signal indices that cannot be green
together (SUMO's `foes`, plus movements that merge into the same lane).

**Attributes**: `pairs` (`dict[str, set[tuple[int, int]]]`) intersection id to conflicting pairs
(both orders).  **Methods**: `conflicts(tls, a, b) -> bool` (a signal never conflicts with itself).

### `conflicts.get_table`, `conflicts.network_conflicts`

`get_table(net_file=DEFAULT_NET) -> ConflictTable` returns a cached table (built once per
process). `network_conflicts(tls, a, b)` is `get_table().conflicts` for the 5x5 grid.

**Examples**

```python
>>> from conflicts import get_table
>>> table = get_table("../network/net/single_4way.net.xml")
>>> len(table.pairs["C"])            # conflicting ordered pairs at the single junction
84
>>> table.conflicts("C", 0, 0)       # a signal never conflicts with itself
False
```

---

## Module `drivers`

### `drivers.DriverPool`

```python
drivers.DriverPool(n_drivers, types, default_carma, rng)
```

A fixed population of persistent drivers. Each driver gets a bidder type drawn in proportion to
the types' `share` and keeps it, its balance and its bid history for life.

| Method | Description |
|---|---|
| `checkout() -> Driver or None` | A random idle driver, now on a trip. `None` if everyone is driving (the vehicle is then a "guest" with a fresh balance; counted in `guest_trips`). |
| `checkin(driver, car)` | Take a finished trip's balance and results back. |
| `balances(cars) -> dict` | Every driver's current balance, including trips in progress. |
| `snapshot(t, cars) -> list[dict]` | Summary of the carma distribution (see `carma_stats`): one row for everyone and one per type. |
| `balance_rows(t, cars) -> list[dict]` | Every driver's balance at time `t`. |
| `to_frame(cars) -> DataFrame` | One row per driver (`trips`, `wins`, `start_carma`, `final_carma`, `mean_wait`, `mean_time_loss`). |

### `drivers.gini`, `drivers.carma_stats`

`gini(values) -> float`: inequality of a list of non-negative numbers, `0` = everyone equal,
`1` = one holder has everything. For ascending values and ranks `1..n`:
`G = 2 * sum(rank * value) / (n * total) - (n + 1) / n`.
`carma_stats(t, name, values) -> dict`: `mean`, `std`, `min`, `median`, `max`, `gini` and
`top10_share` (share held by the richest tenth).

**Examples**

```python
>>> from drivers import gini, carma_stats
>>> gini([1, 1, 1, 1])               # equal
0.0
>>> gini([0, 0, 0, 10])              # one holder has everything: (n-1)/n
0.75
>>> s = carma_stats(0, "all", [50, 100, 150])
>>> s["mean"], s["min"], s["max"], round(s["gini"], 3), s["top10_share"]
(100.0, 50, 150, 0.222, 0.5)
```

---

## Module `history`

### `history.AuctionHistory`

```python
history.AuctionHistory(max_per_tls=100)
```

Every finished auction, so strategies can learn from it. `carma.step_intersection` calls
`record` right after the payment rule runs, so history only contains auctions that have finished:
a strategy never sees the auction it is bidding in.

| Method | Description |
|---|---|
| `record(t, tls, bids, winners, deltas)` | Store an auction and append a `CarBidRecord` to each bidding car's `bid_history`. |
| `last(tls, n=None) -> list[AuctionOutcome]` | The most recent `n` auctions at an intersection, oldest first. |
| `winning_amounts(tls, n=None) -> list[float]` | Their winning amounts. |
| `to_frame() -> DataFrame` | Every bid in every auction, one row per bid (the full record, not limited by `max_per_tls`). |

### Records

| Class | Fields |
|---|---|
| `BidRecord` | `kind`, `members`, `links`, `member_bids`, `amount`, `won`, `net_carma` (the group's total carma change). |
| `AuctionOutcome` | `t`, `tls`, `bids`; properties `winning_amount`, `highest_losing_amount`. |
| `CarBidRecord` | `t`, `tls`, `own_bid`, `group_amount`, `won`, `carma_change`, `wait_time`, `top_other_amount` (best rival amount), `winning_amount`, `balance` and `urgency` when the car bid. |

**Notes**: "what if I had bid differently?" for a car bidding alone: any bid above
`top_other_amount` would have won. In a group the car only changes the group's amount
(`group_amount - own_bid + new_bid`).

**Examples**

```python
>>> from types import SimpleNamespace as NS
>>> from core import Bid
>>> from history import AuctionHistory
>>> a = NS(vid="a", carma=90.0, wait_time=3.0, urgency=1.0, bid_history=[])
>>> c = NS(vid="c", carma=110.0, wait_time=1.0, urgency=1.0, bid_history=[])
>>> ba = Bid(tls="C", members=[a], links={1}, member_bids={"a": 10.0}, amount=10.0)
>>> bc = Bid(tls="C", members=[c], links={5}, member_bids={"c": 4.0}, amount=4.0)
>>> h = AuctionHistory(max_per_tls=10)
>>> h.record(5.0, "C", [ba, bc], [ba], {"a": -10.0, "c": 10.0})
>>> h.winning_amounts("C")
[10.0]
>>> a.bid_history[0].top_other_amount, a.bid_history[0].won, a.bid_history[0].balance
(4.0, True, 100.0)
```

---

## Module `metrics`

| Function | Description |
|---|---|
| `finish_car(car, t) -> CarRecord` | Freeze a car's trip results when it leaves the network at time `t`. |
| `records_to_frame(records) -> DataFrame` | One row per `CarRecord` (becomes `_cars.csv`). |
| `summarize(cars) -> DataFrame` | Per bidder type: `cars`, `mean_travel_time`, `mean_wait_time`, `mean_time_loss`, `mean_wins`, `mean_start_carma`, `mean_final_carma` (becomes `_summary.csv`). Lower travel, wait and time loss is better. |

**Examples**

```python
>>> import pandas as pd
>>> cars = pd.read_csv("/tmp/demo/demo_cars.csv")          # written by the carma.run example
>>> (cars["urgency"] * cars["time_loss"]).mean()           # urgency-weighted time lost
```

---

## Module `blocking`

A signal link is **blocked** when a stopped car sits within `entrance_dist` metres of the start of
the lane it leads into: a car crossing the stop line would have nowhere to go.

| Function | Description |
|---|---|
| `build_out_lanes(tls_ids) -> dict` | Once at start-up: `{tls: {signal index: outgoing lane id}}` from TraCI's controlled links. |
| `find_blocked_links(controlled, out_lanes, cars, entrance_dist) -> dict` | Each step: `{tls: set of blocked signal indices}` from the cached car state (no extra TraCI calls). `entrance_dist=None` disables the check. |

---

## Module `signals`

### `signals.SignalController`

```python
signals.SignalController(tls_ids, yellow=3, all_red=0)
```

Turns "these signal indices should be green" into a safe state string (`G` green, `y` yellow,
`r` red) and sends it to SUMO.

**Methods**: `apply(tls, desired, t) -> None`. Call once per step per intersection with the full
set of links that should be green. Links not in `desired` are red; a link losing green shows yellow
for `yellow` seconds (then `all_red` seconds) before any new green starts. An **empty** `desired`
means "keep the last green", so an intersection never goes all red.

**Notes**: the guarantee "always at least one green or yellow link" holds while `all_red == 0`.

---

## Module `baselines`

Stronger signal baselines than the fixed 90 s program, chosen with `Experiment.control`. All of
them use the phases of the network's own static program (same movements, same `G` / `g`
priorities), so they differ from `default_lights` only in **when** phases switch. `carma.run`
sets them up; you only need this module to change their parameters.

### `baselines.program_file`

```python
baselines.program_file(net: str, control: str) -> str
```

Path of an additional file with a `control` (`'actuated'` or `'delay_based'`) program for every
signalised junction of `net`, written to `network/tls/<net>_<control>.add.xml` if it is missing or
older than the network. Green phases get `minDur = MIN_DUR` (5 s) and `maxDur = MAX_DUR` (50 s);
the controller parameters are SUMO's defaults (`PARAMS`). Junctions whose only phase is all-green
are left on their own program. `carma.run` passes the file to SUMO with `-a` and switches every
junction to it (`activate_programs`).

### `baselines.MaxPressure`

```python
baselines.MaxPressure(tls_ids, min_green=10.0, yellow=3.0)
```

Max-pressure control (Varaiya 2013) over the green phases of each junction's static program. The
pressure of a phase is the sum over the incoming lanes it serves of (vehicles on the lane) minus
(mean vehicles on the outgoing lanes it serves from that lane). Once the current phase has been
green for `min_green` seconds, the junction switches to the highest-pressure phase if it is
strictly higher than the current one; a switch shows `yellow` seconds of yellow on the links that
lose green. There is no maximum green.

**Methods**: `step(t) -> None`, once per simulation step. Must be created after `traci.start`
(it reads the programs and shows each junction's first green phase).

**Notes**: vehicle counts are whole-lane counts (`traci.lane.getLastStepVehicleNumber`), so on
long approaches moving vehicles far upstream count too.

---

## Module `learner_base`

### `learner_base.DecisionLearner`

```python
learner_base.DecisionLearner(base_balance=100.0, gamma=1.0, k_max=8, l_max=2, seed=0,
                             rho_rate=0.002)
```

Base class of learners A and B. It links a driver's decisions into transitions: each decision's
state `(x, k, l, u)` (balance / mean balance, auction-holding intersections left, auctions lost
here, urgency) leads to the driver's next decision, on the same trip or on its next trip. The
reward is minus urgency times the seconds that passed (to arrival for a trip's last decision).
With `gamma=1` (default) the learner minimises the average time per trip: every transition into a
new trip is charged `-rho`, the moving-average reward per trip. With `gamma < 1` it discounts per
trip instead. Pending decisions are kept by the learner, per driver, across days.

Hooks called by `carma.run`: `amount(car, ctx)`, `observe_auction(t, tls, bids, winners, deltas,
cars)`, `on_trip_end(car, t)`, `on_day_end(t)`. Used by `learning_study.py`: `start_day(day)`,
`frozen` (keep bidding, stop learning), `exploration() -> (level, floor)`, `policy_snapshot()`,
`save_state(path)`, `load_state(path)`. Subclasses implement `choose(s, car, ctx) -> (bid, info)`
and `learn(prev, reward, s_next, same_trip, dt)`.

---

## Module `best_response`

### `best_response.BestResponseLearner`

```python
best_response.BestResponseLearner(gamma=1.0, window=500, ema=0.02, t0=0.5, t_decay=0.9,
    t_min=0.02, x_max=50.0, x_step=0.1, k_max=8, l_max=2, alpha_min=0.01, alpha_power=0.6,
    monotone=True, base_balance=100.0, seed=0)
```

Learner A. Learns, per intersection, the highest rival bid (`win_curve(tls, bids_rel, lost)`: the
chance of facing no rival and of beating the best one, ties won with the learned chance per `l`),
the time to the next decision after a win, a loss and (at the last intersection) until arrival,
and the mean rebate; and the value `value(x, k, l, u)` by TD learning on a balance grid (step
`x_step` up to 3 times the mean, coarser up to `x_max`). `q_values(x, k, l, u, tls)` gives the
value of each of 40 candidate bids (0 and 0.5%-100% of the balance); the bid is drawn from
softmax(Q / T), with the temperature `t0 * t_decay^(day-1)` (seconds), never below `t_min`.
`greedy(x, k, l, u, tls=None)` returns the best (fraction, bid relative to the mean); `tls=None`
pools the intersections. `report(path)` writes the policy table.

```python
>>> import numpy as np
>>> from best_response import BestResponseLearner, FRACTIONS
>>> L = BestResponseLearner()
>>> for _ in range(100):
...     L.rivals["J"].append(0.2); L.faced["J"].append(True)      # every rival bids 0.2 x mean
>>> L.times["J"] = {"win": [10.0, 100], "lose": [30.0, 100], "arrive": [10.0, 100],
...                 "rebate": [0.0, 100]}
>>> for key, b0 in (((1, 0, 1.0), -50.0), ((2, 0, 1.0), -100.0), ((2, 1, 1.0), -120.0)):
...     L.V[key] = b0 + 20.0 * L.grid; L.visits[key] = np.ones(len(L.grid))
>>> fraction, bid = L.greedy(1.0, 2, 0, 1.0, "J")    # just outbid 0.2
>>> fraction == FRACTIONS[FRACTIONS > 0.2].min()
True
```

---

## Module `qlearning`

### `qlearning.QLearner`

```python
qlearning.QLearner(gamma=1.0, eps_start=0.2, eps_end=0.02, eps_days=40, alpha_min=0.01,
                   alpha_power=0.6, k_max=8, l_max=2, base_balance=100.0, seed=0)
```

Learner B: one shared Q-table over (balance bucket, k, l, u) x the 10 bid fractions of
`regret.ACTIONS`, semi-Markov Q-learning with the same transitions and objective as A,
epsilon-greedy with epsilon falling from `eps_start` to `eps_end` over days 1..`eps_days`.

---

## Module `prices`

### `prices.summarize`

```python
prices.summarize(decisions, bin_seconds, mean_balance, junction_types=None) -> DataFrame
```

The price table of one run from its decision log: one row per (intersection, time bin) with
`scope="bin"` and one per intersection with `scope="day"`. Columns: `n_auctions`, `n_contested`,
`share_contested`, `n_decisions`; mean, std and 10/25/50/75/90% quantiles of bids (`bid_rel_*`
relative to the mean balance, `bid_*` in carma) and of the winning bid in contested auctions
(`win_rel_*`, `win_*`, the price); `payment_mean`, `rebate_mean`, `share_all_zero`,
`n_rivals_mean`, `share_broke`, `cars_in_range_mean`, `balance_rel_mean`, `junction`.

---

## Module `convergence`

### `convergence.convergence_table`

```python
convergence.convergence_table(prices_by_day, learn_days, settings) -> DataFrame
convergence.Settings(window=14, active_auctions=50.0, m1_max=0.01, drift_max=0.05, share=0.9,
                     consecutive=7, min_days=60, min_visits=20)
convergence.policy_change(previous_snapshot, snapshot, min_visits=20) -> float | None
```

The stopping rule of the learning study (section 7 of the spec). One row per day: `m1` (policy
change), `m2_share` / `m3_share` (share of active intersections with no price drift / no
week-on-week change over the trailing window), `m3_network`, `m4` (exploration at its floor for
the whole window), `passed`, `consecutive` (passing days in a row) and `converged` (day >=
`min_days` and `consecutive` >= 7).

---

## Command-line scripts

All scripts can be run from the repository root. Defaults for `--workers` are the number of CPU
cores minus two. Finished runs (those with a `*_run.csv`) are skipped, so rerunning a command
resumes it. For worked examples see the [tutorial](tutorial.md).

### Study scripts (`controller/`)

#### `sweeps.py`

```
python controller/sweeps.py {carma,demand} [--seeds N ...] [--workers N]
       [--experiment NAME] [--values V ...]                         # carma sweep
       [--configs NAME ...] [--p-start P] [--p-step S] [--p-min P] [--p-values P ...]
       [--sequential] [--rerun] [--max-time T]                      # demand sweep
```

| Option | Default | Description |
|---|---|---|
| `sweep` | required | `carma`: starting balance 50..500; `demand`: lower `p` until each configuration jams. |
| `--seeds` | `0` | Seeds to repeat each run with. |
| `--experiment` | `MIXED` | Carma sweep: the experiment to run. |
| `--values` | 50..500 | Carma sweep: starting balances. |
| `--configs` | all `STUDY_CONFIGS` | Demand sweep: experiment names. |
| `--p-start`, `--p-step`, `--p-min` | `1.0`, `0.05`, `0.05` | Demand ladder: first `p`, decrement, last `p`. |
| `--p-values` | none | Demand sweep: run exactly these `p` values. |
| `--sequential` | off | One level at a time, stopping each configuration at its first jam (slower, fewer runs). |
| `--rerun` | off | Redo runs that already finished. |
| `--max-time` | `7800` | Stop every run at this simulated time (s). |

Output: `results/sweeps/{demand,initial_carma}/`. Analyse: `python analyze/sweep_analysis.py all`.

#### `persistence.py`

```
python controller/persistence.py [--seeds N ...] [--p-values P ...] [--configs NAME ...]
       [--modes {reset,persist} ...] [--drivers N] [--initial-carma X]
       [--no-default] [--stop-signs] [--only-stop-signs] [--workers N] [--rerun]
```

Single 4-way intersection, 4 simulated hours per run. `reset` starts every trip at the initial
balance; `persist` uses `--drivers` (default 300) persistent drivers. Defaults: seeds `0 1 2`,
`p` in `6 4 3 2.5 2`, seven strategies. Output: `results/persistence/`.

#### `multiday.py`

```
python controller/multiday.py {single,grid} [--configs NAME ...] [--p-values P ...]
       [--seeds N ...] [--days N] [--resets K ...] [--no-stop-signs]
       [--trace-interval S] [--results-name NAME] [--workers N]
```

Chains of `--days` (default 7) days: the same drivers, new trips each day, balances carried over.
For every carma strategy, reset chains are also run for each `--resets` interval (default `3 5`;
none = skip): every driver goes back to the starting balance on days 1, k+1, 2k+1, ... Only days
k+1 onwards are run (days 1..k equal the no-reset chain's); their files carry `_r<k>`.
`--trace-interval` sets the seconds between checkpoints (default: a quarter of the trip window,
i.e. 900 s single / 450 s grid); `--results-name` writes to `results/<NAME>/<scenario>` instead
of `results/multiday/<scenario>` (e.g. `multiday14` for the 14-day chains). Defaults: seeds `0 1 2`; `single`: `p` 3.0 and 2.0,
120 drivers; `grid`: `p` 1.0 and 0.9, 1500 drivers. Output: `results/multiday/<scenario>/`.

#### `urgency_study.py`

```
python controller/urgency_study.py [--seeds N ...] [--p-values P ...] [--carma X ...]
       [--configs NAME ...] [--workers N] [--rerun]
```

Single intersection, 2 simulated hours, urgency 1 (probability 0.8) or 6 (0.2), 300 persistent
drivers in every run (baselines and benchmarks included, so per-driver fairness is comparable).
Defaults: seeds `0 1 2`, `p` 3.0 and 2.5, carma strategies at starting balances 100 and 10; the
baselines are the four signal baselines, `stop_signs`, `urg_omniscient` and `urg_coin`.
Output: `results/urgency/`.

#### `lanes_study.py`

```
python controller/lanes_study.py [--seeds N ...] [--lanes N ...] [--bases B ...]
       [--configs NAME ...] [--workers N] [--rerun]
```

Single intersection with `--lanes` (default 1-5) lanes per road. The insertion period is
`base / lanes` seconds, with `--bases` (default 3.0, 2.5, 2.0). Defaults: seeds `0 1`; the four
carma strategies plus the four signal baselines (`default_lights`, `actuated_lights`,
`delay_based_lights`, `max_pressure`), `stop_signs`, `urg_omniscient`, `urg_coin`. Add
`lane_fast_release`, `lane_min_green` and `phase_urgency` through `--configs`. The networks
`network/net/single_<N>lane[_stop].net.xml` are generated on first use. Output: `results/lanes/`.

#### `regret_study.py`

```
python controller/regret_study.py [--configs NAME ...] [--p-values P ...] [--seeds N ...]
       [--days N] [--workers N]
```

5x5 grid, 1500 persistent drivers, 7-day chains with urgency 1 / 6. Defaults: seeds `0 1`, `p`
1.0 and 0.9; learners `regret_lane_0.5 / 1.0 / 2.0`, heuristics `only_lane_urgency` and
`urg_lane_wait`, and the baselines `default_lights`, `actuated_lights`, `delay_based_lights`,
`max_pressure` and `stop_signs`. Every run uses the persistent drivers. The extra experiments
(`regret_lane_4.0`, `regret_lane_8.0`, `frac_lane_0.9_0.15`, `frac_lane_0.9_0.05`,
`mix_learn20`, `mix_learn80`) are run with `--configs`. Output: `results/regret/`.

#### `learning_study.py`

```
python controller/learning_study.py [--scenarios {single,grid3,grid5} ...] [--configs NAME ...]
       [--seeds N ...] [--min-days 60] [--max-days 90] [--extend-to D] [--no-early-stop]
       [--baseline-days 7] [--no-evaluate] [--eval-days 7] [--window 14] [--consecutive 7]
       [--urgency U ...] [--urgency-probs P ...] [--bid-log {full,summary}] [--price-bin 300]
       [--trace-interval 60] [--p P] [--drivers N] [--results-name learning] [--workers N]
       [--mutants] [--mutant-share 0.1] [--mutant-days 14] [--mutant-start {warm,fresh}]
       [--mutant-config learn_br] [--initial-carma 100] [--carma-levels L ...] [--integer-carma]
```

`--initial-carma` sets every driver's starting balance (the learners measure balances against
it). `--carma-levels 10 20 40 ...` runs the whole study once per starting balance, each level in
`results/<results-name>/c<level>/<scenario>/`. `--integer-carma` makes carma whole units (bids,
payments and rebates; learner A then only considers whole bids).

Learning chains (learners `learn_br`, `learn_q`, `learn_hedge`; baselines `front_wait`,
`only_lane_urgency`, `actuated_lights`, `stop_signs`) on the single intersection (120 drivers,
p 3.0), the 3x3 (130 drivers, p 2.3) or the 5x5 (300 drivers, p 1.0). Learners run between
`--min-days` and `--max-days` days and stop when the convergence rule holds (unless
`--no-early-stop`); `--extend-to` continues unconverged chains. Then a frozen evaluation block on
reserved trips. `--mutants` runs the exploitability test on finished chains instead. Output:
`results/<results-name>/<scenario>/` (see the module docstring for the files).

### Analysis scripts (`analyze/`)

| Script | Arguments | Reads | Writes |
|---|---|---|---|
| `sweep_analysis.py` | `{carma,demand,all}` `[--show]` | `results/sweeps/` | tables `analyze/tables/sweeps/`, figures `analyze/figures/sweeps/` |
| `stop_sign_analysis.py` | none | `results/saved/sweeps_v1_single_winner`, `results/sweeps/demand`, `results/persistence` | `stop_signs_grid_demand.png`, `stop_signs_single.png`, `results/analysis/stop_signs_*.csv` |
| `multiday_analysis.py` | `[--results NAME] [single] [grid]` | `results/<NAME>/` (default `multiday`) | `<NAME>_<scenario>_{days,carma_finegrain,timeseries}.png`, `results/analysis/<NAME>_<scenario>_{by_age,by_reset,reset_paired,paired,by_week}.csv` (`by_week` for chains over 7 days). Reports mean wait and mean time lost; compare with stop signs on time lost. |
| `urgency_analysis.py` | none | `results/urgency/` | `urgency_time_lost.png`, `results/analysis/urgency_summary.csv`, `urgency_paired.csv` |
| `lanes_analysis.py` | none | `results/lanes/` | `lanes_cost.png`, `results/analysis/lanes_summary.csv`, `lanes_paired.csv` |
| `regret_analysis.py` | none | `results/regret/` | `regret_days.png`, `results/analysis/regret_by_day.csv`, `regret_paired.csv` |
| `carma_sweep_analysis.py` | `[--results learning_carma] [--scenario grid5]` | `results/<NAME>/c<level>/<scenario>/` and `results/<NAME>/baselines/<scenario>/` | `carma_sweep_{prices,bids,time_lost,by_intersection,evolution}.png/.pdf` in `analyze/figures/learning/`, tables `results/analysis/carma_sweep_{summary,by_intersection,paired}.csv` |
| `learning_analysis.py` | `[--results NAME] [--scenarios ...] [--no-gif]` | `results/<NAME>/` (default `learning`) | figures in `analyze/figures/learning/`, all amounts in carma (per scenario: `_bids_by_intersection`, `_bid_distribution`, `_learners`, `_empirical_policy`; per chain: `_prices_grid`, `_prices_weekly`, `_prices_map` (+ `.gif`), `_within_day`, `_convergence`, `_policy`, `_belief`; `equilibrium_prices`), tables `results/analysis/learning_{bids_summary,convergence_summary,equilibrium_prices,repeatability,empirical_policy,outcomes,paired}.csv` |
| `delay_metrics.py` | `[files ...] [--thresholds T ...] [--no-save]` | `*_cars.csv` (default: the newest in `results/`) | tables in `analyze/tables/` |
| `plot_delay.py` | `[files ...] [--thresholds T ...] [--show]` | `*_cars.csv` | figures in `analyze/figures/` |

#### `paired.py` (helper used by the analysis scripts)

```python
paired.paired(d, metric, reference, units, by=(), name="name", relative=True) -> DataFrame
paired.show(table, by=(), name="name", digits=1) -> str
```

Paired comparisons. Every strategy in a study sees the same traffic for the same seed (and day),
so strategies are compared unit by unit (`units`, e.g. `("seed",)` or `("seed", "day")`) against
`reference`, within each `by` group. `paired` returns one row per compared item with `n`,
`mean_diff` (percent of the reference if `relative`, otherwise absolute), `sd`, a 95% Student-t
interval (`ci_low`, `ci_high`), `better` (units on which the item was strictly lower) and both
means. `show` prints it compactly. Days of one chain are not fully independent, so with
`("seed", "day")` units the interval is somewhat optimistic; `better` (a sign-test count) is a
check on it.

```python
>>> import pandas as pd, sys; sys.path.insert(0, "analyze")
>>> from paired import paired
>>> d = pd.DataFrame({"name": ["a", "b"] * 3, "seed": [0, 0, 1, 1, 2, 2],
...                   "cost": [10, 9, 20, 18, 30, 27]})
>>> paired(d, "cost", "a", units=("seed",))[["name", "n", "mean_diff", "better"]]
  name  n  mean_diff  better
0    b  3      -10.0       3
```
