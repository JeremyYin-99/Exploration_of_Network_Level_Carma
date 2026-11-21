# Spec: learning equilibrium carma bids from repeated play

Status: implemented (revision 3), except the 7x7 grid, which is postponed. Phase P0 (smoke tests,
3x3 calibration) is done; phase P1 (single intersection) is running.

Changes in revision 3 (made while implementing and testing; details in the sections below):
- **Objective: average time per trip, no discount factor** (section 3). Discounting per decision
  made losing on purpose look profitable (every loss adds a decision, so the rest of the future is
  discounted once more); discounting per trip worked in principle but its overall value level
  converged over weeks, and while it was off, "finish the trip" versus "lose and decide again" was
  misjudged. Average cost per trip has neither problem. Discounting per trip remains an option.
- **Balance grid up to 50 times the mean** (S5, S6). With a grid up to 4 times the mean, carma
  concentrated in a few drivers whose balance was off the grid; they valued extra carma at nothing,
  dumped it into bids, and prices ran away (13 times the mean balance by day 10). Balances never
  visited now take the value of the nearest visited one.
- **The chance of winning a tie is learned** per number of losses (ties go to the longest wait).
- **Front-car bidding** is a flag on the bidder type (`front_only=True`) used with `group_by_lane`,
  not a separate grouper.
- **3x3 demand calibrated**: p = 2.3 with 130 drivers (the p ∝ 1/n guess, 1.67, jammed).
- **Exploitability test**: mutants start from the population's learned state by default (warm),
  and the comparison is a difference in differences (S10).
- The price of an intersection is the winning bid in **contested** auctions (an uncontested winner
  pays nothing); the convergence metrics have their own file.

Changes in revision 2:
- The 5-light corridor is replaced by **3x3 and 7x7 grids**. The study now covers four networks:
  single intersection, 3x3, 5x5 and 7x7 (S8, S9).
- **Bid tracking:** every decision bid is stored, per-intersection price summaries are written
  every day, and the price evolution is plotted for presentation (S11, S12).
- **No fixed day cap:** chains run at least 60 and up to 90 days, stop early (from day 60) only
  when the convergence rule holds, and can be extended past 90 days if they haven't converged
  (section 7, S8).

## 1. Goal

Drivers who each want to minimise **their own** travel time play the carma auction game every
day, with the same drivers on new trips, and adjust how they bid from experience. We want to
know:

- **Q1. Convergence.** Starting from naive bidding, does the population settle on a stable
  bidding policy, and how long does that take on each network?
- **Q2. The price at each intersection.** Once settled, what does priority cost at each
  intersection, and how does a driver's bid depend on their state (balance, intersections left,
  auctions already lost there, urgency)? How do prices differ across intersections and across
  network sizes, and is the difference explained by load and junction type? How does each
  intersection's price **evolve day by day**, from the start of learning to equilibrium?
- **Q3. Equilibrium.** Is the settled policy an equilibrium, meaning no driver can lower their
  own travel time by bidding differently?
- **Q4. Outcomes.** What does the equilibrium do to travel times (each driver's, the total, and
  fairness across drivers), compared with `only_lane_urgency`, the myopic learner
  (`RegretHedge`), actuated lights and stop signs?
- **Q5. Later.** Urgency that varies per trip, carma resets, per-intersection features, heavier
  demand, and posted per-intersection prices (adaptive karma tolls).

### Definition of done

1. **A resumable study command** that runs learning chains on four networks (single
   intersection, 3x3, 5x5, 7x7) under the run-length rule in section 7:
   - at least 60 and at most 90 days;
   - an early stop from day 60 when the convergence rule holds;
   - extension past 90 days when it doesn't.
2. **Every decision bid stored** with the bidder's state and the auction outcome, plus a
   per-intersection price summary for every day (S11).
3. **Presentation figures** showing how prices evolve over days at every intersection of every
   network, with the convergence day marked (S12).
4. **A converged learned policy**, saved per day as a table: bid by (balance, intersections
   left, auctions lost here, urgency), in carma and as a share of the mean balance.
5. **An exploitability test** showing that newcomers who learn against the frozen population
   cannot do significantly better than it (S10).
6. **Outcome comparisons** of travel time and fairness against the baselines, paired on common
   evaluation days (S8 evaluation block).
7. **Agreement** between the main learner and plain Q-learning, or a documented disagreement.
8. **Write-up:** results in the README, a tutorial section and API documentation.

### Out of scope (for now)

- Bayesian optimisation.
- Rerouting.
- Changes to signal logic beyond the front-car grouping in S2.
- Deep RL, unless the tabular state proves too coarse.
- Posted prices: that is Q5.

## 2. Approach: why not plain Q-learning

Three approaches can find the policy organically. All three share the plumbing in S1-S4.

| | A. Best-response learner (main) | B. Tabular Q-learning (control) | C. `RegretHedge` (existing) |
|---|---|---|---|
| Learns | The rival-bid distribution and the time cost of a win or loss at each intersection (measured), and the value of carma (TD) | One value per (state, bid) | Weights per (context, bid fraction) |
| Value of saved carma | Learned | Learned | Hand-set price `lam0` (myopic) |
| Uses "what if I had bid x" | Yes, exactly: every auction updates the whole win curve | No: only the bid actually made | Yes |
| Per-intersection price | Directly: the rival-bid distribution *is* the price | Indirect | Indirect |
| Expected speed | Fast | Slow (noisy max, one action per update) | Fast |
| Weak spot | Relies on a model of the auction (checked by B) | Sample-hungry, may cycle | Ignores the future |

**Why A is the main approach.** The auction rules are known: with front-car bidding a bid wins
if it beats the highest rival bid, the winner pays its bid, and losers get a rebate. Only three
things are unknown, and A learns each one separately:

1. The distribution of the highest rival bid at each intersection. This is measured directly from
   the auction outcomes, and it is the price of priority there.
2. How long the driver waits until its next decision after a win and after a loss, at each
   intersection. This is measured.
3. What a unit of carma is worth in future travel time. This is learned by temporal-difference
   (TD) learning. It is the only part that needs reinforcement learning, and it is what
   `RegretHedge` replaces with `lam0`.

Each driver then picks the bid that maximises its expected value. As drivers' bids change, the
rival-bid distributions change. The point where nothing changes any more is a stationary
equilibrium, the same kind of fixed point the karma papers compute, so the results are
comparable with theirs.

**Why B is kept.** B uses the same states with no auction model. If A and B reach the same bids,
the model in A is not driving the answer.

**Why C is kept.** C is the existing myopic baseline.

**Assumption.** Winning bids at each intersection are public, like a price board. A uses them to
learn the rival-bid distributions.

## 3. The game

- **Players.** N persistent drivers (the number per network is in S8). Every trip is driven by a
  random idle driver.
- **Decision points.** A driver decides only when it is the **front car** of its lane at an
  intersection's auction (see S2). If it loses, it decides again at the next auction there.
- **State** s = (x, k, l, u):
  - x = balance / mean balance. Total carma is conserved, so the mean balance stays at its
    starting value (100).
  - k = intersections left on the route, counting the current one (k = 1 at the last).
  - l = auctions already lost at this intersection (0, 1, 2+).
  - u = urgency of the trip (1 in phase 1; 1 or 6 in phase 2).
- **Action.** A bid between 0 and the balance.
- **Outcome at intersection j.** Let R be the highest rival bid.
  - **Win** if bid > R (ties go to the longest wait, S3): pay the bid. The next decision comes
    after a time t_win at the next intersection, or the trip ends if k = 1. The next state is
    (balance - bid, k - 1, 0).
  - **Lose** otherwise: receive a rebate (a share of the winner's payment). The next decision is
    at the same intersection after a time t_lose. The next state is (balance + rebate, k, l + 1).
- **Reward.** Minus u times the seconds until the next decision. For the last decision of a trip,
  it is the seconds until arrival.
- **Between trips.** The last decision of a trip leads to the driver's first decision on its next
  trip, with reward 0 for the gap. Leftover carma therefore keeps its value, and the trip end is
  not terminal.
- **Objective.** Each driver minimises its long-run **average time per trip** (urgency-weighted).
  The learners use relative ("differential") values: every transition into a new trip is charged
  minus rho, the average reward per trip, learned as a moving average of trip totals:
  target = reward - rho [new trip] + V(next state). Nothing is discounted. (Option: gamma < 1
  discounts per trip instead. Never per decision: every lost auction adds a decision, so the rest
  of the future would be discounted once more and losing on purpose would look profitable. Never per
  second of driving: a driver on the road always pays a second per second, so only the cost per trip
  can be lowered.)
- **Equilibrium.** A stationary policy pi* such that pi* is a best response when every other
  driver plays pi*. The test is exploitability (S10).

## 4. Components

Each item has an acceptance check. Existing studies must give identical results after S1-S4, so
every new option defaults to the current behaviour.

### S1. Trip features on Car ([car.py](../controller/car.py))

- At departure, store the full list of upcoming traffic lights (`traci.vehicle.getNextTLS`
  already returns all of them; `update()` keeps only the first), plus `route_len`.
- Each step, keep `intersections_left` (k) and `route_index`.
- Keep `lost_here`: auctions lost at the current intersection, reset when the intersection
  changes.
- Write a **trip-end record** with the arrival time into `bid_history`, so the last decision gets
  its reward.
- *Check:* on a known grid route, k counts down to 1 and `lost_here` resets at each new
  intersection.

### S2. Front-car grouping ([bidders.py](../controller/bidders.py), [auction.py](../controller/auction.py))

- Implemented as `BidderType(front_only=True)` with `group_by_lane` (config helper `front()`).
  There is one group per lane and every car in the lane is a member, so the grant serves the whole
  lane, but **only the front car bids**. Its bid is the group amount; followers bid 0 and pay 0.
- **Why:** with the current lane groups each member pays in proportion to its own bid, so a
  self-interested learner will bid 0 and let its lane-mates pay. Learners would converge to
  free-riding, not to the value of priority. Front-car bidding keeps lane-level green;
  vehicle-level auctions lost badly in the urgency study because of switching.
- **Rebates.** Option `rebate_to`:
  - `"bidders"` (proposed default for this study): losing front cars only, so income is tied to
    decisions.
  - `"members"` (current behaviour): every car in a losing lane.
- **Bid records.** Followers get no decision record, or a record flagged `decision=False` that
  learners skip.
- *Check:* total carma conserved (assert). With a waiting-time bid, time lost is close to
  `only_lane_urgency`'s on the same trips (paired), confirming the grouping didn't cost
  throughput.

### S3. Tie-break option ([auction.py](../controller/auction.py) `_rank`)

- `tie_break="random"` (default, current behaviour) or `"wait"` (longest-waiting bid first).
- Matters when everyone is broke and all bids are 0.
- *Check:* a toy auction with equal bids picks the longest wait.

### S4. Richer bid records ([history.py](../controller/history.py) `CarBidRecord`)

- Add `intersections_left`, `lost_here`, `trip_id`, `auction_id`, `n_rivals`, `rebate`,
  `decision` (bool), and the trip-end record from S1.
- *Check:* fields filled in a smoke run; the existing analyses still run.

### S5. Learner A: best response ([controller/best_response.py](../controller/best_response.py), new)

One learner object is shared by all drivers using it.

**Measured, per intersection j:**

- The rival-bid distribution F_j(b) = P(R < b), with R in units of the mean balance. It is an
  empirical distribution over the last W auctions at j (default W = 500), with an option to
  condition on the number of rivals.
- t_win,j and t_lose,j: moving averages (EMA) of the realised time to the next decision after a
  win and after a loss. Plus t_arrive,j for wins at a driver's last intersection.
- rebate_j: the mean rebate on a loss at j.

**Learned: the value function V(x, k, l, u).**

- A table over a balance grid with linear interpolation: steps of 0.1 up to 3 times the mean, then
  coarser steps up to 50 times the mean (carma can concentrate; see the revision 3 notes). By k
  (1 to 8+), l (0, 1, 2+) and u.
- Updated by TD(0) from realised transitions: V(s) += alpha * (r - rho [new trip] + V(s') - V(s)).
- Learning rate alpha = max(0.01, visits^-0.6).
- Between trips, s' is the first decision state of the driver's next trip.
- Balances never visited take the value of the nearest visited balance; an unseen (k, l, u) row
  starts as a copy of the nearest seen row. Unexplored states never look better than explored ones.
- Values are read as non-decreasing in the balance (more carma can never hurt); the stored values
  are left as learned.
- The chance of winning a tie (equal bids go to the longest wait) is learned per l.

**Choice.** For a grid of about 40 candidate bids between 0 and the balance (geometric spacing):

```
Q(s, b) = P_unc,j * [ -u * t_win,j + V(x, k - 1, 0, u) ]                         (free win)
        + (1 - P_unc,j) * ( F_j(b)       * [ -u * t_win,j  + V(x - b/m, k - 1, 0, u) ]
                          + (1 - F_j(b)) * [ -u * t_lose,j + V(x + rebate_j/m, k, l + 1, u) ] )
```

P_unc,j is the share of bidders at j who face no rival (an uncontested win is free). At k = 1 the
win branch uses -u * t_arrive,j - rho + E[V at the first decision of the next trip].

The bid is drawn from softmax(Q / T) (a smoothed best response). The temperature T anneals from
T0 by a factor of 0.9 per day down to T_min, which is reached by about day 30, well before the
earliest stop at day 60 (section 7). A temperature above zero gives a quantal-response
equilibrium; as T approaches 0 this approaches a Nash equilibrium. Smoothing guards against
cycling.

**State persistence.** The learner state (F, t, V) lives for the whole chain, as in
[regret_study.py](../controller/regret_study.py), and is saved each day to
`<label>_learner.json`. A chain can resume or be extended from it, and a frozen copy can be
loaded (S8 evaluation block, S10).

*Checks:*

- A unit test on a hand-made F and V gives the analytically optimal bid.
- At the single intersection with urgency 1 or 6, urgent drivers learn to bid more than calm
  ones.
- No NaN values, and carma is conserved.

### S6. Learner B: tabular Q-learning ([controller/qlearning.py](../controller/qlearning.py), new)

- One shared table Q[(x bin, k, l, u), a], with a = a share of the balance from the 10 values
  `regret.ACTIONS`.
- Semi-Markov update, since the time between decisions varies:
  Q(s,a) += alpha * (r + gamma * max Q(s', .) - Q(s,a)).
- Same rewards, same between-trip transitions, same gamma.
- Epsilon-greedy exploration, decaying from 0.2 to 0.02 over the **first 40 days**, then held at
  0.02. This is a fixed schedule rather than one tied to the chain length, which now varies.
- Learning lazily at the next decision, like `RegretHedge._learn`.
- Saved each day like S5.
- *Check:* it runs on the same plumbing and produces a policy table in the same format as S5.

### S7. Learner C: `RegretHedge` with front-car grouping

- A config entry only. It uses the same grouping, so all learners are compared on equal terms.

### S8. Study runner ([controller/learning_study.py](../controller/learning_study.py), new)

**Chains.** A chain is (scenario, seed, strategy), using the multiday pattern:

- persistent drivers;
- balances and learner state carried over between days;
- 60 s checkpoints;
- finished days skipped, so an interrupted chain resumes.

**Run length** (section 7 has the details):

- Each learning chain runs until it **converges** (at day 60 or later) or reaches `--max-days`
  (default 90).
- After each day from day 60 onward, the runner evaluates the convergence rule and writes the
  result to `<chain>_convergence.csv` and `<chain>_status.json`.
- A chain that reaches `--max-days` without converging stops and is marked `converged: false`.
- `--extend-to D` (for example 120) continues such chains from their saved learner state and
  balances, and the same rule keeps being checked.
- `--no-early-stop` runs to `--max-days` regardless.
- Trip seeds are 1000 * seed + day, so up to 999 learning days are available.

**Scenarios.** Demand scales with grid size: trip length grows with the grid side n while the
number of intersections grows with n², so the load per intersection stays about constant when
the trip rate is proportional to n, that is p ∝ 1/n. P0 calibrates p on each new grid so that
actuated lights give about the same per-intersection queueing as the 5x5 at p = 1.0. Driver
pools are sized for about 6 trips per driver per day.

| Scenario | Network | Signalised junctions (4-way / T / corner*) | Trip window | Demand p (start, calibrated in P0) | Drivers |
|---|---|---|---|---|---|
| `single` | `single_4way` (exists) | 1 / 0 / 0 | 3600 s | 3.0 | 120 |
| `grid3` | `grid3_200_tls` (built) | 1 / 4 / 4 | 1800 s | **2.3** (calibrated) | **130** |
| `grid5` | `RandomGrid5_200_tls` (exists) | 9 / 12 / 4 | 1800 s | 1.0 | 300 |
| `grid7` | `grid7_200_tls` (postponed) | 25 / 20 / 4 | 1800 s | 0.71 | 420 |

P0 calibration of the 3x3 (one day per level, actuated lights): p = 1.67 jammed (82 s time lost,
35% stopped; front-car bidding gridlocked); p = 2.0: 31 s, 15% stopped; p = 2.3: 23 s, 9-10%
stopped; p = 2.5: 21 s, 8%. The 5x5 at p = 1.0: 35 s, 10.6% stopped (its trips cross more
intersections). The p ∝ 1/n rule does not hold on these small grids, where most trips start and
end at the fringe.

\* Corner junctions join only two roads and usually have nothing to decide; the code already
skips junctions whose lights are all green.

- At the single intersection k = 1 always, so it tests the learner but not the dependence on
  intersections left.
- `guest_trips` (departures that found every driver busy) must stay 0 on every network.

**Strategies:**

- Learners A, B and C.
- Front-car waiting-time bid: the `only_lane_urgency` rule with front-car grouping.
- `only_lane_urgency`.
- Actuated lights.
- Stop signs.

Baselines that don't learn run only the evaluation block. Proposed allocation, to keep the 7x7
affordable:

| Scenario | A | B | C |
|---|---|---|---|
| single, 3x3, 5x5 | 3 seeds | 3 seeds | 3 seeds |
| 7x7 | 3 seeds | 1 seed | 1 seed |

**Evaluation block.** After a chain stops (converged or at the maximum), it runs 7 more days with
all learning frozen, on reserved evaluation trips (trip seeds 100000 + 1000 * seed + e, never
used for learning). Every baseline runs on the same evaluation trips, so outcomes are compared
paired by (seed, evaluation day), even when chains stopped on different days.

**Other settings:**

- Urgency off in phase 1; 1 or 6 in phase 2.
- Each learner's exploration random numbers are separate from the simulation's, so all
  strategies see identical trips.
- Bid logging as in S11.

**CLI**, following the other studies:

```
--scenario --configs --seeds --min-days 60 --max-days 90 --extend-to --no-early-stop
--evaluate --urgency --workers --results-name
```

### S9. Grid networks (`network/net/grid3_200_tls.net.xml`, `grid7_200_tls.net.xml`, plus stop-sign versions)

- Built with the same `netgenerate` options as the existing 5x5 (the options are recorded in
  its file header): `--grid --grid.number 3` (or `7`) `--grid.length 200
  --default.junctions.type traffic_light`, with `allway_stop` for the stop-sign versions.
- Actuated and delay-based programs are generated automatically by
  [baselines.py](../controller/baselines.py) for any network.
- Add `grid3` and `grid7` to the `SCENARIOS` table shared with
  [multiday.py](../controller/multiday.py), with trip windows, demand and drivers from S8.
- *Checks:*
  - the junction counts match the table in S8;
  - the trips' distribution of intersections left covers 1 to about 2n;
  - `guest_trips` is 0;
  - one day under actuated lights shows per-intersection queueing comparable to the 5x5.

### S10. Exploitability test (in `learning_study.py`, `--mutants`)

- Load the converged learner state and **freeze** it, so the population no longer learns.
- Designate 10% of drivers as **mutants**. By default they start from the population's learned
  state with exploration at its floor and keep learning against the frozen population for D_m days
  (default 14); `--mutant-start fresh --mutant-config learn_q` starts them from scratch instead
  (that also tests deviation by a different algorithm, but mixes in how fast newcomers learn).
- Run the same days a second time with the mutants also frozen, playing the population policy.
- Same drivers and same trips in both runs, but which driver takes which trip depends on arrival
  order, so the comparison is a **difference in differences** per day: (mutants - population) when
  the mutants learn, minus the same gap when they don't, relative to the mutants' frozen time lost.
  The direct difference (mutants learning vs frozen) is reported too.
- *Result:* the mutants' gain in percent with a 95% interval over days. If it is not significantly
  below 0%, or is within 2%, the policy is an approximate equilibrium.

### S11. Bid storage and price tracking (in carma.py / learning_study.py)

Bids are stored at three levels, from raw to summarised. All of them are written while the chain
runs, so the evolution can be watched and plotted before a chain finishes.

1. **Every decision bid, per day:** `<label>_bids.csv.gz`, one row per front-car bid.
   - Columns: day, t, tls, auction_id, driver, x, k, l, u, bid (carma), bid_rel (bid / mean
     balance), won, top_rival, winning_bid, payment, rebate, n_rivals, wait.
   - On by default for learning chains (`--bid-log full`); `--bid-log summary` keeps only levels
     2 and 3.
   - Expected size: roughly 1 MB per day on the 5x5 (about 20,000 bids), so about 100 MB per
     90-day chain; the 7x7 is 2-3 times that. The whole plan comes to a few GB. Keep these files
     out of git (see section 9).
2. **Per-intersection prices, per day:** `<label>_prices.csv`, written at the end of each day.
   - One row per (intersection, 300 s time bin), plus one whole-day row per intersection.
   - Columns:
     - n_auctions and n_decision_bids;
     - mean, std, and quantiles 10/25/50/75/90 of bids and of winning bids, both relative to the
       mean balance and in carma;
     - mean payment and mean rebate;
     - share of auctions with all bids at 0;
     - mean n_rivals;
     - share of front cars arriving broke;
     - mean queue length;
     - junction type (4-way, T, corner).
   - Small; always written; can be committed.
3. **Chain-level price history:** `<chain>_prices_by_day.csv`.
   - One row per (day, intersection), appended by the runner at the end of each day.
   - Contains the whole-day columns of level 2; the day's convergence metrics (section 7) are in
     `<chain>_convergence.csv`.
   - This is the table the evolution plots read.

**Learner snapshots, daily:** `<label>_learner.json`, containing:

- each intersection's learned rival-bid quantiles (A's belief, to set against what actually
  happened);
- t_win, t_lose and rebate_j;
- the V table;
- the policy table.

**Units.** Prices are reported as a share of the mean balance (the conserved carma supply), with
carma as a secondary unit. The share is what can be compared across networks and runs.

### S12. Analysis and figures ([analyze/learning_analysis.py](../analyze/learning_analysis.py), new)

The analysis can run at any time, including on chains still in progress (it reads level 3 of
S11). `--chain` plots one chain; the default plots everything. Figures are saved as PNG and PDF
(vector, for papers and slides).

**Price evolution (presentation figures):**

- **Per-intersection small multiples, laid out like the network** (3x3, 5x5 or 7x7 panels in grid
  position):
  - the daily median winning bid with 25-75% and 10-90% bands, against day;
  - a vertical line at the convergence day;
  - a shared y-axis so intersections can be compared.
- **Heatmap:** intersections (rows, ordered by grid position or by load) against day, coloured by
  the mean winning bid.
- **Network map snapshots:** the grid drawn with each junction coloured by its price, on day 1,
  10, 30 and the convergence day.
- **Animated GIF** of the map over all days, made with Pillow (ffmpeg isn't installed).
- **Within-day curves** for chosen days (from the 300 s bins): how the price rises and falls with
  congestion during the day.
- **Learners compared at the same intersection:** A, B and C's price evolution overlaid.
- **Across networks:** the distribution of equilibrium prices by junction type and against load,
  for single, 3x3, 5x5 and 7x7.
- **Belief against reality:** A's learned rival-bid quantiles against the realised ones.

**Convergence:**

- A per-chain dashboard of the section 7 metrics against day, with thresholds, the days each test
  passed and the stop day.
- The day of convergence per network and learner, as a table.

**Learner state, by day:**

- the policy table (bid by x and k, for l = 0) as a heatmap;
- V against balance for each k;
- the price of carma implied by V (the slope of V against balance).

**Outcomes** (from the evaluation block):

- mean and 95th-percentile time lost;
- per-driver cost Gini;
- time lost by state;
- paired against the baselines (reusing [paired.py](../analyze/paired.py)).

**Agreement and repeatability:**

- the policy tables of A, B and C compared on the same states;
- whether independent seeds converge to the same per-intersection prices.

### S13. Documentation and tests

- Unit tests for S2, S3, the S5 bid choice and the section 7 stopping rule (on made-up price
  series: flat, trending, noisy).
- A determinism check: the same seed gives identical cars.
- README findings, a tutorial section, and entries in [api.md](api.md).

## 5. Experiment phases

| Phase | What | Answers | Rough cost* |
|---|---|---|---|
| P0 (done) | Smoke tests; 3x3 built and calibrated; unit tests | Plumbing works, carma conserved exactly, no NaN values. Measured per simulated day: single 19 s, 3x3 about 25-45 s, 5x5 about 65 s (front-car bidding; learners a little slower) | done |
| P1 | Single intersection, A, B and C, urgency 1 and 1/6 | Does A converge, and how fast? Do A, B and C agree? Do urgent drivers bid more? | about 3 h |
| P2 | 3x3 | Does the bid depend on intersections left? How do prices differ between the 4-way centre and the T-junctions? | about 3 h |
| P3 | 5x5 | Prices at all intersections against load; outcomes against the baselines | about 9 h |
| P4 | 7x7 | Does the picture hold on a larger network? How do prices change with network size? | about 9-14 h |
| P5 | Evaluation blocks, mutants (S10), repeatability on all grids | Is the result an equilibrium? Is it repeatable? | about 15% on top |
| P6 | Extensions | Urgency 1 or 6 on the grids, a heavier demand level, resets, intersection features in V, posted prices | later |

\* Worst case of 90 days per chain, 8 workers, with the S8 allocation. Measured in the 14-day
run: a 5x5 day took about 3 minutes, and a single-intersection day about 1 minute at p=3.0.
Estimated: a 3x3 day about 1 minute, a 7x7 day 6-9 minutes. P0 replaces these estimates. Early
stops at day 60 save up to a third. Networks can share the 8 workers. The whole plan is about
25-30 hours of machine time in the worst case.

## 6. Default settings (to tune in P1)

| Setting | Default |
|---|---|
| Objective | average time per trip (gamma = 1); rho moving average rate 0.002 per trip |
| A: window W / EMA rate for times | 500 auctions / 0.02 |
| A: temperature T0, decay, T_min | 0.5 s, 0.9 per day, 0.02 s (T_min reached by about day 30) |
| A, B: learning rate | max(0.01, visits^-0.6) |
| B: epsilon | 0.2 to 0.02 over days 1-40, then 0.02 |
| Balance grid | step 0.1 up to 3 times the mean, coarser up to 50 times, interpolated |
| Candidate bids (A) | about 40, geometric, from 0 to the balance |
| Days | min 60, max 90, early stop by the section 7 rule; extend in steps of 30 if not converged |
| Convergence window | trailing 14 days; the rule must hold on 7 consecutive daily checks |
| Evaluation block | 7 days, learning frozen, reserved trip seeds |
| Drivers | 120 / 130 / 300 (single / 3x3 / 5x5); 7x7 postponed |
| Demand p | 3.0 / 2.3 (calibrated) / 1.0 |
| Urgency | 1 (phase 1); 1 or 6 with probabilities 0.8 / 0.2 (phase 2) |
| Bid log | full (every decision bid), gzip CSV |

## 7. Convergence, stopping and extension

### Daily metrics

These are written to `<chain>_convergence.csv` for every day.

- **Price of intersection j on day d:** P_j(d), the mean winning bid over the day's auctions at
  j, relative to the mean balance.
- **Active intersection:** at least 50 auctions per day on average. Only active intersections
  are tested.
- **M1. Policy change:** the mean absolute change in the policy's bid from the day before,
  relative to the mean balance, weighted by visits, over states visited at least 20 times on both
  days.
  - For A and B this is the greedy bid.
  - For C it is the expected bid fraction.
- **M2. Price drift:** over the trailing 14 days, the least-squares slope of P_j(d) against d.
  - drift_j = |slope| × 14 / mean P_j over the window.
  - Intersection j passes if drift_j < 5%, or if the slope's 95% interval includes 0.
- **M3. Week on week:** the difference between the mean P_j of the second and the first week of
  the window, with a Welch t-test on the 7 daily values each.
  - Intersection j passes if the 95% interval includes 0.
  - Network-wide, the mean of the per-intersection differences must also have an interval that
    includes 0.
- **M4. Exploration finished:** the temperature (A) or epsilon (B) was at its floor for the whole
  window.

### Stopping rule ("strong evidence of convergence")

A chain stops on day d if:

- d ≥ `--min-days` (60); and
- on **each** of the last 7 daily checks:
  - M1 < 1% of the mean balance;
  - M2 passes at 90% or more of the active intersections;
  - M3 passes at 90% or more of the active intersections, and network-wide;
  - M4 holds.

With the defaults, the first possible stop is day 60, and the evidence behind it spans days 41-60.

### Non-convergence and extension

- If the rule hasn't held by `--max-days` (90), the chain stops with status `converged: false`.
  The status file lists which tests failed and at which intersections.
- `--extend-to 120` (or higher) continues the chain from its saved state, and the rule keeps
  being checked daily.
- A chain whose prices still drift after extension is reported as not converged. That is itself
  a result: it could mean cycling, or a very slow drift.

### Success criteria for the study

These are tested after stopping.

- **Equilibrium:** the mutant gain (S10) is not significantly below 0, or is within 2%.
- **Repeatable:** across seeds, each intersection's mean equilibrium price varies by no more than
  its day-to-day spread.
- **Agreement:** A and B give the same policy within tolerance on frequently visited states. If
  they don't, report it and find out why.

The bid standard deviation is **not** expected to reach 0. Arrivals are random and balances
differ, so "converged" means a stable distribution, not a single bid.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Learning cycles or doesn't settle | Smoothed (softmax) choice with annealing; slower updates of the rival-bid distributions; if needed, freeze the population for a few days between updates (iterated best response). Non-convergence is detected and reported (section 7), not hidden |
| Early stop on a slow drift | The minimum of 60 days, a 14-day window, 7 consecutive passing checks, and the drift test (M2) on top of the week-on-week test (M3) |
| The stopping rule fires while exploration is still being reduced | M4: exploration must be at its floor for the whole window, and the floor is reached by about day 30 (A) or day 40 (B) |
| A's model is wrong (for example, rival bids depend on the driver's own state, or V should depend on the intersection) | B checks it; add a load class of the intersection to V's state; condition F_j on the number of rivals |
| B too slow | Expected; B is a check, not the main result. Option: counterfactual updates for B |
| Front-car grouping changes throughput | The S2 check against `only_lane_urgency` on the same trips |
| Followers benefit without paying | Accept: they decide when they reach the front. Report it as a property of the mechanism |
| Broke drivers starve | Rebates recover them; tie-break by wait; log the broke share; add a starvation guard only if needed (and log when it fires) |
| Equal urgency means bids barely affect total time | Expected (earlier studies). Q2 and Q3 are still answerable; phase 2 adds urgency |
| Demand not comparable across grid sizes | Scale p ∝ 1/n, then calibrate in P0 on per-intersection queueing under actuated lights |
| 7x7 is expensive (estimated 9-14 h per 90-day chain) | Allocation in S8 (fewer B and C seeds); early stop; measure the real cost in P0 before committing |
| Bid logs grow to several GB | Gzip; `--bid-log summary` option; keep full logs out of git |
| Long simulation times overall | Smaller networks for development, sized driver pools, finished days skipped, 8 workers shared across networks |

## 8b. Starting-carma sweep (added)

The 5x5 is run with every driver starting at 10, 20, 40, 80, 160, 320 and 640 carma, with
**whole units of carma** (`--integer-carma`): bids are whole numbers, a winner pays its whole bid,
and the payment is shared among the losing front cars as evenly as possible (leftover units go
one each to the longest-waiting losers), so total carma is exactly conserved. With fractional
carma the starting amount would only set the scale: all three learners bid fractions of their
balance and payments are proportional, so every bid and payment would scale with it (a test at
10 and 100 carma gave bids 8-10 times larger at 100). With whole units, a driver starting with 10
can only bid 0, 1, ..., 10 and many bids tie; with 640 bidding is nearly continuous. In a one-day
test learner A made 42% zero bids with 554 ties at 10 carma, and 12% zero bids with 45 ties at
640. The learners measure balances against the starting amount (their `base`), and learner A
considers every whole bid up to a balance of 40 and the rounded-down fractions above.

Run: learner A, `front_wait` and `only_lane_urgency` at every level, 3 seeds, 90 days without early
stopping, then the evaluation week; actuated lights and stop signs once (they don't use carma), on
the same trips. Analysis: `analyze/carma_sweep_analysis.py`.

## 8c. Extending the sweep to 180 days, with a price-only stopping rule (amended 2026-11-19)

Written before the extension was run.

**Why.** At day 90 none of the 21 learner-A chains met the section 7 rule.

- M1 can't pass with learner A as built. It has stayed at about 0.037 (3.7% of the mean balance)
  since day 30, against the 1% bar. Learner A adapts at a constant rate:
  - rival bids come from a sliding window of the last 500;
  - the TD step never falls below 0.01;
  - time costs are exponential averages (0.02).

  Its estimates therefore keep moving, and the greedy bid keeps switching between candidates of
  nearly equal value.
- The price tests (M2, M3) work on a network this size. On synthetic flat prices, with the
  sweep's day-to-day noise (median CV 14% per intersection) and 21 active intersections, the rule
  without M1 holds by day 64-76 in nearly all runs. So the M2/M3 failures in the sweep are real
  price movement:
  - network prices fell about 6% per 28 days over days 61-90, against about 4% over days 31-60;
  - only 3 of 21 chains met the rule without M1 by day 90.

**Rule for the extension (price-only).** A chain counts as converged on day d if:

- d ≥ 60; and
- on each of the last 7 daily checks, M2, M3 (per intersection and network-wide) and M4 pass.

Thresholds and windows are the same as section 7. M1 is still computed and reported, but is no
longer required. The question (Q2) is where the price settles, and the price tests measure that
directly. Section 7's rule still applies everywhere else in the study.

**Run.**

- All learner-A chains (7 levels × 3 seeds) continue from their saved day-90 state to day 180,
  without early stopping:
  `--configs learn_br --carma-levels 10 20 40 80 160 320 640 --integer-carma --no-early-stop
  --extend-to 180 --results-name learning_carma`.
- A new 7-day evaluation follows, with learning frozen, on the reserved trips. The baselines'
  evaluation days use the same trips, so they are not rerun and comparisons stay paired.
- The day-90 summaries and figures are kept as they were (`results/analysis/day90/`,
  `analyze/figures/learning/day90/`, `results/learning_carma/day90_snapshot/`).

**Reported.**

- The first day each chain meets the rule, and whether the rule still holds on day 180.
- Price per junction type at day 90 and at day 180.
- Drift over days 153-180.
- Time lost against actuated lights after day 180.

**Further extension.** Chains still drifting at day 180 are extended only if their drift is clearly
shrinking, and never past day 270. A steady drift at day 180 is reported as a result (for example,
prices sliding toward the floor where most auctions are ties settled by the longest wait).

**Outcome (2026-11-19).** All 21 chains reached day 180 (`results/analysis/carma_sweep_convergence.csv`).

- **The price stopped falling.** Mean drift across chains went from -6.8% per 28 days (days 63-90)
  to +0.5% (days 153-180; 95% interval -3.1% to +4.1%). Prices at 20-640 carma levelled off near
  18% (4-way) and 15% (T) of the starting balance; at 10 carma they may still be falling.
- **The rule still fails.** 9 of 21 chains met it at some point, but it holds on day 180 in none.
- **This is a limit of the test, not evidence of drift.** Each intersection's daily price is
  correlated with the day before (0.34). Synthetic prices that never change, given that
  correlation, show the same result: the rule never holds on day 180 and passes on 22% of days
  151-180, against 26% for the real chains. M2 and M3 treat days as independent.
- **No further extension.** It would not change the verdict. A test on weekly means, or with
  intervals that allow for correlation, is the open fix.
- **Travel time unchanged.** On the same trips, time lost after day 180 differs from day 90 by
  +0.5% (within noise).

## 8d. A further 30 days, to day 210 (requested 2026-11-19)

Written before the run. The 8c outcome said no further extension was needed for the verdict; the
extension is run anyway to see whether the remaining movement at single intersections and at 10
carma dies out.

**Run.** The same 21 chains continue from day 180 to day 210, without early stopping and without
an evaluation week (travel time did not change between day 90 and day 180; the evaluation can be
added later from the saved day-210 state): the 8c command with `--extend-to 210 --no-evaluate`.
The day-180 outputs are kept in `results/analysis/day180/`, `analyze/figures/learning/day180/` and
`results/learning_carma/day180_snapshot/`.

**Checked, against the same numbers at day 180.**

1. Network drift over days 183-210, mean of the 18 chains at 20-640 carma with a 95% interval
   across chains, measured as in `carma_sweep_convergence.csv` (mean winning bid; day 180: +1.3%,
   [-2.8, +5.3] per 28 days; the +0.5% in 8c is all 21 chains). The price counts as settled if
   the interval includes 0 and the mean is within ±5% per 28 days.
2. Per intersection: how many of the 21 have a last-28-day drift interval that includes 0 (day
   180: 17; about 20 if nothing moves), and whether B3, C4, E3 and E1 settle.
3. The gap between the highest and lowest of the 20-640 levels at each intersection, days
   197-210 (day 180: median 1.9, at most 2.3 points).
4. 10 carma: drift over the last 28 days and the price against the other levels.
5. The price-only rule (8c) is still reported, but cannot pass with correlated days, so it does
   not decide anything here.

The cap of day 270 (8c) still applies.

**Outcome (day 210, 2026-11-19).** All 21 chains reached day 210 (115 min on 16 workers).

1. Network drift over days 183-210, 18 chains: -2.1% per 28 days, [-6.0, +1.8] (day 180: +1.3%).
   Passes: the interval includes 0 and the mean is within ±5%.
2. 16 of 21 intersections are flat over the last 28 days (day 180: 17). The ones that move have
   changed: B3 +10, C4 -10, D2 -9, D4 -12, E1 -8 (% per 28 days). E3 settled and C4 reversed.
   Different intersections in each window, with both signs, is wandering, not a trend.
3. Gap between the 20-640 levels per intersection, days 197-210: median 1.8, at most 2.6 points
   (day 180: 1.9 and 2.3). It did not narrow further per intersection; per junction type (30-day
   blocks below) it narrowed to 1.3-1.5 points.
4. 10 carma is still drifting down: -1.0 points per 30-day block, [-3.5, +1.5] with 3 chains; its
   network price was 11.3% of the start in days 181-210.
5. The price-only rule was met at some point in 10 of 21 chains and holds on day 210 in none, as
   expected.

**Added after the plan: 30-day blocks.** Each chain's mean price per 30-day block, compared with
its previous block (paired within the chain, 95% interval across the 18 chains; chains are
independent, so day-to-day correlation does not bias the interval). Network change per block,
in points of starting carma: -0.1, -1.1, -0.9, -0.8, -0.7, then **-0.15 [-0.51, +0.22]** for days
181-210. By junction type, days 181-210: 4-way -0.19 [-0.68, +0.29]; T middle of an edge -0.37
[-0.73, -0.00]; T next to a corner +0.01 [-0.32, +0.34]. Intersections whose interval excludes 0:
12 of 21 at day 180 (all falling), 5 at day 210 (A2, B3, C4, D3 falling; D4 rising). Script:
`analyze/carma_sweep_analysis.py` (`block_changes`), table `results/analysis/carma_sweep_blocks.csv`.

**Correction to 8c.** The blocks show the price was still falling at day 180 (days 151-180 against
121-150: -0.74 points, [-1.14, -0.35], about 4% a month). The 28-day slope used in 8c was too
imprecise to see a fall of that size. The fall ended around day 165-180.

**Verdict.** At 20-640 carma the price has stopped falling, network-wide and per junction type,
to within about ±0.5 points a month: 4-way 18.2%, T middle of an edge 18.0%, T next to a corner
13.0% of starting carma (days 181-210). This rests on one flat block after five falling ones; a
second (days 211-240) would confirm it. At 10 carma the price is still falling.
*Superseded at day 240 (8e): the second block fell again, so this was a pause, not a plateau.*

## 8e. A second flat block, to day 240 (requested 2026-11-20)

Written before the run. The 8d verdict rests on one flat 30-day block after five falling ones.
This run adds a second block to confirm or overturn it.

**Run.** The same 21 chains continue from day 210 to day 240, without early stopping and without
an evaluation week: the 8d command with `--extend-to 240 --no-evaluate`. The day-210 outputs are
kept in `results/analysis/day210/`, `analyze/figures/learning/day210/` and
`results/learning_carma/day210_snapshot/`.

**Checked, all as 30-day block changes (days 211-240 against 181-210, paired within chain, 95%
interval across chains, in points of starting carma), against the day-210 numbers.**

1. Network, 18 chains at 20-640 carma (day 210: -0.15 [-0.51, +0.22]). The plateau is confirmed
   if the interval includes 0 and stays above -0.74, the size of the last falling block (days
   151-180). An interval that excludes 0 on the negative side means the fall has resumed.
2. Per junction type, same test (day 210: 4-way -0.19 [-0.68, +0.29]; T middle of an edge -0.37
   [-0.73, -0.00]; T next to a corner +0.01 [-0.32, +0.34]). The middle-of-an-edge T is the one to
   watch: it was borderline at day 210.
3. Per intersection: how many of the 21 have an interval that excludes 0 (day 210: 5; A2, B3, C4,
   D3 down, D4 up), and whether the same intersections keep moving in the same direction. The
   same movers in the same direction would be a slow trend; different ones would be wandering.
4. The gap between the highest and lowest of the 20-640 levels per junction type (day 210:
   1.3-1.5 points), and whether it keeps narrowing.
5. 10 carma: its block change (day 210: -1.0 [-3.5, +1.5]) and its price as % of start (day 210:
   11.3%).

If check 1 passes, no further extension is planned and the price per junction type is reported
from days 181-240. The cap of day 270 (8c) still applies.

**Outcome (day 240, 2026-11-20).** All 21 chains reached day 240 (116 min on 16 workers).

1. Network, 18 chains: **-0.84 [-1.22, -0.45]** points (-5.2%). Fails: the interval excludes 0
   and the change is as large as the falling blocks of days 61-180. The fall resumed. It is not
   an artefact of restarting the chains from saved state: the step from days 206-210 to 211-215 is
   -0.32 points against a typical 5-day change of -0.09 (sd across chains 0.79), and the fall is
   gradual (10-day means 16.05 for days 201-210, then 15.60, 15.31, 15.24).
2. Every junction type fell: 4-way -0.90 [-1.40, -0.40]; T middle of an edge -0.89 [-1.33,
   -0.46]; T next to a corner -0.74 [-1.06, -0.41]. Each of the 3 seeds fell at each type.
3. 15 of 21 intersections have an interval that excludes 0, all falling (day 210: 5, both signs).
   The rank correlation of per-intersection change with the previous block is -0.38: those that
   fell least in days 181-210 fell most now, i.e. wandering around a common decline.
4. Gap between the 20-640 levels per junction type, days 211-240: 4-way 1.7, T middle of an edge
   1.0, T next to a corner 1.2 points (day 210: 1.5, 1.3, 1.3). Every level fell (network -0.56 to
   -1.15 points); the starting levels move together.
5. 10 carma: -0.29 [-1.55, +0.96], 11.0% of start. That is about 1.1 carma per winning bid, next
   to the 1-unit floor of whole-unit bids, so its slowing is expected from the granularity and is
   not evidence of convergence.

**Shared trips.** All levels of a seed drive the same trips every day (route seed 1000 x seed +
day), so the 18 chains are not fully independent. The within-seed correlation of block changes is
near 0 for most blocks (-0.20 to +0.03) but +0.42 for days 151-180. With the 3 seeds as the units
(df 2): days 211-240 -0.84 [-1.67, -0.00], all three seeds falling (-1.22, -0.68, -0.61). Days
151-180: -0.74 [-2.24, +0.76], with one seed flat, so the 8d correction (still falling at day 180)
was less certain than its 18-chain interval suggested.

**Verdict.** The 8d plateau was a pause. From day 31-60 to day 211-240 the price fell 0.77 points
per 30-day block on average (one flat block in six), from 20.0% to 15.4% of starting carma. Apart
from the pause each block lost 4-6% of the current price, with no clear slowing. Learner A's
price has not settled by day 240 and where it would settle cannot be read from these runs. Report
it as a dated trajectory: days 211-240, 4-way 17.3%, T middle of an edge 17.1%, T next to a corner
12.3% of starting carma, the same at every starting level from 20 to 640 (gap at most 1.7
points). The day-270 cap allows one more block, which at this rate would show one more fall and
not settle the question.

Taken with the proposed defaults when implementation started (change them if you disagree):

1. Rebates go to losing front cars only (`rebate_to="bidders"`).
2. Phase 1 uses urgency 1 only (pure travel time); urgency 1 or 6 comes later (`--urgency 1 6
   --urgency-probs 0.8 0.2`).
3. Compute: the 7x7 is postponed; single, 3x3 and 5x5 with A, B, C on 3 seeds each.
4. Winning bids at each intersection are public (learner A reads them).
5. Full bid logs stay out of git (`results/learning*/**/*_bids.csv.gz` in `.gitignore`); the
   price summaries and convergence tables are kept.

Still open:

6. Are the convergence thresholds in section 7 strict enough for what you want to claim? (1%
   policy change, 5% drift, 90% of intersections, 7 consecutive checks.) With few intersections
   (single: 1, 3x3: 5) a single false alarm of a 5% test fails a day, so convergence is typically
   declared a few days after the earliest possible day. For the 5x5 starting-carma sweep, M1 was
   dropped from the rule (section 8c): learner A's constant-rate updates keep it near 3.7%.
