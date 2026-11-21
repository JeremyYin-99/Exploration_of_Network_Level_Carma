import traci


class Car:
    """A vehicle in the simulation: SUMO-derived state plus a carma balance.

    A Car holds state only. It makes no bidding decisions. How much a car bids is decided by a
    BidStrategy (strategies.py), so strategies can be swapped without touching this class.

    Call update() once per simulation step to refresh the cached TraCI state. Everything else
    (strategies, groupers, auctions) reads these attributes and never calls TraCI itself.

    Attributes:
        vid (str): SUMO vehicle id.
        carma (float): Current carma balance. Only PaymentRule results should change this.
        lane (str): Lane the vehicle is on ('' while inside a junction).
        pos (float): Position along the lane.
        speed (float): Current speed.
        length (float): Vehicle length.
        accel (float): Maximum acceleration.
        wait_time (float): Seconds the vehicle has currently been stopped/waiting.
        tls (str | None): Id of the next traffic light on the vehicle's route.
        link (int | None): Signal index (position in the TLS state string) the vehicle needs.
        dist (float | None): Distance to that traffic light.
        intersections_left (int): Auction-holding intersections still ahead on the route,
            counting the next one (1 at the last). See `decision_tls`.
        route_len (int): intersections_left when the trip started.
        lost_at (dict): auctions this car lost at each intersection on this trip.
    """

    # Intersections that hold auctions in the current run (set by carma.run). Only these count
    # in `intersections_left`; lights with nothing to decide (all-green corners) are skipped.
    # Empty = count every traffic light ahead.
    decision_tls: frozenset = frozenset()

    def __init__(self, vid, carma=100, kind="default"):
        """Create the car for vehicle `vid` with a starting balance and bidder type, and read its
        first state from SUMO.
        """
        # SUMO's id for this vehicle (unique per vehicle; used as the key everywhere else).
        self.vid = vid
        # Name of the bidder type this car was assigned (decides how it bids; see BidderType).
        self.kind = kind
        # Simulation time at which the car entered the network (for travel time).
        self.depart = traci.simulation.getTime()
        # How many auctions this car has been part of the winning group in.
        self.wins = 0
        # This car's own bid in every auction it took part in, oldest first, with the outcome
        # (a list of CarBidRecord, see history.py). Strategies can learn from it.
        self.bid_history = []
        # Persistent-driver runs: the id of the Driver behind this car (None otherwise).
        self.driver = None
        # How much this trip's delay costs the driver (value of time). 1 unless the experiment
        # gives cars an urgency (Experiment.urgency_levels); strategies may read it.
        self.urgency = 1.0
        # The car's carma balance. Starts at `carma` and is changed only by payment rules.
        self.carma = carma
        # The balance the car started with (kept so the results can show how much it used).
        self.start_carma = carma
        # Declare the traffic-light fields up front so they always exist; update() fills them in.
        self.tls = None
        self.link = None
        self.dist = None
        # Auctions lost at each intersection on this trip (filled in by carma.step_intersection),
        # and how many auction-holding intersections are still ahead.
        self.lost_at = {}
        self.intersections_left = 0
        # Pull the first snapshot of state from SUMO so the car is usable immediately.
        self.update()
        # How many auction-holding intersections the whole route has.
        self.route_len = self.intersections_left

    @property
    def lost_here(self) -> int:
        """Auctions this car has already lost at its next intersection on this trip."""
        return self.lost_at.get(self.tls, 0)

    @property
    def route_index(self) -> int:
        """Position of the next intersection along the route (0 = the first one)."""
        return self.route_len - self.intersections_left

    def update(self):
        """Refresh cached state from the simulation. Call once per step."""
        # Local alias, just to keep the TraCI calls below short.
        vid = self.vid

        # Which lane the car is on right now ('' if it is inside the junction itself).
        self.lane = traci.vehicle.getLaneID(vid)
        # How far along that lane the car is, in metres from the lane start.
        self.pos = traci.vehicle.getLanePosition(vid)
        # Current speed in m/s.
        self.speed = traci.vehicle.getSpeed(vid)
        # Physical length of the vehicle in metres.
        self.length = traci.vehicle.getLength(vid)
        # Maximum acceleration of this vehicle type in m/s^2.
        self.accel = traci.vehicle.getAcceleration(vid)
        # Seconds the car has been (nearly) stopped in its current queue. Resets when it moves.
        self.wait_time = traci.vehicle.getWaitingTime(vid)
        # Total seconds spent waiting over the whole trip so far, and SUMO's "time loss" (seconds
        # lost against driving at full speed). Cached here because a vehicle can't be queried
        # once it has arrived; the last values seen become its trip results.
        self.acc_wait = traci.vehicle.getAccumulatedWaitingTime(vid)
        self.time_loss = traci.vehicle.getTimeLoss(vid)

        # getNextTLS returns [(tlsID, linkIndex, distance, state), ...] ordered by distance.
        # Once a car crosses the stop line its next TLS becomes the following light, so a car
        # that has cleared an intersection automatically leaves that intersection's approach.
        upcoming = traci.vehicle.getNextTLS(vid)

        # If there is at least one traffic light ahead on the route, remember the nearest one:
        #   tls  = which intersection it is,
        #   link = which signal index in that intersection's state string this car needs
        #          (i.e. which movement: left / straight / right from this lane),
        #   dist = metres until the stop line. The 4th value (current light colour) is unused.
        if upcoming:
            self.tls, self.link, self.dist, _ = upcoming[0]
        # No traffic light left on the route: clear the fields so nothing treats this car as
        # waiting at an intersection.
        else:
            self.tls = self.link = self.dist = None
        # Auction-holding intersections still ahead, counting the next one.
        if Car.decision_tls:
            self.intersections_left = sum(1 for light in upcoming if light[0] in Car.decision_tls)
        else:
            self.intersections_left = len(upcoming)

    def update_light(self):
        """A cheaper update for runs where nothing bids (default traffic lights): only the speed
        and the trip totals that the results need, not the lane or next traffic light."""
        vid = self.vid
        self.speed = traci.vehicle.getSpeed(vid)
        self.acc_wait = traci.vehicle.getAccumulatedWaitingTime(vid)
        self.time_loss = traci.vehicle.getTimeLoss(vid)

    def __str__(self):
        """A short description for debugging."""
        # Human-readable one-line summary, handy for debugging prints.
        return (f"Car({self.vid}, carma={self.carma:.1f}, lane={self.lane}, "
                f"pos={self.pos:.1f}, speed={self.speed:.1f}, wait={self.wait_time:.0f})")
