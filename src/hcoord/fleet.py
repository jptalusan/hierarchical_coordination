"""Vehicles, route stops, and schedule/feasibility primitives.

A `Vehicle` carries a `home` depot it must return to by `service_end_time`. The
return-home segment is implicit: it's appended virtually during feasibility
checks so route construction code can't forget it.

Stops carry a hard time-window (`earliest`, `latest`) on service start. A route
is feasible iff every stop is reached by its `latest`, capacity stays in
[0, vehicle.capacity] throughout, and the vehicle arrives home by
`service_end_time`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hcoord.travel import TravelTimeOracle

StopKind = Literal["pickup", "dropoff"]


@dataclass(frozen=True)
class Stop:
    kind: StopKind
    zone: int
    request_id: int
    earliest: float
    latest: float
    service_time: float = 0.0

    @property
    def load_delta(self) -> int:
        return 1 if self.kind == "pickup" else -1


@dataclass
class Vehicle:
    id: int
    capacity: int
    home: int
    location: int
    available_time: float
    service_end_time: float
    route: list[Stop] = field(default_factory=list)
    onboard: int = 0


@dataclass(frozen=True)
class ScheduleEntry:
    stop: Stop
    arrival: float
    departure: float
    load_after: int


def schedule(vehicle: Vehicle, oracle: TravelTimeOracle) -> list[ScheduleEntry]:
    """Walk the vehicle's planned route and compute arrival/departure/load."""
    entries: list[ScheduleEntry] = []
    t = vehicle.available_time
    loc = vehicle.location
    load = vehicle.onboard
    for stop in vehicle.route:
        t += oracle.travel_time(loc, stop.zone)
        arrival = max(t, stop.earliest)
        load += stop.load_delta
        t = arrival + stop.service_time
        entries.append(ScheduleEntry(stop=stop, arrival=arrival, departure=t, load_after=load))
        loc = stop.zone
    return entries


def return_arrival(vehicle: Vehicle, oracle: TravelTimeOracle) -> float:
    """Time the vehicle arrives at its home depot if it executes its route."""
    sched = schedule(vehicle, oracle)
    if not sched:
        return vehicle.available_time + oracle.travel_time(vehicle.location, vehicle.home)
    last = sched[-1]
    return last.departure + oracle.travel_time(last.stop.zone, vehicle.home)


def feasible(vehicle: Vehicle, oracle: TravelTimeOracle, tol: float = 1e-9) -> bool:
    """True iff the route respects time windows, capacity, and end-of-day return."""
    sched = schedule(vehicle, oracle)
    for entry in sched:
        if entry.arrival > entry.stop.latest + tol:
            return False
        if entry.load_after < 0 or entry.load_after > vehicle.capacity:
            return False
    return return_arrival(vehicle, oracle) <= vehicle.service_end_time + tol
