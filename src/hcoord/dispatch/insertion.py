"""Exhaustive insertion primitive shared by all dispatchers.

For a candidate vehicle and request, enumerate every (pickup_position,
dropoff_position) pair, check route feasibility (capacity, time windows,
return-home deadline), and pick the cheapest by increase in return-home
arrival time.

This primitive is the apples-to-apples kernel both `MonolithicDispatcher`
and `HierarchicalDispatcher` use; the only difference between them is which
vehicles each considers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hcoord.demand import Request
from hcoord.fleet import Stop, Vehicle, feasible, return_arrival
from hcoord.travel import TravelTimeOracle

InsertionObserver = Callable[[Vehicle, Request, "InsertionResult | None"], None]

DEFAULT_PICKUP_SERVICE: float = 1.0
DEFAULT_DROPOFF_SERVICE: float = 1.0


@dataclass(frozen=True)
class InsertionResult:
    """Indices in the *original* route where pickup/dropoff would land.

    `pickup_at` is the position before which the pickup is inserted. The
    dropoff is inserted before original index `dropoff_at` (which must be
    >= pickup_at). `cost` is the increase in return-home arrival time.
    """

    vehicle_id: int
    pickup_at: int
    dropoff_at: int
    cost: float


def _build_stops(
    request: Request,
    pickup_service: float,
    dropoff_service: float,
) -> tuple[Stop, Stop]:
    pickup = Stop(
        kind="pickup",
        zone=request.origin,
        request_id=request.id,
        earliest=request.earliest_pickup,
        latest=request.latest_arrival,
        service_time=pickup_service,
    )
    dropoff = Stop(
        kind="dropoff",
        zone=request.destination,
        request_id=request.id,
        earliest=request.earliest_pickup,
        latest=request.latest_arrival,
        service_time=dropoff_service,
    )
    return pickup, dropoff


def _splice(route: list[Stop], p: int, q: int, pickup: Stop, dropoff: Stop) -> list[Stop]:
    return list(route[:p]) + [pickup] + list(route[p:q]) + [dropoff] + list(route[q:])


def best_insertion(
    vehicle: Vehicle,
    request: Request,
    oracle: TravelTimeOracle,
    *,
    pickup_service: float = DEFAULT_PICKUP_SERVICE,
    dropoff_service: float = DEFAULT_DROPOFF_SERVICE,
    observer: InsertionObserver | None = None,
) -> InsertionResult | None:
    """Cheapest feasible insertion, or None if no (p, q) pair is feasible.

    If `observer` is provided, it is called once at the end with
    `(vehicle, request, best)` (where `best` may be None) — useful for
    collecting training data without modifying the dispatch loop.
    """
    pickup, dropoff = _build_stops(request, pickup_service, dropoff_service)
    base_return = return_arrival(vehicle, oracle)
    n = len(vehicle.route)
    best: InsertionResult | None = None

    for p in range(n + 1):
        for q in range(p, n + 1):
            new_route = _splice(vehicle.route, p, q, pickup, dropoff)
            candidate = Vehicle(
                id=vehicle.id,
                capacity=vehicle.capacity,
                home=vehicle.home,
                location=vehicle.location,
                available_time=vehicle.available_time,
                service_end_time=vehicle.service_end_time,
                route=new_route,
                onboard=vehicle.onboard,
            )
            if not feasible(candidate, oracle):
                continue
            cost = return_arrival(candidate, oracle) - base_return
            if best is None or cost < best.cost:
                best = InsertionResult(
                    vehicle_id=vehicle.id,
                    pickup_at=p,
                    dropoff_at=q,
                    cost=cost,
                )

    if observer is not None:
        observer(vehicle, request, best)
    return best


def apply_insertion(
    vehicle: Vehicle,
    request: Request,
    result: InsertionResult,
    *,
    pickup_service: float = DEFAULT_PICKUP_SERVICE,
    dropoff_service: float = DEFAULT_DROPOFF_SERVICE,
) -> None:
    """Mutate `vehicle.route` to realize the insertion described by `result`."""
    if result.vehicle_id != vehicle.id:
        raise ValueError(
            f"insertion result is for vehicle {result.vehicle_id}, "
            f"called on vehicle {vehicle.id}"
        )
    pickup, dropoff = _build_stops(request, pickup_service, dropoff_service)
    vehicle.route = _splice(vehicle.route, result.pickup_at, result.dropoff_at, pickup, dropoff)
