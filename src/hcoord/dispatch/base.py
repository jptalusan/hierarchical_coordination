"""Dispatcher interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hcoord.demand import Request
from hcoord.fleet import Vehicle
from hcoord.travel import TravelTimeOracle


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of an assignment attempt.

    `vehicle_id` is None if the request could not be served (no feasible
    insertion in any considered vehicle). `cost` is the increase in the
    chosen vehicle's return-home arrival time; `inf` for a dropped request.
    """

    request_id: int
    vehicle_id: int | None
    cost: float


class Dispatcher(ABC):
    """Base class. Concrete dispatchers implement `assign`; `rebalance` is a no-op by default."""

    def __init__(self, *, fleet: list[Vehicle], oracle: TravelTimeOracle) -> None:
        self.fleet = fleet
        self.oracle = oracle

    @abstractmethod
    def assign(self, request: Request, now: float) -> DispatchResult:
        """Try to insert `request` into some vehicle's route. Mutate on success."""

    def rebalance(self, now: float) -> None:
        """Optional fleet-level rebalancing tick."""
