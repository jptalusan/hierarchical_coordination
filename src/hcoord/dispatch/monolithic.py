"""Monolithic dispatcher — considers every vehicle for every request."""

from __future__ import annotations

from hcoord.demand import Request
from hcoord.dispatch.base import DispatchResult, Dispatcher
from hcoord.dispatch.insertion import apply_insertion, best_insertion


class MonolithicDispatcher(Dispatcher):
    """Full-fleet greedy insertion. Baseline for the scalability comparison."""

    def assign(self, request: Request, now: float) -> DispatchResult:
        for v in self.fleet:
            if v.available_time < now:
                v.available_time = now

        best = None
        for v in self.fleet:
            r = best_insertion(v, request, self.oracle, observer=self.observer)
            if r is None:
                continue
            if best is None or r.cost < best.cost:
                best = r

        if best is None:
            return DispatchResult(request_id=request.id, vehicle_id=None, cost=float("inf"))

        target = next(v for v in self.fleet if v.id == best.vehicle_id)
        apply_insertion(target, request, best)
        return DispatchResult(request_id=request.id, vehicle_id=best.vehicle_id, cost=best.cost)
