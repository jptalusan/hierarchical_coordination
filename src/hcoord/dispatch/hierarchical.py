"""Hub-catchment hierarchical dispatcher.

HLP (heuristic): every `rebalance_interval_min` minutes, target vehicle counts
per region are computed proportional to forecast demand over the next
`forecast_lookahead_min` minutes. Idle vehicles are moved from surplus regions
to deficit regions; the move charges the vehicle the travel time to its new
home hub.

LLP: per-region greedy insertion using the shared insertion primitive.

For v1 the demand forecast is a perfect oracle: the dispatcher is constructed
with the day's full request stream and counts requests in the lookahead window.
A real forecaster can be plugged in by replacing `_compute_targets`.
"""

from __future__ import annotations

from hcoord.demand import Request
from hcoord.dispatch.base import DispatchResult, Dispatcher
from hcoord.dispatch.insertion import apply_insertion, best_insertion
from hcoord.fleet import Vehicle
from hcoord.regions import Partition
from hcoord.travel import TravelTimeOracle


class HierarchicalDispatcher(Dispatcher):
    def __init__(
        self,
        *,
        fleet: list[Vehicle],
        oracle: TravelTimeOracle,
        partition: Partition,
        future_requests: list[Request],
        rebalance_interval_min: float = 30.0,
        forecast_lookahead_min: float = 60.0,
    ) -> None:
        super().__init__(fleet=fleet, oracle=oracle)
        self.partition = partition
        self._future = sorted(future_requests, key=lambda r: r.announce_time)
        self.rebalance_interval = rebalance_interval_min
        self.forecast_lookahead = forecast_lookahead_min
        self._region_of: dict[int, int] = {
            v.id: partition.region(v.location) for v in fleet
        }
        self._last_rebalance: float = -float("inf")

    def region_of_vehicle(self, vehicle_id: int) -> int:
        return self._region_of[vehicle_id]

    def vehicles_in(self, region: int) -> list[Vehicle]:
        return [v for v in self.fleet if self._region_of[v.id] == region]

    def assign(self, request: Request, now: float) -> DispatchResult:
        region = self.partition.region(request.origin)
        candidates = self.vehicles_in(region)
        for v in candidates:
            if v.available_time < now:
                v.available_time = now

        best = None
        for v in candidates:
            r = best_insertion(v, request, self.oracle)
            if r is None:
                continue
            if best is None or r.cost < best.cost:
                best = r

        if best is None:
            return DispatchResult(request_id=request.id, vehicle_id=None, cost=float("inf"))

        target = next(v for v in candidates if v.id == best.vehicle_id)
        apply_insertion(target, request, best)
        return DispatchResult(request_id=request.id, vehicle_id=best.vehicle_id, cost=best.cost)

    def rebalance(self, now: float) -> None:
        if now < self._last_rebalance + self.rebalance_interval:
            return
        self._last_rebalance = now

        targets = self._compute_targets(now)
        current: dict[int, int] = dict.fromkeys(range(self.partition.n_regions), 0)
        for v in self.fleet:
            current[self._region_of[v.id]] += 1

        while True:
            surplus = {r: current[r] - targets[r] for r in current if current[r] > targets[r]}
            deficit = {r: targets[r] - current[r] for r in current if current[r] < targets[r]}
            if not surplus or not deficit:
                break

            move = self._pick_move(surplus, deficit, now)
            if move is None:
                break
            vehicle, src, dst, new_hub, travel = move

            vehicle.available_time = now + travel
            vehicle.location = new_hub
            self._region_of[vehicle.id] = dst
            current[src] -= 1
            current[dst] += 1

    def _pick_move(
        self,
        surplus: dict[int, int],
        deficit: dict[int, int],
        now: float,
    ) -> tuple[Vehicle, int, int, int, float] | None:
        """Choose (vehicle, src, dst, new_hub, travel_time) for the next move."""
        for src in sorted(surplus, key=lambda r: surplus[r], reverse=True):
            idle = [
                v for v in self.vehicles_in(src)
                if not v.route and v.available_time <= now + 1e-9
            ]
            if not idle:
                continue
            dst = max(deficit, key=lambda r: deficit[r])
            dst_hubs = self.partition.hub_groups[dst]
            vehicle = min(
                idle,
                key=lambda vx: min(self.oracle.travel_time(vx.location, h) for h in dst_hubs),
            )
            new_hub = min(dst_hubs, key=lambda h: self.oracle.travel_time(vehicle.location, h))
            travel = self.oracle.travel_time(vehicle.location, new_hub)
            return vehicle, src, dst, new_hub, travel
        return None

    def _compute_targets(self, now: float) -> dict[int, int]:
        end = now + self.forecast_lookahead
        counts: dict[int, int] = dict.fromkeys(range(self.partition.n_regions), 0)
        for req in self._future:
            if req.announce_time < now:
                continue
            if req.announce_time >= end:
                break
            counts[self.partition.region(req.origin)] += 1

        fleet_size = len(self.fleet)
        n_regions = self.partition.n_regions
        total = sum(counts.values())
        if total == 0:
            base = fleet_size // n_regions
            extra = fleet_size - base * n_regions
            return {r: base + (1 if r < extra else 0) for r in range(n_regions)}

        raw = {r: fleet_size * c / total for r, c in counts.items()}
        floor = {r: int(v) for r, v in raw.items()}
        leftover = fleet_size - sum(floor.values())
        remainder = sorted(((raw[r] - floor[r], r) for r in raw), reverse=True)
        targets = dict(floor)
        for _, r in remainder[:leftover]:
            targets[r] += 1
        return targets
