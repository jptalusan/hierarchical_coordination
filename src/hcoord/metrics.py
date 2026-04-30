"""Per-run metrics for dispatcher experiments.

Two kinds of records:
- `DecisionRecord` — one per request, captures wall-clock cost of a single
  `dispatcher.assign` call and whether it succeeded.
- `RunMetrics` — aggregate across the day plus a `FleetSummary` snapshot of
  the final fleet state.

Because v1 uses deterministic travel times and feasibility-checked greedy
insertion, every assigned request also arrives on time, so `assignment_rate`
doubles as on-time-arrival rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from hcoord.fleet import Vehicle, return_arrival
from hcoord.travel import TravelTimeOracle


@dataclass(frozen=True)
class DecisionRecord:
    request_id: int
    announce_time: float
    vehicle_id: int | None
    cost: float
    wall_time_s: float


@dataclass(frozen=True)
class FleetSummary:
    n_vehicles: int
    n_active: int
    mean_route_length: float
    max_route_length: int
    total_deployment_min: float


@dataclass(frozen=True)
class RunMetrics:
    n_requests: int
    n_assigned: int
    assignment_rate: float
    mean_wall_ms: float
    median_wall_ms: float
    p95_wall_ms: float
    max_wall_ms: float
    total_wall_s: float
    fleet: FleetSummary
    decisions: list[DecisionRecord] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("decisions", None)
        return d


def compute_metrics(
    decisions: list[DecisionRecord],
    fleet: list[Vehicle],
    oracle: TravelTimeOracle,
) -> RunMetrics:
    n = len(decisions)
    n_assigned = sum(1 for d in decisions if d.vehicle_id is not None)

    if decisions:
        walls_ms = np.array([d.wall_time_s * 1000.0 for d in decisions])
        mean_w = float(walls_ms.mean())
        med_w = float(np.median(walls_ms))
        p95_w = float(np.percentile(walls_ms, 95))
        max_w = float(walls_ms.max())
        total_s = float(walls_ms.sum() / 1000.0)
    else:
        mean_w = med_w = p95_w = max_w = total_s = 0.0

    route_lengths = [len(v.route) for v in fleet]
    deployments = [return_arrival(v, oracle) for v in fleet]
    fleet_summary = FleetSummary(
        n_vehicles=len(fleet),
        n_active=sum(1 for L in route_lengths if L > 0),
        mean_route_length=float(np.mean(route_lengths)) if route_lengths else 0.0,
        max_route_length=max(route_lengths) if route_lengths else 0,
        total_deployment_min=float(sum(deployments)),
    )

    return RunMetrics(
        n_requests=n,
        n_assigned=n_assigned,
        assignment_rate=n_assigned / n if n > 0 else 0.0,
        mean_wall_ms=mean_w,
        median_wall_ms=med_w,
        p95_wall_ms=p95_w,
        max_wall_ms=max_w,
        total_wall_s=total_s,
        fleet=fleet_summary,
        decisions=decisions,
    )
