"""Single-run experiment harness.

`run_experiment(cfg)` builds the network, demand, fleet, and dispatcher from
a config; streams requests in announce-time order; calls `rebalance` then
`assign` per request; and returns a `RunMetrics` snapshot.

This is the unit a hydra entrypoint or a Python sweep loop calls. It does
not need a discrete-event simulator: with deterministic travel times and
feasibility-checked greedy insertion, the planned route is also the realized
route.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from hcoord.demand import generate_requests
from hcoord.dispatch import HierarchicalDispatcher, MonolithicDispatcher
from hcoord.dispatch.base import Dispatcher
from hcoord.geography import build_memphis_outskirts
from hcoord.metrics import DecisionRecord, RunMetrics, compute_metrics
from hcoord.placement import make_fleet
from hcoord.regions import hub_catchment_partition, merge_nearest_hubs
from hcoord.travel import TravelTimeOracle


@dataclass
class ExperimentConfig:
    seed: int = 11

    # Network
    network: str = "synthetic"  # "synthetic" | "memphis_osm"
    n_outskirts: int = 25
    osm_inner_km: float = 8.0
    osm_outer_km: float = 40.0
    osm_cache_dir: str = "data/osm_cache"

    # Fleet
    fleet_size: int = 30
    capacity: int = 6
    service_end_time: float = 24 * 60.0
    placement: str = "hubs"

    # Demand
    intensity: float = 1.0
    base_rate: float = 0.4
    structural_zero_prob: float = 0.4
    announce_lead_min: float = 90.0
    arrival_buffer_min: float = 15.0

    # Dispatcher
    dispatcher: str = "monolithic"
    n_regions: int = 5
    rebalance_interval_min: float = 30.0
    forecast_lookahead_min: float = 60.0


def _build_dispatcher(
    cfg: ExperimentConfig,
    *,
    network: Any,
    oracle: TravelTimeOracle,
    fleet: list[Any],
    requests: list[Any],
) -> Dispatcher:
    if cfg.dispatcher == "monolithic":
        return MonolithicDispatcher(fleet=fleet, oracle=oracle)
    if cfg.dispatcher == "hierarchical":
        n_hubs = len(network.hubs)
        if not 1 <= cfg.n_regions <= n_hubs:
            raise ValueError(f"n_regions must be in [1, {n_hubs}], got {cfg.n_regions}")
        if cfg.n_regions == n_hubs:
            partition = hub_catchment_partition(network, oracle)
        else:
            partition = hub_catchment_partition(
                network,
                oracle,
                hub_groups=merge_nearest_hubs(network, oracle, cfg.n_regions),
            )
        return HierarchicalDispatcher(
            fleet=fleet,
            oracle=oracle,
            partition=partition,
            future_requests=requests,
            rebalance_interval_min=cfg.rebalance_interval_min,
            forecast_lookahead_min=cfg.forecast_lookahead_min,
        )
    raise ValueError(f"unknown dispatcher: {cfg.dispatcher!r}")


def _build_network(cfg: ExperimentConfig) -> Any:
    if cfg.network == "synthetic":
        return build_memphis_outskirts(seed=cfg.seed, n_outskirts=cfg.n_outskirts)
    if cfg.network == "memphis_osm":
        from hcoord.geography_osm import build_memphis_osm  # lazy import

        return build_memphis_osm(
            seed=cfg.seed,
            n_outskirts=cfg.n_outskirts,
            inner_radius_km=cfg.osm_inner_km,
            outer_radius_km=cfg.osm_outer_km,
            cache_dir=cfg.osm_cache_dir,
        )
    raise ValueError(f"unknown network: {cfg.network!r}")


def run_experiment(cfg: ExperimentConfig) -> RunMetrics:
    network = _build_network(cfg)
    oracle = TravelTimeOracle(network)
    requests = generate_requests(
        network,
        seed=cfg.seed,
        intensity=cfg.intensity,
        base_rate=cfg.base_rate,
        structural_zero_prob=cfg.structural_zero_prob,
        announce_lead_min=cfg.announce_lead_min,
        arrival_buffer_min=cfg.arrival_buffer_min,
    )

    placement_kwargs: dict[str, Any] = {}
    if cfg.placement == "demand_proportional":
        placement_kwargs["requests"] = requests

    fleet = make_fleet(
        network=network,
        fleet_size=cfg.fleet_size,
        capacity=cfg.capacity,
        service_end_time=cfg.service_end_time,
        placement=cfg.placement,
        **placement_kwargs,
    )

    dispatcher = _build_dispatcher(
        cfg, network=network, oracle=oracle, fleet=fleet, requests=requests
    )

    decisions: list[DecisionRecord] = []
    for req in requests:
        dispatcher.rebalance(now=req.announce_time)
        t0 = time.perf_counter()
        result = dispatcher.assign(req, now=req.announce_time)
        wall = time.perf_counter() - t0
        decisions.append(
            DecisionRecord(
                request_id=req.id,
                announce_time=req.announce_time,
                vehicle_id=result.vehicle_id,
                cost=result.cost,
                wall_time_s=wall,
            )
        )

    return compute_metrics(decisions, fleet, oracle)
