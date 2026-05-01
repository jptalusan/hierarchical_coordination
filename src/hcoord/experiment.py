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

    # Learning data collection (optional). If `collect_to` is set, attach an
    # `InsertionCollector` to the dispatcher and dump rows to that CSV path
    # at the end of the run.
    collect_to: str | None = None

    # Learned scorer (optional). `scorer_mode="rank"` ranks candidates by
    # predicted cost and runs exhaustive (p, q) on the top-K (best when
    # cost ranking is reliable). `scorer_mode="filter"` drops candidates
    # the model is confident are infeasible (logit below threshold), then
    # exhaustive on the rest (best when the v1 model's feasibility head
    # is much more reliable than its cost ranking).
    scorer_path: str | None = None
    scorer_mode: str = "rank"
    scorer_top_k: int = 3
    scorer_filter_logit_threshold: float = -2.0


def _build_dispatcher(
    cfg: ExperimentConfig,
    *,
    network: Any,
    oracle: TravelTimeOracle,
    fleet: list[Any],
    requests: list[Any],
    observer: Any = None,
    scorer: Any = None,
) -> Dispatcher:
    if cfg.dispatcher == "monolithic":
        return MonolithicDispatcher(
            fleet=fleet, oracle=oracle, observer=observer,
            scorer=scorer,
            scorer_mode=cfg.scorer_mode,
            scorer_top_k=cfg.scorer_top_k,
            scorer_filter_logit_threshold=cfg.scorer_filter_logit_threshold,
        )
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
            observer=observer,
            scorer=scorer,
            scorer_mode=cfg.scorer_mode,
            scorer_top_k=cfg.scorer_top_k,
            scorer_filter_logit_threshold=cfg.scorer_filter_logit_threshold,
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

    collector = None
    if cfg.collect_to is not None:
        from hcoord.learning.collector import InsertionCollector  # lazy: viz extra

        # run_id makes (run_id, decision_id) globally unique across configs
        # so concatenated dumps can be grouped without collision.
        run_id = (
            f"s{cfg.seed}_n{cfg.n_outskirts}_f{cfg.fleet_size}"
            f"_i{cfg.intensity:g}_{cfg.dispatcher}"
        )
        if cfg.dispatcher == "hierarchical":
            run_id += f"_k{cfg.n_regions}"
        ctx = {
            "run_id": run_id,
            "seed": cfg.seed,
            "network": cfg.network,
            "n_outskirts": cfg.n_outskirts,
            "fleet_size": cfg.fleet_size,
            "intensity": cfg.intensity,
            "dispatcher": cfg.dispatcher,
            "n_regions": cfg.n_regions if cfg.dispatcher == "hierarchical" else 0,
        }
        collector = InsertionCollector(oracle, context=ctx)

    scorer = None
    if cfg.scorer_path is not None:
        from hcoord.learning.inference import LearnedScorer  # lazy: learn extra

        scorer = LearnedScorer.from_checkpoint(cfg.scorer_path)

    dispatcher = _build_dispatcher(
        cfg, network=network, oracle=oracle, fleet=fleet, requests=requests,
        observer=collector, scorer=scorer,
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

    if collector is not None and cfg.collect_to is not None:
        collector.write_csv(cfg.collect_to)

    return compute_metrics(decisions, fleet, oracle)
