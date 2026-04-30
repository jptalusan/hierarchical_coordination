"""hcoord — hierarchical decomposition for multi-modal DVRP."""

from hcoord.demand import DEFAULT_SHIFTS, Request, Shift, generate_requests
from hcoord.dispatch import (
    DispatchResult,
    Dispatcher,
    HierarchicalDispatcher,
    InsertionResult,
    MonolithicDispatcher,
    apply_insertion,
    best_insertion,
)
from hcoord.experiment import ExperimentConfig, run_experiment
from hcoord.fleet import ScheduleEntry, Stop, Vehicle, feasible, return_arrival, schedule
from hcoord.geography import Network, Zone, build_memphis_outskirts
from hcoord.metrics import DecisionRecord, FleetSummary, RunMetrics, compute_metrics
from hcoord.placement import list_strategies, make_fleet, make_placement
from hcoord.regions import Partition, hub_catchment_partition, merge_nearest_hubs
from hcoord.travel import TravelTimeOracle

__all__ = [
    "DEFAULT_SHIFTS",
    "DecisionRecord",
    "DispatchResult",
    "Dispatcher",
    "ExperimentConfig",
    "FleetSummary",
    "HierarchicalDispatcher",
    "InsertionResult",
    "MonolithicDispatcher",
    "Network",
    "Partition",
    "Request",
    "RunMetrics",
    "ScheduleEntry",
    "Shift",
    "Stop",
    "TravelTimeOracle",
    "Vehicle",
    "Zone",
    "apply_insertion",
    "best_insertion",
    "build_memphis_outskirts",
    "compute_metrics",
    "feasible",
    "generate_requests",
    "hub_catchment_partition",
    "list_strategies",
    "make_fleet",
    "make_placement",
    "merge_nearest_hubs",
    "return_arrival",
    "run_experiment",
    "schedule",
]
