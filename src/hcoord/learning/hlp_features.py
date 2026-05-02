"""Per-region state features for the learned HLP.

At each rebalance tick, the HLP sees: how many vehicles are currently
in each region, how busy those vehicles are, and (via the perfect-info
demand oracle) how many requests are coming in the lookahead window.
The learned HLP needs the same information in a flat scalar form.

For v1: produce a fixed-width per-region feature dict for up to
`MAX_REGIONS` slots. Unused slots get zeros so the model input shape
is fixed across K-sensitivity sweeps. The actual `n_regions` is
recorded separately as a context column.
"""

from __future__ import annotations

from typing import Any

from hcoord.fleet import Vehicle, return_arrival
from hcoord.regions import Partition
from hcoord.travel import TravelTimeOracle

# Padding bound. Memphis substrate has 5 hubs (so K ∈ [1, 5]). Bumping this
# requires retraining; keep it tight.
MAX_REGIONS: int = 5

# Per-region scalar features. The model input is the per-region block
# repeated across slots (PER_REGION_FEATURES * MAX_REGIONS scalars) plus
# the global features.
PER_REGION_FEATURES: tuple[str, ...] = (
    "n_vehicles",         # current count of vehicles assigned to this region
    "n_idle_now",         # vehicles with no route AND available_time <= now
    "n_demand_window",    # request count in [now, now + lookahead]
    "mean_route_len",     # mean route length over in-region vehicles
    "mean_slack_min",     # mean (service_end - return_arrival) — service slack
    "mean_tt_to_hub_min", # mean travel time from vehicle locations to nearest hub of region
    "supply_minus_demand",  # n_idle_now - n_demand_window (signed gap)
)

GLOBAL_FEATURES: tuple[str, ...] = (
    "now_min",
    "fleet_size",
    "n_regions",
    "rebalance_interval_min",
    "forecast_lookahead_min",
)


def per_region_features(
    region: int,
    *,
    fleet: list[Vehicle],
    region_of: dict[int, int],
    partition: Partition,
    oracle: TravelTimeOracle,
    demand_count: int,
    now: float,
) -> dict[str, float]:
    """Scalar features for a single region at a single tick."""
    in_region = [v for v in fleet if region_of[v.id] == region]
    n_vehicles = len(in_region)
    if n_vehicles == 0:
        return {
            "n_vehicles": 0.0,
            "n_idle_now": 0.0,
            "n_demand_window": float(demand_count),
            "mean_route_len": 0.0,
            "mean_slack_min": 0.0,
            "mean_tt_to_hub_min": 0.0,
            "supply_minus_demand": -float(demand_count),
        }
    idle = sum(1 for v in in_region if not v.route and v.available_time <= now + 1e-9)
    mean_route_len = sum(len(v.route) for v in in_region) / n_vehicles
    slacks = []
    tts = []
    region_hubs = partition.hub_groups[region]
    for v in in_region:
        slacks.append(v.service_end_time - return_arrival(v, oracle))
        tts.append(min(oracle.travel_time(v.location, h) for h in region_hubs))
    return {
        "n_vehicles": float(n_vehicles),
        "n_idle_now": float(idle),
        "n_demand_window": float(demand_count),
        "mean_route_len": float(mean_route_len),
        "mean_slack_min": float(sum(slacks) / len(slacks)),
        "mean_tt_to_hub_min": float(sum(tts) / len(tts)),
        "supply_minus_demand": float(idle - demand_count),
    }


def extract_hlp_state(
    *,
    fleet: list[Vehicle],
    region_of: dict[int, int],
    partition: Partition,
    oracle: TravelTimeOracle,
    demand_counts: dict[int, int],
    now: float,
    rebalance_interval_min: float,
    forecast_lookahead_min: float,
) -> dict[str, Any]:
    """Flat state dict: padded per-region block + global features.

    Per-region features for region r appear under keys `r{r}_{feat}`.
    Slots for r >= n_regions are filled with 0.0 (padding).
    """
    n_regions = partition.n_regions
    if n_regions > MAX_REGIONS:
        raise ValueError(
            f"n_regions={n_regions} exceeds MAX_REGIONS={MAX_REGIONS}; "
            "bump MAX_REGIONS and retrain"
        )

    state: dict[str, Any] = {
        "now_min": float(now),
        "fleet_size": float(len(fleet)),
        "n_regions": float(n_regions),
        "rebalance_interval_min": float(rebalance_interval_min),
        "forecast_lookahead_min": float(forecast_lookahead_min),
    }
    for slot in range(MAX_REGIONS):
        if slot < n_regions:
            feats = per_region_features(
                slot,
                fleet=fleet,
                region_of=region_of,
                partition=partition,
                oracle=oracle,
                demand_count=demand_counts.get(slot, 0),
                now=now,
            )
        else:
            feats = {name: 0.0 for name in PER_REGION_FEATURES}
        for name, value in feats.items():
            state[f"r{slot}_{name}"] = value
    return state


HLP_STATE_FEATURE_NAMES: tuple[str, ...] = tuple(
    list(GLOBAL_FEATURES)
    + [f"r{slot}_{feat}" for slot in range(MAX_REGIONS) for feat in PER_REGION_FEATURES]
)
