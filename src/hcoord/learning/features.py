"""Per (vehicle, request) feature extraction for the learned LLP.

Features are scalar-only for v1 (no sequence / GNN encoding of the route)
since the dispatcher only needs a cost score per (vehicle, request) pair
to pick the argmin. Route content is summarized through aggregates.

The extractor reads the *pre-insertion* vehicle state, so it must be
called before `apply_insertion` mutates the route.
"""

from __future__ import annotations

from typing import Any

from hcoord.demand import Request
from hcoord.fleet import Vehicle, return_arrival
from hcoord.travel import TravelTimeOracle


def extract_features(
    vehicle: Vehicle,
    request: Request,
    oracle: TravelTimeOracle,
) -> dict[str, Any]:
    """Flat scalar feature dict for one (vehicle, request) candidate.

    Naming: `req_*` for request-only, `veh_*` for vehicle-only,
    `pair_*` for (vehicle, request) interaction, `route_*` for
    summaries over the current route.
    """
    base_return = return_arrival(vehicle, oracle)

    tt_loc_to_origin = oracle.travel_time(vehicle.location, request.origin)
    tt_origin_to_dest = oracle.travel_time(request.origin, request.destination)
    tt_dest_to_home = oracle.travel_time(request.destination, vehicle.home)
    tt_loc_to_home = oracle.travel_time(vehicle.location, vehicle.home)

    # Lower-bound on cost when the route is empty: the request is appended
    # before return-home, so cost = tt_loc->origin + ride + tt_dest->home
    # minus the previously planned return tt_loc->home.
    delta_naive = tt_loc_to_origin + tt_origin_to_dest + tt_dest_to_home - tt_loc_to_home

    feats: dict[str, Any] = {
        # Request
        "req_origin": int(request.origin),
        "req_destination": int(request.destination),
        "req_announce_time": float(request.announce_time),
        "req_earliest_pickup": float(request.earliest_pickup),
        "req_latest_arrival": float(request.latest_arrival),
        "req_window_min": float(request.latest_arrival - request.earliest_pickup),
        "req_shift_id": int(request.shift_id),
        # Vehicle
        "veh_id": int(vehicle.id),
        "veh_capacity": int(vehicle.capacity),
        "veh_home": int(vehicle.home),
        "veh_location": int(vehicle.location),
        "veh_available_time": float(vehicle.available_time),
        "veh_route_len": int(len(vehicle.route)),
        "veh_base_return_time": float(base_return),
        "veh_slack_to_service_end": float(vehicle.service_end_time - base_return),
        # Pair
        "pair_tt_loc_to_origin": float(tt_loc_to_origin),
        "pair_tt_origin_to_dest": float(tt_origin_to_dest),
        "pair_tt_dest_to_home": float(tt_dest_to_home),
        "pair_delta_naive": float(delta_naive),
    }

    if vehicle.route:
        # Route-proximity features are over the planned trajectory the vehicle
        # will actually visit: current location, then every stop. This makes
        # the empty-route fallback below a true special case of the same
        # computation, not a parallel definition.
        zones = [vehicle.location] + [s.zone for s in vehicle.route]
        tts_to_origin = [oracle.travel_time(z, request.origin) for z in zones]
        tts_to_dest = [oracle.travel_time(z, request.destination) for z in zones]
        feats.update(
            {
                "route_min_tt_to_origin": float(min(tts_to_origin)),
                "route_mean_tt_to_origin": float(sum(tts_to_origin) / len(tts_to_origin)),
                "route_min_tt_to_dest": float(min(tts_to_dest)),
                "route_mean_tt_to_dest": float(sum(tts_to_dest) / len(tts_to_dest)),
                "route_last_zone": int(vehicle.route[-1].zone),
                "route_last_tt_to_origin": float(
                    oracle.travel_time(vehicle.route[-1].zone, request.origin)
                ),
                "route_n_pickups": int(sum(1 for s in vehicle.route if s.kind == "pickup")),
                "route_n_dropoffs": int(sum(1 for s in vehicle.route if s.kind == "dropoff")),
            }
        )
    else:
        feats.update(
            {
                "route_min_tt_to_origin": float(tt_loc_to_origin),
                "route_mean_tt_to_origin": float(tt_loc_to_origin),
                "route_min_tt_to_dest": float(
                    oracle.travel_time(vehicle.location, request.destination)
                ),
                "route_mean_tt_to_dest": float(
                    oracle.travel_time(vehicle.location, request.destination)
                ),
                "route_last_zone": int(vehicle.location),
                "route_last_tt_to_origin": float(tt_loc_to_origin),
                "route_n_pickups": 0,
                "route_n_dropoffs": 0,
            }
        )

    return feats


# Features dropped after step-1 review because they are constant in the
# current dispatch model: `pair_tt_loc_to_home` (vehicle.location only
# updates on hierarchical rebalance), `pair_pickup_buffer_min`
# (announce_time == earliest_pickup by construction in demand.py), and
# `veh_onboard` (no event loop ever advances it). Re-add when the model
# gains state that varies them.
FEATURE_NAMES: tuple[str, ...] = (
    "req_origin",
    "req_destination",
    "req_announce_time",
    "req_earliest_pickup",
    "req_latest_arrival",
    "req_window_min",
    "req_shift_id",
    "veh_id",
    "veh_capacity",
    "veh_home",
    "veh_location",
    "veh_available_time",
    "veh_route_len",
    "veh_base_return_time",
    "veh_slack_to_service_end",
    "pair_tt_loc_to_origin",
    "pair_tt_origin_to_dest",
    "pair_tt_dest_to_home",
    "pair_delta_naive",
    "route_min_tt_to_origin",
    "route_mean_tt_to_origin",
    "route_min_tt_to_dest",
    "route_mean_tt_to_dest",
    "route_last_zone",
    "route_last_tt_to_origin",
    "route_n_pickups",
    "route_n_dropoffs",
)
