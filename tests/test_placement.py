from collections import Counter

import pytest

from hcoord.demand import generate_requests
from hcoord.geography import build_memphis_outskirts
from hcoord.placement import list_strategies, make_fleet, make_placement


def test_strategies_registered():
    s = list_strategies()
    assert "hubs" in s
    assert "demand_proportional" in s


def test_hub_placement_round_robin():
    net = build_memphis_outskirts()
    locs = make_placement("hubs", network=net, fleet_size=12)
    hub_ids = {h.id for h in net.hubs}
    assert len(locs) == 12
    counts = Counter(locs)
    assert set(counts) == hub_ids
    assert max(counts.values()) - min(counts.values()) <= 1


def test_demand_proportional_targets_outskirts():
    net = build_memphis_outskirts()
    reqs = generate_requests(net, seed=11)
    locs = make_placement("demand_proportional", network=net, fleet_size=20, requests=reqs)
    outskirts = {o.id for o in net.outskirts}
    assert len(locs) == 20
    assert all(loc in outskirts for loc in locs)


def test_demand_proportional_falls_back_without_requests():
    net = build_memphis_outskirts()
    hub_ids = {h.id for h in net.hubs}
    locs = make_placement("demand_proportional", network=net, fleet_size=5, requests=[])
    assert all(loc in hub_ids for loc in locs)


def test_demand_proportional_volume_tracks_demand():
    net = build_memphis_outskirts()
    reqs = generate_requests(net, seed=11, intensity=3.0)
    locs = make_placement("demand_proportional", network=net, fleet_size=50, requests=reqs)
    demand_by_origin = Counter(r.origin for r in reqs)
    placed_by_origin = Counter(locs)
    top_demand = max(demand_by_origin, key=demand_by_origin.get)
    assert placed_by_origin[top_demand] >= 1


def test_make_fleet_initializes_correctly():
    net = build_memphis_outskirts()
    fleet = make_fleet(network=net, fleet_size=10, capacity=4, service_end_time=1000.0)
    assert len(fleet) == 10
    for v in fleet:
        assert v.location == v.home
        assert v.available_time == 0.0
        assert v.capacity == 4
        assert v.route == []


def test_make_fleet_propagates_placement_kwargs():
    net = build_memphis_outskirts()
    reqs = generate_requests(net, seed=11)
    fleet = make_fleet(
        network=net,
        fleet_size=10,
        capacity=4,
        service_end_time=1000.0,
        placement="demand_proportional",
        requests=reqs,
    )
    outskirts = {o.id for o in net.outskirts}
    assert all(v.home in outskirts for v in fleet)


def test_unknown_strategy_rejected():
    net = build_memphis_outskirts()
    with pytest.raises(KeyError):
        make_placement("nonexistent", network=net, fleet_size=5)


def test_negative_fleet_size_rejected():
    net = build_memphis_outskirts()
    with pytest.raises(ValueError):
        make_placement("hubs", network=net, fleet_size=-1)


def test_zero_fleet_returns_empty():
    net = build_memphis_outskirts()
    assert make_placement("hubs", network=net, fleet_size=0) == []
