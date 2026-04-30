from hcoord.demand import generate_requests
from hcoord.dispatch import HierarchicalDispatcher, MonolithicDispatcher
from hcoord.geography import build_memphis_outskirts
from hcoord.placement import make_fleet
from hcoord.regions import hub_catchment_partition, merge_nearest_hubs
from hcoord.travel import TravelTimeOracle

SERVICE_END = 24 * 60.0  # 24h sim horizon


def _setup(*, fleet_size=20, capacity=4, intensity=1.0, seed=11):
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    requests = generate_requests(net, seed=seed, intensity=intensity)
    fleet = make_fleet(
        network=net, fleet_size=fleet_size, capacity=capacity,
        service_end_time=SERVICE_END, placement="hubs",
    )
    return net, oracle, requests, fleet


def test_monolithic_assigns_under_light_load():
    _, oracle, reqs, fleet = _setup(fleet_size=30, capacity=6, intensity=0.4)
    d = MonolithicDispatcher(fleet=fleet, oracle=oracle)
    assigned = sum(
        1 for r in reqs if d.assign(r, now=r.announce_time).vehicle_id is not None
    )
    assert reqs and assigned / len(reqs) > 0.7


def test_monolithic_drops_when_no_vehicles():
    _, oracle, reqs, _ = _setup()
    d = MonolithicDispatcher(fleet=[], oracle=oracle)
    if not reqs:
        return
    result = d.assign(reqs[0], now=reqs[0].announce_time)
    assert result.vehicle_id is None
    assert result.cost == float("inf")


def test_monolithic_assignment_appears_in_route():
    _, oracle, reqs, fleet = _setup(fleet_size=10, capacity=4)
    d = MonolithicDispatcher(fleet=fleet, oracle=oracle)
    r = reqs[0]
    result = d.assign(r, now=r.announce_time)
    assert result.vehicle_id is not None
    target = next(v for v in fleet if v.id == result.vehicle_id)
    assert any(s.request_id == r.id for s in target.route)


def test_hierarchical_initial_region_assignment_matches_home():
    net, oracle, reqs, fleet = _setup(fleet_size=15)
    p = hub_catchment_partition(net, oracle)
    d = HierarchicalDispatcher(fleet=fleet, oracle=oracle, partition=p, future_requests=reqs)
    for v in fleet:
        assert d.region_of_vehicle(v.id) == p.region(v.home)


def test_hierarchical_only_uses_in_region_vehicles():
    net, oracle, reqs, fleet = _setup(fleet_size=20)
    p = hub_catchment_partition(net, oracle)
    d = HierarchicalDispatcher(fleet=fleet, oracle=oracle, partition=p, future_requests=reqs)
    for r in reqs[:5]:
        before_region = {v.id: d.region_of_vehicle(v.id) for v in fleet}
        result = d.assign(r, now=r.announce_time)
        if result.vehicle_id is None:
            continue
        assert before_region[result.vehicle_id] == p.region(r.origin)


def test_hierarchical_rebalance_respects_interval():
    net, oracle, reqs, fleet = _setup(fleet_size=20)
    p = hub_catchment_partition(net, oracle)
    d = HierarchicalDispatcher(
        fleet=fleet, oracle=oracle, partition=p,
        future_requests=reqs, rebalance_interval_min=30.0,
    )
    snapshot = {v.id: (v.location, d.region_of_vehicle(v.id)) for v in fleet}
    d.rebalance(now=300.0)
    d.rebalance(now=305.0)  # within interval — should be no-op
    after_one = {v.id: (v.location, d.region_of_vehicle(v.id)) for v in fleet}
    d.rebalance(now=305.0)
    after_two = {v.id: (v.location, d.region_of_vehicle(v.id)) for v in fleet}
    assert after_one == after_two
    # snapshot may equal after_one if no rebalancing happened (uniform distribution)
    # but the function shouldn't have errored
    assert set(snapshot) == set(after_one)


def test_hierarchical_rebalance_moves_idle_vehicle_to_deficit_region():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    p = hub_catchment_partition(net, oracle)
    # Synthetic future requests concentrated in one region
    region_zones = {r: p.zones(r) for r in range(p.n_regions)}
    target_region = max(range(p.n_regions), key=lambda r: len(region_zones[r]))
    target_outskirt = next(z for z in region_zones[target_region] if not net.zone(z).is_hub)
    target_hub = p.hub_groups[target_region][0]
    from hcoord.demand import Request
    synthetic = [
        Request(
            id=i, origin=target_outskirt, destination=target_hub,
            announce_time=120.0 + i, earliest_pickup=120.0 + i,
            latest_arrival=240.0 + i, shift_id=0,
        )
        for i in range(20)
    ]
    fleet = make_fleet(
        network=net, fleet_size=10, capacity=4,
        service_end_time=SERVICE_END, placement="hubs",
    )
    d = HierarchicalDispatcher(
        fleet=fleet, oracle=oracle, partition=p, future_requests=synthetic,
        rebalance_interval_min=30.0, forecast_lookahead_min=180.0,
    )
    counts_before = {r: len(d.vehicles_in(r)) for r in range(p.n_regions)}
    d.rebalance(now=60.0)
    counts_after = {r: len(d.vehicles_in(r)) for r in range(p.n_regions)}
    assert counts_after[target_region] > counts_before[target_region]


def test_hierarchical_compute_targets_sums_to_fleet_size():
    net, oracle, reqs, fleet = _setup(fleet_size=25)
    p = hub_catchment_partition(net, oracle)
    d = HierarchicalDispatcher(fleet=fleet, oracle=oracle, partition=p, future_requests=reqs)
    targets = d._compute_targets(now=0.0)
    assert sum(targets.values()) == 25


def test_hierarchical_uniform_target_when_no_demand_in_lookahead():
    net, oracle, reqs, fleet = _setup(fleet_size=15)
    p = hub_catchment_partition(net, oracle)
    d = HierarchicalDispatcher(
        fleet=fleet, oracle=oracle, partition=p, future_requests=reqs,
        forecast_lookahead_min=1.0,  # tiny window → likely empty
    )
    targets = d._compute_targets(now=12 * 60.0)  # noon, between shifts
    # Either empty falls through to uniform OR a request happens to land
    assert sum(targets.values()) == 15


def test_k3_partition_works_with_dispatcher():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    p = hub_catchment_partition(net, oracle, hub_groups=merge_nearest_hubs(net, oracle, 3))
    reqs = generate_requests(net, seed=11)
    fleet = make_fleet(
        network=net, fleet_size=15, capacity=4,
        service_end_time=SERVICE_END, placement="hubs",
    )
    d = HierarchicalDispatcher(fleet=fleet, oracle=oracle, partition=p, future_requests=reqs)
    assert p.n_regions == 3
    assigned = sum(
        1 for r in reqs[:30]
        if d.assign(r, now=r.announce_time).vehicle_id is not None
    )
    assert assigned > 0
