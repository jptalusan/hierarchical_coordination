from hcoord.fleet import Stop, Vehicle, feasible, return_arrival, schedule
from hcoord.geography import build_memphis_outskirts
from hcoord.travel import TravelTimeOracle


def _net_oracle():
    net = build_memphis_outskirts()
    return net, TravelTimeOracle(net)


def test_empty_route_return_arrival():
    _, oracle = _net_oracle()
    v = Vehicle(id=0, capacity=4, home=1, location=0, available_time=0.0, service_end_time=1000.0)
    assert return_arrival(v, oracle) == oracle.travel_time(0, 1)


def test_empty_route_feasible():
    _, oracle = _net_oracle()
    v = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=1000.0)
    assert feasible(v, oracle)


def test_schedule_respects_earliest():
    _, oracle = _net_oracle()
    pickup = Stop(kind="pickup", zone=10, request_id=42, earliest=100.0, latest=200.0, service_time=2.0)
    drop = Stop(kind="dropoff", zone=0, request_id=42, earliest=0.0, latest=400.0, service_time=2.0)
    v = Vehicle(
        id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=500.0,
        route=[pickup, drop],
    )
    s = schedule(v, oracle)
    assert s[0].arrival >= 100.0
    assert s[0].departure == s[0].arrival + 2.0
    assert s[0].load_after == 1
    assert s[1].load_after == 0


def test_capacity_violation_infeasible():
    _, oracle = _net_oracle()
    p1 = Stop("pickup", 10, 1, 0, 1000, 0)
    p2 = Stop("pickup", 11, 2, 0, 1000, 0)
    d1 = Stop("dropoff", 0, 1, 0, 1000, 0)
    d2 = Stop("dropoff", 0, 2, 0, 1000, 0)
    v = Vehicle(
        id=0, capacity=1, home=0, location=0, available_time=0.0, service_end_time=2000.0,
        route=[p1, p2, d1, d2],
    )
    assert not feasible(v, oracle)


def test_window_violation_infeasible():
    _, oracle = _net_oracle()
    pickup = Stop("pickup", 10, 1, earliest=0, latest=0, service_time=0)
    drop = Stop("dropoff", 0, 1, earliest=0, latest=1000, service_time=0)
    v = Vehicle(
        id=0, capacity=1, home=0, location=0, available_time=0.0, service_end_time=2000.0,
        route=[pickup, drop],
    )
    assert not feasible(v, oracle)


def test_return_home_deadline_enforced():
    _, oracle = _net_oracle()
    pickup = Stop("pickup", 10, 1, 0, 1000, 0)
    drop = Stop("dropoff", 0, 1, 0, 1000, 0)
    tight = Vehicle(
        id=0, capacity=1, home=15, location=0, available_time=0.0, service_end_time=1.0,
        route=[pickup, drop],
    )
    assert not feasible(tight, oracle)
    loose = Vehicle(
        id=0, capacity=1, home=15, location=0, available_time=0.0, service_end_time=2000.0,
        route=[pickup, drop],
    )
    assert feasible(loose, oracle)


def test_load_delta_pickup_dropoff():
    pickup = Stop("pickup", 1, 1, 0, 100, 0)
    drop = Stop("dropoff", 2, 1, 0, 100, 0)
    assert pickup.load_delta == 1
    assert drop.load_delta == -1
