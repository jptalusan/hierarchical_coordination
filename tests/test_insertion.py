from hcoord.demand import Request
from hcoord.dispatch.insertion import apply_insertion, best_insertion
from hcoord.fleet import Vehicle, return_arrival
from hcoord.geography import build_memphis_outskirts
from hcoord.travel import TravelTimeOracle


def _setup():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    return net, oracle


def _request(rid: int, origin: int, dest: int, earliest: float, latest: float) -> Request:
    return Request(
        id=rid,
        origin=origin,
        destination=dest,
        announce_time=earliest,
        earliest_pickup=earliest,
        latest_arrival=latest,
        shift_id=0,
    )


def test_insertion_into_empty_route():
    _, oracle = _setup()
    v = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=2000.0)
    req = _request(rid=1, origin=10, dest=0, earliest=0.0, latest=200.0)
    result = best_insertion(v, req, oracle)
    assert result is not None
    assert result.pickup_at == 0
    assert result.dropoff_at == 0
    assert result.cost > 0


def test_apply_insertion_mutates_route():
    _, oracle = _setup()
    v = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=2000.0)
    req = _request(rid=1, origin=10, dest=0, earliest=0.0, latest=200.0)
    result = best_insertion(v, req, oracle)
    apply_insertion(v, req, result)
    assert len(v.route) == 2
    assert v.route[0].kind == "pickup" and v.route[0].request_id == 1
    assert v.route[1].kind == "dropoff" and v.route[1].request_id == 1


def test_apply_insertion_rejects_wrong_vehicle():
    import pytest

    _, oracle = _setup()
    v1 = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=2000.0)
    v2 = Vehicle(id=1, capacity=4, home=0, location=0, available_time=0.0, service_end_time=2000.0)
    req = _request(rid=1, origin=10, dest=0, earliest=0.0, latest=200.0)
    result = best_insertion(v1, req, oracle)
    with pytest.raises(ValueError):
        apply_insertion(v2, req, result)


def test_infeasible_returns_none():
    _, oracle = _setup()
    v = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=0.5)
    req = _request(rid=1, origin=10, dest=0, earliest=0.0, latest=200.0)
    result = best_insertion(v, req, oracle)
    assert result is None


def test_two_requests_can_be_inserted_sequentially():
    _, oracle = _setup()
    v = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=4000.0)
    r1 = _request(rid=1, origin=10, dest=0, earliest=0.0, latest=300.0)
    apply_insertion(v, r1, best_insertion(v, r1, oracle))
    r2 = _request(rid=2, origin=15, dest=1, earliest=0.0, latest=300.0)
    res2 = best_insertion(v, r2, oracle)
    assert res2 is not None
    apply_insertion(v, r2, res2)
    assert len(v.route) == 4
    rids = [s.request_id for s in v.route]
    assert sorted(set(rids)) == [1, 2]


def test_cost_equals_return_arrival_delta():
    _, oracle = _setup()
    v = Vehicle(id=0, capacity=4, home=0, location=0, available_time=0.0, service_end_time=4000.0)
    req = _request(rid=1, origin=10, dest=2, earliest=0.0, latest=400.0)
    base = return_arrival(v, oracle)
    result = best_insertion(v, req, oracle)
    apply_insertion(v, req, result)
    after = return_arrival(v, oracle)
    assert abs((after - base) - result.cost) < 1e-9


def test_cheapest_choice_picked():
    """Insertion should prefer the vehicle/position with lowest cost."""
    _, oracle = _setup()
    near = Vehicle(id=0, capacity=4, home=10, location=10, available_time=0.0, service_end_time=2000.0)
    far = Vehicle(id=1, capacity=4, home=20, location=20, available_time=0.0, service_end_time=2000.0)
    req = _request(rid=1, origin=10, dest=0, earliest=0.0, latest=200.0)
    near_cost = best_insertion(near, req, oracle).cost
    far_cost = best_insertion(far, req, oracle).cost
    assert near_cost < far_cost
