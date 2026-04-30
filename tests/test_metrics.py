from hcoord.fleet import Vehicle
from hcoord.geography import build_memphis_outskirts
from hcoord.metrics import DecisionRecord, compute_metrics
from hcoord.travel import TravelTimeOracle


def test_empty_decisions_yield_zeros():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    fleet = [Vehicle(id=0, capacity=4, home=0, location=0,
                     available_time=0.0, service_end_time=1440.0)]
    m = compute_metrics([], fleet, oracle)
    assert m.n_requests == 0
    assert m.assignment_rate == 0.0
    assert m.mean_wall_ms == 0.0
    assert m.fleet.n_vehicles == 1
    assert m.fleet.n_active == 0


def test_assignment_rate():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    fleet: list[Vehicle] = []
    decisions = [
        DecisionRecord(request_id=0, announce_time=0.0, vehicle_id=1, cost=1.0, wall_time_s=0.001),
        DecisionRecord(request_id=1, announce_time=1.0, vehicle_id=None, cost=float("inf"), wall_time_s=0.002),
        DecisionRecord(request_id=2, announce_time=2.0, vehicle_id=2, cost=2.0, wall_time_s=0.003),
    ]
    m = compute_metrics(decisions, fleet, oracle)
    assert m.n_assigned == 2
    assert abs(m.assignment_rate - 2 / 3) < 1e-9
    assert abs(m.mean_wall_ms - 2.0) < 1e-9
    assert abs(m.total_wall_s - 0.006) < 1e-9


def test_summary_omits_decisions():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    decisions = [
        DecisionRecord(request_id=0, announce_time=0.0, vehicle_id=1, cost=1.0, wall_time_s=0.001),
    ]
    m = compute_metrics(decisions, [], oracle)
    s = m.summary()
    assert "decisions" not in s
    assert "fleet" in s
    assert s["n_assigned"] == 1
