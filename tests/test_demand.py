import pytest

from hcoord.demand import DEFAULT_SHIFTS, Shift, generate_requests
from hcoord.geography import build_memphis_outskirts


def test_deterministic():
    net = build_memphis_outskirts()
    a = generate_requests(net, seed=11)
    b = generate_requests(net, seed=11)
    assert [(r.id, r.origin, r.destination, r.announce_time) for r in a] == [
        (r.id, r.origin, r.destination, r.announce_time) for r in b
    ]


def test_one_direction_outskirt_to_hub():
    net = build_memphis_outskirts()
    hub_ids = {h.id for h in net.hubs}
    outskirt_ids = {o.id for o in net.outskirts}
    reqs = generate_requests(net, seed=11)
    assert reqs, "expected at least one request at default settings"
    for r in reqs:
        assert r.origin in outskirt_ids
        assert r.destination in hub_ids


def test_time_consistency():
    net = build_memphis_outskirts()
    for r in generate_requests(net, seed=11):
        assert r.announce_time <= r.earliest_pickup + 1e-9
        assert r.earliest_pickup < r.latest_arrival


def test_structural_zeros_persist_across_shifts():
    net = build_memphis_outskirts()
    reqs = generate_requests(net, seed=11, intensity=10.0, structural_zero_prob=0.5)
    pairs_per_shift: dict[int, set[tuple[int, int]]] = {}
    for r in reqs:
        pairs_per_shift.setdefault(r.shift_id, set()).add((r.origin, r.destination))
    if len(pairs_per_shift) < 2:
        pytest.skip("need at least two active shifts for this check")
    seen = set.union(*pairs_per_shift.values())
    inactive = seen - set.intersection(
        *(pairs_per_shift[s] | (seen - pairs_per_shift[s]) for s in pairs_per_shift)
    )
    assert inactive == set()


def test_intensity_scales_volume():
    net = build_memphis_outskirts()
    low = len(generate_requests(net, seed=11, intensity=1.0))
    high = len(generate_requests(net, seed=11, intensity=5.0))
    assert high > low * 2


def test_zero_intensity_yields_no_requests():
    net = build_memphis_outskirts()
    assert generate_requests(net, seed=11, intensity=0.0) == []


def test_sorted_by_announce_time():
    net = build_memphis_outskirts()
    times = [r.announce_time for r in generate_requests(net, seed=11)]
    assert times == sorted(times)


def test_shift_count_respected():
    net = build_memphis_outskirts()
    custom = (Shift(0, 7 * 60.0),)
    reqs = generate_requests(net, seed=11, shifts=custom)
    assert {r.shift_id for r in reqs} <= {0}
    default_reqs = generate_requests(net, seed=11)
    assert {r.shift_id for r in default_reqs} == {s.id for s in DEFAULT_SHIFTS}


def test_invalid_params_rejected():
    net = build_memphis_outskirts()
    with pytest.raises(ValueError):
        generate_requests(net, structural_zero_prob=1.5)
    with pytest.raises(ValueError):
        generate_requests(net, intensity=-1.0)
    with pytest.raises(ValueError):
        generate_requests(net, announce_lead_min=10.0, arrival_buffer_min=15.0)
