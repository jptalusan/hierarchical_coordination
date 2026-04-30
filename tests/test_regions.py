import pytest

from hcoord.geography import build_memphis_outskirts
from hcoord.regions import hub_catchment_partition, merge_nearest_hubs
from hcoord.travel import TravelTimeOracle


def test_default_partition_one_region_per_hub():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    p = hub_catchment_partition(net, oracle)
    assert p.n_regions == len(net.hubs)
    for r, group in enumerate(p.hub_groups):
        assert len(group) == 1


def test_every_zone_assigned_exactly_once():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    p = hub_catchment_partition(net, oracle)
    assigned = sum(len(zs) for zs in p.zones_in.values())
    assert assigned == len(net.zones)
    assert set(p.region_of) == {z.id for z in net.zones}


def test_outskirt_assigned_to_nearest_hub():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    p = hub_catchment_partition(net, oracle)
    hub_ids = [h.id for h in net.hubs]
    for o in net.outskirts:
        nearest_hub = min(hub_ids, key=lambda hid: oracle.travel_time(o.id, hid))
        expected_region = next(r for r, group in enumerate(p.hub_groups) if nearest_hub in group)
        assert p.region(o.id) == expected_region


def test_custom_hub_groups_partition_zones():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    hub_ids = [h.id for h in net.hubs]
    groups = [[hub_ids[0], hub_ids[1]], [hub_ids[2]], [hub_ids[3], hub_ids[4]]]
    p = hub_catchment_partition(net, oracle, hub_groups=groups)
    assert p.n_regions == 3
    for h, group in zip([hub_ids[0], hub_ids[1]], [0, 0]):
        assert p.region(h) == group


def test_invalid_hub_groups_rejected():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    hub_ids = [h.id for h in net.hubs]
    with pytest.raises(ValueError):
        hub_catchment_partition(net, oracle, hub_groups=[hub_ids[:2]])
    with pytest.raises(ValueError):
        hub_catchment_partition(net, oracle, hub_groups=[hub_ids, [hub_ids[0]]])


def test_merge_nearest_hubs_yields_requested_region_count():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    for k in (1, 2, 3, 4, 5):
        groups = merge_nearest_hubs(net, oracle, n_regions=k)
        assert len(groups) == k
        flat = sorted(hid for g in groups for hid in g)
        assert flat == sorted(h.id for h in net.hubs)


def test_merge_nearest_hubs_invalid_k():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    with pytest.raises(ValueError):
        merge_nearest_hubs(net, oracle, n_regions=0)
    with pytest.raises(ValueError):
        merge_nearest_hubs(net, oracle, n_regions=99)


def test_k3_partition_via_merge_is_valid():
    net = build_memphis_outskirts()
    oracle = TravelTimeOracle(net)
    groups = merge_nearest_hubs(net, oracle, n_regions=3)
    p = hub_catchment_partition(net, oracle, hub_groups=groups)
    assert p.n_regions == 3
    assigned = sum(len(zs) for zs in p.zones_in.values())
    assert assigned == len(net.zones)
