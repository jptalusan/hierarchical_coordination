from hcoord.geography import build_memphis_outskirts
from hcoord.travel import TravelTimeOracle


def test_self_loop_zero():
    oracle = TravelTimeOracle(build_memphis_outskirts())
    for z in oracle.network.zones:
        assert oracle.travel_time(z.id, z.id) == 0.0


def test_symmetric_on_undirected_graph():
    oracle = TravelTimeOracle(build_memphis_outskirts())
    zones = oracle.network.zones
    for i in range(0, len(zones), 5):
        for j in range(0, len(zones), 7):
            if i == j:
                continue
            assert abs(oracle.travel_time(i, j) - oracle.travel_time(j, i)) < 1e-9


def test_triangle_inequality():
    oracle = TravelTimeOracle(build_memphis_outskirts())
    n = len(oracle.network.zones)
    for i in (0, 5, 12, 20):
        for j in (1, 7, 15, 25):
            for k in range(0, n, 6):
                if len({i, j, k}) < 3:
                    continue
                direct = oracle.travel_time(i, j)
                via = oracle.travel_time(i, k) + oracle.travel_time(k, j)
                assert direct <= via + 1e-9


def test_reachable_from_covers_all_zones():
    oracle = TravelTimeOracle(build_memphis_outskirts())
    n = len(oracle.network.zones)
    reachable = oracle.reachable_from(0)
    assert len(reachable) == n
