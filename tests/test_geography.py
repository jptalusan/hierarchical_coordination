import networkx as nx
import pytest

from hcoord.geography import build_memphis_outskirts


def test_default_size():
    net = build_memphis_outskirts()
    assert len(net) == 30
    assert len(net.hubs) == 5
    assert len(net.outskirts) == 25


def test_graph_connected():
    net = build_memphis_outskirts()
    assert nx.is_connected(net.graph)


def test_edge_attributes_present():
    net = build_memphis_outskirts()
    for _, _, data in net.graph.edges(data=True):
        assert "distance" in data and data["distance"] > 0
        assert "travel_time" in data and data["travel_time"] > 0


def test_deterministic_from_seed():
    a = build_memphis_outskirts(seed=7)
    b = build_memphis_outskirts(seed=7)
    assert [(z.id, z.x, z.y, z.is_hub, z.name) for z in a.zones] == [
        (z.id, z.x, z.y, z.is_hub, z.name) for z in b.zones
    ]


def test_seeds_differ():
    a = build_memphis_outskirts(seed=7)
    b = build_memphis_outskirts(seed=8)
    assert [(z.x, z.y) for z in a.zones] != [(z.x, z.y) for z in b.zones]


def test_hubs_inside_central_disk():
    net = build_memphis_outskirts(hub_radius_km=4.0)
    for h in net.hubs:
        assert (h.x**2 + h.y**2) ** 0.5 <= 4.0 + 1e-9


def test_outskirts_outside_central_disk():
    net = build_memphis_outskirts(hub_radius_km=4.0)
    for o in net.outskirts:
        assert (o.x**2 + o.y**2) ** 0.5 >= 4.0 + 1.0 - 1e-9


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        build_memphis_outskirts(n_hubs=0)
    with pytest.raises(ValueError):
        build_memphis_outskirts(n_outskirts=0)
