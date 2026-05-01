"""Tests for the OSM substrate.

The inner builder is tested against a synthetic NetworkX graph (no network
required). The end-to-end Memphis pull is exercised by an opt-in smoke test
that is skipped unless `osmnx` is installed and the cache exists, since CI
shouldn't depend on the OSM Overpass API being reachable.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from hcoord.geography_osm import (
    HubSpec,
    build_network_from_osm_graph,
    sample_outskirts,
)


def _toy_osm_graph() -> nx.MultiDiGraph:
    """A 6-node grid where node coords double as lat/lon."""
    g = nx.MultiDiGraph()
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (0.0, 1.0),
        4: (1.0, 1.0),
        5: (2.0, 1.0),
    }
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)
    edges = [
        (0, 1, 60.0), (1, 2, 60.0),
        (3, 4, 60.0), (4, 5, 60.0),
        (0, 3, 90.0), (1, 4, 90.0), (2, 5, 90.0),
    ]
    for u, v, t in edges:
        g.add_edge(u, v, travel_time=t)
        g.add_edge(v, u, travel_time=t)
    return g


def _stub_nearest(g: nx.MultiDiGraph, lon: float, lat: float) -> int:
    return min(
        g.nodes,
        key=lambda n: (g.nodes[n]["x"] - lon) ** 2 + (g.nodes[n]["y"] - lat) ** 2,
    )


def test_inner_builder_constructs_zones_and_complete_graph():
    g = _toy_osm_graph()
    hubs = [HubSpec("A", lat=0.0, lon=0.0), HubSpec("B", lat=0.0, lon=2.0)]
    outskirts = [(1.0, 1.0)]  # snaps to node 4
    net = build_network_from_osm_graph(
        g, hubs, outskirts, nearest_node_fn=_stub_nearest
    )
    assert len(net.zones) == 3
    assert len(net.hubs) == 2
    assert nx.is_connected(net.graph)
    n = len(net.zones)
    assert net.graph.number_of_edges() == n * (n - 1) // 2


def test_inner_builder_dedupes_collisions():
    g = _toy_osm_graph()
    hubs = [HubSpec("A", lat=0.0, lon=0.0)]
    outskirts = [(0.0, 0.0), (1.0, 1.0)]  # first collides with hub
    net = build_network_from_osm_graph(
        g, hubs, outskirts, nearest_node_fn=_stub_nearest
    )
    assert len(net.zones) == 2


def test_inner_builder_travel_times_match_dijkstra_minutes():
    g = _toy_osm_graph()
    hubs = [HubSpec("A", lat=0.0, lon=0.0), HubSpec("B", lat=0.0, lon=2.0)]
    net = build_network_from_osm_graph(
        g, hubs, outskirt_lat_lons=[], nearest_node_fn=_stub_nearest
    )
    expected_min = (60.0 + 60.0) / 60.0
    assert math.isclose(net.graph[0][1]["travel_time"], expected_min)


def test_inner_builder_disconnected_raises():
    g = nx.MultiDiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=10.0, y=10.0)
    hubs = [HubSpec("A", lat=0.0, lon=0.0), HubSpec("B", lat=10.0, lon=10.0)]
    with pytest.raises(RuntimeError, match="disconnected"):
        build_network_from_osm_graph(
            g, hubs, outskirt_lat_lons=[], nearest_node_fn=_stub_nearest
        )


def test_sample_outskirts_count_and_annulus():
    pts = sample_outskirts(
        center_lat=35.0, center_lon=-90.0, n=20,
        inner_km=10.0, outer_km=40.0, seed=3,
    )
    assert len(pts) == 20
    for lat, lon in pts:
        d_lat_km = (lat - 35.0) * 111.0
        d_lon_km = (lon - -90.0) * 111.0 * math.cos(math.radians(35.0))
        r = math.hypot(d_lat_km, d_lon_km)
        assert 10.0 - 1e-6 <= r <= 40.0 + 1e-6


def test_sample_outskirts_deterministic_from_seed():
    a = sample_outskirts(center_lat=35.0, center_lon=-90.0, n=10,
                         inner_km=5, outer_km=20, seed=11)
    b = sample_outskirts(center_lat=35.0, center_lon=-90.0, n=10,
                         inner_km=5, outer_km=20, seed=11)
    assert a == b


def test_sample_outskirts_invalid_radii():
    with pytest.raises(ValueError):
        sample_outskirts(center_lat=35.0, center_lon=-90.0, n=5,
                         inner_km=20, outer_km=10, seed=1)


def test_memphis_osm_cached():
    """Smoke test against the real Memphis network.

    Skipped unless osmnx is installed AND the cache exists, so CI doesn't hit
    the OSM Overpass API.
    """
    from pathlib import Path

    pytest.importorskip("osmnx")
    from hcoord.geography_osm import DEFAULT_CACHE_DIR, build_memphis_osm
    from hcoord.travel import TravelTimeOracle

    cache_path = (
        Path(DEFAULT_CACHE_DIR)
        / "memphis_n25_in8.0_out40.0_seed7_drive.pkl"
    )
    if not cache_path.exists():
        pytest.skip(f"OSM cache absent ({cache_path}); run build_memphis_osm() locally")

    net = build_memphis_osm(seed=7, n_outskirts=25, outer_radius_km=40.0)
    assert len(net) == 30
    assert len(net.hubs) == 5
    oracle = TravelTimeOracle(net)
    times = [
        oracle.travel_time(o.id, h.id) for o in net.outskirts for h in net.hubs
    ]
    assert min(times) > 0
    assert max(times) > 30  # real OSM has wider spread than synthetic
