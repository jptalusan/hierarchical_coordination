"""OSM-grounded Memphis geography.

Pulls a real drive network for the Memphis area via osmnx, snaps hub and
outskirt centroids to nearest OSM nodes, and computes zone-to-zone travel
times by Dijkstra on the OSM `travel_time` edge weight. The returned object
is a standard `hcoord.Network` — same interface as the synthetic one — so
dispatchers, metrics, and the experiment runner work unchanged.

Caches the pulled OSM graph and the resulting `Network` to disk; subsequent
calls with the same parameters return instantly.

Requires osmnx (optional `[osm]` extra). Install with `uv sync --extra osm`.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from hcoord.geography import Network, Zone


@dataclass(frozen=True)
class HubSpec:
    name: str
    lat: float
    lon: float


# Approximate Memphis hub centroids.
MEMPHIS_HUBS: tuple[HubSpec, ...] = (
    HubSpec("Downtown", 35.1495, -90.0490),
    HubSpec("Medical", 35.1410, -90.0210),
    HubSpec("FedEx", 35.0524, -89.9756),
    HubSpec("Airport", 35.0421, -89.9792),
    HubSpec("UofM", 35.1180, -89.9370),
)

DEFAULT_CACHE_DIR = Path("data/osm_cache")


def _km_to_deg_lat(km: float) -> float:
    return km / 111.0


def _km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.0 * math.cos(math.radians(lat_deg)))


def sample_outskirts(
    *,
    center_lat: float,
    center_lon: float,
    n: int,
    inner_km: float,
    outer_km: float,
    seed: int,
) -> list[tuple[float, float]]:
    """Sample n (lat, lon) points uniformly in an annulus around the center."""
    if inner_km >= outer_km:
        raise ValueError(f"inner_km ({inner_km}) must be < outer_km ({outer_km})")
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    for _ in range(n):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        r_km = rng.uniform(inner_km, outer_km)
        lat = center_lat + _km_to_deg_lat(r_km * math.cos(angle))
        lon = center_lon + _km_to_deg_lon(r_km * math.sin(angle), center_lat)
        points.append((float(lat), float(lon)))
    return points


def build_network_from_osm_graph(
    osm_graph: nx.MultiDiGraph,
    hub_specs: list[HubSpec],
    outskirt_lat_lons: list[tuple[float, float]],
    *,
    travel_time_attr: str = "travel_time",
    nearest_node_fn: Callable[[nx.MultiDiGraph, float, float], int] | None = None,
) -> Network:
    """Build an `hcoord.Network` from an OSM graph plus hub / outskirt centroids.

    Centroids are snapped to nearest OSM nodes; zone-to-zone travel times are
    Dijkstra distances on `travel_time_attr` (seconds in the osmnx convention),
    converted to minutes. The returned `Network`'s graph is complete, with
    edge `travel_time` in minutes — the same shape `TravelTimeOracle` expects.

    `nearest_node_fn(graph, lon, lat) -> node` defaults to osmnx's; tests pass
    a custom one for synthetic graphs.
    """
    if nearest_node_fn is None:
        import osmnx as ox  # lazy import

        def nearest_node_fn(g: nx.MultiDiGraph, lon: float, lat: float) -> int:
            return int(ox.distance.nearest_nodes(g, X=lon, Y=lat))

    zones: list[Zone] = []
    zone_to_osm: dict[int, int] = {}

    for i, hub in enumerate(hub_specs):
        node = nearest_node_fn(osm_graph, hub.lon, hub.lat)
        zones.append(
            Zone(
                id=i,
                x=float(osm_graph.nodes[node]["x"]),
                y=float(osm_graph.nodes[node]["y"]),
                is_hub=True,
                name=hub.name,
            )
        )
        zone_to_osm[i] = node

    n_hubs = len(hub_specs)
    seen_nodes = set(zone_to_osm.values())
    next_outskirt_idx = 0
    for lat, lon in outskirt_lat_lons:
        node = nearest_node_fn(osm_graph, lon, lat)
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        zid = n_hubs + next_outskirt_idx
        next_outskirt_idx += 1
        zones.append(
            Zone(
                id=zid,
                x=float(osm_graph.nodes[node]["x"]),
                y=float(osm_graph.nodes[node]["y"]),
                is_hub=False,
                name=f"Outskirt-{zid - n_hubs:02d}",
            )
        )
        zone_to_osm[zid] = node

    zone_graph = nx.Graph()
    for z in zones:
        zone_graph.add_node(z.id, x=z.x, y=z.y, is_hub=z.is_hub, name=z.name)

    for i, zi in enumerate(zones):
        lengths_s = nx.single_source_dijkstra_path_length(
            osm_graph, zone_to_osm[zi.id], weight=travel_time_attr
        )
        for zj in zones[i + 1 :]:
            target = zone_to_osm[zj.id]
            if target not in lengths_s:
                continue
            tt_min = float(lengths_s[target]) / 60.0
            zone_graph.add_edge(zi.id, zj.id, travel_time=tt_min, distance=0.0)

    if not nx.is_connected(zone_graph):
        n_components = nx.number_connected_components(zone_graph)
        raise RuntimeError(
            f"OSM zone graph is disconnected ({n_components} components); "
            "the bounding box may be too tight or outskirts sampled across a barrier"
        )

    return Network(zones=zones, graph=zone_graph)


def build_memphis_osm(
    *,
    seed: int = 7,
    n_outskirts: int = 25,
    inner_radius_km: float = 8.0,
    outer_radius_km: float = 50.0,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    network_type: str = "drive",
    hub_speeds_kmh: dict[str, float] | None = None,
    fallback_speed_kmh: float = 50.0,
) -> Network:
    """Pull a real Memphis OSM drive network and build an hcoord Network.

    First call performs a network pull (~30 s – 2 min) and caches the OSM
    graph + the built Network to `cache_dir`. Subsequent calls with the same
    parameters return instantly.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = (
        f"memphis_n{n_outskirts}"
        f"_in{inner_radius_km}_out{outer_radius_km}"
        f"_seed{seed}_{network_type}.pkl"
    )
    cache_path = cache_dir / cache_key
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    import osmnx as ox  # lazy import

    center_lat = float(np.mean([h.lat for h in MEMPHIS_HUBS]))
    center_lon = float(np.mean([h.lon for h in MEMPHIS_HUBS]))

    graphml_path = cache_dir / f"memphis_{network_type}_r{outer_radius_km}.graphml"
    if graphml_path.exists():
        G = ox.load_graphml(graphml_path)
    else:
        G = ox.graph_from_point(
            (center_lat, center_lon),
            dist=int(outer_radius_km * 1000) + 5000,
            network_type=network_type,
        )
        G = ox.add_edge_speeds(G, hwy_speeds=hub_speeds_kmh, fallback=fallback_speed_kmh)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, graphml_path)

    outskirts = sample_outskirts(
        center_lat=center_lat,
        center_lon=center_lon,
        n=n_outskirts,
        inner_km=inner_radius_km,
        outer_km=outer_radius_km,
        seed=seed,
    )

    network = build_network_from_osm_graph(G, list(MEMPHIS_HUBS), outskirts)

    with cache_path.open("wb") as f:
        pickle.dump(network, f)

    return network
