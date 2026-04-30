"""Synthetic Memphis-outskirts → Memphis-hubs geography.

Deterministic from a seed. Five hub zones in a central cluster; the rest
scattered in an annulus. A k-nearest-neighbor road graph connects them, with
edge travel times derived from a fixed average speed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx

DEFAULT_HUB_NAMES: tuple[str, ...] = ("Downtown", "Medical", "FedEx", "Airport", "UofM")


@dataclass(frozen=True)
class Zone:
    id: int
    x: float
    y: float
    is_hub: bool
    name: str


class Network:
    """A geography of zones connected by a road graph.

    Edges carry `distance` (km) and `travel_time` (minutes) attributes.
    """

    def __init__(self, zones: list[Zone], graph: nx.Graph) -> None:
        self.zones = zones
        self.graph = graph
        self._by_id = {z.id: z for z in zones}

    def zone(self, zone_id: int) -> Zone:
        return self._by_id[zone_id]

    @property
    def hubs(self) -> list[Zone]:
        return [z for z in self.zones if z.is_hub]

    @property
    def outskirts(self) -> list[Zone]:
        return [z for z in self.zones if not z.is_hub]

    def __len__(self) -> int:
        return len(self.zones)


def build_memphis_outskirts(
    seed: int = 7,
    n_outskirts: int = 25,
    n_hubs: int = 5,
    hub_radius_km: float = 4.0,
    max_radius_km: float = 25.0,
    k_nearest: int = 4,
    avg_speed_kmh: float = 50.0,
) -> Network:
    """Build the default 30-zone synthetic geography.

    Hubs sit in a central disk of radius `hub_radius_km`. Outskirts sit in an
    annulus between `hub_radius_km + 1` and `max_radius_km`. Each zone connects
    to its `k_nearest` Euclidean neighbors. If the resulting graph is
    disconnected, components are stitched with a single bridge edge.
    """
    if n_hubs < 1 or n_outskirts < 1:
        raise ValueError("need at least one hub and one outskirt zone")
    rng = random.Random(seed)
    zones: list[Zone] = []

    for i in range(n_hubs):
        angle = 2.0 * math.pi * i / n_hubs + rng.uniform(-0.1, 0.1)
        r = rng.uniform(0.0, hub_radius_km)
        name = DEFAULT_HUB_NAMES[i] if i < len(DEFAULT_HUB_NAMES) else f"Hub-{i}"
        zones.append(Zone(i, r * math.cos(angle), r * math.sin(angle), True, name))

    for i in range(n_outskirts):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        r = rng.uniform(hub_radius_km + 1.0, max_radius_km)
        zid = n_hubs + i
        zones.append(Zone(zid, r * math.cos(angle), r * math.sin(angle), False, f"Outskirt-{i:02d}"))

    g = nx.Graph()
    for z in zones:
        g.add_node(z.id, x=z.x, y=z.y, is_hub=z.is_hub, name=z.name)

    for zi in zones:
        ranked = sorted(
            ((math.hypot(zi.x - zj.x, zi.y - zj.y), zj.id) for zj in zones if zj.id != zi.id)
        )
        for d_km, zjid in ranked[:k_nearest]:
            if not g.has_edge(zi.id, zjid):
                g.add_edge(
                    zi.id,
                    zjid,
                    distance=d_km,
                    travel_time=(d_km / avg_speed_kmh) * 60.0,
                )

    if not nx.is_connected(g):
        comps = [sorted(c) for c in nx.connected_components(g)]
        comps.sort(key=lambda c: c[0])
        for ca, cb in zip(comps, comps[1:]):
            best: tuple[float, int, int] | None = None
            for a in ca:
                za = zones[a]
                for b in cb:
                    zb = zones[b]
                    d = math.hypot(za.x - zb.x, za.y - zb.y)
                    if best is None or d < best[0]:
                        best = (d, a, b)
            assert best is not None
            d, a, b = best
            g.add_edge(a, b, distance=d, travel_time=(d / avg_speed_kmh) * 60.0)

    return Network(zones=zones, graph=g)
