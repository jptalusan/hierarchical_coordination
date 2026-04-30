"""Travel-time oracle.

Wraps a `Network` and exposes shortest-path travel times in minutes between any
two zones. All-pairs shortest paths are computed eagerly at construction (cheap
for ~30 nodes) and then served from the cache.

The interface is deliberately small so a stochastic oracle (e.g., the GMM
scenario sampler in `stgnn-drt-scenarios`) can be swapped in without changing
dispatch code.
"""

from __future__ import annotations

import networkx as nx

from hcoord.geography import Network


class TravelTimeOracle:
    """Deterministic travel-time oracle backed by cached APSP."""

    def __init__(self, network: Network, weight: str = "travel_time") -> None:
        self.network = network
        self.weight = weight
        self._tt: dict[int, dict[int, float]] = {
            src: dict(targets)
            for src, targets in nx.all_pairs_dijkstra_path_length(network.graph, weight=weight)
        }

    def travel_time(self, origin: int, dest: int) -> float:
        """Travel time from `origin` to `dest` in minutes. Self-loops are 0."""
        if origin == dest:
            return 0.0
        return self._tt[origin][dest]

    def reachable_from(self, origin: int) -> dict[int, float]:
        """Travel times from `origin` to every reachable zone."""
        return dict(self._tt[origin])
