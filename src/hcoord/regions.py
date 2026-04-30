"""Hub-catchment region partitions for the HLP.

Each outskirt zone is assigned to its travel-time-nearest hub. K = n_hubs by
default. For ablations with K < n_hubs, hubs can be agglomerated by min-link
travel time, or grouped explicitly via `hub_groups`.
"""

from __future__ import annotations

from dataclasses import dataclass

from hcoord.geography import Network
from hcoord.travel import TravelTimeOracle


@dataclass(frozen=True)
class Partition:
    """A zone-to-region assignment.

    `region_of` maps zone id → region id in [0, n_regions). `zones_in` is the
    inverse. `hub_groups[r]` lists the hub zone ids assigned to region r.
    """

    region_of: dict[int, int]
    zones_in: dict[int, list[int]]
    hub_groups: list[list[int]]

    @property
    def n_regions(self) -> int:
        return len(self.hub_groups)

    def region(self, zone_id: int) -> int:
        return self.region_of[zone_id]

    def zones(self, region_id: int) -> list[int]:
        return list(self.zones_in[region_id])


def hub_catchment_partition(
    network: Network,
    oracle: TravelTimeOracle,
    hub_groups: list[list[int]] | None = None,
) -> Partition:
    """Partition zones into hub catchments.

    If `hub_groups` is None, every hub forms its own region. Otherwise each
    inner list of hub ids forms a single region; the lists must partition the
    set of hubs exactly once each.
    """
    hub_ids = [h.id for h in network.hubs]
    if hub_groups is None:
        groups: list[list[int]] = [[h] for h in hub_ids]
    else:
        flat = [hid for group in hub_groups for hid in group]
        if sorted(flat) != sorted(hub_ids):
            raise ValueError("hub_groups must partition the set of hubs exactly")
        groups = [list(group) for group in hub_groups]

    hub_to_region = {hid: r for r, group in enumerate(groups) for hid in group}

    region_of: dict[int, int] = {}
    for z in network.zones:
        if z.is_hub:
            region_of[z.id] = hub_to_region[z.id]
        else:
            nearest = min(hub_ids, key=lambda hid: oracle.travel_time(z.id, hid))
            region_of[z.id] = hub_to_region[nearest]

    zones_in: dict[int, list[int]] = {r: [] for r in range(len(groups))}
    for zid, r in region_of.items():
        zones_in[r].append(zid)
    for r in zones_in:
        zones_in[r].sort()

    return Partition(region_of=region_of, zones_in=zones_in, hub_groups=groups)


def merge_nearest_hubs(
    network: Network,
    oracle: TravelTimeOracle,
    n_regions: int,
) -> list[list[int]]:
    """Agglomerate hubs into `n_regions` groups by min-link travel time.

    Use the result as the `hub_groups` argument to `hub_catchment_partition`
    for K < n_hubs ablations.
    """
    n_hubs = len(network.hubs)
    if not 1 <= n_regions <= n_hubs:
        raise ValueError(f"n_regions must be in [1, {n_hubs}], got {n_regions}")

    groups: list[list[int]] = [[h.id] for h in network.hubs]
    while len(groups) > n_regions:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = min(oracle.travel_time(a, b) for a in groups[i] for b in groups[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        assert best is not None
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]

    return groups
