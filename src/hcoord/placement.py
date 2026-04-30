"""Initial fleet placement strategies.

Strategies are registered by name so a config-driven runner (e.g., hydra) can
select between them without code changes. Each vehicle's `home` depot is set
to its initial location, so the same strategy controls both where the fleet
starts and where it must return at end of day.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from hcoord.demand import Request
from hcoord.fleet import Vehicle
from hcoord.geography import Network

PlacementFn = Callable[..., list[int]]
_REGISTRY: dict[str, PlacementFn] = {}


def register(name: str) -> Callable[[PlacementFn], PlacementFn]:
    def decorator(fn: PlacementFn) -> PlacementFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate placement strategy {name!r}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)


def make_placement(
    strategy: str,
    *,
    network: Network,
    fleet_size: int,
    **kwargs: Any,
) -> list[int]:
    """Return a list of `fleet_size` zone ids — one home depot per vehicle."""
    if strategy not in _REGISTRY:
        raise KeyError(f"unknown strategy {strategy!r}; options: {list_strategies()}")
    if fleet_size < 0:
        raise ValueError(f"fleet_size must be >= 0, got {fleet_size}")
    return _REGISTRY[strategy](network=network, fleet_size=fleet_size, **kwargs)


def make_fleet(
    *,
    network: Network,
    fleet_size: int,
    capacity: int,
    service_end_time: float,
    placement: str = "hubs",
    **placement_kwargs: Any,
) -> list[Vehicle]:
    """Build `fleet_size` vehicles using the named placement strategy."""
    homes = make_placement(
        placement, network=network, fleet_size=fleet_size, **placement_kwargs
    )
    return [
        Vehicle(
            id=i,
            capacity=capacity,
            home=h,
            location=h,
            available_time=0.0,
            service_end_time=service_end_time,
        )
        for i, h in enumerate(homes)
    ]


@register("hubs")
def _hub_placement(*, network: Network, fleet_size: int, **_: Any) -> list[int]:
    """Round-robin across hub zones."""
    hub_ids = [h.id for h in network.hubs]
    if not hub_ids:
        raise ValueError("network has no hubs")
    return [hub_ids[i % len(hub_ids)] for i in range(fleet_size)]


@register("demand_proportional")
def _demand_proportional_placement(
    *,
    network: Network,
    fleet_size: int,
    requests: list[Request] | None = None,
    **_: Any,
) -> list[int]:
    """Allocate vehicles to outskirt origins proportional to request volume.

    Uses largest-remainder rounding to hit `fleet_size` exactly. Falls back to
    hub placement if no requests are supplied.
    """
    if not requests:
        return _hub_placement(network=network, fleet_size=fleet_size)
    counts = Counter(r.origin for r in requests)
    total = sum(counts.values())
    raw = {z: fleet_size * c / total for z, c in counts.items()}
    floor = {z: int(v) for z, v in raw.items()}
    leftover = fleet_size - sum(floor.values())
    remainder = sorted(((raw[z] - floor[z], z) for z in raw), reverse=True)

    locations: list[int] = []
    for z, n in floor.items():
        locations.extend([z] * n)
    for _, z in remainder[:leftover]:
        locations.append(z)
    return locations[:fleet_size]
