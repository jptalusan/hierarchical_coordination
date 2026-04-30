"""One-direction shift-commute demand model.

Generates outskirt → hub workforce requests anchored to a small set of shift
start times. Demand is shift-synchronized (announce times cluster shortly
before shift start) and structurally sparse: a fraction of OD pairs is
structurally inactive across all shifts.

Parameters are exposed for sweeping the scalability plot:
    - intensity: multiplier on the Poisson mean (drives request volume).
    - structural_zero_prob: fraction of OD pairs with zero demand.
    - shifts: list of Shift objects (count and times).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hcoord.geography import Network


@dataclass(frozen=True)
class Shift:
    """A workforce shift identified by an integer id and its start time (min)."""

    id: int
    start_time: float


@dataclass(frozen=True)
class Request:
    """A single workforce-commute trip request.

    The dispatcher learns about the request at `announce_time`. The worker can
    be picked up any time at or after `earliest_pickup` and must arrive at the
    destination hub by `latest_arrival`.
    """

    id: int
    origin: int
    destination: int
    announce_time: float
    earliest_pickup: float
    latest_arrival: float
    shift_id: int


DEFAULT_SHIFTS: tuple[Shift, ...] = (
    Shift(0, 6 * 60.0),
    Shift(1, 14 * 60.0),
    Shift(2, 22 * 60.0),
)


def generate_requests(
    network: Network,
    *,
    shifts: tuple[Shift, ...] = DEFAULT_SHIFTS,
    base_rate: float = 0.4,
    intensity: float = 1.0,
    structural_zero_prob: float = 0.4,
    announce_lead_min: float = 90.0,
    arrival_buffer_min: float = 15.0,
    announce_jitter_min: float = 10.0,
    seed: int = 11,
) -> list[Request]:
    """Generate one-direction outskirt → hub commute requests.

    Structural zeros are sampled once per OD pair and persist across all shifts
    (a worker who never commutes from origin o to hub h doesn't commute there
    at any shift).

    For each active (outskirt, hub, shift) triple, the number of requests is
    drawn from Poisson(`intensity` * `base_rate`). Each request announces in a
    small jitter window before the shift, with `announce_lead_min` of slack
    between earliest pickup and the shift start, minus `arrival_buffer_min`.
    """
    if base_rate < 0 or intensity < 0:
        raise ValueError("base_rate and intensity must be non-negative")
    if not 0.0 <= structural_zero_prob <= 1.0:
        raise ValueError("structural_zero_prob must be in [0, 1]")
    if announce_lead_min <= arrival_buffer_min:
        raise ValueError("announce_lead_min must exceed arrival_buffer_min")

    rng = np.random.default_rng(seed)
    hub_ids = [h.id for h in network.hubs]
    outskirt_ids = [o.id for o in network.outskirts]

    od_active = {
        (o, h): bool(rng.random() >= structural_zero_prob)
        for o in outskirt_ids
        for h in hub_ids
    }

    lam = float(intensity * base_rate)
    requests: list[Request] = []
    rid = 0
    for shift in shifts:
        earliest_pickup = shift.start_time - announce_lead_min
        latest_arrival = shift.start_time - arrival_buffer_min
        for o in outskirt_ids:
            for h in hub_ids:
                if not od_active[(o, h)]:
                    continue
                k = int(rng.poisson(lam)) if lam > 0 else 0
                for _ in range(k):
                    jitter = float(rng.uniform(0.0, announce_jitter_min))
                    requests.append(
                        Request(
                            id=rid,
                            origin=o,
                            destination=h,
                            announce_time=earliest_pickup + jitter,
                            earliest_pickup=earliest_pickup + jitter,
                            latest_arrival=latest_arrival,
                            shift_id=shift.id,
                        )
                    )
                    rid += 1

    requests.sort(key=lambda r: r.announce_time)
    return requests
