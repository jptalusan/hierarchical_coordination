"""InsertionCollector: observer that records training rows from `best_insertion`.

Each row corresponds to one (vehicle, request) candidate evaluation: a flat
feature dict (see `features.extract_features`) plus the ground-truth label
(best feasible cost, or `feasible=False` if no (p, q) pair was feasible).

Usage:

    collector = InsertionCollector(oracle, context={"seed": 11, ...})
    dispatcher = MonolithicDispatcher(fleet, oracle, observer=collector)
    # ... run experiment ...
    df = collector.to_dataframe()
    df.to_csv("path.csv", index=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hcoord.demand import Request
from hcoord.dispatch.insertion import InsertionResult
from hcoord.fleet import Vehicle
from hcoord.learning.features import extract_features
from hcoord.travel import TravelTimeOracle


@dataclass
class InsertionRow:
    features: dict[str, Any]
    feasible: bool
    cost: float  # 0.0 if infeasible (sentinel; use `feasible` to mask)
    pickup_at: int  # -1 if infeasible
    dropoff_at: int  # -1 if infeasible


class InsertionCollector:
    """Observer that records (features, label) rows from each `best_insertion` call.

    `context` is a flat dict of metadata stamped onto every row (e.g., seed,
    fleet_size, intensity, dispatcher type, region). This makes the resulting
    CSV easy to slice by configuration.
    """

    def __init__(
        self,
        oracle: TravelTimeOracle,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.oracle = oracle
        self.context = dict(context or {})
        self._rows: list[dict[str, Any]] = []
        self._decision_id = 0
        self._last_request_id: int | None = None

    def __call__(
        self,
        vehicle: Vehicle,
        request: Request,
        result: InsertionResult | None,
    ) -> None:
        # Group candidates by request: bump decision_id when the request changes.
        if request.id != self._last_request_id:
            self._decision_id += 1
            self._last_request_id = request.id

        feats = extract_features(vehicle, request, self.oracle)
        row: dict[str, Any] = dict(self.context)
        row.update(feats)
        row.update(
            {
                "decision_id": self._decision_id,
                "request_id": int(request.id),
                "feasible": bool(result is not None),
                "cost": float(result.cost) if result is not None else 0.0,
                "pickup_at": int(result.pickup_at) if result is not None else -1,
                "dropoff_at": int(result.dropoff_at) if result is not None else -1,
            }
        )
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def to_dataframe(self):
        """Return a `pandas.DataFrame` of collected rows. Requires the [viz] extra."""
        import pandas as pd

        return pd.DataFrame(self._rows)

    def write_csv(self, path: str | Path) -> None:
        """Write collected rows to `path` as CSV. Requires the [viz] extra."""
        df = self.to_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
