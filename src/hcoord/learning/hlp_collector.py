"""HLPCollector: observer that records (state, allocation) per rebalance tick.

Each row corresponds to one call of `HierarchicalDispatcher.rebalance`:
the per-region state features plus the heuristic's chosen target counts.
Outcome / hindsight-optimal labels are NOT recorded here — that's step 2,
which re-runs the day with perturbed allocations and overwrites the
target column with the post-hoc optimum.

The observer is a callable invoked at the start of `rebalance(now)`,
*before* any moves are executed, so the recorded state is the pre-move
snapshot the HLP sees when deciding.

Usage:

    collector = HLPCollector(context={"seed": 11, ...})
    dispatcher = HierarchicalDispatcher(..., hlp_observer=collector)
    # ... run experiment ...
    df = collector.to_dataframe()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcoord.learning.hlp_features import MAX_REGIONS


@dataclass
class HLPRow:
    state: dict[str, float]
    targets: dict[int, int]  # region_id -> target vehicle count
    now: float


class HLPCollector:
    """Records (state, allocation) per rebalance tick. Targets are recorded
    as `target_r0 ... target_r{MAX_REGIONS-1}`; unused slots are NaN."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        self.context = dict(context or {})
        self._rows: list[dict[str, Any]] = []
        self._tick_id = 0

    def __call__(
        self,
        state: dict[str, float],
        targets: dict[int, int],
        now: float,
    ) -> None:
        # `now_min` already lives in `state` (set by extract_hlp_state). We
        # don't overwrite it. Step 2 (hindsight-optimal perturbations) joins
        # perturbed rollouts to the original ticks via (run_id, now_min) —
        # which is byte-stable across re-runs of the same config since
        # request announce times are pre-generated and rebalance throttling
        # is deterministic. `tick_id` is a convenience monotonic counter;
        # do not key on it across runs.
        self._tick_id += 1
        row: dict[str, Any] = dict(self.context)
        row.update(state)
        for slot in range(MAX_REGIONS):
            row[f"target_r{slot}"] = (
                float(targets[slot]) if slot in targets else float("nan")
            )
        row["tick_id"] = self._tick_id
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._rows)

    def write_csv(self, path: str | Path) -> None:
        df = self.to_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
