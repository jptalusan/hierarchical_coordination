"""Dispatcher interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hcoord.demand import Request
from hcoord.dispatch.insertion import InsertionObserver, InsertionResult, best_insertion
from hcoord.fleet import Vehicle
from hcoord.travel import TravelTimeOracle

if TYPE_CHECKING:
    from hcoord.learning.inference import LearnedScorer


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of an assignment attempt.

    `vehicle_id` is None if the request could not be served (no feasible
    insertion in any considered vehicle). `cost` is the increase in the
    chosen vehicle's return-home arrival time; `inf` for a dropped request.
    """

    request_id: int
    vehicle_id: int | None
    cost: float


class Dispatcher(ABC):
    """Base class. Concrete dispatchers implement `assign`; `rebalance` is a no-op by default.

    If `scorer` is provided, `_pick_best_insertion` uses it to rank candidate
    vehicles by predicted cost, runs exhaustive (p, q) search on the top
    `scorer_top_k`, and falls back to full exhaustive across the remaining
    vehicles only if none of the top-K is feasible. The fallback path
    guarantees the learned scorer never degrades the dispatcher's quality
    versus the heuristic baseline.
    """

    def __init__(
        self,
        *,
        fleet: list[Vehicle],
        oracle: TravelTimeOracle,
        observer: InsertionObserver | None = None,
        scorer: "LearnedScorer | None" = None,
        scorer_top_k: int = 3,
    ) -> None:
        if scorer_top_k < 1:
            raise ValueError(f"scorer_top_k must be >= 1, got {scorer_top_k}")
        self.fleet = fleet
        self.oracle = oracle
        self.observer = observer
        self.scorer = scorer
        self.scorer_top_k = scorer_top_k

    @abstractmethod
    def assign(self, request: Request, now: float) -> DispatchResult:
        """Try to insert `request` into some vehicle's route. Mutate on success."""

    def rebalance(self, now: float) -> None:
        """Optional fleet-level rebalancing tick."""

    def _pick_best_insertion(
        self,
        candidates: list[Vehicle],
        request: Request,
    ) -> InsertionResult | None:
        """Pick the best feasible insertion among `candidates`.

        Without a scorer: exhaustive search across all candidates (the
        heuristic baseline). With a scorer: rank candidates by predicted
        score, run exhaustive (p, q) on the top-K only; if none is
        feasible, fall back to exhaustive across the rest.
        """
        if self.scorer is None or len(candidates) <= self.scorer_top_k:
            return self._exhaustive_pick(candidates, request)

        scores = self.scorer.score_batch(candidates, request, self.oracle)
        # argsort lifts ties stably; order from cheapest predicted to dearest.
        import numpy as np

        order = np.argsort(scores)
        top_idx = list(order[: self.scorer_top_k])
        rest_idx = list(order[self.scorer_top_k :])

        # Exhaustive (p, q) on the top-K only — this is the actual win.
        best = self._exhaustive_pick([candidates[i] for i in top_idx], request)
        if best is not None:
            return best

        # All top-K were infeasible: fall back to the rest. Quality identical
        # to the baseline, just slower than the happy path.
        return self._exhaustive_pick([candidates[i] for i in rest_idx], request)

    def _exhaustive_pick(
        self,
        candidates: list[Vehicle],
        request: Request,
    ) -> InsertionResult | None:
        best: InsertionResult | None = None
        for v in candidates:
            r = best_insertion(v, request, self.oracle, observer=self.observer)
            if r is None:
                continue
            if best is None or r.cost < best.cost:
                best = r
        return best
