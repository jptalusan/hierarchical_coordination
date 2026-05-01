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

    If `scorer` is provided, `_pick_best_insertion` uses one of two modes:

    - `mode="rank"`: rank candidates by predicted cost, exhaustive (p, q) on
      the top `scorer_top_k`, fall back to exhaustive on the rest only if
      all top-K are infeasible. Best when the model's cost ranking is
      reliable.
    - `mode="filter"`: drop candidates whose feasibility logit is below
      `scorer_filter_logit_threshold` (i.e., model is confident they're
      infeasible), then exhaustive over the survivors. Best when the
      model's feasibility head is much more reliable than its cost ranking
      — typical for the v1 MLP. Quality is bounded by the false-negative
      rate of the feasibility classifier; the threshold lets the caller
      tune the speed/quality tradeoff explicitly.

    Either way, exhaustive (p, q) feasibility checks run on every retained
    vehicle, so the dispatcher never returns an actually-infeasible result.
    """

    def __init__(
        self,
        *,
        fleet: list[Vehicle],
        oracle: TravelTimeOracle,
        observer: InsertionObserver | None = None,
        scorer: "LearnedScorer | None" = None,
        scorer_mode: str = "rank",
        scorer_top_k: int = 3,
        scorer_filter_logit_threshold: float = -2.0,
    ) -> None:
        if scorer_top_k < 1:
            raise ValueError(f"scorer_top_k must be >= 1, got {scorer_top_k}")
        if scorer_mode not in ("rank", "filter"):
            raise ValueError(
                f"scorer_mode must be 'rank' or 'filter', got {scorer_mode!r}"
            )
        self.fleet = fleet
        self.oracle = oracle
        self.observer = observer
        self.scorer = scorer
        self.scorer_mode = scorer_mode
        self.scorer_top_k = scorer_top_k
        self.scorer_filter_logit_threshold = scorer_filter_logit_threshold

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

        Dispatches to the configured mode. See class docstring.
        """
        if self.scorer is None:
            return self._exhaustive_pick(candidates, request)
        if self.scorer_mode == "filter":
            return self._filter_pick(candidates, request)
        return self._rank_pick(candidates, request)

    def _rank_pick(
        self,
        candidates: list[Vehicle],
        request: Request,
    ) -> InsertionResult | None:
        if len(candidates) <= self.scorer_top_k:
            return self._exhaustive_pick(candidates, request)

        import numpy as np

        scores = self.scorer.score_batch(candidates, request, self.oracle)
        order = np.argsort(scores)
        top_idx = list(order[: self.scorer_top_k])
        rest_idx = list(order[self.scorer_top_k :])

        best = self._exhaustive_pick([candidates[i] for i in top_idx], request)
        if best is not None:
            return best
        return self._exhaustive_pick([candidates[i] for i in rest_idx], request)

    def _filter_pick(
        self,
        candidates: list[Vehicle],
        request: Request,
    ) -> InsertionResult | None:
        """Drop confident-infeasible candidates, run exhaustive on the rest.

        If the survivor set is empty (every vehicle was predicted infeasible),
        fall back to exhaustive over the full candidate list — guarantees we
        never silently drop a feasible insertion just because the classifier
        was uniformly pessimistic.
        """
        if not candidates:
            return None

        logits = self.scorer.feasibility_logits(candidates, request, self.oracle)
        keep = logits >= self.scorer_filter_logit_threshold
        survivors = [c for c, k in zip(candidates, keep) if k]
        if not survivors:
            return self._exhaustive_pick(candidates, request)
        best = self._exhaustive_pick(survivors, request)
        if best is not None:
            return best
        # All survivors were actually infeasible — try the dropped ones.
        rest = [c for c, k in zip(candidates, keep) if not k]
        return self._exhaustive_pick(rest, request)

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
