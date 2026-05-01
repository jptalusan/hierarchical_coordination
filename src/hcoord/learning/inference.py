"""Inference-side adapter: standardize + score a (vehicle, request) candidate.

The dispatcher plug-in (step 3) holds one `LearnedScorer` and calls
`score_pair(vehicle, request)` per candidate. This module hides the
extract-features → vector-in-CONTINUOUS_FEATURES-order → standardize →
model.score plumbing so the dispatcher doesn't need to know about it.

Loading from a checkpoint:

    scorer = LearnedScorer.from_checkpoint("outputs/learning/scorer.pt")
    score = scorer.score_pair(vehicle, request, oracle)  # lower is better

Batch inference (preferred for a full fleet at one decision):

    scores = scorer.score_batch(vehicles, request, oracle)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hcoord.demand import Request
from hcoord.fleet import Vehicle
from hcoord.learning.dataset import CONTINUOUS_FEATURES, Standardizer
from hcoord.learning.features import extract_features
from hcoord.learning.model import InsertionScorer
from hcoord.learning.train import load_checkpoint
from hcoord.travel import TravelTimeOracle


class LearnedScorer:
    """Wraps a trained `InsertionScorer` and its training standardizer.

    `feature_names` are the feature columns the model was trained on, in
    order. We re-extract via `features.extract_features` at inference time
    and pull only those columns, so the call site doesn't need to memorize
    the order.
    """

    def __init__(
        self,
        model: InsertionScorer,
        standardizer: Standardizer,
        feature_names: tuple[str, ...] = CONTINUOUS_FEATURES,
    ) -> None:
        if standardizer.feature_names != feature_names:
            raise ValueError(
                "standardizer.feature_names doesn't match LearnedScorer.feature_names; "
                "checkpoint may have been trained with a different feature set"
            )
        self.model = model.eval()
        self.standardizer = standardizer
        self.feature_names = feature_names

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "LearnedScorer":
        model, standardizer, _ = load_checkpoint(path)
        return cls(model=model, standardizer=standardizer,
                   feature_names=standardizer.feature_names)

    def _vectorize(self, feats: dict) -> np.ndarray:
        return np.array(
            [feats[name] for name in self.feature_names], dtype=np.float32
        )

    @torch.no_grad()
    def score_pair(
        self,
        vehicle: Vehicle,
        request: Request,
        oracle: TravelTimeOracle,
    ) -> float:
        """Lower-is-better score for one (vehicle, request) candidate."""
        feats = extract_features(vehicle, request, oracle)
        x = self._vectorize(feats)
        x = self.standardizer.transform(x[None, :]).astype(np.float32)
        return float(self.model.score(torch.from_numpy(x))[0])

    @torch.no_grad()
    def score_batch(
        self,
        vehicles: list[Vehicle],
        request: Request,
        oracle: TravelTimeOracle,
    ) -> np.ndarray:
        """Lower-is-better scores for a list of vehicles, one shared request."""
        if not vehicles:
            return np.empty(0, dtype=np.float32)
        rows = [
            self._vectorize(extract_features(v, request, oracle)) for v in vehicles
        ]
        X = self.standardizer.transform(np.stack(rows, axis=0)).astype(np.float32)
        return self.model.score(torch.from_numpy(X)).numpy()
