"""Tests for the LearnedScorer inference adapter."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hcoord.demand import generate_requests  # noqa: E402
from hcoord.geography import build_memphis_outskirts  # noqa: E402
from hcoord.learning.dataset import CONTINUOUS_FEATURES, Standardizer  # noqa: E402
from hcoord.learning.inference import LearnedScorer  # noqa: E402
from hcoord.learning.model import InsertionScorer, ScorerConfig  # noqa: E402
from hcoord.learning.train import save_checkpoint  # noqa: E402
from hcoord.placement import make_fleet  # noqa: E402
from hcoord.travel import TravelTimeOracle  # noqa: E402


def _toy_scorer() -> tuple[LearnedScorer, dict]:
    """Build a fresh untrained scorer + standardizer for plumbing tests."""
    # Pin RNG: untrained weights can otherwise put log1p_cost in a regime
    # where expm1 overflows to inf, making score_pair return inf.
    torch.manual_seed(0)
    cfg = ScorerConfig(n_features=len(CONTINUOUS_FEATURES), hidden_dim=16,
                       n_hidden_layers=2)
    model = InsertionScorer(cfg)
    # Identity standardizer: mean=0, std=1 — makes the inputs equal raw features
    # so any plumbing bug shows up loudly.
    std = Standardizer(
        mean=np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float32),
        std=np.ones(len(CONTINUOUS_FEATURES), dtype=np.float32),
        feature_names=CONTINUOUS_FEATURES,
    )
    return LearnedScorer(model=model, standardizer=std), {"cfg": cfg, "model": model}


def _toy_setup():
    net = build_memphis_outskirts(seed=7, n_outskirts=10)
    oracle = TravelTimeOracle(net)
    fleet = make_fleet(network=net, fleet_size=4, capacity=4,
                       service_end_time=1440.0, placement="hubs")
    requests = generate_requests(net, seed=7, intensity=1.0,
                                 announce_lead_min=90.0, arrival_buffer_min=15.0)
    return oracle, fleet, requests


def test_score_pair_returns_finite_float():
    scorer, _ = _toy_scorer()
    oracle, fleet, requests = _toy_setup()
    s = scorer.score_pair(fleet[0], requests[0], oracle)
    assert np.isfinite(s)


def test_score_batch_matches_score_pair_per_vehicle():
    scorer, _ = _toy_scorer()
    oracle, fleet, requests = _toy_setup()
    batch = scorer.score_batch(fleet, requests[0], oracle)
    assert batch.shape == (len(fleet),)
    for i, v in enumerate(fleet):
        single = scorer.score_pair(v, requests[0], oracle)
        np.testing.assert_allclose(batch[i], single, rtol=1e-5, atol=1e-6)


def test_score_batch_empty_returns_empty_array():
    scorer, _ = _toy_scorer()
    oracle, _, requests = _toy_setup()
    out = scorer.score_batch([], requests[0], oracle)
    assert out.shape == (0,)


def test_from_checkpoint_round_trip(tmp_path):
    cfg = ScorerConfig(n_features=len(CONTINUOUS_FEATURES), hidden_dim=8,
                       n_hidden_layers=1)
    model = InsertionScorer(cfg)
    std = Standardizer(
        mean=np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float32),
        std=np.ones(len(CONTINUOUS_FEATURES), dtype=np.float32),
        feature_names=CONTINUOUS_FEATURES,
    )
    save_checkpoint(tmp_path / "scorer.pt", model, std, eval_dict={})
    loaded = LearnedScorer.from_checkpoint(tmp_path / "scorer.pt")
    assert loaded.feature_names == CONTINUOUS_FEATURES

    oracle, fleet, requests = _toy_setup()
    a = loaded.score_pair(fleet[0], requests[0], oracle)
    # Reconstruct manually from the original model + std for byte-equality.
    direct = LearnedScorer(model=model, standardizer=std)
    b = direct.score_pair(fleet[0], requests[0], oracle)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_mismatched_feature_names_rejected():
    cfg = ScorerConfig(n_features=len(CONTINUOUS_FEATURES))
    model = InsertionScorer(cfg)
    bad_std = Standardizer(
        mean=np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float32),
        std=np.ones(len(CONTINUOUS_FEATURES), dtype=np.float32),
        feature_names=("foo", "bar"),  # wrong shape — distinct from CONTINUOUS_FEATURES
    )
    with pytest.raises(ValueError, match="feature_names"):
        LearnedScorer(model=model, standardizer=bad_std)
