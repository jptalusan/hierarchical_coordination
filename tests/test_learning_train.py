"""Tests for the LLP scorer training pipeline.

Uses synthetic data so the test is fast and doesn't depend on the OSM
collection. Verifies dataset prep, no-leak split, model forward shape,
that one training step decreases the loss, and checkpoint round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hcoord.learning.dataset import (  # noqa: E402
    CONTINUOUS_FEATURES,
    Standardizer,
    prepare,
    split_by_run,
)
from hcoord.learning.model import InsertionScorer, ScorerConfig  # noqa: E402
from hcoord.learning.train import (  # noqa: E402
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
    train_scorer,
)


def _synth_df(n_per_run: int = 50, n_runs: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_runs):
        run_id = f"run_{r}"
        for d in range(n_per_run // 5):  # 5 vehicles per decision
            for v in range(5):
                feats = {f: float(rng.standard_normal()) for f in CONTINUOUS_FEATURES}
                # Make pair_delta_naive a strong predictor of cost.
                cost = abs(feats["pair_delta_naive"]) * 5.0 + rng.uniform(0, 5)
                feasible = bool(cost < 50.0)
                row = dict(feats)
                row.update(
                    {
                        "run_id": run_id,
                        "seed": r,
                        "decision_id": d,
                        "feasible": feasible,
                        "cost": cost if feasible else 0.0,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_split_by_seed_no_overlap():
    df = _synth_df(n_per_run=20, n_runs=4)
    train, test = split_by_run(df, test_seeds=(0, 1))
    assert set(train["seed"].unique()).isdisjoint(set(test["seed"].unique()))
    assert len(train) + len(test) == len(df)


def test_split_by_seed_rejects_empty_side():
    df = _synth_df(n_per_run=20, n_runs=2)
    with pytest.raises(ValueError, match="empty side"):
        split_by_run(df, test_seeds=(99,))


def test_split_by_seed_requires_one_arg():
    df = _synth_df(n_per_run=20, n_runs=2)
    with pytest.raises(ValueError, match="exactly one"):
        split_by_run(df, test_seeds=(0,), test_run_filter={"seed": 0})
    with pytest.raises(ValueError, match="exactly one"):
        split_by_run(df)


def test_standardizer_zero_variance_safe():
    X = np.array([[1.0, 5.0], [1.0, 5.0]], dtype=np.float32)
    std = Standardizer.fit(X, ("a", "b"))
    Z = std.transform(X)
    # Zero-variance columns become 0 (after subtracting mean), not NaN.
    assert np.isfinite(Z).all()
    assert (Z == 0).all()


def test_prepare_round_trip_and_train_test_use_same_standardizer():
    df = _synth_df()
    train_df, test_df = split_by_run(df, test_seeds=(3,))
    train_split, std = prepare(train_df)
    test_split, _ = prepare(test_df, standardizer=std)
    # Train z-scores have ~zero mean + unit std (within sampling noise).
    assert abs(train_split.X.mean()) < 0.5
    # Test z-scores reuse train's mean/std (so test stats may differ).
    assert train_split.feature_names == test_split.feature_names
    assert train_split.X.shape[1] == len(CONTINUOUS_FEATURES)


def test_model_forward_shapes():
    cfg = ScorerConfig(n_features=len(CONTINUOUS_FEATURES))
    model = InsertionScorer(cfg)
    X = torch.zeros((7, cfg.n_features))
    feas, log1p_cost = model(X)
    assert feas.shape == (7,)
    assert log1p_cost.shape == (7,)
    score = model.score(X)
    assert score.shape == (7,)


def test_one_short_training_run_improves_metrics(tmp_path):
    df = _synth_df(n_per_run=400, n_runs=4, seed=42)
    train_df, test_df = split_by_run(df, test_seeds=(3,))
    train_split, std = prepare(train_df)
    test_split, _ = prepare(test_df, standardizer=std)
    cfg = ScorerConfig(n_features=train_split.X.shape[1], hidden_dim=64,
                       n_hidden_layers=2)
    train_cfg = TrainConfig(n_epochs=25, batch_size=64, seed=0)
    model, eval_dict = train_scorer(train_split, test_split,
                                    train_cfg=train_cfg, model_cfg=cfg,
                                    log_every=10)

    history = eval_dict["history"]
    # Loss decreases from start to end.
    assert history[-1]["loss"] < history[0]["loss"]
    # Top-1 accuracy beats chance (1/5 = 0.2). Synthetic cost is
    # abs(pair_delta_naive) * 5 + noise, so the model must learn that
    # particular feature dominates ranking — strong but not trivial.
    assert eval_dict["final"]["top1_acc"] > 0.35
    # Top-3 accuracy should beat top-1.
    assert eval_dict["final"]["top3_acc"] >= eval_dict["final"]["top1_acc"]

    # Checkpoint round-trip.
    ckpt = tmp_path / "scorer.pt"
    save_checkpoint(ckpt, model, std, eval_dict)
    assert ckpt.exists()
    assert ckpt.with_suffix(".eval.json").exists()
    loaded_model, loaded_std, loaded_eval = load_checkpoint(ckpt)
    assert loaded_eval["final"]["top1_acc"] == eval_dict["final"]["top1_acc"]
    # Same prediction up to numerical noise.
    X = torch.from_numpy(test_split.X[:5])
    with torch.no_grad():
        s1 = model.score(X).numpy()
        s2 = loaded_model.score(X).numpy()
    np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-6)
