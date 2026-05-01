"""Train an InsertionScorer on the dumped LLP dataset.

Splits by seed: trains on seeds 11 + 23, tests on seed 37 (held-out).
The held-out seed has different outskirt zone IDs (substrate is sampled
per-seed), so this measures generalization across both seed-level demand
randomness AND the spatial layout of outskirt zones.

Outputs:
    outputs/learning/scorer.pt        # checkpoint
    outputs/learning/scorer.eval.json # final metrics + history

Usage:
    uv run python scripts/train_scorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.learning.dataset import (  # noqa: E402
    CONTINUOUS_FEATURES,
    load_dataset,
    prepare,
    split_by_run,
)
from hcoord.learning.model import ScorerConfig  # noqa: E402
from hcoord.learning.train import TrainConfig, save_checkpoint, train_scorer  # noqa: E402

DATA_PATH = REPO_ROOT / "outputs" / "learning" / "dataset.csv"
CKPT_PATH = REPO_ROOT / "outputs" / "learning" / "scorer.pt"
TEST_SEEDS = (37,)


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"missing dataset at {DATA_PATH}; run scripts/collect_dataset.py first"
        )

    print(f"Loading {DATA_PATH}...")
    df = load_dataset(DATA_PATH)
    print(f"  rows: {len(df):,}, runs: {df['run_id'].nunique()}")

    train_df, test_df = split_by_run(df, test_seeds=TEST_SEEDS)
    print(f"  train: {len(train_df):,} rows ({train_df['run_id'].nunique()} runs)")
    print(f"  test:  {len(test_df):,} rows ({test_df['run_id'].nunique()} runs)")

    train_split, standardizer = prepare(train_df, feature_cols=CONTINUOUS_FEATURES)
    test_split, _ = prepare(test_df, standardizer=standardizer,
                            feature_cols=CONTINUOUS_FEATURES)

    model_cfg = ScorerConfig(
        n_features=train_split.X.shape[1],
        hidden_dim=64,
        n_hidden_layers=2,
    )
    train_cfg = TrainConfig(n_epochs=30, batch_size=1024, lr=1e-3, seed=0)

    print(f"\nTraining {model_cfg}\n         {train_cfg}\n")
    model, eval_dict = train_scorer(train_split, test_split,
                                    train_cfg=train_cfg, model_cfg=model_cfg)

    print("\n=== Final test metrics ===")
    for k, v in eval_dict["final"].items():
        print(f"  {k}: {v}")

    save_checkpoint(CKPT_PATH, model, standardizer, eval_dict)
    print(f"\nSaved checkpoint: {CKPT_PATH}")
    print(f"Saved eval log:   {CKPT_PATH.with_suffix('.eval.json')}")


if __name__ == "__main__":
    main()
