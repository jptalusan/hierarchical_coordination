"""Load + prepare collected (vehicle, request) → cost rows for training.

Splits by `run_id` so no individual decision leaks across train/test. The
held-out runs share the OSM substrate and demand model with training runs,
so the test set measures generalization across seeds + operating points,
not across geographies.

Categorical zone IDs (req_origin, req_destination, veh_home, veh_location,
route_last_zone, veh_id) are excluded for v1 because outskirt zone numbering
varies per seed — the same integer means different physical places across
runs. Re-add as embeddings if a future version uses per-substrate models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CATEGORICAL_ID_COLS: tuple[str, ...] = (
    "req_origin",
    "req_destination",
    "veh_id",
    "veh_home",
    "veh_location",
    "route_last_zone",
)

CONTINUOUS_FEATURES: tuple[str, ...] = (
    "req_announce_time",
    "req_earliest_pickup",
    "req_latest_arrival",
    "req_window_min",
    "req_shift_id",
    "veh_capacity",
    "veh_available_time",
    "veh_route_len",
    "veh_base_return_time",
    "veh_slack_to_service_end",
    "pair_tt_loc_to_origin",
    "pair_tt_origin_to_dest",
    "pair_tt_dest_to_home",
    "pair_delta_naive",
    "route_min_tt_to_origin",
    "route_mean_tt_to_origin",
    "route_min_tt_to_dest",
    "route_mean_tt_to_dest",
    "route_last_tt_to_origin",
    "route_n_pickups",
    "route_n_dropoffs",
)


@dataclass
class Standardizer:
    """Z-score features using train-set statistics. Reusable at inference."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    @classmethod
    def fit(cls, X: np.ndarray, feature_names: Iterable[str]) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard against zero-variance columns (constant context that slipped through).
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std, feature_names=tuple(feature_names))


@dataclass
class PreparedSplit:
    X: np.ndarray  # (n_rows, n_features) float32, standardized
    feasible: np.ndarray  # (n_rows,) bool
    cost: np.ndarray  # (n_rows,) float32, 0 where infeasible (mask with `feasible`)
    run_id: np.ndarray  # (n_rows,) str
    decision_id: np.ndarray  # (n_rows,) int — local to (run_id, decision_id)
    feature_names: tuple[str, ...]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in CONTINUOUS_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required feature columns: {missing}")
    return df


def split_by_run(
    df: pd.DataFrame,
    *,
    test_run_filter: dict[str, object] | None = None,
    test_seeds: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by `run_id` so all rows of a run go to the same side.

    Either pass `test_seeds=(...)` to hold out runs by seed, or
    `test_run_filter={"col": value}` for a custom predicate. Exactly one.
    """
    if (test_run_filter is None) == (test_seeds is None):
        raise ValueError("specify exactly one of test_run_filter or test_seeds")

    if test_seeds is not None:
        mask = df["seed"].isin(test_seeds)
    else:
        assert test_run_filter is not None
        mask = pd.Series(True, index=df.index)
        for col, val in test_run_filter.items():
            mask &= df[col] == val
    train = df[~mask].reset_index(drop=True)
    test = df[mask].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("split produced an empty side")
    return train, test


def prepare(
    df: pd.DataFrame,
    *,
    standardizer: Standardizer | None = None,
    feature_cols: tuple[str, ...] = CONTINUOUS_FEATURES,
) -> tuple[PreparedSplit, Standardizer]:
    """Pull features + labels out of `df`. Fits a standardizer if not provided.

    Returns the prepared split AND the standardizer (so the caller can reuse
    the train-set fit on the test split).
    """
    X_raw = df[list(feature_cols)].to_numpy(dtype=np.float32)
    if standardizer is None:
        standardizer = Standardizer.fit(X_raw, feature_cols)
    X = standardizer.transform(X_raw).astype(np.float32)
    feasible = df["feasible"].to_numpy(dtype=bool)
    cost = df["cost"].to_numpy(dtype=np.float32)
    return (
        PreparedSplit(
            X=X,
            feasible=feasible,
            cost=cost,
            run_id=df["run_id"].to_numpy(),
            decision_id=df["decision_id"].to_numpy(dtype=np.int64),
            feature_names=tuple(feature_cols),
        ),
        standardizer,
    )
