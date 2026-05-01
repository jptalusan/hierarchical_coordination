"""Collect (vehicle, request) → cost training data for the learned LLP.

Runs a small grid of configs with `collect_to=` set, dumping per-config
CSVs into `outputs/learning/` and concatenating them into `dataset.csv`.

Each row is one (vehicle, request) candidate evaluation: features (see
`hcoord.learning.features`) + label (cost, feasible). Context columns
(seed, fleet_size, intensity, dispatcher, n_regions) make the dataset
easy to slice.

Usage:
    uv run python scripts/collect_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.experiment import ExperimentConfig, run_experiment  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "learning"
SEEDS = [11, 23, 37]
FLEET_SIZES = [60, 120]
INTENSITIES = [1.0, 3.0, 5.0]
# Use only monolithic for collection: it sees the full fleet per decision,
# giving denser per-(vehicle, request) coverage. Hierarchical would only
# log within-region candidates, which biases the feature distribution.
DISPATCHERS = [("monolithic", 0)]


def configs():
    base = dict(
        network="memphis_osm",
        n_outskirts=25,
        osm_outer_km=40.0,
        capacity=6,
        service_end_time=24 * 60.0,
        placement="hubs",
    )
    for seed in SEEDS:
        for fleet in FLEET_SIZES:
            for intensity in INTENSITIES:
                for disp, k in DISPATCHERS:
                    if disp == "hierarchical":
                        yield ExperimentConfig(
                            **base, seed=seed, fleet_size=fleet,
                            intensity=intensity, dispatcher=disp, n_regions=k,
                        )
                    else:
                        yield ExperimentConfig(
                            **base, seed=seed, fleet_size=fleet,
                            intensity=intensity, dispatcher=disp,
                        )


def _slug(cfg: ExperimentConfig) -> str:
    bits = [
        f"s{cfg.seed}",
        f"f{cfg.fleet_size}",
        f"i{cfg.intensity:g}",
        cfg.dispatcher,
    ]
    if cfg.dispatcher == "hierarchical":
        bits.append(f"k{cfg.n_regions}")
    return "_".join(bits)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = list(configs())
    print(f"Collecting from {len(cfgs)} configs into {OUT_DIR}/")

    paths: list[Path] = []
    for i, cfg in enumerate(cfgs):
        path = OUT_DIR / f"{_slug(cfg)}.csv"
        cfg.collect_to = str(path)
        print(f"[{i + 1}/{len(cfgs)}] {_slug(cfg)}", flush=True)
        run_experiment(cfg)
        paths.append(path)

    print("\nConcatenating per-config CSVs...")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / "dataset.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {out} ({len(df):,} rows, {df.shape[1]} columns)")
    print("\n--- per-config row counts ---")
    print(df.groupby(["seed", "fleet_size", "intensity"]).size().to_string())
    print("\n--- feasibility breakdown ---")
    print(df["feasible"].value_counts().to_string())
    print("\n--- cost stats (feasible only) ---")
    print(df.loc[df["feasible"], "cost"].describe().to_string())


if __name__ == "__main__":
    main()
