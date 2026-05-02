"""Collect rebalance-tick state + heuristic-allocation rows for the learned HLP.

Runs hierarchical configs with `collect_hlp_to=` set, dumps per-config
CSVs into `outputs/learning/hlp/`, concatenates into `dataset.csv`.

Each row is one rebalance tick: per-region state (max 5 regions, padded
with zeros) + heuristic-chosen target counts. Hindsight-optimal labels
come in step 2.

Usage:
    uv run python scripts/collect_hlp_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.experiment import ExperimentConfig, run_experiment  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "learning" / "hlp"
SEEDS = [11, 23, 37]
FLEET_SIZES = [60, 120]
INTENSITIES = [1.0, 3.0, 5.0]
N_REGIONS_VALUES = [3, 5]


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
                for k in N_REGIONS_VALUES:
                    yield ExperimentConfig(
                        **base, seed=seed, fleet_size=fleet,
                        intensity=intensity,
                        dispatcher="hierarchical", n_regions=k,
                    )


def _slug(cfg: ExperimentConfig) -> str:
    return f"s{cfg.seed}_f{cfg.fleet_size}_i{cfg.intensity:g}_k{cfg.n_regions}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = list(configs())
    print(f"Collecting HLP rows from {len(cfgs)} configs into {OUT_DIR}/")

    paths: list[Path] = []
    for i, cfg in enumerate(cfgs):
        path = OUT_DIR / f"{_slug(cfg)}.csv"
        cfg.collect_hlp_to = str(path)
        print(f"[{i + 1}/{len(cfgs)}] {_slug(cfg)}", flush=True)
        run_experiment(cfg)
        paths.append(path)

    print("\nConcatenating per-config CSVs...")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    out = OUT_DIR / "dataset.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {out} ({len(df):,} rows, {df.shape[1]} columns)")
    print("\n--- per-config tick counts ---")
    print(df.groupby(["seed", "fleet_size", "intensity", "n_regions"]).size().to_string())
    print("\n--- target column non-null fractions ---")
    for slot in range(5):
        col = f"target_r{slot}"
        if col in df.columns:
            frac = df[col].notna().mean()
            print(f"  {col}: {frac:.2%} non-null")


if __name__ == "__main__":
    main()
