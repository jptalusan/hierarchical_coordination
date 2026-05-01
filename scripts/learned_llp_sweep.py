"""Compare heuristic vs learned-LLP plug-in on the scaleup grid.

For each (fleet, intensity), runs:
  - monolithic baseline (heuristic LLP, full exhaustive over fleet)
  - hierarchical K=5 baseline (heuristic LLP, full exhaustive within region)
  - monolithic + learned LLP (top-K candidate selection then exhaustive (p,q))
  - hierarchical K=5 + learned LLP (same, restricted to region)

Outputs results.csv + comparison plots in outputs/learned_llp/.

Requires the trained scorer at outputs/learning/scorer.pt
(produced by `scripts/train_scorer.py`).

Usage:
    uv run python scripts/learned_llp_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.analysis import run_sweep, save_figure  # noqa: E402
from hcoord.experiment import ExperimentConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "learned_llp"
SCORER_PATH = REPO_ROOT / "outputs" / "learning" / "scorer.pt"
SCORER_TOP_K = 5
FLEET_SIZES = [60, 120, 240]
INTENSITIES = [1.0, 3.0, 5.0]
SEED = 11


def configs():
    base = dict(
        seed=SEED,
        network="memphis_osm",
        n_outskirts=25,
        osm_outer_km=40.0,
        capacity=6,
        service_end_time=24 * 60.0,
        placement="hubs",
    )
    learned = dict(scorer_path=str(SCORER_PATH), scorer_top_k=SCORER_TOP_K)

    for fleet in FLEET_SIZES:
        for intensity in INTENSITIES:
            yield ExperimentConfig(**base, fleet_size=fleet, intensity=intensity,
                                   dispatcher="monolithic")
            yield ExperimentConfig(**base, **learned, fleet_size=fleet, intensity=intensity,
                                   dispatcher="monolithic")
            yield ExperimentConfig(**base, fleet_size=fleet, intensity=intensity,
                                   dispatcher="hierarchical", n_regions=5)
            yield ExperimentConfig(**base, **learned, fleet_size=fleet, intensity=intensity,
                                   dispatcher="hierarchical", n_regions=5)


def _arm_label(row: pd.Series) -> str:
    learned = "learned" if pd.notna(row.get("scorer_path", None)) and str(row["scorer_path"]) != "nan" else "heuristic"
    if row["dispatcher"] == "monolithic":
        return f"mono ({learned})"
    return f"hier K={int(row['n_regions'])} ({learned})"


def main() -> None:
    if not SCORER_PATH.exists():
        raise SystemExit(
            f"missing scorer at {SCORER_PATH}; run scripts/train_scorer.py first"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = list(configs())
    print(f"Running {len(cfgs)} configs with scorer at {SCORER_PATH}, top_k={SCORER_TOP_K}")
    df = run_sweep(cfgs)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    df["arm"] = df.apply(_arm_label, axis=1)

    print(f"\nWrote results: {OUT_DIR / 'results.csv'}")
    print(df[[
        "arm", "fleet_size", "intensity",
        "assignment_rate", "mean_wall_ms", "p95_wall_ms", "max_wall_ms",
    ]].to_string(index=False))

    # Pivot speedup factor: learned vs heuristic at the same (dispatcher, fleet, intensity).
    print("\n--- learned vs heuristic mean-wall speedup ---")
    for disp in ("monolithic", "hierarchical"):
        sub = df[df["dispatcher"] == disp]
        heur = sub[sub["scorer_path"].isna() | (sub["scorer_path"] == "")].set_index(
            ["fleet_size", "intensity"]
        )["mean_wall_ms"]
        learned = sub[sub["scorer_path"].notna() & (sub["scorer_path"] != "")].set_index(
            ["fleet_size", "intensity"]
        )["mean_wall_ms"]
        ratio = (heur / learned).unstack().round(2)
        print(f"\n{disp}:")
        print(ratio.to_string())

    # Quality drop check: assignment_rate (learned) should be >= assignment_rate (heuristic)
    # within rounding (fallback guarantees no quality loss).
    print("\n--- assignment-rate delta (learned - heuristic, in pp) ---")
    for disp in ("monolithic", "hierarchical"):
        sub = df[df["dispatcher"] == disp]
        heur = sub[sub["scorer_path"].isna() | (sub["scorer_path"] == "")].set_index(
            ["fleet_size", "intensity"]
        )["assignment_rate"]
        learned = sub[sub["scorer_path"].notna() & (sub["scorer_path"] != "")].set_index(
            ["fleet_size", "intensity"]
        )["assignment_rate"]
        delta_pp = ((learned - heur) * 100).unstack().round(2)
        print(f"\n{disp}:")
        print(delta_pp.to_string())

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(6 * len(INTENSITIES), 4),
                             sharey=True)
    for ax, intensity in zip(axes, INTENSITIES):
        sub = df[df["intensity"] == intensity].copy()
        for arm, g in sub.groupby("arm"):
            g = g.sort_values("fleet_size")
            ax.plot(g["fleet_size"], g["mean_wall_ms"], marker="o", label=arm)
        ax.set_xlabel("fleet size")
        ax.set_ylabel("mean wall (ms)")
        ax.set_title(f"intensity={intensity}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    save_figure(fig, OUT_DIR / "wall_vs_fleet.png")

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(6 * len(INTENSITIES), 4),
                             sharey=True)
    for ax, intensity in zip(axes, INTENSITIES):
        sub = df[df["intensity"] == intensity].copy()
        for arm, g in sub.groupby("arm"):
            g = g.sort_values("fleet_size")
            ax.plot(g["fleet_size"], g["assignment_rate"] * 100, marker="o", label=arm)
        ax.set_xlabel("fleet size")
        ax.set_ylabel("assignment rate (%)")
        ax.set_title(f"intensity={intensity}")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    save_figure(fig, OUT_DIR / "assignment_vs_fleet.png")

    print(f"\nSaved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
