"""Compare heuristic vs learned-LLP (rank vs filter mode) on the scaleup grid.

The rank-mode plug-in (top-K by predicted cost) lost up to -21pp of
assignment rate at saturation because the v1 model can't discriminate
between similar empty-route vehicles at the same hub. Filter-mode uses
only the model's strong head — feasibility classification at 98.9% — to
drop confident-infeasibles and run exhaustive on the rest. This salvages
the speedup at saturation while preserving quality.

Three arms per cell:
  - heuristic baseline (no scorer)
  - learned-LLP, rank mode (top-K = 5)
  - learned-LLP, filter mode (logit threshold = -2.0)

Usage:
    uv run python scripts/learned_llp_filter_sweep.py
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

OUT_DIR = REPO_ROOT / "outputs" / "learned_llp_filter"
SCORER_PATH = REPO_ROOT / "outputs" / "learning" / "scorer.pt"
SEEDS = [11, 23, 37]  # 3 seeds for tighter saturation numbers
FLEET_SIZES = [60, 120, 240]
INTENSITIES = [1.0, 3.0, 5.0]
SCORER_TOP_K = 5
# Threshold = -4.0 → drop only vehicles with predicted P(feasible) < 1.8%.
# A scan of {-1,-2,-3,-4,-5} at the saturation cell (mono fleet=120 int=5)
# showed -4.0 closes the residual quality gap (~0pp) while still giving 2x
# speedup (the heavy-route vehicles, which dominate exhaustive (p,q) cost,
# are still the ones most confidently classified infeasible at saturation).
FILTER_THRESHOLD = -4.0


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
                # Heuristic
                yield ExperimentConfig(
                    **base, seed=seed, fleet_size=fleet, intensity=intensity,
                    dispatcher="monolithic",
                )
                # Learned, rank
                yield ExperimentConfig(
                    **base, seed=seed, fleet_size=fleet, intensity=intensity,
                    dispatcher="monolithic",
                    scorer_path=str(SCORER_PATH),
                    scorer_mode="rank",
                    scorer_top_k=SCORER_TOP_K,
                )
                # Learned, filter
                yield ExperimentConfig(
                    **base, seed=seed, fleet_size=fleet, intensity=intensity,
                    dispatcher="monolithic",
                    scorer_path=str(SCORER_PATH),
                    scorer_mode="filter",
                    scorer_filter_logit_threshold=FILTER_THRESHOLD,
                )


def _arm_label(row) -> str:
    has_path = pd.notna(row.get("scorer_path", None)) and str(row["scorer_path"]) != "nan"
    if not has_path:
        return "heuristic"
    return f"learned ({row['scorer_mode']})"


def main() -> None:
    if not SCORER_PATH.exists():
        raise SystemExit(
            f"missing scorer at {SCORER_PATH}; run scripts/train_scorer.py first"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = list(configs())
    print(f"Running {len(cfgs)} configs (3 seeds × 9 cells × 3 arms)")
    df = run_sweep(cfgs)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    df["arm"] = df.apply(_arm_label, axis=1)

    print(f"\nWrote results: {OUT_DIR / 'results.csv'}")

    # Aggregate over seeds.
    agg = df.groupby(["arm", "fleet_size", "intensity"]).agg(
        assignment_mean=("assignment_rate", "mean"),
        assignment_std=("assignment_rate", "std"),
        wall_mean=("mean_wall_ms", "mean"),
        wall_std=("mean_wall_ms", "std"),
        max_wall_mean=("max_wall_ms", "mean"),
        n=("seed", "count"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "aggregated.csv", index=False)
    print("\n--- aggregated (mean across 3 seeds) ---")
    print(agg.to_string(index=False))

    # Speedup factors (heur / learned) and quality deltas (learned - heur).
    print("\n--- vs heuristic: speedup factor (mean wall) ---")
    h = agg[agg["arm"] == "heuristic"].set_index(["fleet_size", "intensity"])
    for arm in ("learned (rank)", "learned (filter)"):
        a = agg[agg["arm"] == arm].set_index(["fleet_size", "intensity"])
        ratio = (h["wall_mean"] / a["wall_mean"]).unstack().round(2)
        print(f"\n{arm}:")
        print(ratio.to_string())

    print("\n--- vs heuristic: assignment delta (pp, mean) ---")
    for arm in ("learned (rank)", "learned (filter)"):
        a = agg[agg["arm"] == arm].set_index(["fleet_size", "intensity"])
        delta = ((a["assignment_mean"] - h["assignment_mean"]) * 100).unstack().round(2)
        print(f"\n{arm}:")
        print(delta.to_string())

    # Plots: same layout as the prior comparison sweep but with 3 arms.
    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(6 * len(INTENSITIES), 4),
                             sharey=True)
    for ax, intensity in zip(axes, INTENSITIES):
        sub = agg[agg["intensity"] == intensity]
        for arm, g in sub.groupby("arm"):
            g = g.sort_values("fleet_size")
            ax.errorbar(g["fleet_size"], g["wall_mean"], yerr=g["wall_std"],
                        marker="o", capsize=3, label=arm)
        ax.set_xlabel("fleet size")
        ax.set_ylabel("mean wall (ms)")
        ax.set_title(f"intensity={intensity}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    save_figure(fig, OUT_DIR / "wall_vs_fleet.png")

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(6 * len(INTENSITIES), 4),
                             sharey=True)
    for ax, intensity in zip(axes, INTENSITIES):
        sub = agg[agg["intensity"] == intensity]
        for arm, g in sub.groupby("arm"):
            g = g.sort_values("fleet_size")
            ax.errorbar(g["fleet_size"], g["assignment_mean"] * 100,
                        yerr=g["assignment_std"] * 100,
                        marker="o", capsize=3, label=arm)
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
