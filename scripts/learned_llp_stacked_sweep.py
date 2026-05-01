"""Full-stack speedup measurement: hierarchical × learned LLP.

Five arms per cell so the layering is visible:
  - mono + heuristic LLP            (the absolute baseline; used as denominator)
  - mono + learned LLP filter
  - hier K=5 + heuristic LLP        (write-up §1-3 result, "decomposition only")
  - hier K=5 + learned LLP filter   (decomposition + filter)
  - hier K=5 + learned LLP stacked  (decomposition + filter + top-K rank)

The compounding question: does learned LLP keep adding speedup *on top of*
hierarchical decomposition, or does it overlap?

Usage:
    uv run python scripts/learned_llp_stacked_sweep.py
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

OUT_DIR = REPO_ROOT / "outputs" / "learned_llp_stacked"
SCORER_PATH = REPO_ROOT / "outputs" / "learning" / "scorer.pt"
SEEDS = [11, 23, 37]
FLEET_SIZES = [60, 120, 240]
INTENSITIES = [1.0, 3.0, 5.0]
SCORER_TOP_K = 5
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
    learned_filter = dict(
        scorer_path=str(SCORER_PATH),
        scorer_mode="filter",
        scorer_filter_logit_threshold=FILTER_THRESHOLD,
    )
    learned_stacked = dict(
        scorer_path=str(SCORER_PATH),
        scorer_mode="stacked",
        scorer_top_k=SCORER_TOP_K,
        scorer_filter_logit_threshold=FILTER_THRESHOLD,
    )

    for seed in SEEDS:
        for fleet in FLEET_SIZES:
            for intensity in INTENSITIES:
                # mono + heuristic
                yield ExperimentConfig(
                    **base, seed=seed, fleet_size=fleet, intensity=intensity,
                    dispatcher="monolithic",
                )
                # mono + learned filter
                yield ExperimentConfig(
                    **base, **learned_filter, seed=seed, fleet_size=fleet,
                    intensity=intensity, dispatcher="monolithic",
                )
                # hier + heuristic
                yield ExperimentConfig(
                    **base, seed=seed, fleet_size=fleet, intensity=intensity,
                    dispatcher="hierarchical", n_regions=5,
                )
                # hier + learned filter
                yield ExperimentConfig(
                    **base, **learned_filter, seed=seed, fleet_size=fleet,
                    intensity=intensity, dispatcher="hierarchical", n_regions=5,
                )
                # hier + learned stacked
                yield ExperimentConfig(
                    **base, **learned_stacked, seed=seed, fleet_size=fleet,
                    intensity=intensity, dispatcher="hierarchical", n_regions=5,
                )


def _arm_label(row) -> str:
    has_path = pd.notna(row.get("scorer_path", None)) and str(row["scorer_path"]) != "nan"
    if row["dispatcher"] == "monolithic":
        if not has_path:
            return "mono + heur"
        return f"mono + {row['scorer_mode']}"
    # hierarchical
    if not has_path:
        return "hier + heur"
    return f"hier + {row['scorer_mode']}"


def main() -> None:
    if not SCORER_PATH.exists():
        raise SystemExit(
            f"missing scorer at {SCORER_PATH}; run scripts/train_scorer.py first"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = list(configs())
    print(f"Running {len(cfgs)} configs (3 seeds × 9 cells × 5 arms)")
    df = run_sweep(cfgs)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    df["arm"] = df.apply(_arm_label, axis=1)

    print(f"\nWrote results: {OUT_DIR / 'results.csv'}")

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

    # Compounding speedup: each arm vs the mono+heur absolute baseline.
    print("\n--- speedup factor vs mono + heur baseline (mean wall) ---")
    base_arm = "mono + heur"
    base = agg[agg["arm"] == base_arm].set_index(["fleet_size", "intensity"])
    for arm in agg["arm"].unique():
        if arm == base_arm:
            continue
        a = agg[agg["arm"] == arm].set_index(["fleet_size", "intensity"])
        ratio = (base["wall_mean"] / a["wall_mean"]).unstack().round(2)
        print(f"\n{arm}:")
        print(ratio.to_string())

    # Quality vs mono+heur baseline.
    print("\n--- assignment delta vs mono + heur baseline (pp) ---")
    for arm in agg["arm"].unique():
        if arm == base_arm:
            continue
        a = agg[agg["arm"] == arm].set_index(["fleet_size", "intensity"])
        delta = ((a["assignment_mean"] - base["assignment_mean"]) * 100).unstack().round(2)
        print(f"\n{arm}:")
        print(delta.to_string())

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(7 * len(INTENSITIES), 4.5),
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
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left")
    save_figure(fig, OUT_DIR / "wall_vs_fleet.png")

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(7 * len(INTENSITIES), 4.5),
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
        ax.legend(fontsize=7, loc="lower right")
    save_figure(fig, OUT_DIR / "assignment_vs_fleet.png")

    print(f"\nSaved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
