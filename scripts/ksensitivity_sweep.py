"""K-sensitivity ablation on real Memphis OSM.

Sweeps n_regions K=1..5 (with monolithic as a separate baseline) across
multiple seeds at a fixed (fleet, intensity) operating point chosen from the
scale-up results to be informative — saturated enough that quality differences
emerge, not so saturated that everything ties at 98%.

Usage:
    uv run python scripts/ksensitivity_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.analysis import run_sweep, save_figure  # noqa: E402
from hcoord.experiment import ExperimentConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "ksensitivity"
SEEDS = [11, 23, 37, 53, 71]
K_VALUES = [1, 2, 3, 4, 5]
OPERATING_POINTS = [
    (60, 3.0),   # constrained corner — quality differences expected
    (120, 5.0),  # heavier, partially saturated
]


def configs():
    base = dict(
        network="memphis_osm",
        n_outskirts=25,
        osm_outer_km=40.0,
        capacity=6,
        service_end_time=24 * 60.0,
        placement="hubs",
    )
    for fleet, intensity in OPERATING_POINTS:
        for seed in SEEDS:
            yield ExperimentConfig(
                **base, seed=seed, fleet_size=fleet, intensity=intensity,
                dispatcher="monolithic",
            )
            for k in K_VALUES:
                yield ExperimentConfig(
                    **base, seed=seed, fleet_size=fleet, intensity=intensity,
                    dispatcher="hierarchical", n_regions=k,
                )


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds for each (fleet, intensity, dispatcher, K)."""
    df = df.copy()
    df["k_label"] = df.apply(
        lambda r: 0 if r["dispatcher"] == "monolithic" else int(r["n_regions"]),
        axis=1,
    )
    grouped = df.groupby(["fleet_size", "intensity", "dispatcher", "k_label"]).agg(
        assignment_mean=("assignment_rate", "mean"),
        assignment_std=("assignment_rate", "std"),
        wall_mean=("mean_wall_ms", "mean"),
        wall_std=("mean_wall_ms", "std"),
        n=("seed", "count"),
    ).reset_index()
    return grouped


def plot_k_sensitivity_dual(
    agg: pd.DataFrame,
    *,
    fleet_size: int,
    intensity: float,
    ax_left,
):
    """Mean wall (ms) and assignment rate vs K, with seed error bars.

    Monolithic plotted as horizontal reference lines.
    """
    sub = agg[(agg["fleet_size"] == fleet_size) & (agg["intensity"] == intensity)]
    hier = sub[sub["dispatcher"] == "hierarchical"].sort_values("k_label")
    mono = sub[sub["dispatcher"] == "monolithic"]

    ax_right = ax_left.twinx()

    ax_left.errorbar(
        hier["k_label"], hier["assignment_mean"] * 100,
        yerr=hier["assignment_std"] * 100,
        marker="o", color="tab:blue", capsize=3, label="hier assignment %",
    )
    if not mono.empty:
        m = mono.iloc[0]
        ax_left.axhline(
            m["assignment_mean"] * 100, ls="--", color="tab:blue", alpha=0.5,
            label="mono assignment %",
        )
        ax_left.fill_between(
            [min(K_VALUES), max(K_VALUES)],
            (m["assignment_mean"] - m["assignment_std"]) * 100,
            (m["assignment_mean"] + m["assignment_std"]) * 100,
            color="tab:blue", alpha=0.08,
        )

    ax_right.errorbar(
        hier["k_label"], hier["wall_mean"], yerr=hier["wall_std"],
        marker="s", color="tab:red", capsize=3, label="hier mean wall",
    )
    if not mono.empty:
        m = mono.iloc[0]
        ax_right.axhline(
            m["wall_mean"], ls="--", color="tab:red", alpha=0.5,
            label="mono mean wall",
        )

    ax_left.set_xlabel("K (regions)")
    ax_left.set_ylabel("assignment rate (%)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.set_ylim(0, 105)
    ax_left.set_xticks(K_VALUES)

    ax_right.set_ylabel("mean wall (ms)", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")

    ax_left.set_title(f"K-sensitivity (fleet={fleet_size}, intensity={intensity}, n_seeds={len(SEEDS)})")
    ax_left.grid(True, alpha=0.3)

    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_l + handles_r, labels_l + labels_r, loc="lower right", fontsize=8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = list(configs())
    print(f"Running {len(cfgs)} configs ({len(SEEDS)} seeds × "
          f"{len(OPERATING_POINTS)} ops × ({len(K_VALUES)} K + monolithic))")
    df = run_sweep(cfgs)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    print(f"\nWrote results: {OUT_DIR / 'results.csv'}")

    agg = _aggregate(df)
    agg.to_csv(OUT_DIR / "aggregated.csv", index=False)
    print("\n--- aggregated (mean over seeds) ---")
    cols = ["fleet_size", "intensity", "dispatcher", "k_label",
            "assignment_mean", "assignment_std", "wall_mean", "wall_std", "n"]
    print(agg[cols].to_string(index=False))

    fig, axes = plt.subplots(1, len(OPERATING_POINTS),
                             figsize=(7 * len(OPERATING_POINTS), 4.5))
    if len(OPERATING_POINTS) == 1:
        axes = [axes]
    for ax, (fleet, intensity) in zip(axes, OPERATING_POINTS):
        plot_k_sensitivity_dual(agg, fleet_size=fleet, intensity=intensity, ax_left=ax)
    save_figure(fig, OUT_DIR / "k_sensitivity.png")
    print(f"\nSaved figure to {OUT_DIR / 'k_sensitivity.png'}")


if __name__ == "__main__":
    main()
