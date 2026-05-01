"""Geographic scale-up: 60 outskirts × larger fleets on real Memphis OSM.

The 30-zone scale-up + K-sensitivity sweeps already cover the headline
result. This pushes the geography (60 outskirts → 65 zones) and fleet
(up to 480) to confirm the speedup story holds at larger problem sizes
and to see whether the constrained-corner quality gap widens.

The OSM graphml is cached, so this only re-snaps centroids and recomputes
APSP for the new outskirt count — much faster than the original pull.

Usage:
    uv run python scripts/largescale_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.analysis import (  # noqa: E402
    plot_assignment_vs_fleet,
    plot_speedup_heatmap,
    plot_wall_vs_fleet,
    run_sweep,
    save_figure,
)
from hcoord.experiment import ExperimentConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "largescale"
N_OUTSKIRTS = 60
FLEET_SIZES = [120, 240, 480]
INTENSITIES = [1.0, 3.0, 5.0]


def configs():
    base = dict(
        seed=11,
        network="memphis_osm",
        n_outskirts=N_OUTSKIRTS,
        osm_outer_km=40.0,
        capacity=6,
        service_end_time=24 * 60.0,
        placement="hubs",
    )
    for fleet in FLEET_SIZES:
        for intensity in INTENSITIES:
            yield ExperimentConfig(**base, fleet_size=fleet, intensity=intensity,
                                   dispatcher="monolithic")
            for k in (3, 5):
                yield ExperimentConfig(**base, fleet_size=fleet, intensity=intensity,
                                       dispatcher="hierarchical", n_regions=k)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_sweep(list(configs()))
    df.to_csv(OUT_DIR / "results.csv", index=False)
    print(f"\nWrote results: {OUT_DIR / 'results.csv'}")
    print(df[[
        "dispatcher", "n_regions", "fleet_size", "intensity",
        "assignment_rate", "n_requests", "mean_wall_ms", "p95_wall_ms", "max_wall_ms",
    ]].to_string(index=False))

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(6 * len(INTENSITIES), 4),
                             sharey=True)
    for ax, intensity in zip(axes, INTENSITIES):
        plot_wall_vs_fleet(df, intensity=intensity, ax=ax)
    save_figure(fig, OUT_DIR / "wall_vs_fleet.png")

    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(6 * len(INTENSITIES), 4),
                             sharey=True)
    for ax, intensity in zip(axes, INTENSITIES):
        plot_assignment_vs_fleet(df, intensity=intensity, ax=ax)
    save_figure(fig, OUT_DIR / "assignment_vs_fleet.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_speedup_heatmap(df, ax=ax)
    save_figure(fig, OUT_DIR / "speedup_heatmap.png")

    print(f"Saved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
