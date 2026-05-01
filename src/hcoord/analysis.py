"""Sweep runner + plotting helpers.

`run_sweep(configs)` runs a list of `ExperimentConfig` in-process (no hydra
overhead), returns a tidy `pandas.DataFrame` of per-run metrics. `plot_*`
helpers produce the figures that go in the proposal writeup.

Both pandas and matplotlib live in the `[viz]` extra; this module is
imported lazily by callers that opt into them.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from hcoord.experiment import ExperimentConfig, run_experiment

_FLAT_FIELDS = {f.name for f in dataclasses.fields(ExperimentConfig)}


def run_sweep(
    configs: Iterable[ExperimentConfig],
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """Run experiments sequentially and return a tidy DataFrame.

    One row per config. Columns: every `ExperimentConfig` field plus
    `assignment_rate`, `mean_wall_ms`, `p95_wall_ms`, `max_wall_ms`,
    `total_wall_s`, `n_assigned`, `n_requests`, `n_active`,
    `mean_route_length`, `total_deployment_min`, `runtime_s`.
    """
    rows: list[dict[str, Any]] = []
    configs = list(configs)
    for i, cfg in enumerate(configs):
        if progress:
            print(f"[{i + 1}/{len(configs)}] {_label(cfg)}", flush=True)
        t0 = time.perf_counter()
        m = run_experiment(cfg)
        elapsed = time.perf_counter() - t0
        row: dict[str, Any] = {k: v for k, v in asdict(cfg).items() if k in _FLAT_FIELDS}
        row.update(
            {
                "assignment_rate": m.assignment_rate,
                "n_assigned": m.n_assigned,
                "n_requests": m.n_requests,
                "mean_wall_ms": m.mean_wall_ms,
                "median_wall_ms": m.median_wall_ms,
                "p95_wall_ms": m.p95_wall_ms,
                "max_wall_ms": m.max_wall_ms,
                "total_wall_s": m.total_wall_s,
                "n_active": m.fleet.n_active,
                "mean_route_length": m.fleet.mean_route_length,
                "total_deployment_min": m.fleet.total_deployment_min,
                "runtime_s": elapsed,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _label(cfg: ExperimentConfig) -> str:
    bits = [
        f"net={cfg.network}",
        f"disp={cfg.dispatcher}",
    ]
    if cfg.dispatcher == "hierarchical":
        bits.append(f"K={cfg.n_regions}")
    bits += [f"F={cfg.fleet_size}", f"I={cfg.intensity}"]
    return " ".join(bits)


def dispatcher_label(row: pd.Series) -> str:
    if row["dispatcher"] == "monolithic":
        return "monolithic"
    return f"hier K={int(row['n_regions'])}"


# ---------- Plotting ----------


def plot_wall_vs_fleet(
    df: pd.DataFrame,
    *,
    intensity: float,
    ax=None,
    metric: str = "mean_wall_ms",
):
    """Mean per-decision wall-clock vs fleet size, one line per dispatcher.

    Filter to a single intensity for a clean line plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sub = df[df["intensity"] == intensity].copy()
    sub["disp_label"] = sub.apply(dispatcher_label, axis=1)
    for label, g in sub.groupby("disp_label"):
        g = g.sort_values("fleet_size")
        ax.plot(g["fleet_size"], g[metric], marker="o", label=label)

    ax.set_xlabel("fleet size")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"per-decision wall-time vs fleet (intensity={intensity})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_assignment_vs_fleet(
    df: pd.DataFrame,
    *,
    intensity: float,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sub = df[df["intensity"] == intensity].copy()
    sub["disp_label"] = sub.apply(dispatcher_label, axis=1)
    for label, g in sub.groupby("disp_label"):
        g = g.sort_values("fleet_size")
        ax.plot(g["fleet_size"], g["assignment_rate"] * 100, marker="o", label=label)

    ax.set_xlabel("fleet size")
    ax.set_ylabel("assignment rate (%)")
    ax.set_title(f"on-time arrival vs fleet (intensity={intensity})")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_speedup_heatmap(df: pd.DataFrame, *, ax=None):
    """Hierarchical-vs-monolithic speedup factor across (fleet, intensity)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    mono = (
        df[df["dispatcher"] == "monolithic"]
        .pivot(index="fleet_size", columns="intensity", values="mean_wall_ms")
    )
    hier = (
        df[df["dispatcher"] == "hierarchical"]
        .groupby(["fleet_size", "intensity"])["mean_wall_ms"]
        .mean()
        .unstack()
    )
    speedup = mono / hier

    im = ax.imshow(speedup.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(speedup.columns)))
    ax.set_xticklabels([f"{c:g}" for c in speedup.columns])
    ax.set_yticks(range(len(speedup.index)))
    ax.set_yticklabels([str(i) for i in speedup.index])
    ax.set_xlabel("intensity")
    ax.set_ylabel("fleet size")
    ax.set_title("hierarchical speedup factor (mono / hier)")
    for i, fleet in enumerate(speedup.index):
        for j, intensity in enumerate(speedup.columns):
            v = speedup.loc[fleet, intensity]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}×", ha="center", va="center", color="white")
    plt.colorbar(im, ax=ax)
    return ax


def plot_k_sensitivity(
    df: pd.DataFrame,
    *,
    fleet_size: int,
    intensity: float,
    ax=None,
):
    """Assignment rate and mean wall-time vs K, at fixed fleet/intensity."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sub = df[
        (df["dispatcher"] == "hierarchical")
        & (df["fleet_size"] == fleet_size)
        & (df["intensity"] == intensity)
    ].sort_values("n_regions")

    ax.plot(sub["n_regions"], sub["assignment_rate"] * 100, marker="o", color="tab:blue", label="assignment %")
    ax.set_xlabel("K (regions)")
    ax.set_ylabel("assignment rate (%)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_ylim(0, 105)

    ax2 = ax.twinx()
    ax2.plot(sub["n_regions"], sub["mean_wall_ms"], marker="s", color="tab:red", label="mean wall ms")
    ax2.set_ylabel("mean wall (ms)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    mono = df[
        (df["dispatcher"] == "monolithic")
        & (df["fleet_size"] == fleet_size)
        & (df["intensity"] == intensity)
    ]
    if not mono.empty:
        ax.axhline(
            mono["assignment_rate"].iloc[0] * 100,
            ls="--", color="tab:blue", alpha=0.4,
        )
        ax2.axhline(
            mono["mean_wall_ms"].iloc[0],
            ls="--", color="tab:red", alpha=0.4,
        )

    ax.set_title(f"K-sensitivity (fleet={fleet_size}, intensity={intensity})")
    ax.grid(True, alpha=0.3)
    return ax


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
