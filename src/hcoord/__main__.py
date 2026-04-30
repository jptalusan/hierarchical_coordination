"""Hydra entrypoint.

Run a single experiment:
    uv run python -m hcoord

Override fields:
    uv run python -m hcoord dispatcher=hierarchical n_regions=3 fleet_size=50

Sweep (hydra --multirun):
    uv run python -m hcoord --multirun \
        dispatcher=monolithic,hierarchical \
        fleet_size=20,40,80,160 \
        intensity=1,2,4
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hcoord.experiment import ExperimentConfig, run_experiment


def _to_typed(cfg: DictConfig) -> ExperimentConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    fields = ExperimentConfig.__dataclass_fields__
    return ExperimentConfig(**{k: v for k, v in raw.items() if k in fields})


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    typed = _to_typed(cfg)
    metrics = run_experiment(typed)

    summary = metrics.summary()
    summary["config"] = asdict(typed)

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "decisions.json").write_text(
        json.dumps([asdict(d) for d in metrics.decisions], default=str)
    )

    label = typed.dispatcher
    if typed.dispatcher == "hierarchical":
        label += f" K={typed.n_regions}"
    print(f"\n=== {label} ===")
    print(
        f"  fleet={typed.fleet_size}  capacity={typed.capacity}  "
        f"intensity={typed.intensity}  placement={typed.placement}"
    )
    print(
        f"  assigned: {metrics.n_assigned}/{metrics.n_requests} "
        f"({metrics.assignment_rate:.1%})"
    )
    print(
        f"  wall ms (mean / p95 / max): "
        f"{metrics.mean_wall_ms:.3f} / {metrics.p95_wall_ms:.3f} / {metrics.max_wall_ms:.3f}"
    )
    print(f"  total wall: {metrics.total_wall_s * 1000.0:.1f} ms")
    print(f"  active vehicles: {metrics.fleet.n_active}/{metrics.fleet.n_vehicles}")
    print(f"  output dir: {out_dir}")


if __name__ == "__main__":
    main()
