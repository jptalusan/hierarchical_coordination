"""Spot-check: is the heuristic HLP allocation beatable at all?

Before committing to a full perturbation-based label generation run, this
quick check asks: at each rebalance tick, can we find an allocation that
beats the heuristic's choice on end-of-day assignment rate?

Procedure per (config, tick):
  1. Run the heuristic to capture its target allocation at every tick.
  2. For the chosen tick, generate N random ±k perturbations on top of
     the heuristic targets (subject to: non-negative, sum to fleet_size).
  3. Re-run the day with the override applied at that single tick.
  4. Record best perturbation's downstream assignment rate vs heuristic's.

Decision rule: if <20% of (config, tick) cells have *any* perturbation
that beats the heuristic by >0.5pp, supervised step 2 (hindsight labels)
won't have enough signal. If >50%, full rollouts are worth running.
20-50% is a gray zone — examine where the wins land.

Cost: ~75 perturbed runs ≈ 40 sec.

Usage:
    uv run python scripts/hlp_spot_check.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hcoord.demand import generate_requests  # noqa: E402
from hcoord.dispatch.hierarchical import HierarchicalDispatcher  # noqa: E402
from hcoord.experiment import ExperimentConfig, _build_network  # noqa: E402
from hcoord.fleet import Vehicle  # noqa: E402
from hcoord.metrics import DecisionRecord, compute_metrics  # noqa: E402
from hcoord.placement import make_fleet  # noqa: E402
from hcoord.regions import hub_catchment_partition, merge_nearest_hubs  # noqa: E402
from hcoord.travel import TravelTimeOracle  # noqa: E402

import time  # noqa: E402

# Five configs across saturation regimes. The "control" cell (light load)
# should show ~no signal; the saturated cells should reveal whether
# allocation perturbations move the needle.
CONFIGS_TO_PROBE = [
    # (fleet, intensity, n_regions, label)
    (60, 1.0, 5, "light_load"),
    (60, 3.0, 3, "constrained_lowK"),
    (60, 5.0, 5, "constrained_highK"),
    (120, 3.0, 5, "moderate"),
    (120, 5.0, 5, "heavy_saturation"),
]
SEED = 11
N_PERTURBATIONS_PER_TICK = 5
PERTURB_MAGNITUDE = 2  # |delta| per region


class _PerturbedHLP(HierarchicalDispatcher):
    """Subclass that injects a custom allocation at a specific tick index.

    `override_at_tick` is 0-indexed over the actual rebalance ticks
    (after throttle filtering). Other ticks fall through to the heuristic.
    """

    def __init__(self, *, override_at_tick: int, override_targets: dict[int, int],
                 **kwargs):
        super().__init__(**kwargs)
        self._override_at_tick = override_at_tick
        self._override_targets = override_targets
        self._compute_tick_idx = -1

    def _compute_targets(self, now: float) -> dict[int, int]:
        self._compute_tick_idx += 1
        if self._compute_tick_idx == self._override_at_tick:
            return dict(self._override_targets)
        return super()._compute_targets(now)


def _build_world(fleet_size: int, intensity: float, n_regions: int, seed: int):
    base = ExperimentConfig(
        seed=seed,
        network="memphis_osm",
        n_outskirts=25,
        osm_outer_km=40.0,
        capacity=6,
        service_end_time=24 * 60.0,
        placement="hubs",
        fleet_size=fleet_size,
        intensity=intensity,
        dispatcher="hierarchical",
        n_regions=n_regions,
    )
    network = _build_network(base)
    oracle = TravelTimeOracle(network)
    requests = generate_requests(
        network, seed=seed, intensity=intensity,
        announce_lead_min=base.announce_lead_min,
        arrival_buffer_min=base.arrival_buffer_min,
    )
    if n_regions == len(network.hubs):
        partition = hub_catchment_partition(network, oracle)
    else:
        partition = hub_catchment_partition(
            network, oracle,
            hub_groups=merge_nearest_hubs(network, oracle, n_regions),
        )
    return base, network, oracle, requests, partition


def _fresh_fleet(base: ExperimentConfig, network):
    return make_fleet(
        network=network, fleet_size=base.fleet_size, capacity=base.capacity,
        service_end_time=base.service_end_time, placement=base.placement,
    )


def _run(disp: HierarchicalDispatcher, requests, oracle):
    decisions: list[DecisionRecord] = []
    for req in requests:
        disp.rebalance(now=req.announce_time)
        t0 = time.perf_counter()
        result = disp.assign(req, now=req.announce_time)
        wall = time.perf_counter() - t0
        decisions.append(DecisionRecord(
            request_id=req.id, announce_time=req.announce_time,
            vehicle_id=result.vehicle_id, cost=result.cost, wall_time_s=wall,
        ))
    return compute_metrics(decisions, disp.fleet, oracle)


def _capture_heuristic_targets(network, oracle, requests, partition,
                               base: ExperimentConfig) -> list[dict[int, int]]:
    """Run heuristic once with a target-capture observer, return per-tick targets."""
    fleet = _fresh_fleet(base, network)
    captured: list[dict[int, int]] = []
    disp = HierarchicalDispatcher(
        fleet=fleet, oracle=oracle, partition=partition,
        future_requests=requests,
        rebalance_interval_min=base.rebalance_interval_min,
        forecast_lookahead_min=base.forecast_lookahead_min,
        hlp_observer=lambda state, targets, now: captured.append(dict(targets)),
    )
    metrics = _run(disp, requests, oracle)
    return captured, metrics.assignment_rate


def _random_perturbation(targets: dict[int, int], n_regions: int,
                         magnitude: int, rng: np.random.Generator) -> dict[int, int]:
    """Move `magnitude` vehicles from region a to region b, both random.
    Result has same sum as input, all entries non-negative."""
    new = dict(targets)
    a, b = rng.choice(n_regions, size=2, replace=False)
    a, b = int(a), int(b)
    take = min(magnitude, new.get(a, 0))
    if take == 0:
        return new
    new[a] = new[a] - take
    new[b] = new.get(b, 0) + take
    return new


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"Probing {len(CONFIGS_TO_PROBE)} configs × N rebalance ticks × "
          f"{N_PERTURBATIONS_PER_TICK} perturbations of ±{PERTURB_MAGNITUDE}\n")

    results = []
    for fleet, intensity, K, label in CONFIGS_TO_PROBE:
        base, network, oracle, requests, partition = _build_world(fleet, intensity, K, SEED)
        heur_targets, heur_assign = _capture_heuristic_targets(
            network, oracle, requests, partition, base,
        )
        n_ticks = len(heur_targets)
        print(f"\n=== {label} (fleet={fleet}, int={intensity}, K={K}) ===")
        print(f"  heuristic: {heur_assign:.4f} ({n_ticks} rebalance ticks)")

        for tick_idx in range(n_ticks):
            heur_target = heur_targets[tick_idx]
            best_delta_pp = 0.0
            n_beats = 0
            for p in range(N_PERTURBATIONS_PER_TICK):
                perturbed = _random_perturbation(heur_target, K, PERTURB_MAGNITUDE, rng)
                if perturbed == heur_target:
                    continue
                fleet_p = _fresh_fleet(base, network)
                disp = _PerturbedHLP(
                    fleet=fleet_p, oracle=oracle, partition=partition,
                    future_requests=requests,
                    rebalance_interval_min=base.rebalance_interval_min,
                    forecast_lookahead_min=base.forecast_lookahead_min,
                    override_at_tick=tick_idx,
                    override_targets=perturbed,
                )
                m = _run(disp, requests, oracle)
                delta_pp = (m.assignment_rate - heur_assign) * 100
                if delta_pp > 0.5:
                    n_beats += 1
                if delta_pp > best_delta_pp:
                    best_delta_pp = delta_pp
            results.append({
                "config": label, "tick_idx": tick_idx,
                "heuristic_assign": heur_assign,
                "best_delta_pp": best_delta_pp,
                "n_beats_by_0.5pp": n_beats,
            })
            print(f"  tick {tick_idx}: heuristic targets={heur_target}, "
                  f"best perturbation Δ = {best_delta_pp:+.2f}pp, "
                  f"beats by >0.5pp: {n_beats}/{N_PERTURBATIONS_PER_TICK}")

    # Verdict
    print("\n=== Verdict ===")
    n_total = len(results)
    n_beatable = sum(1 for r in results if r["n_beats_by_0.5pp"] > 0)
    n_meaningful = sum(1 for r in results if r["best_delta_pp"] > 1.0)
    print(f"Beatable (any perturbation > +0.5pp): {n_beatable}/{n_total} = {n_beatable/n_total:.1%}")
    print(f"Meaningful (best perturbation > +1.0pp): {n_meaningful}/{n_total} = {n_meaningful/n_total:.1%}")
    deltas = [r["best_delta_pp"] for r in results]
    print(f"Best-delta-pp stats: mean={np.mean(deltas):.2f}, "
          f"median={np.median(deltas):.2f}, max={np.max(deltas):.2f}")

    # Decision rule
    if n_beatable / n_total < 0.20:
        print("\n→ DECISION: heuristic is rarely beatable; supervised step 2 has no signal.")
        print("  Recommend: pivot to RL (DDQN) or accept heuristic as v1.")
    elif n_beatable / n_total > 0.50:
        print("\n→ DECISION: heuristic is often beatable; full perturbation rollouts worth running.")
    else:
        print("\n→ DECISION: gray zone; examine which regime has signal before committing.")


if __name__ == "__main__":
    main()
