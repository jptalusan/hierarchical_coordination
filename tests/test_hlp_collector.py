"""Tests for the HLP rebalance-decision collector.

Exercises:
  - per-region feature extractor produces sensible scalars
  - extract_hlp_state pads unused region slots with zeros
  - HLPCollector records one row per rebalance tick with state + targets
  - end-to-end via run_experiment with cfg.collect_hlp_to set
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hcoord.demand import generate_requests
from hcoord.experiment import ExperimentConfig, run_experiment
from hcoord.geography import build_memphis_outskirts
from hcoord.learning.hlp_collector import HLPCollector
from hcoord.learning.hlp_features import (
    GLOBAL_FEATURES,
    HLP_STATE_FEATURE_NAMES,
    MAX_REGIONS,
    PER_REGION_FEATURES,
    extract_hlp_state,
    per_region_features,
)
from hcoord.placement import make_fleet
from hcoord.regions import hub_catchment_partition
from hcoord.travel import TravelTimeOracle


def _toy_setup():
    net = build_memphis_outskirts(seed=7, n_outskirts=15)
    oracle = TravelTimeOracle(net)
    partition = hub_catchment_partition(net, oracle)
    fleet = make_fleet(network=net, fleet_size=12, capacity=4,
                       service_end_time=1440.0, placement="hubs")
    region_of = {v.id: partition.region(v.location) for v in fleet}
    return net, oracle, partition, fleet, region_of


def test_per_region_features_empty_region_returns_zeros():
    _, oracle, partition, fleet, region_of = _toy_setup()
    # Pick a region id outside the partition's range and pretend it's empty.
    # Use region 0 but force region_of to map nothing to it.
    feats = per_region_features(
        99,
        fleet=fleet,
        region_of={v.id: 0 for v in fleet},  # everyone in region 0
        partition=partition,
        oracle=oracle,
        demand_count=3,
        now=0.0,
    )
    assert feats["n_vehicles"] == 0.0
    assert feats["n_idle_now"] == 0.0
    assert feats["n_demand_window"] == 3.0
    assert feats["supply_minus_demand"] == -3.0


def test_per_region_features_idle_count_respects_now():
    _, oracle, partition, fleet, region_of = _toy_setup()
    feats = per_region_features(
        0,
        fleet=fleet,
        region_of=region_of,
        partition=partition,
        oracle=oracle,
        demand_count=2,
        now=0.0,
    )
    # Initial fleet has empty routes; at now=0, all are "idle".
    in_region = sum(1 for v in fleet if region_of[v.id] == 0)
    assert feats["n_vehicles"] == float(in_region)
    assert feats["n_idle_now"] == float(in_region)
    assert feats["mean_route_len"] == 0.0


def test_per_region_features_idle_count_excludes_busy_vehicles():
    """Negative direction: a vehicle whose `available_time > now` must NOT
    be counted as idle. The positive-only test above would silently pass a
    bug that returned `n_vehicles` for `n_idle_now` unconditionally."""
    _, oracle, partition, fleet, region_of = _toy_setup()
    # Force the first 3 region-0 vehicles to be "busy" (available_time later
    # than the snapshot time `now=120`).
    in_region_0 = [v for v in fleet if region_of[v.id] == 0]
    assert len(in_region_0) >= 3
    for v in in_region_0[:3]:
        v.available_time = 200.0  # > now=120
    feats_busy = per_region_features(
        0,
        fleet=fleet,
        region_of=region_of,
        partition=partition,
        oracle=oracle,
        demand_count=0,
        now=120.0,
    )
    # n_vehicles stays the same (region membership unchanged).
    assert feats_busy["n_vehicles"] == float(len(in_region_0))
    # But idle count drops by exactly 3.
    assert feats_busy["n_idle_now"] == float(len(in_region_0) - 3)


def test_per_region_features_idle_count_at_epsilon_boundary():
    """The implementation tolerates `available_time <= now + 1e-9`. Vehicles
    sitting exactly at `now` (or within epsilon) should be counted idle;
    vehicles further out should not."""
    _, oracle, partition, fleet, region_of = _toy_setup()
    in_region_0 = [v for v in fleet if region_of[v.id] == 0]
    assert len(in_region_0) >= 2
    in_region_0[0].available_time = 120.0          # at boundary -> idle
    in_region_0[1].available_time = 120.0 + 1e-3   # past tolerance -> NOT idle
    feats = per_region_features(
        0,
        fleet=fleet,
        region_of=region_of,
        partition=partition,
        oracle=oracle,
        demand_count=0,
        now=120.0,
    )
    assert feats["n_idle_now"] == float(len(in_region_0) - 1)


def test_extract_hlp_state_pads_unused_region_slots():
    _, oracle, partition, fleet, region_of = _toy_setup()
    state = extract_hlp_state(
        fleet=fleet,
        region_of=region_of,
        partition=partition,
        oracle=oracle,
        demand_counts={0: 1, 1: 0},
        now=120.0,
        rebalance_interval_min=30.0,
        forecast_lookahead_min=60.0,
    )
    # All declared feature names present with finite floats.
    for name in HLP_STATE_FEATURE_NAMES:
        assert name in state
        assert isinstance(state[name], float)
    # Slots beyond the actual partition are zero-filled.
    n_regions = partition.n_regions
    for slot in range(n_regions, MAX_REGIONS):
        for feat in PER_REGION_FEATURES:
            assert state[f"r{slot}_{feat}"] == 0.0
    assert state["n_regions"] == float(n_regions)
    assert state["now_min"] == 120.0


def test_hlp_collector_records_targets_and_state():
    collector = HLPCollector(context={"tag": "smoke"})
    state = {f"r{i}_{f}": 0.0 for i in range(MAX_REGIONS) for f in PER_REGION_FEATURES}
    state.update({f: 0.0 for f in GLOBAL_FEATURES})
    targets = {0: 6, 1: 4, 2: 2}
    collector(state, targets, now=120.0)
    collector(state, targets, now=240.0)
    rows = collector.rows
    assert len(rows) == 2
    assert rows[0]["tag"] == "smoke"
    assert rows[0]["target_r0"] == 6.0
    assert rows[0]["target_r1"] == 4.0
    assert rows[0]["target_r2"] == 2.0
    # Slot 3 unused → NaN (using float("nan") sentinel).
    import math
    assert math.isnan(rows[0]["target_r3"])
    # tick_id increments.
    assert rows[1]["tick_id"] == 2


def test_collect_hlp_to_writes_csv(tmp_path: Path):
    out = tmp_path / "hlp.csv"
    cfg = ExperimentConfig(
        seed=7, fleet_size=8, intensity=1.0,
        n_outskirts=10, network="synthetic",
        dispatcher="hierarchical", n_regions=3,
        collect_hlp_to=str(out),
    )
    run_experiment(cfg)
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "tick_id" in rows[0]
    assert "now_min" in rows[0]
    assert "target_r0" in rows[0]
    assert "target_r2" in rows[0]
    assert "r0_n_vehicles" in rows[0]
    assert rows[0]["run_id"] == "s7_n10_f8_i1_k3"
    # First-tick at now ≈ first announce_time. Check it's a positive number.
    assert float(rows[0]["now_min"]) > 0.0
    # Targets per row sum (over actual regions) to fleet_size.
    total = sum(float(rows[0][f"target_r{i}"]) for i in range(3))
    assert total == 8.0


def test_collect_hlp_to_unset_does_not_write(tmp_path: Path):
    cfg = ExperimentConfig(
        seed=7, fleet_size=8, intensity=1.0,
        n_outskirts=10, network="synthetic",
        dispatcher="hierarchical", n_regions=3,
    )
    run_experiment(cfg)
    assert not list(tmp_path.iterdir())


def test_collect_hlp_to_ignored_for_monolithic(tmp_path: Path):
    """Monolithic dispatcher has no HLP — observer is wired into the
    hierarchical dispatcher only. Setting collect_hlp_to on a monolithic
    config should be a no-op (no error, no file)."""
    out = tmp_path / "hlp.csv"
    cfg = ExperimentConfig(
        seed=7, fleet_size=8, intensity=1.0,
        n_outskirts=10, network="synthetic",
        dispatcher="monolithic",
        collect_hlp_to=str(out),
    )
    run_experiment(cfg)
    # No rebalance ticks means no rows; we still write an empty CSV header.
    # Either no-file or empty-file is acceptable; assert no exception.
    if out.exists():
        with out.open() as f:
            content = f.read().strip()
        # Empty or header-only.
        assert content.count("\n") <= 0


def test_max_regions_guard():
    """If we hit MAX_REGIONS=5, K=6 with our hub set should raise. There
    are only 5 hubs in the substrate so this is a structural assertion."""
    _, oracle, partition, fleet, region_of = _toy_setup()
    assert partition.n_regions <= MAX_REGIONS


def test_collect_hlp_to_now_min_is_byte_stable_across_reruns(tmp_path: Path):
    """Step 2 will perturb the heuristic's allocation and re-run the day to
    measure downstream-assignment-rate. To match perturbed-rollout ticks
    back to the original collector's rows, the join key MUST be stable.

    The chosen key is `(run_id, now_min)`. This test pins down the
    contract: two identical-config runs produce identical `now_min`
    values at every tick. If a future change introduces nondeterminism
    (RNG leak, dict ordering, etc.), this test catches it before step 2's
    hindsight labels go silently miscoded."""
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    cfg = ExperimentConfig(
        seed=7, fleet_size=8, intensity=2.0,
        n_outskirts=10, network="synthetic",
        dispatcher="hierarchical", n_regions=3,
    )
    for path in (out_a, out_b):
        cfg.collect_hlp_to = str(path)
        run_experiment(cfg)
    with out_a.open() as f:
        rows_a = list(csv.DictReader(f))
    with out_b.open() as f:
        rows_b = list(csv.DictReader(f))
    assert len(rows_a) == len(rows_b)
    assert len(rows_a) > 0
    for a, b in zip(rows_a, rows_b):
        assert a["now_min"] == b["now_min"]
        assert a["tick_id"] == b["tick_id"]
        # All target columns also match — the heuristic allocation is
        # deterministic from the same state.
        for slot in range(3):
            assert a[f"target_r{slot}"] == b[f"target_r{slot}"]
