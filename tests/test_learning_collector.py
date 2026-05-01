"""Tests for the LLP training-data collector.

Exercises:
  - feature extractor on an empty-route and non-empty-route vehicle
  - pair_delta_naive matches the hand-computed lower bound
  - InsertionCollector observing both feasible and infeasible insertions
  - cost label matches `best_insertion`'s returned cost
  - decision_id grouping (one bump per request)
  - end-to-end via run_experiment with cfg.collect_to set, run_id present
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hcoord.demand import Request, generate_requests
from hcoord.dispatch.insertion import best_insertion
from hcoord.dispatch.monolithic import MonolithicDispatcher
from hcoord.experiment import ExperimentConfig, run_experiment
from hcoord.fleet import Stop, Vehicle
from hcoord.geography import build_memphis_outskirts
from hcoord.learning.collector import InsertionCollector
from hcoord.learning.features import FEATURE_NAMES, extract_features
from hcoord.placement import make_fleet
from hcoord.travel import TravelTimeOracle


def _toy_setup():
    net = build_memphis_outskirts(seed=7, n_outskirts=10)
    oracle = TravelTimeOracle(net)
    fleet = make_fleet(
        network=net, fleet_size=4, capacity=4, service_end_time=1440.0,
        placement="hubs",
    )
    requests = generate_requests(
        net, seed=7, intensity=1.0, structural_zero_prob=0.4,
        announce_lead_min=90.0, arrival_buffer_min=15.0,
    )
    return net, oracle, fleet, requests


def test_extract_features_empty_route_uses_location_as_proxy():
    _, oracle, fleet, requests = _toy_setup()
    v = fleet[0]
    r = requests[0]
    feats = extract_features(v, r, oracle)
    assert feats["veh_route_len"] == 0
    # Empty-route fallback: route_min_tt_to_origin equals tt(loc, origin).
    assert feats["route_min_tt_to_origin"] == feats["pair_tt_loc_to_origin"]
    assert feats["route_n_pickups"] == 0
    assert feats["route_n_dropoffs"] == 0
    # All declared feature names present.
    for name in FEATURE_NAMES:
        assert name in feats, f"missing feature {name}"


def test_pair_delta_naive_matches_hand_computation():
    _, oracle, fleet, requests = _toy_setup()
    v = fleet[0]
    r = requests[0]
    feats = extract_features(v, r, oracle)
    expected = (
        oracle.travel_time(v.location, r.origin)
        + oracle.travel_time(r.origin, r.destination)
        + oracle.travel_time(r.destination, v.home)
        - oracle.travel_time(v.location, v.home)
    )
    assert feats["pair_delta_naive"] == pytest.approx(expected)


def test_extract_features_nonempty_route_summarizes_stops():
    _, oracle, fleet, requests = _toy_setup()
    v = fleet[0]
    r = requests[0]
    # Plant a pickup + dropoff manually.
    v.route = [
        Stop(kind="pickup", zone=r.origin, request_id=r.id,
             earliest=r.earliest_pickup, latest=r.latest_arrival, service_time=1.0),
        Stop(kind="dropoff", zone=r.destination, request_id=r.id,
             earliest=r.earliest_pickup, latest=r.latest_arrival, service_time=1.0),
    ]
    feats = extract_features(v, requests[1], oracle)
    assert feats["veh_route_len"] == 2
    assert feats["route_n_pickups"] == 1
    assert feats["route_n_dropoffs"] == 1
    assert feats["route_last_zone"] == requests[0].destination


def test_collector_records_feasible_and_infeasible():
    _, oracle, fleet, requests = _toy_setup()
    collector = InsertionCollector(oracle, context={"tag": "smoke"})
    v = fleet[0]
    r = requests[0]
    # Feasible call: cost recorded must match the InsertionResult's cost.
    truth = best_insertion(v, r, oracle, observer=collector)
    assert truth is not None
    # Force infeasibility: vehicle leaves so late no insertion can satisfy
    # the latest_arrival window.
    v_late = Vehicle(
        id=99, capacity=v.capacity, home=v.home, location=v.location,
        available_time=r.latest_arrival + 100.0,
        service_end_time=v.service_end_time,
    )
    best_insertion(v_late, r, oracle, observer=collector)
    assert len(collector) == 2
    rows = collector.rows
    assert rows[0]["feasible"] is True
    assert rows[0]["cost"] == pytest.approx(truth.cost)
    assert rows[0]["pickup_at"] == truth.pickup_at
    assert rows[0]["dropoff_at"] == truth.dropoff_at
    assert rows[1]["feasible"] is False
    assert rows[1]["pickup_at"] == -1
    assert rows[1]["dropoff_at"] == -1
    assert rows[0]["tag"] == "smoke"
    # Both rows belong to the same decision (same request).
    assert rows[0]["decision_id"] == rows[1]["decision_id"]


def test_collector_groups_by_decision_id():
    _, oracle, fleet, requests = _toy_setup()
    collector = InsertionCollector(oracle, context={})
    dispatcher = MonolithicDispatcher(fleet=fleet, oracle=oracle, observer=collector)
    # Process the first three requests.
    for req in requests[:3]:
        dispatcher.assign(req, now=req.announce_time)
    decision_ids = sorted({row["decision_id"] for row in collector.rows})
    assert decision_ids == [1, 2, 3]
    # Each decision has exactly len(fleet) rows (one per vehicle considered).
    counts = {}
    for row in collector.rows:
        counts.setdefault(row["decision_id"], 0)
        counts[row["decision_id"]] += 1
    assert all(c == len(fleet) for c in counts.values())


def test_run_experiment_writes_csv_when_collect_to_set(tmp_path: Path):
    out = tmp_path / "dump.csv"
    cfg = ExperimentConfig(
        seed=7, fleet_size=4, intensity=1.0,
        n_outskirts=10, network="synthetic",
        dispatcher="monolithic",
        collect_to=str(out),
    )
    run_experiment(cfg)
    assert out.exists()
    with out.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    # Context columns present.
    assert "seed" in rows[0]
    assert "fleet_size" in rows[0]
    assert "dispatcher" in rows[0]
    assert "run_id" in rows[0]
    assert rows[0]["run_id"] == "s7_n10_f4_i1_monolithic"
    # Feature + label columns present.
    assert "pair_delta_naive" in rows[0]
    assert "feasible" in rows[0]
    assert "cost" in rows[0]
    # All declared FEATURE_NAMES are columns.
    for name in FEATURE_NAMES:
        assert name in rows[0], f"missing feature column {name}"


def test_run_experiment_no_csv_when_collect_to_unset(tmp_path: Path):
    cfg = ExperimentConfig(
        seed=7, fleet_size=4, intensity=1.0,
        n_outskirts=10, network="synthetic",
        dispatcher="monolithic",
    )
    run_experiment(cfg)
    # Just confirms no exception and no spurious files (collect_to=None path).
    assert not list(tmp_path.iterdir())
