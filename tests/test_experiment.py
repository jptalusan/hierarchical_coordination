import pytest

from hcoord.experiment import ExperimentConfig, run_experiment


def test_default_config_runs():
    m = run_experiment(ExperimentConfig())
    assert m.n_requests > 0
    assert 0.0 <= m.assignment_rate <= 1.0
    assert m.fleet.n_vehicles == 30


def test_monolithic_and_hierarchical_serve_same_demand():
    mono = run_experiment(ExperimentConfig(dispatcher="monolithic"))
    hier = run_experiment(ExperimentConfig(dispatcher="hierarchical", n_regions=5))
    assert mono.n_requests == hier.n_requests


def test_hierarchical_k_values():
    for k in (1, 2, 3, 4, 5):
        m = run_experiment(ExperimentConfig(dispatcher="hierarchical", n_regions=k))
        assert m.fleet.n_vehicles == 30


def test_demand_proportional_placement_works():
    m = run_experiment(
        ExperimentConfig(placement="demand_proportional", fleet_size=20)
    )
    assert m.n_requests > 0
    assert m.fleet.n_vehicles == 20


def test_intensity_scales_workload():
    light = run_experiment(ExperimentConfig(intensity=0.5))
    heavy = run_experiment(ExperimentConfig(intensity=3.0))
    assert heavy.n_requests > light.n_requests * 2


def test_invalid_dispatcher_rejected():
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(dispatcher="bogus"))


def test_invalid_n_regions_rejected():
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(dispatcher="hierarchical", n_regions=99))


def test_hierarchical_faster_than_monolithic_at_scale():
    """At larger fleet/load, hierarchical should be faster per decision."""
    cfg_mono = ExperimentConfig(dispatcher="monolithic", fleet_size=100, intensity=3.0)
    cfg_hier = ExperimentConfig(
        dispatcher="hierarchical", n_regions=5, fleet_size=100, intensity=3.0
    )
    mono = run_experiment(cfg_mono)
    hier = run_experiment(cfg_hier)
    assert hier.mean_wall_ms < mono.mean_wall_ms
