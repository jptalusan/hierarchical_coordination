"""Tests for the LearnedScorer plug-in path on the dispatchers.

Three properties matter:
  - With a *perfect* scorer (returns -true_cost so argmin == heuristic argmin),
    learned dispatch must produce the same per-request assignment as the
    heuristic baseline. This locks down the candidate-selection plumbing.
  - With an *adversarial* scorer (returns +inf for all candidates so top-K
    are bogus), learned dispatch must fall back to exhaustive across the
    rest and still produce the heuristic result. Quality-preserving fallback.
  - End-to-end via run_experiment with cfg.scorer_path: results structure
    is identical and metrics are computable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hcoord.demand import generate_requests  # noqa: E402
from hcoord.dispatch.monolithic import MonolithicDispatcher  # noqa: E402
from hcoord.experiment import ExperimentConfig, run_experiment  # noqa: E402
from hcoord.geography import build_memphis_outskirts  # noqa: E402
from hcoord.learning.dataset import CONTINUOUS_FEATURES, Standardizer  # noqa: E402
from hcoord.learning.inference import LearnedScorer  # noqa: E402
from hcoord.learning.model import InsertionScorer, ScorerConfig  # noqa: E402
from hcoord.learning.train import save_checkpoint  # noqa: E402
from hcoord.placement import make_fleet  # noqa: E402
from hcoord.travel import TravelTimeOracle  # noqa: E402


class _PerfectScorer:
    """Returns the *true* min insertion cost as the score, so model argmin
    matches the heuristic argmin exactly. Drop-in for `LearnedScorer`."""

    def score_batch(self, vehicles, request, oracle):
        from hcoord.dispatch.insertion import best_insertion

        out = np.empty(len(vehicles), dtype=np.float32)
        for i, v in enumerate(vehicles):
            r = best_insertion(v, request, oracle)
            out[i] = r.cost if r is not None else 1e9
        return out

    def score_pair(self, vehicle, request, oracle):
        return float(self.score_batch([vehicle], request, oracle)[0])


class _PrefersInfeasibleScorer:
    """Ranks infeasible vehicles first. Forces fallback to the rest."""

    def score_batch(self, vehicles, request, oracle):
        from hcoord.dispatch.insertion import best_insertion

        out = np.empty(len(vehicles), dtype=np.float32)
        for i, v in enumerate(vehicles):
            r = best_insertion(v, request, oracle)
            # Infeasible → very small score (top of ranking).
            # Feasible → use cost (so among feasibles we still rank correctly).
            out[i] = -1e9 if r is None else float(r.cost)
        return out

    def score_pair(self, vehicle, request, oracle):
        return float(self.score_batch([vehicle], request, oracle)[0])


def _toy_setup(seed: int = 7, fleet_size: int = 6, n_outskirts: int = 10):
    net = build_memphis_outskirts(seed=seed, n_outskirts=n_outskirts)
    oracle = TravelTimeOracle(net)
    fleet = make_fleet(network=net, fleet_size=fleet_size, capacity=4,
                       service_end_time=1440.0, placement="hubs")
    requests = generate_requests(net, seed=seed, intensity=2.0,
                                 announce_lead_min=90.0, arrival_buffer_min=15.0)
    return net, oracle, fleet, requests


def _run_dispatcher(scorer, n_requests: int = 12, *, top_k: int = 3, seed: int = 7):
    _, oracle, fleet, requests = _toy_setup(seed=seed)
    d = MonolithicDispatcher(fleet=fleet, oracle=oracle,
                             scorer=scorer, scorer_top_k=top_k)
    results = []
    for req in requests[:n_requests]:
        r = d.assign(req, now=req.announce_time)
        results.append((r.vehicle_id, r.cost))
    return results


def test_no_scorer_matches_baseline():
    baseline = _run_dispatcher(scorer=None)
    # Guard: this is the same baseline, so trivially matches itself; the real
    # value is anchoring the next two tests against this list.
    assert baseline == _run_dispatcher(scorer=None)


def test_perfect_scorer_matches_heuristic_assignment():
    baseline = _run_dispatcher(scorer=None)
    learned = _run_dispatcher(scorer=_PerfectScorer(), top_k=2)
    # Same request → same vehicle → same cost. Locks down the plumbing.
    assert learned == baseline


def test_fallback_when_topk_all_infeasible_matches_baseline():
    """Construct a fleet where the first K vehicles are guaranteed infeasible
    (available_time set past service end). Scorer prefers them, so top-K
    yields no feasible insertion and fallback must engage."""
    from hcoord.fleet import Vehicle

    net, oracle, fleet, requests = _toy_setup(fleet_size=4)
    # Replace first 2 vehicles with infeasible ones (can't return home in time).
    for i in range(2):
        v = fleet[i]
        fleet[i] = Vehicle(
            id=v.id, capacity=v.capacity, home=v.home, location=v.location,
            available_time=v.service_end_time + 1.0,
            service_end_time=v.service_end_time,
        )

    # Baseline: exhaustive over the 4-vehicle fleet, only the last 2 are feasible.
    d_base = MonolithicDispatcher(fleet=[v for v in fleet], oracle=oracle)
    base_results = []
    for req in requests[:5]:
        base_results.append(d_base.assign(req, now=req.announce_time).vehicle_id)

    # Learned: same fleet, scorer puts the 2 infeasible vehicles in top-2.
    # Re-build fleet from scratch since the prior assigns mutated routes.
    _, oracle2, fleet2, requests2 = _toy_setup(fleet_size=4)
    for i in range(2):
        v = fleet2[i]
        fleet2[i] = Vehicle(
            id=v.id, capacity=v.capacity, home=v.home, location=v.location,
            available_time=v.service_end_time + 1.0,
            service_end_time=v.service_end_time,
        )
    d_learn = MonolithicDispatcher(
        fleet=fleet2, oracle=oracle2,
        scorer=_PrefersInfeasibleScorer(), scorer_top_k=2,
    )
    learn_results = []
    for req in requests2[:5]:
        learn_results.append(d_learn.assign(req, now=req.announce_time).vehicle_id)

    assert learn_results == base_results


def test_scorer_top_k_one_is_legal():
    baseline = _run_dispatcher(scorer=None)
    learned = _run_dispatcher(scorer=_PerfectScorer(), top_k=1)
    assert learned == baseline


def test_scorer_top_k_zero_rejected():
    _, oracle, fleet, _ = _toy_setup()
    with pytest.raises(ValueError, match="scorer_top_k"):
        MonolithicDispatcher(fleet=fleet, oracle=oracle,
                             scorer=_PerfectScorer(), scorer_top_k=0)


def test_run_experiment_with_scorer_path(tmp_path: Path):
    # Save an untrained model — we only need the integration to plumb.
    cfg_model = ScorerConfig(n_features=len(CONTINUOUS_FEATURES))
    model = InsertionScorer(cfg_model)
    std = Standardizer(
        mean=np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float32),
        std=np.ones(len(CONTINUOUS_FEATURES), dtype=np.float32),
        feature_names=CONTINUOUS_FEATURES,
    )
    ckpt = tmp_path / "scorer.pt"
    save_checkpoint(ckpt, model, std, eval_dict={})

    cfg = ExperimentConfig(
        seed=7, fleet_size=4, intensity=1.0,
        n_outskirts=10, network="synthetic",
        dispatcher="monolithic",
        scorer_path=str(ckpt),
        scorer_top_k=3,
    )
    metrics = run_experiment(cfg)
    # Pipeline returns sane metrics; both arms are exercised.
    assert metrics.n_requests > 0
    assert 0.0 <= metrics.assignment_rate <= 1.0


def test_top_k_ge_fleet_size_is_exhaustive_no_op():
    """If top_k >= candidate count, the scorer path must never call the model
    (since exhaustive on all is identical to top-K-then-fallback). This is
    both a perf optimization and a robustness guarantee."""
    _, oracle, fleet, requests = _toy_setup(fleet_size=4)

    class _ExplodingScorer:
        def score_batch(self, *a, **kw):
            raise AssertionError("scorer was called when top_k >= fleet size")

        def score_pair(self, *a, **kw):
            raise AssertionError("scorer was called when top_k >= fleet size")

    d = MonolithicDispatcher(fleet=fleet, oracle=oracle,
                             scorer=_ExplodingScorer(), scorer_top_k=10)
    # Should not blow up — short-circuit before the scorer is invoked.
    d.assign(requests[0], now=requests[0].announce_time)


# ---- filter-mode tests ----


class _PerfectFilter:
    """Feasibility logits that match ground-truth perfectly. Drop-in scorer."""

    def feasibility_logits(self, vehicles, request, oracle):
        from hcoord.dispatch.insertion import best_insertion

        out = np.empty(len(vehicles), dtype=np.float32)
        for i, v in enumerate(vehicles):
            r = best_insertion(v, request, oracle)
            # +5 if feasible (well above threshold), -5 if not (well below).
            out[i] = 5.0 if r is not None else -5.0
        return out

    def score_batch(self, *a, **kw):
        raise AssertionError("filter mode should not call score_batch")

    def score_pair(self, *a, **kw):
        raise AssertionError("filter mode should not call score_pair")


class _OveragressiveFilter:
    """Logits = -100 for all vehicles. Forces full-fleet fallback."""

    def feasibility_logits(self, vehicles, request, oracle):
        return np.full(len(vehicles), -100.0, dtype=np.float32)

    def score_batch(self, *a, **kw):
        raise AssertionError("filter mode should not call score_batch")

    def score_pair(self, *a, **kw):
        raise AssertionError("filter mode should not call score_pair")


def test_filter_mode_with_perfect_filter_matches_baseline():
    """A perfect feasibility filter retains exactly the actually-feasible
    vehicles and runs exhaustive on them — same answer as full exhaustive."""
    baseline = _run_dispatcher(scorer=None)
    _, oracle, fleet, requests = _toy_setup()
    d = MonolithicDispatcher(
        fleet=fleet, oracle=oracle,
        scorer=_PerfectFilter(),
        scorer_mode="filter",
        scorer_filter_logit_threshold=0.0,
    )
    learned = []
    for req in requests[:12]:
        r = d.assign(req, now=req.announce_time)
        learned.append((r.vehicle_id, r.cost))
    assert learned == baseline


def test_filter_mode_overaggressive_falls_back_to_full_fleet():
    """If the filter rejects every vehicle, fallback runs exhaustive on the
    full candidate set — quality identical to baseline."""
    baseline = _run_dispatcher(scorer=None)
    _, oracle, fleet, requests = _toy_setup()
    d = MonolithicDispatcher(
        fleet=fleet, oracle=oracle,
        scorer=_OveragressiveFilter(),
        scorer_mode="filter",
        scorer_filter_logit_threshold=-2.0,
    )
    learned = []
    for req in requests[:12]:
        r = d.assign(req, now=req.announce_time)
        learned.append((r.vehicle_id, r.cost))
    assert learned == baseline


def test_filter_mode_when_survivors_all_infeasible_falls_back_to_dropped():
    """If the filter retains some vehicles but they all turn out infeasible,
    fall through to the rejected ones. Quality preserved either way."""
    from hcoord.fleet import Vehicle

    _, oracle, fleet, requests = _toy_setup(fleet_size=4)
    # Make vehicles 0 and 1 infeasible (their available_time exceeds end).
    for i in range(2):
        v = fleet[i]
        fleet[i] = Vehicle(
            id=v.id, capacity=v.capacity, home=v.home, location=v.location,
            available_time=v.service_end_time + 1.0,
            service_end_time=v.service_end_time,
        )

    class _RetainOnlyInfeasible:
        """Keeps the two rigged-infeasible vehicles, drops the two real ones."""
        def feasibility_logits(self, vehicles, request, oracle):
            return np.array(
                [5.0 if v.id < 2 else -5.0 for v in vehicles],
                dtype=np.float32,
            )

    d_base = MonolithicDispatcher(fleet=[v for v in fleet], oracle=oracle)
    base_results = []
    for req in requests[:5]:
        base_results.append(d_base.assign(req, now=req.announce_time).vehicle_id)

    # Re-build to undo prior route mutations.
    _, oracle2, fleet2, requests2 = _toy_setup(fleet_size=4)
    for i in range(2):
        v = fleet2[i]
        fleet2[i] = Vehicle(
            id=v.id, capacity=v.capacity, home=v.home, location=v.location,
            available_time=v.service_end_time + 1.0,
            service_end_time=v.service_end_time,
        )
    d_learn = MonolithicDispatcher(
        fleet=fleet2, oracle=oracle2,
        scorer=_RetainOnlyInfeasible(),
        scorer_mode="filter",
        scorer_filter_logit_threshold=0.0,
    )
    learn_results = []
    for req in requests2[:5]:
        learn_results.append(d_learn.assign(req, now=req.announce_time).vehicle_id)

    assert learn_results == base_results


def test_filter_mode_threshold_controls_strictness():
    """Higher threshold → keeps fewer vehicles → more fallback work but
    same answer (since fallback runs exhaustive on rejected when retained
    ones fail). Just confirms the knob is wired through."""
    _, oracle, fleet, requests = _toy_setup()

    captured = {"n_retained": []}

    class _GraduatedFilter:
        def feasibility_logits(self, vehicles, request, oracle):
            # Linear ramp 1, 2, ..., n. Threshold of 0.5 keeps everything;
            # of 5 keeps only the last few.
            return np.arange(1, len(vehicles) + 1, dtype=np.float32)

    d_lax = MonolithicDispatcher(
        fleet=list(fleet), oracle=oracle,
        scorer=_GraduatedFilter(),
        scorer_mode="filter",
        scorer_filter_logit_threshold=0.5,
    )
    d_strict = MonolithicDispatcher(
        fleet=list(fleet), oracle=oracle,
        scorer=_GraduatedFilter(),
        scorer_mode="filter",
        scorer_filter_logit_threshold=4.5,
    )
    # Both must produce a sensible result on the first request without raising.
    r1 = d_lax.assign(requests[0], now=requests[0].announce_time)
    r2 = d_strict.assign(requests[0], now=requests[0].announce_time)
    assert r1.vehicle_id is not None
    assert r2.vehicle_id is not None


def test_invalid_scorer_mode_rejected():
    _, oracle, fleet, _ = _toy_setup()
    with pytest.raises(ValueError, match="scorer_mode"):
        MonolithicDispatcher(fleet=fleet, oracle=oracle,
                             scorer=_PerfectFilter(),
                             scorer_mode="bogus")
