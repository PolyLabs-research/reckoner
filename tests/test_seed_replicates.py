"""The seed reaches the campaign's streams, and a replicate is a registered run.

Until 2026-11-11 ``cfg.seed`` reached none of them: problems, pool draws,
episodes, search noise and training batches were all numbered from the iteration
index alone, so two campaigns under different seeds would have been one
campaign run twice — and the door refused the second anyway, because the seed is
in the fingerprint. First gate first (AGENTS §5 rider a): the wire does its
central job, moving the streams, and at the registered seed it moves nothing.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from reckoner import campaign
from reckoner.campaign import (
    CAMPAIGN_FINGERPRINT,
    CampaignRefusal,
    assert_campaign_profile,
    assert_eval_profile,
    assert_resumed_seed,
)
from reckoner.config import (
    REGISTERED_SEED,
    SEED_SPACE,
    STREAM_STRIDE,
    Config,
    config_fingerprint,
    stream_seed,
    validate,
)
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.replay import ReplayRing
from reckoner.resume import RunState
from reckoner.runner import run_iteration
from reckoner.search import uniform_stub
from reckoner.valuegate import ValueHeadState

REPO = Path(__file__).resolve().parents[1]


def _seeded(seed: int) -> Config:
    return replace(Config(), seed=seed)


def _ring_digest(cfg: Config, n: int) -> str:
    rows = read_suite(REPO / "runs" / "suites" / "solve_in_3.jsonl")
    problems = [suite_problem(rows[i]) for i in sample_indices(len(rows), 6, 0)]
    ring = ReplayRing(2000, cfg)
    run_iteration(problems, uniform_stub(cfg), cfg, ring, sims=16, seed=stream_seed(cfg, n))
    digest = hashlib.sha256()
    for name in ("tokens", "visit_actions", "visit_counts", "improved_probs"):
        digest.update(getattr(ring, name)[: len(ring)].tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ the wire


def test_the_registered_seed_numbers_its_streams_exactly_as_before() -> None:
    """At seed 42 the stream seed IS the iteration index, for every iteration a
    campaign could reach. That equality is the whole inertness claim."""
    assert Config().seed == REGISTERED_SEED
    assert all(stream_seed(Config(), n) == n for n in range(5000))


def test_another_seed_moves_the_self_play_stream() -> None:
    """The central job, measured on a real iteration rather than on integers:
    the same problems under the same evaluator fill a different ring."""
    registered = _ring_digest(Config(), 0)
    assert _ring_digest(Config(), 0) == registered, "the fixture is not deterministic"
    assert _ring_digest(_seeded(43), 0) != registered


def test_no_two_seeds_or_iterations_share_a_stream() -> None:
    seeds = (0, 1, 41, 42, 43, 1000, SEED_SPACE - 1)
    streams = {stream_seed(_seeded(s), n) for s in seeds for n in range(200)}
    assert len(streams) == len(seeds) * 200
    assert min(streams) >= 0, "random.Random drops an integer seed's sign"
    assert max(streams) * 100_003 < 2**63, "the runner's derived seeds must fit int64"


def test_the_stream_numbering_refuses_what_it_cannot_number() -> None:
    with pytest.raises(ValueError, match="stream numbering"):
        stream_seed(Config(), STREAM_STRIDE)
    with pytest.raises(ValueError, match="seed must be in"):
        validate(_seeded(SEED_SPACE))
    with pytest.raises(ValueError, match="seed must be in"):
        validate(_seeded(-1))
    validate(_seeded(SEED_SPACE - 1))


def test_the_campaign_loop_draws_every_stream_from_the_wire() -> None:
    """A bare ``seed=n`` in the loop is the defect itself, so it fails by existing."""
    source = inspect.getsource(campaign.run)
    assert "stream = stream_seed(cfg, n)" in source
    assert "seed=n" not in source and source.count("seed=stream") == 3


# ------------------------------------------------------------------ the door


def test_the_door_admits_the_registered_run_unchanged() -> None:
    assert assert_campaign_profile(Config()) == CAMPAIGN_FINGERPRINT


def test_an_unregistered_seed_is_refused_by_name() -> None:
    with pytest.raises(CampaignRefusal, match="seed 43 is not a registered"):
        assert_campaign_profile(_seeded(43))


def test_a_registered_replicate_is_admitted_under_its_own_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(campaign, "REGISTERED_SEEDS", frozenset({42, 43}))
    got = assert_campaign_profile(_seeded(43))
    assert got == config_fingerprint(_seeded(43)) != CAMPAIGN_FINGERPRINT
    assert assert_eval_profile(_seeded(43)) != assert_eval_profile(Config())


def test_a_replicate_may_differ_in_its_seed_and_in_nothing_else(monkeypatch) -> None:
    monkeypatch.setattr(campaign, "REGISTERED_SEEDS", frozenset({42, 43}))
    cfg = _seeded(43)
    moved = replace(cfg, train=replace(cfg.train, lr=cfg.train.lr * 2))
    with pytest.raises(CampaignRefusal, match="not the one the campaign was"):
        assert_campaign_profile(moved)


def test_a_replicate_s_eval_profile_is_still_held_to_the_pin(monkeypatch) -> None:
    """A replicate is a campaign config, so the eval pin still applies to it. As
    first wired, a non-registered seed read as 'not the campaign' and the eval
    check was skipped for exactly the runs that needed it."""
    monkeypatch.setattr(campaign, "REGISTERED_SEEDS", frozenset({42, 43}))
    monkeypatch.setattr(campaign, "EVAL_FINGERPRINT", "0" * 64)
    with pytest.raises(CampaignRefusal, match="eval profile fingerprint"):
        assert_eval_profile(_seeded(43))


# ---------------------------------------------------------------- the resume


def test_a_run_resumes_only_under_the_seed_it_started_with() -> None:
    state = RunState(iteration=3, value_head=ValueHeadState(), seed=42)
    assert_resumed_seed(state, Config())
    with pytest.raises(CampaignRefusal, match="started under seed 42"):
        assert_resumed_seed(state, _seeded(43))
