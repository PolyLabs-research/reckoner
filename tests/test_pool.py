"""The snapshot league, on both polarities for each of its three load-bearing rules.

Each rule has a failure this project has already met one layer down, so each gets
its accepting case beside its rejecting one.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from reckoner.absence import Absent
from reckoner.config import Config, with_m1_search
from reckoner.dataset import read_suite, suite_problem
from reckoner.model import Reckoner, save_checkpoint
from reckoner.pool import CheckpointPool, PoolError
from reckoner.valuegate import ValueHeadState

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
#: The declaration these tests are about is the win/draw/loss head's, and the
#: evaluators below report no steps-to-go: both belong to the search that reads 'wdl'.
CFG = with_m1_search(Config())

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated"
)


def a_snapshot(tmp_path: Path, name: str, *, step: int, live: bool = False) -> Path:
    torch.manual_seed(step)
    path = tmp_path / f"{name}.pt"
    save_checkpoint(
        path,
        Reckoner(CFG),
        CFG,
        step,
        value_head=ValueHeadState(live=live, switched_at_iteration=0 if live else None).as_dict(),
    )
    return path


def recording_factory(seen: list[float]):
    """An evaluator factory that records the value scale it was handed."""

    def factory(model, value_scale: float):
        seen.append(value_scale)
        width = 7 * CFG.model.max_sites

        def evaluate(leaves):
            return [(np.zeros(width, dtype=np.float32), value_scale * 0.5) for _ in leaves]

        return evaluate

    return factory


# ---------------------------------------------------------------------------
# Rule 1 — version refusal, counted, both polarities
# ---------------------------------------------------------------------------


def test_a_matching_snapshot_loads(tmp_path: Path) -> None:
    pool = CheckpointPool(CFG)
    assert pool.add(a_snapshot(tmp_path, "ok", step=100)) is not None
    assert len(pool) == 1
    assert pool.stats.refusals == 0


def test_a_version_mismatch_is_refused_and_counted(tmp_path: Path) -> None:
    """Never a silently smaller pool — the chunk-6 registration, discharged."""
    path = a_snapshot(tmp_path, "stale", step=50)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["meta"]["ruleset_version"] = 999
    torch.save(blob, path)

    pool = CheckpointPool(CFG)
    with pytest.raises(PoolError, match="refusal is counted"):
        pool.add(path)
    assert pool.stats.refusals == 1
    assert pool.stats.refused_paths == [str(path)]
    assert len(pool) == 0


def test_a_vocab_mismatch_is_refused_too(tmp_path: Path) -> None:
    path = a_snapshot(tmp_path, "othervocab", step=50)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["meta"]["vocab_version"] = 999
    torch.save(blob, path)
    pool = CheckpointPool(CFG)
    assert pool.try_add(path) is False
    assert pool.stats.refusals == 1


# ---------------------------------------------------------------------------
# Rule 2 — a snapshot solves under ITS OWN declaration
# ---------------------------------------------------------------------------


@needs_suites
def test_a_pre_switch_snapshot_plays_value_silent_inside_a_post_switch_run(
    tmp_path: Path,
) -> None:
    """The running run has switched; this member has not. It plays silent.

    Re-solving a pre-switch snapshot with post-switch wiring produces a par that
    snapshot never achieved — a wrong number under a provenance tag true in its
    own terms, which is F-02 one layer up.
    """
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "old", step=10, live=False))
    assert member is not None
    assert member.value_head.live is False

    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_1.jsonl")[0])
    pool.par_for(problem, member, recording_factory(seen), random.Random(0), sims=8, m=5)
    assert seen and all(scale == 0.0 for scale in seen), (
        "a pre-switch member was handed a live value scale — it solved under the "
        "run's wiring, not its own"
    )


@needs_suites
def test_a_post_switch_snapshot_plays_value_live(tmp_path: Path) -> None:
    """The other polarity — otherwise "honours its own declaration" would just
    mean "always silent"."""
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "new", step=90, live=True))
    assert member is not None
    assert member.value_head.live is True

    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_1.jsonl")[0])
    pool.par_for(problem, member, recording_factory(seen), random.Random(0), sims=8, m=5)
    assert seen and all(scale == 1.0 for scale in seen)


def test_the_declaration_rides_in_the_checkpoint(tmp_path: Path) -> None:
    """It is part of the snapshot's identity, so it must be IN the snapshot."""
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "live", step=7, live=True))
    assert member is not None
    assert member.meta["value_head"]["live"] is True
    assert member.meta["value_head"]["switched_at_iteration"] == 0


# ---------------------------------------------------------------------------
# Rule 3 — unavailability is a COUNTED fallback, never a silent substitution
# ---------------------------------------------------------------------------


@needs_suites
def test_an_unsolved_problem_falls_back_with_the_provenance_flipped(
    tmp_path: Path,
) -> None:
    """The fallback is allowed. The silence is not.

    A par tagged `pool` that no pool member produced is the defect this module
    exists to avoid.
    """
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "weak", step=1))
    assert member is not None

    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_6.jsonl")[0])
    result = pool.par_for(
        problem, member, recording_factory(seen), random.Random(0), sims=2, m=1, budget=1
    )
    assert result.fell_back is True
    assert result.par_source == problem.par_source != "pool"
    assert isinstance(result.par_asof, Absent), "a fallback par_asof is Absent, never None"
    assert result.par_asof.kind == "inapplicable"
    assert "timeless" in result.par_asof.reason
    assert result.reason == "snapshot_capped"
    assert result.par == problem.par
    assert pool.stats.pool_par_unavailable_capped == 1
    assert pool.stats.pool_par_unavailable_empty == 0
    assert pool.stats.pool_par_solved == 0


@needs_suites
def test_a_solved_problem_is_tagged_pool_and_dated(tmp_path: Path) -> None:
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "able", step=42))
    assert member is not None
    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_1.jsonl")[0])
    result = pool.par_for(problem, member, recording_factory(seen), random.Random(0), sims=16, m=16)
    assert result.fell_back is False
    assert result.par_source == "pool"
    assert result.par_asof == 42, "pool par is as-of the snapshot that produced it"
    assert result.par >= 1
    assert pool.stats.pool_par_solved == 1


def test_a_problem_with_no_own_par_has_no_honest_fallback(tmp_path: Path) -> None:
    """Absence carries a reason: there is no label for this pair, so it raises."""
    from reckoner.episode import Problem
    from reckoner.expr import num
    from reckoner.vocab import GOAL_EVALUATE

    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "x", step=1))
    assert member is not None
    unlabelled = Problem(goal=GOAL_EVALUATE, expr=num(5), par=None, par_source="unverified")
    with pytest.raises(PoolError, match="no honest label"):
        pool.par_for(
            unlabelled, member, recording_factory([]), random.Random(0), sims=2, m=1, budget=1
        )


# ---------------------------------------------------------------------------
# Membership and sampling are DECLARED
# ---------------------------------------------------------------------------


def test_the_pool_is_bounded_and_keeps_the_most_recent(tmp_path: Path) -> None:
    cfg = Config()
    cfg.league.pool_size = 3
    pool = CheckpointPool(cfg)
    for step in (10, 20, 30, 40, 50):
        pool.add(a_snapshot(tmp_path, f"s{step}", step=step))
    assert pool.composition()["steps"] == [30, 40, 50]


def test_composition_is_loggable_and_names_the_live_members(tmp_path: Path) -> None:
    pool = CheckpointPool(CFG)
    pool.add(a_snapshot(tmp_path, "a", step=1, live=False))
    pool.add(a_snapshot(tmp_path, "b", step=2, live=True))
    assert pool.composition() == {
        "size": 2,
        "steps": [1, 2],
        "order": [1, 2],
        "value_head_live": [2],
    }


def test_composition_order_is_the_list_sample_sees_not_the_sorted_view(tmp_path: Path) -> None:
    """F-23's subtle half, asserted. `sample` is ``rng.choice`` over `members`, so
    membership and SEQUENCE are both load-bearing: the same pool assembled in a
    different order draws a different snapshot. A composition column carrying only
    the sorted view could not see the defect it was added for."""
    pool = CheckpointPool(CFG)
    for step in (50, 10, 30):  # anchor-like member first, as the driver enrols
        pool.add(a_snapshot(tmp_path, f"m{step}", step=step))
    composition = pool.composition()
    assert composition["order"] == [50, 10, 30], "order must be insertion order"
    assert composition["steps"] == [10, 30, 50], "steps must remain the sorted identity"
    assert composition["order"] != composition["steps"], "the two views must be distinguishable"


def test_sampling_an_empty_pool_returns_nothing_rather_than_raising() -> None:
    """An empty pool is a state, not an error — par_from_pool_frac simply cannot
    be honoured yet, and the caller falls back with that recorded."""
    assert CheckpointPool(CFG).sample(random.Random(0)) is None


def test_an_unknown_sampling_policy_is_refused_by_validate() -> None:
    from reckoner.config import validate

    cfg = Config()
    cfg.league.pool_sample = "recency"
    with pytest.raises(ValueError, match="one-lever round"):
        validate(cfg)


def test_the_solve_budget_is_accounted(tmp_path: Path) -> None:
    """Pool par costs wall clock, and the cost shows up where costs show up."""
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "t", step=1))
    assert member is not None
    from reckoner.episode import Problem
    from reckoner.expr import num
    from reckoner.vocab import GOAL_EVALUATE

    problem = Problem(goal=GOAL_EVALUATE, expr=num(5), par=1, par_source="bfs")
    pool.par_for(problem, member, recording_factory([]), random.Random(0), sims=2, m=1, budget=1)
    assert pool.stats.as_dict()["seconds_solving"] >= 0.0
    assert "pool_par_unavailable" in pool.stats.as_dict()


# ---------------------------------------------------------------------------
# The three confirms: hatch scope, absence at every layer, distinct causes
# ---------------------------------------------------------------------------


def test_the_league_loader_has_no_escape_hatch_to_pass() -> None:
    """ "Must not pass False" is a convention until it is impossible.

    The league path calls `load_league_checkpoint`, which has no
    `strict_versions` parameter at all — so the hatch's scope is enforced by the
    signature rather than by remembering.
    """
    import inspect

    from reckoner.model import load_checkpoint, load_league_checkpoint

    assert "strict_versions" in inspect.signature(load_checkpoint).parameters
    assert "strict_versions" not in inspect.signature(load_league_checkpoint).parameters
    # The meaningful check is that the league module never *passes* it. Its
    # docstring names it deliberately, to say the hatch is unreachable from here.
    source = (REPO / "src" / "reckoner" / "pool.py").read_text()
    assert "strict_versions=" not in source, (
        "the league module passes the escape hatch — it must not be able to"
    )


def test_an_empty_pool_is_a_different_disease_from_a_capped_snapshot(tmp_path: Path) -> None:
    """Different causes, different counters — the capped/stuck lesson, one layer up.

    Pooling them would hide which is happening, and the fixes differ: wait for
    snapshots versus raise the solve budget.
    """
    from reckoner.episode import Problem
    from reckoner.expr import num
    from reckoner.vocab import GOAL_EVALUATE

    problem = Problem(goal=GOAL_EVALUATE, expr=num(5), par=1, par_source="bfs")

    empty = CheckpointPool(CFG)
    result = empty.par_for_episode(problem, recording_factory([]), random.Random(0))
    assert result.reason == "pool_empty"
    assert empty.stats.pool_par_unavailable_empty == 1
    assert empty.stats.pool_par_unavailable_capped == 0

    stocked = CheckpointPool(CFG)
    stocked.add(a_snapshot(tmp_path, "m", step=1))
    stocked.par_for_episode(problem, recording_factory([]), random.Random(0), sims=2, m=1, budget=1)
    assert stocked.stats.pool_par_unavailable_empty == 0
    assert stocked.stats.pool_par_unavailable_capped == 1


def test_an_absent_par_asof_refuses_to_be_read_as_a_date(tmp_path: Path) -> None:
    """A None read as 0 would date a fallback par to the beginning of time."""
    from reckoner.absence import AbsenceError
    from reckoner.episode import Problem
    from reckoner.expr import num
    from reckoner.vocab import GOAL_EVALUATE

    pool = CheckpointPool(CFG)
    problem = Problem(goal=GOAL_EVALUATE, expr=num(5), par=1, par_source="bfs")
    result = pool.par_for_episode(problem, recording_factory([]), random.Random(0))
    with pytest.raises(AbsenceError, match="test for Absent"):
        bool(result.par_asof)


# ---------------------------------------------------------------------------
# Enrollment — a pool the loop never feeds is a league in name only
# ---------------------------------------------------------------------------


def test_enrollment_grows_the_pool_and_evicts_the_oldest(tmp_path: Path) -> None:
    """Par escalation is the amendment's core mechanism, and it is enrollment.

    A pool that stays at size 1 for a whole run is a fixed opponent wearing the
    word "league", and `par_from_pool_frac` would price a mechanism that never
    engages.
    """
    cfg = Config()
    cfg.league.pool_size = 3
    pool = CheckpointPool(cfg)
    torch.manual_seed(0)
    model = Reckoner(cfg)

    sizes, compositions = [], []
    for step in (100, 200, 300, 400, 500):
        pool.enroll(model, step, ValueHeadState(), tmp_path / f"snap-{step}.pt")
        sizes.append(len(pool))
        compositions.append(pool.composition()["steps"])

    assert sizes == [1, 2, 3, 3, 3], "the pool must grow, then hold at pool_size"
    assert compositions[-1] == [300, 400, 500], "oldest evicted first"


def test_an_enrolled_snapshot_carries_the_declaration_it_was_played_under(
    tmp_path: Path,
) -> None:
    """Rule 2 depends on this: the snapshot must remember its own wiring."""
    pool = CheckpointPool(CFG)
    torch.manual_seed(0)
    live = ValueHeadState(live=True, switched_at_iteration=4)
    member = pool.enroll(Reckoner(CFG), 900, live, tmp_path / "live.pt")
    assert member is not None
    assert member.value_head.live is True
    assert member.value_head.switched_at_iteration == 4
    assert pool.composition()["value_head_live"] == [900]
