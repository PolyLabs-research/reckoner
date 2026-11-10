"""The episode runner: splits, the descent identity, the H split, and parity.

The descent identity is chunk 7's forward obligation landing. It is asserted, not
watched — and the one legitimate exception to `evals == nodes` is *named* rather
than absorbed into an inequality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config, config_fingerprint
from reckoner.dataset import git_sha, read_suite, suite_problem
from reckoner.logschema import SCHEMA_ERA, validate_row
from reckoner.replay import ReplayRing
from reckoner.rules import RULESET_VERSION
from reckoner.runner import IterationStats, entropy, iteration_row, run_iteration
from reckoner.search import uniform_stub
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_2.jsonl").exists(), reason="suites not generated"
)


def problems(depth: int, n: int) -> list:
    return [suite_problem(r) for r in read_suite(SUITES / f"solve_in_{depth}.jsonl")[:n]]


# ---------------------------------------------------------------------------
# The descent identity — chunk 7's forward obligation
# ---------------------------------------------------------------------------


@needs_suites
def test_the_descent_identity_holds_over_a_real_iteration() -> None:
    """`nodes - evaluations == terminals`, asserted rather than watched."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.nodes - stats.evaluations == stats.terminals
    assert stats.terminals > 0, "an iteration with no terminals would not exercise the identity"


def test_the_identity_is_checked_and_can_fail() -> None:
    """The other polarity. An invariant nobody has seen refuse is a comment."""
    stats = IterationStats(nodes=10, evaluations=10, terminal_solved=3)
    with pytest.raises(AssertionError, match="descent identity violated"):
        stats.check_descent_identity()


def test_evals_equals_nodes_only_when_no_terminal_is_reached() -> None:
    """The named exception, pinned so it cannot quietly become an inequality.

    Chunk 7's `evals == nodes` is the zero-terminal special case of the identity.
    A terminal leaf creates a node without an evaluation, which is why the
    equality does not generalise to real episodes.
    """
    zero_terminals = IterationStats(nodes=49, evaluations=49)
    zero_terminals.check_descent_identity()
    assert zero_terminals.nodes == zero_terminals.evaluations

    with_terminals = IterationStats(nodes=52, evaluations=49, terminal_solved=3)
    with_terminals.check_descent_identity()
    assert with_terminals.nodes != with_terminals.evaluations


# ---------------------------------------------------------------------------
# Splits sum — the schema invariant, met by the producer
# ---------------------------------------------------------------------------


@needs_suites
def test_outcomes_sum_to_episodes() -> None:
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.episodes_solved + stats.episodes_capped + stats.episodes_stuck == stats.episodes


@needs_suites
def test_the_histogram_accounts_for_exactly_the_solved_episodes() -> None:
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert sum(stats.steps_minus_par.values()) == stats.episodes_solved


@needs_suites
def test_the_row_it_produces_validates_against_the_schema() -> None:
    """The producer and the schema must agree without a translation step."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    row = iteration_row(
        stats,
        iteration=0,
        run_name="test",
        git_sha=git_sha(REPO),
        config_fingerprint=config_fingerprint(CFG),
        cfg=CFG,
        ruleset_version=RULESET_VERSION,
        vocab_version=VOCAB_VERSION,
        schema_era=SCHEMA_ERA,
        evaluator_checkpoint_sha256="d" * 64,
        pool_composition={"size": 0, "steps": [], "order": [], "value_head_live": []},
        absent={
            "pool_par_fraction": "no pool in this fixture",
            "ladder_pass": "not a ladder iteration",
        },
    )
    assert validate_row(row) == [], "a clean iteration must raise no alarms"


@needs_suites
def test_exact_par_cannot_be_beaten_in_the_produced_row() -> None:
    """The tripwire's fourth sighting: the producer must not even construct it."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.z_by_par_source["bfs"]["+1"] == 0


# ---------------------------------------------------------------------------
# The entropy split
# ---------------------------------------------------------------------------


def test_entropy_is_zero_on_a_degenerate_distribution() -> None:
    assert entropy(np.array([1.0])) == pytest.approx(0.0)
    assert entropy(np.array([])) == 0.0


def test_entropy_is_maximal_on_a_uniform_distribution() -> None:
    assert entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))


@needs_suites
def test_both_entropy_populations_are_measured() -> None:
    """Start and reached must both be populated, or the split is a single number.

    Depth-2 problems take at least two steps, so every episode contributes to
    both populations — which is what makes the split meaningful here.
    """
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.h_prior_start and stats.h_prior_reached
    assert stats.h_target_start and stats.h_target_reached
    values = stats.entropies()
    assert values["entropy_prior_step1_start"] != values["entropy_prior_step1_reached"], (
        "start and reached entropies are identical — the split is not splitting"
    )


@needs_suites
def test_prior_and_visit_entropy_are_different_measurements() -> None:
    """Prior entropy alone cannot tell a confident policy from a collapsed one.

    Rewritten for F-42. This asserted H(prior) != H(target) while the target was
    the visit share. Under π′ the two are EQUAL wherever search found nothing —
    correctly — and against the flat-prior stub they then differ only by float32
    rounding, which is not a measurement. The visit share is still a different
    quantity from the prior; π′'s separation is tested below, on KL.
    """
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    values = stats.entropies()
    assert values["entropy_prior_step1_start"] != values["entropy_visits_step1_start"]


# ---------------------------------------------------------------------------
# Batching is across searches, and it changes nothing
# ---------------------------------------------------------------------------


@needs_suites
def test_batch_size_does_not_change_the_iteration() -> None:
    """Pooling across episodes has no coupling, so parity is an equality.

    Not a tolerance: if this ever needs one, batching has leaked inside a tree.
    """
    kwargs = {"sims": 16, "m": 5, "seed": 3}
    small = run_iteration(problems(2, 8), uniform_stub(CFG), CFG, batch_size=1, **kwargs)
    large = run_iteration(problems(2, 8), uniform_stub(CFG), CFG, batch_size=64, **kwargs)
    assert small.episodes_solved == large.episodes_solved
    assert small.nodes == large.nodes and small.evaluations == large.evaluations
    assert small.steps_minus_par == large.steps_minus_par
    assert small.z_by_par_source == large.z_by_par_source


@needs_suites
def test_seeding_is_a_per_episode_fan_out_not_one_shared_stream() -> None:
    """F-11's precedent generalised to the runner.

    Every episode drawing from one stream is not a sample of independent
    episodes. Different seeds must move the iteration; the same seed must
    reproduce it exactly.
    """
    kwargs = {"sims": 16, "m": 5}
    a = run_iteration(problems(2, 10), uniform_stub(CFG), CFG, seed=1, **kwargs)
    b = run_iteration(problems(2, 10), uniform_stub(CFG), CFG, seed=1, **kwargs)
    c = run_iteration(problems(2, 10), uniform_stub(CFG), CFG, seed=2, **kwargs)

    # Same seed reproduces exactly.
    assert a.nodes == b.nodes
    assert a.evaluations == b.evaluations
    assert a.z_by_par_source == b.z_by_par_source
    assert a.h_visits_start == b.h_visits_start

    # A different seed must move something. The quantities it can move are
    # narrower than they look: against `uniform_stub` the priors are FLAT, so
    # H_prior is a function of the legal-action count alone and is seed-blind by
    # construction, node counts are budget-driven, and π′ equals the flat prior
    # wherever the search holds no proof (F-42). What the Gumbel draw actually
    # moves is which actions get visited — so the discriminating quantities are
    # the evaluation count and the visit-share entropy.
    assert (a.evaluations, a.h_visits_start) != (c.evaluations, c.h_visits_start), (
        "a different seed changed neither the evaluation count nor the visit "
        "distribution — the fan-out is not fanning out"
    )


# ---------------------------------------------------------------------------
# The ring receives the episode
# ---------------------------------------------------------------------------


@needs_suites
def test_every_step_of_every_episode_reaches_the_ring() -> None:
    ring = ReplayRing(2000, CFG)
    stats = run_iteration(problems(2, 8), uniform_stub(CFG), CFG, ring, sims=16, m=5, seed=1)
    assert len(ring) > 0
    assert len(ring) >= stats.episodes_solved, "at least one row per solved episode"
    for slot in range(len(ring)):
        record = ring.get(slot)
        assert record["z"] in (-1, 0, 1)
        assert record["par_source"] == "bfs"


@needs_suites
def test_the_ring_learns_z_only_when_the_episode_settles() -> None:
    """z is an episode-level outcome; every step of one episode carries the same z."""
    ring = ReplayRing(2000, CFG)
    run_iteration(problems(2, 4), uniform_stub(CFG), CFG, ring, sims=16, m=5, seed=1)
    zs = {ring.get(slot)["z"] for slot in range(len(ring))}
    assert zs <= {-1, 0, 1} and zs


# ---------------------------------------------------------------------------
# The two dormant levers — ported, inert, and tested on BOTH settings
# ---------------------------------------------------------------------------


@needs_suites
def test_resign_is_inert_by_default() -> None:
    """Default off, per v1.1: calibration of k is deferred to campaign evidence."""
    assert CFG.par.concede_enabled is False
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.episodes_conceded == 0


@needs_suites
def test_resign_fires_when_enabled_and_is_a_distinct_outcome() -> None:
    """The other polarity — "implemented" must mean something happens when on.

    Conceding is a DECISION where capping is an exhaustion; pooling them would
    hide which the policy is doing, so it gets its own counter and the splits
    still sum.
    """
    cfg = Config()
    cfg.par.concede_enabled = True
    cfg.par.concede_k = 0  # resign the moment the line exceeds par
    stats = run_iteration(problems(2, 12), uniform_stub(cfg), cfg, sims=16, m=5, seed=1)
    assert stats.episodes_conceded > 0, "resign enabled but nothing ever conceded"
    assert (
        stats.episodes_solved
        + stats.episodes_capped
        + stats.episodes_stuck
        + stats.episodes_conceded
        == stats.episodes
    )


@needs_suites
def test_a_conceded_episode_is_not_counted_as_capped() -> None:
    cfg = Config()
    cfg.par.concede_enabled = True
    cfg.par.concede_k = 0
    stats = run_iteration(problems(2, 12), uniform_stub(cfg), cfg, sims=16, m=5, seed=1)
    assert stats.episodes_capped == 0, "conceding must not be filed as exhaustion"


# ------------------------------------------- absence is the caller's claim (F-29)


def _row(**kwargs) -> dict:
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    return iteration_row(
        stats,
        iteration=0,
        run_name="test",
        git_sha=git_sha(REPO),
        config_fingerprint=config_fingerprint(CFG),
        cfg=CFG,
        ruleset_version=RULESET_VERSION,
        vocab_version=VOCAB_VERSION,
        schema_era=SCHEMA_ERA,
        evaluator_checkpoint_sha256="d" * 64,
        pool_composition={"size": 0, "steps": [], "order": [], "value_head_live": []},
        **kwargs,
    )


@needs_suites
def test_an_empty_absent_means_nothing_is_absent() -> None:
    """F-29, the polarity that cost a cadence unit.

    `iteration_row` defaulted `absent` to a fabricated mapping naming
    ``pool_par_fraction`` and ``ladder_pass``, via ``absent or {...}`` — which
    cannot distinguish "unspecified" from "specified as empty". The campaign
    reached the first iteration where nothing WAS absent (a cadence iteration
    with a live pool: the ladder fires, pool par is drawn) and the row was
    refused for naming two columns it had just written.

    An empty mapping means every column has a value.
    """
    row = _row(absent={})
    assert row["absent"] == {}, "an empty absent must not be replaced by an invented one"

    row["pool_par_fraction"] = 0.2
    row["ladder_pass"] = 0
    assert validate_row(row) == [], "a row with nothing absent must validate"


@needs_suites
def test_a_stated_absence_is_carried_through_verbatim() -> None:
    """The other polarity: a caller that DOES declare an absence keeps its own
    reason, rather than the library's idea of one."""
    reason = "league.par_from_pool_frac is 0, or no snapshot has been taken yet"
    row = _row(
        absent={
            "pool_par_fraction": reason,
            "ladder_pass": "not a ladder iteration",
        }
    )
    assert row["absent"]["pool_par_fraction"] == reason
    assert validate_row(row) == []


@needs_suites
def test_omitting_absent_entirely_does_not_invent_absences() -> None:
    """A caller who passes nothing gets nothing — not two fabricated claims about
    a run the library has never seen."""
    assert _row()["absent"] == {}


# ---------------------------------------------------------------------------
# What the ring stores for a root's visits — F-41, pinned absolutely
# ---------------------------------------------------------------------------


def test_ring_visits_stores_action_indices_not_root_slots() -> None:
    """**Pinned reference vector**, because a round trip cannot see this bug.

    ``SearchResult.visits`` is indexed by root slot (position in ``actions``); the
    ring stores dense action indices (``rule * max_sites + site``). The writer
    and reader agreed with each other's *types* and disagreed about *coordinates*
    for all of M1 (F-41). Absolute expected values, computed by hand:

    =====  ==========  ======  ================================
    slot   action      visits  dense index (max_sites = 192)
    =====  ==========  ======  ================================
    0      (1, 2)      3       1 * 192 + 2 = 194
    1      (3, 0)      10      3 * 192 + 0 = 576
    2      (0, 5)      0       0 * 192 + 5 = 5
    =====  ==========  ======  ================================

    Ordered by visits, descending: slot 1, slot 0, slot 2. The slot reading would
    store ``[1, 0, 2]`` — cells 1, 0 and 2, which are ``eval_add`` at sites 1, 0
    and 2, none of them the action visited.
    """
    from reckoner.runner import ring_visits
    from reckoner.search import SearchResult

    result = SearchResult(
        visits=np.array([3, 10, 0], dtype=np.int32),
        values=np.zeros(3, dtype=np.float32),
        actions=[(1, 2), (3, 0), (0, 5)],
        chosen=(3, 0),
        root_value=0.0,
        improved=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    actions, counts = ring_visits(result, width=16, max_sites=192)
    assert actions.tolist() == [576, 194, 5]
    assert counts.tolist() == [10, 3, 0]
    assert actions.dtype == np.int32 and counts.dtype == np.int32


def test_ring_visits_keeps_only_the_top_width() -> None:
    """The ring's slot count bounds what is stored; the top visits survive."""
    from reckoner.runner import ring_visits
    from reckoner.search import SearchResult

    result = SearchResult(
        visits=np.array([1, 7, 4], dtype=np.int32),
        values=np.zeros(3, dtype=np.float32),
        actions=[(0, 0), (2, 1), (4, 3)],
        chosen=(2, 1),
        root_value=0.0,
        improved=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    actions, counts = ring_visits(result, width=2, max_sites=192)
    assert actions.tolist() == [2 * 192 + 1, 4 * 192 + 3]
    assert counts.tolist() == [7, 4]


# ---------------------------------------------------------------------------
# F-42: the target instruments — H(π′), H(visit share), KL(π′ ‖ prior)
# ---------------------------------------------------------------------------


def _root(visits, values):
    from reckoner.search import SearchResult, SearchStats, _improved_target

    visits = np.asarray(visits, np.int32)
    values = np.asarray(values, np.float32)
    n = len(visits)
    return SearchResult(
        visits,
        values,
        [(0, s) for s in range(n)],
        (0, 0),
        float(values.max()),
        SearchStats(),
        root_priors=np.full(n, 1 / n),
        improved=_improved_target(np.zeros(n), visits, values, 0.0, CFG),
    )


def test_the_target_entropy_is_pi_prime_and_the_visit_entropy_is_the_share() -> None:
    from reckoner.runner import _Live, _record

    result = _root([19, 3, 3, 3, 20], [-1, -1, -1, -1, 0])
    stats = IterationStats()
    _record(stats, result, _Live(problem=None, expr=None, rng=None, index=0))  # type: ignore[arg-type]
    assert stats.h_target_start == [entropy(result.improved)]
    assert stats.h_visits_start == [entropy(result.visit_share())]
    # H(π′) = 4 · e^-70 · (70 + log(1 + 4e^-70)) + ... ≈ 1.11e-28: by hand, not a tolerance.
    assert stats.h_target_start[0] == pytest.approx(4 * np.exp(-70.0) * 70.0, rel=1e-3)
    assert stats.h_visits_start[0] > 1.0


def test_kl_separates_a_root_that_taught_something_from_one_that_did_not() -> None:
    """Mute vs live: two roots with the SAME visit profile, one holding proofs.
    The visit-share entropy cannot tell them apart; KL can, and is exactly zero
    on the root where π′ is the prior."""
    from reckoner.runner import _Live, _record

    nothing = _root([19, 3, 3, 3, 20], [0, 0, 0, 0, 0])
    proofs = _root([19, 3, 3, 3, 20], [-1, -1, -1, -1, 0])
    a, b = IterationStats(), IterationStats()
    live = _Live(problem=None, expr=None, rng=None, index=0)  # type: ignore[arg-type]
    _record(a, nothing, live)
    _record(b, proofs, live)
    assert a.h_visits_start == b.h_visits_start
    assert a.kl_start == [0.0]
    assert b.kl_start[0] == pytest.approx(np.log(5), rel=1e-6)
    assert a.h_target_start != b.h_target_start


def test_kl_divergence_refuses_mass_the_prior_does_not_have() -> None:
    from reckoner.runner import kl_divergence

    assert kl_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    with pytest.raises(ValueError, match="no"):
        kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0]))


def test_kl_survives_a_prior_that_underflows_in_float32() -> None:
    """A confident network: a legal action 110 nats below the favourite has a
    positive float64 prior (e^-110) that underflows to 0 in float32. The search
    proves the favourite a loss, so π′ keeps mass on the underdog. The
    diagnostic must stay finite rather than halt self-play (F-42 review)."""
    from reckoner.runner import _Live, _record
    from reckoner.search import SearchResult, SearchStats, _improved_target

    logits = np.array([0.0, -110.0])
    visits = np.array([24, 24], np.int32)
    values = np.array([-1.0, 0.0], np.float32)
    prior = np.exp(logits - logits.max())
    prior = prior / prior.sum()
    assert np.float32(prior[1]) == 0.0 < prior[1], "the premise: float32 underflow"
    improved = _improved_target(logits, visits, values, 0.0, CFG)
    assert improved[1] > 0.0
    result = SearchResult(
        visits,
        values,
        [(0, 0), (0, 1)],
        (0, 1),
        0.0,
        SearchStats(),
        root_priors=prior,
        improved=improved,
    )
    stats = IterationStats()
    _record(stats, result, _Live(problem=None, expr=None, rng=None, index=0))  # type: ignore[arg-type]
    assert np.isfinite(stats.kl_start[0]) and stats.kl_start[0] > 0.0
