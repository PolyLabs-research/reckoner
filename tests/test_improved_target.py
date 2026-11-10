"""F-54: the Phase-2 policy target is the Gumbel improved policy π′.

π′ = softmax(logits + σ(completed Q)) over **every** legal root action, with
σ(q) = (c_visit + max N) · c_scale · q, visited actions completed by their own
(max-backed) value and unvisited ones by v_mix (Danihelka et al. 2022, eq. 33).
Before F-54 the target was the Sequential-Halving visit share, whose shape is a
function of the visited count alone (`FINDINGS.md` F-54).

Every number below is an **absolute** reference value computed by hand (or by
``math.exp`` in the test), never by the code under test: a target that is wrong
in the same way on both sides of a round trip survives every round trip.

No gitignored data: problems come from the tracked suites, the generator, or the
expression constructors.
"""

from __future__ import annotations

import inspect
import math
import random
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from reckoner import search as search_module
from reckoner.config import POLICY_TARGETS, Config
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.evaluate import model_evaluator
from reckoner.model import N_RULES, Reckoner, action_index
from reckoner.rules import apply
from reckoner.search import (
    SearchResult,
    SearchStats,
    _improved_target,
    _leaf_outcome,
    _root_improved,
    _Tree,
    search,
    uniform_stub,
)

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()
CAP = CFG.episode.step_cap


def _quiet(cfg: Config = CFG) -> Config:
    return replace(cfg, search=replace(cfg.search, root_noise=False))


# ---------------------------------------------------------------------------
# The central job first (rider a): π′ moves mass off proven losses
# ---------------------------------------------------------------------------


def test_improved_target_is_the_hand_computed_vector_on_a_proof_root() -> None:
    """**Pinned reference vector**, the F-54 constructed case in pure form.

    Uniform logits, visits [19, 3, 3, 3, 20], values [-1, -1, -1, -1, 0] (four
    proven losses and one at-par solve), v̂ = 0. Every action is visited, so the
    completion does not enter. max N = 20, so σ = (50 + 20) · 1 · q = 70 q and

        π′(loser)  = e^-70 / (1 + 4 e^-70)
        π′(solver) = 1     / (1 + 4 e^-70)

    The visit share would teach 28/48 = 0.5833 toward the losses.
    """
    visits = np.array([19, 3, 3, 3, 20], dtype=np.int32)
    values = np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
    pi = _improved_target(np.zeros(5), visits, values, 0.0, CFG)

    assert pi.dtype == np.float32
    loser = math.exp(-70.0) / (1.0 + 4.0 * math.exp(-70.0))
    assert pi[:4].tolist() == [float(np.float32(loser))] * 4
    assert float(np.float32(loser)) == pytest.approx(3.975449954226706e-31, rel=1e-6)
    assert pi[4] == np.float32(1.0)
    # The counter-reading, so the test says what it displaced.
    share = visits / visits.sum()
    assert float(share[:4].sum()) == pytest.approx(28 / 48)


def _five_action_cap_roots():
    """All five-action single-solver solve_in_1 states, re-posed at the step cap.

    Par = cap and steps_taken = cap - 1, so the solver is an exact at-par solve
    (z = 0) and every other action runs into the cap (z = -1): every root action
    is a proof, and nothing about the value head enters.
    """
    for row in read_suite(SUITES / "solve_in_1.jsonl"):
        problem = replace(suite_problem(row), par=CAP, par_source="pool")
        result = search(
            problem,
            problem.expr,
            uniform_stub(_quiet()),
            _quiet(),
            random.Random(0),
            steps_taken=CAP - 1,
        )
        if len(result.actions) == 5:
            yield problem, result


def test_improved_target_takes_the_mass_off_proven_losses_end_to_end() -> None:
    """The real search on all 40 constructed roots: the four-tuple.

    floor 0 · null 28/48 (the visit share, measured in F-54) · threshold <= 1e-20
    · measured: each loser at e^-70 / (1 + 4 e^-70), the solver at 1.0.
    """
    loser = float(np.float32(math.exp(-70.0) / (1.0 + 4.0 * math.exp(-70.0))))
    roots = 0
    for problem, result in _five_action_cap_roots():
        roots += 1
        z = [
            _leaf_outcome(
                problem,
                apply(problem.expr, *a),
                CFG,
                random.Random(0),
                SearchStats(),
                total_steps=CAP,
            )
            for a in result.actions
        ]
        assert sorted(z) == [-1.0, -1.0, -1.0, -1.0, 0.0]
        solver = z.index(0.0)
        assert sorted(result.visits.tolist(), reverse=True) == [20, 19, 3, 3, 3]
        assert result.values.tolist() == z, "terminal root children must carry their exact z"
        assert result.chosen == result.actions[solver]
        assert result.improved[solver] == np.float32(1.0)
        assert [float(result.improved[s]) for s in range(5) if s != solver] == [loser] * 4
        assert result.visit_share()[solver] == pytest.approx(20 / 48)
    assert roots == 40, f"the constructed population is 40 roots; found {roots}"


# ---------------------------------------------------------------------------
# The completion for unvisited actions is v_mix — and the alternatives fail
# ---------------------------------------------------------------------------


def test_unvisited_actions_are_completed_with_v_mix() -> None:
    """Vector V1, by hand. logits 0 x 4, visits [3, 1, 0, 0], values [1, 0, -1, -1]
    (the -1s under visits == 0 are the initialiser, not values), v̂ = 0.

    v_mix = (0 + 4 * (0.5 * 1 + 0.5 * 0)) / (1 + 4) = 0.4; max N = 3, so σ = 53 q
    and π′ ∝ [e^53, e^0, e^21.2, e^21.2].
    """
    visits = np.array([3, 1, 0, 0], dtype=np.int32)
    values = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    pi = _improved_target(np.zeros(4), visits, values, 0.0, CFG)

    logits = np.array([53.0, 0.0, 21.2, 21.2])
    expected = np.exp(logits - logits.max())
    expected = (expected / expected.sum()).astype(np.float32)
    assert pi.tolist() == expected.tolist()
    assert pi[2] > pi[1], "an unknown action must outrank a visited 0"
    assert float(pi[2] + pi[3]) < 1e-12

    # What each rejected completion would have produced — and that it differs.
    def with_completion(c: float) -> np.ndarray:
        q = np.where(visits > 0, values.astype(np.float64), c)
        z = 53.0 * q
        e = np.exp(z - z.max())
        return (e / e.sum()).astype(np.float32)

    assert pi.tolist() != with_completion(1.0).tolist(), "max-backed root value (A)"
    assert pi.tolist() != with_completion(0.0).tolist(), "v̂ alone (B)"
    assert pi.tolist() != with_completion(-1.0).tolist(), "the -1.0 initialiser leaking"


def test_v_mix_uses_v_hat_and_prior_weights() -> None:
    """Vector V2: visits [10, 6, 0, 0], values [0, -1, ., .], v̂ = 0.2, logits 0.

    v_mix = (0.2 + 16 * (0.5 * 0 + 0.5 * -1)) / 17 = -7.8 / 17. A and B would both
    give 0.2. σ scale = 50 + 10 = 60.
    """
    visits = np.array([10, 6, 0, 0], dtype=np.int32)
    values = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
    pi = _improved_target(np.zeros(4), visits, values, 0.2, CFG)
    v_mix = -7.8 / 17
    z = 60.0 * np.array([0.0, -1.0, v_mix, v_mix])
    e = np.exp(z - z.max())
    assert pi.tolist() == (e / e.sum()).astype(np.float32).tolist()


def test_unvisited_values_never_reach_the_target() -> None:
    """Poison the unvisited slots: π′ must not move. Mask by visits, never by value."""
    visits = np.array([3, 1, 0, 0], dtype=np.int32)
    clean = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    poisoned = np.array([1.0, 0.0, 7.0, np.nan], dtype=np.float32)
    a = _improved_target(np.zeros(4), visits, clean, 0.0, CFG)
    b = _improved_target(np.zeros(4), visits, poisoned, 0.0, CFG)
    assert a.tolist() == b.tolist()


# ---------------------------------------------------------------------------
# Where search found nothing, π′ is the prior
# ---------------------------------------------------------------------------


def test_improved_is_the_prior_where_search_found_nothing() -> None:
    """Uniform stub, no proofs at the root: every visited q is 0 and v̂ is 0, so
    σ adds nothing and π′ equals the prior exactly. This is Gumbel's intended
    behaviour and the reason the target-entropy instrument alone cannot see it."""
    rows = read_suite(SUITES / "solve_in_3.jsonl")
    checked = 0
    for i in sample_indices(len(rows), 12, 0):
        problem = suite_problem(rows[i])
        result = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(i))
        visited = result.visits > 0
        if np.any(result.values[visited] != 0.0):
            continue  # a root holding a proof is not this test's population
        checked += 1
        assert np.array_equal(result.improved, result.root_priors.astype(np.float32))
    assert checked >= 8, f"only {checked} proof-free roots; the population is too thin"


# ---------------------------------------------------------------------------
# The tripwire, both polarities
# ---------------------------------------------------------------------------


def _two_action_tree() -> tuple[_Tree, int]:
    from reckoner.expr import num

    tree = _Tree(8)
    root = tree.add(num(1), -1, -1)
    tree.open(root, [(0, 0), (1, 0)], np.zeros(2, dtype=np.float32))
    tree.own_eval[root] = 0.0
    tree.value[root] = 0.0
    return tree, root


def test_a_child_without_visits_trips_the_root_tripwire() -> None:
    """A root child that exists with zero visits means the visited mask and the
    tree disagree, and the completion would read an initialiser as a value."""
    from reckoner.expr import num

    tree, root = _two_action_tree()
    tree.children[root][0] = tree.add(num(2), root, 0)  # type: ignore[index]
    with pytest.raises(RuntimeError, match="visited"):
        _root_improved(tree, root, np.zeros(2), CFG)


def test_a_consistent_tree_passes_the_tripwire() -> None:
    from reckoner.expr import num

    tree, root = _two_action_tree()
    tree.children[root][0] = tree.add(num(2), root, 0)  # type: ignore[index]
    tree.child_visits[root][0] = 1  # type: ignore[index]
    tree.child_values[root][0] = 0.0  # type: ignore[index]
    pi = _root_improved(tree, root, np.zeros(2), CFG)
    assert pi.shape == (2,) and abs(float(pi.sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# π′ never feeds back into the search, and the Gumbel draw never enters π′
# ---------------------------------------------------------------------------


def _signature(result: SearchResult) -> tuple:
    return (
        result.visits.tobytes(),
        result.values.tobytes(),
        result.chosen,
        result.root_value,
        tuple(sorted(result.stats.__dict__.items())),
        np.asarray(result.root_priors).tobytes(),
    )


def _fixture_problems(n: int = 3) -> list:
    out = []
    for name in ("solve_in_2.jsonl", "solve_in_4.jsonl", "scripted_in_7.jsonl"):
        rows = read_suite(SUITES / name)
        out += [suite_problem(rows[i]) for i in sample_indices(len(rows), n, 0)]
    return out


def _evaluators():
    torch.manual_seed(0)
    model = Reckoner(CFG)
    model.eval()
    return {
        "uniform": uniform_stub(CFG),
        "net_value_0": model_evaluator(model, CFG, 0.0),
        "net_value_1": model_evaluator(model, CFG, 1.0),
    }


def test_the_target_never_feeds_back_into_the_search(monkeypatch) -> None:
    """Replace π′ with a constant: everything the search DID must be bit-identical.
    That is the one-lever property at the search: this change moves the target,
    never which actions are visited, chosen, or how values back up."""
    for name, evaluator in _evaluators().items():
        for i, problem in enumerate(_fixture_problems()):
            base = search(problem, problem.expr, evaluator, CFG, random.Random(i))
            with monkeypatch.context() as m:
                m.setattr(
                    search_module,
                    "_root_improved",
                    lambda tree, root, logits, cfg: np.full(
                        len(tree.actions[root]), 1 / len(tree.actions[root]), np.float32
                    ),
                )
                patched = search(problem, problem.expr, evaluator, CFG, random.Random(i))
            assert _signature(base) == _signature(patched), f"{name}, problem {i}"


def test_the_feedback_test_can_see_a_feedback_path(monkeypatch) -> None:
    """The other polarity. The non-root rule shares the softmax formula; break it
    and the search must move — otherwise the test above could not fail."""
    problem = _fixture_problems(1)[1]
    base = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(0))
    monkeypatch.setattr(
        search_module,
        "_gumbel_improved",
        lambda priors, completed, max_visits, cfg: np.eye(len(priors))[-1],
    )
    moved = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(0))
    assert _signature(base) != _signature(moved)


def _legacy_select(tree, node, cfg):
    """`_select` VERBATIM as of 6a9b18f (search.py:260-266), before F-54."""
    priors, visits, values = tree.priors[node], tree.child_visits[node], tree.child_values[node]
    assert priors is not None and visits is not None and values is not None
    completed = np.where(visits > 0, values, float(tree.value[node]))
    improved = search_module._softmax(
        priors + search_module._sigma(completed, int(visits.max(initial=0)), cfg)
    )
    return int(np.argmax(improved - visits / (1 + visits.sum())))


@pytest.mark.parametrize(("sims", "m"), [(48, 16), (16, 3), (6, 16)])
def test_the_refactored_select_is_bit_identical_to_the_legacy_rule(
    monkeypatch, sims: int, m: int
) -> None:
    """The non-root rule now calls the shared formula. It must choose exactly as
    the pre-F-54 code did, under every evaluator, with root noise on and off."""
    for noise in (True, False):
        cfg = replace(CFG, search=replace(CFG.search, root_noise=noise))
        for name, evaluator in _evaluators().items():
            for i, problem in enumerate(_fixture_problems(2)):
                new = search(
                    problem, problem.expr, evaluator, cfg, random.Random(i), sims=sims, m=m
                )
                with monkeypatch.context() as patch:
                    patch.setattr(search_module, "_select", _legacy_select)
                    old = search(
                        problem, problem.expr, evaluator, cfg, random.Random(i), sims=sims, m=m
                    )
                assert _signature(new) == _signature(old), f"{name} noise={noise} problem {i}"
                assert new.improved.tobytes() == old.improved.tobytes()


def test_the_gumbel_draw_never_enters_the_target() -> None:
    """A two-action root visits [24, 24] whatever the draw, so π′ must be the same
    under every seed. The draw must still have moved the search — here, which of
    two equal actions it acted on — or the test proves nothing."""
    rows = read_suite(SUITES / "solve_in_2.jsonl")
    problem = None
    for row in rows:
        candidate = suite_problem(row)
        if len(search_module.legal_actions(candidate.expr)) == 2:
            problem = candidate
            break
    assert problem is not None
    targets, chosen = set(), set()
    for seed in range(8):
        result = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(seed))
        targets.add(result.improved.tobytes())
        chosen.add(result.chosen)
    assert len(chosen) > 1, "the draw never changed the act; this root cannot test independence"
    assert len(targets) == 1


def test_sequential_and_batched_search_agree_on_the_target() -> None:
    """The batched runner pools evaluator calls only; under an EXACT evaluator π′
    must match bit for bit.

    Under a network it does not, and the cause is upstream of π′: the leaf VALUES
    already differ between the two paths (by up to 1.2e-7 on the F-54 fixture),
    because a network's output depends on the batch it is evaluated in. The
    integer visit share never carried that; π′ does (F-54, recorded). The
    campaign uses one path consistently, so it reproduces run to run.
    """
    problems = _fixture_problems(2)
    evaluator = uniform_stub(CFG)
    items = [(p, p.expr, random.Random(i), 0) for i, p in enumerate(problems)]
    batched = search_module.run_batched(items, evaluator, CFG)
    for i, (p, result) in enumerate(zip(problems, batched, strict=True)):
        single = search(p, p.expr, evaluator, CFG, random.Random(i))
        assert single.values.tobytes() == result.values.tobytes()
        assert single.improved.tobytes() == result.improved.tobytes()


# ---------------------------------------------------------------------------
# The API: one name for one concept
# ---------------------------------------------------------------------------


def test_improved_is_required_and_keyword_only() -> None:
    """A default would be a provenance default claiming 'computed' (F-02); a
    positional slot would be rebound by the next field someone inserts."""
    spec = {f.name: f for f in fields(SearchResult)}["improved"]
    assert spec.kw_only
    parameter = inspect.signature(SearchResult).parameters["improved"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    args = (np.zeros(1, np.int32), np.zeros(1, np.float32), [(0, 0)], (0, 0), 0.0, SearchStats())
    with pytest.raises(TypeError):
        SearchResult(*args)  # improved omitted
    with pytest.raises(TypeError):
        SearchResult(*args, np.zeros(1), np.zeros(1))  # an 8th positional


def test_the_visit_share_has_its_own_name_and_the_old_one_is_gone() -> None:
    result = SearchResult(
        np.array([3, 1, 0], np.int32),
        np.zeros(3, np.float32),
        [(0, 0), (1, 0), (2, 0)],
        (0, 0),
        0.0,
        SearchStats(),
        improved=np.full(3, 1 / 3, np.float32),
    )
    assert not hasattr(result, "improved_policy")
    assert result.visit_share().tolist() == [0.75, 0.25, 0.0]


def test_the_config_names_exactly_the_targets_the_reader_dispatches() -> None:
    from reckoner.train import TARGET_FIELDS

    assert set(TARGET_FIELDS) == set(POLICY_TARGETS) == {"improved", "visits"}
    assert Config().train.policy_target == "improved"


# ---------------------------------------------------------------------------
# The writer: π′ in action coordinates, the tail recorded exactly
# ---------------------------------------------------------------------------


def _result(actions, improved, visits=None) -> SearchResult:
    n = len(actions)
    return SearchResult(
        np.asarray(visits if visits is not None else [1] * n, np.int32),
        np.zeros(n, np.float32),
        list(actions),
        actions[0],
        0.0,
        SearchStats(),
        improved=np.asarray(improved, np.float32),
    )


def test_ring_improved_stores_action_indices_by_mass_with_the_tail_exact() -> None:
    """Pinned: actions [(0,0), (2,1), (6,3)], π′ [0.2, 0.5, 0.3] at max_sites 192.

    Dense indices 0, 2*192+1 = 385, 6*192+3 = 1155. Sorted by mass: 385 (0.5),
    1155 (0.3), 0 (0.2). Action 0 is included on purpose: it is also the padding
    value, and a writer that confused the two would pass a test without it.
    """
    from reckoner.runner import ring_improved

    actions, probs, tail = ring_improved(_result([(0, 0), (2, 1), (6, 3)], [0.2, 0.5, 0.3]), 8, 192)
    assert actions.tolist() == [385, 1155, 0, 0, 0, 0, 0, 0]
    assert probs.tolist() == [np.float32(0.5), np.float32(0.3), np.float32(0.2), 0, 0, 0, 0, 0]
    assert tail == 0.0
    assert actions.dtype == np.int32 and probs.dtype == np.float32


def test_ring_improved_records_the_dropped_mass_exactly() -> None:
    from reckoner.runner import ring_improved

    n = 131
    weights = np.arange(1, n + 1, dtype=np.float64)
    pi = (weights / weights.sum()).astype(np.float32)
    acts = [(r, s) for r in range(N_RULES) for s in range(CFG.model.max_sites)][:n]
    actions, probs, tail = ring_improved(_result(acts, pi), 128, CFG.model.max_sites)
    kept = np.argsort(-pi, kind="stable")[:128]
    assert actions.tolist() == [action_index(*acts[s], CFG.model.max_sites) for s in kept]
    assert tail == float(np.sort(pi)[:3].astype(np.float64).sum())


def test_ring_improved_refuses_a_target_not_aligned_with_the_actions() -> None:
    from reckoner.runner import ring_improved

    with pytest.raises(ValueError, match="aligned"):
        ring_improved(_result([(0, 0), (1, 0)], [1.0]), 8, 192)


def test_both_targets_are_storable_in_one_record() -> None:
    """Sanity on the dense width the reader expands into."""
    assert N_RULES * CFG.model.max_sites == 1344
