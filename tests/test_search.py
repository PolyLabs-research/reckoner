"""Search: the backup arithmetic first, then budget, equivalence, determinism.

The backup test is written first and stands first, because the port's sharpest
hazard is silent: a tree that negates per ply, or averages where it should
maximise, still returns plausible numbers.

**Which search.** Everything down to the last section plays ``uniform_stub``
under the search chunk 7's gates were measured with, built outright as ``CFG``:
no past-par rule, and the stub's own value read as given. WEAK-w1 added two
levers to the search (the past-par rule, the default since 2026-11-17, and the
steps-head value). The last section runs the same mechanics under each, on a
real network, because the stub reports no steps-to-go.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config, with_m1_search
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import Problem
from reckoner.expr import add, eq, mul, num, var
from reckoner.gates import descent_identity_holds, search_found_a_solve
from reckoner.search import _backup, _Tree, run_batched, search, uniform_stub
from reckoner.vocab import GOAL_SOLVE, VAR_X

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
#: The search chunk 7 measured: the stub's value is read as given ('wdl').
CFG = with_m1_search(Config())
X = var(VAR_X)

PROBLEM = Problem(
    goal=GOAL_SOLVE,
    target=VAR_X,
    par=3,
    par_source="bfs",
    expr=eq(add(mul(num(3), X), num(6)), num(21)),
)


# ---------------------------------------------------------------------------
# The backup arithmetic — hand-computed, mixed leaf types
# ---------------------------------------------------------------------------


def _wire(tree, parent, slot, expr):
    """Add a child AND wire it into its parent's slot.

    The pre-F-13 fixture added children without wiring them, which the
    max-accumulate backup did not notice because it never looked at
    `children`. Recompute does look, so the fixture now builds the tree it
    always claimed to.
    """
    child = tree.add(expr, parent, slot)
    tree.children[parent][slot] = child
    return child


def test_backup_is_max_and_never_negates() -> None:
    """A hand-computed three-level tree with **both** leaf kinds in it.

        root  (own-eval -1.0, fully expanded)
        ├── A          terminal-solved leaf        +1.0   proven
        └── B          own-eval -0.5, fully expanded
            ├── B1     value-estimated leaf        +0.25
            └── B2     terminal (too large) leaf   -1.0   proven

    Three distinct mistakes a blind port makes, and this catches all three:

    * **negation per ply** would make root = -1 x max(children).
    * **mean backup** would make root = mean(+1, -0.5, +0.25, -1) = -0.0625.
    * **own-eval flooring** would let a fully expanded node keep its own
      estimate in the max — the stale-prior-over-proof defect (F-13).

    By hand under the ruled semantics: B is fully expanded, so its own -0.5 does
    NOT participate and B = max(+0.25, -1.0) = +0.25. root is fully expanded, so
    its own -1.0 does not participate either, and root = max(+1.0, +0.25) = +1.0.
    """
    priors = np.zeros(2, dtype=np.float32)
    tree = _Tree(8)
    root = tree.add(PROBLEM.expr, -1, -1)
    tree.open(root, [(0, 0), (0, 1)], priors)
    tree.own_eval[root] = -1.0
    a = _wire(tree, root, 0, PROBLEM.expr)
    b = _wire(tree, root, 1, PROBLEM.expr)
    tree.open(b, [(0, 0), (0, 1)], priors)
    b1 = _wire(tree, b, 0, PROBLEM.expr)
    b2 = _wire(tree, b, 1, PROBLEM.expr)

    tree.terminal[a] = True
    tree.terminal[b2] = True
    tree.value[a] = 1.0
    tree.value[b2] = -1.0

    _backup(tree, a, 1.0)  # terminal-solved
    _backup(tree, b1, 0.25)  # value-estimated
    _backup(tree, b2, -1.0)  # terminal, too large

    assert tree.value[b1] == pytest.approx(0.25)
    assert tree.value[b2] == pytest.approx(-1.0)
    assert tree.value[b] == pytest.approx(0.25), "B must be the max under B, not the mean"
    assert tree.value[root] == pytest.approx(1.0), "root must be the max, and unnegated"

    # The negation a blind port would introduce would have produced this:
    assert tree.value[root] != pytest.approx(-1.0)
    # ...and the mean a different blind port would have produced, this:
    assert tree.value[root] != pytest.approx(np.mean([1.0, -0.5, 0.25, -1.0]))


def test_own_eval_participates_only_while_actions_remain() -> None:
    """**The ruled rule, both polarities, in one tree.**

    Own-eval is a proxy for what UNTRIED actions might yield. While one exists it
    belongs in the max; once the legal set is fully expanded it represents
    nothing and must drop out. Here own-eval is deliberately optimistic (+0.9)
    against a single child worth -1.0.
    """
    priors = np.zeros(2, dtype=np.float32)
    tree = _Tree(8)
    root = tree.add(PROBLEM.expr, -1, -1)
    tree.open(root, [(0, 0), (0, 1)], priors)
    tree.own_eval[root] = 0.9

    first = _wire(tree, root, 0, PROBLEM.expr)
    tree.terminal[first] = True
    tree.value[first] = -1.0
    _backup(tree, first, -1.0)
    # Polarity one: slot 1 is still untried, so the optimistic prior stands in.
    assert tree.value[root] == pytest.approx(0.9), "own-eval must hold while an action is untried"

    second = _wire(tree, root, 1, PROBLEM.expr)
    tree.terminal[second] = True
    tree.value[second] = -1.0
    _backup(tree, second, -1.0)
    # Polarity two: fully expanded, so the prior has nothing left to represent.
    assert tree.value[root] == pytest.approx(-1.0), (
        "a fully expanded node kept its own estimate — stale prior over proof (F-13)"
    )


def test_a_proof_may_lower_a_node_below_its_own_estimate() -> None:
    """Replaces the old monotonicity test, whose premise was the defect.

    The pre-F-13 test asserted a later worse line could never demote an earlier
    value. That monotonicity was a CONSEQUENCE of max-accumulating into a floor
    the node could not escape. Under the ruled semantics a proof is allowed to
    lower a node, and must be — otherwise a trained net's optimism could never be
    contradicted by evidence.
    """
    priors = np.zeros(1, dtype=np.float32)
    tree = _Tree(4)
    root = tree.add(PROBLEM.expr, -1, -1)
    tree.open(root, [(0, 0)], priors)
    tree.own_eval[root] = 0.8
    child = _wire(tree, root, 0, PROBLEM.expr)
    tree.terminal[child] = True
    tree.value[child] = -1.0
    _backup(tree, child, -1.0)
    assert tree.value[root] == pytest.approx(-1.0)
    assert tree.visits[root] == 1


def test_proofs_propagate_to_a_fully_expanded_parent() -> None:
    """A terminal's z is a proof; a fully expanded node of proven children is proven."""
    priors = np.zeros(2, dtype=np.float32)
    tree = _Tree(8)
    root = tree.add(PROBLEM.expr, -1, -1)
    tree.open(root, [(0, 0), (0, 1)], priors)
    tree.own_eval[root] = 0.0

    a = _wire(tree, root, 0, PROBLEM.expr)
    tree.terminal[a] = True
    tree.value[a] = 0.0
    _backup(tree, a, 0.0)
    assert not tree.proven[root], "one proven child of two is not a proof"

    b = _wire(tree, root, 1, PROBLEM.expr)
    tree.terminal[b] = True
    tree.value[b] = -1.0
    _backup(tree, b, -1.0)
    assert tree.proven[root], "every child proven and fully expanded — the node is proven"
    assert tree.value[root] == pytest.approx(0.0), "proven at max(children)"


# ---------------------------------------------------------------------------
# The depth-1 gate, at the arithmetic-verified (sims, m)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_the_gate_arithmetic_is_recorded_and_the_gate_matches_it() -> None:
    """**Chunk 7 gate.** 100% on depth 1 at the (sims, m) the arithmetic supports.

    Measured on the real suite, not chunk 2's stand-in samplers: B_max = 5, so
    a uniform stub considers the winning action with certainty only at m >= 5.
    At m = 3 the worst-case per-problem probability is 3/5 and P(all 200) is
    4e-45 — a 100% gate there would be a gate the arithmetic cannot support.
    """
    record = json.loads((REPO / "runs" / "gate_arithmetic.json").read_text())
    b_max = record["B_max"]
    assert b_max == 5
    assert record["by_goal"]["SOLVE"]["min"] == 5
    assert record["by_goal"]["EVALUATE"]["max"] == 1

    rows = read_suite(SUITES / "solve_in_1.jsonl")
    evaluator = uniform_stub(CFG)
    solved = 0
    for i, row in enumerate(rows):
        problem = suite_problem(row)
        result = search(
            problem, problem.expr, evaluator, CFG, random.Random(1000 + i), sims=16, m=b_max
        )
        # ONE predicate, imported. Expressing it here and again in
        # scripts/chunk7_gate_table.py is what produced the 0/200 re-run: two
        # gates wearing one name, and only one of them re-expressed.
        if search_found_a_solve(result):
            solved += 1
    assert solved == len(rows), f"depth-1 gate: {solved}/{len(rows)} at m={b_max}"


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_below_b_max_the_gate_is_not_reachable() -> None:
    """The other polarity: the arithmetic predicts a miss, and there is one.

    Without this, "100% at m=5" would be consistent with the gate being trivially
    satisfiable at any m — which would mean the arithmetic decided nothing.
    """
    rows = read_suite(SUITES / "solve_in_1.jsonl")
    evaluator = uniform_stub(CFG)
    solved = 0
    for i, row in enumerate(rows):
        problem = suite_problem(row)
        result = search(
            problem, problem.expr, evaluator, CFG, random.Random(1000 + i), sims=16, m=3
        )
        if search_found_a_solve(result):
            solved += 1
    assert solved < len(rows), "m=3 solved everything — B_max is wrong or the stub is not uniform"


# ---------------------------------------------------------------------------
# Budget identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sims", [6, 16, 31, 48])
def test_budget_identity(sims: int) -> None:
    """A terminal short-circuit still consumes a simulation. Budget means budget."""
    evaluator = uniform_stub(CFG)
    result = search(PROBLEM, PROBLEM.expr, evaluator, CFG, random.Random(7), sims=sims, m=5)
    assert result.stats.sims_requested == sims
    assert result.stats.sims_used <= sims
    assert int(result.visits.sum()) == result.stats.sims_used


# ---------------------------------------------------------------------------
# Sync vs batched — the S2-class parity test, full grid, odd pairs included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 5, 12, 16])
@pytest.mark.parametrize("sims", [6, 16, 31, 48])
def test_sequential_and_batched_agree_exactly(m: int, sims: int) -> None:
    """Batching is across *searches*, so parity is exact rather than a tolerance.

    Pooling leaves inside one tree would change what the tree does — simulation
    k+1's selection depends on k's backup — and the test would then compare two
    algorithms. Across trees there is no coupling, so this is an equality.
    Parametrised over the full m x sims grid, odd pairs included.
    """
    evaluator = uniform_stub(CFG)
    problems = [PROBLEM] * 4
    sequential = [
        search(p, p.expr, evaluator, CFG, random.Random(20 + i), sims=sims, m=m)
        for i, p in enumerate(problems)
    ]
    batched = run_batched(
        [(p, p.expr, random.Random(20 + i)) for i, p in enumerate(problems)],
        evaluator,
        CFG,
        sims=sims,
        m=m,
        batch_size=3,
    )
    for a, b in zip(sequential, batched, strict=True):
        assert np.array_equal(a.visits, b.visits), f"visit parity failed at m={m} sims={sims}"
        assert np.allclose(a.values, b.values)
        assert a.chosen == b.chosen
        assert a.stats.sims_used == b.stats.sims_used
        assert a.stats.nodes == b.stats.nodes
        assert a.stats.evaluations == b.stats.evaluations


# ---------------------------------------------------------------------------
# The tree must actually deepen — the gate that F-06 was missing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (SUITES / "solve_in_5.jsonl").exists(), reason="suites not generated")
def test_the_tree_deepens_past_one_ply() -> None:
    """**The gate F-06 was missing.** A search that does not search passes the rest.

    The first version of the search expanded only the root's children and then
    re-backed-up their values, so ``nodes`` was flat across every budget: on a
    6-action root at ``m=5`` it read 6, 6, 6, 6 for sims 6, 16, 31, 48. Every
    other gate passed anyway — depth-1 needs one ply, budget identity counts
    visits regardless, and sync-vs-batched compared two equally shallow paths.

    ``nodes`` was in ``SearchStats`` the whole time. Nothing asserted on it.
    That is what this test is for. (See ``ERRATA-chunk7.md`` §3 for why the
    originally published exhibit — "48 simulations produced 2 nodes" — is not a
    demonstration of the defect unless the problem is named.)
    """
    rows = [r for r in read_suite(SUITES / "solve_in_5.jsonl") if r["goal"] == GOAL_SOLVE]
    problem = suite_problem(rows[0])
    evaluator = uniform_stub(CFG)

    small = search(problem, problem.expr, evaluator, CFG, random.Random(1), sims=6, m=5)
    large = search(problem, problem.expr, evaluator, CFG, random.Random(1), sims=48, m=5)

    assert large.stats.nodes > small.stats.nodes, "more simulations built no more tree"
    assert large.stats.nodes > 5 + 1, "the tree is only the root and its children"
    assert large.stats.max_depth >= 2, f"max_depth {large.stats.max_depth} — one ply only"
    # Every simulation must do work: nodes grow roughly with sims, not with m.
    assert large.stats.nodes >= 0.5 * large.stats.sims_used


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_seeds_give_identical_results_in_the_eval_profile() -> None:
    """The plan's 2-seed identity: with root noise off, the draw cannot matter."""
    cfg = with_m1_search(Config())
    cfg.search.root_noise = False
    evaluator = uniform_stub(cfg)
    a = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(1), sims=16, m=5)
    b = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(999), sims=16, m=5)
    assert np.array_equal(a.visits, b.visits)
    assert a.chosen == b.chosen


def test_root_noise_actually_varies_the_search() -> None:
    """Both polarities: if the seed never mattered, the identity above is vacuous."""
    cfg = with_m1_search(Config())
    assert cfg.search.root_noise is True
    evaluator = uniform_stub(cfg)
    seen = {
        tuple(search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(s), sims=16, m=3).visits)
        for s in range(12)
    }
    assert len(seen) > 1, "root noise changed nothing — the eval-profile test proves nothing"


def test_search_is_byte_identical_across_processes() -> None:
    """Cross-process under varied PYTHONHASHSEED — the construction-gate standard."""
    # The probe is a FILE, not an embedded string. It was a quoted `-c` program
    # until a blind text edit matched inside the literal and broke it — twice.
    # Documentation warns, methods help, helpers prevent: a program in its own
    # file has nothing for an edit to corrupt silently, and it lints.
    program = REPO / "tests" / "subprocess_probes" / "search_determinism.py"
    outputs = set()
    for seed in ("0", "1", "7777", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.add(
            subprocess.run(
                [sys.executable, str(program)],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            ).stdout.strip()
        )
    assert len(outputs) == 1, f"search varied across PYTHONHASHSEED: {outputs}"


# ---------------------------------------------------------------------------
# StateTooLarge is a counted terminal loss
# ---------------------------------------------------------------------------


def test_an_over_long_state_is_a_counted_terminal_loss_not_a_crash() -> None:
    """Step-cap-equivalent semantics: the budget cannot represent where it went.

    Never a crash mid-batch, and never an edit to the legality mask — legality
    stays the engine's alone. add_both_sides makes this reachable from the first
    stub rollout (FINDINGS.md F-05), so the counter is in SearchStats from
    field one.
    """
    cfg = with_m1_search(Config())
    cfg.model.seq_len = 26  # the start state fits; one add_both_sides does not
    evaluator = uniform_stub(cfg)
    result = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(3), sims=24, m=5)

    assert result.stats.state_too_large > 0, "the over-long case was never reached"
    assert result.stats.sims_used > 0
    assert np.isfinite(result.values).all()
    assert result.values.min() >= -1.0
    # It is a loss, not a discard: the budget was spent and the visits recorded.
    assert int(result.visits.sum()) == result.stats.sims_used


def test_search_stats_name_every_terminal_kind() -> None:
    stats = search(
        PROBLEM, PROBLEM.expr, uniform_stub(CFG), CFG, random.Random(1), sims=16, m=5
    ).stats
    for field_name in ("terminal_solved", "terminal_no_actions", "state_too_large"):
        assert field_name in stats.as_dict()
    assert stats.as_dict()["sims_requested"] == 16


def test_a_root_with_no_legal_action_returns_cleanly() -> None:
    """Empty action set, not "solved" — the two differ and only one is here.

    A *solved* equation still has legal rewrites: `x = 5` is goal form, and the
    both-sides rules remain legal on it. Whether to stop there is the episode's
    decision, not search's — search's contract is that its root is non-terminal.
    The genuinely empty case is a SIMPLIFY normal form.
    """
    from reckoner.rules import legal_actions
    from reckoner.vocab import GOAL_SIMPLIFY

    assert legal_actions(eq(X, num(5))), "a solved equation still has legal rewrites"

    stuck = Problem(goal=GOAL_SIMPLIFY, par=1, par_source="bfs", expr=X)
    assert legal_actions(stuck.expr) == []
    result = search(stuck, stuck.expr, uniform_stub(CFG), CFG, random.Random(1), sims=16, m=5)
    assert result.chosen is None and result.actions == []
    assert result.stats.terminal_no_actions == 1
    assert result.stats.sims_used == 0, "an empty root spends nothing"


# ---------------------------------------------------------------------------
# WEAK-w1's two levers: the same mechanics with each pulled
# ---------------------------------------------------------------------------
# ``bound`` is the past-par rule alone, which is ``Config()`` as it stands;
# ``steps`` adds the steps-head value. The evaluator is a real network at random
# weights: under ``'steps'`` the search needs a steps-to-go with every answer,
# and the stub has none to give.

LEVERS = {"bound": ("loss", "wdl"), "steps": ("loss", "steps")}


def _with_levers(name: str):
    """One of WEAK-w1's searches in the measurement profile, and a network to play it."""
    import torch

    from reckoner.evaluate import model_evaluator
    from reckoner.model import Reckoner

    cfg = Config()
    cfg.search.past_par, cfg.search.value_source = LEVERS[name]
    cfg.search.root_noise = False
    torch.manual_seed(0)
    return cfg, model_evaluator(Reckoner(cfg).eval(), cfg, 0.0)


def test_the_default_search_is_the_past_par_rule_alone() -> None:
    cfg = Config()
    assert (cfg.search.past_par, cfg.search.value_source) == LEVERS["bound"]


@pytest.mark.parametrize("lever", sorted(LEVERS))
@pytest.mark.parametrize("sims", [6, 31, 48])
def test_the_levered_search_keeps_its_budget_and_its_books(lever: str, sims: int) -> None:
    """Budget identity and the descent identity, with the past-par terminals in it."""
    cfg, evaluator = _with_levers(lever)
    result = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(7), sims=sims, m=5)
    assert result.stats.sims_requested == sims
    assert result.stats.sims_used <= sims
    assert int(result.visits.sum()) == result.stats.sims_used
    assert descent_identity_holds(result)
    assert result.values.min() >= -1.0 and result.values.max() <= 0.0, "exact par has no +1"


@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_the_levered_search_scores_states_past_par_and_counts_them(lever: str) -> None:
    """The rule's central job on a whole search: with enough budget on a par-3
    problem the tree reaches depth 3 unsolved, and those nodes are terminals the
    evaluator never sees. Without the count the descent identity would not close."""
    cfg, evaluator = _with_levers(lever)
    result = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(7), sims=48, m=16)
    assert result.stats.past_par > 0, "no state past par was reached, so nothing was tested"
    assert result.stats.max_depth <= PROBLEM.par, "the search expanded below a certain loss"
    assert descent_identity_holds(result)


@pytest.mark.parametrize("lever", sorted(LEVERS))
@pytest.mark.parametrize(("m", "sims"), [(3, 31), (5, 16), (16, 48)])
def test_the_levered_search_agrees_with_itself_batched(lever: str, m: int, sims: int) -> None:
    """Sequential against batched, odd pairs included. The batch crops to its own
    longest state, so a row can be padded differently from run to run: priors
    and values agree to rounding, the decisions exactly."""
    cfg, evaluator = _with_levers(lever)
    problems = [PROBLEM] * 3
    sequential = [
        search(p, p.expr, evaluator, cfg, random.Random(20 + i), sims=sims, m=m)
        for i, p in enumerate(problems)
    ]
    batched = run_batched(
        [(p, p.expr, random.Random(20 + i)) for i, p in enumerate(problems)],
        evaluator,
        cfg,
        sims=sims,
        m=m,
        batch_size=2,
    )
    for a, b in zip(sequential, batched, strict=True):
        assert np.array_equal(a.visits, b.visits), f"visit parity failed at m={m} sims={sims}"
        assert np.allclose(a.values, b.values, atol=1e-5)
        assert a.chosen == b.chosen
        assert a.stats.as_dict() == b.stats.as_dict()


@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_the_levered_search_ignores_the_seed_in_the_eval_profile(lever: str) -> None:
    cfg, evaluator = _with_levers(lever)
    a = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(1), sims=16, m=5)
    b = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(999), sims=16, m=5)
    assert np.array_equal(a.visits, b.visits)
    assert a.chosen == b.chosen
