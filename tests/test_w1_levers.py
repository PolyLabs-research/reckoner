"""WEAK-w1's two levers: what the search does past par, and where its value comes from.

Both are config keys that default to the search M1 ran. First gate first (AGENTS
§5 rider a): each lever is shown doing its central job on a real search, and
doing nothing at its default.

* ``search.past_par = 'loss'`` — an unsolved state at or past par is scored -1.
  Its job: stop an unexplored line outranking a solution once the episode is
  over par.
* ``search.value_source = 'steps'`` — the value of an unproven state comes from
  the steps head against par. Its job: let the search refute a bad move at
  first sight, where a silent value head makes it exhaust everything beneath.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from reckoner.config import PAST_PAR_RULES, VALUE_SOURCES, Config, validate
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import Problem, bfs_par, verify
from reckoner.evaluate import model_evaluator
from reckoner.gates import descent_identity_holds
from reckoner.model import Reckoner, encode
from reckoner.rules import apply, legal_actions
from reckoner.search import SearchStats, _leaf_outcome, _state_value, search, uniform_stub

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated"
)


def with_search(**changes) -> Config:
    return replace(CFG, search=replace(CFG.search, root_noise=False, **changes))


def suite(depth: int) -> list[Problem]:
    return [suite_problem(row) for row in read_suite(SUITES / f"solve_in_{depth}.jsonl")]


def one_step_from_solved_with_a_way_to_dawdle() -> tuple[Problem, int]:
    """A depth-1 problem with a solving action AND an action that does not solve.

    Returns the problem and the slot of the solving action.
    """
    rng = random.Random(0)
    for problem in suite(1):
        actions = legal_actions(problem.expr)
        solving = [
            k
            for k, (rule_id, site_id) in enumerate(actions)
            if verify(problem, apply(problem.expr, rule_id, site_id), CFG, rng)
        ]
        if solving and len(solving) < len(actions):
            return problem, solving[0]
    raise AssertionError("no depth-1 suite problem has both a solving and a non-solving action")


# ------------------------------------------------------------------ config


@pytest.mark.parametrize(
    ("key", "legal", "default"),
    [("past_par", PAST_PAR_RULES, "evaluate"), ("value_source", VALUE_SOURCES, "wdl")],
)
def test_the_levers_are_validated_both_ways_and_default_to_m1s_search(key, legal, default) -> None:
    assert getattr(CFG.search, key) == default
    for good in legal:
        validate(replace(CFG, search=replace(CFG.search, **{key: good})))
    for bad in ("", "Loss", "value", "step"):
        with pytest.raises(ValueError, match=key):
            validate(replace(CFG, search=replace(CFG.search, **{key: bad})))


# ---------------------------------------------------------------- past par


@needs_suites
def test_past_par_loss_scores_an_unsolved_state_at_par_and_only_there() -> None:
    """Both polarities on the function itself, and the count that keeps the books."""
    problem = suite(3)[0]
    assert problem.par == 3
    rng = random.Random(0)
    for rule, at_par, counted in (("evaluate", None, 0), ("loss", -1.0, 2)):
        cfg = with_search(past_par=rule)
        stats = SearchStats()
        # Before par: undecided under both rules. The state is the start state,
        # which is unsolved; only the step count differs between the calls.
        assert _leaf_outcome(problem, problem.expr, cfg, rng, stats, total_steps=2) is None
        assert _leaf_outcome(problem, problem.expr, cfg, rng, stats, total_steps=3) == at_par
        assert _leaf_outcome(problem, problem.expr, cfg, rng, stats, total_steps=7) == at_par
        assert stats.past_par == counted


@needs_suites
def test_a_solved_state_is_scored_by_par_under_both_rules() -> None:
    """The lever must not touch a solution: its z is the par game's, as before."""
    problem, slot = one_step_from_solved_with_a_way_to_dawdle()
    rule_id, site_id = legal_actions(problem.expr)[slot]
    solved = apply(problem.expr, rule_id, site_id)
    rng = random.Random(0)
    for rule in PAST_PAR_RULES:
        cfg = with_search(past_par=rule)
        stats = SearchStats()
        assert _leaf_outcome(problem, solved, cfg, rng, stats, total_steps=1) == 0.0
        assert _leaf_outcome(problem, solved, cfg, rng, stats, total_steps=5) == -1.0
        assert stats.past_par == 0 and stats.terminal_solved == 2


@needs_suites
def test_past_par_the_merged_search_ranks_an_unexplored_line_above_a_solution() -> None:
    """The lever's central job, on a real search.

    The episode is already at par, one step from solved. The solving move now
    lands over par, so it is worth -1 exactly. Under ``'evaluate'`` every move
    that does NOT solve is handed to the evaluator, which (value head silent)
    says 0: the search ranks dawdling above solving. Under ``'loss'`` nothing
    outranks the solution.
    """
    problem, solver = one_step_from_solved_with_a_way_to_dawdle()

    merged = search(
        problem,
        problem.expr,
        uniform_stub(CFG),
        with_search(),
        random.Random(0),
        steps_taken=problem.par,
    )
    assert merged.values[solver] == -1.0
    others = [v for k, v in enumerate(merged.values) if k != solver and merged.visits[k] > 0]
    assert others and max(others) == 0.0, "the defect this lever exists for is not present"
    assert merged.chosen != merged.actions[solver], "the merged search chose to solve after all"

    fixed = search(
        problem,
        problem.expr,
        uniform_stub(CFG),
        with_search(past_par="loss"),
        random.Random(0),
        steps_taken=problem.par,
    )
    visited = [v for k, v in enumerate(fixed.values) if fixed.visits[k] > 0]
    assert max(visited) == -1.0 == fixed.values[solver]
    assert fixed.stats.past_par > 0
    assert descent_identity_holds(fixed) and descent_identity_holds(merged)


@needs_suites
def test_past_par_loss_changes_nothing_before_par() -> None:
    """From a start state with the full par budget ahead, a search that never
    reaches par is the same search under both rules."""
    problem = suite(6)[0]
    kwargs = {"sims": 8, "m": 4}
    a = search(problem, problem.expr, uniform_stub(CFG), with_search(), random.Random(0), **kwargs)
    b = search(
        problem,
        problem.expr,
        uniform_stub(CFG),
        with_search(past_par="loss"),
        random.Random(0),
        **kwargs,
    )
    assert a.stats.max_depth < problem.par, "the fixture reached par; pick a deeper problem"
    assert np.array_equal(a.visits, b.visits) and np.array_equal(a.values, b.values)
    assert np.array_equal(a.improved, b.improved) and a.chosen == b.chosen


# ------------------------------------------------------------ value source


@needs_suites
def test_state_value_is_the_par_games_arithmetic() -> None:
    bfs = replace(suite(5)[0], par=5, par_source="bfs")
    pool = replace(bfs, par_source="pool")
    steps = with_search(value_source="steps")
    prior = np.zeros(1, dtype=np.float32)

    # 'wdl': the evaluator's own value, whatever else the answer carries.
    assert _state_value((prior, 0.25), bfs, 2, with_search()) == 0.25
    assert _state_value((prior, 0.25, 99.0), bfs, 2, with_search()) == 0.25

    # 'steps', against exact par: 2 steps in, 3 left to stay level.
    assert _state_value((prior, 0.9, 3.0), bfs, 2, steps) == 0.0  # exactly on par
    assert _state_value((prior, 0.9, 3.5), bfs, 2, steps) == -0.5  # half a step over
    assert _state_value((prior, 0.9, 4.0), bfs, 2, steps) == -1.0  # a step over
    assert _state_value((prior, 0.9, 9.0), bfs, 2, steps) == -1.0  # floored
    assert _state_value((prior, 0.9, 1.0), bfs, 2, steps) == 0.0  # "under exact par": capped
    # Against a par that can be beaten, finishing under it is worth up to +1.
    assert _state_value((prior, 0.9, 2.5), pool, 2, steps) == 0.5
    assert _state_value((prior, 0.9, 1.0), pool, 2, steps) == 1.0

    with pytest.raises(ValueError, match="no steps-to-go"):
        _state_value((prior, 0.0), bfs, 2, steps)


@needs_suites
def test_the_uniform_stub_is_refused_under_steps_rather_than_read_as_zero() -> None:
    problem = suite(2)[0]
    with pytest.raises(ValueError, match="no steps-to-go"):
        search(
            problem,
            problem.expr,
            uniform_stub(CFG),
            with_search(value_source="steps"),
            random.Random(0),
        )


@needs_suites
def test_the_evaluator_reports_the_steps_head_only_when_asked() -> None:
    torch.manual_seed(0)
    model = Reckoner(CFG).eval()
    leaves = [(p, p.expr) for p in suite(3)[:5]]
    plain = model_evaluator(model, CFG, 1.0)(leaves)
    asked = model_evaluator(model, with_search(value_source="steps"), 1.0)(leaves)

    encoded = [encode(problem, expr, CFG) for problem, expr in leaves]
    with torch.no_grad():
        _, _, steps = model(
            torch.stack([e.tokens for e in encoded]),
            torch.stack([e.site_positions for e in encoded]),
        )
    for (prior, value), (prior_s, value_s, to_go), expected in zip(
        plain, asked, steps.tolist(), strict=True
    ):
        assert np.array_equal(prior, prior_s) and value == value_s
        assert to_go == pytest.approx(expected, abs=1e-6)


@needs_suites
def test_value_source_steps_refutes_a_bad_move_at_first_sight() -> None:
    """The lever's central job, with a true cost-to-go standing in for the head.

    The evaluator here answers every state with its exact BFS distance. Under
    ``'steps'`` each root move is then worth exactly what it is: 0 if it keeps
    the problem on par, -1 if it does not, after ONE visit. Under ``'wdl'`` the
    same evaluator's value slot is 0 (a silent head), and a move that throws par
    away still reads 0 unless the search exhausts everything beneath it.
    """
    rng = random.Random(0)
    problem = next(
        p
        for p in suite(3)
        if 2 <= len(legal_actions(p.expr)) <= 8
        and any(
            bfs_par(replace(p, expr=apply(p.expr, r, s)), CFG, cap=2) is None
            for r, s in legal_actions(p.expr)
        )
    )
    width = 7 * CFG.model.max_sites

    def oracle(leaves):
        out = []
        for leaf_problem, expr in leaves:
            if verify(leaf_problem, expr, CFG, rng):
                distance = 0
            else:
                found = bfs_par(replace(leaf_problem, expr=expr), CFG, cap=6)
                distance = 24 if found is None else found
            out.append((np.zeros(width, dtype=np.float32), 0.0, float(distance)))
        return out

    actions = legal_actions(problem.expr)
    keeps_par = [
        bfs_par(replace(problem, expr=apply(problem.expr, r, s)), CFG, cap=2) is not None
        for r, s in actions
    ]
    assert any(keeps_par) and not all(keeps_par)

    budget = {"sims": len(actions), "m": len(actions)}  # one visit per root move
    informed = search(
        problem, problem.expr, oracle, with_search(value_source="steps"), random.Random(0), **budget
    )
    for slot, good in enumerate(keeps_par):
        assert informed.visits[slot] == 1
        assert informed.values[slot] == (0.0 if good else -1.0), (slot, good, informed.values)

    silent = search(problem, problem.expr, oracle, with_search(), random.Random(0), **budget)
    assert all(v == 0.0 for v in silent.values), "a silent value head should not tell them apart"
    assert descent_identity_holds(informed) and descent_identity_holds(silent)


# ------------------------------------------------------------- the round


def test_the_w1_configs_are_the_brief() -> None:
    """WEAK-w1.md registers one set of benchmark settings and two levers. The arm
    files' diffs are exactly those, so a second lever added quietly fails here,
    by name (AGENTS §6)."""
    from reckoner.config import config_diff, load_config

    a, b, v = (load_config(REPO / "configs" / f"w1_{arm}.yaml") for arm in "abv")
    for cfg in (a, b, v):
        validate(cfg)
    assert config_diff(Config(), a) == {
        "train.rehearsal_frac": (0.65, 0.0),
        "train.train_steps_per_iter": (400, 200),
        "train.lr_warmup_steps": (200, 100),
        "train.replay_capacity": (500_000, 60_000),
        "campaign.iterations": (20, 10),
        "campaign.episodes_per_iteration": (400, 200),
        "ladder.ladder_every": (5, 99),
    }
    assert config_diff(a, b) == {"search.past_par": ("evaluate", "loss")}
    assert config_diff(b, v) == {"search.value_source": ("wdl", "steps")}


def test_the_w1_readings_count_at_par_and_pair_by_problem() -> None:
    """``collect``'s arithmetic on rows small enough to check by hand."""
    import weak_w1

    def rows(outcomes: dict[str, int]) -> list[dict]:
        return [
            {
                "arm": f"model@{sims}",
                "z": z,
                "par": int(key[1]),
                "steps": 3 if z == 0 else 24,
                "problem_key": key,
            }
            for sims in weak_w1.BUDGETS
            for key, z in outcomes.items()
        ]

    x = weak_w1.at_par_table(rows({"d1-a": 0, "d1-b": 0, "d5-a": 0, "d5-b": -1}))
    y = weak_w1.at_par_table(rows({"d1-a": 0, "d1-b": -1, "d5-a": -1, "d5-b": -1}))
    assert x["at_par"]["1"]["1"] == 2 and x["at_par"]["1"]["5"] == 1
    assert x["solved"]["48"]["5"] == 1 and y["at_par"]["48"]["5"] == 0
    everything = weak_w1._paired([x], [y], 1, weak_w1.DEPTHS, resamples=200)
    assert everything["pairs"] == 4 and everything["mean_difference"] == pytest.approx(0.5)
    deep = weak_w1._paired([x], [y], 1, (5, 6), resamples=200)
    assert deep["pairs"] == 2 and deep["mean_difference"] == pytest.approx(0.5)
    same = weak_w1._paired([x], [x], 1, weak_w1.DEPTHS, resamples=200)
    assert same["mean_difference"] == 0.0 and not same["excludes_zero"]
