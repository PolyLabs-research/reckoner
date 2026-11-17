"""Which problems does each dataset hold, by depth and goal? (F-60)

A census, not a measurement of any network: exact counts over every row of the
frozen datasets and suites, reproducible from the anchored bytes.

It exists because a from-scratch exploration (WEAK-w1.md, "Exploration (not
evidence)") left a network unable to play about half of ``solve_in_6`` however
long it trained, and the reason turned out to be in the data. ``train_100k``
holds no depth-6 SOLVE problem, and ``solve_in_6`` is 95 of them in 200.

What it counts:

* every dataset and suite by ``(par, goal)``;
* the SOLVE problems of the ``solve_both_sides_product`` shape (x on both sides
  and a product of two numerals still to evaluate), at any depth: the template
  ``scripts/generate.py`` aims at depth 6;
* where in ``train_100k``'s stored order each ``(par, goal)`` first and last
  appears, and what the final rows are. The generator writes rows in the order of
  its work plan, which goes depth by depth, and keeps the first ``count`` of them.

It also re-reads the per-problem records WEAK-w1 already holds, by ``(par,
goal)``: how each recorded network did on the stratum it never practised. Those
rows are registered; the split is made here, after the fact.

    python scripts/strata_census.py            # writes runs/strata_census_result.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from reckoner.config import Config
from reckoner.dataset import (
    anchored_data,
    git_sha,
    read_anchors,
    read_dataset,
    read_suite,
    write_record,
)
from reckoner.episode import decode_state
from reckoner.expr import Num, Op, Var
from reckoner.train import SupervisionSet
from reckoner.vocab import EQ, GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE, MUL

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
GOAL = {GOAL_EVALUATE: "EVALUATE", GOAL_SOLVE: "SOLVE", GOAL_SIMPLIFY: "SIMPLIFY"}
#: ``scripts/generate.py``: the plan over-requests by this factor, then keeps the first ``count``.
OVER_REQUEST = 1.6


def _has_var(expr) -> bool:
    if isinstance(expr, Var):
        return True
    return isinstance(expr, Op) and any(_has_var(child) for child in expr.children)


def _has_numeral_product(expr) -> bool:
    if not isinstance(expr, Op):
        return False
    if expr.kind == MUL and all(isinstance(child, Num) for child in expr.children):
        return True
    return any(_has_numeral_product(child) for child in expr.children)


def is_both_sides_product(tokens: tuple[int, ...]) -> bool:
    """A SOLVE problem with x on both sides and a product of numerals to evaluate.

    The shape ``solve_both_sides_product`` emits and no other v1 template does:
    ``solve_both_sides`` has x on both sides and no such product, and
    ``solve_product_rhs`` has the product and x on one side only.
    """
    goal, _, expr = decode_state(tokens)
    if goal != GOAL_SOLVE or not isinstance(expr, Op) or expr.kind != EQ:
        return False
    left, right = expr.children
    return _has_var(left) and _has_var(right) and _has_numeral_product(expr)


def by_par_and_goal(par: np.ndarray, goal: np.ndarray) -> dict[str, dict[str, int]]:
    counts: Counter = Counter(zip(par.tolist(), goal.tolist(), strict=True))
    out: dict[str, dict[str, int]] = {}
    for (p, g), n in sorted(counts.items()):
        out.setdefault(str(p), {})[GOAL[g]] = n
    return out


def census_dataset(name: str) -> dict:
    dataset = read_dataset(anchored_data(name))
    par, goal = np.asarray(dataset.par), np.asarray(dataset.goal)
    solve_rows = np.nonzero(goal == GOAL_SOLVE)[0]
    shaped = [int(i) for i in solve_rows if is_both_sides_product(dataset.state(int(i)))]
    out = {
        "rows": len(par),
        "by_par_and_goal": by_par_and_goal(par, goal),
        "both_sides_product_shape": {
            "rows": len(shaped),
            "by_par": {str(p): n for p, n in sorted(Counter(par[shaped].tolist()).items())},
        },
    }
    # The stored order is the generator's plan order.
    first_last = {}
    for (p, g), _ in sorted(Counter(zip(par.tolist(), goal.tolist(), strict=True)).items()):
        rows = np.nonzero((par == p) & (goal == g))[0]
        first_last[f"par {p} {GOAL[g]}"] = [int(rows[0]), int(rows[-1])]
    tail = slice(len(par) - 1000, len(par))
    out["stored_order"] = {
        "first_and_last_row": first_last,
        "last_1000_rows": by_par_and_goal(par[tail], goal[tail]),
    }
    return out


def census_suite(path: Path) -> dict:
    rows = read_suite(path)
    par = np.array([row["par"] for row in rows])
    goal = np.array([row["goal"] for row in rows])
    shaped = [row for row in rows if is_both_sides_product(tuple(row["tokens"]))]
    return {
        "rows": len(rows),
        "par_source": sorted({row["par_source"] for row in rows}),
        "by_par_and_goal": by_par_and_goal(par, goal),
        "both_sides_product_shape": len(shaped),
    }


def census_supervision(name: str) -> dict:
    """``phase1_train`` / ``phase1_eval`` hold STATES along solutions; count their problems."""
    data = SupervisionSet(anchored_data(name))
    depth = np.asarray(data.depth)
    starts = np.nonzero(depth == np.asarray(data.steps_remaining))[0]
    goal = np.asarray(data.goal)
    return {
        "rows": len(data),
        "problems": len(starts),
        "problems_by_depth_and_goal": by_par_and_goal(depth[starts], goal[starts]),
    }


def at_par_by_stratum(pair_scores: Path) -> dict[str, dict[str, dict[str, list[int]]]]:
    """``{arm: {par: {goal: [at par, problems]}}}`` from one pass's per-problem rows.

    The goal is the first token of the row's ``problem_key``, which is the
    problem's encoded state.
    """
    out: dict[str, dict[str, dict[str, list[int]]]] = {}
    for line in pair_scores.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        goal = GOAL[int(row["problem_key"].split(",")[0])]
        cell = (
            out.setdefault(row["arm"], {}).setdefault(str(row["par"]), {}).setdefault(goal, [0, 0])
        )
        cell[0] += int(int(row["z"]) >= 0)
        cell[1] += 1
    return out


def weak_w1_readings() -> dict[str, dict]:
    """Every WEAK-w1 reading on record, by stratum. Empty where the records are absent."""
    root = REPO / "runs" / "w1"
    passes = {
        "reference (the Phase-1 anchor)": root / "reference" / "pass-0000",
        **{
            f"start network, arm {arm}'s search": root / "baseline" / arm / "pass-0000"
            for arm in "ABV"
        },
        **{
            f"{arm}-{seed}": root / f"{arm}-{seed}" / "ladder" / "w1" / "pass-0009"
            for arm in "ABV"
            for seed in (42, 43, 44)
        },
    }
    return {
        name: at_par_by_stratum(path / "pair_scores.jsonl")
        for name, path in passes.items()
        if (path / "pair_scores.jsonl").exists()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "strata_census_result.json")
    args = parser.parse_args()

    cfg = Config()
    requested = int(cfg.generator.train_set_size * OVER_REQUEST) + 64
    report = {
        "git_sha": git_sha(REPO),
        "anchors_sha256_entries": len(read_anchors()),
        "generator_plan": {
            "train_set_size": cfg.generator.train_set_size,
            "candidates_requested": requested,
            "candidates_per_depth": requested // 6,
            "order": "depth 1's candidates first, depth 6's last; the first train_set_size "
            "problems that survive labelling and de-duplication are kept",
            "depth_6_candidates_begin_at": 5 * (requested // 6),
        },
        "datasets": {name: census_dataset(name) for name in ("train_100k", "eval_held_out")},
        "supervision": {name: census_supervision(name) for name in ("phase1_train", "phase1_eval")},
        "suites": {path.stem: census_suite(path) for path in sorted(SUITES.glob("*.jsonl"))},
        "weak_w1_at_par_by_stratum": weak_w1_readings(),
    }
    write_record(args.out, report)

    train = report["datasets"]["train_100k"]
    print("  train_100k by par and goal:")
    for par, goals in train["by_par_and_goal"].items():
        print(f"    par {par}: {goals}")
    print(f"  train_100k, both-sides-product shape: {train['both_sides_product_shape']}")
    print(f"  train_100k, last 1,000 rows: {train['stored_order']['last_1000_rows']}")
    for name in ("solve_in_5", "solve_in_6"):
        suite = report["suites"][name]
        print(
            f"  {name}: {suite['by_par_and_goal']}  both-sides-product shape "
            f"{suite['both_sides_product_shape']}"
        )
    print(
        "  WEAK-w1's records, at par on the 95 depth-6 SOLVE problems (policy alone / with search):"
    )
    for name, table in report["weak_w1_at_par_by_stratum"].items():
        alone = table["model@1"]["6"]["SOLVE"]
        searched = table["model@48"]["6"]["SOLVE"]
        print(f"    {name:<34} {alone[0]:>3} / {searched[0]:>3}   of {alone[1]}")
    print(f"  wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
