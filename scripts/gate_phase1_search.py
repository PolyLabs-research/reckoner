"""Gates 11-13: does the warm-started net actually solve, under real search?

Every row here carries rider (c)'s **four-tuple** — floor, null-model baseline,
threshold, measured value — because a solve-rate number without its null is the
F-10 defect wearing a different metric. The null is not hypothetical: it is the
uniform-prior stub at the same budget, run on the same problems, which is the
chess stub-baseline precedent applied here as law.

Three rows per suite:

* **trained**    — the Phase-1 net's priors and value
* **stub**       — ``uniform_stub``, flat priors and neutral value (the null)
* **zero-value** — the net's priors with value forced to 0, the ablation. The
  W/D/L head was trained at loss weight **0** by design, so its output is an
  untrained head's noise. This row measures whether that noise is helping,
  hurting, or inert in backup, which is a diagnostic the gate needs and the
  gate's own number cannot supply.

Solving means *playing the episode* — search, take the chosen action, apply,
repeat until solved or the step budget is spent. It does not mean "a solve
appeared somewhere in the tree", which is a weaker claim wearing the same word.

Writes ``runs/gate_phase1_search.json``.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from reckoner.config import Config, with_m1_search
from reckoner.dataset import git_sha, read_suite, suite_problem, write_record
from reckoner.episode import Problem, verify
from reckoner.expr import Expr
from reckoner.model import Reckoner, StateTooLarge, encode
from reckoner.rules import apply, legal_actions
from reckoner.search import Evaluation, Evaluator, search, uniform_stub

REPO = Path(__file__).resolve().parents[1]


def model_evaluator(
    model: Reckoner, cfg: Config, *, zero_value: bool = False, value_scale: float | None = None
) -> Evaluator:
    """Batch the trained net behind the search's evaluator contract.

    ``StateTooLarge`` returns flat priors and a neutral value rather than
    raising: chunk 7 pinned oversized states as a counted terminal loss inside
    the search, and an evaluator that raises would turn that into a crash
    mid-batch. Legality stays the engine's alone — this returns priors over the
    whole action layout and never touches a mask.
    """
    width = 7 * cfg.model.max_sites

    def evaluate(leaves: Sequence[tuple[Problem, Expr]]) -> list[Evaluation]:
        encoded, keep = [], []
        for i, (problem, expr) in enumerate(leaves):
            try:
                encoded.append(encode(problem, expr, cfg))
                keep.append(i)
            except StateTooLarge:
                continue
        out: list[Evaluation] = [(np.zeros(width, dtype=np.float32), 0.0) for _ in leaves]
        if not encoded:
            return out
        tokens = torch.stack([e.tokens for e in encoded])
        sites = torch.stack([e.site_positions for e in encoded])
        with torch.no_grad():
            policy, value_logits, _steps = model(tokens, sites)
            probs = torch.softmax(value_logits, dim=1)
            # W/D/L vs par -> expected z in [-1, +1]. Column order is the head's:
            # 0 = under par (+1), 1 = equal (0), 2 = over par (-1).
            expected = (probs[:, 0] - probs[:, 2]).tolist()
        # The declaration, applied. `value_scale` comes from
        # valuegate.value_contribution(state) — the SAME function the loss reads,
        # so an untrained head cannot be silent in one consumer and loud in the
        # other. `zero_value` remains the explicit ablation lever.
        scale = 0.0 if zero_value else (1.0 if value_scale is None else value_scale)
        for slot, i in enumerate(keep):
            out[i] = (
                policy[slot].numpy().astype(np.float32),
                scale * float(expected[slot]),
            )
        return out

    return evaluate


def play(
    problem: Problem,
    evaluator: Evaluator,
    cfg: Config,
    *,
    sims: int,
    m: int,
    budget: int,
    seed: int,
) -> tuple[bool, int]:
    """Search-guided play. Returns ``(solved, steps_taken)``.

    **The search rng is seeded per (problem, step), never re-seeded to a constant.**
    A fresh ``Random(0)`` for every search makes the root Gumbel draw a function of
    the action *count* alone, so every 5-action problem in the suite considers the
    same slots and chooses the same one — measured: chosen slot 0 on all six probe
    problems, visits ``[8, 8, 0, 0, 0]`` every time. Against flat stub priors that
    turns the null from "uniform-random action" into "always the first legal
    action", which is a different and weaker null wearing the same name. Caught by
    the inherited law: two ``m`` values returned byte-identical rates.

    ``verify`` takes its own rng (SIMPLIFY equivalence is random-assignment with
    k draws), seeded per problem so both arms are judged by the same draws.
    """
    expr = problem.expr
    checker = random.Random(12345)
    for step in range(budget):
        if not legal_actions(expr):
            return False, step
        rng = random.Random(seed * 10_000 + step)
        result = search(problem, expr, evaluator, cfg, rng, sims=sims, m=m)
        if result.chosen is None:
            return False, step
        rule_id, site_id = result.chosen
        expr = apply(expr, rule_id, site_id)
        if verify(problem, expr, cfg, checker):
            return True, step + 1
    return False, budget


def run_suite(
    rows: list[dict], evaluator: Evaluator, cfg: Config, *, sims: int, m: int, budget_mult: int
) -> dict:
    solved = 0
    steps: list[int] = []
    for index, row in enumerate(rows):
        problem = suite_problem(row)
        budget = min(cfg.episode.step_cap, budget_mult * problem.par + 2)
        ok, took = play(problem, evaluator, cfg, sims=sims, m=m, budget=budget, seed=1000 + index)
        solved += int(ok)
        if ok:
            steps.append(took)
    return {
        "n": len(rows),
        "solved": solved,
        "rate": round(solved / len(rows), 4),
        "median_steps_when_solved": int(np.median(steps)) if steps else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs" / "phase1" / "phase1.pt")
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--budget-mult", type=int, default=2)
    parser.add_argument("--out", default="gate_phase1_search")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # The gate's three evaluators answer (policy, value) and its recorded numbers
    # were taken under the search that reads that value: the default until 2026-11-16.
    cfg = with_m1_search(Config())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Reckoner(cfg)
    model.load_state_dict(state["state_dict"])
    model.eval()

    arms = {
        "trained": model_evaluator(model, cfg),
        "stub_null": uniform_stub(cfg),
        "zero_value": model_evaluator(model, cfg, zero_value=True),
    }

    print(f"  GATE 11-13 — search-guided solve, sims={args.sims}, m={args.m}")
    print(f"  step budget = min(step_cap, {args.budget_mult} x par + 2), declared\n")

    out: dict = {
        "sims": args.sims,
        "m": args.m,
        "budget_mult": args.budget_mult,
        "checkpoint_step": state["meta"]["step"],
        "git_sha": git_sha(REPO),
        "suites": {},
    }

    for depth in args.depths:
        rows = read_suite(REPO / "runs" / "suites" / f"solve_in_{depth}.jsonl")
        if args.limit:
            rows = rows[: args.limit]
        print(f"  solve_in_{depth}  (n={len(rows)})")
        print(f"    {'arm':>12} {'solved':>8} {'rate':>8} {'median steps':>13}")
        entry = {}
        for name, evaluator in arms.items():
            res = run_suite(
                rows, evaluator, cfg, sims=args.sims, m=args.m, budget_mult=args.budget_mult
            )
            entry[name] = res
            print(
                f"    {name:>12} {res['solved']:>8} {res['rate']:>8.4f} "
                f"{str(res['median_steps_when_solved']):>13}"
            )
        out["suites"][f"solve_in_{depth}"] = entry
        print()

    # Gate 11's four-tuple, over the depth<=2 union.
    if set(args.depths) >= {1, 2}:
        tot = sum(out["suites"][f"solve_in_{d}"]["trained"]["n"] for d in (1, 2))
        tr = sum(out["suites"][f"solve_in_{d}"]["trained"]["solved"] for d in (1, 2))
        nl = sum(out["suites"][f"solve_in_{d}"]["stub_null"]["solved"] for d in (1, 2))
        zv = sum(out["suites"][f"solve_in_{d}"]["zero_value"]["solved"] for d in (1, 2))
        gate = {
            "metric": "depth<=2 search-guided solve rate",
            "floor": 0.0,
            "null_stub": round(nl / tot, 4),
            "threshold": 0.95,
            "measured": round(tr / tot, 4),
            "zero_value_ablation": round(zv / tot, 4),
            "n": tot,
            "verdict": "PASS" if tr / tot >= 0.95 else "SHORT",
        }
        out["gate_11"] = gate
        print("  GATE 11 four-tuple (rider (c)):")
        print(
            f"    floor      {gate['floor']:.4f}   (no state solves itself; a wrong move is a miss)"
        )
        print(f"    null stub  {gate['null_stub']:.4f}")
        print(f"    threshold  {gate['threshold']:.4f}")
        print(f"    measured   {gate['measured']:.4f}   -> {gate['verdict']}")
        print(f"    ablation   {gate['zero_value_ablation']:.4f}  (value head forced to 0)")

    write_record(REPO / "runs" / f"{args.out}.json", out)
    print(f"\n  wrote runs/{args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
