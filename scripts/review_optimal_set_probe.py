"""REVIEW-2026-11-15's second probe: optimal action SETS, the policy, and the steps head.

On frozen suite start states with exact BFS distance d (suites ``solve_in_1..3``,
60 each via ``sample_indices``), every legal action is labelled by the exact BFS
distance of the state it leads to. An action is optimal when that distance is
``d - 1``. The probe then asks three things of the Phase-1 anchor:

1. how many states have more than one optimal action (the single BFS label that
   gate 10b scores against is then one of several right answers);
2. whether the policy's top action is in the optimal set;
3. whether the steps head, which the search never reads, ranks the children
   correctly: is the child with the lowest predicted steps-to-solve optimal, and
   how far are its predictions from the exact distances.

A child further than ``d + 1`` is counted, not measured: BFS stops there.

A probe for a review, not a registered reading: 180 states, one seed.

    python scripts/review_optimal_set_probe.py --out runs/review_20261115/optimal_set_probe_result.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from reckoner.campaign import ANCHOR, SUITES
from reckoner.config import Config
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.episode import Problem, bfs_par, verify
from reckoner.model import StateTooLarge, encode, load_checkpoint
from reckoner.rules import apply, legal_actions

REPO = Path(__file__).resolve().parents[1]
DEPTHS = (1, 2, 3)
PER_SUITE = 60
BEYOND = 99


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    torch.set_num_threads(4)
    model, _ = load_checkpoint(ANCHOR, cfg)
    model.eval()
    rng = random.Random(0)

    rows = []
    for depth in DEPTHS:
        suite = read_suite(SUITES / f"solve_in_{depth}.jsonl")
        for i in sample_indices(len(suite), PER_SUITE, 0):
            problem = suite_problem(suite[i])
            actions = legal_actions(problem.expr)
            children, distances = [], []
            for rule_id, site_id in actions:
                child = apply(problem.expr, rule_id, site_id)
                children.append(child)
                posed = Problem(
                    goal=problem.goal,
                    expr=child,
                    par=problem.par,
                    target=problem.target,
                    par_source="bfs",
                )
                if verify(posed, child, cfg, rng):
                    distances.append(0)
                else:
                    found = bfs_par(posed, cfg, cap=depth + 1)
                    distances.append(BEYOND if found is None else found)
            try:
                encoded = [encode(problem, expr, cfg) for expr in [problem.expr, *children]]
            except StateTooLarge:
                continue
            with torch.no_grad():
                policy, _, steps = model(
                    torch.stack([e.tokens for e in encoded]),
                    torch.stack([e.site_positions for e in encoded]),
                )
            best = min(distances)
            optimal = [k for k, dist in enumerate(distances) if dist == best]
            masked = policy[0].masked_fill(~encoded[0].legal_mask, float("-inf"))
            dense = [rule_id * cfg.model.max_sites + site_id for rule_id, site_id in actions]
            top = dense.index(int(masked.argmax()))
            predicted = steps[1:].numpy()
            known = [k for k, dist in enumerate(distances) if dist < BEYOND]
            rows.append(
                {
                    "depth": depth,
                    "bfs_consistent": best == depth - 1,
                    "legal_actions": len(actions),
                    "optimal_actions": len(optimal),
                    "policy_top_is_optimal": top in optimal,
                    "steps_argmin_is_optimal": int(np.argmin(predicted)) in optimal,
                    "steps_error_root": abs(float(steps[0]) - depth),
                    "steps_error_children": [
                        abs(float(predicted[k]) - distances[k]) for k in known
                    ],
                    "children_beyond_cap": sum(dist == BEYOND for dist in distances),
                }
            )

    def summarise(group: list[dict]) -> dict:
        errors = [e for r in group for e in r["steps_error_children"]]
        return {
            "states": len(group),
            "bfs_consistent": sum(r["bfs_consistent"] for r in group),
            "legal_actions_mean": round(float(np.mean([r["legal_actions"] for r in group])), 2),
            "states_with_several_optimal_actions": sum(r["optimal_actions"] > 1 for r in group),
            "policy_top_is_optimal": sum(r["policy_top_is_optimal"] for r in group),
            "steps_argmin_is_optimal": sum(r["steps_argmin_is_optimal"] for r in group),
            "steps_abs_error_root": round(
                float(np.mean([r["steps_error_root"] for r in group])), 3
            ),
            "steps_abs_error_children": round(float(np.mean(errors)), 3),
            "children_beyond_cap": sum(r["children_beyond_cap"] for r in group),
        }

    report = {
        "git_sha": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "per_suite": PER_SUITE,
        "by_depth": {str(d): summarise([r for r in rows if r["depth"] == d]) for d in DEPTHS},
        "all": summarise(rows),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
