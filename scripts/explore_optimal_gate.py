"""EXPLORATION (AGENTS §7.1) — gate 10b's own states, scored against optimal SETS.

**Not evidence.** Gate 10b counts a state as a hit when the policy's top action is
the one BFS happened to return. Where several actions are optimal, a different
optimal action is counted as a miss. This script takes the same population (the
held-out supervision rows from problems of depth <= 3, minus the rows that also
occur in training: 1,229 states) and labels every legal action exactly:

* a row's ``steps_remaining`` is its exact distance to a solution, because it
  lies on a BFS-optimal path;
* an action is optimal when the state it leads to is solved, or is solvable in
  ``steps_remaining - 1`` steps (BFS with that cap).

It reports the single-label hits (which must reproduce the gate's own count), the
optimal-set hits, and what the single-label misses actually are.

    python scripts/explore_optimal_gate.py --checkpoint runs/phase1/phase1.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from train_phase1 import shared_state_indices

from reckoner.config import Config
from reckoner.dataset import anchored_data
from reckoner.episode import Problem, bfs_par, decode_state, verify
from reckoner.model import load_checkpoint
from reckoner.rules import apply, legal_actions
from reckoner.train import SupervisionSet, make_batch

REPO = Path(__file__).resolve().parents[1]


def optimal_actions(
    tokens: tuple[int, ...], distance: int, cfg: Config, rng: random.Random
) -> list[int]:
    goal, target, expr = decode_state(tokens)
    out = []
    for rule_id, site_id in legal_actions(expr):
        child = apply(expr, rule_id, site_id)
        posed = Problem(goal=goal, expr=child, par=distance, target=target, par_source="bfs")
        if verify(posed, child, cfg, rng):
            reach = True
        elif distance <= 1:
            reach = False
        else:
            reach = bfs_par(posed, cfg, cap=distance - 1) is not None
        if reach:
            out.append(rule_id * cfg.model.max_sites + site_id)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs" / "phase1" / "phase1.pt")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = Config()
    eval_path = anchored_data("phase1_eval")
    held = SupervisionSet(eval_path)
    shared = shared_state_indices(eval_path)
    rows = [i for i in range(len(held)) if i not in shared and int(held.depth[i]) <= 3]

    model, _ = load_checkpoint(args.checkpoint, cfg)
    model.eval()
    top: list[int] = []
    label: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = make_batch(held, rows[start : start + 256], cfg)
            policy, _, _ = model(batch.tokens, batch.site_positions)
            masked = policy.masked_fill(~batch.legal_mask, float("-inf"))
            top += masked.argmax(dim=1).tolist()
            label += batch.policy_target.argmax(dim=1).tolist()

    rng = random.Random(0)
    several = label_hits = optimal_hits = label_not_optimal = 0
    miss_but_optimal = miss_and_wrong = 0
    for i, t, bfs_action in zip(rows, top, label, strict=True):
        tokens = tuple(int(x) for x in held.tokens[i, : held.lengths[i]])
        optimal = optimal_actions(tokens, int(held.steps_remaining[i]), cfg, rng)
        several += len(optimal) > 1
        label_not_optimal += bfs_action not in optimal
        label_hits += t == bfs_action
        optimal_hits += t in optimal
        if t != bfs_action:
            if t in optimal:
                miss_but_optimal += 1
            else:
                miss_and_wrong += 1

    report = {
        "checkpoint": str(args.checkpoint),
        "states": len(rows),
        "states_with_several_optimal_actions": several,
        "bfs_label_not_in_optimal_set": label_not_optimal,
        "single_label_hits": label_hits,
        "optimal_set_hits": optimal_hits,
        "single_label_misses_that_are_optimal": miss_but_optimal,
        "single_label_misses_that_are_wrong": miss_and_wrong,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
