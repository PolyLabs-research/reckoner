"""F-54's instrument: what the Phase-2 policy target carries, and what it cannot.

The ring trains the policy toward ``SearchResult``'s root distribution. Before
F-54 that was the **Sequential-Halving visit share**: ``visits / visits.sum()``.
The spec promises an *improved-policy* target (spec §1 thesis, §5 Phase 2), the
Gumbel π′ = softmax(logits + σ(completed Q)), and the method that returned the
visit share was named ``improved_policy``.

Three readings, each run against three evaluators (the uniform stub, and a fixed
random-init network with its value contribution at 0.0 and at 1.0), over frozen
suite start states (``sample_indices``, never a prefix):

1. **Schedule census.** The sorted visit profile, grouped by the number of
   actions the root considered. If every group holds one profile whatever the
   evaluator, the target's *shape* is fixed by Sequential Halving's schedule and
   only *which action* gets which share carries search information.
2. **Ceiling.** The largest share any target places on one action.
3. **Proven-worse mass.** Root actions whose child is a *proof* (a terminal, so
   its value is an exact z) are compared: target mass on an action proven
   strictly worse than another visited, proven action. A target that has read
   the search puts ~0 there.

When ``SearchResult`` carries a π′ (``improved``), readings 2 and 3 are repeated
for it, beside the visit share, so the same probe reads both sides of the fix.

    python scripts/f54_policy_target_probe.py --per-suite 20 --out /tmp/f54.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import reckoner
from reckoner.config import Config
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.evaluate import model_evaluator
from reckoner.model import Reckoner
from reckoner.rules import apply
from reckoner.search import SearchStats, _leaf_outcome, run_batched, search, uniform_stub

REPO = Path(__file__).resolve().parents[1]
SUITES = sorted((REPO / "runs" / "suites").glob("*.jsonl"))


def _proven_children(result, problem, cfg: Config, steps_taken: int = 0) -> dict[int, float]:
    """Visited root slots whose child is a terminal proof, with its exact z.

    Recomputed with the search's own terminal oracle (``_leaf_outcome``) rather
    than read from the result, so the probe needs nothing the pre-fix search did
    not expose. Cross-checked: the search must have stored that same exact z.
    """
    out = {}
    for slot in np.flatnonzero(result.visits):
        successor = apply(problem.expr, *result.actions[int(slot)])
        z = _leaf_outcome(
            problem, successor, cfg, random.Random(0), SearchStats(), total_steps=steps_taken + 1
        )
        if z is None:
            continue
        if float(result.values[slot]) != z:
            raise RuntimeError(f"stored value {result.values[slot]} != exact z {z} at slot {slot}")
        out[int(slot)] = z
    return out


def _read(results, problems, cfg: Config, distribution) -> dict:
    """Ceiling and proven-worse mass of one target distribution over many roots."""
    tops, worse_mass, states_with_worse = [], [], 0
    for result, problem in zip(results, problems, strict=True):
        if len(result.actions) == 0:
            continue
        p = distribution(result)
        tops.append(float(p.max()))
        proofs = _proven_children(result, problem, cfg)
        if len(proofs) >= 2:
            best = max(proofs.values())
            worse = [s for s, z in proofs.items() if z < best]
            if worse:
                states_with_worse += 1
                worse_mass.append(float(p[worse].sum()))
    return {
        "roots": len(tops),
        "max_share_mean": round(float(np.mean(tops)), 4),
        "max_share_max": round(float(np.max(tops)), 4),
        "roots_with_a_proven_worse_action": states_with_worse,
        "mass_on_proven_worse_mean": round(float(np.mean(worse_mass)), 6) if worse_mass else None,
        "mass_on_proven_worse_max": round(float(np.max(worse_mass)), 6) if worse_mass else None,
    }


def cap_construction(cfg: Config) -> dict:
    """**Constructed** proofs, because start states almost never have two.

    Every five-action ``solve_in_1`` start state, re-posed one step before the
    step cap with par set to the cap: the one solving action is then an exact
    at-par solve (z = 0) and each of the other four runs into the cap (z = -1).
    Every root action is a proof, so nothing about the value head enters. Root
    noise off, uniform stub, so the search is deterministic and the only
    information in play is the proofs themselves.
    """
    cap = cfg.episode.step_cap
    quiet = replace(cfg, search=replace(cfg.search, root_noise=False))
    shares, improved, chosen_solver, states = [], [], 0, 0
    for row in read_suite(REPO / "runs" / "suites" / "solve_in_1.jsonl"):
        problem = replace(suite_problem(row), par=cap, par_source="pool")
        result = search(
            problem, problem.expr, uniform_stub(quiet), quiet, random.Random(0), steps_taken=cap - 1
        )
        if len(result.actions) != 5:
            continue
        proofs = _proven_children(result, problem, quiet, steps_taken=cap - 1)
        losses = [s for s, z in proofs.items() if z == -1.0]
        winners = [s for s, z in proofs.items() if z == 0.0]
        if len(proofs) != 5 or len(winners) != 1:
            continue
        states += 1
        shares.append(float(visit_share(result)[losses].sum()))
        chosen_solver += result.chosen == result.actions[winners[0]]
        if hasattr(result, "improved"):
            improved.append(float(np.asarray(result.improved, np.float64)[losses].sum()))
    out = {
        "states": states,
        "search_chose_the_solver": f"{chosen_solver}/{states}",
        "visit_share_mass_on_proven_losses": sorted({round(x, 6) for x in shares}),
    }
    if improved:
        out["improved_mass_on_proven_losses_max"] = max(improved)
    return out


def visit_share(result) -> np.ndarray:
    total = result.visits.sum()
    return result.visits / total if total else np.full(len(result.actions), 1 / len(result.actions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-suite", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    problems = []
    for path in SUITES:
        rows = read_suite(path)
        problems += [
            suite_problem(rows[i]) for i in sample_indices(len(rows), args.per_suite, args.seed)
        ]

    torch.manual_seed(0)
    network = Reckoner(cfg)
    network.eval()
    evaluators = {
        "uniform_stub": uniform_stub(cfg),
        "random_net_value_0": model_evaluator(network, cfg, 0.0),
        "random_net_value_1": model_evaluator(network, cfg, 1.0),
    }

    profiles: dict[int, set] = defaultdict(set)
    per_evaluator = {}
    for name, evaluator in evaluators.items():
        items = [
            (p, p.expr, random.Random(args.seed * 100_003 + i), 0) for i, p in enumerate(problems)
        ]
        results = run_batched(items, evaluator, cfg)
        considered = Counter()
        for result in results:
            n = int(np.count_nonzero(result.visits))
            considered[n] += 1
            profiles[n].add(tuple(sorted(result.visits.tolist(), reverse=True)))
        reading = {"visit_share": _read(results, problems, cfg, visit_share)}
        if hasattr(results[0], "improved"):
            reading["improved"] = _read(
                results, problems, cfg, lambda r: np.asarray(r.improved, np.float64)
            )
        reading["roots_by_actions_visited"] = {str(k): v for k, v in sorted(considered.items())}
        per_evaluator[name] = reading

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    report = {
        "git_sha": head.stdout.strip() if head.returncode == 0 else "not-a-git-checkout",
        "reckoner_source": str(Path(reckoner.__file__).resolve().parent),
        "sims": cfg.search.sims,
        "gumbel_m": cfg.search.gumbel_m,
        "problems": len(problems),
        "cap_construction": cap_construction(cfg),
        "schedule_census": {
            str(n): {
                "distinct_sorted_profiles": len(p),
                "profile": list(next(iter(p))) if len(p) == 1 else None,
            }
            for n, p in sorted(profiles.items())
        },
        "per_evaluator": per_evaluator,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
