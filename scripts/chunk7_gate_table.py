"""Re-run every chunk-7 gate under the FIXED search. One table, all verbatim.

The pre-F-06 numbers do not transfer. Selection is now path-dependent where
before there was no selection to depend on anything, so determinism, budget and
the StateTooLarge semantics are all being exercised for the first time on a
search that descends.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from reckoner.config import Config, with_m1_search
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import Problem
from reckoner.expr import add, eq, mul, num, var
from reckoner.gates import search_found_a_solve
from reckoner.search import run_batched, search, uniform_stub
from reckoner.vocab import GOAL_SOLVE, VAR_X

REPO = Path(__file__).resolve().parents[1]
#: Chunk 7's gates were measured with `uniform_stub`, which reports no steps-to-go,
#: so every config here is the search that reads 'wdl' (the default until 2026-11-16).
CFG = with_m1_search(Config())
X = var(VAR_X)
PROBLEM = Problem(
    goal=GOAL_SOLVE,
    target=VAR_X,
    par=3,
    par_source="bfs",
    expr=eq(add(mul(num(3), X), num(6)), num(21)),
)


def main() -> int:
    ev = uniform_stub(CFG)
    out: dict = {"search": "post-F-06 (descending)"}
    print("  CHUNK-7 GATE TABLE — re-run under the FIXED (descending) search\n")

    # --- budget identity ------------------------------------------------
    print("  budget identity")
    print(
        f"    {'sims':>5} {'used':>6} {'visits.sum':>11} {'nodes':>6} {'max_depth':>10}  identity"
    )
    rows = []
    for sims in (6, 16, 31, 48):
        r = search(PROBLEM, PROBLEM.expr, ev, CFG, random.Random(7), sims=sims, m=5)
        ok = int(r.visits.sum()) == r.stats.sims_used <= r.stats.sims_requested
        rows.append(
            {
                "sims": sims,
                "used": r.stats.sims_used,
                "visits": int(r.visits.sum()),
                "nodes": r.stats.nodes,
                "max_depth": r.stats.max_depth,
                "holds": ok,
            }
        )
        print(
            f"    {sims:>5} {r.stats.sims_used:>6} {int(r.visits.sum()):>11} "
            f"{r.stats.nodes:>6} {r.stats.max_depth:>10}  {'HOLDS' if ok else 'FAILS'}"
        )
    out["budget_identity"] = rows

    # --- determinism: 2-seed in the eval profile ------------------------
    cfg_eval = with_m1_search(Config())
    cfg_eval.search.root_noise = False
    a = search(
        PROBLEM, PROBLEM.expr, uniform_stub(cfg_eval), cfg_eval, random.Random(1), sims=16, m=5
    )
    b = search(
        PROBLEM, PROBLEM.expr, uniform_stub(cfg_eval), cfg_eval, random.Random(999), sims=16, m=5
    )
    same = bool(np.array_equal(a.visits, b.visits) and a.chosen == b.chosen)
    out["two_seed_eval_identity"] = {"identical": same, "visits": a.visits.tolist()}
    print(
        f"\n  2-seed eval-profile identity (root_noise=False): "
        f"{'IDENTICAL' if same else 'DIFFER'}  visits={a.visits.tolist()}"
    )

    # --- determinism: cross-process -------------------------------------
    # The probe is a FILE (tests/subprocess_probes/search_determinism.py), not an
    # embedded string — a blind text edit matched inside the old literal twice.
    probe = REPO / "tests" / "subprocess_probes" / "search_determinism.py"
    seen = set()
    for seed in ("0", "1", "7777", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        seen.add(
            subprocess.run(
                [sys.executable, str(probe)], capture_output=True, text=True, check=True, env=env
            ).stdout.strip()
        )
    out["cross_process"] = {"distinct_outputs": len(seen), "value": sorted(seen)[0]}
    print(
        f"  cross-process identity (4 PYTHONHASHSEEDs): "
        f"{'IDENTICAL' if len(seen) == 1 else 'DIFFER'}  {sorted(seen)[0]}"
    )

    # --- depth-1 gate at m >= B_max --------------------------------------
    suite1 = read_suite(REPO / "runs" / "suites" / "solve_in_1.jsonl")
    print(f"\n  {'m':>4} {'depth-1 solved':>16} {'rate':>8}")
    gate = {}
    for m in (3, 5, 16):
        solved = 0
        for i, row in enumerate(suite1):
            p = suite_problem(row)
            r = search(p, p.expr, ev, CFG, random.Random(1000 + i), sims=16, m=m)
            # The SAME predicate the test imports. This script and that test
            # disagreeing is exactly what produced a 0/200 re-run for a search
            # that was working (FINDINGS.md F-14's closing note).
            if search_found_a_solve(r):
                solved += 1
        gate[m] = solved
        print(f"    {m:>2} {solved:>14}/{len(suite1)} {100 * solved / len(suite1):>7.1f}%")
    out["depth1_gate"] = gate

    # --- StateTooLarge under real descent ---------------------------------
    cfg_small = with_m1_search(Config())
    cfg_small.model.seq_len = 26
    r = search(
        PROBLEM, PROBLEM.expr, uniform_stub(cfg_small), cfg_small, random.Random(3), sims=48, m=5
    )
    out["state_too_large"] = {
        "count": r.stats.state_too_large,
        "sims_used": r.stats.sims_used,
        "visits": int(r.visits.sum()),
        "nodes": r.stats.nodes,
    }
    print(
        f"\n  StateTooLarge under descent (seq_len=26, sims=48): "
        f"{r.stats.state_too_large} counted, sims_used={r.stats.sims_used}, "
        f"visits.sum={int(r.visits.sum())}, nodes={r.stats.nodes}"
    )

    # --- the descent gate --------------------------------------------------
    rows5 = [
        r
        for r in read_suite(REPO / "runs" / "suites" / "solve_in_5.jsonl")
        if r["goal"] == GOAL_SOLVE
    ]
    p5 = suite_problem(rows5[0])
    print(
        f"\n  descent gate (branchy depth-5 SOLVE, {len(__import__('reckoner.rules', fromlist=['x']).legal_actions(p5.expr))} root actions)"
    )
    print(f"    {'sims':>5} {'nodes':>6} {'max_depth':>10} {'evals':>6}")
    descent = []
    for sims in (6, 16, 31, 48):
        r = search(p5, p5.expr, ev, CFG, random.Random(1), sims=sims, m=5)
        descent.append(
            {
                "sims": sims,
                "nodes": r.stats.nodes,
                "max_depth": r.stats.max_depth,
                "evals": r.stats.evaluations,
            }
        )
        print(f"    {sims:>5} {r.stats.nodes:>6} {r.stats.max_depth:>10} {r.stats.evaluations:>6}")
    out["descent"] = descent

    # --- parity (credited, re-verified here for the single table) ---------
    seq = [
        search(PROBLEM, PROBLEM.expr, ev, CFG, random.Random(20 + i), sims=48, m=5)
        for i in range(4)
    ]
    bat = run_batched(
        [(PROBLEM, PROBLEM.expr, random.Random(20 + i)) for i in range(4)],
        ev,
        CFG,
        sims=48,
        m=5,
        batch_size=3,
    )
    parity = all(
        np.array_equal(a.visits, b.visits) and a.chosen == b.chosen
        for a, b in zip(seq, bat, strict=True)
    )
    out["parity_spot"] = parity
    print(
        f"\n  sequential/batched parity spot-check (4 searches, batch 3): "
        f"{'EXACT' if parity else 'DIFFER'}"
    )

    (REPO / "runs" / "chunk7_gate_table.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print("\n  wrote runs/chunk7_gate_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
