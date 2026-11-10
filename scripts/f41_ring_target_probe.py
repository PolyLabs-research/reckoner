"""F-41's instrument: does the ring train the policy toward the actions search chose?

Measures the Phase-2 policy target **end to end on the real path** — the batched
runner writes the ring through ``runner._settle``, ``train.ring_batch`` turns each
record into the dense target the policy loss consumes — and scores every stored
visit against ground truth captured from the search that produced it.

Ground truth is taken at the one place it exists. ``_settle`` is wrapped so that,
for each episode, the ``SearchResult`` behind every record the original appends is
captured in the order the original appends it (the same ``encode`` skip the runner
applies is applied here, and the count is asserted equal). Nothing about what
``visit_actions`` *means* is assumed, so the same probe reads the writer before and
after the F-41 fix. For the k-th stored visit of a record:

* ``trained cell`` = the dense cell ``ring_batch`` writes that visit's mass into;
* ``true cell``    = ``action_index(*result.actions[slot])`` for the slot the
  search ranked k-th by visits.

Each stored visit is then **correct** (trained == true), **wrong-legal** (a
different legal action got the mass) or **dropped** (an illegal cell — the policy
loss masks it, and its ``nan_to_num`` makes the loss silently ignore it). Mass is
weighted by the stored visit share. The record-level reading is whether the
target's argmax over legal cells is one of the search's max-visit actions.

A second, **policy-independent** census runs over every frozen suite start state:
for each root slot j, the HEAD writer stores j and ``ring_batch`` trains cell j,
which is right only when ``action_index(*legal_actions(s)[j]) == j``. That number
depends on the rule system and the site enumeration alone, so it holds for any
network and any budget.

Coverage is published beside the result (AGENTS §5): the n_legal histogram, and
records by par stratum and par source.

    python scripts/f41_ring_target_probe.py --per-suite 20 --out /tmp/f41.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import reckoner
from reckoner import runner
from reckoner.config import Config
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.episode import bfs_solution, decode_state
from reckoner.model import N_RULES, StateTooLarge, action_index, encode
from reckoner.replay import ReplayRing
from reckoner.rules import RULES, legal_actions
from reckoner.search import uniform_stub
from reckoner.train import ring_batch

REPO = Path(__file__).resolve().parents[1]
SUITES = sorted((REPO / "runs" / "suites").glob("*.jsonl"))


def slot_census(cfg: Config) -> dict:
    """Policy-independent: root slot j trained as dense cell j, over every suite state."""
    max_sites = cfg.model.max_sites
    legal_cells: Counter = Counter()
    n_legal_hist: Counter = Counter()
    states_all_correct = 0
    states = 0
    capped = Counter()
    # States where NO storable slot reaches a legal cell: the whole target is
    # dropped whatever the network or budget, because which slots get visits
    # cannot rescue a state none of whose slot numbers is a legal cell.
    no_legal_slot: Counter = Counter()
    by_family: Counter = Counter()
    for path in SUITES:
        family = "solve_in" if path.name.startswith("solve_in_") else "scripted_in"
        for row in read_suite(path):
            _goal, _target, expr = decode_state(tuple(row["tokens"]))
            actions = legal_actions(expr)
            cells = {action_index(r, s, max_sites) for r, s in actions}
            states += 1
            by_family[family] += 1
            n_legal_hist[len(actions)] += 1
            ok = True
            reachable = False
            for j, (r, s) in enumerate(actions):
                kind = (
                    "correct"
                    if action_index(r, s, max_sites) == j
                    else ("wrong_legal" if j in cells else "dropped")
                )
                legal_cells[kind] += 1
                if j < cfg.search.gumbel_m:
                    capped[kind] += 1
                    reachable = reachable or kind != "dropped"
                ok = ok and kind == "correct"
            states_all_correct += ok
            no_legal_slot[family] += not reachable
    return {
        "states": states,
        "states_by_family": dict(by_family),
        "states_with_no_storable_slot_on_a_legal_cell": dict(no_legal_slot),
        "states_every_slot_correct": states_all_correct,
        "all_slots": dict(legal_cells),
        f"slots_below_m_{cfg.search.gumbel_m}": dict(capped),
        "n_legal_histogram": {str(k): v for k, v in sorted(n_legal_hist.items())},
    }


def optimal_move_census(cfg: Config) -> dict:
    """Where the HEAD writer sends the move a *good* policy's target concentrates on.

    A trained network's search puts most of its visits on one action, so what
    decides whether the policy learns anything is whether THAT action's slot lands
    on its own cell. Over the BFS-exact suites (solve_in_1..6), the BFS-optimal
    first move stands in for it. BFS returns *an* optimal move (its tie-break), so
    this is one optimal move per state, not every optimal move.
    """
    max_sites = cfg.model.max_sites
    kinds: Counter = Counter()
    by_rule: Counter = Counter()
    states = 0
    for path in SUITES:
        if not path.name.startswith("solve_in_"):
            continue
        for row in read_suite(path):
            problem = suite_problem(row)
            derivation = bfs_solution(problem, cfg)
            if not derivation:
                continue
            (rule_id, site_id), _state = derivation[0]
            actions = legal_actions(problem.expr)
            j = actions.index((rule_id, site_id))
            cells = {action_index(r, s, max_sites) for r, s in actions}
            true = action_index(rule_id, site_id, max_sites)
            kind = "correct" if j == true else ("wrong_legal" if j in cells else "dropped")
            kinds[kind] += 1
            by_rule[(RULES[rule_id].name, kind)] += 1
            states += 1
    return {
        "states": states,
        "optimal_move_slot": dict(kinds),
        "by_rule": {f"{name}:{kind}": n for (name, kind), n in sorted(by_rule.items())},
    }


def end_to_end(cfg: Config, per_suite: int, seed: int) -> dict:
    """Real runner -> ring -> ring_batch, scored against captured search results."""
    problems = []
    for path in SUITES:
        rows = read_suite(path)
        problems += [suite_problem(rows[i]) for i in sample_indices(len(rows), per_suite, seed)]

    ring = ReplayRing(200_000, cfg)
    truth: dict[int, tuple] = {}
    original = runner._settle

    def capturing(stats, ring_, e, trail, cfg_, *, solved, capped):
        entries = []
        for result, expr, _steps in trail[e.index]:
            try:
                encode(e.problem, expr, cfg_)
            except StateTooLarge:
                continue
            entries.append(result)
        start = len(ring_)
        original(stats, ring_, e, trail, cfg_, solved=solved, capped=capped)
        if len(ring_) - start != len(entries):
            raise RuntimeError("probe lost alignment with the runner's appends")
        for k, result in enumerate(entries):
            truth[start + k] = result

    runner._settle = capturing
    try:
        stats = runner.run_iteration(problems, uniform_stub(cfg), cfg, ring, seed=seed)
    finally:
        runner._settle = original

    max_sites = cfg.model.max_sites
    width = ring.visit_actions.shape[1]
    mass = Counter()
    visits_n = Counter()
    argmax_hits = 0
    legal_mass = []
    records = 0
    by_stratum: Counter = Counter()
    by_source: Counter = Counter()
    # The pooled split mixes strata whose ring rows behave differently, so it is
    # also published by (par source, par band). A campaign's own ring is drawn
    # from train_100k (BFS par 1-6), and only the matching stratum stands in
    # for it.
    mass_by_stratum: dict[str, Counter] = {}
    records_by_stratum: Counter = Counter()
    for i in range(len(ring)):
        result = truth[i]
        record = ring.get(i)
        # F-41 is about the VISIT fields; since F-42 the default target is π′, so
        # the visit arm is pinned where the field exists (the probe must still run
        # against pre-F-42 trees, whose ring_batch takes no target).
        if hasattr(cfg.train, "policy_target"):
            batch = ring_batch(ring, [i], cfg, target="visits")
        else:
            batch = ring_batch(ring, [i], cfg)
        if batch is None:
            raise RuntimeError(f"record {i} produced no batch")
        target = batch.policy_target[0]
        mask = batch.legal_mask[0]
        cells = {action_index(r, s, max_sites) for r, s in result.actions}
        if set(torch.nonzero(mask).flatten().tolist()) != cells:
            raise RuntimeError(f"record {i}: ring_batch's legal mask is not the root's legal set")

        order = np.argsort(-result.visits)[:width]
        total = float(result.visits[order].sum())
        par = int(record["par"])
        stratum = f"{record['par_source']}:par_{'1-6' if par <= 6 else '7-10'}"
        records_by_stratum[stratum] += 1
        cell = mass_by_stratum.setdefault(stratum, Counter())
        for k, slot in enumerate(order):
            count = int(result.visits[slot])
            if count == 0:
                continue
            trained = int(record["visit_actions"][k])
            true = action_index(*result.actions[int(slot)], max_sites)
            kind = (
                "correct" if trained == true else ("wrong_legal" if trained in cells else "dropped")
            )
            visits_n[kind] += 1
            mass[kind] += count / total
            cell[kind] += count / total

        legal_mass.append(float((target * mask).sum()))
        top = int(result.visits.max())
        best = {
            action_index(*result.actions[s], max_sites)
            for s in np.flatnonzero(result.visits == top)
        }
        masked = target.masked_fill(~mask, -1.0)
        argmax_hits += int(torch.argmax(masked)) in best
        records += 1
        by_stratum[int(record["par"])] += 1
        by_source[str(record["par_source"])] += 1

    return {
        "problems": len(problems),
        "per_suite": per_suite,
        "episodes_solved": stats.episodes_solved,
        "records": records,
        "stored_visits": dict(visits_n),
        "visit_mass": {k: round(v / records, 4) for k, v in mass.items()},
        "target_mass_on_legal_cells_mean": round(float(np.mean(legal_mass)), 4),
        "target_mass_on_legal_cells_min": round(float(np.min(legal_mass)), 4),
        "argmax_is_a_max_visit_action": f"{argmax_hits}/{records}",
        "records_by_par": {str(k): v for k, v in sorted(by_stratum.items())},
        "records_by_par_source": dict(by_source),
        "visit_mass_by_stratum": {
            name: {
                "records": records_by_stratum[name],
                **{k: round(v / records_by_stratum[name], 4) for k, v in sorted(cell.items())},
            }
            for name, cell in sorted(mass_by_stratum.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-suite", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if head.returncode == 0:
        sha = head.stdout.strip()
        dirty: bool | str = bool(
            subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    else:
        # An exported tree (e.g. `git archive` of an older commit, to read the
        # pre-fix writer). No checkout means no SHA to vouch for: say so, and
        # name the source tree actually imported instead.
        sha, dirty = "not-a-git-checkout", "unknown"
    report = {
        "git_sha": sha,
        "dirty": dirty,
        "reckoner_source": str(Path(reckoner.__file__).resolve().parent),
        "sims": cfg.search.sims,
        "gumbel_m": cfg.search.gumbel_m,
        "dense_width": N_RULES * cfg.model.max_sites,
        "slot_census": slot_census(cfg),
        "optimal_move_census": optimal_move_census(cfg),
        "end_to_end": end_to_end(cfg, args.per_suite, args.seed),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
