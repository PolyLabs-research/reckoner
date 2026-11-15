"""EXPLORATION (AGENTS §7.1) — can the Phase-2 loop extend a weak starting network?

**Nothing this script prints is evidence.** It exists to find out whether an
effect is there before a registered round is written for it, and its outputs go
under ``runs/explore/``, which git ignores.

The question, from ``REVIEW-2026-11-15.md`` step 1: the Phase-1 anchor was
supervised on depths 1-6, the same band the suites measure, so nothing is left
for the loop to learn there. A network supervised only on problems of depth <= K
has never seen the early states of deeper problems. Does self-play teach it
those, and how fast, graded against exact BFS answers?

    explore_weak_start.py optimal-sets --out runs/explore/weak/optimal_sets.json
    explore_weak_start.py train-anchor --max-depth 3 --out runs/explore/weak/anchor-d3.pt
    explore_weak_start.py loop --anchor runs/explore/weak/anchor-d3.pt --arm A --out-dir runs/explore/weak/d3-A
    explore_weak_start.py summary --root runs/explore/weak

Arms of ``loop``. Each adds one thing to the one before, so a difference between
neighbours has one cause. The patches are applied from here, by replacing module
functions, so the library is untouched until a registered round asks for them:

    A  the loop as merged: π′ target, value head silent, no rehearsal
       (rehearsal would feed the weak network BFS labels for the deep problems)
    B  A + the search scores an unsolved state at or past par as a loss. Past par
       the episode's z is -1 whatever happens next; the merged search asks the
       silent value head instead, which answers 0, so an unexplored line outranks
       a solution that is over par.
    C  B + the steps head trains only on rows from solved episodes. ``steps_loss``
       says it is masked that way; ``ring_batch`` passes a mask of all ones.
    D  C + the value head's loss is switched off, so nothing but the policy and
       the (clean) steps target reaches the shared layers.

The loop is the campaign's iteration body without its league and ladder: problems
from ``train_100k`` at their exact BFS par, self-play into one ring, then
``train_on_ring``. It is smaller than the campaign by default (200 episodes, 200
steps) so that a curve costs minutes.

What is read after every iteration, by problem depth 1-6:
* greedy play (one simulation): solved, at par, capped, on frozen suite problems;
* the same with the full search budget, every ``--search-every`` iterations;
* is the policy's top action OPTIMAL at the start state (needs ``optimal-sets``);
* agreement with BFS's single label on ``phase1_eval``, by distance to solution.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import reckoner.runner as runner_module
import reckoner.search as search_module
import reckoner.train as train_module
from reckoner.campaign import SUITES, eval_profile
from reckoner.config import Config, validate
from reckoner.dataset import (
    anchored_data,
    read_suite,
    sample_indices,
    suite_problem,
    training_problems,
)
from reckoner.episode import Problem, bfs_par, verify
from reckoner.evaluate import model_evaluator
from reckoner.model import (
    Reckoner,
    crop_to_content,
    encode,
    load_checkpoint,
    save_checkpoint,
)
from reckoner.replay import ReplayRing
from reckoner.rules import apply, legal_actions
from reckoner.runner import run_iteration
from reckoner.train import SupervisionSet, make_batch, train, train_on_ring

REPO = Path(__file__).resolve().parents[1]
DEPTHS = (1, 2, 3, 4, 5, 6)
ARMS = ("A", "B", "C", "D")


# --------------------------------------------------------------------- data


class DepthSubset:
    """The rows of a supervision set whose PROBLEM depth is at most ``max_depth``.

    A view ``make_batch`` and ``train`` can read in place of the full set. The
    filter is on the problem's depth, not the row's distance to the solution: a
    depth-6 problem's last three states look like a depth-3 problem, and they are
    left out with the rest of it, so the network has seen no part of a deeper
    problem.
    """

    def __init__(self, data: SupervisionSet, max_depth: int) -> None:
        rows = np.flatnonzero(np.asarray(data.depth) <= max_depth)
        self.tokens = np.asarray(data.tokens)[rows]
        self.lengths = np.asarray(data.lengths)[rows]
        self.action = np.asarray(data.action)[rows]
        self.steps_remaining = np.asarray(data.steps_remaining)[rows]
        self.depth = np.asarray(data.depth)[rows]
        self.goal = np.asarray(data.goal)[rows]
        self.meta = data.meta
        self.path = data.path

    def __len__(self) -> int:
        return int(self.lengths.shape[0])


def suite_problems(depth: int, k: int) -> list[Problem]:
    rows = read_suite(SUITES / f"solve_in_{depth}.jsonl")
    return [suite_problem(rows[i]) for i in sample_indices(len(rows), min(k, len(rows)), 0)]


# ----------------------------------------------------------- optimal action sets


def _optimal_actions(row: dict) -> dict:
    """Every action from a suite start state that keeps the problem on par."""
    cfg = Config()
    rng = random.Random(0)
    problem = suite_problem(row)
    depth = int(problem.par)
    optimal = []
    for rule_id, site_id in legal_actions(problem.expr):
        child = apply(problem.expr, rule_id, site_id)
        posed = Problem(
            goal=problem.goal, expr=child, par=problem.par, target=problem.target, par_source="bfs"
        )
        if verify(posed, child, cfg, rng):
            reach = True
        elif depth <= 1:
            reach = False
        else:
            reach = bfs_par(posed, cfg, cap=depth - 1) is not None
        if reach:
            optimal.append(rule_id * cfg.model.max_sites + site_id)
    return {"tokens": row["tokens"], "par": depth, "optimal": optimal}


def cmd_optimal_sets(args: argparse.Namespace) -> int:
    started = time.monotonic()
    out: dict[str, list[dict]] = {}
    with multiprocessing.Pool(args.workers) as pool:
        for depth in DEPTHS:
            rows = read_suite(SUITES / f"solve_in_{depth}.jsonl")
            picked = [rows[i] for i in sample_indices(len(rows), min(args.per_suite, len(rows)), 0)]
            states = pool.map(_optimal_actions, picked, chunksize=1)
            empty = sum(not s["optimal"] for s in states)
            several = sum(len(s["optimal"]) > 1 for s in states)
            print(
                f"  depth {depth}: {len(states)} states, {several} with several optimal actions, "
                f"{empty} with none (must be 0), {time.monotonic() - started:.0f}s",
                flush=True,
            )
            out[str(depth)] = states
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    return 0


# ------------------------------------------------------------------ anchors


def cmd_train_anchor(args: argparse.Namespace) -> int:
    cfg = Config()
    validate(cfg)
    torch.set_num_threads(args.threads)
    data = DepthSubset(SupervisionSet(anchored_data("phase1_train")), args.max_depth)
    torch.manual_seed(args.seed)
    model = Reckoner(cfg)
    started = time.monotonic()

    def on_log(step: int, loss: float, lr: float) -> None:
        print(
            f"  step {step:>5}  loss {loss:.4f}  lr {lr:.2e}  {time.monotonic() - started:.0f}s",
            flush=True,
        )

    print(
        f"  weak anchor: problems of depth <= {args.max_depth}, {len(data):,} rows, {args.steps} steps"
    )
    stats = train(
        model,
        data,
        cfg,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        log_every=250,
        on_log=on_log,
    )
    model.to("cpu").eval()
    save_checkpoint(args.out, model, cfg, stats.steps, explore_max_depth=args.max_depth)
    print(f"  saved {args.out}  final loss {stats.losses[-1]:.4f}  nan skips {stats.nan_skips}")
    return 0


# --------------------------------------------------------------- the arm patches


def install_certain_loss_bound() -> None:
    """Arm B: an unsolved state at or past par is a loss, and the search is told so."""
    original = search_module._leaf_outcome

    def bounded(problem, expr, cfg, rng, stats, *, total_steps):
        outcome = original(problem, expr, cfg, rng, stats, total_steps=total_steps)
        if outcome is None and total_steps >= problem.par:
            # Counted with the step-cap terminals: the runner's descent identity
            # requires every unevaluated node to be a counted terminal, and this
            # exploration adds no counter of its own. A registered round would.
            stats.step_cap_reached += 1
            return -1.0
        return outcome

    search_module._leaf_outcome = bounded


class SolvedRows:
    """Arm C: which ring rows came from an episode that solved its problem.

    The ring does not store it, so it is captured where the runner settles each
    episode and applied where ``train_on_ring`` builds its batch.
    """

    def __init__(self, capacity: int) -> None:
        self.flags = np.zeros(capacity, dtype=bool)

    def install(self) -> None:
        settle = runner_module._settle
        flags = self.flags

        def recording(stats, ring, e, trail, cfg, *, solved, capped):
            before = len(ring) if ring is not None else 0
            settle(stats, ring, e, trail, cfg, solved=solved, capped=capped)
            if ring is not None:
                if len(ring) >= ring.capacity:
                    raise RuntimeError("the exploration ring wrapped: raise --capacity")
                flags[before : len(ring)] = solved

        runner_module._settle = recording
        ring_batch = train_module.ring_batch

        def masked(ring, indices, cfg, *, target=None):
            batch = ring_batch(ring, indices, cfg, target=target)
            if batch is not None:
                if batch.skipped:
                    raise RuntimeError("a ring row sat out, so the solved mask cannot be aligned")
                batch.solved_mask = torch.from_numpy(flags[np.asarray(indices)])
            return batch

        train_module.ring_batch = masked


# ------------------------------------------------------------------ readings


def read_play(model, cfg: Config, device: str, per_suite: int, sims: int, m: int) -> dict:
    """Episodes on frozen suite problems, measurement profile (no root noise)."""
    ev_cfg = eval_profile(cfg)
    evaluator = model_evaluator(model, ev_cfg, 0.0, device=device)
    out = {}
    for depth in DEPTHS:
        problems = suite_problems(depth, per_suite)
        stats = run_iteration(problems, evaluator, ev_cfg, None, sims=sims, m=m, seed=0)
        cell = stats.z_by_par_source.get("bfs", {})
        out[str(depth)] = {
            "n": stats.episodes,
            "solved": stats.episodes_solved,
            "at_par": int(cell.get("0", 0)) + int(cell.get("+1", 0)),
            "capped": stats.episodes_capped,
        }
    return out


def read_label_agreement(model, cfg: Config, device: str, data: SupervisionSet) -> dict:
    """Top-1 agreement with BFS's single label, by distance to the solution."""
    hits: dict[int, int] = {}
    totals: dict[int, int] = {}
    with torch.no_grad():
        for start in range(0, len(data), 256):
            batch = make_batch(data, list(range(start, min(start + 256, len(data)))), cfg)
            policy, _, _ = model(batch.tokens.to(device), batch.site_positions.to(device))
            masked = policy.cpu().masked_fill(~batch.legal_mask, float("-inf"))
            correct = masked.argmax(dim=1) == batch.policy_target.argmax(dim=1)
            for distance, ok in zip(batch.steps_target.tolist(), correct.tolist(), strict=True):
                key = int(distance)
                hits[key] = hits.get(key, 0) + int(ok)
                totals[key] = totals.get(key, 0) + 1
    return {str(k): {"n": totals[k], "hits": hits[k]} for k in sorted(totals)}


def read_optimal(model, cfg: Config, device: str, optimal_sets: dict | None) -> dict | None:
    """Is the policy's top action at the start state one of the optimal ones?"""
    if optimal_sets is None:
        return None
    out = {}
    for depth, states in optimal_sets.items():
        problems = [suite_problem({**state, "par_source": "bfs"}) for state in states]
        encoded = [encode(problem, problem.expr, cfg) for problem in problems]
        sites = torch.stack([e.site_positions for e in encoded])
        tokens = crop_to_content(torch.stack([e.tokens for e in encoded]), sites)
        with torch.no_grad():
            policy, _, _ = model(tokens.to(device), sites.to(device))
        legal = torch.stack([e.legal_mask for e in encoded])
        top = policy.cpu().masked_fill(~legal, float("-inf")).argmax(dim=1).tolist()
        out[depth] = {
            "n": len(states),
            "optimal": sum(int(t in s["optimal"]) for t, s in zip(top, states, strict=True)),
        }
    return out


def reading(model, cfg, device, args, eval_set, optimal_sets, with_search: bool) -> dict:
    model.eval()
    row = {
        "greedy": read_play(model, cfg, device, args.eval_per_suite, 1, 1),
        "label_agreement": read_label_agreement(model, cfg, device, eval_set),
        "optimal_top": read_optimal(model, cfg, device, optimal_sets),
    }
    if with_search:
        row["search"] = read_play(
            model, cfg, device, args.search_per_suite, cfg.search.sims, cfg.search.gumbel_m
        )
    return row


# ---------------------------------------------------------------------- loop


def cmd_loop(args: argparse.Namespace) -> int:
    torch.set_num_threads(args.threads)
    cfg = Config()
    overrides = {
        "rehearsal_frac": 0.0,
        "train_steps_per_iter": args.train_steps,
        "lr_warmup_steps": args.train_steps // 2,
    }
    if args.arm == "D":
        overrides["value_loss_weight"] = 0.0
    cfg = replace(cfg, train=replace(cfg.train, **overrides))
    validate(cfg)

    if args.arm in ("B", "C", "D"):
        install_certain_loss_bound()
    if args.arm in ("C", "D"):
        SolvedRows(args.capacity).install()

    model, _ = load_checkpoint(args.anchor, cfg)
    model.to(args.device).eval()
    eval_set = SupervisionSet(anchored_data("phase1_eval"))
    optimal_sets = json.loads(args.optimal_sets.read_text()) if args.optimal_sets else None
    ring = ReplayRing(args.capacity, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve = (args.out_dir / "curve.jsonl").open("w")

    def emit(row: dict) -> None:
        curve.write(json.dumps(row, sort_keys=True) + "\n")
        curve.flush()
        greedy = row["reading"]["greedy"]
        at_par = " ".join(f"{greedy[str(d)]['at_par']:>3}" for d in DEPTHS)
        solved = " ".join(f"{greedy[str(d)]['solved']:>3}" for d in DEPTHS)
        print(
            f"  [{args.arm}] iter {row['iteration']:>2}  greedy at par by depth: {at_par}  "
            f"solved: {solved}  (of {greedy['1']['n']})  {row.get('seconds', {})}",
            flush=True,
        )

    for n in range(args.iterations + 1):
        started = time.monotonic()
        with_search = n % args.search_every == 0 or n == args.iterations
        row = {
            "arm": args.arm,
            "anchor": str(args.anchor),
            "iteration": n,
            "reading": reading(model, cfg, args.device, args, eval_set, optimal_sets, with_search),
        }
        seconds = {"read": round(time.monotonic() - started, 1)}
        if n == args.iterations:
            row["seconds"] = seconds
            emit(row)
            break

        t0 = time.monotonic()
        problems = training_problems(anchored_data("train_100k"), args.episodes, seed=n)
        stats = run_iteration(
            problems, model_evaluator(model, cfg, 0.0, device=args.device), cfg, ring, seed=n
        )
        seconds["self_play"] = round(time.monotonic() - t0, 1)
        row["self_play"] = {
            "episodes": stats.episodes,
            "solved": stats.episodes_solved,
            "capped": stats.episodes_capped,
            "z": stats.z_by_par_source.get("bfs", {}),
            "solved_by_depth": {str(k): v for k, v in sorted(stats.solved_by_depth.items())},
            "ring_rows": len(ring),
            "entropies": stats.entropies(),
        }

        t0 = time.monotonic()
        trained = train_on_ring(
            model, ring, cfg, steps=cfg.train.train_steps_per_iter, seed=n, device=args.device
        )
        model.eval()
        seconds["train"] = round(time.monotonic() - t0, 1)
        tail = trained.losses[-20:]
        row["train"] = {
            "steps": trained.steps,
            "nan_skips": trained.nan_skips,
            "encode_skips": trained.encode_skips,
            "loss_first": trained.losses[0] if trained.losses else None,
            "loss_last20_mean": sum(tail) / len(tail) if tail else None,
        }
        row["seconds"] = seconds
        emit(row)

    curve.close()
    model.to("cpu")
    save_checkpoint(args.out_dir / "final.pt", model, cfg, args.iterations, explore_arm=args.arm)
    return 0


# ------------------------------------------------------------------- summary


def cmd_summary(args: argparse.Namespace) -> int:
    for path in sorted(args.root.glob("*/curve.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        print(f"\n  {path.parent.name}   (arm {rows[0]['arm']}, {len(rows) - 1} iterations)")
        print("    greedy AT PAR by problem depth 1..6, then solved; search at par where read")
        for row in rows:
            greedy = row["reading"]["greedy"]
            n = greedy["1"]["n"]
            line = (
                f"    iter {row['iteration']:>2}  "
                + " ".join(f"{greedy[str(d)]['at_par']:>3}" for d in DEPTHS)
                + f"  /{n}   solved "
                + " ".join(f"{greedy[str(d)]['solved']:>3}" for d in DEPTHS)
            )
            if "search" in row["reading"]:
                search = row["reading"]["search"]
                line += (
                    "   search at par "
                    + " ".join(f"{search[str(d)]['at_par']:>3}" for d in DEPTHS)
                    + f" /{search['1']['n']}"
                )
            optimal = row["reading"].get("optimal_top")
            if optimal:
                line += "   top optimal " + " ".join(
                    f"{optimal[str(d)]['optimal']:>3}" for d in DEPTHS
                )
            if "self_play" in row:
                play = row["self_play"]
                line += f"   self-play solved {play['solved']}/{play['episodes']} capped {play['capped']}"
            print(line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("optimal-sets")
    p.add_argument("--per-suite", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_optimal_sets)

    p = sub.add_parser("train-anchor")
    p.add_argument("--max-depth", type=int, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_train_anchor)

    p = sub.add_parser("loop")
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-per-suite", type=int, default=100)
    p.add_argument("--search-per-suite", type=int, default=40)
    p.add_argument("--search-every", type=int, default=5)
    p.add_argument("--capacity", type=int, default=200_000)
    p.add_argument("--optimal-sets", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=cmd_loop)

    p = sub.add_parser("summary")
    p.add_argument("--root", type=Path, required=True)
    p.set_defaults(run=cmd_summary)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
