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
neighbours has one cause. **B's and V's mechanisms are config keys since WEAK-w1**
(``search.past_par``, ``search.value_source``) and are set here by name, for
every arm, so an arm means the same thing whatever the defaults are. Until
2026-11-16 this script applied them as patches; the arithmetic is the same.
What the library still does not have (C and F) is patched in from here, by
replacing module functions:

    A  the loop as M1 ran it: π′ target, value head silent, no rehearsal
       (rehearsal would feed the weak network BFS labels for the deep problems)
    B  A + ``past_par: loss``: the search scores an unsolved state at or past par
       as a loss. Past par the episode's z is -1 whatever happens next; the M1
       search asks the silent value head instead, which answers 0, so an
       unexplored line outranks a solution that is over par.
    C  B + the steps head trains only on rows from solved episodes. ``steps_loss``
       says it is masked that way; ``ring_batch`` passes a mask of all ones.
    D  C + the value head's loss is switched off, so nothing but the policy and
       the (clean) steps target reaches the shared layers.
    E  D + ``value_source: steps``: the search's value comes from the STEPS head.
       A state reached after g steps, with J predicted steps to go, is worth
       clip(par - g - J, -1, 0) against exact par. The win/draw/loss head is not
       read at all. (Step 3 of the review.)
    F  E + the steps head is trained by one-step lookahead through the exact
       rules: target(s) = 1 + min over legal actions of J(child), a solved child
       counting 0, J from the weights the iteration started with; where the
       episode solved, the observed steps-to-go caps it. No solved episode is
       needed to give a state a target.

    V  B + ``value_source: steps`` and nothing else: the training step is the
       merged one (steps trained on every row, value loss on). WEAK-w1's third
       arm. E is V + C + D, so V against E says whether C and D matter once the
       search has a value.
    S  V + C: the steps head trains only on rows from solved episodes, and the
       value loss stays on. An unsolved episode's rows carry "steps to the cap"
       as their steps target, which under ``value_source: steps`` tells the
       search that a family it cannot yet solve is hopeless. S against V
       isolates that; S against E isolates the value loss.

``--sims`` overrides the search budget for self-play and for the search reading,
so "is it the budget or the value" has its own arm (B at a larger budget).

The loop is the campaign's iteration body without its league and ladder: problems
from ``train_100k`` at their exact BFS par, self-play into one ring, then
``train_on_ring``. It is smaller than the campaign by default (200 episodes, 200
steps) so that a curve costs minutes.

What is read after every iteration, by problem depth 1-6:
* greedy play (one simulation): solved, at par, capped, on frozen suite problems,
  in total and for the SOLVE-goal problems alone (see below);
* the same with the full search budget, every ``--search-every`` iterations;
* is the policy's top action OPTIMAL at the start state (needs ``optimal-sets``);
* agreement with BFS's single label on ``phase1_eval``, by distance to solution.

**The stratum no training set has.** ``train_100k`` holds no depth-6 SOLVE
problem (the ``solve_both_sides_product`` template): counted on all 100,000
rows, 0 of 3,550 at par 6, against 95 of the 200 in ``solve_in_6``. So the
readings split out the SOLVE goal, and two more commands exist to test what
practising that stratum does:

    explore_weak_start.py mint --template solve_both_sides_product --count 400 \
        --out runs/explore/weak4/extra_d6_solve.jsonl
    explore_weak_start.py loop ... --extra runs/explore/weak4/extra_d6_solve.jsonl --extra-count 16

``mint`` writes exploration-only problems, BFS-labelled, with every problem that
appears in a frozen instrument or in ``train_100k`` left out. ``--extra-count``
of each iteration's episodes are then drawn from that file instead of from
``train_100k``.

    explore_weak_start.py read --checkpoint runs/phase1/phase1.pt --search m1 bound steps

reads one checkpoint on all ten suites (the four scripted mid strata included)
under each named search, with no training at all.
"""

from __future__ import annotations

import argparse
import copy
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
    dataset_keys,
    problem_key,
    read_dataset,
    read_suite,
    sample_indices,
    state_keys,
    suite_problem,
    suite_row,
    training_problems,
)
from reckoner.episode import Problem, bfs_par, decode_state, verify
from reckoner.evaluate import model_evaluator
from reckoner.generator import TEMPLATES, generate
from reckoner.model import (
    Reckoner,
    StateTooLarge,
    crop_to_content,
    encode,
    load_checkpoint,
    save_checkpoint,
)
from reckoner.replay import ReplayRing
from reckoner.rules import apply, legal_actions
from reckoner.runner import run_iteration
from reckoner.train import SupervisionSet, make_batch, train, train_on_ring
from reckoner.vocab import GOAL_SOLVE

REPO = Path(__file__).resolve().parents[1]
DEPTHS = (1, 2, 3, 4, 5, 6)
ARMS = ("A", "B", "C", "D", "E", "F", "V", "S")
#: Each arm's search, as the two keys WEAK-w1 added: (past_par, value_source).
SEARCH_OF_ARM = {
    "A": ("evaluate", "wdl"),
    "B": ("loss", "wdl"),
    "C": ("loss", "wdl"),
    "D": ("loss", "wdl"),
    "E": ("loss", "steps"),
    "F": ("loss", "steps"),
    "V": ("loss", "steps"),
    "S": ("loss", "steps"),
}
#: Named searches for ``read``: the one M1 ran, the past-par rule alone, both WEAK-w1
#: levers, and both with the steps value capped at par whatever the par source
#: (``ceiling0``, a patch: see :func:`install_zero_ceiling`).
SEARCHES = {
    "m1": ("evaluate", "wdl"),
    "bound": ("loss", "wdl"),
    "steps": ("loss", "steps"),
    "ceiling0": ("loss", "steps"),
}
#: The four mid strata: scripted par, which a solution can beat (z = +1).
MID_SUITES = tuple(f"scripted_in_{d}" for d in (7, 8, 9, 10))


def search_as(cfg: Config, past_par: str, value_source: str) -> Config:
    return replace(cfg, search=replace(cfg.search, past_par=past_par, value_source=value_source))


# --------------------------------------------------------------------- data


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
    torch.manual_seed(args.seed)
    model = Reckoner(cfg)
    if args.steps == 0:
        # No supervision at all: the network as initialised. The loop's hardest start.
        model.eval()
        save_checkpoint(args.out, model, cfg, 0, explore_max_depth=0)
        print(f"  saved {args.out}: random initialisation, seed {args.seed}, no training")
        return 0
    data = SupervisionSet(anchored_data("phase1_train")).up_to_depth(args.max_depth)
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


def install_zero_ceiling() -> None:
    """``read --search ceiling0``: no UNPROVEN state is worth more than par.

    Under ``value_source: steps`` the library caps an estimate at 0 against exact
    par and at +1 against any other par, where finishing under par is possible.
    On the scripted strata that lets a hopeful estimate outrank a proven at-par
    solution. This caps the estimate at 0 everywhere; a solution the tree has
    actually found under par still scores +1, because that is a proof, not an
    estimate. Not a library key: an exploration of F-59's "registered, not done".
    """
    original = search_module._state_value

    def capped(answer, problem, total_steps, cfg):
        return min(0.0, original(answer, problem, total_steps, cfg))

    search_module._state_value = capped


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


class BootstrappedSteps:
    """Arm F: steps targets by one-step lookahead through the exact rules."""

    def __init__(self, cfg: Config, device: str) -> None:
        self.cfg = cfg
        self.device = device
        self.frozen: Reckoner | None = None
        self.rng = random.Random(0)

    def refresh(self, model: Reckoner) -> None:
        """Freeze the weights the lookahead reads: the ones this iteration started with."""
        self.frozen = copy.deepcopy(model).eval()

    def _lookahead(self, ring, indices) -> torch.Tensor:
        cfg, cap = self.cfg, float(self.cfg.episode.step_cap)
        owners: list[int] = []
        encoded = []
        best = [float("inf")] * len(indices)
        for row, i in enumerate(indices):
            record = ring.get(i)
            goal, target, expr = decode_state(tuple(int(t) for t in record["tokens"]))
            problem = Problem(
                goal=goal, expr=expr, par=int(record["par"]), target=target, par_source="bfs"
            )
            for rule_id, site_id in legal_actions(expr):
                child = apply(expr, rule_id, site_id)
                if verify(problem, child, cfg, self.rng):
                    best[row] = 1.0
                    continue
                try:
                    encoded.append(encode(problem, child, cfg))
                    owners.append(row)
                except StateTooLarge:
                    continue
        if encoded:
            assert self.frozen is not None
            predicted: list[float] = []
            with torch.no_grad():
                for start in range(0, len(encoded), 1024):
                    chunk = encoded[start : start + 1024]
                    sites = torch.stack([e.site_positions for e in chunk])
                    tokens = crop_to_content(torch.stack([e.tokens for e in chunk]), sites)
                    _, _, steps = self.frozen(tokens.to(self.device), sites.to(self.device))
                    predicted += steps.cpu().tolist()
            for row, to_go in zip(owners, predicted, strict=True):
                # An unsolved child is at least one step from solved, whatever the head says.
                best[row] = min(best[row], 1.0 + max(1.0, to_go))
        return torch.tensor([min(b, cap) if b != float("inf") else float("inf") for b in best])

    def install(self) -> None:
        ring_batch = train_module.ring_batch

        def bootstrapped(ring, indices, cfg, *, target=None):
            batch = ring_batch(ring, indices, cfg, target=target)
            if batch is None:
                return None
            if batch.skipped:
                raise RuntimeError("a ring row sat out, so the lookahead cannot be aligned")
            lookahead = self._lookahead(ring, indices)
            observed = torch.where(
                batch.solved_mask,
                batch.steps_target,
                torch.full_like(batch.steps_target, float("inf")),
            )
            target_steps = torch.minimum(observed, lookahead)
            known = torch.isfinite(target_steps)
            batch.steps_target = torch.where(known, target_steps, torch.zeros_like(target_steps))
            batch.solved_mask = known
            return batch

        train_module.ring_batch = bootstrapped


# ------------------------------------------------------------------ readings


def _cell(stats, source: str = "bfs") -> dict:
    cell = stats.z_by_par_source.get(source, {})
    return {
        "n": stats.episodes,
        "solved": stats.episodes_solved,
        "at_par": int(cell.get("0", 0)) + int(cell.get("+1", 0)),
        "under_par": int(cell.get("+1", 0)),
        "capped": stats.episodes_capped,
    }


def read_play(model, cfg: Config, make_evaluator, per_suite: int, sims: int, m: int) -> dict:
    """Episodes on frozen suite problems, measurement profile (no root noise).

    Each depth is played in two parts, the SOLVE-goal problems and the rest, so
    the stratum no training set holds (depth-6 SOLVE) can be read on its own.
    ``solve`` is that part; the top-level numbers are the two parts summed.
    """
    ev_cfg = eval_profile(cfg)
    evaluator = make_evaluator(model, ev_cfg)
    out = {}
    for depth in DEPTHS:
        problems = suite_problems(depth, per_suite)
        parts = {
            "solve": [p for p in problems if p.goal == GOAL_SOLVE],
            "other": [p for p in problems if p.goal != GOAL_SOLVE],
        }
        cells = {
            name: _cell(run_iteration(part, evaluator, ev_cfg, None, sims=sims, m=m, seed=0))
            for name, part in parts.items()
            if part
        }
        total = {
            key: sum(cell[key] for cell in cells.values())
            for key in ("n", "solved", "at_par", "capped")
        }
        out[str(depth)] = total | {
            "solve": cells.get("solve", {"n": 0, "solved": 0, "at_par": 0, "capped": 0})
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


def reading(model, cfg, device, make_evaluator, args, eval_set, optimal_sets, with_search) -> dict:
    model.eval()
    row = {
        "greedy": read_play(model, cfg, make_evaluator, args.eval_per_suite, 1, 1),
        "label_agreement": read_label_agreement(model, cfg, device, eval_set),
        "optimal_top": read_optimal(model, cfg, device, optimal_sets),
    }
    if with_search:
        row["search"] = read_play(
            model, cfg, make_evaluator, args.search_per_suite, cfg.search.sims, cfg.search.gumbel_m
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
    if args.arm in ("D", "E", "F"):
        overrides["value_loss_weight"] = 0.0
    cfg = replace(cfg, train=replace(cfg.train, **overrides))
    cfg = search_as(cfg, *SEARCH_OF_ARM[args.arm])
    if args.sims is not None:
        cfg = replace(cfg, search=replace(cfg.search, sims=args.sims))
    validate(cfg)

    if args.arm in ("C", "D", "E", "F", "S"):
        SolvedRows(args.capacity).install()
    bootstrap = BootstrappedSteps(cfg, args.device) if args.arm == "F" else None
    if bootstrap is not None:
        bootstrap.install()

    def make_evaluator(net, profile):
        return model_evaluator(net, profile, 0.0, device=args.device)

    extra = [suite_problem(row) for row in read_suite(args.extra)] if args.extra else []
    if bool(extra) != bool(args.extra_count):
        raise SystemExit("--extra and --extra-count go together")

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
            "sims": cfg.search.sims,
            "search": list(SEARCH_OF_ARM[args.arm]),
            "extra": [str(args.extra), args.extra_count] if args.extra else None,
            "reading": reading(
                model, cfg, args.device, make_evaluator, args, eval_set, optimal_sets, with_search
            ),
        }
        seconds = {"read": round(time.monotonic() - started, 1)}
        if n == args.iterations:
            row["seconds"] = seconds
            emit(row)
            break

        t0 = time.monotonic()
        stream = n + args.seed_offset * 1000
        problems = training_problems(
            anchored_data("train_100k"), args.episodes - args.extra_count, seed=stream
        )
        if extra:
            # The same number of episodes, some of them from the minted stratum.
            problems += random.Random(stream).sample(extra, args.extra_count)
        stats = run_iteration(problems, make_evaluator(model, cfg), cfg, ring, seed=stream)
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
        if bootstrap is not None:
            bootstrap.refresh(model)
        trained = train_on_ring(
            model, ring, cfg, steps=cfg.train.train_steps_per_iter, seed=stream, device=args.device
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


# ------------------------------------------------- the stratum training lacks


def _instrument_keys() -> set[tuple[int, ...]]:
    """Every problem a frozen instrument holds, and every one ``train_100k`` holds.

    A minted problem that matched an instrument would be training on the
    measuring stick; one that matched ``train_100k`` would not be new.
    """
    keys: set[tuple[int, ...]] = set()
    for path in sorted(SUITES.glob("*.jsonl")) + sorted((REPO / "runs" / "paired").glob("*.jsonl")):
        keys |= state_keys(read_suite(path))
    for name in ("eval_held_out", "train_100k"):
        keys |= dataset_keys(read_dataset(anchored_data(name)))
    return keys


def _mint_one(job: tuple[str, int]) -> dict | None:
    template, seed = job
    problem = generate(template, random.Random(seed), Config())
    return None if problem is None else suite_row(problem)


def cmd_mint(args: argparse.Namespace) -> int:
    if not args.out.resolve().is_relative_to((REPO / "runs" / "explore").resolve()):
        raise SystemExit("minted problems are exploration-only: --out must be under runs/explore/")
    started = time.monotonic()
    barred = _instrument_keys()
    rng = random.Random(args.seed)
    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    dropped = {"unlabelled": 0, "duplicate": 0, "in_an_instrument_or_train": 0}
    with multiprocessing.Pool(args.workers) as pool:
        while len(rows) < args.count:
            jobs = [(args.template, rng.randrange(2**62)) for _ in range(args.workers * 8)]
            for row in pool.map(_mint_one, jobs, chunksize=1):
                if row is None:
                    dropped["unlabelled"] += 1
                    continue
                key = problem_key(suite_problem(row))
                if key in barred:
                    dropped["in_an_instrument_or_train"] += 1
                elif key in seen:
                    dropped["duplicate"] += 1
                else:
                    seen.add(key)
                    rows.append(row)
            print(f"  {len(rows)} minted, {dropped}, {time.monotonic() - started:.0f}s", flush=True)
    rows = rows[: args.count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    pars: dict[int, int] = {}
    for row in rows:
        pars[row["par"]] = pars.get(row["par"], 0) + 1
    print(
        f"  wrote {args.out}: {len(rows)} problems from {args.template}, by par {sorted(pars.items())}"
    )
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    """One checkpoint, every suite, each named search. No training."""
    torch.set_num_threads(args.threads)
    base = Config()
    if args.sims is not None:
        base = replace(base, search=replace(base.search, sims=args.sims))
    model, _ = load_checkpoint(args.checkpoint, base)
    model.to(args.device).eval()
    suites = [f"solve_in_{d}" for d in DEPTHS] + list(MID_SUITES)
    report = {"checkpoint": str(args.checkpoint), "sims": base.search.sims, "searches": {}}
    if "ceiling0" in args.search and len(args.search) > 1:
        raise SystemExit("ceiling0 patches the search for the whole process: read it on its own")
    if args.search == ["ceiling0"]:
        install_zero_ceiling()
    for name in args.search:
        cfg = eval_profile(search_as(base, *SEARCHES[name]))
        validate(cfg)
        evaluator = model_evaluator(model, cfg, 0.0, device=args.device)
        rows = {}
        for suite in suites:
            all_rows = read_suite(SUITES / f"{suite}.jsonl")
            picked = sample_indices(len(all_rows), min(args.per_suite, len(all_rows)), 0)
            problems = [suite_problem(all_rows[i]) for i in picked]
            started = time.monotonic()
            stats = run_iteration(problems, evaluator, cfg, None, seed=0)
            source = "bfs" if suite.startswith("solve_in") else problems[0].par_source
            rows[suite] = _cell(stats, source) | {"seconds": round(time.monotonic() - started, 1)}
            cell = rows[suite]
            print(
                f"  [{name:>7}] {suite:<14} at par {cell['at_par']:>3}  under par "
                f"{cell['under_par']:>3}  solved {cell['solved']:>3}  capped {cell['capped']:>3}  "
                f"of {cell['n']}  ({cell['seconds']}s)",
                flush=True,
            )
        report["searches"][name] = rows
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
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
            if "solve" in greedy["1"]:
                line += "   SOLVE goal at par " + " ".join(
                    f"{greedy[str(d)]['solve']['at_par']:>3}/{greedy[str(d)]['solve']['n']:<3}"
                    for d in DEPTHS
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
    p.add_argument("--sims", type=int, default=None, help="override the search budget")
    p.add_argument("--seed-offset", type=int, default=0, help="shift every stream: a replicate")
    p.add_argument("--capacity", type=int, default=200_000)
    p.add_argument("--optimal-sets", type=Path, default=None)
    p.add_argument("--extra", type=Path, default=None, help="minted problems (see `mint`)")
    p.add_argument("--extra-count", type=int, default=0, help="episodes per iteration from --extra")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(run=cmd_loop)

    p = sub.add_parser("mint")
    p.add_argument("--template", choices=sorted(TEMPLATES), required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=cmd_mint)

    p = sub.add_parser("read")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--search", nargs="+", choices=sorted(SEARCHES), required=True)
    p.add_argument("--per-suite", type=int, default=200)
    p.add_argument("--sims", type=int, default=None, help="override the search budget")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(run=cmd_read)

    p = sub.add_parser("summary")
    p.add_argument("--root", type=Path, required=True)
    p.set_defaults(run=cmd_summary)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
