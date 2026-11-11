"""The Phase-2 episode runner: many episodes in lockstep, one pooled evaluator.

Batching is **across concurrent episode searches, never within one tree.** The
plan's "batched leaves" wording predates F-06 and is superseded: inside a tree,
simulation *k+1*'s selection depends on *k*'s backup, so pooling there changes
what the tree does. Across episodes there is no coupling, so parity with
sequential play stays an equality rather than a tolerance — and it is the shape
AGENTS.md §8 prescribes anyway. `search.batch_searches` is the key and the law.

Every live episode takes one search per round; their leaf requests are pooled by
:func:`reckoner.search.run_batched`; each applies its chosen action and the round
repeats until no episode is live.

The descent gate rides in
-------------------------
Chunk 7 measured `evaluations == nodes` at every budget and the forward
obligation was to assert it here. **The equality does not generalise, and the
exception has a name rather than a relaxation:** a *terminal* leaf creates a node
without an evaluation. The invariant that does hold universally, verified against
the search before being asserted, is

    nodes - evaluations
        == terminal_solved + terminal_no_actions + state_too_large + step_cap_reached

Chunk 7's `evals == nodes` is exactly its zero-terminal special case, which is why
the descent gate saw it: a branchy depth-5 SOLVE at 48 sims reaches no terminal.
Real episodes reach terminals constantly — that is the point of them — so
asserting the equality here would have failed on correct behaviour, and widening
it to an inequality would have asserted nothing. The identity is the honest form.

The entropy split
-----------------
`logschema` has four waiting columns: prior and target entropy, each on start
states and on reached states. Prior entropy alone cannot distinguish a confident
policy from a collapsed one — that is the chess lesson — so the search-improved
target is logged beside it. "Start" is an episode's first search; "reached" is
every search after at least one rewrite. That mapping is the faithful analog of
chess's start-vs-book split and is an interpretation, not a transplant: there is
no openings book here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from statistics import fmean

import numpy as np

from reckoner.config import Config
from reckoner.episode import Problem, encode_state, verify
from reckoner.expr import Expr
from reckoner.logschema import STEPS_MINUS_PAR_BINS
from reckoner.model import StateTooLarge, action_index, encode
from reckoner.replay import IMPROVED_TAIL_TOL, ReplayRing
from reckoner.rules import apply, legal_actions
from reckoner.search import Evaluator, SearchResult, run_batched


def entropy(distribution: np.ndarray) -> float:
    """Shannon entropy in nats. An empty or degenerate distribution is 0.0."""
    p = np.asarray(distribution, dtype=np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) in nats, float64. Terms with p == 0 contribute 0.

    The instrument that moves while the value head is silent (F-54): π′ equals
    the prior exactly where search found nothing, so H(π′) and H(prior) agree
    there and only a divergence can tell a root that taught something from one
    that did not. Raises on a non-finite result rather than logging one: p > 0
    where q == 0 means the target named an action the prior had no mass on.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    support = p > 0.0
    with np.errstate(divide="ignore"):
        value = float(np.sum(p[support] * (np.log(p[support]) - np.log(q[support]))))
    if not np.isfinite(value):
        raise ValueError(f"KL(π′ ‖ prior) is {value}: π′ has mass where the prior has none")
    return max(0.0, value)


@dataclass
class _Live:
    problem: Problem
    expr: Expr
    rng: random.Random
    steps: int = 0
    index: int = 0


@dataclass
class IterationStats:
    """Everything an iteration row needs, and the identity that guards it."""

    episodes: int = 0
    episodes_solved: int = 0
    episodes_capped: int = 0
    episodes_stuck: int = 0
    episodes_conceded: int = 0
    nodes: int = 0
    evaluations: int = 0
    terminal_solved: int = 0
    terminal_no_actions: int = 0
    state_too_large: int = 0
    step_cap_reached: int = 0
    solved_by_depth: dict[int, int] = field(default_factory=dict)
    seen_by_depth: dict[int, int] = field(default_factory=dict)
    z_by_par_source: dict[str, dict[str, int]] = field(default_factory=dict)
    steps_minus_par: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(STEPS_MINUS_PAR_BINS, 0)
    )
    h_prior_start: list[float] = field(default_factory=list)
    h_prior_reached: list[float] = field(default_factory=list)
    h_target_start: list[float] = field(default_factory=list)
    h_target_reached: list[float] = field(default_factory=list)
    # F-54: H(π′) is the target entropy; the visit-share entropy that column held
    # before is kept under its own name, and KL(π′ ‖ prior) is the instrument
    # that can tell an informative target from the prior when the entropies tie.
    h_visits_start: list[float] = field(default_factory=list)
    h_visits_reached: list[float] = field(default_factory=list)
    kl_start: list[float] = field(default_factory=list)
    kl_reached: list[float] = field(default_factory=list)
    #: Ring records whose π′ dropped more than IMPROVED_TAIL_TOL beyond the
    #: stored slots (F-54). Premise-zero: they sit out of policy training.
    ring_improved_truncated: int = 0
    seconds: float = 0.0

    @property
    def terminals(self) -> int:
        return (
            self.terminal_solved
            + self.terminal_no_actions
            + self.state_too_large
            + self.step_cap_reached
        )

    def check_descent_identity(self) -> None:
        """`nodes - evaluations == terminals`. Chunk 7's forward obligation.

        Raises rather than warns: a violation means a node was created without
        being either evaluated or resolved, which is the F-06 family — a search
        doing something other than searching, reported as if it had.
        """
        if self.nodes - self.evaluations != self.terminals:
            raise AssertionError(
                f"descent identity violated: nodes {self.nodes} - evaluations "
                f"{self.evaluations} = {self.nodes - self.evaluations}, but terminals "
                f"= {self.terminals} (solved {self.terminal_solved}, no-actions "
                f"{self.terminal_no_actions}, too-large {self.state_too_large}, cap "
                f"{self.step_cap_reached}). Every node is evaluated exactly once "
                "unless it is terminal; a mismatch means one was neither."
            )

    def solve_rate_by_depth(self) -> dict[str, float]:
        return {
            str(d): round(self.solved_by_depth.get(d, 0) / n, 6)
            for d, n in sorted(self.seen_by_depth.items())
        }

    def entropies(self) -> dict[str, float]:
        """Means, or 0.0 for a population that did not occur.

        A 0.0 here is a *premise-dependent* reading, not a measured entropy: an
        iteration with no reached states has no reached entropy. The runner
        surfaces the population sizes beside it so the row's reader can tell.
        """

        def mean(values: list[float]) -> float:
            return round(fmean(values), 6) if values else 0.0

        return {
            "entropy_visits_step1_start": mean(self.h_visits_start),
            "entropy_visits_step1_reached": mean(self.h_visits_reached),
            "kl_improved_prior_step1_start": mean(self.kl_start),
            "kl_improved_prior_step1_reached": mean(self.kl_reached),
            "entropy_prior_step1_start": round(fmean(self.h_prior_start), 6)
            if self.h_prior_start
            else 0.0,
            "entropy_prior_step1_reached": round(fmean(self.h_prior_reached), 6)
            if self.h_prior_reached
            else 0.0,
            "entropy_target_step1_start": round(fmean(self.h_target_start), 6)
            if self.h_target_start
            else 0.0,
            "entropy_target_step1_reached": round(fmean(self.h_target_reached), 6)
            if self.h_target_reached
            else 0.0,
        }


def _bin_for(delta: int) -> str:
    if delta < 0:
        return "<0"
    return str(delta) if delta <= 5 else "6+"


def _record(stats: IterationStats, result: SearchResult, live: _Live) -> None:
    """Fold one search's numbers into the iteration, including the H split."""
    s = result.stats
    stats.nodes += s.nodes
    stats.evaluations += s.evaluations
    stats.terminal_solved += s.terminal_solved
    stats.terminal_no_actions += s.terminal_no_actions
    stats.state_too_large += s.state_too_large
    stats.step_cap_reached += s.step_cap_reached

    h_prior = entropy(result.root_priors)
    h_target = entropy(result.improved)
    h_visits = entropy(result.visit_share())
    # Compared at the TARGET's precision: π′ is stored float32, and where it is
    # the prior both come from the identical float64 softmax, so the float32
    # values agree exactly and the divergence is exactly 0 — not float noise.
    # Where the float32 cast of the prior underflows to 0 while the float64
    # prior is positive, the float64 value is used: a precision artefact must
    # not read as a target naming an action the prior has no mass on, and must
    # never halt self-play over a diagnostic column (F-54 review).
    prior64 = np.asarray(result.root_priors, dtype=np.float64)
    prior32 = prior64.astype(np.float32).astype(np.float64)
    reference = np.where(prior32 > 0.0, prior32, prior64)
    kl = kl_divergence(result.improved, reference) if len(result.actions) else 0.0
    if live.steps == 0:
        stats.h_prior_start.append(h_prior)
        stats.h_target_start.append(h_target)
        stats.h_visits_start.append(h_visits)
        stats.kl_start.append(kl)
    else:
        stats.h_prior_reached.append(h_prior)
        stats.h_target_reached.append(h_target)
        stats.h_visits_reached.append(h_visits)
        stats.kl_reached.append(kl)


def run_iteration(
    problems: list[Problem],
    evaluator: Evaluator,
    cfg: Config,
    ring: ReplayRing | None = None,
    *,
    sims: int | None = None,
    m: int | None = None,
    seed: int = 0,
    batch_size: int | None = None,
) -> IterationStats:
    """Play every problem to a terminal outcome, pooling searches across episodes.

    Seeding is a **per-episode, per-step fan-out**, never one shared stream —
    F-11's precedent generalised: a null computed with shared randomness is not a
    null, and an episode set driven by one stream is not a sample of independent
    episodes.
    """
    sims = sims if sims is not None else cfg.search.sims
    m = m if m is not None else cfg.search.gumbel_m
    stats = IterationStats(episodes=len(problems))
    started = time.perf_counter()

    live = [
        _Live(problem=p, expr=p.expr, rng=random.Random(seed * 100_003 + i), index=i)
        for i, p in enumerate(problems)
    ]
    for problem in problems:
        stats.seen_by_depth[problem.par or 0] = stats.seen_by_depth.get(problem.par or 0, 0) + 1

    trail: dict[int, list] = {e.index: [] for e in live}

    while live:
        items = [
            (e.problem, e.expr, random.Random(seed * 100_003 + e.index * 977 + e.steps), e.steps)
            for e in live
        ]
        results = run_batched(items, evaluator, cfg, sims=sims, m=m, batch_size=batch_size)

        still: list[_Live] = []
        for e, result in zip(live, results, strict=True):
            _record(stats, result, e)

            if result.chosen is None:
                stats.episodes_stuck += 1
                _settle(stats, ring, e, trail, cfg, solved=False, capped=False)
                continue

            trail[e.index].append((result, e.expr, e.steps))
            e.expr = apply(e.expr, *result.chosen)
            e.steps += 1

            if verify(e.problem, e.expr, cfg, e.rng):
                stats.episodes_solved += 1
                _settle(stats, ring, e, trail, cfg, solved=True, capped=False)
                continue
            # RESIGN-VS-PAR, implemented and default OFF. Once the best line
            # already needs >= par + concede_k, the outcome is settled and the
            # remaining budget buys nothing. Calibration of k is deferred to
            # campaign evidence per v1.1 — it is inert until enabled, and the
            # counter is distinct because conceding is a DECISION where capping
            # is an exhaustion.
            if (
                cfg.par.concede_enabled
                and e.problem.par is not None
                and e.steps >= e.problem.par + cfg.par.concede_k
            ):
                stats.episodes_conceded += 1
                _settle(stats, ring, e, trail, cfg, solved=False, capped=False)
                continue
            if e.steps >= cfg.episode.step_cap or not legal_actions(e.expr):
                capped = e.steps >= cfg.episode.step_cap
                if capped:
                    stats.episodes_capped += 1
                else:
                    stats.episodes_stuck += 1
                _settle(stats, ring, e, trail, cfg, solved=False, capped=capped)
                continue
            still.append(e)
        live = still

    stats.seconds = round(time.perf_counter() - started, 3)
    stats.check_descent_identity()
    return stats


def ring_visits(result: SearchResult, width: int, max_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """A root's top-``width`` visits as ``(dense action indices, counts)``.

    Two coordinate systems meet here, and this is the one place they are
    converted. ``result.visits`` is indexed by **root slot** — position in
    ``result.actions``, which is ``legal_actions(expr)`` order. The ring stores
    **dense action indices** (``rule * max_sites + site``), the layout the policy
    head emits and ``train.ring_batch`` expands into.

    Storing the slot was F-53: slot j can only ever reach cell j, which is
    ``eval_add`` at site j, so 93% of stored visit mass landed on illegal cells
    and the policy loss dropped it without a sound, for all of M1.
    """
    order = np.argsort(-result.visits)[:width]
    return _dense_actions(result.actions, order, max_sites), result.visits[order].astype(np.int32)


def _dense_actions(actions: list[tuple[int, int]], slots: np.ndarray, max_sites: int) -> np.ndarray:
    """Root slots → dense action indices. **The one conversion** (F-53's rule)."""
    return np.fromiter(
        (action_index(*actions[int(slot)], max_sites) for slot in slots),
        dtype=np.int32,
        count=len(slots),
    )


def ring_improved(
    result: SearchResult, width: int, max_sites: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """π′'s top-``width`` actions as ``(dense indices, probs, tail)`` (F-54).

    Sorted by mass, ties by slot (stable), padded with action 0 / mass 0.0. The
    tail is the float64 sum of the π′ entries NOT stored — exactly 0.0 whenever
    the root has at most ``width`` legal actions — so a reader can always tell a
    complete target from a cropped one, and never has to guess.
    """
    pi = np.asarray(result.improved)
    if len(pi) != len(result.actions):
        raise ValueError(
            f"π′ has {len(pi)} entries for {len(result.actions)} root actions; it must "
            "be aligned with them slot for slot"
        )
    order = np.argsort(-pi, kind="stable")
    kept = order[:width]
    actions = np.zeros(width, dtype=np.int32)
    probs = np.zeros(width, dtype=np.float32)
    actions[: len(kept)] = _dense_actions(result.actions, kept, max_sites)
    probs[: len(kept)] = pi[kept]
    tail = float(pi[order[width:]].astype(np.float64).sum())
    return actions, probs, tail


def _settle(
    stats: IterationStats,
    ring: ReplayRing | None,
    e: _Live,
    trail: dict[int, list],
    cfg: Config,
    *,
    solved: bool,
    capped: bool,
) -> None:
    """Score the episode and write its steps into the ring, z known at last."""
    par = e.problem.par or 0
    if solved:
        delta = e.steps - par
        z = 1 if delta < 0 else (0 if delta == 0 else -1)
        stats.steps_minus_par[_bin_for(delta)] += 1
        stats.solved_by_depth[par] = stats.solved_by_depth.get(par, 0) + 1
    else:
        z = -1

    source = e.problem.par_source
    cell = stats.z_by_par_source.setdefault(source, {"+1": 0, "0": 0, "-1": 0})
    cell[{1: "+1", 0: "0", -1: "-1"}[z]] += 1

    if ring is None:
        return
    for result, expr, steps in trail[e.index]:
        # THE STATE ITSELF. An earlier version stored empty token arrays here —
        # the ring filled with visits, z and root_q for states it did not
        # contain, so every row was untrainable and `len(ring) > 0` still passed.
        # Rider (a) at the ring boundary: "received rows" is not "received steps".
        try:
            encoded = encode(e.problem, expr, cfg)
        except StateTooLarge:
            continue  # counted upstream; never cropped
        seq = np.asarray(encode_state(e.problem.goal, expr, e.problem.target), dtype=np.int16)
        visit_actions, visit_counts = ring_visits(
            result, ring.visit_actions.shape[1], cfg.model.max_sites
        )
        # Both targets, always: the ring serves either arm of train.policy_target
        # (F-54), and this writer reads no training config.
        improved_actions, improved_probs, improved_tail = ring_improved(
            result, ring.improved_actions.shape[1], cfg.model.max_sites
        )
        if improved_tail > IMPROVED_TAIL_TOL:
            stats.ring_improved_truncated += 1
        ring.append(
            tokens=seq,
            site_positions=np.asarray(encoded.site_positions[: encoded.n_sites], dtype=np.int16),
            visit_actions=visit_actions,
            visit_counts=visit_counts,
            root_q=result.root_value,
            z=z,
            par_source=source,
            par=par,
            steps_remaining=max(0, e.steps - steps),
            depth=par,
            goal=e.problem.goal,
            improved_actions=improved_actions,
            improved_probs=improved_probs,
            improved_tail=improved_tail,
        )


def iteration_row(
    stats: IterationStats,
    *,
    iteration: int,
    run_name: str,
    git_sha: str,
    config_fingerprint: str,
    cfg: Config,
    ruleset_version: int,
    vocab_version: int,
    schema_era: int,
    evaluator_checkpoint_sha256: str,
    pool_composition: dict,
    seconds_train: float = 0.0,
    absent: dict[str, str] | None = None,
) -> dict:
    """Assemble a `logschema` row. The schema validates it; this only fills it.

    **`absent` is the caller's claim about the caller's run, and this function
    does not invent one** (F-29). It used to: an `absent or {...}` default filed
    ``pool_par_fraction`` and ``ladder_pass`` as absent whenever a caller passed
    nothing — including when the caller passed an empty mapping *meaning* nothing
    is absent. `or` cannot tell "unspecified" from "specified as empty", and here
    the empty case is the meaningful one: it says every column has a value.

    The campaign hit it at the first iteration where that was true — a cadence
    iteration with a live pool, where the ladder fires and pool par is drawn, so
    neither column is absent. The default then named both while the driver had
    just written both, and the row was refused after the cadence had been paid
    for. Two chunks of golden runs never reached it, because golden's
    ``ladder_every = 99`` guarantees ``ladder_pass`` is always absent.

    An empty mapping now means what it says.
    """
    return {
        "evaluator_checkpoint_sha256": evaluator_checkpoint_sha256,
        "pool_composition": pool_composition,
        "iteration": iteration,
        "schema_era": schema_era,
        "run_name": run_name,
        "git_sha": git_sha,
        "config_fingerprint": config_fingerprint,
        "ruleset_version": ruleset_version,
        "vocab_version": vocab_version,
        "measure_dtype": cfg.numerics.measure_dtype,
        "train_dtype": cfg.numerics.train_dtype,
        "solve_rate_by_depth": stats.solve_rate_by_depth(),
        "z_by_par_source": stats.z_by_par_source,
        "steps_minus_par_histogram": stats.steps_minus_par,
        "episodes": stats.episodes,
        "episodes_solved": stats.episodes_solved,
        "episodes_capped": stats.episodes_capped,
        "episodes_stuck": stats.episodes_stuck,
        "episodes_conceded": stats.episodes_conceded,
        "search_nodes_total": stats.nodes,
        "search_evaluations_total": stats.evaluations,
        "terminal_no_actions": stats.terminal_no_actions,
        "state_too_large": stats.state_too_large,
        "ring_improved_truncated": stats.ring_improved_truncated,
        "nan_skips": 0,
        "pool_refusals": 0,
        "seconds_self_play": stats.seconds,
        "seconds_train": seconds_train,
        "seconds_total": round(stats.seconds + seconds_train, 3),
        "absent": dict(absent) if absent is not None else {},
        **stats.entropies(),
    }


__all__ = [
    "IterationStats",
    "entropy",
    "iteration_row",
    "kl_divergence",
    "ring_improved",
    "ring_visits",
    "run_iteration",
]
