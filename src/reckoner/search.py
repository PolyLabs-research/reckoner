"""Search: array-tree MCTS, Gumbel root, Sequential Halving — the careful port.

There is no opponent, and that changes two things
-------------------------------------------------
**Backup does not negate per ply.** The sharpest porting hazard in the project
(plan §8 decision 5), pinned by a config key ``config.validate()`` refuses to set
to anything else.

**Backup takes the max, not the mean.** With no opponent there is nothing to
average over: every choice below a node is ours, so a node is worth the best
thing reachable from it. Mean backup would be quietly pessimistic — averaging a
winning line against siblings we would never pick. ``tests/test_search.py``
hand-computes a mixed three-level tree, because "no negation" and "max not mean"
are two distinct mistakes and only one of them is loud.

**Proof-directed breadth is N/A** and is deliberately not ported: it exists to
force a proof through every opponent reply, and there are no replies here.

The tree must actually deepen
-----------------------------
The first version of this module expanded only the root's children and then
re-backed-up the same values: 48 simulations produced **2 nodes**
(``FINDINGS.md`` F-06). Every chunk-7 gate passed anyway — a depth-1 suite needs
one ply, budget identity counts visits regardless, and sync-vs-batched compared
two equally shallow paths — so the gates were all satisfiable by a search that
does not search. ``stats.max_depth`` and ``stats.nodes`` are now reported, and a
test asserts that ``sims >> m`` builds a tree deeper than one ply.

Where the batching lives, and why it is there
---------------------------------------------
**Batching is across concurrent searches, never within one search.** Inside one
tree, simulation *k+1*'s selection depends on simulation *k*'s backup, so pooling
leaves within a tree would change what the tree does — and the equivalence test
would then be comparing two algorithms and would have to be weakened to a
tolerance. Batching across trees has no such coupling, so sync-vs-batched parity
is **exact by construction**, and it is the shape AGENTS.md §8 prescribes anyway:
many CPU-side actors feeding one process that owns the GPU.

``_simulate`` is therefore a generator that *yields* leaf requests and receives
evaluations. :func:`search` drives one; :func:`run_batched` drives many and pools
their requests.

Terminal states
---------------
A terminal leaf is not expanded and its value is its outcome, but **it still
consumes a simulation** — budget means budget.

``StateTooLarge`` is a **counted terminal loss**, step-cap-equivalent semantics.
Never a crash mid-batch, and never an edit to the legality mask — legality stays
the engine's alone. ``add_both_sides`` makes it reachable from the first stub
rollout (F-05), so the counter is in ``SearchStats`` from field one.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

import numpy as np

from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.expr import Expr
from reckoner.model import StateTooLarge, action_index, encode
from reckoner.rules import apply, legal_actions

Evaluation = tuple[np.ndarray, float]
Evaluator = Callable[[Sequence[tuple[Problem, Expr]]], list[Evaluation]]


@dataclass
class SearchStats:
    """What a search spent and what it met. Absence carries a reason."""

    sims_requested: int = 0
    sims_used: int = 0
    step_cap_reached: int = 0
    nodes: int = 0
    max_depth: int = 0
    evaluations: int = 0
    terminal_solved: int = 0
    terminal_no_actions: int = 0
    state_too_large: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class SearchResult:
    """One root search. ``visits`` and ``values`` are indexed by ROOT SLOT, the
    position in ``actions`` (``legal_actions`` order). ``values`` holds -1.0 where
    ``visits == 0``: that is the initialiser, not a value — mask by visits."""

    visits: np.ndarray
    values: np.ndarray
    actions: list[tuple[int, int]]
    chosen: tuple[int, int] | None
    root_value: float
    stats: SearchStats = field(default_factory=SearchStats)
    #: Softmaxed network prior over the LEGAL actions at the root. Needed for the
    #: H_prior half of logschema's entropy split — prior entropy alone cannot
    #: distinguish a confident policy from a collapsed one, so both halves are
    #: logged and both need their source exposed here. Declared AFTER `stats`:
    #: every existing call site passes `stats` sixth positionally, and inserting
    #: a defaulted field ahead of it silently rebinds that argument.
    root_priors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: **The Gumbel improved policy π′** over every legal root action, slot-indexed
    #: like ``actions``, float32, summing to 1 — the Phase-2 policy target under
    #: ``train.policy_target = 'improved'`` (F-42). softmax(logits + σ(completed
    #: Q)): visited actions completed by their own value, unvisited ones by v_mix.
    #: Required and keyword-only: a default would be a provenance default claiming
    #: "computed" (F-02), and a positional slot would be rebound by the next field
    #: someone inserts ahead of it (see ``root_priors``).
    improved: np.ndarray = field(kw_only=True)

    def visit_share(self) -> np.ndarray:
        """The Sequential-Halving visit share, ``visits / visits.sum()``.

        Named for what it is. Before F-42 this method was called
        ``improved_policy`` and was the training target; its shape is a function
        of the visited count alone, which is why it is not the improved policy and
        why it survives only as the ``'visits'`` control arm.
        """
        total = self.visits.sum()
        if total == 0:
            return np.full(len(self.actions), 1.0 / max(1, len(self.actions)), dtype=np.float32)
        return (self.visits / total).astype(np.float32)


class _Tree:
    """Per-node stats in arrays; per-node child stats sized by the legal count."""

    __slots__ = (
        "visits",
        "value",
        "own_eval",
        "proven",
        "terminal",
        "depth",
        "parent",
        "from_slot",
        "expr",
        "actions",
        "priors",
        "children",
        "child_visits",
        "child_values",
        "size",
    )

    def __init__(self, capacity: int) -> None:
        self.visits = np.zeros(capacity, dtype=np.int32)
        self.value = np.full(capacity, -1.0, dtype=np.float32)
        # The node's OWN evaluation, kept apart from its value. It is a proxy for
        # what unexplored actions might yield, and it participates in the max only
        # while such actions exist — see `_recompute`.
        self.own_eval = np.full(capacity, -1.0, dtype=np.float32)
        # A node whose value is a PROOF, not an estimate: a terminal, or a fully
        # expanded node all of whose children are proven.
        self.proven = np.zeros(capacity, dtype=bool)
        self.terminal = np.zeros(capacity, dtype=bool)
        self.depth = np.zeros(capacity, dtype=np.int32)
        self.parent = np.full(capacity, -1, dtype=np.int32)
        self.from_slot = np.full(capacity, -1, dtype=np.int32)
        self.expr: list[Expr | None] = [None] * capacity
        self.actions: list[list[tuple[int, int]]] = [[] for _ in range(capacity)]
        self.priors: list[np.ndarray | None] = [None] * capacity
        self.children: list[np.ndarray | None] = [None] * capacity
        self.child_visits: list[np.ndarray | None] = [None] * capacity
        self.child_values: list[np.ndarray | None] = [None] * capacity
        self.size = 0

    def add(self, expr: Expr, parent: int, slot: int) -> int:
        index = self.size
        self.size += 1
        self.expr[index] = expr
        self.parent[index] = parent
        self.from_slot[index] = slot
        self.depth[index] = 0 if parent < 0 else int(self.depth[parent]) + 1
        return index

    def open(self, node: int, actions: list[tuple[int, int]], priors: np.ndarray) -> None:
        self.actions[node] = actions
        self.priors[node] = priors
        self.children[node] = np.full(len(actions), -1, dtype=np.int32)
        self.child_visits[node] = np.zeros(len(actions), dtype=np.int32)
        self.child_values[node] = np.full(len(actions), -1.0, dtype=np.float32)


def _recompute(tree: _Tree, node: int) -> None:
    """A node's value from its children, its own estimate, and its expansion state.

    **Own-eval has exactly one legitimate role: a proxy for actions not yet
    tried.** While a node still has unexpanded actions, its own estimate stands in
    for what those might yield, and taking part in the max is epistemically sound.
    Once the legal set is fully expanded there is nothing left for it to
    represent, and a fully expanded node is worth ``max(children)`` alone.

    Letting own-eval floor a fully expanded node is a **stale prior outranking a
    proof** — the same defect family as chess's FINALE bug, which was proven
    evidence mishandled at action *selection*; this is the same thing at value
    *aggregation* (`FINDINGS.md` F-13). With a trained evaluator it is worse than
    untidy: ``root_q`` could never report a position as worse than the net already
    believed, so the net's belief would floor its own MSE target — self-referential
    optimism aimed at exactly the half of the blend the currency ruling exists to
    keep from fighting the other half.

    **Proofs propagate.** A terminal's z is a proof; a fully expanded node whose
    children are all proven is itself proven at ``max(children)``. That is the
    single-agent MCTS-Solver, in a max tree, in a paragraph.
    """
    if tree.terminal[node]:
        tree.proven[node] = True
        return
    children = tree.children[node]
    if children is None:  # opened for nothing, or not yet opened
        tree.value[node] = tree.own_eval[node]
        tree.proven[node] = False
        return

    best = -np.inf
    unexpanded = False
    all_proven = True
    for child in children:
        index = int(child)
        if index == -1:
            unexpanded = True
            all_proven = False
            continue
        best = max(best, float(tree.value[index]))
        if not tree.proven[index]:
            all_proven = False

    if unexpanded:
        best = max(best, float(tree.own_eval[node]))
    if best == -np.inf:
        best = float(tree.own_eval[node])

    tree.value[node] = best
    tree.proven[node] = bool(all_proven and len(children) > 0)


def _backup(tree: _Tree, node: int, value: float) -> None:
    """To the root: visits increment, values **recompute**, nothing negates.

    The leaf's value is set from ``value``; every ancestor is recomputed rather
    than max-accumulated, because "fully expanded" is a property that changes as
    the tree grows and a running max cannot un-floor itself once it has taken an
    own-eval it should no longer be holding.
    """
    if not tree.terminal[node]:
        tree.own_eval[node] = value
    tree.value[node] = value if tree.terminal[node] else max(value, float(tree.value[node]))
    _recompute(tree, node)

    current = node
    while current != -1:
        tree.visits[current] += 1
        parent = int(tree.parent[current])
        if parent != -1:
            slot = int(tree.from_slot[current])
            tree.child_visits[parent][slot] += 1  # type: ignore[index]
            _recompute(tree, parent)
            tree.child_values[parent][slot] = tree.value[current]  # type: ignore[index]
        current = parent


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max())
    return exp / exp.sum()


def _sigma(q: np.ndarray, max_visits: int, cfg: Config) -> np.ndarray:
    return (cfg.search.c_visit + max_visits) * cfg.search.c_scale * q


def _gumbel_improved(
    priors: np.ndarray, completed: np.ndarray, max_visits: int, cfg: Config
) -> np.ndarray:
    """softmax(priors + σ(completed)) — **the one implementation** of the formula.

    The non-root rule and the root target both call it, each passing its own
    completion. No dtype coercion here: the non-root rule must stay bit-identical
    to what it was before F-42, so the caller's dtypes are the caller's.
    """
    return _softmax(priors + _sigma(completed, max_visits, cfg))


def _improved_target(
    logits: np.ndarray, visits: np.ndarray, values: np.ndarray, v_hat: float, cfg: Config
) -> np.ndarray:
    """The root's π′ over EVERY legal action, slot-indexed, float32 (F-42).

    Visited actions are completed by their own (max-backed) value; that is the
    value Sequential Halving ranked them by, and a terminal child's is its exact
    z. **Unvisited actions are completed by v_mix** (Danihelka et al. 2022, eq.
    33): the network's own root estimate v̂ mixed with the prior-weighted mean of
    the visited values,

        v_mix = (v̂ + ΣN · Σ_visited softmax(logits)_a · q_a) / (1 + ΣN).

    Not the root's own value: that is a max, so one proof would complete every
    unexplored action at the proof's level and π′ would fall back to the prior.
    Not v̂ alone: that is 0 while the value head is silent, which ranks an unknown
    action level with a proven solve and well above a proven loss it might be.

    Visited is ``visits > 0`` and nothing else. The -1.0 initialiser of unvisited
    child values is also a legitimate proven-loss value, so no value test can
    tell them apart, and an unvisited slot's value is never read.

    Pure: no rng, no config beyond σ's constants. The Gumbel draw never enters.
    """
    visited = visits > 0
    total = int(visits.sum())
    q = np.where(visited, values.astype(np.float64), 0.0)
    if total == 0:
        v_mix = float(v_hat)
    else:
        weights = _softmax(logits[visited].astype(np.float64))
        v_mix = (float(v_hat) + total * float(np.dot(weights, q[visited]))) / (1 + total)
    completed = np.where(visited, q, v_mix)
    return _gumbel_improved(
        logits.astype(np.float64), completed, int(visits.max(initial=0)), cfg
    ).astype(np.float32)


def _root_improved(tree: _Tree, root: int, logits: np.ndarray, cfg: Config) -> np.ndarray:
    """π′ at the root, after the structural tripwire.

    Every root child is created and backed up within the same simulation, so a
    child exists exactly where it has visits. If that ever fails, the visited
    mask and the tree disagree, and the completion would read an initialiser as
    a value — refused here rather than trained on.
    """
    children, visits = tree.children[root], tree.child_visits[root]
    values = tree.child_values[root]
    assert children is not None and visits is not None and values is not None
    if not np.array_equal(children >= 0, visits > 0):
        raise RuntimeError(
            "root children and root visits disagree: a child exists without being "
            "visited, or was visited without existing. The visited mask is the only "
            "thing that separates a value from the -1.0 initialiser (F-42)."
        )
    return _improved_target(logits, visits, values, float(tree.own_eval[root]), cfg)


def _select(tree: _Tree, node: int, cfg: Config) -> int:
    """Gumbel-AZ's non-root rule: follow the improved policy, minus visit share.

    Its completion is the node's own (max-backed) value — unchanged by F-42,
    which moved only the root's training target. Unifying the two completions
    would change the search, a second lever, and is registered separately.
    """
    priors, visits, values = tree.priors[node], tree.child_visits[node], tree.child_values[node]
    assert priors is not None and visits is not None and values is not None
    completed = np.where(visits > 0, values, float(tree.value[node]))
    improved = _gumbel_improved(priors, completed, int(visits.max(initial=0)), cfg)
    return int(np.argmax(improved - visits / (1 + visits.sum())))


def _leaf_outcome(
    problem: Problem,
    expr: Expr,
    cfg: Config,
    rng: random.Random,
    stats: SearchStats,
    *,
    total_steps: int,
) -> float | None:
    """Terminal value **on the z scale**, or ``None`` if the leaf needs evaluating.

    **One currency (spec §6).** A solved leaf is not worth ``+1.0`` because it is
    solved — it is worth what the *par game* pays for solving it in
    ``total_steps``: ``+1`` under par, ``0`` at par, ``-1`` over par. A flat
    ``+1.0`` for every solve makes the searcher indifferent between an over-par
    solve and an under-par one, which means it cannot race, and the par game's
    whole premise dies at the backup rule while every test stays green
    (`FINDINGS.md` F-13).

    This is chess's FINALE bug in this project's terms: there, several proven
    wins tied at ``+1`` and the engine chose mate-in-5 over mate-in-2. Here the
    tie is between solutions of different lengths, and the thing that goes
    missing is the only quantity the campaign measures.

    ``total_steps`` is steps already taken in the episode plus the leaf's depth in
    this tree — the real step count of the line, not the tree-local one.
    """
    if problem.par is None:
        raise ValueError(
            "search needs par: the in-tree terminal value is z against par, and a "
            "problem without par cannot be scored on the scale the loop trains on."
        )
    if verify(problem, expr, cfg, rng):
        stats.terminal_solved += 1
        if total_steps < problem.par:
            return 1.0
        return 0.0 if total_steps == problem.par else -1.0
    try:
        encode(problem, expr, cfg)
    except StateTooLarge:
        stats.state_too_large += 1
        return -1.0
    if not legal_actions(expr):
        stats.terminal_no_actions += 1
        return -1.0
    if total_steps >= cfg.episode.step_cap:
        # The cap is a loss, identically to going over par (plan chunk 3). A leaf
        # at the cap cannot be continued, so it is terminal and it is -1.
        stats.step_cap_reached += 1
        return -1.0
    return None


def _priors_for(raw: np.ndarray, actions: list[tuple[int, int]], cfg: Config) -> np.ndarray:
    return np.array(
        [raw[action_index(r, s, cfg.model.max_sites)] for r, s in actions], dtype=np.float32
    )


def _simulate(
    problem: Problem,
    expr: Expr,
    cfg: Config,
    rng: random.Random,
    sims: int,
    m: int,
    steps_taken: int = 0,
) -> Generator[tuple[Problem, Expr], Evaluation, SearchResult]:
    """Yields leaves needing evaluation; returns the finished result."""
    stats = SearchStats(sims_requested=sims)
    tree = _Tree(sims + 4)
    root = tree.add(expr, -1, -1)
    stats.nodes = 1

    root_actions = legal_actions(expr)
    if not root_actions:
        stats.terminal_no_actions += 1
        return SearchResult(
            np.zeros(0, np.int32),
            np.zeros(0, np.float32),
            [],
            None,
            -1.0,
            stats,
            improved=np.zeros(0, np.float32),
        )

    raw, root_value = yield (problem, expr)
    stats.evaluations += 1
    logits = _priors_for(raw, root_actions, cfg).astype(np.float64)
    tree.open(root, root_actions, logits.astype(np.float32))
    tree.own_eval[root] = root_value
    tree.value[root] = root_value

    gumbel = (
        np.array([-np.log(-np.log(rng.random())) for _ in root_actions])
        if cfg.search.root_noise
        else np.zeros(len(root_actions))
    )
    considered = list(np.argsort(-(logits + gumbel))[: min(m, len(root_actions))])

    budget, remaining = sims, list(considered)
    while budget > 0 and remaining:
        rounds_left = max(1, int(np.ceil(np.log2(max(2, len(remaining))))))
        per_action = max(1, budget // (rounds_left * len(remaining)))
        for slot in remaining:
            for _ in range(per_action):
                if budget <= 0:
                    break
                budget -= 1
                stats.sims_used += 1

                node, chosen = root, slot
                while True:
                    child = int(tree.children[node][chosen])  # type: ignore[index]
                    if child == -1:
                        rule_id, site_id = tree.actions[node][chosen]
                        successor = apply(tree.expr[node], rule_id, site_id)  # type: ignore[arg-type]
                        child = tree.add(successor, node, chosen)
                        tree.children[node][chosen] = child  # type: ignore[index]
                        stats.nodes += 1
                        stats.max_depth = max(stats.max_depth, int(tree.depth[child]))
                        outcome = _leaf_outcome(
                            problem,
                            successor,
                            cfg,
                            rng,
                            stats,
                            total_steps=steps_taken + int(tree.depth[child]),
                        )
                        if outcome is not None:
                            tree.terminal[child] = True
                            tree.value[child] = outcome
                            _backup(tree, child, outcome)
                        else:
                            child_raw, child_value = yield (problem, successor)
                            stats.evaluations += 1
                            child_actions = legal_actions(successor)
                            tree.open(
                                child, child_actions, _priors_for(child_raw, child_actions, cfg)
                            )
                            _backup(tree, child, float(child_value))
                        break
                    if tree.terminal[child]:
                        _backup(tree, child, float(tree.value[child]))
                        break
                    node = child
                    chosen = _select(tree, node, cfg)

        visits, values = tree.child_visits[root], tree.child_values[root]
        assert visits is not None and values is not None
        if len(remaining) == 1:
            break
        remaining = sorted(
            remaining,
            key=lambda s: (
                logits[s] + gumbel[s] + _sigma(values[s : s + 1], int(visits.max()), cfg)[0]
            ),
            reverse=True,
        )[: max(1, len(remaining) // 2)]

    visits, values = tree.child_visits[root], tree.child_values[root]
    assert visits is not None and values is not None
    best = max(
        considered,
        key=lambda s: logits[s] + gumbel[s] + _sigma(values[s : s + 1], int(visits.max()), cfg)[0],
    )
    # π′ is computed AFTER the act and feeds nothing back into it: the search's
    # visits, values and choice are exactly what they were before F-42.
    improved = _root_improved(tree, root, logits, cfg)
    return SearchResult(
        visits.copy(),
        values.copy(),
        root_actions,
        root_actions[best],
        float(tree.value[root]),
        stats,
        root_priors=_softmax(logits),
        improved=improved,
    )


def search(
    problem: Problem,
    expr: Expr,
    evaluator: Evaluator,
    cfg: Config,
    rng: random.Random,
    *,
    sims: int | None = None,
    m: int | None = None,
    steps_taken: int = 0,
) -> SearchResult:
    """One search, evaluating one leaf at a time.

    ``steps_taken`` is how many steps the episode has already spent. It is not
    cosmetic: the in-tree terminal value is z against par, so a leaf's worth
    depends on the total line length, not the tree-local depth. Defaulting it to
    0 is correct only for a search from a problem's start state.
    """
    generator = _simulate(
        problem,
        expr,
        cfg,
        rng,
        sims if sims is not None else cfg.search.sims,
        m if m is not None else cfg.search.gumbel_m,
        steps_taken,
    )
    try:
        request = next(generator)
        while True:
            request = generator.send(evaluator([request])[0])
    except StopIteration as stop:
        return stop.value  # type: ignore[no-any-return]


def run_batched(
    items: Sequence[tuple[Problem, Expr, random.Random]],
    evaluator: Evaluator,
    cfg: Config,
    *,
    sims: int | None = None,
    m: int | None = None,
    batch_size: int | None = None,
) -> list[SearchResult]:
    """Many searches concurrently, pooling their leaf evaluations.

    Exact parity with :func:`search` by construction: each tree advances
    sequentially and only the *evaluator calls* are pooled across trees, so there
    is no coupling to approximate away and the equivalence test is an equality.
    """
    sims = sims if sims is not None else cfg.search.sims
    m = m if m is not None else cfg.search.gumbel_m
    size = batch_size if batch_size is not None else cfg.search.batch_searches

    # Per-item steps_taken: an episode mid-flight is not at step 0, and the
    # terminal value is z against par over the WHOLE line. Items may supply it as
    # a 4th element; 3-tuples mean a search from a start state.
    generators = [
        _simulate(item[0], item[1], cfg, item[2], sims, m, item[3] if len(item) > 3 else 0)
        for item in items
    ]
    results: list[SearchResult | None] = [None] * len(generators)
    pending: list[tuple[int, tuple[Problem, Expr]]] = []

    for index, generator in enumerate(generators):
        try:
            pending.append((index, next(generator)))
        except StopIteration as stop:
            results[index] = stop.value

    while pending:
        chunk, rest = pending[:size], pending[size:]
        answers = evaluator([leaf for _index, leaf in chunk])
        resumed: list[tuple[int, tuple[Problem, Expr]]] = []
        for (index, _leaf), answer in zip(chunk, answers, strict=True):
            try:
                resumed.append((index, generators[index].send(answer)))
            except StopIteration as stop:
                results[index] = stop.value
        pending = resumed + rest

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


def uniform_stub(cfg: Config) -> Evaluator:
    """Flat priors, neutral value — the stub the depth-1 gate runs against."""
    width = 7 * cfg.model.max_sites

    def evaluate(leaves: Sequence[tuple[Problem, Expr]]) -> list[Evaluation]:
        return [(np.zeros(width, dtype=np.float32), 0.0) for _ in leaves]

    return evaluate
