"""Gate predicates, defined once and imported by every consumer.

**A gate expressed in a test and separately in a script is two gates wearing one
name.** Proof, and the reason this module exists: F-13 changed the terminal value
scale, the depth-1 predicate was re-expressed in `tests/test_search.py`, and
`scripts/chunk7_gate_table.py` kept the old expression — so the first re-run
reported **0/200 at every m** for a search that was working correctly. One edit
had to be two, and only one was made.

Same family as the one-formatter law (a caption describes or it calls
`render_expr()`; there is no third path) and the one-legality-oracle rule. A
predicate that decides whether a gate passed is exactly the kind of thing that
must have one implementation, because the two copies fail apart silently and the
failure looks like a result.

Predicates here are **scale-independent by construction** wherever they can be:
they read what the search *did*, not what a value happened to equal. That is what
survived F-13 — the chunk-7 gates were never scale-dependent in what they meant,
only in how one of them was written.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal, localcontext

from reckoner.search import SearchResult

#: Working precision for floor arithmetic. Generous, because the whole point is
#: that no intermediate rounding happens anywhere between the counts and the
#: integerization.
_FLOOR_PRECISION = 40

#: One-sided 95%.
Z_95 = Decimal("1.96")


def one_sided_lower_bound(successes: int, trials: int, z: Decimal = Z_95) -> Decimal:
    """The anchor's one-sided band, **in count units**, at full precision.

    ``p - z * sqrt(p(1-p)/n)``, multiplied back into counts. Returned as a
    ``Decimal`` rather than a float so the caller cannot lose the digits that
    decide the integerization — the whole failure mode this construction exists
    to prevent is an intermediate rounding changing the floor.
    """
    with localcontext() as ctx:
        ctx.prec = _FLOOR_PRECISION
        n = Decimal(trials)
        p = Decimal(successes) / n
        se = (p * (1 - p) / n).sqrt()
        return (p - z * se) * n


def ceil_count(count: Decimal) -> int:
    """Integerize a floor by **ceiling**, and never by rounding.

    PREREG-m1 §4.1, and the justification is semantic rather than numeric, which
    is what lets it travel to every future floor without re-litigation:
    **a floor is an inequality.** The gate is ``count >= b`` for a real ``b``;
    counts are integers; ceiling is the only integerization that never admits a
    count the bound itself excludes. Round-half-up would admit 1166 against a
    bound of 1166.4945 — a count the declared construction rejects — and would do
    so or not depending on where the decimals happened to fall.

    An exact integer is its own ceiling. That case is pinned by test, because it
    is the boundary every hand-rolled ceiling gets wrong.
    """
    with localcontext() as ctx:
        ctx.prec = _FLOOR_PRECISION
        return int(count.to_integral_value(rounding=ROUND_CEILING))


def no_regress_floor(successes: int, trials: int, z: Decimal = Z_95) -> int:
    """The declared no-regress construction, end to end.

    **This is an indistinguishability floor.** Holding it means *not below the
    anchor's own one-sided 95% band* — it does not mean *at least as good on
    every problem*, and it does not mean *at least as good on average*. Three
    different gates, three different licensed sentences, so the kind is named
    where the floor is computed as well as where it is declared.

    Defined here rather than in the script that checks it, per this module's
    reason for existing: a floor expressed in a prereg and separately in a
    checker is two floors wearing one name.
    """
    return ceil_count(one_sided_lower_bound(successes, trials, z))


def search_found_a_solve(result: SearchResult) -> bool:
    """Did the tree reach a solved terminal?

    **The depth-1 / B_max gate predicate.** Its arithmetic was always about
    whether the winning action is *considered and found* at ``m >= B_max``, never
    about a value crossing a threshold — and reading it from `terminal_solved`
    makes that explicit and immune to the value scale.

    The superseded expression was ``result.values.max() >= 1.0``. Under F-13's z
    currency an at-par solve scores ``0.0``, which is also what a neutral
    evaluator predicts everywhere else, so a solve stopped standing out by value
    and the old predicate read False on every correct solve.
    """
    return result.stats.terminal_solved > 0


def descent_identity_holds(result: SearchResult) -> bool:
    """``nodes - evaluations == terminals``, for a single search.

    Every node is evaluated exactly once unless it is terminal. Chunk 7's
    ``evals == nodes`` is the zero-terminal special case; the identity is the form
    that generalises, and the runner asserts it per iteration.
    """
    s = result.stats
    terminals = (
        s.terminal_solved
        + s.terminal_no_actions
        + s.state_too_large
        + s.step_cap_reached
        + s.past_par
    )
    return s.nodes - s.evaluations == terminals
