"""The single schema definition for run logs. Inherited law, and it comes first.

`logschema.py as single schema definition with role fields and
absence-carries-a-reason notes` is in the chunk-0 inherited block. It leads the
chunk-9 build because **"from field one" is only achievable if the schema exists
before any writer does** — a column added after the first row is a column with a
hole in it, and the hole is exactly where the interesting iteration was.

Three properties, each of which has a failure this project has already met:

**1. One definition, imported by writers and readers alike.** A dashboard that
knows the column names independently is a second schema that can disagree with
the first. Analysis code imports `ITERATION_FIELDS` and nothing else.

**2. Every field carries a role.** A number's role is what it is *for*, and it is
not inferable from its name: `solve_rate_by_depth` is an outcome the campaign is
judged on, `entropy_prior_step1_start` is a diagnostic that explains an outcome,
and `nan_skips` is health. A reader six weeks out needs to know which numbers can
move the verdict and which only explain it — and mixing them is how a diagnostic
becomes a target.

**3. Absence carries a reason.** A missing field is never silently null. Either
it is required and its absence is an error, or it is optional and the row must
say *why* it is absent, in `absent: {field: reason}`. F-02 is the ancestor here:
a value that says nothing about its own provenance gets read as though it said
the strongest thing. A `null` in a JSONL column reads as zero to the next
histogram that touches it.

The schema is data, not code that runs at write time — `validate_row` is called
by the writer, and by tests, and by anything that reads a log it did not write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Closed vocabulary. A role outside this set is a schema bug, not a new kind of
#: number — the point of the list is that adding a role is a decision.
ROLES = frozenset(
    {
        "identity",  # which run, which iteration — how a row is addressed
        "provenance",  # what produced it: shas, fingerprints, versions, dtype
        "outcome",  # what the campaign is judged on
        "diagnostic",  # instrumentation that explains an outcome
        "counter",  # event tallies
        "timing",  # wall clock
        "health",  # skips, refusals, aborts — the run's own vital signs
    }
)


#: Why an era-3 row's ``entropy_target_*`` is not this column's quantity (F-54).
_F54_TARGET_REDEFINED = (
    "before schema_era 4 this held H of the Sequential-Halving visit share — a "
    "function of the visited count alone (F-54) — not of π′"
)

#: Current schema era. Bumped whenever a field is ADDED, a pinned binning
#: CHANGES, or a column's DEFINITION changes — the column then moves its
#: ``since`` and states why in ``since_reason`` (F-54: a redefinition is an era
#: boundary, the lesson F-53 taught the ring). A row records the era it was
#: written under, which is what lets a reader distinguish "this run predates the
#: column" from "this run dropped it" — and "this value meant something else".
#:
#: 4 (F-54): ``entropy_target_*`` redefined from H(visit share) to H(π′);
#: ``entropy_visits_*``, ``kl_improved_prior_*`` and ``ring_improved_truncated``
#: added.
SCHEMA_ERA = 4

#: Bins for ``steps_minus_par_histogram``, **pinned here and versioned with the
#: schema — never in config.** A histogram whose bins moved is two instruments
#: sharing a name, and config is where numbers drift. Changing these is an era
#: bump, which is exactly the friction the change deserves.
#:
#: ``<0`` is reachable only against non-exact par (pool par), because nothing
#: beats BFS-exact par by construction. It is present so that when it becomes
#: non-zero the row says which source allowed it.
STEPS_MINUS_PAR_BINS: tuple[str, ...] = ("<0", "0", "1", "2", "3", "4", "5", "6+")

#: **Definitional zero.** par_source values that are EXACT — the minimum in this rule system. Beating
#: one is impossible by construction, and `EpisodeResult.__post_init__` already
#: refuses it. Listing them here makes the log a SECOND, independent tripwire on
#: the project's most load-bearing invariant (F-02's descendant).
EXACT_PAR_SOURCES = frozenset({"bfs"})


class SchemaError(ValueError):
    """A row that does not match the schema. Raised, never warned."""


@dataclass(frozen=True, slots=True)
class Field:
    """One column. ``absence`` is required exactly when the field is optional."""

    name: str
    kind: type | tuple[type, ...]
    role: str
    doc: str
    required: bool = True
    absence: str | None = None
    #: Which SPECIES of zero this column carries, when zero is the expected
    #: reading. Two kinds, two dispositions, because they are different diseases:
    #:
    #: * ``"definitional"`` — nonzero is impossible *by the meaning of the terms*.
    #:   A nonzero reading means the pipeline is broken, so the row is REFUSED at
    #:   write time. Suppressing it loses nothing: the number could not have been
    #:   evidence of anything except its own wrongness.
    #: * ``"premise"`` — zero is *proven under premises that can break*. A nonzero
    #:   reading does not mean the pipeline lied; it means a premise did (a round
    #:   fired, an emission constraint slipped, a rule changed). That row IS the
    #:   evidence, so it WRITES and ALARMS — refusing it would suppress exactly
    #:   the observation worth having. ``zero_premises`` names what must hold.
    zero_class: str | None = None
    zero_premises: str | None = None
    #: Era in which this column first existed. A row from an earlier era may
    #: omit it, and that absence is valid with the synthetic reason
    #: ``predates_field`` — computed from (row era, this number), never asserted
    #: by the row, because a row written before a field existed cannot have
    #: explained the absence of something nobody had named yet.
    since: int = 1
    #: Why rows before ``since`` cannot be read as this column, when "the column
    #: did not exist yet" would be false: a column REDEFINED in place existed
    #: before, under another meaning, and its old values are not this quantity.
    since_reason: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SchemaError(f"{self.name}: role {self.role!r} is not in {sorted(ROLES)}")
        if self.required and self.absence is not None:
            raise SchemaError(f"{self.name}: a required field cannot declare an absence reason")
        if self.zero_class not in (None, "definitional", "premise"):
            raise SchemaError(
                f"{self.name}: zero_class must be None, 'definitional' or 'premise'; "
                f"got {self.zero_class!r}"
            )
        if self.zero_class == "premise" and not self.zero_premises:
            raise SchemaError(
                f"{self.name}: a premise-dependent zero MUST name its premises. "
                "A nonzero reading is a finding about which premise broke, and a "
                "finding nobody can attribute is an alarm nobody can act on."
            )
        if self.zero_class != "premise" and self.zero_premises:
            raise SchemaError(f"{self.name}: only a premise-dependent zero names premises")
        if not self.required and not self.absence:
            raise SchemaError(
                f"{self.name}: an optional field MUST declare why it may be absent. "
                "Absence carries a reason — a column that is sometimes missing and "
                "never says why is a column that reads as zero."
            )


#: The Phase-2 loop's per-iteration row: ``runs/<name>/iterations.jsonl``.
ITERATION_FIELDS: tuple[Field, ...] = (
    # --- identity ---------------------------------------------------------
    Field("iteration", int, "identity", "0-based loop index."),
    Field(
        "schema_era",
        int,
        "identity",
        "Schema era this row was written under. Distinguishes 'predates the "
        "column' from 'dropped the column' — without it, era handling is a "
        "guess about history.",
    ),
    Field("run_name", str, "identity", "Run directory name; rows are portable without it."),
    # --- provenance -------------------------------------------------------
    Field("git_sha", str, "provenance", "Repo SHA at the moment the row was written."),
    Field("config_fingerprint", str, "provenance", "sha256 of the resolved config."),
    Field("ruleset_version", int, "provenance", "Par and actions are denominated in this."),
    Field("vocab_version", int, "provenance", "States are denominated in this."),
    Field(
        "measure_dtype",
        str,
        "provenance",
        "Regime that produced every measured number in this row. fp32 is the "
        "regime the chunk-9 equivalence gate licensed (F-12).",
    ),
    Field("train_dtype", str, "provenance", "Regime the optimiser step ran in."),
    # --- outcome ----------------------------------------------------------
    Field(
        "solve_rate_by_depth",
        dict,
        "outcome",
        "depth -> fraction solved. Stratified because an aggregate hides the "
        "deep tail, and the deep tail is the experiment.",
    ),
    Field(
        "z_by_par_source",
        dict,
        "outcome",
        "par_source -> {'+1': n, '0': n, '-1': n}. **The draw-inflation watch.** "
        "Splitting by par_source is what makes a rising draw rate readable: "
        "draws against exact BFS par mean the policy reached par, draws against "
        "pool par mean the pool stopped improving, and one aggregate number "
        "cannot tell those apart.",
    ),
    Field(
        "steps_minus_par_histogram",
        dict,
        "outcome",
        "(steps - par) -> count. **Population: SOLVED episodes only.** Registered "
        "at chunk-8 close — the median alone says at least half drew; this is the "
        "loss tail. Capped and stuck episodes do NOT appear here, in any bin: an "
        "unsolved episode has no steps-to-solve, and filing it under '6+' would "
        "pool two different diseases into one instrument. Draw-inflation (the 0 "
        "bin rising) and timeout-composition (episodes_capped rising) have "
        "different causes and different fixes, so they are counted separately and "
        "the totals are cross-checked: sum(bins) == episodes_solved, and "
        "episodes_solved + episodes_capped + episodes_stuck == episodes.",
    ),
    # --- diagnostic -------------------------------------------------------
    Field(
        "entropy_prior_step1_start",
        float,
        "diagnostic",
        "H of the network prior at step 1, on **problem start states**.",
    ),
    Field(
        "entropy_prior_step1_reached",
        float,
        "diagnostic",
        "H of the network prior at step 1, on states reached after >= 1 rewrite. "
        "The chess split was start-position vs book-position; this is the "
        "faithful analog here, and the mapping is an interpretation rather than "
        "a transplant — stated so a reader does not assume an openings book.",
    ),
    Field(
        "entropy_target_step1_start",
        float,
        "diagnostic",
        "H of the **search-improved** target π′ at step 1 on start states. Prior "
        "and target must both be logged: the chess lesson is that prior entropy "
        "alone cannot distinguish a confident policy from a collapsed one. Equal "
        "to H(prior) exactly where search found nothing — read it beside "
        "kl_improved_prior_step1_start, which is zero there and only there.",
        since=4,
        since_reason=_F54_TARGET_REDEFINED,
    ),
    Field(
        "entropy_target_step1_reached",
        float,
        "diagnostic",
        "H of the search-improved target π′ at step 1 on reached states.",
        since=4,
        since_reason=_F54_TARGET_REDEFINED,
    ),
    Field(
        "entropy_visits_step1_start",
        float,
        "diagnostic",
        "H of the Sequential-Halving visit share at step 1 on start states — the "
        "'visits' control arm's target, and the quantity entropy_target_* held "
        "before era 4. A function of the visited count alone (F-54).",
        since=4,
    ),
    Field(
        "entropy_visits_step1_reached",
        float,
        "diagnostic",
        "H of the visit share at step 1 on reached states (see the start column).",
        since=4,
    ),
    Field(
        "kl_improved_prior_step1_start",
        float,
        "diagnostic",
        "Mean KL(π′ ‖ prior) in nats at step 1 on start states. Zero exactly where "
        "search found nothing to improve on, so it is the policy-signal instrument "
        "that still moves while the value head is silent (F-54).",
        since=4,
    ),
    Field(
        "kl_improved_prior_step1_reached",
        float,
        "diagnostic",
        "Mean KL(π′ ‖ prior) at step 1 on reached states.",
        since=4,
    ),
    # --- counters ---------------------------------------------------------
    Field("episodes", int, "counter", "Episodes completed this iteration."),
    Field(
        "episodes_solved",
        int,
        "counter",
        "Episodes the CHECKER accepted. The population of "
        "steps_minus_par_histogram, and asserted equal to its total.",
    ),
    Field(
        "episodes_capped",
        int,
        "counter",
        "Episodes that hit episode.step_cap unsolved. Distinct from over-par "
        "solves: both give z = -1, and pooling them hides which is happening.",
    ),
    Field(
        "episodes_conceded",
        int,
        "counter",
        "Episodes the policy RESIGNED: the best line already needs >= par + "
        "`par.concede_k` steps, so the outcome is settled and the remaining "
        "budget buys nothing. A distinct outcome from `capped` — conceding is a "
        "decision and capping is an exhaustion, and pooling them would hide "
        "which the policy is doing. Inert while `par.concede_enabled` is False.",
    ),
    Field(
        "episodes_stuck",
        int,
        "counter",
        "Episodes that ran out of legal actions before solving. An EPISODE "
        "outcome — not to be confused with `terminal_no_actions`, which counts "
        "no-action nodes encountered INSIDE a search tree and does not end an "
        "episode. **Premise-dependent zero: this row writes and alarms rather "
        "than refusing.**",
        zero_class="premise",
        zero_premises=(
            "chunk 5's proof sketch: dead ends are unreachable from well-formed "
            "v1 problems, because every non-goal state admits at least one legal "
            "rewrite. A nonzero reading means a premise broke, not that the "
            "counter lied — candidates: ROUND-02 fired, an emission constraint "
            "slipped, or the rule set changed under a dataset built before it."
        ),
    ),
    Field(
        "search_nodes_total",
        int,
        "counter",
        "Summed SearchStats.nodes — the descent gate's number, carried per iteration.",
    ),
    Field(
        "search_evaluations_total",
        int,
        "counter",
        "Summed SearchStats.evaluations; equals nodes when every node is evaluated once.",
    ),
    Field(
        "terminal_no_actions",
        int,
        "counter",
        "No-action nodes met inside search trees. Search-internal; an episode "
        "ending with no legal move is `episodes_stuck`, a different column.",
    ),
    # --- health -----------------------------------------------------------
    Field(
        "state_too_large",
        int,
        "health",
        "Counted terminal losses from encode overflow. Never cropped (chunk 6).",
    ),
    Field(
        "ring_improved_truncated",
        int,
        "health",
        "Ring records whose π′ dropped more than IMPROVED_TAIL_TOL beyond the "
        "stored slots. They sit out of policy training under 'improved' — counted, "
        "never cropped (F-54).",
        zero_class="premise",
        zero_premises="every ring record's root had at most IMPROVED_SLOTS = 128 legal "
        "actions, or the π′ mass beyond the stored slots was at most IMPROVED_TAIL_TOL",
        since=4,
    ),
    Field("nan_skips", int, "health", "Optimiser steps skipped on non-finite gradients."),
    Field(
        "pool_refusals",
        int,
        "health",
        "Snapshots the CheckpointPool refused on a version mismatch. A counted "
        "event, never a silently smaller pool (BRIEF-chunk9 registration).",
    ),
    # --- timing -----------------------------------------------------------
    Field("seconds_self_play", float, "timing", "Wall clock generating episodes."),
    Field("seconds_train", float, "timing", "Wall clock in the optimiser."),
    Field("seconds_total", float, "timing", "Wall clock for the whole iteration."),
    # --- optional, each with its reason -----------------------------------
    Field(
        "pool_par_fraction",
        float,
        "diagnostic",
        "Fraction of episodes whose par came from a pool snapshot.",
        required=False,
        absence="league.par_from_pool_frac is 0, or no snapshot has been taken yet",
    ),
    Field(
        "ladder_pass",
        int,
        "outcome",
        "Ladder pass index, when this iteration ran one.",
        required=False,
        absence="not a ladder iteration (ladder runs on ladder.ladder_every cadence)",
    ),
    Field(
        "evaluator_checkpoint_sha256",
        str,
        "provenance",
        "sha256 of the checkpoint the SEARCH EVALUATOR was built from this "
        "iteration. **Provenance, not configuration**: a config field records "
        "what was asked for and a digest records what ran. D-A1 §1.1 — golden "
        "asserted that the model MOVES and never that the moved model is USED, "
        "so the loop's entire subject (model -> search -> ring) went unexercised "
        "while every iteration trained on episodes generated by flat priors. "
        "This column is the assertion that the trained weights are the weights "
        "playing.",
        since=2,
    ),
    Field(
        "pool_composition",
        dict,
        "diagnostic",
        "The par-escalation pool as it stood when this iteration drew from it: "
        "``size``, ``steps`` (sorted — the pool's identity), ``order`` (the "
        "members list as ``sample`` sees it) and ``value_head_live``. "
        "**Registered by M1-A3, on F-23's argument.** `composition()` existed "
        "from chunk 9 and was logged by nothing, so when resume rebuilt the pool "
        "with the anchor alone, every column in this row agreed with an "
        "uninterrupted run while the ring the model trains on differed — the "
        "campaign's own escalation mechanism silently reset, invisible at any "
        "strictness a row comparison can reach. M1-A2 §6's two-population split "
        "is analytically dependent on this: a pool/bfs ratio is uninterpretable "
        "without knowing how many rungs the pool held. ``size`` 0 is a VALUE "
        "(the pool is empty, and par has nothing to escalate from); it is not "
        "absence, which is why this column is required rather than optional.",
        since=3,
    ),
)


#: ``runs/<name>/value_switch.jsonl`` — one row per evaluation of the value-head
#: switch criterion, **including the ones that abstain.**
#:
#: An abstention nobody records is indistinguishable from a criterion nobody ran,
#: and a long abstention streak through a draw-dominated stretch is exactly the
#: data the campaign will want to have had. Three outcomes, three distinguishable
#: rows: fired, refused, abstained — never a silence.
VALUE_SWITCH_FIELDS: tuple[Field, ...] = (
    Field("iteration", int, "identity", "Loop index this evaluation belongs to."),
    Field("schema_era", int, "identity", "Schema era this row was written under."),
    Field("metric", str, "identity", "Held-out z balanced accuracy."),
    Field(
        "abstained",
        bool,
        "outcome",
        "True when the rarest class had too little support to judge. **Not a "
        "failure** — 'not evaluable yet' and 'evaluated and refused' are "
        "different states and only one of them is evidence.",
    ),
    Field("fired", bool, "outcome", "True on the single evaluation that ratcheted the head live."),
    Field("already_live", bool, "outcome", "True once the ratchet has fired; the criterion idles."),
    Field("clears", bool, "outcome", "Whether measured met threshold (False when abstained)."),
    Field("n", int, "diagnostic", "Held-out slice size."),
    Field(
        "class_census",
        dict,
        "diagnostic",
        "z -> support. **The cause of an abstention travels with it**, so a "
        "streak can be read back as 'the minority class was thin' rather than "
        "guessed at.",
    ),
    Field("k_classes_with_support", int, "diagnostic", "K — the null is 1/K."),
    Field("smallest_class_support", int, "diagnostic", "What the abstention rule tests."),
    Field(
        "abstain_reason",
        str,
        "diagnostic",
        "Why the criterion declined to judge: `no_variance` (z took one value, so "
        "nothing distinguishes a head from a constant) or `thin_minority`. Two "
        "causes, two fixes — wait for the policy to start losing, versus wait for "
        "more episodes.",
        required=False,
        absence="the criterion judged, so there is no abstention to explain",
    ),
    Field("floor", float, "diagnostic", "0.0 — an accuracy has no structural minimum."),
    Field("null", float, "diagnostic", "1/K: a constant predictor's balanced accuracy."),
    Field("threshold", float, "diagnostic", "1/K + margin, priced as a one-way door's error rate."),
    Field("measured", float, "diagnostic", "The head's balanced accuracy."),
)


# ---------------------------------------------------------------------------
# The ladder's two currencies. They never mix, and the schema is what stops them.
# ---------------------------------------------------------------------------

#: **z-vs-par.** For RULE-DENOMINATED arms only — our model, RandomRewriter,
#: GreedyHeuristic. They act in our rule system, so their steps ARE steps and z
#: against par is meaningful.
CURRENCY_Z = "z_vs_par"

#: **solve-vs-budget.** For arms denominated in someone else's rules. A sympy
#: derivation is not a sequence of our rewrites, so counting its "steps" against
#: our par would be comparing lengths measured in different units and calling the
#: difference skill. Spec: sympy is a rung, never par.
CURRENCY_BUDGET = "solve_vs_budget"

CURRENCIES = frozenset({CURRENCY_Z, CURRENCY_BUDGET})

#: Fields that exist ONLY in one currency. A row carrying the other currency's
#: score field is refused — that is the whole mechanism by which "two currencies
#: never silently mix" is enforced rather than remembered.
_Z_ONLY = ("z", "steps", "par", "par_source")
_BUDGET_ONLY = ("solved", "steps_used", "budget", "cas_version")

_LADDER_COMMON: tuple[Field, ...] = (
    Field("pass_index", int, "identity", "Ladder pass this row belongs to."),
    Field("schema_era", int, "identity", "Schema era this row was written under."),
    Field("arm", str, "identity", "Which rung produced it."),
    Field("problem_key", str, "identity", "The paired-set problem, by its identity key."),
    Field(
        "currency",
        str,
        "identity",
        "**Stated on every row, never inferred.** z_vs_par for rule-denominated "
        "arms; solve_vs_budget for external ones. A reader that has to guess a "
        "row's units will eventually guess wrong, and the two are not comparable.",
    ),
    Field("role", str, "provenance", "What this arm is FOR: baseline, rung, or the subject."),
    Field(
        "nondeterministic",
        bool,
        "provenance",
        "True for arms stochastic BY DESIGN (RandomRewriter). Carried from day "
        "one rather than added after a surprise: a repeated number from a "
        "deterministic arm is evidence, and from a stochastic one it is luck.",
    ),
    Field("seed", int, "provenance", "Per-problem derived seed; recorded even when unused."),
    Field(
        "calibration_note",
        str,
        "provenance",
        "What this number does and does not license. Free text, required — a "
        "score without its caveat gets quoted without its caveat.",
    ),
)

LADDER_FIELDS_Z: tuple[Field, ...] = _LADDER_COMMON + (
    Field("z", int, "outcome", "+1 under par, 0 at par, -1 over par or capped."),
    Field("steps", int, "outcome", "Steps taken, in OUR rules."),
    Field("par", int, "outcome", "The par this z is against."),
    Field("par_source", str, "provenance", "bfs | scripted | pool — never sympy."),
)

LADDER_FIELDS_BUDGET: tuple[Field, ...] = _LADDER_COMMON + (
    Field("solved", bool, "outcome", "Did the external solver produce a correct answer?"),
    Field("steps_used", int, "outcome", "Its own steps, in ITS units. Never compared to par."),
    Field("budget", int, "outcome", "The step budget it was given."),
    Field(
        "cas_version",
        str,
        "provenance",
        "**Part of the rung's identity.** A sympy upgrade changes the opponent, "
        "so a pass compared across versions is a pass compared against two "
        "different rungs wearing one name.",
    ),
)


def ladder_fields(currency: str) -> tuple[Field, ...]:
    if currency == CURRENCY_Z:
        return LADDER_FIELDS_Z
    if currency == CURRENCY_BUDGET:
        return LADDER_FIELDS_BUDGET
    raise SchemaError(f"unknown currency {currency!r}; known: {sorted(CURRENCIES)}")


def validate_ladder_row(row: dict[str, Any]) -> list[str]:
    """Validate against the row's OWN currency, and refuse any mixing.

    Two guards, and the second is the one that makes the ruling structural:

    1. the row validates against the field set its ``currency`` selects
    2. it carries **none** of the other currency's score fields — so a
       solve-vs-budget row can never be read as a z row, and no aggregation can
       accidentally average a z against a solve rate
    """
    currency = row.get("currency")
    if currency not in CURRENCIES:
        raise SchemaError(
            f"row declares currency {currency!r}. Every ladder row states its "
            "currency; a row that does not is a number whose units are a guess."
        )
    forbidden = _BUDGET_ONLY if currency == CURRENCY_Z else _Z_ONLY
    trespass = sorted(set(row) & set(forbidden))
    if trespass:
        raise SchemaError(
            f"a {currency} row carries {trespass}, which belong to the other "
            "currency. The two are not comparable — sympy's steps are not our "
            "steps — and a row that carries both invites exactly the average "
            "nobody should compute."
        )
    return validate_row(row, ladder_fields(currency))


def field_map(fields: tuple[Field, ...] = ITERATION_FIELDS) -> dict[str, Field]:
    return {f.name: f for f in fields}


def validate_row(row: dict[str, Any], fields: tuple[Field, ...] = ITERATION_FIELDS) -> list[str]:
    """Raise :class:`SchemaError` unless ``row`` matches the schema exactly.

    Four checks, and the third is the one that earns this module's place:

    1. no unknown keys — a mistyped column is a column nobody reads
    2. every required field present, with the declared type
    3. **every absent optional field named in ``absent`` with a reason** — and
       conversely, nothing named in ``absent`` that is actually present
    4. no ``None`` values: absence is expressed in ``absent``, never as null,
       because a null reads as zero to the next thing that aggregates it

    Returns the list of **alarms** — premise-dependent zeros that came back
    nonzero. Alarms do not raise: that row is the evidence a premise broke, and
    refusing it would suppress the observation. Hard violations still raise.
    """
    known = field_map(fields)
    era = row.get("schema_era")
    if era is not None and not isinstance(era, int):
        raise SchemaError("schema_era must be an int")
    absent = row.get("absent", {})
    if not isinstance(absent, dict):
        raise SchemaError("'absent' must be a mapping of field -> reason")

    unknown = set(row) - set(known) - {"absent", "alarms"}
    if unknown:
        raise SchemaError(f"unknown columns: {sorted(unknown)}")

    for name, spec in known.items():
        present = name in row
        if present and row[name] is None:
            raise SchemaError(
                f"{name}: null is not how absence is expressed — put it in "
                f"'absent' with a reason. A null reads as zero downstream."
            )
        if present and not isinstance(row[name], spec.kind):
            raise SchemaError(f"{name}: expected {spec.kind}, got {type(row[name])}")
        if not present:
            # Absence by ERA is valid, and its reason is computed rather than
            # asserted: a row written before a field existed could not have
            # explained the absence of something nobody had named yet.
            if era is not None and era < spec.since:
                continue
            if spec.required:
                raise SchemaError(
                    f"{name}: required field missing"
                    + (f" (row era {era} >= field since {spec.since})" if era is not None else "")
                )
            if name not in absent or not str(absent[name]).strip():
                raise SchemaError(
                    f"{name}: optional field is absent without a reason. "
                    f"Declared reason for absence is: {spec.absence!r}"
                )

    for name in absent:
        if name not in known:
            raise SchemaError(f"'absent' names unknown column {name!r}")
        if name in row:
            raise SchemaError(f"{name}: named in 'absent' but present in the row")

    _check_pinned_bins(row)
    _check_exact_par_cannot_be_beaten(row)
    _check_splits_sum(row)
    return _premise_zero_alarms(row, known)


def _premise_zero_alarms(row: dict[str, Any], known: dict[str, Field]) -> list[dict[str, Any]]:
    """Premise-dependent zeros that came back nonzero — the row still ships.

    **Structured, not prose.** An alarm the census has to parse out of a sentence
    is an alarm the census can misattribute; the field name is a key, so counting
    by field is exact rather than regex-shaped.
    """
    alarms = []
    for name, spec in known.items():
        if spec.zero_class != "premise" or name not in row:
            continue
        if row[name]:
            alarms.append(
                {
                    "field": name,
                    "value": row[name],
                    "expected": 0,
                    "premises": spec.zero_premises,
                    "message": (
                        f"{name} = {row[name]}, expected 0. This zero holds only under "
                        "premises that can break. The row is written and flagged rather "
                        "than refused — it is the evidence."
                    ),
                }
            )
    return alarms


def alarm_census(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Alarms by field across rows. The last leg of the alarm's journey.

    Fired at write, carried in the row, **surfaced at the run**. Without this an
    alarm written is an alarm stored: the first nonzero would be found by whoever
    happened to grep the JSONL, which is not a level anyone acts at. Rider (b),
    one more layer up — a signal nobody aggregates is a comment that happens to
    be recorded.

    Returns ``{}`` when nothing fired, and ``{}`` is the expectation line a run
    report asserts against: ``alarms: 0``.
    """
    census: dict[str, int] = {}
    for row in rows:
        for alarm in row.get("alarms", []):
            field_name = alarm["field"] if isinstance(alarm, dict) else str(alarm)
            census[field_name] = census.get(field_name, 0) + 1
    return census


def switch_event_row(event: dict[str, Any], *, schema_era: int) -> dict[str, Any]:
    """Project a `valuegate` event onto ``VALUE_SWITCH_FIELDS``. Every outcome writes."""
    return {
        "iteration": event["iteration"],
        "schema_era": schema_era,
        "metric": event["metric"],
        "abstained": bool(event.get("abstained", False)),
        "fired": bool(event.get("fired", False)),
        "already_live": bool(event["already_live"]),
        "clears": bool(event["clears"]),
        "n": event["n"],
        "class_census": {str(k): v for k, v in event["class_census"].items()},
        "k_classes_with_support": event["k_classes_with_support"],
        "smallest_class_support": event["smallest_class_support"],
        **({"abstain_reason": event["abstain_reason"]} if event.get("abstain_reason") else {}),
        "floor": float(event["floor"]),
        "null": float(event["null"]),
        "threshold": float(event["threshold"]),
        "measured": float(event["measured"]),
        **(
            {}
            if event.get("abstain_reason")
            else {
                "absent": {
                    "abstain_reason": "the criterion judged, so there is no abstention to explain"
                }
            }
        ),
    }


def abstention_census(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many evaluations abstained, refused, fired. The streak, made readable."""
    census = {"abstained": 0, "refused": 0, "fired": 0, "idle": 0}
    for row in rows:
        if row.get("already_live") and not row.get("fired"):
            census["idle"] += 1
        elif row.get("abstained"):
            census["abstained"] += 1
        elif row.get("fired"):
            census["fired"] += 1
        else:
            census["refused"] += 1
    return census


def _check_pinned_bins(row: dict[str, Any]) -> None:
    """The histogram's bins are the schema's, not the row's."""
    hist = row.get("steps_minus_par_histogram")
    if hist is None:
        return
    if set(hist) != set(STEPS_MINUS_PAR_BINS):
        missing = sorted(set(STEPS_MINUS_PAR_BINS) - set(hist))
        extra = sorted(set(hist) - set(STEPS_MINUS_PAR_BINS))
        raise SchemaError(
            f"steps_minus_par_histogram bins must be exactly {list(STEPS_MINUS_PAR_BINS)}; "
            f"missing {missing}, unexpected {extra}. Every bin is present in every row "
            "including the zeros, so rows are directly comparable and a missing bin is "
            "never read as a zero that was measured."
        )


def _check_splits_sum(row: dict[str, Any]) -> None:
    """Every episode lands in exactly one outcome, and the histogram's population
    is named rather than inferred.

    This is the brief's "splits sum" plumbing expectation, enforced by the schema
    instead of by a reviewer reading a row. A split that does not sum is either a
    lost episode or a double-counted one, and both look like a plausible number.
    """
    needed = (
        "episodes",
        "episodes_solved",
        "episodes_capped",
        "episodes_stuck",
        "episodes_conceded",
    )
    if any(k not in row for k in needed):
        return
    total = (
        row["episodes_solved"]
        + row["episodes_capped"]
        + row["episodes_stuck"]
        + row["episodes_conceded"]
    )
    if total != row["episodes"]:
        raise SchemaError(
            f"outcomes do not sum: solved {row['episodes_solved']} + capped "
            f"{row['episodes_capped']} + stuck {row['episodes_stuck']} + conceded "
            f"{row['episodes_conceded']} = {total}, "
            f"but episodes = {row['episodes']}. An episode ends in exactly one "
            "outcome; a split that does not sum is a lost or double-counted episode."
        )
    hist = row.get("steps_minus_par_histogram")
    if hist is not None and sum(hist.values()) != row["episodes_solved"]:
        raise SchemaError(
            f"steps_minus_par_histogram totals {sum(hist.values())} but "
            f"episodes_solved = {row['episodes_solved']}. The histogram's "
            "population is SOLVED episodes only — capped and stuck episodes have "
            "no steps-to-solve and must not be filed in any bin."
        )


def _check_exact_par_cannot_be_beaten(row: dict[str, Any]) -> None:
    """The log layer as a second tripwire on the win-condition invariant.

    **Zero species: definitional.** Nothing beats exact par by the meaning of the
    words, so a nonzero cell means the pipeline is broken and the row is REFUSED.
    Contrast ``episodes_stuck``, whose zero is proven only under premises that can
    break — that one writes and alarms, because the row is the evidence. Two
    zeros, two diseases, two dispositions.

    ``EpisodeResult.__post_init__`` already refuses ``z > 0`` against an exact
    ``par_source``; this refuses it again at write time, from a different module
    with a different code path. The cell is **present and zero**, never absent —
    an absent cell cannot be checked, and this is the one number in the project
    that must never drift silently (F-02's descendant).
    """
    table = row.get("z_by_par_source")
    if table is None:
        return
    for source, cells in table.items():
        if source not in EXACT_PAR_SOURCES:
            continue
        if "+1" not in cells:
            raise SchemaError(
                f"z_by_par_source[{source!r}] must carry an explicit '+1' cell. "
                "It is structurally impossible to beat exact par, so the cell is "
                "present at zero — an absent cell is a tripwire that cannot fire."
            )
        if cells["+1"] != 0:
            raise SchemaError(
                f"z_by_par_source[{source!r}]['+1'] == {cells['+1']}, but {source} is "
                "EXACT par: beating it is impossible by construction. Either the "
                "labeller is wrong or the outcome is — this is F-02's shape and the "
                "row does not ship."
            )


def append_row(
    path: Path, row: dict[str, Any], fields: tuple[Field, ...] = ITERATION_FIELDS
) -> list[str]:
    """Validate, then append one JSONL row. Validation is not optional.

    Writing an invalid row and validating later is how a run produces 50
    iterations of unreadable log and discovers it at analysis time.
    """
    alarms = validate_row(row, fields)
    if alarms:
        # The alarm travels IN the row. An alarm returned but not recorded is a
        # comment that happens to be computed (rider (b)); this one cannot be
        # dropped by a caller who forgets to look at the return value.
        row = {**row, "alarms": alarms}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return alarms


def read_rows(path: Path, fields: tuple[Field, ...] = ITERATION_FIELDS) -> list[dict[str, Any]]:
    """Read and validate. A reader that does not validate is trusting a writer
    it did not run — including an older era of the same writer."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        try:
            validate_row(row, fields)
        except SchemaError as exc:
            raise SchemaError(f"{path}:{i + 1}: {exc}") from exc
    return rows


def describe(fields: tuple[Field, ...] = ITERATION_FIELDS) -> str:
    """The schema as a table — for the run directory, so a log is readable
    without this package."""
    lines = ["| column | role | required | meaning / absence |", "|---|---|---|---|"]
    for f in fields:
        req = "yes" if f.required else f"no — {f.absence}"
        lines.append(f"| `{f.name}` | {f.role} | {req} | {f.doc} |")
    return "\n".join(lines)
