# PREREG-chunk9-shakedown.md — plumbing expectations, written before the run

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `SH-A<n>` — document prefix, then number (`P0-A5`'s
convention, applied from the start rather than after a collision).

**Status at commit: the shakedown has not run.** Every line below is an
expectation, not a result. Pre-registration applies to shakedowns *precisely
because* they are the runs nobody thinks need it.

---

## What is NOT here, and why that is the point

Six landings moved these from things a reviewer checks in a row to things a row
**cannot violate**. They are absent from this file deliberately — re-listing a
structural invariant as an expectation would imply it might not hold:

| former expectation | now enforced by |
|---|---|
| splits sum | `logschema._check_splits_sum` — `solved + capped + stuck == episodes`, and `sum(bins) == episodes_solved` |
| exact par is never beaten | three independent layers: `EpisodeResult.__post_init__`, `logschema`, `ReplayRing.append` |
| the search descends | `IterationStats.check_descent_identity` — `nodes − evals == terminals`, asserted per iteration |
| no null, no unexplained absence | `logschema.validate_row` refuses both |
| histogram bins comparable across runs | `STEPS_MINUS_PAR_BINS`, pinned in schema and versioned with it |
| the ring is not read as zeros | `Absent.__bool__` raises |
| a record lands untracked | `dataset.write_record` check-ignores its own path |

**An expectation that can be a schema invariant should be one.** What remains
below is only what genuinely cannot be.

---

## The expectations

### E1 — rows populate

`runs/<name>/iterations.jsonl` has exactly one row per committed iteration, each
validating against `ITERATION_FIELDS`, with no field absent except the two that
declare a reason (`pool_par_fraction`, `ladder_pass`).

### E2 — `alarms: 0`

`logschema.alarm_census(rows) == {}`. Premise-dependent zeros stay zero. A
non-zero reading is **not a shakedown failure** — the row writes, carries its
alarm, and the alarm names the premise. It is a finding, filed.

### E3 — snapshot loads, and refuses a mismatch

A `CheckpointPool` snapshot with matching `ruleset_version` and `vocab_version`
loads; one with either differing **raises**, and the refusal is a **counted
event**, never a silently smaller pool. Both polarities, per the chunk-6
registration.

### E4 — pool par carries provenance

Every episode labelled from a snapshot carries `par_source="pool"` **and**
`par_asof`. A pool-labelled episode may legitimately produce `z = +1` — beating
pool par is the escalation mechanism — and the invariant must **not** be widened
to silence it. `z_by_par_source` splits by source so a rising draw rate is
readable.

### E5 — the value-head switch criterion, frozen

| slot | value |
|---|---|
| metric | held-out z **balanced** accuracy |
| **floor** | **0.0**, marked uninformative — an accuracy has no structural minimum; an anti-correlated head scores *below* the trivial model |
| **null** | **1/K**, K = classes with support, census on the row |
| **threshold** | **1/K + 0.15** |
| min class support | **100** in the rarest class, else **abstain** |
| behaviour | fires **once**, **ratchets**, every evaluation writes a row |

The margin is priced as a **one-way door's error rate**, over campaign length
rather than per evaluation: at support 100, `se = 0.0354`, `z = 4.24`,
`P(fire | null) = 0.001%` per evaluation and **0.02% over 20 iterations**. The
rejected alternative is recorded: margin 0.10 at support 30 gives 6.07% per
evaluation and **71.4% over 20** — a detector that fires on noise in seven of ten
campaigns.

### E6 — abstention rows

Every evaluation writes to `runs/<name>/value_switch.jsonl`, including the ones
that abstain, each carrying `class_census` and `smallest_class_support`. An
abstention nobody records is indistinguishable from a criterion nobody ran.
`abstention_census` reports fired / refused / abstained / idle.

### E7 — the cross-switch watch

Solve-rate-by-depth is compared across the switch iteration. A regression is
**noted and diagnosed**; there is **no auto-revert**. The ratchet stands — a
detector that undoes itself on a bad reading is a detector with a feedback loop,
and the diagnosis is the deliverable.

### E8 — ring retention

`KEEP_RINGS = 3` behind `LATEST`. One ring at `replay_capacity` 500,000 is
**0.73 GiB**; unpruned that is 14.5 GiB by iteration 20 and **36.3 GiB by 50**,
against 114 GiB free. Steady state is 2.18 GiB. Pruning runs **after** the commit
and never touches the committed iteration.

---

## The shakedown is deleted after recording

Per the plan. Its artifacts are evidence of plumbing, not of performance, and
keeping them invites a later reader to cite a three-iteration run as a result.
The **numbers recorded in the report** survive; the run directory does not.

## Blocked on

`CheckpointPool` (brief §4) does not exist yet. **E3 and E4 cannot be evaluated
until it does**, and the shakedown will not be run half-way and reported as
though it covered them — a partial run reported as a whole one is exactly the
coverage claim this file exists to prevent. E1, E2, E5, E6, E8 are evaluable now;
E3, E4 and E7 follow the pool.

---

# Amendment SH-A1 — 2026-08-15, the omissions register

**Not a response to a measurement**; the shakedown has not run. This adds the
register line the omissions section needs.

Pure silence about the seven structural guarantees invites the opposite failure
to the one the omission avoids: the next reader, not knowing *why* they are
absent, re-adds one as an expectation and regresses it to reviewer-checked.
**Absence carries a reason, including in an expectations file.** The register
line, which is the absence *with* its reason rather than a list of things that
might not hold:

> **Structural by construction, not expectations:** splits-sum, exact-par
> tripwire (×3 layers), descent identity, null/absence rules, pinned bins,
> `Absent.__bool__`, `write_record` trackedness — see `logschema`, `replay`,
> `runner`, `dataset`.

---

## Comparands

- `runs/shakedown_result.json` — the shakedown's record
- `runs/<name>/iterations.jsonl`, `runs/<name>/value_switch.jsonl` — the record
  shapes the kill-point harness asserts against (patterns, not instances)

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
