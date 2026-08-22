# PREREG-chunk10-smoke.md — the ladder smoke pass, written before the run

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `S-A<n>`.

**Status at commit: the smoke pass has not run.** Every number below is a
threshold or a shape, never a result.

---

## What is NOT here, and why

Per the chunk-9 precedent — **an expectation that can be a schema invariant should
be one** — the following are absent deliberately. Re-listing a structural
guarantee as an expectation implies it might not hold.

> **Structural by construction, not expectations:** currency non-mixing
> (`logschema.validate_ladder_row` refuses a z row carrying budget fields and
> vice versa), per-row validation at write (`append_row`), the strict pairing key
> (`dataset.problem_key`), duplicate-problem refusal at the freeze
> (`pairedset.freeze`), cross-currency pairing refusal (`ladder.pair`), and
> partial-overlap refusal (`ladder.pair`).

---

## The paired set

### S1 — frozen at birth, verified at read

`runs/paired/smoke_v1.jsonl` is written once, its sha256 appended to
`runs/ANCHORS.sha256`, and a second `freeze` of the same path **raises**. `load`
re-digests and refuses a drift. An anchor nobody consults on the path that uses
the file is a comment.

### S2 — censused at BOTH levels, decision rule pre-stated

Candidates are censused against:

| level | reference | question |
|---|---|---|
| problem-level | `runs/data/train_100k` | is this a training *problem*? |
| state-level | `runs/data/phase1_train` | is this an intermediate *state* of a training derivation? |

**The rule, fixed before the numbers:** a candidate colliding at *either* level is
**dropped before the freeze**. Contamination discovered *after* the freeze is a
**finding, not an adjustment** — editing a frozen instrument to make it clean
produces a set whose digest no longer describes the thing that was measured.

Both counts are reported, and so is `state_level_beyond_problem_level` — the
candidates the problem-level test alone would have passed. That number is F-08's
mechanism measured at this boundary; **it may be zero**, and zero here is a
premise-dependent zero, not a definitional one.

Candidates are drawn from `runs/data/eval_held_out`. This implements F-09's
ruling ("read the unseen subset") for the ladder: the held-out set was 21.28%
seen, so the paired set is the censused remainder rather than the whole file.

---

## The pass

### S3 — every arm scored on its own declared subset, completely

Each arm writes exactly one row per problem it **plays**. `SympySolver.plays`
declares SOLVE and EVALUATE and declines SIMPLIFY; declined problems are
**counted skips**, never rows saying the rung failed. A short arm raises.

### S4 — `pair_scores` from row one

Rows land as they happen. Asserted directly: at unit *k* of the first arm, the
file already holds *k* rows. A pass that buffers to the end and is killed at 90%
leaves nothing, and a pass that finishes leaves means the bootstrap cannot
consume.

### S5 — resume at a **real** kill point

The pass runs in a subprocess which is **SIGKILLed** mid-way — not an exception,
which would leave every line complete. The resumed pass must produce a row set
**identical** to an uninterrupted run of the same pass, row for row, with no
duplicated `(arm, problem_key)` unit. A torn final line is truncated; a torn
line anywhere else raises rather than being dropped.

---

## The gates

### S6 — the null run (rider (c): the null is a **run**, not an estimate)

Self-match, eval profile: the deterministic arm against itself over the paired
set.

| slot | value |
|---|---|
| floor | **−2.0** — z ∈ {−1, 0, +1}, so a paired difference is bounded below by −2 |
| **null** | **0.0**, *measured by running it*, not assumed |
| threshold | every paired difference **exactly 0**, and the bootstrap reports `saturated` |
| measured | filled by the run |

Its contrast case runs beside it: the same comparison under the self-play profile
must **not** be identically zero, or the detector is vacuous.

### S7 — greedy vs random, four-tupled

| slot | value |
|---|---|
| metric | mean paired difference in z, `greedy − random`, over the frozen paired set |
| floor | **−2.0** (as above) |
| **null** | **0.0** — established by S6's self-match run and by `rigged_null` |
| threshold | the paired-difference bootstrap's **95% CI excludes zero**, and the mean is **positive** |
| measured | filled by the run |

**Direction pre-stated: greedy > random.** A largest-subtree-first heuristic
should beat uniform legal play under the par game.

**This chunk's DONE-WHEN does not depend on S7's direction.** If greedy loses, or
if the CI straddles zero, that is a **finding about the heuristic**, filed with
its numbers — the ladder machinery is what chunk 10 ships, and a ladder that
reports an unflattering result is the ladder working. The distinction is stated
here, before the run, precisely so it cannot be drawn afterwards to explain a
number away.

---

## Reported verbatim

Wall clock, split by arm and by phase (census, freeze, pass, kill-resume). Row
counts and skip counts per arm. The census's two levels and their delta. Every
gate's four slots. `sympy`'s `cas_version`.

The paired set is **kept**, unlike the chunk-9 shakedown run directory: it is an
instrument, and its whole purpose is to be the same set next time.

---

## Comparands

- `runs/ANCHORS.sha256`
- `runs/data/eval_held_out`
- `runs/data/phase1_train`
- `runs/data/train_100k`
- `runs/paired/smoke_v1.jsonl`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
