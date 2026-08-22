# BRIEF-chunk9.md — Phase 2: the loop (2026-08-15)
# Plan chunk 9 + Amendment v1.1 deltas are the spine. This file governs.

## Part 0 — ROCm (A2 executed here)
0a. Index swap per §4 step 2; `rm -rf .venv && make install`; uv.lock change
    → clean-clone gate re-run. Blessed stack:
    `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; `torch.compile` stays OFF
    (gfx1151). check_env output in the run dir.
0b. GPU/CPU equivalence smoke on a fixed batch, numeric tolerance DECLARED
    before measuring (bit-identity is not expected across devices; say what
    is). Both polarities: a perturbed weight must break it.
0c. Box-occupancy declaration vs the chess project before any GPU run.
0d. Four-tuple discipline applies to every gate this chunk declares.

## Design (decided; deviations via BLOCKED)
1. Batching is ACROSS concurrent episode searches, never within a tree —
   the plan's "batched leaves" wording predates F-06; `batch_searches` is the
   key and the law. Descent gate (evals == nodes) rides into the runner.
2. Replay ring: `root_q` from field one; `_FIELDS_SINCE` era handling; one
   absence semantics (masked, reason carried).
3. z/q blend ON by default (`value_q_mse_weight: 0.5`) — docstring states why
   the self-referential hazard is structurally absent here (the checker,
   not Q, decides solved). W/D/L head wakes on real z.
4. Snapshot league: `CheckpointPool`; `league.par_from_pool_frac = 0.20`; pool
   par re-solved fresh at episode time; POOL LOADS ASSERT
   `ruleset_version`/`vocab_version` MATCH (registered; F-02 one layer up).
   `par_source="pool"` with `par_asof`; z>0-vs-exact invariant already guards.
5. All harness randomness is per-problem derived seed fan-out, never a
   shared stream — gate 11's null defect, generalized. Eval profile
   deterministic; self-play profile stochastic by design.
6. Rehearsal ported DORMANT (`rehearsal_frac: 0.0`) — the lever exists before
   it's needed. Resign-vs-par implemented, default OFF, calibration
   deferred to campaign evidence.
7. `logschema.py` from field one: policy entropy at step 1 (H_prior AND
   target-H — the chess lesson), start-position vs book-position entropy
   split, solve rates by depth, z composition BY `par_source` (draw-inflation
   watch), StateTooLarge counters, wall-clock split, skips,
   absence-carries-a-reason throughout. The z step-distribution registered
   at chunk-8 close lands in these columns.
8. Crash-resume: write-ordering contract, two SIGKILL points, resume test
   through both; any dedup key goes through the shared identity normalizer.
9. Golden mini-run < 3 min in `make golden`, runs on CPU regardless of
   training device.

## DONE-WHEN
- Part 0 battery green with declared tolerance; occupancy declared.
- Golden green; both kill-points resume clean.
- 3-iteration shakedown at default config with pre-registered plumbing
  expectations (rows populate, splits sum, snapshot loads, pool par carries
  provenance) — expectations written BEFORE the shakedown runs; shakedown
  deleted after recording, per plan.
- Four-tuples on any declared gate; verbatim numbers; tree-state block.
Three failed attempts at any gate → BLOCKED, never a weakened gate.

---

# APPENDED 2026-08-15 — obligations registered by earlier chunks

Preserved verbatim from this file's previous contents. The brief above governs;
these are prior registrations it must satisfy, not competing instructions. They
are appended rather than merged because amendments append and never edit
(AGENTS.md §7). Items 4 and 3 of the brief above are the same obligations
restated by the principal; the original wording is kept so nothing is lost in
the restatement.

## CheckpointPool must refuse a version mismatch

**Registered:** 2026-08-14, chunk 6.

`CheckpointPool` loads snapshots to supply **pool par** (`league.par_from_pool_frac`,
default 0.20). A snapshot whose `ruleset_version` or `vocab_version` differs from
the running environment must be **refused, not loaded**.

**Why, precisely.** Pool par is a par *re-solved by an old snapshot at episode
time*. If that snapshot indexes a different action space or a different symbol
space, the par it produces is denominated in a rule system that is not the one
being played — which is `FINDINGS.md` F-02 reborn one layer up, and harder to
see: F-02 shipped a wrong number with a provenance tag that said `bfs`, and this
would ship a wrong number with a provenance tag that says `pool` and is, in its
own terms, true.

**Already in place:** `model.load_checkpoint` refuses on mismatch by default and
`checkpoint_meta` stamps both versions. Chunk 9 must not pass
`strict_versions=False` in the pool path — the escape hatch exists for offline
inspection, not for the league.

**Test it on both polarities**, as always: a matching snapshot loads, a mismatched
one raises, and the pool surfaces the refusal as a counted event rather than a
silently smaller pool.

## The z/q blend's hazard is structurally absent — say so where it is used

`train.value_q_mse_weight` is 0.5 from day one. The self-referential hazard that
makes blending dangerous elsewhere does not exist here because the **checker**,
not the model's own Q, decides solved. State that in the loss docstring so a
later reader does not "fix" it by turning the blend off.

## Snapshot par and the win-condition invariant

`EpisodeResult` raises when `z > 0` against an exact `par_source`. Pool par is
*not* exact — beating it is the whole escalation mechanism — so pool-labelled
episodes will legitimately produce `z = +1`. Do not widen the invariant to
silence them; that is the difference the invariant exists to preserve.

---

## Comparands

- `uv.lock` — the blessed stack Part 0 pins and re-gates on change

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
