# GATE-chunk8-VERDICT.md — Phase 1: supervised warm start

Verdicts for chunk 8's DONE-WHEN gates, recorded under the principal's ruling of
2026-08-15. Every row carries rider (c)'s **four-tuple** — floor, null-model
baseline, threshold, measured value — because a threshold nobody computed the
floor of is not a gate.

Two words are used precisely throughout. **PASSED** means the measured value met
the registered threshold. **VACUOUSLY** means it would have done so for any
model, including an untrained one. They are recorded together where both apply,
per literal-verdict law: the literal result stands, and the finding files beside
it rather than replacing it.

---

## Gate 10 — held-out policy accuracy

### 10a. As registered: `top-8 rule-site ≥ 0.90 on depth ≤ 3`

| | |
|---|---:|
| floor (fraction of depth ≤ 3 states with ≤ 8 legal actions) | **0.9897** |
| null model (untrained network, random init, seed 0) | **1.0000** |
| threshold | 0.9000 |
| measured (trained) | 1.0000 |

**Verdict: PASSED — VACUOUSLY.** The threshold sits below the floor. Top-k is
ranked over the legal action set, 98.97% of depth ≤ 3 held-out states have ≤ 8
legal actions, and a randomly initialised network scores 1.0000. The gate cannot
distinguish 5,000 steps of training from none. Filed as `FINDINGS.md` F-10.

### 10b. As amended, under the principal's signature via the plan's revision channel

**`top-1 rule-site ≥ 0.90 on depth ≤ 3, evaluated on the F-09 unseen subset.`**
The amendment folds F-09 and F-10 into one instrument: the metric that
discriminates, measured on the states the model has not seen. Its arithmetic is
stated **at declaration**, not afterwards:

| | |
|---|---:|
| floor (fraction of depth ≤ 3 states with ≤ 1 legal action) | **0.5241** |
| null model — uniform-random, E[1/B] | **0.6803** |
| null model — untrained network, measured | **0.6950** |
| threshold | 0.9000 |
| headroom demanded above floor | **0.3759** |
| **measured (trained, unseen subset)** | **0.9699** |

**Verdict: PASS. Discriminating by construction.**

### Depth-stratified top-1, unseen subset (n = 5,172)

| depth | n | top-1 |
|---:|---:|---:|
| 1 | 368 | 1.0000 |
| 2 | 345 | 0.9652 |
| 3 | 516 | 0.9516 |
| 4 | 1,606 | 0.9782 |
| 5 | 1,725 | 0.9913 |
| 6 | 612 | 1.0000 |
| **≤ 3 (the gate)** | **1,229** | **0.9699** |

### All three numbers, side by side, per the F-09 ruling's both-numbers form

| measurement | n | depth ≤ 3 top-1 |
|---|---:|---:|
| all held-out states | 1,744 | 0.9782 |
| **unseen subset (the gate)** | **1,229** | **0.9699** |
| inflation delta | | **+0.0083** |

The contaminated instrument reads 0.83 points high. Unseen n = 1,229 ≥ 1,000, so
`eval_held_out_v2` is **not triggered** and v1 stands unmodified. The exclusion is
applied at measurement time, not by deleting data, so one artifact serves both
numbers permanently.

---

## Gate 11 — depth ≤ 2 solve rate under 16-sim search

Registered config, `m ≥ 5` per item 9's B_max arithmetic. Solving means *playing
the episode* — search, apply, repeat — not "a solve appeared somewhere in the
tree". Step budget `min(step_cap, 2 × par + 2)`, declared.

| | |
|---|---:|
| floor (a-priori: no state solves itself) | 0.0000 |
| **null model — uniform-prior stub, same budget, same problems** | **0.9175** |
| threshold | 0.9500 |
| **measured (trained)** | **1.0000** |
| null-to-threshold margin | 0.0325 |

**Verdict at the time of measurement: PASS.** Superseded — see the three-row
ledger below. Per suite as measured then: trained 200/200 on both `solve_in_1`
and `solve_in_2`; stub 200/200 and 167/200.

### Gate 11's three rows — verdicts do not retro-edit

| # | wiring | measured | threshold | verdict |
|---|---|---:|---:|---|
| 1 | solved-flat terminal scale (as chunk 8 ran) | 1.0000 | 0.9500 | **PASSED** |
| 2 | z currency, value-head noise **live** (F-13 re-run) | 0.9200 | 0.9500 | **SHORT** |
| 3 | z currency, **value-silent-until-criterion** (the ruled permanent wiring) | **1.0000** | 0.9500 | **PASS** |

Row 1 stands as measured — a verdict is a record of what was true on the wiring
it ran on, and rows 2 and 3 are *added*, not substituted. Row 2 is the honest
reading on wiring that has since been ruled non-permanent. **Row 3 is the gate on
the wiring the project will actually run**, and its number is already in the
record: it is the zero-value arm of the F-14 table, which is what
value-silent-until-criterion *is*.

`FINDINGS.md` F-14 carries the mechanism and the gate-12 erratum.

**The first null measured for this gate was wrong** and said 1.0000, which would
have made gate 11 vacuous like gate 10. The cause was a constant-seeded search
rng; see `FINDINGS.md` F-11. The correction changed the verdict, not just the
number.

At `m = 1` the same gate reads null 0.7725 against measured 1.0000 — a far wider
margin, because at `m = 1` the prior does the work the search does at `m = 5`.
The tension between item 9's reachability requirement and rider (c)'s
discrimination requirement is recorded in F-11.

---

## Gate 12 — zero-value ablation (diagnostic, not a threshold)

The W/D/L head is trained at loss weight **0** by design (spec §5; Phase-1 value
is the steps-to-solve regression), so its output at inference is an untrained
head's noise. The ablation forces value to 0 and re-runs.

| arm | depth ≤ 2 solve rate |
|---|---:|
| trained (value head live) | 1.0000 |
| **zero-value ablation** | **1.0000** |

**Result as reported: the untrained value head is inert.** Identical to the digit
at both `m = 1` and `m = 5`.

> **ERRATUM, 2026-08-15 — that reading was an artifact of the scale it was taken
> on.** Under the solved-flat terminal value a solve backed up `+1.0` and drowned
> any noise the head could emit, so "inert" was unmeasurable rather than true.
> Under the z currency an at-par solve is `0.0`, noise around zero beats it, and
> the ablation separates a working search from a broken one by **8 points**
> (`FINDINGS.md` F-14). The ablation was measuring a quantity the scale had made
> unmeasurable. The design decision it was planted to test — brief item 11, *"if
> the untrained head hurts, that is a finding, filed"* — fired one semantics era
> late and eight points wide, and it fired.

---

## Gate 13 — reproducible from seed

**Declared tolerance: exact.** Not "close" — bit-identical loss curves. On CPU
with every RNG seeded from config there is no source of nondeterminism to
tolerate, and a tolerance would be a place for real drift to hide.

**Declared scope:** 5 steps at batch 16, exercising sampling, batch construction,
the crop, both losses, clipping and the optimiser step. It does not cover
long-run accumulation; a 2 × 83-minute repeat is not a `make test` cost, and
saying so is cheaper than implying coverage that does not exist.

**Verdict: PASS.** Seed-identical runs produce identical loss lists; a different
seed produces a different one, so the assertion cannot pass on a loop that
ignores its inputs. Pinned permanently as
`tests/test_train.py::test_training_is_reproducible_from_seed`.

---

## The six informational baselines — `solve_in_1..6` @ 16 sims, m = 5

Not gates. Recorded so chunk 9 opens against numbers rather than impressions.
Record: `runs/gate_phase1_baselines.json`.

| suite | trained | stub null | zero-value | trained median steps | par |
|---|---:|---:|---:|---:|---:|
| `solve_in_1` | **1.0000** | 1.0000 | 1.0000 | 1 | 1 |
| `solve_in_2` | **1.0000** | 0.8350 | 1.0000 | 2 | 2 |
| `solve_in_3` | **1.0000** | 0.5350 | 1.0000 | 3 | 3 |
| `solve_in_4` | **1.0000** | 0.5200 | 1.0000 | 4 | 4 |
| `solve_in_5` | **1.0000** | 0.5450 | 1.0000 | 5 | 5 |
| `solve_in_6` | **1.0000** | 0.5350 | 1.0000 | 6 | 6 |

**1,200 of 1,200**, and the median step count equals par at every depth — the
warm start is not merely solving, it is solving at par on at least half of every
suite. The stub null collapses to ~0.53 from depth 3 onward, so the separation
is 0.465 at the deep end where it matters. The zero-value ablation is identical
at every depth: the untrained W/D/L head is inert throughout, not just at depth 2.

**What these numbers do not say, said here rather than left to be assumed.**

1. **The step distribution against par is not captured — only its median.** In
   this project that distribution *is* z, the game's own currency: equal to par
   is a draw, over par is a loss. A median at par tells us at least half draw; it
   does not give the loss tail. The instrument records `median_steps_when_solved`
   and should record the full histogram. **Registered for chunk 9** rather than
   re-run here — it is a 30-minute measurement and it was not what the brief
   asked for, but the closing report should not imply coverage it does not have.
2. **The step budget is `2 × par + 2`, declared.** A solve at depth 6 had up to
   14 steps available, so "solved" here permits recovery from a wrong move. That
   is the right budget for a *solve rate* and the wrong one for a *par rate*,
   which is the same gap as (1) seen from the other side.
3. **Suite start states are verified absent from training (F-08); their
   intermediate states are not.** A suite problem's solution path may pass through
   states that other problems' derivations contributed. That is not contamination
   under the registered definition — the problem is unseen — but it is the
   honest boundary of what "unseen" covers here.

---

## Gate 14 — anchor by digest

See `runs/phase1/anchor.json` and the `runs/phase1/phase1.pt` line in
`runs/ANCHORS.sha256`. The anchor refuses to write if `phase1_train`'s digests no
longer match what the run recorded — an anchor whose training data was rebuilt
under it is not the anchor anyone thinks it is.

---

## The run

| | |
|---|---:|
| steps | 5,000 (A4 bound 10,000, **one extension unspent**) |
| wall clock | 82.8 min (pilot projected 83.0 — 0.3% error) |
| device / batch | CPU / 128, cosine + 200 warmup |
| NaN skips / encode skips | 0 / 0 |
| loss, first → last windowed mean | 1.4901 → 0.0673 |
| overall top-1, untrained → trained | 0.4791 → 0.9800 |

**No extension**, by a rule fixed before the numbers landed: loss improvement
over the final 500 steps was 1.8% against a 5% bar, and the registered gate was
not short. F-10 supplies a third reason — extending toward a gate that random
initialisation already passes is meaningless.

---

## Spec-clarifying erratum — "z/q blend on by default from day one"

**A clarification, not a change.** Plan chunk 9 and `train.value_q_mse_weight = 0.5`
carry the clause *"on from day one"*, imported from p2_c. That clause assumed a
**trained** value head — in the source project the head had been trained on real
outcomes before the blend was switched on.

Phase 1 breaks the premise rather than the clause: spec §5 trains Phase-1 value on
steps-to-solve and leaves W/D/L at loss weight 0, because imitation data has
degenerate z (every row is on an optimal derivation, so every episode is a draw).
So at iteration 0 the head is not a weak predictor of z — it is **not a predictor
of z at all**.

The F-14 ruling restores the clause's intent: **day one of a trained head.** The
blend is on by default, and the head it blends becomes live when the pre-registered
criterion clears. Nothing about `value_q_mse_weight` changes; what changes is that
"day one" now names the day the premise holds.

---

## Comparands

- `runs/ANCHORS.sha256`
- `runs/gate_phase1_baselines.json`
- `runs/phase1/anchor.json`
- `runs/phase1/phase1.pt`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
