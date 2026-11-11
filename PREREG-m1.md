# PREREG-m1.md — the M1 campaign, frozen 2026-08-16

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `M1-A<n>`.

**Status at commit: M1 has not run.** No number below is a campaign result. Every
number below is either a pre-launch measurement with its own record, or a
threshold derived from one on this page. Per standing law — *rule before
measurement, all the way down* — this file is committed first and the loop is run
against it.

---

## 1. How the primary was chosen, by mechanism rather than choice

`RULING-chunk11-primary.md` (2026-08-15) named search economy as the primary and
froze the rule that would site it: sweep the anchor's at-par rate over
`sims ∈ {6, 8, 12, 16, 48}`, take `s* =` the budget landing nearest 0.55 within
`[0.4, 0.7]`. It also pre-registered the succession in advance: *if the economy
primary saturates, the mid-strata suite is M2's primary, declared now, in
PREREG-m1's own text.*

The rule ran and returned nothing to select:

| sims | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| at-par | 1176 | 1176 | 1176 | 1176 | 1189 | 1189 | 1189 | 1189 | 1193 |
| rate | .9800 | .9800 | .9800 | .9800 | .9908 | .9908 | .9908 | .9908 | .9942 |

Every rung sits above the window's upper edge of 0.70. The rule fired
`all_above_window` and extended downward over `(1, 2, 3, 4)` under `P11B-A3`; at
`sims = 1` the domain has no floor left to extend into, and `P11B-A4`'s
reconciled `select()` returned:

```
s_star: null      needs: succession      at_domain_floor: true
successor_strata: [7, 8, 10]
```

**The anchor pars 98.00% of the 1,200-problem instrument at one simulation.**
Phase-1 imitation of BFS-optimal derivations already amortized the search into the
net — the AlphaZero effect arrived before the loop ran — so the suite-economy axis
is saturated at the floor of its own domain. Succession fires at iteration 0.

The record of that chain: `runs/chunk11_part0b_sweep.json` (union of two
invocations, both source digests and the judging `select()` version stamped
in-file), `PREREG-chunk11-part0bc.md` amendments A3/A4/A5.

---

## 2. P1 — the primary

**CI-separated improvement in pooled beat-par rate on the scripted strata
`{7, 8, 10}`, paired against the Phase-1 anchor.**

| slot | value |
|---|---|
| population | `scripted_in_7`, `scripted_in_8`, `scripted_in_10` — 200 each, **600 problems** |
| metric | pooled beat-par rate (the `<0` bin: strictly under scripted par) |
| pairing | **per problem**, against the anchor's Part-0d outcomes |
| test of record | **paired-difference bootstrap**, criterion **CI excludes zero** |
| reporting | pooled rate **and** per-stratum rates, every ladder pass |
| campaign | ~20 iterations at `--ladder-every 5` |

`scripted_in_9` is **informational only** and is not in the primary. Part-0d found
it the sole stratum born saturated from above — beat-par 164/200 = 0.82, past the
≥0.50 definition — so its trajectory is ceiling-bound at birth and would decorate
the RUNLOG without informing it.

**Why beat-par is the live cell here.** Scripted par is a *provisional floor*, not
an exact minimum, so beating it is legal on these strata — which is precisely what
the instrument was minted to allow, and what the exact-par suites cannot offer.

### 2.1 The four-tuple, filled by measurement

| slot | value | source |
|---|---:|---|
| **floor** | **0.000000** | a-priori: a beat-par rate has no value below zero |
| **null** | **0.000000** | **measured**, Part-0e: uniform stub, `0/600` |
| **baseline** | **0.168333** | measured, Part-0d: anchor `101/600` |
| **ceiling** | **1.000000** | distance from baseline: **0.831667** |

The null does not merely sit low on the scale — **it sits on the scale's floor.**
At `sims = 48` the uniform stub solves nothing at all on these strata: 600 of 600
unsolved, every episode `z = −1`, mean z exactly `−1.0000` on each of the three.
So every point of beat-par rate the loop earns is signal over a reference
contributing literally zero, and the anchor's 101 beats and 12 outright failures
demonstrate both ends of the instrument's working range before the campaign
touches it.

Record: `runs/chunk11_part0e_null.json`, `runs/chunk11_part0d_scripted_strata.json`.

---

## 3. The evaluation protocol, printed verbatim

Inherited from Part-0d without restatement, because **the baseline's protocol is
the primary's protocol** — a baseline measured under one protocol and a primary
measured under another are not paired, whatever the pairing code does. Printed
rather than referenced, because a protocol referenced is a protocol someone will
one day reconstruct differently.

```
sims                48
gumbel_m            16          via the declared clamp m = min(16, sims)
step_cap            24
root_noise          false
value_scale         0.0         value-silent until the switch criterion fires
measure_dtype       fp32
seed                0
device              cpu
model               runs/phase1/phase1.pt
config_fingerprint  f1d258a1161c5ba17f031a3ad89fae0e688ab135f8ced44c1dcd01b1e89491c1
```

The m-clamp is `min(16, sims)`, the frozen rule — **not** `min(5, sims)`. Over
the extended domain it binds trivially (`m = sims` for `sims ≤ 4`); at the
protocol budget of 48 it binds at 16.

---

## 4. No-regress

**The loop may not buy low-budget competence by selling high-budget competence** —
and, per the sweep's own gift, it may not sell the one-simulation competence
either. Two at-par floors on the 1,200-problem exact-par instrument, both rows
carrying their derivation.

### 4.1 The integerization convention, declared once for both rows

**Ceiling on the full-precision bound. No intermediate rounding anywhere.**

The justification is semantic, not numeric, which is what lets it travel to every
future floor without re-litigation: **a floor is an inequality.** The gate is
`count ≥ b` where `b` is a real number and counts are integers. Ceiling is the
only integerization that never admits a count the bound itself excludes.
Round-half-up would admit 1166 against a bound of 1166.4945 — a count the declared
construction rejects — and would do so or not depending on where the decimals
happened to fall.

Two notes kept because the record is the argument:

- **The convention had never been chosen.** The existing `sims = 48` row
  (`1187.8295 → 1188`) rounds to 1188 under nearest *and* under ceiling, so it
  never discriminated. The `sims = 1` row is the first case that does.
- **Intermediate rounding flips this row.** At four decimals,
  `0.9721 × 1200 = 1166.52`; at full precision, `1166.4945`. The full-precision
  bound is authoritative — `P11B-A2`'s lesson, on the row `P11B-A2`'s own
  arithmetic sits beside.

### 4.2 Row one — the sims = 48 floor

```
p            = 1193 / 1200                    = 0.9941666667
1 - p                                         = 0.0058333333
p(1-p)/n     = 0.9941666667 x 0.0058333333 / 1200
             = 0.0000048327546296
SE                                            = 0.0021983527
1.96 x SE                                     = 0.0043087713
lower bound  = p - 1.96 SE                    = 0.9898578954
as a count   = lower x 1200                   = 1187.8294744303
CEILING                                       = 1188
```

**Floor: 1188 / 1200 at sims = 48.** Reconfirmed unchanged under the
now-declared rule; this is not a retro-edit of an existing threshold.

### 4.3 Row two — the sims = 1 floor

```
p            = 1176 / 1200                    = 0.9800000000
1 - p                                         = 0.0200000000
p(1-p)/n     = 0.9800000000 x 0.0200000000 / 1200
             = 0.0000163333333333
SE                                            = 0.0040414519
1.96 x SE                                     = 0.0079212457
lower bound  = p - 1.96 SE                    = 0.9720787543
as a count   = lower x 1200                   = 1166.4945051681
CEILING                                       = 1167
```

**Floor: 1167 / 1200 at sims = 1.**

### 4.4 Both floors name their kind

Per the standing law countersigned at `P11B-A2`: every no-regress floor names its
kind where it is declared. **Both rows above are indistinguishability floors.**

> Holding 1188 at `sims = 48`, or 1167 at `sims = 1`, means **not below the
> anchor's own one-sided 95% band** at that budget. It does **not** mean *at least
> as good on every problem*, and it does **not** mean *at least as good on
> average*. Those are three different gates licensing three different sentences,
> and the derivations above make "distinguishable" mean exactly one computable
> thing.

### 4.5 Gates 10b and 11, re-declared on the ruled wiring

| gate | floor | null | threshold | measured |
|---|---:|---:|---:|---:|
| **10b** top-1 rule-site, depth ≤ 3, F-09 unseen subset | 0.5241 | 0.6803 uniform / 0.6950 untrained | 0.9000 | **0.9699** |
| **11** depth ≤ 2 solve rate, 16-sim search, value-silent-until-criterion (row 3) | 0.0000 | 0.9175 | 0.9500 | **1.0000** |

Gate 11 is carried at **row 3** — the ruled permanent wiring — because rows 1 and
2 are records of what was true on wiring since ruled non-permanent, and verdicts
do not retro-edit.

---

## 5. Informational rungs

At matched `s*` the ladder rungs report but do not gate: greedy and sympy against
the model, **both currencies in their lanes**. sympy is never par — a ladder rung
only, unless its derivations compile into our rule vocabulary.

`scripted_in_9` joins them as informational, per §2.

---

## 6. The entropy funnel signature, named

`p2_d`'s central finding — *the loop collapses when left alone* — enters here as a
design input rather than a surprise. Phase 3's diversity treatment fires when the
signature appears, so the signature is named before it can be recognised
retrospectively.

**Which H, on which population, by how much.** Four columns are already logged:

```
entropy_prior_step1_start      entropy_prior_step1_reached
entropy_target_step1_start     entropy_target_step1_reached
```

"Start" is an episode's first search; "reached" is every search after at least one
rewrite — the faithful analog of chess's start-vs-book split, an interpretation
and not a transplant, since there is no openings book here.

**The signature is the closing gap on the reached population, not a falling
entropy.** Prior entropy alone cannot distinguish a confident policy from a
collapsed one — that is the chess lesson, and it is why the search-improved target
is logged beside the prior. A *confident* policy has low prior-H while the search
still finds improvement, so the prior→target gap stays open. A *collapsed* policy
has low prior-H **and** low target-H: the search has stopped disagreeing with the
policy because there is nothing left to explore.

**Declared trigger** — both conditions, on the reached population, sustained over
**3 consecutive ladder passes**:

1. `entropy_target_step1_reached` ≤ **50%** of its iteration-0 value, **and**
2. the gap `entropy_target_step1_reached − entropy_prior_step1_reached` ≤ **25%**
   of its iteration-0 value.

`entropy_prior_step1_start` on the fixed start population is the **control**: if
start-H holds while the reached population collapses, the collapse is in the
trajectory the loop generates, not in the policy as such. Sustained over three
passes because Phase 3 is a one-way door and a single noisy pass must not open it.

A `0.0` in any column is a **premise-dependent reading, not a measured entropy** —
an iteration with no reached states has no reached entropy — and the population
sizes are logged beside it so the row's reader can tell.

> **First declared here.** The ruling asked for the signature to be named with its
> arithmetic; these two thresholds and the three-pass persistence are new on this
> page and have never been measured against. They are a trigger for a treatment,
> not a gate on the primary.

---

## 7. The switch criterion, its trajectory, and the watches

### 7.1 The criterion, carried unchanged from `PREREG-chunk9-shakedown.md` E5

| slot | value |
|---|---|
| metric | held-out z **balanced** accuracy |
| floor | **0.0**, marked uninformative — an anti-correlated head scores *below* the trivial model |
| null | **1/K**, K = classes with support, census on the row |
| threshold | **1/K + 0.15** |
| min class support | **100** in the rarest class, else **abstain** |
| behaviour | fires **once**, **ratchets**, every evaluation writes a row |

The margin is priced as a one-way door's error rate over campaign length:
at support 100, `se = 0.0354`, `z = 4.24`, `P(fire | null) = 0.001%` per
evaluation and **0.02% over 20 iterations**.

**Expected trajectory over M1.** The head is value-silent until the criterion
fires (`value_scale = 0.0`), so early iterations are expected to **abstain** on
class support rather than refuse on accuracy: the `+1` cell is the rare class on
the scripted strata, and it is the anchor's 101/600 that must grow before the
rarest class reaches 100. Abstention on support early, then either a fire or a
recorded refusal — and **an abstention is data, not silence**.

### 7.2 Abstention rows

Every evaluation writes to `runs/m1/value_switch.jsonl`, **including the ones
that abstain**, each carrying `class_census` and `smallest_class_support`.
`abstention_census` reports fired / refused / abstained / idle. An abstention
nobody records is indistinguishable from a criterion nobody ran.

### 7.3 The cross-switch watch, armed

Solve-rate-by-depth is compared across the switch iteration. A regression is
**noted and diagnosed**; there is **no auto-revert**. The ratchet stands — a
detector that undoes itself on a bad reading is a detector with a feedback loop,
and the diagnosis is the deliverable.

### 7.4 Pool composition growth

Reported every ladder pass: pool membership and its growth, with sampling
declared (`league.pool_sample: uniform`). A pool that grows without its
composition reported is a par source nobody can audit.

### 7.5 z-by-par_source — the draw-inflation watch

`z_by_par_source` is reported every pass, `+1 / 0 / −1` per source. **Par sources
do not pool.** The watch is for the failure mode where a rising z is bought by
shifting mass between par sources of different strictness rather than by solving
better — scripted par being a beatable provisional floor and BFS-exact par not
being beatable at all. A z that improves while its `par_source` mix moves is a
finding to diagnose before it is a result to report.

---

## 8. The BLOCKED branches

Three failed attempts at any gate → **BLOCKED**, never a weakened gate. A halt is
information; `BLOCKED-<date>-<slug>.md` is committed and tracked like any other
record.

Declared in advance:

- **P1 shows no CI separation over ~20 iterations.** Not a failure to hide: the
  successor axis was certified live by Part-0d — three strata with real headroom
  and 12 outright anchor failures — so a null result on a live instrument is a
  finding about the loop, and it is reported as one.
- **Either no-regress floor breaks** (at-par < 1188 at `sims = 48`, or < 1167 at
  `sims = 1`). BLOCKED, diagnosed before any further iteration. The loop does not
  trade one budget's competence for another's in either direction.
- **Gate 10b or gate 11 regresses** on the ruled wiring. BLOCKED.
- **The draw-inflation watch fires** — z improving while the `par_source` mix
  moves. BLOCKED pending diagnosis; a result bought by mix-shift is not a result.
- **The funnel signature fires.** Not BLOCKED: this is Phase 3's declared trigger,
  and it fires the diversity treatment, which is the pre-registered response.

---

## 9. What is open at the freeze

**Rider (b) item 2 — the failing-set identity — is an open finding and does not
gate this freeze.** Its bar was on the suite-economy axis, which succession has
made moot as a primary, and the no-regress floors it would inform (1188, 1167) are
identity-free aggregates. The diagnostic runs in parallel;
`runs/chunk11_misses_diagnostic.json` is its record.

Its question, registered before its run: the anchor at `sims = 1` misses 24 of
1,200 and the scripted solver misses 24 of the same 1,200. Two systems, one count,
one population — coincidence to dispose of in a line, or the same problems hard
for a greedy symbolic solver and a one-simulation neural policy alike.

---

*The primary was chosen by a rule. The rule was amended only where its inputs did
not exist. The successor was certified live by measurement, not by expectation.
Both floors carry their derivation, their kind, and the convention that
integerized them. Every branch that fired was written before its number was.*

---

# Amendment M1-A1 — 2026-08-16, the trigger's disambiguator, the convention's test, and a hypothesis registered mid-diagnostic

**Disclosure, first sentence, two parts.** §1 and §2 below affect **no measurement
that exists** — M1 has not run and the funnel trigger has never been evaluated.
§3 **is** written after seeing partial output: the misses diagnostic has returned
its `sims = 1` row and its scripted column, and their per-depth distributions
match cell for cell. It has **not** yet returned the by-key intersection, which is
the thing §3's predictions are about. The hypothesis is therefore registered
after the distributions and **before the keys**, and it is written down now
precisely so the keys score it rather than confirm it.

## 1. The funnel trigger's row carries the contemporaneous beat-par delta

§6 named the signature as the closing prior-to-target gap on the reached
population, and argued that a falling entropy alone cannot distinguish
confidence from collapse. **The same ambiguity applies to the gap, one level up.**
A gap that closes because the policy has learned what the search knows is
mastery; a gap that closes because the search has nothing left to explore is
collapse. The entropy columns cannot tell those apart either.

The disambiguator is whether the primary moved while the gap closed. So:

> **Every row that evaluates the funnel trigger carries the contemporaneous
> pooled beat-par delta on `{7, 8, 10}` against the anchor baseline of
> `101/600 = 0.168333`, and the per-stratum deltas beside it.**

Gap closing **with** the primary rising is mastery — the loop is learning, and the
Phase-3 treatment would be firing at success. Gap closing **with the primary flat
or falling** is collapse, which is what the treatment exists for. The trigger says
*look*; the row must carry *what to look at*, or the reader is left inferring the
distinction from two numbers that were never printed together.

The trigger's firing conditions in §6 are **unchanged**. This adds a required
column, not a threshold.

## 2. The convention is now executable-verified, and dual derivation is protocol

**What happened.** The ceiling in the derivation script was written
``-((-x) // 1)`` — correct for `int` and `float`, and **floor** for `Decimal`,
whose `__floordiv__` truncates toward zero. It produced 1187 and 1166 where the
declared construction gives 1188 and 1167. Two of this record's own rulings
composed into a defect at their seam: `P11B-A5` migrated the arithmetic to
`Decimal`, and the ceiling idiom silently changed meaning underneath it.

**What caught it.** Exactly one mechanism: the numbers had to agree with an
independent derivation. No test existed. Review would have read `ceiling` in the
name and moved on.

Two consequences, both landed:

**L1 treatment — the construction moves into code with reference vectors.**
`reckoner.gates.no_regress_floor`, `one_sided_lower_bound` and `ceil_count`,
tested in `tests/test_no_regress_floor.py` against both frozen rows to ten
places, an exact-integer boundary case, the property that ceiling never admits a
count the bound excludes, and the `Decimal //` trap itself pinned so nobody
re-enters it. It lives in `gates.py` for that module's own stated reason: a floor
expressed in a prereg and separately in a checker is two floors wearing one name.

**Dual derivation is the review, for any threshold that reaches a frozen page.**
The reviewer computes the number independently **before** reading the derivation,
deliberately rather than incidentally. This has been happening informally since
chunk 0; from here it is the protocol, because it is the only thing that caught
this and it caught it twice in one day.

## 3. The template-family hypothesis, registered with its discriminating predictions

**What is already visible.** The anchor at `sims = 1` misses 24 of 1,200. The
scripted solver misses 24 of the same 1,200. Their distributions:

```
scripted solver   by depth   {2: 4, 3: 4, 4: 10, 5: 6}   depths 1 and 6 perfect
anchor sims = 1   by par     {2: 4, 3: 4, 4: 10, 5: 6}   pars   1 and 6 perfect
```

Cell for cell, four cells, both endpoints perfect — from two systems sharing no
code, no representation and no search.

**Candidate mechanism, named so it can be killed.** *Imitation error concentrates
where BFS-optimality contradicts greedy salience.* These are the states whose
locally attractive move costs a step. The scripted solver takes that move by
design — it is greedy, and that is exactly why its par is a provisional floor
rather than an exact minimum. The anchor's residual 2% policy error pools in the
same states because the Phase-1 training signal fights salience hardest there. On
this reading the anchor's residual **is the greedy prior showing through**, and
six simulations of search rescue thirteen of the twenty-four.

**Discriminating predictions, registered before the keys return:**

| # | prediction | kills the hypothesis if |
|---|---|---|
| 1 | intersection is **24/24** — the same problems, not merely the same count | the sets differ materially by key |
| 2 | goal split of the shared set is **7 EVALUATE / 17 SIMPLIFY**, matching the scripted table | the shared set's goals do not match |
| 3 | the 48-sim residual **7 ⊂ the 24** | the stubborn seven include problems outside the shared set |

If the keys come back matching, this is not trivia: it is a portrait of what
Phase 1 did not learn, drawn before Phase 2 starts learning it. If they do not,
the coincidence dies in one table and this amendment records that it was a
coincidence.

**This remains an open finding and gates nothing.** Its record is
`runs/chunk11_misses_diagnostic.json`.

---

# Amendment M1-A2 — 2026-08-16, CampaignConfig, the thread pins, and the watchlist

**Disclosure.** M1 has not run; no campaign number exists. This amendment's
`episodes_per_iteration` is **derived from measurements taken after the freeze** —
the campaign-cost pilot of 2026-08-16, run on the campaign host — and those
measurements are pre-launch instrument timings, not results. The derivation
direction is the licensed one: thresholds freeze before measurement, treatment
values derive from it.

**One diff, one fingerprint, one era.** Every fingerprint change closes
comparison paths, so pending changes batch: `CampaignConfig` and the thread pins
land together rather than as two eras.

## 1. The exact diff, and both fingerprints of both profiles

The whole of it, additive — **no existing key changed value**:

```diff
+ campaign:
+   episodes_per_iteration: 400
+   interop_threads: 32
+   intra_op_threads: 8
+   iterations: 20
```

This project carries **two** fingerprints, and §3 above records only one of them:

| profile | pre-A2 | post-A2 |
|---|---|---|
| **eval** (`root_noise=False`) — what §3 records | `f1d258a1161c5ba1…` | `314fbeb99b6640f6…` |
| **campaign** (`root_noise=True`) — what the driver runs | `09157f706fcc3d0b…` | `ce41af96ee85f0a2…` |

**The driver asserts the campaign fingerprint `ce41af96…`**, not the eval one.
Stated because BRIEF-chunk11-driver's ruling 3 says "asserted == PREREG-m1's
recorded fingerprint", and taken literally that would have the driver assert an
eval-profile value while running the campaign profile — a check that could never
pass. Both profiles are now recorded so neither is inferred.

**Every pre-launch measurement ran under the pre-A2 values**, which their own
protocol blocks already state: Part-0b, Part-0d, Part-0e and the misses
diagnostic all record `f1d258a1…`, and the campaign-cost pilot deliberately cites
the pre-A2 era in its own record. Nothing is retro-stamped.

**The retired merge path stays retired.** `chunk11_part0b.load_prior()` will now
refuse a union against the existing sweep record, because the fingerprint moved.
That is the guard working, not a defect, and it is not special-cased.

## 2. The campaign's measured cost

Every term prices from a measurement already on the record. Host-scaling uses the
single dual-measured point: `1814.8 / 1475.88 = 1.2296` pod-over-local at
`sims=48`, basis stated rather than assumed.

**The cadence unit — one ladder pass, every fifth iteration:**

| term | cost | source |
|---|---:|---|
| no-regress suite @48 (1,200) | 30.2 min | pilot, pod-measured, 1.5123 s/ep |
| no-regress suite @1 (1,200) | 3.3 min | sweep `sims=1`, pod-scaled |
| **primary scripted {7,8,10} @48 (600)** | **70.9 min** | Part-0d per-stratum, pod-scaled |
| model rung (200 @48) | 5.0 min | pilot rate |
| greedy rung (200) | 0.0 min | pilot, 0.0062 s/arm-problem |
| **total** | **109.5 min** | |

**The primary's own measurement pass is the single largest term in the
campaign** — 70.9 of every 109.5 minutes. The scripted strata cost 5.844
s/problem, **4.75× the suite rate**, because par 7–10 problems run far longer
episodes. A rate carried across populations keeps its number and loses its
denominator; per-episode cost is a property of the problems, not of the machine.

**The campaign total**, 20 iterations, four cadence units, training fixed at
623.8 s/iteration:

| episodes/iter | self-play/it | train/it | 20 iters | instruments | **total** |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.5 m | 10.4 m | 4.3 h | 7.3 h | 11.6 h |
| 200 | 5.0 m | 10.4 m | 5.1 h | 7.3 h | 12.4 h |
| **400** | **10.1 m** | **10.4 m** | **6.8 h** | **7.3 h** | **14.1 h** |
| 800 | 20.2 m | 10.4 m | 10.2 h | 7.3 h | 17.5 h |

The instrument column is roughly half the campaign and is **invariant in
`episodes_per_iteration`** — it cannot be traded against.

## 3. `episodes_per_iteration = 400`, and which word applies to it

**CHOSEN, not derived.** No measured staleness optimum exists, so this is a
judgement. What is *computed* is its consequences, and those are shown:

| computed consequence | at 400 |
|---|---|
| fresh ring rows / iteration | 1,304 (3.26 rows/episode, pilot-measured) |
| training draws / iteration | 51,200 (400 steps × 128 batch) |
| **reuse ratio** | **≈ 40×** |
| ring fill after 20 iterations | 26,080 of 500,000 — **10%, never wraps** |
| **holdout rows / iteration** | **130.4** (`ring_holdout_frac` 0.1) |
| holdout rows over the campaign | 2,608 |

**The holdout line is the deciding one.** The switch criterion requires **100 in
the rarest class**. At 100 episodes that is 32.6 holdout rows per iteration — a
criterion structurally starved by its own support requirement, unfirable for most
of the campaign regardless of what the value head learns. At 400 it is 130.4, and
the criterion can wake.

**Registered as an M2 lever:** the reuse ratio, if staleness symptoms appear in
the loss curves or the funnel columns. Chosen here, revisable there, and named so
the revision is a decision rather than a drift.

## 4. The three pins, each with its evidential class

A pin is a claim about the future wearing evidence from the past, and the classes
license different sentences — the floor taxonomy's move, applied to pins.

**`intra_op_threads = 8` — EXERCISED.** The hosts' defaults *differed*: 16 on the
local box, 32 on the pod. Only `min(8, …)` made the licence's parity possible, and
the reproduction gate returned `1193/1200` with identical failing sets across
both. The clamp was load-bearing before anyone declared it; this pin declares it
with the measured counterfactual attached. Licenses: *this value was varied
across hosts and its effect measured.*

**`interop_threads = 32` — OBSERVED.** Constant at 32 on both hosts by coincidence
of default, never varied, never discriminated. Licenses **only** *this is what
ran*. It says nothing about whether interop affects numerics. A future host with
a different interop default stands **outside** this evidence, not inside it.

**The OMP family — RECORDED ABSENT.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` were **unset** on both hosts
throughout the licence. Unset is a value: setting them later is a gated
configuration change, not a tidy-up. The driver asserts they are unset at
startup.

**Raising any of these is not a config tweak.** It is a new configuration whose
reproduction gate re-runs — the set-growth law, standing at the thread knob.

## 5. The watchlist — informational, no gate contact

The 24 problems certified on 2026-08-16 as the shared miss set of the anchor at
`sims=1` and the scripted solver (intersection 24/24, Jaccard 1.0) enter the
RUNLOG as a **frozen** named watchlist. Two columns, computed per ladder pass
from data the pass already produces:

```
family_remaining = |pass_miss_set ∩ frozen_24|      the family shrinking
novel_misses     = |pass_miss_set \ frozen_24|      a new family growing
```

Neither column re-derives the family. A re-derived family per pass would conflate
*"the family shrank"* with *"the family's definition moved"* — a histogram whose
bins drift, cured the same way: freeze the reference, observe against it.

The pair exists because the aggregate no-regress floor catches net regression and
**cannot attribute it**. Together these two can, for free.

The question they answer: does self-play training shrink the trap family — does
search correcting the policy at root-adjacent traps teach the correction back
into the priors? That is the amortization thesis at its most localized.

**No gate contact. Informational only.**

## 6. The switch criterion's classes are two-population, not two-class

Stated because the machinery is already correct and the *wording* is what would
mislead a reader of the first `+1`.

Self-play draws par from two populations: `par_from_pool_frac = 0.2` with
`seed_pool_with_anchor = True`, so **80% of ring rows carry `bfs` par and 20%
carry `pool` par.**

* Against **`bfs`** par, `+1` is impossible by construction, and the tripwire
  scopes the impossibility to exactly those rows — `replay.py` raises on
  `par_source == "bfs" and z > 0`, the third layer of the same check.
* Against **`pool`** par, `+1` is **beatable by construction** and merely rare by
  dynamics early: at iteration 1 the model ≈ the anchor that seeded the pool, so
  it matches its own snapshot's step count and pool rows score 0.

So `K` — classes *with support* — starts at **2** and becomes **3** when the model
begins beating pool snapshots, at which point the threshold `1/K + 0.15` moves
from **0.65** toward **0.483** by the criterion's own definition. No machinery
changes; `valuegate` already anticipates the two-class start.

**The first `pool`/`+1` row in the ring is not an anomaly. It is the escalation
mechanism working, and a campaign milestone** — greeted, not investigated.

---

# M1-A3 — Pool composition becomes a logged column (schema era 3)

**Ratified 2026-08-17, on F-23's argument.**

## 1. What changes

`ITERATION_FIELDS` gains **`pool_composition`** (`dict`, role `diagnostic`,
`since=3`), and `SCHEMA_ERA` becomes **3**. The column carries the pool *as the
iteration drew from it* — captured before that iteration's own enrolment, since
reading it after `enroll` would log a pool one rung deeper than the one that
actually supplied par.

```
{"size": int, "steps": [int], "order": [int], "value_head_live": [int]}
```

`size` 0 is a **value**, not an absence: it says the pool is empty and par has
nothing to escalate from. That is why the column is required rather than
optional — the alternative reintroduces exactly the ambiguity between *empty* and
*unrecorded* that the absence-with-reason discipline exists to remove.

## 2. Why a frozen page is amended for it

`CheckpointPool.composition()` has existed since chunk 9 and was logged by
nothing. F-23 is the demonstration of what that costs: resume rebuilt the pool
with the anchor alone, and **every column in the iteration row agreed with an
uninterrupted run while the ring the model trains on differed.** Par escalation —
the mechanism §6 is denominated in — silently reset, and no row comparison at any
strictness could have reported it, because the divergent state was not in the
schema.

§6 is analytically dependent on this. A pool/bfs ratio is uninterpretable without
knowing how many rungs the pool held: the same 80/20 split means one thing
against a pool of eight snapshots and another against a pool of one, and the
transition of `K` from 2 to 3 — the milestone §6 registers — is a claim about the
pool's contents that the artifacts could not previously substantiate.

This is rider (b) standing on a state variable rather than an artifact: the
capability existed, the consumer was never written.

## 3. `order` is not redundant with `steps`

`steps` is sorted — the pool's identity, stable under how it was assembled.
`order` is the members list as `sample` sees it, and `sample` is `rng.choice`
over that list, so **the same membership in a different sequence draws a
different snapshot.** F-23 turned on precisely that: the fix replays the original
enrolment sequence rather than restoring a set. A composition column carrying
only the sorted view would have been structurally unable to see the defect it was
added for.

The two agree once eviction has run, since eviction re-sorts by step. They differ
while the pool is under capacity — the campaign's opening iterations.

## 4. Cost

Era machinery: zero lines. A row written under era 1 or 2 may omit the column,
and that absence is computed from (row era, `field.since`) rather than asserted
by the row — a run that predates a column cannot have explained the absence of
something nobody had named. The one-line consequence at each of the three
`iteration_row` call sites is a required keyword argument, which is the point:
the column cannot be forgotten by a new caller.

**No gate contact.** No threshold, floor or primary is denominated in this
column. It is the evidence that §6's denominator was what it claims.

---

# Amendment M1-A4 — 2026-08-21 — the fingerprint moves

**Full text: `AMENDMENT-m1a4.md`.** Recorded here because M1-A2 §1's table is the
frozen page's copy of the fingerprints, and a page whose recorded value has been
superseded elsewhere is a page that lies to the next reader.

| profile | M1-A2 recorded | **M1-A4 records** |
|---|---|---|
| **eval** (`root_noise=False`) | `314fbeb99b6640f6…` | **`c8aa1fccd1c97dd8…`** |
| **campaign** (`root_noise=True`) | `ce41af96ee85f0a2…` | **`8443847bb8c41218…`** |

**The superseded values are not deleted.** Every measurement taken through the
rehearsal, the equivalence gate and Part-0d's re-run ran under `ce41af96…` /
`314fbeb9…`, and those numbers remain readable only against the config that
produced them. Verdicts do not retro-edit.

## What moved it — one change, six decisions

| decision | effect on the fingerprint |
|---|---|
| §5 `train.rehearsal_frac` 0.0 → **0.65** | **value change** |
| §4 `ladder.sympy_step_budget`, `sympy_time_budget_s` **deleted** | **fields removed** |
| §6 `campaign.interop_threads` OBSERVED → EXERCISED | role only, value unchanged |
| §4 `ladder.problems_per_pass` → per-file population cap | role only, inert at 200 |
| §2 `league.snapshot_every` ignored → honoured | role only, inert at 1 |
| §3 watchlist per leg on the pass record | not config |

**The batching rule is why this is one row and not five.** A fingerprint that
moves five times carries five partial explanations; this one moves once, with a
page that explains all of it.

## What did NOT move, and why that is the point

§4's census originally proposed deleting eight fields. Six were **kept**:
`model.param_budget_min/max` are enforced by a guard test, and the four
`generator.*` fields were **wired** to `generate.py` instead of deleted — which
converts F-33's coincidental provenance agreement into a causal one, and is
provably inert because the CLI defaults already equalled the config values.

The census had reported all eight dead. It answers *"is this consumed by the
campaign?"*, and its output was read as *"is this doing any work?"* — the same
species of true-in-one-dimension claim this project has now catalogued nine
times. The census's scope is fixed rather than footnoted: it reports
**runtime-live** and **guard-live**, and only *neither* is a deletion candidate.

---

## Comparands

- `runs/chunk11_misses_diagnostic.json`
- `runs/chunk11_part0b_sweep.json`
- `runs/chunk11_part0d_scripted_strata.json`
- `runs/chunk11_part0e_null.json`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*

---

## Post-close note (2026-11-10, F-54). A note, not an amendment: M1 is closed

- **M1's `entropy_target_*` columns hold H(visit share).** Since schema era 4,
  that column holds H(π′), and the visit-share entropy moved to
  `entropy_visits_*`. Read M1's rows by their era (3); never compare them with
  era-4 rows under one name.
- **§6's funnel arithmetic does not transfer.** Under π′, the target entropy
  equals the prior's exactly wherever the search found nothing, so a
  target-versus-prior entropy gap at iteration 0 is about zero by construction.
- **The next campaign's PREREG re-declares the funnel.** It uses
  `kl_improved_prior_*` (and `entropy_target_*`), each with its four-tuple:
  floor, a null run (the uniform stub at the same budget), threshold, and
  measured value.
- **The campaign door no longer pins M1's pair.** Its fingerprints have moved
  twice since M1, one key each time: `campaign.device` (2026-08-22, after
  F-51), then `train.policy_target` (F-54). M1's pair survives as
  `M1_ERA_FINGERPRINTS` in `campaign.py`, for the scripts that verify M1's own
  rows.

**Correction to the note above (2026-11-11, F-54 review).** "The target entropy
equals the prior's exactly wherever the search found nothing" should read: equals
it up to float32 rounding. `kl_improved_prior_*` is the exact test, and it is
exactly 0 there.
