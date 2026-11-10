# Amendment M1-A4 — 2026-08-21

**Six decisions, each carrying its measurement.** Batched deliberately: every
item moves the config fingerprint, and the batching rule exists so a fingerprint
moves once with a page explaining all of it rather than six times with six
partial explanations.

---

## 1. The rung passes are dropped from M1 (F-34)

**Decided. Both options priced; (b) taken on measurement, not cost.**

M1-A2 §2 prices a cadence term — model rung 200 @48 at 5.0 min, greedy rung at
0.0 — that corresponds to no code. Building it needs a population and a budget,
and **both referents evaporated**: PREREG calls the rungs *informational at
matched `s*`*, and `s*` resolved to **null** when branch (c) fired.

| option | verdict |
|---|---|
| **(a)** `smoke_v1` entire — 389 problems, born clean of F-09, ANCHORS-registered, purpose-built for the ladder, ~10–17 min | **rejected** |
| **(b)** drop the rungs from M1, carry to M2 | **taken** |

(a) is rejected because M-B already measured the instrument **saturated**: par is
BFS-exact so `+1` is impossible, the model sits at 387/389, and the entire
remaining headroom is **12% of one CI half-width**. A rung pinned at its ceiling
reports the ceiling. The suites are saturated too (F-20), and the scripted strata
are the only live population — already the primary, where a rung would duplicate
rather than inform.

Nothing gates on their absence; PREREG calls them informational. M2 mints a rung
population with **headroom demonstrated before freezing**, which is the lesson the
succession to the scripted strata already taught once.

## 2. `league.snapshot_every` is honoured by the driver (F-36)

**Decided, built, and proved inert.**

`shakedown.py` honoured it; the driver enrolled unconditionally. They agreed only
because the default is **1** — correct by coincidence, the fourth appearance of
that shape, and the first caught before it could diverge. At 5 the two
compositions would differ in pool growth and therefore **par escalation**, the
mechanism the primary is denominated against.

Proved inert at the current value: **pool composition bit-identical** across the
change at `snapshot_every = 1`. Tested at both values, because "agrees at the
default" is precisely the evidence that let the divergence sit. One predicate
serves the enrolment site and resume's pool rebuild, since a rebuild replaying a
different cadence would reconstruct a pool the run never had.

## 3. The watchlist emits a pair per leg, on the pass record (F-35)

**Decided and built.**

Per pass, not at `min(budgets)`: each leg has its own miss set, and the 48-sim
reading is the one **commensurate with the primary's budget**. Reading only at
`min(budgets)` is what let a prediction registered at 48 sims be scored against a
1-sim number — 24/266 predicted, 20/458 measured at 1 sim, while the 48-sim
reading was **7/283** and told a different story entirely.

It rides on the **pass record**, not as budget-named iteration columns: a budget
in a column name puts the budget in two homes — config and schema — where they
can drift. The partition identity holds where its inputs live, asserted at
construction.

**Era 4 is redefined away.** No committed artifact carried an era-4 row, and the
triple moved off the iteration row entirely, so the iteration schema returns to
its era-3 shape and `SCHEMA_ERA` returns to 3. An era is frozen only once
something durable claims it.

### 3.1 The first reading, and what it establishes

| | misses | `family_remaining` | `novel_misses` |
|---|---|---|---|
| ckpt-4 @48 | 290 | **7** | 283 |
| ckpt-4 @1 | 478 | 20 | 458 |
| anchor @48 | 7 | 7 | **0** |
| anchor @1 | 24 | 24 | **0** |

ckpt-4's seven family misses at 48 sims are the **identical set** to the anchor's
seven; both rescue the same 17 of 24.

**So the frozen family is not where the degradation lives.** All 283 novel misses
are problems the anchor solved. Top-1 ordering carries the easy population;
search carries the hard family; forgetting destroys the former and leaves the
latter untouched.

**Consequence for how the campaign's own numbers are read:** the primary is a
**low-sensitivity instrument for this failure mode.** Its population is
search-carried, so it *rose by 7* while the model lost 283 easy problems. §8's
"either direction" clause names budgets; this is the **population** version of the
same trade, measured rather than assumed. The floors caught it only because they
happen to sit on the sensitive population.

## 4. Dead-key dispositions (F-31's census)

**Decided.** Batched here because each deletion moves the fingerprint.

| field | disposition |
|---|---|
| `train.rehearsal_frac` | **wired** (F-31) — value in §5 |
| `campaign.interop_threads` | **kept, now verified** — OBSERVED per M1-A2 §4, asserted against the runtime |
| `ladder.problems_per_pass` | **wired** as the per-file population cap; inert at campaign config, where every file is exactly 200 |
| `league.snapshot_every` | **wired** (§2) |
| `ladder.bootstrap_resamples` | **kept** — live once F-30's baseline arm is consumed by the bootstrap |
| `search.perspective` | **kept, exempt** — pinned invariant, legal range is one value |
| `generator.{train_set_size, suite_depths, suite_problems_per_depth, max_bfs_depth}` | **WIRED** — `generate.py` now reads them instead of hardcoding literals |
| `model.{param_budget_min, param_budget_max}` | **KEPT** — a guard test enforces the envelope |
| `ladder.{sympy_step_budget, sympy_time_budget_s}` | **DELETED** — the rung they configure left M1 with F-34 |

### 4.0 The census proposed eight deletions and six were wrong

The disposition table above is the **corrected** one. The census originally
reported all eight as unbacked and dead, and the count-before-touching rule
caught it:

* **`param_budget_min/max` are guard-live.** `test_model.py` asserts the built
  model's 5,073,156 parameters against that envelope. Varying the field across
  its legal range fails the guard — which is "changes what runs" under this
  project's own adopted rule.
* **The four `generator.*` fields are wired, not deleted.** Deleting would move
  the only declaration of dataset sizes out of fingerprinted config and into a
  script's argument default — the opposite direction from `DATA_ROOT`, `ANCHORS`
  and every pin this project has laid down. Wiring converts F-33's *coincidental*
  provenance agreement into a **causal** one, costs three lines, and is provably
  inert: the CLI defaults already equalled the config values, asserted in
  `generate.py` so the equality cannot drift back into coincidence.
* **The two `ladder.sympy_*` fields are deleted**, with the reason recorded: they
  configure a rung F-34 dropped from M1, and when the sympy rung returns in M2
  its budgets get declared at the site that consumes them rather than riding as
  dead weight through a campaign fingerprint.

**The census's scope is fixed rather than footnoted.** It now reports
**runtime-live** and **guard-live** separately, and only a field that is
*neither* is a deletion candidate. It also gains a **known-negative reference
vector** — a field live only through a test guard — beside the known-positive it
already had. Its scope has now been corrected five times, every one found by use
rather than by review, which is the argument for both-polarity vectors.

### 4.1 Ratified: the `assert_eval_profile` split (touches D-A2 §2)

The registered eval fingerprint was asserted unconditionally, which made the
shared seam **unrunnable at golden config** — and that is how the cadence path
came to have no test at all, and how two campaign-blocking defects reached the
pod.

Split: **the profile must ACT** (root noise off) is asserted on every pass,
whoever calls; the **registered pin** is asserted when the pass is a campaign
measurement. Every campaign pass still asserts the registered value at the
boundary where the profile acts, so §2 is preserved rather than weakened.

The predicate is **derived from the config**, never passed — a `campaign=`
argument would be a caller-supplied claim about evidential status, which is
F-19's currency tag wearing a keyword. A test asserts the *signature*, so
reintroducing a caller flag fails rather than being caught in review.

---

## 5. `train.rehearsal_frac = 0.65`

**Taken mechanically by the frozen rule.** No judgment was exercised, because
none was needed.

### The sweep, re-run with the confound fixed

The first sweep drew supervised and ring indices from **one generator**, so arms
differing in `f` also saw different ring sample sequences. Fixed; a residual
remains and is named in `SWEEP-m1a4.md` rather than removed, because the clean
fix would break the inertness proof that makes the rehearsal wiring a defect fix.

| `f` | steps | seed | top-1 | hits / n | ring-epochs | vh/step | band |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 0.00 | 400 | 0 | 0.8942 | 1099/1229 | 39.2 | 128 | no |
| 0.00 | 400 | 1 | 0.8934 | — | 39.2 | 128 | no |
| 0.10 | 400 | 0 | 0.8755 | — | 35.2 | 115 | no |
| 0.15 | 400 | 0 | 0.8755 | — | 33.4 | 109 | no |
| 0.25 | 400 | 0 | 0.9048 | — | 29.4 | 96 | no |
| 0.35 | 400 | 0 | 0.8934 | — | 25.4 | 83 | no |
| 0.50 | 400 | 0 | 0.9601 | 1180/1229 | 19.6 | 64 | no |
| 0.50 | 400 | 1 | 0.9691 | — | 19.6 | 64 | yes |
| **0.65** | 400 | 0,1,2 | **0.9699 ×3** | **1192/1229** | 13.8 | 45 | **YES** |
| 0.75 | 400 | 0,1 | 0.9699 ×2 | 1192/1229 | 9.8 | 32 | yes |
| 0.00 | 200 | 0 | 0.8942 | 1099/1229 | 19.6 | 128 | *mechanism* |

Control reproduced `ckpt-0`'s **0.8942** exactly, so the sweep is valid.

### The estimator, declared before the seeds existed

§2 named no estimator, and with three seeds that ambiguity would have decided the
answer. Declared at `9b1b342`, while the values did not yet exist: **the worst
seed at that `f` must hold the band** — derived from §8, because the campaign
runs a *single* seed against a hard BLOCKED, so an `f` that fails on some seeds
can halt the campaign at its first cadence unit.

It could not be a back-fit: `f = 0.50` fails on **worst (0.9601) and on mean
(0.9646)**; only best-case would have accepted it. The one estimator that would
have changed an already-observed disposition is the one not chosen.

**It turned out not to bite.** Spread at `f = 0.65` is **0.0000** across three
seeds, and at 0.75 across two.

### A limitation of the band, recorded because it is now visible

Five runs at two fractions all land on **1192/1229 — the anchor's own count.**
The band is coarse and saturated up there: 1192 hits against a band of ~1190, and
the metric is pinned at the anchor's value above `f = 0.65`. **It cannot
discriminate in that region.** So §3.1 still governs — **the band SCREENS, the
cadence DECIDES** — and attempt 2's cadence unit is the real verification.

### Does the selected `f` leave an experiment? Yes — with a stated limit

`‖θ_f − θ_anchor‖ / ‖θ_anchor‖`, relative to the control's 0.01268:

| arm | movement | relative |
|---|---:|---:|
| `f = 0.00` | 0.01268 | 100.0% |
| **`f = 0.65`** | 0.01196 / 0.01154 / 0.01178 | **94.3% / 91.0% / 92.9%** |
| `f = 0.75` | 0.01124 / 0.01127 | 88.6% / 88.8% |

The model moves **91–94% as far** as with no rehearsal at all, in directions that
do not cost supervised top-1. That is the encouraging reading.

**The limitation, stated rather than implied:** movement magnitude rules out the
failure mode — a model pinned in place, testing nothing — but it **does not
establish useful learning.** A model can move a long way in directions that
matter to nothing. Sufficiency comes from attempt 2's own numbers: self-play
beat-par, the funnel columns, and the floors.

## 6. `campaign.interop_threads` becomes EXERCISED (ruled 2026-08-19)

**Decided, built, and the reproduction gate re-ran.**

M1-A2 §4 classed interop as **OBSERVED** — *"licenses only 'this is what ran'"* —
because it sat constant at 32 across both hosts and was undiscriminated. F-32
added the check that the record's claim actually matched the runtime.

The campaign host then changed. The new host defaults interop to **48**, and
`assert_threads` **refused**:

> interop threads are 48, but the licence recorded 32 and every measurement on
> the record ran under that value. Unset is a value and OBSERVED is a claim: a
> host that differs stands outside this evidence, not inside it.

That is the assertion working. §4 also names the disposition, so no new rule was
needed:

> "Raising any of these is not a config tweak. It is a **new configuration whose
> reproduction gate re-runs** — the set-growth law, standing at the thread knob."

**Taken: apply the pin and re-run the gate.** Rejected alternatives, and why:

| option | rejected because |
|---|---|
| leave it OBSERVED and amend §4 to record 48 | widens the claim after the fact; every prior measurement ran at 32 |
| set it silently to 32 | converts an observation into a configuration with no gate — exactly what §4 forbids |

### What changed, precisely

**The field's value is unchanged at 32, so the config fingerprint does not move**
— `ce41af96…` is still the registered value, verified. What changed is whether
the runtime is *made to match* the record or merely *asked whether it happens
to*. The pin is applied and then read back, so the returned record is what the
runtime holds rather than what the caller hoped for.

**Inert on the old host**, where ambient interop was already 32 — confirmed
locally, where the same code returns the same record it always did.

A pin that cannot take now refuses with the reason: torch fixes the interop pool
at first parallel work, so a process that has already measured under the wrong
value cannot be corrected into the record.

---

## 7. Cadence cost tracks model quality — recorded, no gate contact

M1-A2 §2 prices the cadence at 109.5 min. The rehearsal's took **227 min**, and
the same factor appeared independently three times:

| measurement | factor |
|---|---:|
| rehearsal cadence, 227 min against 109.5 modelled | 2.07× |
| `ckpt-4`'s scripted-strata rate, 12.0 s/problem against the anchor's 5.844 | 2.05× |
| the anchor's equivalence pass against `ckpt-4`'s | ~2× |

**Cadence cost is a function of model quality**, because a degraded model caps
more episodes and a capped episode runs a search at every step to the cap. Three
measurements from three different runs agree to within a percent.

**No gate contact.** M1-A2 §2's table is descriptive and nothing is denominated
in it (F-27). It is recorded because a cost model that is wrong in a *known
direction* is more useful than one silently assumed right — and because a
campaign that slows down is now a readable signal rather than a puzzle.

---

## What this amendment does NOT settle

* **Whether M1 launches as specified.** §5 establishes that `f = 0.65` leaves the
  model moving; it does not establish that the movement is useful. That is
  attempt 2's cadence unit to answer.
* **`escalation-outruns-learner`** (F-27's watch item) stays open and is
  confounded by the value-head-silent mechanism; it is re-evaluated only after
  attempt 2.
* **The rung population** for M2 (F-34), which must be minted with headroom
  demonstrated *before* freezing.

## The set-growth ledger for this amendment

Fields whose **value** changes: `train.rehearsal_frac` 0.0 → **0.65**.
Fields whose **role** changes at unchanged value: `campaign.interop_threads`
(OBSERVED → EXERCISED), `ladder.problems_per_pass` (dead → per-file population
cap, inert at campaign scale), `league.snapshot_every` (ignored → honoured,
inert at 1).
Fields **deleted**: the eight unbacked keys in §4.

**The campaign fingerprint moves once, here, and PREREG-m1's recorded value must
be updated with it.**

---

## Erratum (2026-11-10, F-41)

§5's screen selected `rehearsal_frac = 0.65` from arms whose **ring** policy
targets were in the wrong coordinates:
- 93.25% of stored visit mass sat on illegal cells and was dropped by the loss;
- 4.17% trained the wrong legal action;
- 2.58% trained the right one (`FINDINGS.md` F-41).

The supervised rows were the only real policy signal in every arm. That fits top-1
being pinned at the anchor's 1192/1229 for `f ≥ 0.65`.

The selection rule was applied as registered, and its arithmetic stands. What it
selected among was not the treatment it was meant to compare. Re-run the screen
under correct targets before 0.65 is treated as tuned rather than inherited.
