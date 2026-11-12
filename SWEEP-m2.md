# The rehearsal-fraction sweep, second reading — registered before any arm runs

**Written 2026-11-12, before any arm has been run.** DRAFT until the principal
approves it; the approval is recorded at the foot of this page, and nothing
below is edited after it. Amendments are appended, dated.

**Commissioned** by the principal, verbatim, on the proposal *"keep the same
rule and band, have the control arm declared as a fresh measurement instead of
a reproduction target, run both targets, and add the LR-controlled 200/400
pair"*: *"Agree with your suggestion and yes, run on a pod"*.

---

## 1. Why this sweep is run again

`SWEEP-m1a4.md` chose `rehearsal_frac = 0.65`, and F-37 concluded that
forgetting is immediate. Both were measured on a ring whose policy targets were
stored in the wrong coordinates: 93.25% of the visit mass sat on illegal cells
and the loss dropped it (F-53). Since then the target itself has changed (F-54).

So M1's sweep numbers stand as readings of that defect. What they say about
training on a correct ring is unknown, and the next campaign's
`rehearsal_frac` rests on them. This page re-measures, once per policy target.

M1's sweep scripts and `SWEEP-m1a4.md` are not edited. This sweep has its own
script, `scripts/sweep_m2.py`.

## 2. The selection rule (unchanged)

> **The smallest `f` whose gate-10b top-1 holds the band.**

Applied **separately to each policy target**. Top-1 is monotone in `f` by
construction, so the rule is fixed before the table exists, for the reason
`SWEEP-m1a4.md` §1 gives.

## 3. The band (unchanged, restated as a count)

`SWEEP-m1a4.md` §3: gate-10b top-1 ≥ 0.968. The instrument is 1,229 rows
(depth ≤ 3, F-09 unseen subset), so the band is

> ### BAND: hits ≥ **1190 / 1229**

(1189 / 1229 = 0.9675 does not hold; 1190 / 1229 = 0.9683 does.)

**Caveat, registered rather than buried.** The band's slope (2.7615 at-par
points per top-1 point) was fitted to M1's rehearsal checkpoints, which trained
under F-53. It is kept because the principal ruled "same band" and because it
is the conservative of the two models on that page. It still only SCREENS: a
passing arm is a candidate, and the campaign's own cadence unit decides.

### The four-tuple (AGENTS §5, rider c)

| | value |
|---|---|
| floor | 0 / 1229 |
| null | the `f = 0.00` arm of the same target: training on the ring alone |
| threshold | 1190 / 1229 |
| measured | this sweep; the untrained anchor is 1192 / 1229 on the local box |

The band sits 2 hits under the anchor. An arm holds it only by forgetting
almost nothing.

## 4. Arms

Thirteen arms per policy target, **twenty-six in all**. Every arm trains from
the Phase-1 anchor on the same fixed ring-0 and is scored on gate 10b.

| # | `f` | steps | warmup | seed | kind |
|---:|---:|---:|---:|---:|---|
| 0 | 0.00 | 400 | 200 | 0 | candidate (the control) |
| 1 | 0.10 | 400 | 200 | 0 | candidate |
| 2 | 0.15 | 400 | 200 | 0 | candidate |
| 3 | 0.25 | 400 | 200 | 0 | candidate |
| 4 | 0.35 | 400 | 200 | 0 | candidate |
| 5 | 0.50 | 400 | 200 | 0 | candidate |
| 6 | 0.65 | 400 | 200 | 0 | candidate (M1-A4's value) |
| 7 | 0.75 | 400 | 200 | 0 | candidate |
| 8 | 0.00 | 400 | 200 | 1 | variance |
| 9 | 0.65 | 400 | 200 | 1 | variance |
| 10 | 0.65 | 400 | 200 | 2 | variance |
| 11 | 0.00 | 200 | 200 | 0 | mechanism: F-37's arm |
| 12 | 0.00 | 200 | 100 | 0 | mechanism: LR-controlled |

- **Only candidates can be selected.** Variance and mechanism arms are
  diagnostics. They can find an answer and cannot move the criterion.
- **The control is a fresh measurement.** M1's sweep required `f = 0.00` to
  reproduce 0.8942. That number is a property of F-53's defect and is not a
  target here. Nothing voids this sweep by its control's value.
- **The anchor is scored on the host that runs the arms**, and every delta is
  taken against that reading.
- **The ring** is `runs/ring0_replay/ring-0`, regenerated 2026-11-11 in ring
  format 3 (F-55: 398 / 400 solved, 99 pool-par, 1,305 rows, reproduced). It
  holds both targets, so both sweeps train on identical rows.
- **`seed`** is `train_on_ring`'s seed, as in M1's sweep. It is not
  `cfg.seed`, and no campaign seed is registered by this page.

### What the two mechanism arms separate

The schedule is 200 warmup steps, then cosine decay. So:

- **Arm 11** (200 steps, warmup 200) is exactly the first 200 steps of the
  control. It never leaves warmup and its summed learning rate is half the
  control's. F-37 read "identical hits from half the training" off this arm.
- **Arm 12** (200 steps, warmup 100) runs the control's whole schedule shape,
  peak included, in half the steps.

If forgetting is immediate, arms 0, 11 and 12 agree. If it follows the learning
rate actually applied, arm 12 sits with arm 0 and arm 11 sits above both. If it
follows the step count, arms 11 and 12 agree and differ from arm 0.

## 5. Both branches, registered now

Per target:

- **Some candidate holds the band** → the smallest such `f` is that target's
  candidate value for the next campaign's PREREG.
- **No candidate holds the band** → a finding, not a failed round. As in
  `SWEEP-m1a4.md` §5, the next round changes one lever, never two.

Across targets:

- **The next campaign trains `'improved'`** (F-54), so the `'improved'` table
  supplies its `rehearsal_frac`. The `'visits'` table is the control reading:
  it isolates the F-53 fix on M1's own target.
- **If the two targets select different `f`,** both are reported and the
  PREREG states which it uses and why. No value is averaged or carried across.

## 6. Reading the variance arms

A candidate within the seed spread of the band has not been shown to hold it.
If the selected `f` is 0.65, its three seeds are reported beside it. If it is
another value, that value's seeds are a follow-up arm set, registered by an
amendment here before they run, and the selection stays provisional until then.

## 7. Registered predictions

Written before any arm runs, because a prediction written afterwards is a
description.

| # | prediction | basis |
|---|---|---|
| P1 | The `'visits'` control does **not** read 1099 / 1229 (M1's 0.8942). | M1's control trained on about 7% of its target mass. |
| P2 | The `'improved'` control forgets **less** than the `'visits'` control: more hits at `f = 0.00`. | Ring-0 was filled with the value head silent, so π′ equals the prior wherever the search held no proof (F-54), and those rows give about zero policy gradient. |
| P3 | Under `'improved'`, the selected `f` is **at most** the one selected under `'visits'`. | Follows from P2 if the band is reached at all. |

No direction is predicted for P1. The informative outcomes are P2 failing, or
arm 11 and arm 12 disagreeing.

## 8. What each arm records

`runs/sweep_m2.json` (tracked) carries, per arm: hits and n, band verdict,
ring-epochs, value-head examples per step, steps taken, rehearsal batches,
encode / NaN skips, `improved_truncated`, first loss and the mean of the last
twenty, weight digest, movement relative to the anchor, seconds, and
provenance (git SHA, clean-tree flag, torch build, threads, anchor digest, ring
digest, config fingerprint).

A sweep with a missing arm makes no selection.

## 9. Where it runs

A rented CPU pod, torch `+cpu` build (`make install ACCEL=cpu`), 8 intra-op
threads per arm as in M1, arms side by side in separate processes. One host
runs all twenty-six arms and the anchor. Results are pulled back to the box and
the pod is deleted.

- Weight digests are comparable within this sweep. They are not comparable
  with digests taken on the local box, whose torch is a different build (F-54,
  box-side amendment).
- Ring-0 was generated on the local box and is copied to the pod byte for
  byte; its digest is in every row.

---

**Three failed attempts at any gate → BLOCKED, never a weakened gate.**

---

## Approval (2026-11-12)

**Approved by the principal, verbatim:** *"agree, proceed"*, on the page as
committed at `12e45ce` and the pod plan offered with it (CPU `cpu5c`, 32 vCPU,
ceiling $1.25/h, 3 h hard cap, worst case $3.36). From here the page above is
frozen; changes are appended below as dated amendments.

---

## Amendment (2026-11-12): attempt 1 aborted, attempt 2 complete — the readings

### Attempt 1, aborted on a slow host

Pod `8uhbld1tnafe7d` (cpu5c, AMD EPYC 9575F), launched as approved. Setup
passed: data matched ANCHORS, tree clean at `c01f549`, the anchor scored
1192 / 1229. The host then ran plain Python about 100× slower than the box (a
5,000,000-step loop: 24.6 s against 0.24 s; the 73 sanity tests: 145 s against
2.8 s), with no cgroup throttling recorded. At that speed the 26 arms could not
finish inside the 3 h cap, and an unfinished arm writes nothing, so it was
terminated after about an hour (≈ $1.13). No arm result came from it.

The principal approved a relaunch with a 4 h cap (worst case $4.48) and a speed
gate: the same loop is run after boot, and a host more than 2× slower than the
box is terminated. **Verbatim:** *"relaunch"*.

### Attempt 2

Pod `tt1aff65bef9tu`, same flavour and CPU model. Speed gate 0.24 s before
setup, 0.23 s after; sanity tests 2.8 s. Code `c01f549`, clean tree, torch
2.13.0+cpu, 8 threads per arm, 4 arms side by side. Started 09:25 UTC, all 26
arms done 10:19 UTC, pulled 10:24 UTC; job exit 0, no arm failed.

**Integrity, per arm:** steps taken = steps requested; rehearsal batches = steps
where `f > 0` and 0 where `f = 0`; encode skips, NaN skips and
`improved_truncated` all 0; every arm's ring digest is `4ff19a7d…`, the box's
ring-0; all 26 weight digests distinct. The anchor scored **1192 / 1229** on the
pod. The record is `runs/sweep_m2.json`.

| # | `f` | steps | warm | seed | kind | `'visits'` hits | `'improved'` hits |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.00 | 400 | 200 | 0 | candidate | **1000** | **1192** |
| 1 | 0.10 | 400 | 200 | 0 | candidate | 1186 | 1192 |
| 2 | 0.15 | 400 | 200 | 0 | candidate | **1192** | 1192 |
| 3 | 0.25 | 400 | 200 | 0 | candidate | 1192 | 1192 |
| 4 | 0.35 | 400 | 200 | 0 | candidate | 1192 | 1192 |
| 5 | 0.50 | 400 | 200 | 0 | candidate | 1192 | 1192 |
| 6 | 0.65 | 400 | 200 | 0 | candidate | 1192 | 1192 |
| 7 | 0.75 | 400 | 200 | 0 | candidate | 1192 | 1192 |
| 8 | 0.00 | 400 | 200 | 1 | variance | 999 | 1192 |
| 9 | 0.65 | 400 | 200 | 1 | variance | 1192 | 1192 |
| 10 | 0.65 | 400 | 200 | 2 | variance | 1192 | 1192 |
| 11 | 0.00 | 200 | 200 | 0 | mechanism | 1128 | 1192 |
| 12 | 0.00 | 200 | 100 | 0 | mechanism | 1054 | 1192 |

Band: hits ≥ 1190. Weights moved in every arm by 1.0–1.3% of the anchor's norm
under both targets.

### Selection, by the frozen rule (§2, §5)

- **`'visits'`: 0.15.** Arm 1 (`f = 0.10`) misses by 4 hits. The variance
  arms at 0.65 read 1192 on all three seeds; no seed was run at 0.15, so per §6
  the selection is **provisional until 0.15's seeds are run**.
- **`'improved'`: 0.00.** Every arm, seed and mechanism variant reads the
  anchor's 1192 exactly.

### The predictions (§7)

| # | prediction | result |
|---|---|---|
| P1 | `'visits'` control does not read M1's 1099 | **holds**: 1000 (seed 1: 999). It forgets more than M1's control did. |
| P2 | `'improved'` control forgets less than `'visits'` | **holds**: 1192 against 1000 |
| P3 | `'improved'` selects `f` ≤ `'visits'` | **holds**: 0.00 ≤ 0.15 |

### What the mechanism arms say (`'visits'`)

| arm | what it is | hits |
|---|---|---:|
| 0 | 400 steps, full schedule | 1000 |
| 12 | 200 steps, full schedule shape (warmup 100) | 1054 |
| 11 | 200 steps, warmup only (F-37's arm) | 1128 |

The three disagree by far more than the measured seed spread (1 hit at
`f = 0.00`). Forgetting is **not immediate**: it grows with the steps taken at
a fixed schedule shape (12 → 0), and with the learning rate applied at a fixed
step count (11 → 12). §4's three readings named one of these as the outcome;
the measurement shows both terms acting.

### Why `'improved'` does not forget here, measured on the box

Ring-0 was filled by the anchor itself with the value head silent. Each stored
target, scored against the anchor's own policy on the same states (all 1,305
rows):

| target | KL(target ‖ anchor), median | mean | rows with KL > 0.01 | argmax = anchor's |
|---|---:|---:|---:|---:|
| π′ (`'improved'`) | 0.000000 | 0.210 | 57 | 1277 / 1305 |
| visit share (`'visits'`) | 2.31 | 3.96 | 987 | 1168 / 1305 |

So on ring-0, π′ is the anchor's own policy on all but 57 rows: training toward
it is self-distillation, and the policy head is taught what it already says.
That is F-54's formula doing what it said (π′ departs from the prior only where
the search holds a proof), and it is why every `'improved'` arm reads 1192.

**What this licenses, and what it does not.**
- At iteration 0 under `'improved'`, the warm start does not need rehearsal to
  survive.
- It says nothing about later iterations. Once the value head goes live, or
  the network has moved off the anchor, π′ will depart from the policy that
  generated the ring, and the forgetting `'visits'` shows can return. This
  sweep cannot measure that: its ring is fixed at iteration 0 by design.
- So the band does not discriminate between `f` values under `'improved'` on
  this ring. Reading 0.00 as "the campaign needs no rehearsal" would be a
  reading the instrument cannot support.
