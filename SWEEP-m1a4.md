# The rehearsal-fraction sweep — selection rule, frozen before the sweep runs

**Written 2026-08-17, before any arm has been run.** F-31 wired the rehearsal
path and proved it inert at `f = 0.0`. Choosing `f > 0` is the treatment
decision, and it lands as M1-A4 carrying this sweep's numbers.

The rule below is registered **first** for a reason specific to this metric.

---

## 1. Why the rule cannot be chosen after the numbers

**Top-1 is monotone in `f` by construction.** More supervised data per batch
means a better supervised metric. So scoring arms by top-1 alone drives `f`
toward 1: maximum preservation, zero learning, and a "winner" that has stopped
doing the experiment.

An objective that a knob trivially maximises is not an objective, and the moment
to notice that is before the table exists — the same reason `s*` was frozen
before its sweep rather than read off it.

## 2. The selection rule

> **The smallest `f` whose gate-10b top-1 holds the band.**

Smallest, not best: the sweep is buying the *minimum* supervision that protects
the warm start, because every supervised row is a self-play row not trained on,
and the campaign's subject is what self-play teaches.

## 3. The band, derived from §8 rather than chosen

§8 makes a no-regress breach BLOCKED, so the tolerable top-1 loss is whatever
keeps at-par above the floor — not a number anyone picks.

**Measured sensitivity**, from the rehearsal's own checkpoints:

| | anchor | `ckpt-4` | Δ |
|---|---:|---:|---:|
| gate-10b top-1 | 0.9699 | 0.8845 | −0.0854 |
| at-par @48 | 1193/1200 = 0.99417 | 910/1200 = 0.75833 | −0.23583 |

**2.7615** at-par points per top-1 point.

The floor's slack is `1193 → 1188` = **5 problems** = 0.42 points, so the
tolerated top-1 loss is `0.004167 / 2.7615` = **0.151 points**:

> ### BAND: gate-10b top-1 ≥ **0.968**

**Caveat, registered rather than buried:** this is a linear extrapolation across
a wide interval. An independent multiplicative model — at-par ≈ mean of `tᵏ` over
suite depths 1–3, whose local slope at `t = 0.9699` is 1.9206 — gives a band of
0.9677. The chord is the steeper slope and therefore the tighter band, and
**0.968 is the conservative of the two.**

The direction is unambiguous whichever model is used: **§8 permits essentially no
forgetting at all.**

## 3.1 The band SCREENS; the cadence DECIDES

Stated here so a passing arm cannot be over-read later. The sweep measures **one
training iteration on ring-0** — the right probe, because F-27 established the
damage is one-shot at iteration 0 — but the actual gate is `at-par ≥ 1188`
measured at **iteration 4**, which only rehearsal attempt 2 can produce.

So a passing arm is a **candidate**, and M1-A4's value stays **provisional** until
the rehearsal's cadence unit confirms it.

Conservative banding is right here for an asymmetric reason, not a general one:
§8 is a BLOCKED branch. A band too tight costs one rejected arm and perhaps a
second round; a band too loose costs a 110-minute cadence unit, a breached floor,
and a halted campaign.

## 4. Arms

`f ∈ {0.00, 0.10, 0.15, 0.25, 0.35, 0.50}`, each trained from the anchor on the
**same fixed ring-0**, same seeds — so `f` is the only variable — and scored on
gate 10b, top-1 depth ≤ 3, F-09 unseen subset.

`f = 0.00` is the control and must reproduce `ckpt-0`'s **0.8942**. If it does
not, the ring-0 replay is not the same measurement and the sweep is void.

### The table M1-A4 will carry

Three columns, so the amendment shows preservation, mechanism and cost together
rather than asserting the trade-off in prose:

| column | meaning |
|---|---|
| **gate-10b top-1** | preservation — does the warm start survive |
| **ring-epochs** | the mechanism variable, `(1 − f) × 39.2` at iteration 0 |
| **value-head examples/step** | the cost the head pays, `(1 − f) × 128` |

The third column is why this is a trade rather than a free win: Phase-1 rows
carry no `z`, so every point of `f` is a point off the head whose criterion
accrual is already the binding constraint.

## 5. Both branches, registered now

**Some `f` clears the band** → that `f` is M1-A4's value, with the sweep table as
its evidence.

**No `f` clears the band** → **a finding, not a failed round.** It says the
frozen page admits no viable training configuration at 400 steps per iteration,
and lever 2 — epoch scaling, steps proportional to ring size — becomes the *next*
round's single lever. Never a simultaneous change: two levers at once is a
confound, and the whole point of registering this branch is that discovering it
must not become a licence to turn both.

## 6. A consequence to carry onto M1-A4's page

At `f > 0` the value head sees `(1 − f)` of its examples per step: Phase-1 rows
carry no `z` by construction, so they contribute policy and steps loss only.
That slows precisely the head whose criterion accrual is already the binding
constraint — the rehearsal's switch criterion abstained at every iteration with
the rarest class at 1, 6, 7, 12, 17 against a floor of 100.

Not a blocker, and not an argument against rehearsal. It is a cost that belongs
on the amendment's page rather than in a footnote discovered later.

---

**Three failed attempts at any gate → BLOCKED, never a weakened gate.**

---

# Registered prediction — the watchlist's first reading

**Written 2026-08-17, before the equivalence gate runs.**

Fix 2 gives the `ckpt-4` equivalence gate a second job it was not built for: it
is also the **first watchlist measurement**, and it retroactively attributes the
rehearsal's misses — the exact question PREREG §5 was written for and could not
answer when it mattered (F-35).

At `sims = 48`, `ckpt-4` missed **290** of 1200 (at-par 910). The frozen family
is 24, certified at `sims = 1`.

| quantity | registered call |
|---|---|
| `family_remaining` | **24** — the anchor's entire hard family still missed by a degraded model |
| `novel_misses` | **≈ 266** — the complement |

**The informative branch is the low one.** `family_remaining < 24` would mean the
degraded model *solved* some of the anchor's hardest problems while losing 283
easier ones — genuinely strange, and worth its own look rather than a shrug.

Registered because a prediction written after the number arrives is a
description, and because this is the first reading of an instrument that has
never produced one.

---

# Amendment — the sweep re-runs, extended, 2026-08-19

**The first sweep's result stands as reported and is superseded as evidence.** It
returned no arm holding the band, with the control reproducing `ckpt-0`'s 0.8942
exactly. Three things change; **the rule and the band do not.**

## 1. The confound the first sweep carried

Supervised indices were drawn from the **same generator** as ring indices, so
every extra supervised draw shifted the ring stream: two arms differing in `f`
also saw different *ring sample sequences*. `f` changed the mixture and the data
at once, and the low-`f` dip — 0.8755 at `f = 0.10`, **below** the `f = 0`
control — could not be told from sampling noise.

Fixed: supervised draws take their own generator, seeded off `seed`, constructed
only when `n_sup > 0` so `f = 0.0` stays bit-identically inert.

**A residual remains, and it is named rather than fixed.** Arms with different
`f` still draw different *counts* of ring indices per step from one stream, so
their sample sequences diverge. The clean fix — per-step seeding, making each
arm's draws a prefix of the control's — would change `f = 0.0`'s behaviour and
**break the inertness proof that makes the wiring a defect fix rather than a
treatment change**. The governance property is worth more than the sharper
comparison, so the residual stays and the variance arms below are what bound it.

## 2. Variance, because the finding rests on 0.0062

The whole result is a 0.0062 shortfall against an **unknown noise floor**. A
second seed on the control and on the best arm bounds it: if seed spread is
±0.005 the first sweep established nothing.

## 3. Arms extended to 0.65 and 0.75

*"The `f` that would clear the band is the `f` at which training barely moves the
model"* was an **argument, not a measurement**. At `f = 0.75` the loop still
draws 12,800 self-play samples over 1,305 fresh rows — **9.8 ring-epochs**, more
than iteration 3 had when the damage was small.

**The criterion cannot move.** Rule and band are frozen above and selection is
mechanical, so extra arms can only find an answer.

## 4. The mechanism arm — `f = 0.00` at 200 steps

A **diagnostic, not a treatment**, so it is no one-lever violation, and it is
excluded from candidacy by construction (selection reads only `steps == 400`,
`seed == 0`).

The first sweep's table shows top-1 tracking **ring-epochs**, not obviously the
mixture:

| ring-epochs | 33.4 | 29.4 | 25.4 | 19.6 |
|---|---|---|---|---|
| top-1 | 0.8747 | 0.8820 | 0.9479 | 0.9618 |

`f = 0.00` at 200 steps gives **19.6 ring-epochs — identical to `f = 0.50`** —
with no supervision at all. A matched pair differing in exactly one thing:

* lands near **0.96** → epoch scaling is the mechanism, rehearsal is redundant,
  and **lever 2 is round two's lever**;
* lands near **0.89** → supervised anchoring is doing the work, and **lever 1
  is**.

One arm, ten minutes, and round two's lever is chosen by measurement rather than
by anyone's intuition.

---

# Amendment — the estimator, declared before the seeds exist, 2026-08-21

**Written while the 0.65 and 0.75 seeds are running and their values do not yet
exist.** The verification job was launched at `b263613`; this page is amended
before its output is read.

## The gap the frozen rule left

§2 says *"the smallest `f` whose gate-10b top-1 holds the band"* and **never named
an estimator.** With one seed per arm that ambiguity was invisible. With three
seeds at `f = 0.65` it decides the answer — worst, mean and best can disagree,
and choosing among them after seeing them is choosing the answer.

## Declared: the WORST seed at that `f` must hold the band

**Derived from §8, not from the data.** The campaign runs a **single seed**, and
§8's no-regress breach is a hard **BLOCKED**. So an `f` that holds on some seeds
and fails on others is an `f` that can halt the campaign at its first cadence
unit. **Worst-case is the decision-relevant statistic precisely when the
downstream consequence is a halt** — the same asymmetry that set the band
conservatively in §3.1.

### It cannot be a back-fit, and here is the arithmetic

`f = 0.50` is the only multi-seed arm already observed: 0.9601 and 0.9691.

| estimator | value | holds the 0.968 band? |
|---|---|---|
| **worst** | 0.9601 | **no** |
| mean | 0.9646 | no |
| best | 0.9691 | yes |

Worst-case and mean both **reject** 0.50; only best-case would have accepted it.
So declaring worst-case **cannot rescue an arm already observed to fail**, and
changes no disposition the table already contains. The one estimator that would
have changed an answer is the one not chosen.

## Branches, pre-stated

**The steps knob**

* **digests identical** → the mechanism arm is **void**, the epoch falsification
  is **withdrawn**, and it is a **live-config gate miss in its own right**: the
  census marked the steps field live on evidence from a call site other than the
  sweep's, which is a finding worth its own line.
* **digests differ** → the falsification **stands**, epoch scaling drops as a
  lever, and **rehearsal fraction is the lever**.

**The selected `f`**

* `f = 0.65`'s **worst** seed holds → the rule takes **0.65**, mechanically.
* else `f = 0.75`'s worst seed holds → **0.75**.
* neither → **§5's second branch fires after all**, this time on an adequate
  estimate rather than an extrapolation.

**Movement**

Does **not** change the selected `f`; selection stays mechanical. The number goes
on M1-A4's page regardless. If movement at the selected `f` is near zero, that
may change **whether M1 launches as specified** — a separate ruling, made on the
number, **never folded into the band**.

---

## Comparands

- `runs/m1_final/sweep_rehearsal_frac.json` — the arms
- `runs/m1_final/sweep_verify.json` — the verification
- `runs/gate10b_trajectory.json` — the sensitivity table in §3
- `runs/phase1/phase1_result.json` — the anchor's frozen 0.9699
- `runs/pod_artifacts/m1_rehearsal/ckpt-4.pt` — the `ckpt-4` of §3's Δ row
- **GONE:** attempt 1's `ckpt-0` — §4's control target

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
