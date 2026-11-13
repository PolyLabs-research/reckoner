# The rehearsal sweep under 'improved', on a ring the network disagrees with — registered before any arm runs

**Written 2026-11-12, before any arm has been run.** DRAFT until the principal
approves it; the approval is recorded at the foot of this page, and nothing
above it is edited afterwards. Amendments are appended, dated.

**Commissioned** by the principal, verbatim: *"Start the next steps"*, on the
proposal *"An `improved` reading on a ring the network disagrees with. This is
what actually decides the campaign's `rehearsal_frac`. It needs a ring from a
later iteration, or one filled with the value head live, so it is a new
registered page."*

---

## 1. Why SWEEP-m2's 'improved' table cannot be used

Every `'improved'` arm of SWEEP-m2 read the anchor's 1192, at every `f`, seed
and schedule (F-56). On ring-0 the π′ target equals the anchor's own policy on
all but 57 of 1,305 rows, because the ring was filled with the value head
silent and the search held a proof almost nowhere. Training toward it is
self-distillation, so the band cannot separate one `f` from another.

The next campaign trains `'improved'`. Its `rehearsal_frac` has to be read on a
ring where π′ says something the network does not already say. Two such rings
were offered: a later iteration's, or one filled with the value head live.

## 2. The ring: ring-V, iteration 0 searched with the value head trusted

`scripts/ring0_replay.py --value-scale 1.0 --out runs/ring_v`: iteration 0
exactly as ring-0 was made, except that the self-play search trusts the
anchor's value head at contribution 1.0 instead of 0.0.

- **The same episodes start.** The 400 problems and their pars come from
  `campaign_problems(…, seed=0)`, and pool par is drawn by the pool member's
  own evaluator, not the self-play one. Only the self-play search differs.
- **Why this ring and not a later iteration's.** A later iteration under
  `'improved'` with the head silent is filled by a network that, by F-56, is
  still the anchor in all but its value and steps heads. Its π′ would again be
  that network's own policy. The value switch is what moves π′ off the
  network, and M1's switch abstained at every iteration, so a short campaign
  would not reach that regime either.
- **What it stands for.** A ring filled after the value switch has fired. It is
  a harder version of that regime: the anchor's value head has passed no
  criterion, so its q is less accurate than a head the switch admitted would
  be. A reading here is a stress test of rehearsal, not a forecast of the
  campaign.

### Measured on ring-V before registration (inputs, not results)

Generated on the box 2026-11-12 at `d5856bb` plus the uncommitted
`--value-scale` option this page commits; torch 2.13.0+rocm7.2 on CPU. Ring
digest `2374f85b…` (`sweep_m2.ring_digest`), ring format 3.

| | ring-0 | ring-V |
|---|---:|---:|
| episodes solved | 398 / 400 | **287 / 400** |
| pool-par episodes | 99 | 99 (the same draws) |
| ring rows | 1,305 | **3,601** |
| rows with z = −1 / 0 / +1 | 60 / 1,240 / 5 | **2,736** / 860 / 5 |
| π′ targets truncated (`improved_truncated`) | 0 | 0 |
| π′: rows with KL(π′ ‖ anchor) > 0.01 | 57 | **1,125** |
| π′: best action = anchor's | 1,277 / 1,305 | **2,613 / 3,601** |
| π′: KL(π′ ‖ anchor), median / mean / p90 | 0.000000 / 0.210 / 0.001 | 0.000002 / 5.27 / 22.3 |
| first-step loss, `f = 0`, policy target withheld (2-step smoke) | — | ≈ 124 |
| first-step loss, `f = 0.65` (SWEEP-m2 smoke) / `f = 0` here | ≈ 1.0 | ≈ 130 |

So π′ teaches a best action different from the anchor's on 988 rows (27%),
against 28 on ring-0. That is the disagreement this sweep needs.

**It is not the only difference, and the page says so before the numbers.**
With its value head trusted, the anchor's search is misled: it solves 287
episodes, the rest run long or cap, and the ring is 2.76× larger and 76% z = −1.
So ring-V also differs from ring-0 in

- **ring-epochs:** 400 steps at `f = 0` are 14.2 epochs here, 39.2 on ring-0;
- **the value head's labels:** mostly −1, against mostly 0;
- **the states themselves:** those a misled search reaches.

Comparisons **within** this sweep (across `f`, and candidate against control)
change one thing. Comparisons **across** the two sweeps change all of the above
at once and are reported as descriptions, never as effects.

## 3. Rule and band (SWEEP-m2's, with §6 built in)

> **The smallest `f` at which every one of three seeds holds the band.**

The band is SWEEP-m2 §3's: **hits ≥ 1190 / 1229** on gate 10b. SWEEP-m2 §6
required seeds at the selected value as a follow-up round. Here every
candidate `f` runs at three seeds in the same round, so the selection needs no
follow-up. This is stricter than SWEEP-m2's seed-0 rule, never looser.

The four-tuple: floor 0 / 1229; null, the `f = 0.00` arms; threshold 1190;
measured, this sweep (the anchor reads 1192 / 1229).

## 4. Arms

Twenty-eight arms, all under `'improved'`, all trained from the Phase-1 anchor on
ring-V, scored on gate 10b, by `scripts/sweep_m3.py`. The arm machinery,
scorer and provenance are SWEEP-m2's (`sweep_m2.run_arm`).

| arms | `f` | steps | warmup | seeds | kind |
|---|---|---:|---:|---|---|
| 0–23 | 0.00, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75 | 400 | 200 | 0, 1, 2 | candidate |
| 24 | 0.00 | 200 | 200 | 0 | mechanism: warmup only |
| 25 | 0.00 | 200 | 100 | 0 | mechanism: full schedule shape |
| 26–27 | 0.00 | 400 | 200 | 0, 1 | control: policy target withheld |

**The control arms** set `train.policy_loss_weight = 0.0` and change nothing
else, so only the value and steps heads train, on ring-V's labels, through the
shared trunk. Their hits against the `f = 0.00` candidates at the same seeds
separate the forgetting π′ causes from the forgetting the label mix causes. The
first-step losses above say that second term is not negligible here.
`'visits'` is not run on ring-V. The decision is for the `'improved'` campaign,
and SWEEP-m2 already read `'visits'` on a correct ring.

## 5. Both branches, registered now

- **Some `f` holds on all three seeds** → the smallest such `f` is the
  `'improved'` campaign's candidate `rehearsal_frac` **for the regime after the
  value switch**. Before the switch, F-56 already shows no `f` is needed to
  hold the band at iteration 0.
- **No `f` holds** → a finding. It would say rehearsal at these fractions does
  not protect the warm start once π′ departs this far, and the next round
  changes one lever, never two.

**What this sweep cannot decide, stated now:** whether the campaign should
change `rehearsal_frac` when the switch fires. That is a second lever
conditioned on the switch, and it is the principal's decision once both
readings exist.

## 6. Registered predictions

Written before any arm runs.

| # | prediction | basis |
|---|---|---|
| P1 | At `f = 0.00` no seed holds the band. | π′ teaches a best action different from the anchor's on 988 of 3,601 rows. |
| P2 | The policy-withheld control keeps more hits than the `f = 0.00` candidate, seed for seed. | It is that candidate minus the one input that departs from the anchor. |
| P3 | The selected `f` is at least 0.15 and at most 0.65. | 0.15 is the smallest `f` that held under `'visits'` on ring-0, where the target named a different best action on 137 of 1,305 rows (10%), against 27% here. 0.65 is M1-A4's value. |
| P4 | If `f = 0.00` forgets, the mechanism arms order as under `'visits'`: the 400-step control lowest, arm 25 next, arm 24 highest. | F-56. |

The informative failures are P2 (the label mix, not π′, would be doing the
damage) and P3's lower bound.

## 7. What each arm records, and where it runs

As SWEEP-m2 §8. The record is `runs/sweep_m3.json` (tracked). A missing arm
means no selection.

It runs on a rented CPU pod with the same image, torch build (`+cpu`) and 8
threads per arm as SWEEP-m2, four arms at a time, after the speed gate (a
plain-Python loop within 2× of the box, or the pod is terminated). The same pod
runs SWEEP-m2 §6's three follow-up arms first, by `cloud/run_round_m3.sh`. Ring-V is generated on the box and
copied to the pod byte for byte; its digest is in every row.

---

**Three failed attempts at any gate → BLOCKED, never a weakened gate.**

---

## Approval (2026-11-12)

**Approved by the principal**, choosing *"Approve and launch (Recommended)"* on
the question: approve this page as committed at `2396a60` and launch one CPU
pod running SWEEP-m2 §6's three follow-up arms and this sweep's 28 arms (cpu5c,
32 vCPU, $1.12/h with a $1.25/h ceiling, 3 h hard cap, worst case $3.36, speed
gate at boot). From here the page above is frozen; changes are appended below
as dated amendments.

---

## Amendment (2026-11-13): attempt 1 ran and its results were lost before they were pulled

Pod `8shb0r5satb6by` passed the speed gate (0.23 s), set up cleanly at
`d6fc2f7` (73 sanity tests in 2.7 s), and started the round at 20:22:48 UTC.
Ninety seconds later both anchors had scored and the first four arms were
running.

**No arm result reached the box.** The local puller that copies results home
did not start until 23:54:58 UTC, three and a half hours after the round, in a
separate step that waited that long to run. By then the pod had deleted
itself: either the job guard's bounded wait for a pull (1,800 s after the job
ends) or the deadline watchdog at 23:20 UTC. The pod had no volume, so its
results went with it. The puller ran for two hours and found nothing.

**Not a reading.** No arm of this sweep has a recorded result, so nothing
above is scored by this attempt.

**What changes for attempt 2, as process, not as protocol:**
- the puller starts in the same step that starts the job, so the two cannot be
  separated by a delay on the box;
- the job guard waits for a pull until the pod's deadline instead of 1,800 s,
  so a late puller still finds the results. The cost of a pod waiting idle is
  bounded by the same deadline the principal approved.

The arms, rule, band and predictions are unchanged.
