# RUNLOG — M1, 2026-08-21/22

**The campaign ran to completion: 20 iterations, 9.6 hours, alarms 0, all four
cadence units holding both floors, and the primary CI-separated.**

**It also ran without an interlock in front of it, and that is on the face of
this log rather than in a footnote.**

---

## 1. The deviation

M1's DONE-WHEN placed a five-iteration dress rehearsal before the campaign. The
rehearsal was **never a separate program**: it is the campaign with an external
stop at `LATEST = 4`, built as a watchdog rather than an `iterations = 5` config
precisely because `iterations` is fingerprinted — a config rehearsal would have
run *at a different fingerprint from the thing it was rehearsing*.

**The stop was never launched.** The checkpoint watcher was; its sibling was not.

### What that cost, and what it did not

A gate does two jobs. It **verifies** — scores registered expectations against
fresh data — and it **interlocks** — holds a refusal in front of the next thing.

The omission destroyed the interlock and left the verification **completely
intact**. Iterations 0–4 on the volume *are* the rehearsal: same program, same
seed, same fingerprint (`8443847bb8c41218…`), same four-step commit order, scored
against `REHEARSAL-m1-attempt2.md` as committed at `f48fe1f` **before the run**.

**What the interlock would have refused:** nothing. Every registered item passed.
**And it never had the chance to.** No re-run can restore an interlock in front of
a run that has already happened.

Cost: six hours and roughly $6. Recorded as **F-38** — the watchdog whose only
output is a kill is indistinguishable from a healthy run, and the liveness-marker
requirement written for one supervisor was never extended to its sibling.

---

## 2. The rehearsal, scored in full against `f48fe1f`

| expectation | result |
|---|---|
| ring-0 reproduces | **1,305 rows, 398/400 solved, 99 pool-par** ✓ |
| `ckpt-0` digest == the sweep's `f = 0.65` seed-0 arm | **`5870537e58fcc609`** ✓ |
| top-1 at every checkpoint ≥ 0.968 | **0.9699 (1192/1229) ×5** ✓ |
| floors at iteration 4 | **1193 ≥ 1188**, **1176 ≥ 1167** ✓ |
| caps through iteration 4 | **2 / 0 / 0 / 0 / 0** ✓ |
| per-stratum at iteration 4 | 28 / 84 / 70 = **182** ✓ |
| alarms | **0** ✓ |

The digest match is the load-bearing one: the sweep harness that **selected** the
treatment and the driver that **applies** it produce bit-identical weights from
the anchor on ring-0. They had never been compared before.

**Counterfactual arm.** Attempt 1 ran the same driver, same instruments, same five
iterations, at the config differing in exactly the treatment — top-1 0.8942 →
0.8845, caps 2/3/3/**43**, floors 910 and 722. This is a two-arm comparison across
a deliberate configuration change, not a floor held in isolation.

---

## 3. The campaign

| iteration | @48 (floor 1188) | @1 (floor 1167) | primary |
|---:|---:|---:|---:|
| 4 | **1193** | **1176** | 182 / 600 |
| 9 | 1193 | 1176 | 199 / 600 |
| 14 | 1193 | 1176 | 199 / 600 |
| 19 | 1193 | **1175** | **209 / 600** |

Caps: 2 at iteration 0, **zero thereafter**. Cadence units ~79 min against attempt
1's 227 — F-27's 2.07× factor shrinking toward 1, as predicted, which is the
scored form of cadence-cost-as-model-quality.

### P1 — the test of record, runnable for the first time since the freeze

```
paired-difference bootstrap, 600 pairs, 10,000 resamples
anchor 101/600   campaign 209/600
mean difference 0.18   CI [0.1367, 0.2217]
excludes_zero: True    saturated: False
```

**CI-separated improvement.** It exists only because F-30 was found and Part-0d
re-run to capture the baseline arm per problem.

### The negative arm

`scripted_in_7`: **43 → 35**, mean **−0.0400**, CI **[−0.0900, +0.0050]** — a
decline, itself not separated. The pooled +0.18 is carried by strata 8 and 10.
**F-39** decomposes it: 149 shorter paths against 50 longer, zero anchor-only, six
campaign-only. The gain is **faster at par**, not materially more capable — and
stratum 7, whose true par is 5–6, is where beating par needs a genuinely shorter
proof rather than a tighter walk.

### The one novel miss

`@48` closes at **7 misses, 0 novel** — the anchor's exact miss set. `@1` closes at
**25: 24 family + 1 novel**, reconciling to the 1175 above. **One new failure in
twenty iterations**, visible in two independent instruments.

---

## 4. A withdrawn sentence

I wrote that the caps result held "against a harder par than attempt 1 faced". The
between-run comparison is **not testable**: attempt 1's ring died with the pod, and
its surviving rows carry `pool_composition` (membership), `steps_minus_par_histogram`
and `z_by_par_source` — **none of which yields mean par**.

What *is* measured, within attempt 2: pool par 4.036 → 3.528 with the gap to
BFS-exact closing from +0.260 to +0.029, caps at zero throughout. Par hardened
while the model stopped capping. And the mechanism runs the right way — pool par
is drawn from the model's own snapshots, so a degraded model produces *looser*
par, meaning attempt 1's escalation was likely **slower**, which strengthens the
degradation reading rather than threatening it.

**The within-run form supports the de-confound; only the between-run magnitude is
unavailable**, and F-37's immediate-forgetting result carries that weight instead.

Recorded as **F-40**: attempt 1's ring became load-bearing when a page cited it as
the counterfactual arm, and registration follows creation rather than citation, so
nothing ever registered it.

---

## Erratum (2026-11-10, F-41)

**Every number above stands as a measurement of its checkpoint.** That includes
P1's 101/600 → 209/600, CI [0.1367, 0.2217], the floors, and top-1 0.9699. The
instruments read the network's priors in the correct coordinates.

**What the campaign trained on did not stand.** Throughout M1, the replay ring
stored root slot positions where `ring_batch` read dense action indices.
- 93.25% of the ring's visit mass was dropped as illegal.
- Of the remainder, more trained the wrong legal action than the right one.
- Under a trained network, the target was discarded entirely in 91.5% of states
  (`FINDINGS.md` F-41).

M1 therefore improved **without the policy target the loop exists to learn
from**. Its +0.18 is real. Its cause is open: value and steps trunk gradients,
supervised rehearsal, or the systematic push toward `eval_add` at low sites.

A resume of this run is refused by code from ring_format 2 onward.

**Amendment (2026-11-10, F-41 review).** Two lines of the erratum above are
corrected.

- "Of the remainder, more trained the wrong legal action than the right one" is
  withdrawn. It was a pooled uniform-stub reading over all ten suites. In the
  stratum matching M1's training problems (BFS par 1–6), correct exceeds
  wrong-legal: 6.00% against 4.61%, with 89.39% dropped. The split under M1's own
  network is unmeasured.
- "Under a trained network … 91.5% of states" named the wrong mechanism. The
  correct statement is network-independent: on 1,094 of 1,200 `solve_in` start
  states, no storable slot lands on a legal cell, so the whole target was dropped
  whatever the network.

The conclusion is unchanged. M1 improved while roughly 90% of its ring's policy
target was discarded.
