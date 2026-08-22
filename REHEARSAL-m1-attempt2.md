# M1 dress rehearsal, attempt 2 — expectations, committed before the run

**Written 2026-08-21, before the run.** Attempt 1 failed at iteration 4 on F-29,
after paying its full cadence unit. Everything below is a prediction; the run
scores them.

**Attempt 2 of 3.** Three failed attempts at any gate → BLOCKED, never a weakened
gate.

---

## 1. What runs, and what changed under it

Five iterations at the registered campaign config — now **`8443847bb8c41218…`**,
moved once by M1-A4 — stopped after iteration 4 commits. Same mechanism as
attempt 1: `scripts/campaign.py`, no flags, `LATEST` polled, killed at 4.

The treatment is **`train.rehearsal_frac = 0.65`**, taken mechanically by a rule
frozen before its arms existed.

### Attempt 1 is now a counterfactual arm, not a historical anecdote

It ran **the same driver, the same instruments, the same five iterations**, under
a config differing in exactly the treatment. That makes this a **pre-registered
two-arm comparison across a deliberate configuration change**, which is much
stronger evidence than a floor held in isolation.

---

## 2. THE ITERATION-0 TRIPWIRE — halt here, not at hour six

The single most valuable check on this page. It costs **minutes at iteration 0**
against **six hours at iteration 4**.

| quantity | required |
|---|---|
| ring after iteration 0 | **1,305 rows** |
| episodes solved | **398 / 400** |
| pool-par draws | **99** |
| `ckpt-0` parameter digest | **`5870537e58fcc609…`** — the sweep's `f = 0.65`, seed 0 arm, confirmed present on the volume |
| `ckpt-0` gate-10b top-1 | **1192 / 1229** |

**Halt if any fails.** A driver that does not reproduce iteration 0 is not the
loop the sweep measured, and every downstream number would describe a different
program.

### What the digest check does that nothing else has

The sweep trained from the anchor on ring-0 at `f = 0.65`, seed 0, 400 steps. The
driver's iteration 0 does **the same thing by a different path**. These are **two
implementations of training that have never been compared to each other** — the
sweep harness and the campaign driver — and a bit-identical checkpoint is the
only evidence that the treatment the sweep selected is the treatment the campaign
applies.

The mode differs on entry — the driver calls `.eval()` after loading, the sweep
does not — and **cannot explain a mismatch even in principle**: `module.training`
is not in `state_dict()`, so it cannot reach a parameter digest at all. There is
therefore **no benign reading of a divergence here.** If the digests differ, that
difference is the finding, and it is one no amount of downstream agreement would
have exposed.

The comparand is recorded, not recomputed: `runs/sweep_verify.json` carries the
arm's digest, so the tripwire runs at full strength rather than quietly
downgrading to the weaker count comparison.

---

## 2.1 TOP-1 AT EVERY CHECKPOINT — a six-hour failure becomes a forty-minute one

Read at **every** `ckpt-i`, not only at iteration 4. About a minute each.

**The halt condition is derived, not invented.** The band's derivation — top-1 →
at-par at 2.76 points per point → the floor — **never references an iteration
count**. So a `ckpt-i` below **0.968** cannot hold `at-par >= 1188` at the
cadence, and continuing to iteration 4 spends hours confirming what is already
known. **Halt and diagnose there.**

| iteration | attempt 1 read | attempt 2 expectation |
|---|---:|---|
| 0 | 0.8942 | **~0.9699** |
| 1 | 0.8942 | ~0.9699 |
| 2 | 0.8877 | ~0.9699 |
| 3 | 0.8893 | ~0.9699 |
| 4 | 0.8845 | ~0.9699 |

§8 answers the obvious objection in advance: the loop *"does not trade one
budget's competence for another's in either direction"*, so a model legitimately
learning cannot buy scripted competence with suite competence. **This halt comes
from the frozen page, not from impatience.**

## 3. The two-arm comparison — iteration 4 against attempt 1

| quantity | attempt 1 | **attempt 2 expectation** |
|---|---:|---|
| no-regress @48 | 910 / 1200 | **≥ 1188** — the floor, as a scored prediction |
| no-regress @1 | 722 / 1200 | **≥ 1167** — likewise |
| primary, pooled | 108 / 600 | **≥ 101/600**, no regression against the anchor |
| per stratum 7 / 8 / 10 | 22 / 50 / 36 | recovery toward 43 / 26 / 32 |
| **`ckpt-4` gate-10b top-1** | **0.8845** | **near 0.9699** (the anchor's) |
| caps by iteration | 2 / 3 / 3 / **43** | **single digits through iteration 4** |
| wall clock | 6 h 01 m | **≈ 3.5 h** |
| cadence unit | 227 min (2.07×) | **near anchor rates**, ≈ 110–130 min |

### Why caps matter beyond cost

Attempt 1's caps went 2 → 3 → 3 → **43**, and F-27 registered
`escalation-outruns-learner` as a candidate: par rising faster than the model
improves. That hypothesis is **confounded** with degradation, because a forgetting
model also caps more.

**Single-digit caps under a healthy model de-confound it.** If caps stay low, the
explosion was degradation, and escalation was never the driver. If caps explode
*anyway*, escalation is real and the watch item becomes live.

### Wall clock is the scored form of cadence-cost-as-model-quality

F-27 measured 2.07× three independent ways, all tracking model degradation.
**The prediction is that the factor shrinks toward 1** — which is a falsifiable
consequence of a model-quality theory of cost, not a scheduling note.

---

## 4. The branch table — priors attached, so results are read rather than debated

| floors | top-1 | reading | prior |
|---|---|---|---|
| **hold** | **recovers** | the fix worked | expected |
| **hold** | flat | **suspect the instruments, not the science** | **near-excluded** — 2.76 at-par points per top-1 point makes this combination close to arithmetically impossible |
| breach | recovers | the one-iteration proxy failed to predict four iterations → **§8, BLOCKED, diagnosed** | possible; it is what §3.1's "the band screens, the cadence decides" was written for |
| breach | flat | **the fix did not act** → check `rehearsal_batches` per step | possible; countable precisely for this |

The third row is the one the sweep's design anticipated: a single training
iteration on ring-0 is a **proxy**, and the cadence is the gate.

The fourth row is why `rehearsal_batches` was made a counted field rather than an
assertion — "the treatment ran" must be observable in the artifacts, not inferred
from the config.

---

## 5. The remaining row classes

**Switch criterion** — still abstains, on `thin_minority`, with rarest-class
support far below 100. M1-A4 §5 notes the head sees `(1 − f) = 35%` of its
examples per step at this treatment, so accrual is **slower** than attempt 1's
1 / 6 / 7 / 12 / 17, not faster. An abstention here is expected and is not a
failure.

**Watchlist**, per leg, against the anchor's reference rows `(24, 0)` @1 and
`(7, 0)` @48. Attempt 1 read **7 / 283** @48 — the family intact, the damage
entirely novel. **The expectation is `novel_misses` collapsing toward 0**, since
those 283 were problems the anchor solved and forgetting lost.

**Funnel columns** present in all five rows. Attempt 1's reached-population gap
*opened* (0.812 → 0.978) as the prior flattened; the expectation is that it
**stays near the anchor's** rather than opening.

**`alarms == 0`**, `schema_era == 3`, `config_fingerprint == 8443847b…` in every
row, `pool_composition.size == n + 1`, `instruments.jsonl` carrying one row at
iteration 4.

---

## 6. What gates, and what is recorded

**PASSES iff** the tripwire holds, every row class is present, `alarms == 0`, and
the run reaches `LATEST = 4` without raising.

**Recorded, and carried to a ruling rather than filed as a pass:** the floors,
the primary, top-1, and the branch table's reading. A floor breach is **§8's
BLOCKED branch**, diagnosed before any further iteration.

Artifacts **rsync back before anything is deleted**, and the volume is a network
volume, so a vanished pod is no longer a lost run.

---

## Comparands

- `instruments.jsonl`
- `runs/sweep_verify.json`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
