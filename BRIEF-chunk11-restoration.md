# BRIEF-chunk11-restoration — 2026-08-17

**Four defect fixes. No frozen page is touched, and no treatment is chosen.**

The rehearsal's first attempt found a campaign-blocking defect (F-29), and
diagnosing it found four more. Every item below restores behaviour some page
already specifies; the implementation went around it. The one thing that *is* a
treatment decision — the value of `rehearsal_frac` — is explicitly **not** here.
It lands as M1-A4 after the sweep, carrying measured numbers.

**The governance test each item must pass:** if the fix changes what the campaign
computes at the current fingerprint, it is not a restoration, and it becomes an
amendment. Each item below states how that is *proved* rather than argued.

---

## 1. Wire the rehearsal path (F-31)

`train.rehearsal_frac` is fingerprinted config. `rehearsal_split` implements it.
`train_on_ring` never calls it. Setting the field moves the fingerprint and
changes nothing.

**Build:** supervised batches mixed into `train_on_ring` at the split
`rehearsal_split` already computes.

**The missing polarity.** The four existing tests prove the split is *computed*.
Nothing proves it is *consumed* — which is exactly how a tested function stayed
disconnected. The new assertion is that the supervised gradient **arrives**:
a batch drawn at `f > 0` must produce a different parameter update than the same
batch at `f = 0`, demonstrable at unit scale.

**INERTNESS PROOF — the governance condition.** At `f = 0.0` the wired path must
be *provably inert*: same ring, same seed, **bit-identical checkpoint digest**
against pre-wiring code. If wiring alters training at the current fingerprint,
this is not a restoration.

## 2. Route the cadence through the ladder (F-33 → F-30, F-32)

`run_instruments` calls `run_iteration` and folds the result into `{beat, of}`.
`ladderpass.run_pass` — with per-problem `pair_scores`, the paired-difference
bootstrap, rung passes, mid-pass resumability, declared skips and currency
refusal — is referenced in the driver once, inside a docstring.

**Build:** `run_instruments` delegates to `run_pass`.

**Why this is measurement-only, and why that settles the governance question:**
routing touches **measurement, never loop state.** The ring, the checkpoints and
the pool are untouched, so the campaign's trajectory is identical whether or not
the cadence goes through `run_pass`. That is the routing equivalent of §1's
inertness proof.

Three conditions:

**2a. The seam stays mono-instance.** `run_instruments` remains the single entry
point and *delegates*; it is not bypassed. This preserves D-A2 §3 and improves on
it — `run_pass`'s arms × problems is the unit axis Lever B wants, so the
parallelism story lands behind the seam it was reserved for.

**2b. Equivalence gate, against numbers already recorded.** Routed through
`run_pass` on `ckpt-4`, the aggregates must reproduce the rehearsal's cadence
unit **exactly**:

| quantity | required |
|---|---|
| no-regress @48 | **910 / 1200** |
| no-regress @1 | **722 / 1200** |
| primary pooled | **108 / 600** |
| per stratum 7 / 8 / 10 | **22 / 50 / 36** |

A divergence means the thin implementation was not merely thin but *wrong*, and
that is a finding to diagnose before anything proceeds.

**2c. The instrument passes must not perturb the loop's RNG state.** Measurement
that shifts self-play's draws couples the two silently, and is the one way
routing could reach the trajectory after all. Asserted, not assumed.

**What routing delivers for free:** per-problem pairing (F-30's campaign arm),
gates 10b and 11 actually evaluated (F-32 — two of §8's five BLOCKED branches can
currently never fire), and `ladder.bootstrap_resamples` acquiring a consumer.

**Part-0d still needs its deterministic re-run.** The baseline arm has no
per-problem records and no routing change retro-fits them. Its aggregates
reproducing — `101/600`, and every per-stratum count — is the verification that
the capture is the same measurement.

## 3. The live-config gate (F-31's mechanism rung)

Seven instances of *built, documented, never wired*, and a class chunk 8 declared
closed while two dead keys sat under the declaration. That is the ladder's
terminal signal, so the remedy is mechanical.

**The rule, in its adopted form:** *the question is not whether a field is read,
but whether **varying it across its legal range** changes what runs.*

That formulation replaced a two-tier version, because the two-tier version passes
`rehearsal_frac`. It also makes "observable only at campaign scale" unsayable
without a demonstration.

**Build:** `scripts/config_census.py` becomes a **gate**. Every fingerprinted
field either names the evidence that varying it changes what runs, or fails.
Exemptions live in a registry and carry evidence, never an assertion.

The worked example, because it is the distinction in one pair:

* **`search.perspective`** — exempt. Legal range is exactly one value, enforced
  by `validate`. Varying it within legal values is impossible: *vacuously
  behavioural.*
* **`train.rehearsal_frac`** — fails. Legal range `[0, 1)`, every value passes
  validation, none changes anything: *vacuously inert.*

The census's own construction is part of the record — it needed three refinements
(guard reads excluded, script reporting separated from library consumption,
classes made callable) and its acceptance test is that it must flag
`rehearsal_frac`, a known positive proved dead by hand.

## 4. Assert the interop pin (F-32, related)

`campaign.interop_threads` is read by nothing. M1-A2 §4 classes it **OBSERVED** —
"licenses only 'this is what ran'" — so it is provenance, not a pin, and is
**not** wired. But `assert_threads` reports `torch.get_num_interop_threads()` and
never compares it to the declared 32, so the record's claim is unverified where
one line verifies it.

---

## Not in this round

* **`rehearsal_frac`'s value** — the treatment decision. M1-A4, after the sweep.
* **Dead-key deletions** — every one moves the fingerprint, so they batch into
  M1-A4 with `rehearsal_frac`'s value: one diff, one era, per the batching rule.
* **Epoch-scaling** (steps proportional to ring size) — registered as the second
  lever, not taken. One lever per round; if rehearsal fails to hold top-1, that
  is the next round's candidate rather than a simultaneous confound.
* **`escalation-outruns-learner`** — confounded by the value-head-silent
  mechanism, re-evaluated only after the fix.

## Bookkeeping

M1-A2 §2's rung terms — model rung 5.0 min, greedy 0.0 — currently correspond to
nothing in the code. Routing makes that table **true rather than wrong**; a
descriptive table catching up with reality needs no amendment.

---

## DONE-WHEN

- Rehearsal path wired; gradient-arrival asserted in both polarities; **inertness
  at `f = 0.0` proved by bit-identical checkpoint digest.**
- Cadence routed through `run_pass`; seam still mono-instance; **equivalence gate
  reproduces 910 / 722 / 108 and 22-50-36 exactly**; RNG non-perturbation
  asserted.
- Live-config gate green, with every exemption carrying evidence.
- Interop pin asserted against its declared value.
- **Ring-0 replay reproduces iteration 0 — 398/400 solved, 99 pool-par, 1,305
  ring rows** — which is the regression check on all four fixes at once: those
  numbers can only reproduce if none of them touched the episode path. It also
  answers F-28's draw question exactly.
- `make lint test` green; `golden` green.

Three failed attempts at any gate → BLOCKED, never a weakened gate.

---

## Comparands

- `runs/pod_artifacts/m1_rehearsal/ckpt-4.pt` — §2b's equivalence-gate subject,
  the checkpoint 910 / 722 / 108 and 22-50-36 must reproduce on
- `runs/pod_artifacts/equivalence_gate/` — the gate's own output
- `runs/part0d_paired/` — §2's "Part-0d still needs its deterministic re-run":
  the baseline arm whose `101/600` and per-stratum counts are the verification
- `runs/config_census.json` — §3's live-config gate, the class report
- `runs/ring0_replay/replay.json` — the DONE-WHEN halt gate: 398/400, 99
  pool-par, 1,305 ring rows. **GONE, and never committed** (F-45): the numbers
  survive on this page; the record that produced them does not.

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
