# PREREG-chunk9-part0.md — ROCm migration: tolerances declared before measuring

**Amendment policy.** This file is pre-registration. Anything below may be
amended *only* by appending a dated amendment block that states what changed,
why, and what was already known when it changed. No line above an amendment is
edited. An amendment written after seeing a measurement it affects must say so in
its first sentence — that is the whole point of the header, and a pre-registration
that can be quietly edited is a lab notebook written in pencil.

**Status at commit: nothing has been measured.** The venv is still
`torch 2.13.0+cpu`. Every number below is a threshold, not a result. This file is
committed *before* `rm -rf .venv`, so the declaration cannot be retrofitted to
whatever the hardware turns out to do.

---

## 0c. Box-occupancy declaration — made first, because it constrains the rest

Read 2026-08-15, before any GPU work, `rocm-smi`:

| | |
|---|---:|
| VRAM total | 103,079,215,104 B (96.0 GiB) |
| VRAM in use | 1,811,087,360 B (1.69 GiB) |
| KFD processes | **none** |
| Host RAM free / available | 14 GiB / 20 GiB of 30 GiB |
| Swap in use | 3.4 GiB of 8.0 GiB |

**Declaration.** The chess project is **not resident** — no KFD processes, 1.69 GiB
held by the desktop compositor. reckoner may therefore take the GPU, and this
chunk claims:

* **VRAM ceiling: 16 GiB.** A 5.07 M-parameter model with batch 128 at width ≤ 64
  cannot legitimately need more; anything approaching the ceiling is a bug, not a
  capacity need, and the ceiling exists to make it show up as one.
* **reckoner yields to a resident chess process.** Before any GPU run, `rocm-smi
  --showpids` is checked and a non-empty KFD list means reckoner runs CPU or
  waits. It does not co-tenant a 96 GiB pool with a project that assumes it owns
  it.
* **Host RAM stays the scarce pool** (AGENTS.md §3). Swap is already 3.4 GiB deep;
  moving training to the GPU must not be taken as licence to load datasets that
  are currently mapped.

## 0a. The blessed stack

| | |
|---|---|
| index | `https://download.pytorch.org/whl/rocm7.2` (≥ rocm7.1 required for gfx1151) |
| fallback | `https://repo.amd.com/rocm/whl/gfx1151/` |
| install | `rm -rf .venv && make install` — rebuild, never surgery (§4 step 2) |
| env | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
| `torch.compile` | **OFF**, and stays off this chunk (gfx1151) |
| `HSA_OVERRIDE_GFX_VERSION` | **never** — needing it means an old wheel snuck in |

`uv.lock` changes, so the clean-clone gate is re-run and `make lint` (`uv lock
--check`) must be green. `check_env.py` output lands in the run directory, and it
**must not** report a CUDA wheel — it exits non-zero on one.

## 0b. GPU/CPU equivalence — the tolerances, declared

**Bit-identity across devices is not expected and is not the gate.** Different
reduction orders in matmul and attention produce different last bits in fp32.
What must hold is that the *decisions the system consumes* are unchanged.

Fixed input: **512 states sampled via `dataset.sample_indices(n, 512, seed=0)`**
from `phase1_eval` — the blessed subsampler, not a prefix. Same
`runs/phase1/phase1.pt` weights, `model.eval()`, fp32, both devices.

### Tier 1 — decision equivalence. This is the gate.

| | |
|---|---:|
| legal-masked **argmax action** identical, CPU vs GPU | **512 / 512 (1.0000)** |
| legal-masked **top-8 set** identical (as a set) | **512 / 512 (1.0000)** |

Threshold is exact agreement. These are the quantities search and the top-k
metric actually read; if they agree, the device change is invisible to every
gate this project has.

### Tier 2 — numeric divergence. Diagnostic, with bounds.

| quantity | declared bound (max abs) |
|---|---:|
| policy logits | ≤ **1e-3** |
| value probabilities (post-softmax) | ≤ **1e-4** |
| steps head | ≤ **1e-3** |

Logits measured on this model span ~10.3, so 1e-3 absolute is ~1e-4 relative.
**If tier 2 is exceeded while tier 1 holds, that is a finding, not a failure** —
it means the margins are wider than the noise and the gate still stands, and the
number gets recorded rather than the bound quietly widened.

### Both polarities — the null

A comparison that cannot fail is not a comparison. **Perturbation: add `1e-2` to
`rule_embedding.weight[0, 0]`** on the GPU copy only, then re-run tier 1.

| | |
|---|---:|
| perturbed-arm argmax agreement | **must be < 1.0000** |

If a perturbation that size cannot move a single argmax across 512 states, the
comparison is measuring nothing and the smoke is void regardless of what the
unperturbed arm says.

### Four-tuple for 0b (rider (c))

| | |
|---|---|
| floor | not applicable a-priori; two identical computations agree trivially, which is why the null is a *run* |
| null | perturbed-weight arm's argmax agreement, **measured**, must be < 1.0000 |
| threshold | 1.0000 argmax agreement, unperturbed |
| measured | — *(not yet measured; this file is the declaration)* |

### bf16, stated separately and not gated here

AGENTS.md §4.5 prescribes bf16 autocast with fp32 master weights for GPU
training. **Tier 1 is not expected to hold under bf16** and is not asserted
there. The bf16 argmax agreement rate is recorded as an informational number in
the same artifact, because "we did not test it" and "we tested it and it was
0.97" are different states and only one of them is knowledge.

## 0d. Scope

Four-tuples on every gate this chunk declares. Three failed attempts at any gate
⇒ `BLOCKED-<date>-<topic>.md`, never a weakened gate — and for this file
specifically, never a widened tolerance.

## Registered risk, stated before it can be claimed as foresight

`rm -rf .venv` destroys the environment every chunk-8 number was measured in. The
`uv.lock` currently committed pins the CPU stack, so the rebuild is reversible by
`git checkout` of `pyproject.toml` + `uv.lock` followed by `make install` — that
is the rollback, and it is written here rather than discovered under pressure.
The chunk-8 artifacts are already committed and pushed (`ff53afa`), so nothing
measured depends on the venv surviving.

---

# Amendment A1 — 2026-08-15, before any measurement

**Nothing has been measured.** This amendment is written while the venv is still
`torch 2.13.0+cpu`; it is not a response to a result. It answers three defects the
principal found in the declaration above, and it tightens rather than widens every
one of them.

## A1.1 — The null was falsifiable in the wrong direction. **Form (a) is chosen.**

The declared null — "a 1e-2 nudge to `rule_embedding.weight[0,0]` must flip at
least one argmax across 512 states" — can fail while the pipeline is perfectly
correct. Post-training decision margins are wide (top-1 is 0.98), logit spreads
are ~10.3, and a single-scalar 1e-2 nudge is plausibly sub-margin everywhere. The
null would then read "the comparison measures nothing" about a comparison that was
measuring fine. **A null that can fail spuriously is the mirror of a null that
reads high: both misattribute, one in each direction** (F-11 was the other
direction).

**The null's actual job** is to prove the harness is wired to two different
things rather than comparing a tensor to itself. It is therefore asserted on the
*numeric* channel, which responds to any nonzero change, not on the *decision*
channel, which responds only above margin. Two parts, both required:

**(i) Comparator self-check — exact, and guaranteed by construction.** Given a
tensor `T` and `T + 1.0`, the comparison harness must report `max|Δ| == 1.0`
exactly. This cannot pass on a harness that compares a tensor to itself, and its
result does not depend on the model, the device, or any margin.

**(ii) End-to-end perturbation — sized to clear the tier-2 bound, not to flip a
decision.** Perturbation is enlarged from one scalar to **`rule_embedding.weight[0] += 1.0`**,
the whole 128-dim row. Since `logit[b,0,s] = Σ_d site_vec[b,s,d] · rule_vec[0,d]`,
this shifts every rule-0 logit by `Σ_d site_vec[b,s,d]` — a sum of 128 terms of
O(0.1–1), so a change of order 1–10 against a 1e-3 bound.

| | |
|---|---:|
| comparator self-check, `max|Δ|` on `T` vs `T + 1.0` | **== 1.0000 exactly** |
| perturbed-arm `max|Δ|` on policy logits | **> 1e-3** (the tier-2 bound) |
| perturbed-arm argmax agreement | **recorded, not asserted** |

The argmax figure is kept as an **informational** row precisely because it may
legitimately read 1.0000. If it does, that is a measurement of how wide the
trained margins are, and it is a finding about the model rather than a failure of
the smoke.

## A1.2 — The tier-1 sample must span the widths chunk 9 will actually reach

512 states drawn from Phase-1-flavoured data sit almost entirely at width ≤ 64,
because optimal derivations never apply `add_both_sides` (F-05, ROUND-01). Chunk
9's self-play does apply it, and L = 64 versus L = 300 exercise different kernel
paths. **Equivalence proven on the narrow set is a property proven for a set that
is about to grow** — and the set grows the moment the loop starts.

The tier-1 sample is therefore **stratified by width bucket**:

| bucket | source | minimum states |
|---|---|---:|
| 1–64 | `phase1_eval`, blessed subsampler | 128 |
| 65–128 | random legal walks, depth-5/6 suites | 128 |
| 129–256 | random legal walks | 128 |
| **≥ 257** | random legal walks | **128** |

512 total, ≥ 128 per bucket. The wide buckets come from the same capped random
legal play that produced `runs/state_extent.json`, where `add_both_sides` is
always available.

**Measured ceiling, declared so the span is not overclaimed.** `state_extent.json`
puts reachable tokens at p99 216, **p100 332**. Widths 333–512 are not reachable
under the v1 rule set within the step cap, so the top bucket is ≥257 rather than
"up to 512", and equivalence at literal 512 columns is untestable with real states.
That gap is covered from the other side: `_crop_to_content` is already pinned
bit-exact against full 512-width padding (F-07), so the columns beyond 332 are
proven inert rather than proven equivalent.

**The test asserts its own contrast premise**, as the subsampler test does: the
stratified sample must span all four buckets, **and** a `phase1_eval`-only sample
must fail the span check. If Phase-1 data ever starts producing wide states, that
assertion fails loudly rather than the guarantee quietly weakening.

## A1.3 — The 16 GiB ceiling, restated at L ≤ 512 instead of L ≤ 64

The ceiling's framing survives — a ceiling is a bug detector — but its arithmetic
was computed on the Phase-1 premise. Restated at chunk-9 shapes, `B = 128`,
`L = 512`, `H = 8`, 6 layers, `d_model = 256`, `d_ff = 1024`:

| term | fp32 | bf16 |
|---|---:|---:|
| attention scores, `B·H·L²` per layer × 6 | 6.0 GiB | 3.0 GiB |
| per-token activations saved for backward | ~1.1 GiB | ~0.6 GiB |
| params + grads + AdamW moments (5.07 M) | 0.08 GiB | 0.08 GiB |
| **subtotal** | **~7.2 GiB** | **~3.7 GiB** |
| pessimistic (softmax input *and* output retained) | ~13.2 GiB | ~6.7 GiB |

**16 GiB is kept**, because the pessimistic fp32 accounting still clears it. What
changes is that a reading is now classifiable instead of merely alarming:

| observed peak | reading |
|---|---|
| ≤ 7 GiB | expected under bf16 at full width |
| 7–14 GiB | **premise change, not a bug** — most likely fp32 where bf16 was intended, or a wider batch than declared. Check dtype before checking for leaks |
| > 16 GiB | alarm: exceeds what these shapes can legitimately need |

Rider (c) applied to a resource: the envelope is computed before the alarm is
sited, so the middle band reads as "check the premise" rather than "something
leaked".

---

**Frozen.** Amendments A1.1–A1.3 close the declaration. Execution of 0a follows.

---

# Amendment A2 — 2026-08-15, written AFTER a measurement

**Stated first, per this file's own policy: this amendment was written after
seeing a measurement it affects.** The comparator self-check declared in A1.1 was
run and reported `1.000000238418579` against a required exact `1.0`. Nothing else
had been measured — the run aborts at the self-check, by design, before any
device comparison.

**What was wrong: the declaration, not the result.** A1.1 said "given a tensor
`T` and `T + 1.0`, the harness must report `max|Δ| == 1.0` exactly". That is
**false for arbitrary `T`** in fp32: for a non-integer `x`, `x + 1.0` rounds, so
`x - (x + 1.0)` is `-1.0 ± ulp`. The harness used `torch.randn`, and the property
it was asserting cannot hold for that input.

**The threshold is unchanged and is not weakened.** Exact equality to `1.0` is
achievable and remains the requirement; the amendment is that the probe must be
**exactly representable**, so the arithmetic is exact rather than approximately
exact. Measured:

| probe | `max|Δ|(T, T + 1.0)` | `== 1.0` |
|---|---|---|
| `torch.randn(16, 32)` | 1.000000238418579 | **False** |
| `torch.arange(512).reshape(16, 32)` | 1.0 | **True** |
| `torch.zeros(16, 32)` | 1.0 | True |

The harness now uses the integer probe. **No tolerance was added** — the
alternative fix, asserting `|self_check − 1.0| < 1e-6`, was available and
rejected: it would have replaced an exact check with an approximate one to
accommodate a defect in the probe rather than fixing the probe, which is the
shape of weakening a gate to pass it.

**Worth recording as the thing that happened:** the self-check's entire purpose is
to prove the comparison machinery is wired to two different things, and the first
time it ran it failed on its own implementation and aborted the smoke before any
result could be produced. A null that can detect its own harness defect is doing
more than its job description.

---

# RESULTS — 0a and 0b, 2026-08-15

Recorded against the declaration above and amendments A1–A2. No threshold was
changed to accommodate a result; A2 is the only amendment written after a
measurement and it says so in its first sentence.

## 0a — the blessed stack, installed

**Three attempts, per the execution law's counter.**

1. Repin `torch` to `https://download.pytorch.org/whl/rocm7.2`. **Failed** — uv
   could not resolve `triton-rocm==3.7.1`.
2. Add `triton-rocm` and `pytorch-triton-rocm` to `[tool.uv.sources]`. **Failed
   identically.**
3. **Diagnosis before the last attempt** (diagnosis is not an attempt): the wheel
   `triton_rocm-3.7.1-cp312-cp312-linux_x86_64.whl` *does* exist on the index, so
   this was never a missing package. `tool.uv.sources` is honoured for **direct**
   dependencies only; `triton-rocm` arrives transitively through torch, so its
   source mapping was ignored and uv looked on PyPI, where it does not exist —
   which is exactly what "there is no version of triton-rocm==3.7.1" was saying.
   **Attempt 3: promote `triton-rocm` to a direct dependency.** Succeeded.

| | |
|---|---|
| torch | **2.13.0+rocm7.2** |
| triton | **triton-rocm 3.7.1** |
| build | ROCm |
| device | AMD Radeon 8060S, **gfx1151** |
| `HSA_OVERRIDE_GFX_VERSION` | not set, not needed |
| `torch.compile` | off |
| matmul 2048³ fp32 | **2.98 TFLOP/s on GPU** vs 1.05 on CPU |

Clean-clone gate re-run after the `uv.lock` change: **`make lint test` — 569
passed in 134.48 s**, still CPU-only, GPU never a test dependency.

## 0b — equivalence, against the frozen tolerances

**Width-bucket census of the tier-1 sample** (A1.2), n = 512:

| bucket | taken | available |
|---|---:|---:|
| 1–64 | 128 | 50,040 |
| 65–128 | 128 | 35,562 |
| 129–256 | 128 | 12,512 |
| ≥ 257 | 128 | 171 |

Spans all four buckets: **True**. `phase1_eval`-only sample spans: **False** —
the contrast premise holds, so the span assertion is not vacuous.

### Four-tuple (rider (c))

| | |
|---|---|
| floor | not applicable a-priori; two identical computations agree trivially, which is why the null is a run |
| **null** | perturbed arm: max abs Δ **2.057e+01** — DETECTED, 4 orders above the 1e-3 bound. Comparator self-check exact at **1.0** |
| threshold | 1.000000 argmax agreement, unperturbed, fp32 |
| **measured** | **1.000000** (512/512 argmax, 512/512 top-8 set) |

**VERDICT: 0b PASSES.** Tier 1 holds at every width bucket.

**Tier 2 exceeded on policy logits — 3.721e-03 against a 1e-3 bound.** Handled by
the disposition declared before measuring: a finding, not a failure; recorded;
**the bound is not widened**. `FINDINGS.md` F-12 carries the per-bucket
breakdown and the correction that divergence does **not** track width, so A1.2's
stratification is not what surfaced it.

bf16, informational and ungated: argmax agreement **0.998047** (511/512) at max
abs Δ **1.369e-01**.

---

# Amendment A3 — 2026-08-15, tier-2 recalibration under the principal's signature

**Written after a measurement**, and it is a recalibration of a bound that fired,
so the policy header applies in full: the numbers in F-12 were known when this was
written, and they are its derivation.

**This is not the author widening a bound that inconvenienced them.** The freeze
was honoured: tier 2 was reported EXCEEDED, the bound was left at 1e-3, and the
result shipped as a finding. The change below carries the principal's signature by
the same path as gate 10b's amendment, and it rests on a principle that cuts the
other way from convenience — **a bound that always fires is not a gate** (AGENTS.md
§5, rider (b) corollary). An estimate the measurement falsified by 3.7× is not a
reference; the measurement is.

| | before | after |
|---|---:|---:|
| tier-2 policy-logit bound | 1e-3 *(estimate)* | **7.5e-3** *(2 × observed max 3.721e-3)* |
| tier-2 value-probability bound | 1e-4 | 1e-4 — **unchanged**, measured 3.779e-05 |
| tier-2 steps bound | 1e-3 | 1e-3 — **unchanged**, measured 3.204e-04 |

Only the channel that fired moves. **Tier 1 is untouched and remains exact** —
tier 1 is the gate; tier 2 is a change detector, and a detector compares against a
baseline.

**The four-bucket table becomes the committed drift reference.**
`runs/gpu_equivalence_reference.json` carries the per-bucket maxima measured at
`6dbb019`, and tier 2 now reports drift against that reference as well as against
the absolute bound. A future run that comes in at 3.7e-3 is *unchanged*; one that
comes in at 7e-3 is *within bound but drifting*, and only the second reading is
information the absolute bound alone could never carry.

# Amendment A4 — 2026-08-15, A1.3's arithmetic was wrong and is corrected here

**Not a response to a measurement** — no GPU run has happened. This corrects a
derivation I stated and the principal asked to see shown rather than referenced.
Showing it is what exposed the error.

A1.3 put per-token saved activations at "~1.1 GiB". That undercounts by ~5×.
Per token per layer a `norm_first` encoder layer retains roughly
`6·d_model + 2·d_ff` = 6·256 + 2·1024 = **3,584 floats**, not the ~700 the earlier
figure implies.

**At the tensor bound, B = 128, L = 512, H = 8, 6 layers:**

| term | elements | fp32 | bf16 |
|---|---:|---:|---:|
| attention scores, `B·H·L²·layers` | 1,610,612,736 | 6.00 GiB | 3.00 GiB |
| per-token activations, `3584·(B·L)·layers` | 1,409,286,144 | 5.25 GiB | 2.62 GiB |
| params + grads + AdamW (fp32 master) | — | 0.08 GiB | 0.08 GiB |
| **subtotal** | | **11.33 GiB** | **5.70 GiB** |
| pessimistic (softmax input *and* output retained) | | **17.33 GiB** | **8.70 GiB** |

**fp32 at the tensor bound exceeds the 16 GiB ceiling.** The earlier table said
13.2 GiB and cleared it. That conclusion was wrong.

**At the measured reachable width** — `state_extent.json` p100 is **332 tokens**,
and 512 is the envelope, not an attainable state:

| | fp32 | bf16 |
|---|---:|---:|
| L = 332 subtotal | 6.00 GiB | 3.04 GiB |
| L = 332 pessimistic | 8.53 GiB | 4.30 GiB |

**The ceiling holds, and the reason is now stated precisely rather than
approximately:** 16 GiB clears bf16 at *any* width and fp32 at every *reachable*
width. It does not clear fp32 at a width no state can reach. Since AGENTS.md §4.5
prescribes bf16 for GPU training anyway, the ceiling is not load-bearing on the
unreachable case — but the previous table implied a safety margin that did not
exist, and a margin nobody has is worse than a margin nobody claimed.

**Corrected classification bands:**

| observed peak | reading |
|---|---|
| ≤ 4.5 GiB | expected: bf16 at reachable widths |
| 4.5–9 GiB | expected: bf16 at the tensor bound, or fp32 at reachable widths |
| 9–16 GiB | **premise change** — fp32 at or near the tensor bound. Check dtype and batch width before checking for leaks |
| > 16 GiB | alarm: exceeds what any declared configuration needs |

---

# Amendment P0-A5 — 2026-08-15, label disambiguation

**Not a response to a measurement.** This fixes a naming collision I created.

`A1`–`A4` exist in **two documents with unrelated referents**:

| label | in `BRIEF-chunk8.md` | in this file |
|---|---|---|
| A1 | supervision contamination census | the null / width-span / VRAM-ceiling amendment |
| A2 | this run: CPU; ROCm swap is chunk 9 Part 0 | the comparator-probe correction |
| A3 | `batch_leaves` → `batch_searches` | tier-2 recalibration |
| A4 | **extension bound: 5,000 steps, one extension to 10,000** | **A1.3's arithmetic corrected** |

So `grep A4` returns two unrelated rulings, and a reference to "A4" is ambiguous
without naming its document. That is **same-name-different-referent** — the dedup
key hazard inverted, and just as fatal to a grep. One spelling per referent
(AGENTS.md §6) covers amendment labels, and it did not occur to me that it did
until the collision was pointed out.

**Convention, in force from here:** amendments to a PREREG are labelled
`P0-A<n>` — document prefix, then number. The four already committed above are to
be read as **`P0-A1` … `P0-A4`**, and this table is the mapping. They are **not
renamed in place**: amendments append and are never edited, and that rule does not
get suspended for the convenience of the person who broke the naming.

**Which amendment said what**, since I also mis-cited one in a report: the
resource envelope was first declared in **`P0-A1.3`** and its arithmetic was
corrected in **`P0-A4`**. A report of mine referred to "A4" for the
resource-arithmetic lesson; within this file that was right, and globally it was
ambiguous, because `A4` also names the brief's extension bound. Both halves of the
correction are recorded: the reference was under-qualified, and the label was
mine to have qualified.

---

## Comparands

- `runs/gpu_equivalence_reference.json`
- `runs/phase1/phase1.pt`
- `runs/state_extent.json`
- `state_extent.json`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
