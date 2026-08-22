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

## Host — what the record carries, and what it does not

**M1 has no host description, and that is a gap rather than an omission of
convenience** (F-48). `BRIEF-m1-host.md` §0a existed to prevent exactly this: a
reader in three months cannot say what hardware produced M1.

**Vouched, from committed artifacts:**

| | | source |
|---|---|---|
| interop / intra-op threads | 32 / 8 | `campaign_summary.json` → `threads` |
| train / measure dtype | bf16 / fp32 | `iterations.jsonl` → `train_dtype`, `measure_dtype` |
| wall clock | 34,565.2 s (9.6 h), 20 iterations | `campaign_summary.json` → `seconds` |
| `omp_family` | `unset` | `campaign_summary.json` → `threads` |

**Absent, with reason:** GPU model, CPU, OS, driver, container image. **No
committed artifact records them.** They are not written here from recollection —
a value belongs on a page quoted from the record, never reconstructed, and the
whole point of F-48 is that an assertion with nothing behind it is what produced
this gap.

**Capture deferred to M2 launch.** The pod must come up anyway, and
stop-not-terminate means it is the same host — so the facts will be recorded
**with that caveat**, as *"the host M2 launched on, believed the same as M1's by
stop-not-terminate"*, and never as contemporaneous with M1. A caveated fact is
worth more than a confident reconstruction.

---

## Comparands

- `runs/m1_final/m1_attempt2/` — the run this log records, in full
- `runs/m1_final/m1_attempt2/campaign_summary.json` — the identity and the totals
- `runs/m1_final/m1_attempt2/checkpoint_watch.json` — top-1 at every checkpoint
- `runs/pod_artifacts/m1_rehearsal/` — attempt 1's surviving records
- **GONE: attempt 1's ring** (`runs/m1_rehearsal/ring-*`) — the between-run
  par-escalation comparand. Cited **in prose only**, which is why the sweep's
  extractor could not see it (F-44), and why it is written here by hand.

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
