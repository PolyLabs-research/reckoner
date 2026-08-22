# M1 dress rehearsal — expectations, committed before the run

**Written 2026-08-17, before the run.** This is the last row of
`BRIEF-chunk11-driver.md`'s DONE-WHEN table, as amended by D-A2 §1 from three
iterations to five.

Every number below is a prediction. The point of writing them first is that the
run scores them; a rehearsal whose expectations are written afterwards is a
description.

---

## 1. What is run, and how five iterations happen at a twenty-iteration config

`campaign.iterations = 20` is **under the fingerprint**. Setting it to 5 changes
the config fingerprint from `ce41af96…` to `7ba86c45…`, and `run_campaign`
refuses it — that refusal is the gate working, not an obstacle to route around.

So the rehearsal is **the campaign's first five iterations at the registered
config**, stopped after iteration 4 commits:

```
run_campaign(run_dir, Config(), anchor=runs/phase1/phase1.pt)     # 20 iterations
# poll LATEST; when it reads 4, kill the process.
```

Killing is safe by construction, and the resume gate is what licenses that
sentence: iteration 5's provisional artifacts are exactly the debris
`test_campaign_resume.py` kills into and resumes from, at both boundaries.

**No behavioural flag is introduced, and no config is edited.** The rehearsal
config *is* the campaign config, byte for byte, which is what makes the rehearsal
a rehearsal rather than a smaller different thing.

## 2. Cost

From M1-A2 §2's measured, pod-scaled rates:

| term | per iteration | × 5 |
|---|---:|---:|
| self-play @ 400 episodes | 10.1 min | 50.5 min |
| training @ 400 steps | 10.4 min | 52.0 min |
| cadence unit, at iteration 4 only | — | 109.5 min |
| **total** | | **≈ 3.5 h** |

The cadence unit is a third of the rehearsal, and iteration 4 is the only
iteration that pays it: `ladder_every = 5`, so `(n + 1) % 5 == 0` first holds at
`n = 4`. **Five is the minimal count that buys a cadence unit**, which is D-A2's
whole argument and is re-derived here rather than quoted.

---

## 3. Predictions — artifacts

At the stop, `run_dir` contains:

| artifact | expected |
|---|---|
| `LATEST` | `4` |
| `iterations.jsonl` | 5 rows, iterations `0…4` |
| `value_switch.jsonl` | 5 rows, iterations `0…4` |
| `instruments.jsonl` | **1 row, iteration 4** |
| `ckpt-0…4.pt`, `snap-0…4.pt` | 5 each |
| `state-4.json`, `ring-4/` | present |
| `_preflight/` | **absent** — the pre-flight removes its own scratch |

## 4. Predictions — every iteration row

1. `schema_era == 3` in all five.
2. `config_fingerprint == ce41af96ee85f0a2…` in all five — and now because the
   config that ran *is* that config, F-25 having removed the constant that used
   to say so regardless.
3. `pool_composition` present in all five, with **`size == n + 1`**: the anchor
   seeds the pool, and iteration *n* draws before enrolling its own snapshot.

   | iteration | 0 | 1 | 2 | 3 | 4 |
   |---|---|---|---|---|---|
   | `size` | 1 | 2 | 3 | 4 | 5 |

4. At iteration 4, **`order` and `steps` disagree** —
   `order == [5000, 0, 1, 2, 3]`, `steps == [0, 1, 2, 3, 5000]` — because the
   anchor enters first at training-step 5000 and the snapshots enrol at their
   iteration index. This is M1-A3 §3's argument observed rather than asserted,
   and it is also F-24's registered denomination mismatch showing up in an
   artifact for the first time.
5. `ladder_pass` **absent with its reason** for iterations 0–3, and `== 0` at
   iteration 4.
6. `pool_par_fraction` present in all five and `≤ 0.2`, since
   `par_from_pool_frac = 0.2` is the draw rate and a drawn snapshot that fails to
   solve falls back to `bfs` with the cause counted.
7. All four funnel columns present in all five rows:
   `entropy_prior_step1_{start,reached}`, `entropy_target_step1_{start,reached}`.
8. `evaluator_checkpoint_sha256` at iteration 0 `== 45333caa…` (the anchor), and
   **at least two distinct values across the five** — a digest that never moves is
   the stub defect wearing a provenance field.
9. **`alarms == 0`.**

## 5. Predictions — the switch log

Five rows, one per iteration, **including the abstentions**: an abstention nobody
records is indistinguishable from a criterion nobody ran. Each carries
`abstained`, `fired`, `already_live`, `clears`, `n`, `class_census` and
`k_classes_with_support`.

Abstention is *expected* early and is **not** a failure: M1-A2 §6 predicts `K`
starts at 2 because `+1` against `bfs` par is impossible by construction, and the
criterion needs support it cannot have at iteration 0.

## 6. Predictions — the cadence unit at iteration 4

One `instruments.jsonl` row carrying:

- `config_fingerprint == 314fbeb99b6640f6…` (the **eval** profile) and
  `profile == "eval"` — asserted at the seam on entry, because a cadence
  measurement is comparable to its baseline only if it provably ran that profile.
- `no_regress_sims_48`: `of == 1200`, `floor == 1188`, `floor_recomputed == 1188`
- `no_regress_sims_1`: `of == 1200`, `floor == 1167`, `floor_recomputed == 1167`
- `primary`: pooled over 600 problems, `anchor_baseline == "101/600"`, with
  `per_stratum` for `scripted_in_7`, `scripted_in_8`, `scripted_in_10`, and
  `delta_vs_anchor` beside it — M1-A2 §1's requirement that the funnel trigger's
  row carry the contemporaneous beat-par delta, which became satisfiable from
  artifacts only with F-26's `instruments.jsonl`.

---

## 7. What gates, and what is merely recorded

**The rehearsal PASSES iff** every row class above is present with the structure
predicted, `alarms == 0`, the artifacts are complete, and the run reaches
`LATEST = 4` without raising.

**Recorded but NOT gating:** the `at_par` counts, whether the floors held, and
the primary's pooled beat-par and delta.

Five iterations of a twenty-iteration campaign is not the campaign, and reading
its numbers as a result is exactly the retrospective move pre-registration
exists to prevent. The rehearsal is a test of the *instrument*, not a measurement
*with* it.

**But a breach is not nothing.** If either no-regress floor fails at iteration 4,
that is recorded and **carried to a ruling before M1 launches** — PREREG-m1 §8
makes a gate-10b/11 regression a BLOCKED condition for the campaign, and a
rehearsal that saw it coming and said nothing would be worse than one that never
ran. Not a rehearsal failure; a launch-blocking observation.

## 8. Disposition

Artifacts are **deleted after recording**, per plan. Recorded means: this file
gains a results section, `RUNLOG-chunk11-driver-part0.md` gains the run, and the
`instruments.jsonl` row and the five iteration rows are transcribed into the
record before the directory is removed.

**Three failed attempts at any gate → BLOCKED, never a weakened gate.**

---

# RESULTS — attempt 1, **FAILED**

**Run:** 2026-08-17 09:54–15:55 UTC, pod `jshsnmwop6pgn`, git `102eef7`, config
`ce41af96…`. Artifacts recovered to `runs/m1_rehearsal_recovered/`.

## Verdict

**FAILED — attempt 1 of 3.** §7 requires the run to reach `LATEST = 4` without
raising. It reached `LATEST = 3` and raised at iteration 4's row write, *after*
the cadence unit completed. The cause is **F-29**, now fixed with both polarities
tested.

`iterations.jsonl` carries 4 rows; `value_switch.jsonl` carries 5 (the criterion
runs before the commit); `instruments.jsonl` carries iteration 4's row, which is
provisional against `LATEST = 3` and would be correctly truncated by resume —
**recovered off the pod before anything touched it.**

## What passed, at four iterations

`schema_era == 3` throughout · `config_fingerprint == ce41af96…` throughout ·
`pool_composition.size == n + 1` (1, 2, 3, 4) · `ladder_pass` absent with its
reason for 0–3 · all four funnel columns present · iteration 0's evaluator digest
`45333caa` and moving thereafter · **alarms 0** · every switch row written,
abstentions included.

## Predictions that failed

**§4.6 — `pool_par_fraction ≤ 0.2`.** Observed 0.2475, 0.185, 0.1925, 0.205.
**The prediction was wrong, not the code.** `par_from_pool_frac = 0.2` is the
Bernoulli parameter of the draw, so the realised fraction is Binomial(400, 0.2)/400
— mean 0.2, sd 0.02 — and exceeds 0.2 about half the time. The page reasoned that
fallbacks can only reduce it and treated a one-sided effect as a ceiling.

Not amended. It is committed at `0d31989`, it failed, and it is recorded as
failed — amending a prediction at the moment it fails is the move D-A2 §1 named.
What it exposes is a drafting error: **a distributional claim was placed in a
structural checklist and thereby given gating status.** §4.6's gating content was
"the column is present", which passed. The numeric claim belonged in §7.

## RECORDED, NOT GATING — the cadence unit

| instrument | measured | floor | anchor | held |
|---|---:|---:|---:|:--:|
| no-regress @48 | **910 / 1200** | 1188 | 1193 | **NO** |
| no-regress @1 | **722 / 1200** | 1167 | 1176 | **NO** |
| primary {7,8,10} | **108 / 600** | — | 101 / 600 | **+1.17 pp** |

Per stratum: `scripted_in_7` 22/200 (anchor 43), `scripted_in_8` 50/200
(anchor 26), `scripted_in_10` 36/200 (anchor 32).

**Both indistinguishability floors are breached, and by margins that are not
marginal** — 278 and 445 problems below floors set at the anchor's own one-sided
95% band. At-par on the suites fell from 99.4% to 75.8% at `sims = 48`.

**And the primary rose.** The model is worse on the easy suites where the anchor
was saturated, better on the hard scripted strata — losing 21 at stratum 7 while
gaining 24 at 8 and 4 at 10.

**Disposition, pre-stated in §7 and applied without amendment:** a breach is a
finding plus PREREG's decision rules, never an adjustment. PREREG-m1 §8 makes a
gate-10b/11 regression a **BLOCKED** condition for the campaign. This is four
iterations of a twenty-iteration run and is *not* the campaign's result — but it
is exactly the observation §7 says is carried to a ruling **before M1 launches**,
and it is carried, not filed as a pass.

The floors did the job they were frozen for.

## Also recorded

- **F-27** — the cost model is flat where the curve is not; total 6 h 01 m
  against a 3.5 h prediction, the cadence unit at 2.07× its model.
- **F-28** — `pool_par_unavailable` reaches no artifact; iteration 0's exact
  draw count is recoverable by offline replay.
- **The switch criterion cleared its threshold at every iteration and abstained
  every time.** Measured balanced accuracy 0.600, 0.553, 0.579, 0.550, 0.560
  against a threshold of 0.483 — margin 0.07–0.12 throughout, never the
  discriminating quantity. `clears` is False solely because `evaluable` is False.
  `k_classes_with_support` counts classes **present** (`k = len(census)`), not
  classes clearing `MIN_CLASS_SUPPORT = 100`; the phrase and the behaviour
  disagree. The rarest class runs 1, 6, 7, 12, 17 against a floor of 100, so on
  this trajectory the head never goes live inside 20 iterations.

---

## Comparands

- `instruments.jsonl`
- `iterations.jsonl`
- `runs/m1_rehearsal_recovered/`
- `state-4.json`
- `value_switch.jsonl`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
