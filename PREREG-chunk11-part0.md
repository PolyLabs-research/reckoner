# PREREG-chunk11-part0.md — the protocol and the decision rule, before the numbers

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `P11-A<n>`.

**Status at commit: Part 0 has not run.** No number below is a result.

---

## Why this file exists at all

Part 0 is a *measurement*, not a gate, so there are no thresholds to pass. There
is one thing that must be fixed in advance and it is the important one: **the rule
that turns the measurement into a choice of primary criterion.** The principal's
ruling states it in words —

> if the suite tail is big enough to carry CI separation over ~20 iterations, the
> primary is tail-reduction (paired, bootstrap as test of record); if it's thin,
> the primary moves to the rung trajectory

— and "big enough" is a threshold nobody has computed the floor of. Rider (c)
applies with the most force here, because this is the criterion the whole campaign
answers to. So the arithmetic is done now, against no data.

---

## The protocol (M1's exact eval protocol, stated so the measurement is repeatable)

| slot | value | source |
|---|---|---|
| model | `runs/phase1/phase1.pt`, the anchor | `ANCHORS.sha256` |
| `search.sims` | **48** | `config.SearchConfig.sims` |
| `search.gumbel_m` | **16** | `config.SearchConfig.gumbel_m` |
| `search.root_noise` | **False** — the **eval** profile | set explicitly; the config *default* is `True`, the self-play value |
| `search.perspective` | `single` | `validate()` enforces it |
| `episode.step_cap` | **24** | `config.EpisodeConfig.step_cap` |
| value wiring | **value-silent**: `value_scale = 0.0` | `valuegate.value_contribution` on an unswitched head (F-14) |
| measure dtype | fp32 | `numerics.measure_dtype`, licensed |
| device | CPU | the golden/measurement convention |

`root_noise = False` is stated rather than inherited **because the default is the
other value**. A measuring rung that silently took the self-play profile would
report a distribution the campaign's eval passes will never see.

## What is measured

**M-A — the anchor's suite z-composition.** The full `steps − par` histogram over
all six frozen suites (`solve_in_1` … `solve_in_6`, 200 problems each, **1,200
total**), reported per suite and pooled, in the pinned `STEPS_MINUS_PAR_BINS`
(`<0`, `0`, `1`, `2`, `3`, `4`, `5`, `6+`), alongside solved / capped / stuck.

The `<0` bin is expected to be **structurally empty** and is reported anyway: the
suites carry BFS-exact par, nothing beats exact par, and three independent layers
refuse a row claiming otherwise. Reporting it is the definitional zero doing its
job — its absence from the table would be indistinguishable from nobody having
looked.

**M-B — the rung baselines.** The anchor against `greedy` (z lane) and `sympy`
(solve-vs-budget lane) on the frozen paired set `runs/paired/smoke_v1.jsonl`
(389 problems, digest `f0b10fd1…`), `pair_scores` from row one, each currency in
its own lane and never differenced across.

## The decision rule, computed now

Let **T** = the number of over-par problems the anchor leaves on the suites, out
of 1,200. T is the *entire* movable mass for a tail-reduction primary: solved is
already 1,200/1,200 and the `<0` bin is structurally empty, so the campaign can
only move problems from the over-par bins into the `0` bin.

For a paired comparison of binary outcomes (over-par / not) the mean paired
difference is `D / n` with `D = k_improved − k_worsened`, and its bootstrap
standard error is driven by the **discordant** pairs `k_d = k_improved +
k_worsened` — concordant pairs contribute zero difference and no variance. The CI
excludes zero when

```
D > 1.96 · sqrt(k_d)
```

Assume, before seeing T:

* the campaign fixes a fraction **f = 0.25** of the tail over ~20 iterations — a
  quarter, which is modest for 20 iterations of self-play improvement;
* regressions run at **25% of improvements** (`k_worsened = 0.25 · k_improved`) —
  not zero, because a policy that changes changes in both directions.

Then `k_improved = 0.25·T`, `D = 0.75·k_improved`, `k_d = 1.25·k_improved`, and

```
0.75·k_i > 1.96·sqrt(1.25·k_i)
0.5625·k_i² > 4.802·k_i
k_i > 8.54   →   k_i ≥ 9   →   T ≥ 36
```

**The rule, fixed:**

| measured T | primary |
|---|---|
| **T ≥ 36** (≥ 3.0% of 1,200) | **tail-reduction** on the frozen suites: mean paired difference in over-par indicator, iteration 0 vs iteration ~20, paired bootstrap as the test of record |
| **T < 36** | **rung trajectory** on the frozen paired set: the model's z against the rungs over iterations, paired bootstrap as the test of record |

**T = 0 is a live possibility and it is not a failure of Part 0.** It would mean
the anchor is already at par everywhere the suites can see, which is exactly the
chess succession lesson — an anchor rung saturating — arriving *by arithmetic
before the campaign* instead of by surprise during it. That is the outcome this
measurement exists to be able to report.

## What is NOT decided here

The rung-trajectory primary's own four-tuple, if it is selected. Its null is a
**run** (the self-match, already implemented) and its floor is computed from the
z domain, but its *threshold* depends on M-B's measured baselines, which is the
point of measuring them. Fixing it now would be inventing a number to look
pre-registered.

## Reported verbatim

Per-suite and pooled histograms; T and its percentage; solved/capped/stuck splits;
the M-B pair counts, means, CIs and currencies; wall clock split by measurement;
the resolved protocol table above as it was actually applied, read back from the
config rather than re-typed.

---

## Comparands

- `runs/paired/smoke_v1.jsonl`
- `runs/phase1/phase1.pt`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
