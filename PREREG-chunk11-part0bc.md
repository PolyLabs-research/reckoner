# PREREG-chunk11-part0bc.md — the s\* rule and the mid-strata protocol, before either runs

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `P11B-A<n>`.

**Status at commit: neither Part-0b nor Part-0c has run.** No number below is a
result. Per the ruling's order of operations — *rule before measurement, all the
way down* — this file is committed first and the sweep is run against it.

---

## Inherited from F-20's general form

> Branch rules need four-tuples on **every** branch, including the ones they fall
> back to. A pre-registered rule that can only interrogate one of its own arms is
> half a rule.

`PREREG-chunk11-part0.md`'s rule failed exactly there, so every branch below is
given its own disposition — including the branches where the rule finds nothing.

---

# Part 0b — the s\* sweep

## The protocol

The chunk-11 Part 0 protocol, unchanged, with `sims` varying:

| slot | value |
|---|---|
| model | `runs/phase1/phase1.pt`, the anchor |
| `sims` | swept over **{6, 8, 12, 16, 48}** |
| `gumbel_m` | **`min(16, sims)`** — see below |
| `root_noise` | **False** (eval profile), set explicitly |
| `step_cap` | 24 |
| value wiring | value-silent, `value_scale = 0.0` |
| instrument | the six frozen suites, 1,200 problems |
| seed | 0 |

**`m = min(16, sims)` is stated because the default is incoherent below sims 16.**
Gumbel sequential halving considers `m` root actions and halves them under a
simulation budget; `search` clamps `m` to the number of legal root actions but
**not** to `sims`, so `sims=6, m=16` would nominate sixteen candidates and have
six simulations to separate them. That is not a low-budget search, it is a
different algorithm. Clamping is declared here rather than discovered in the
numbers.

## The metric

**At-par rate** = (problems finishing with `steps == par`) / 1,200, i.e. the `0`
bin of the pinned `STEPS_MINUS_PAR_BINS`. The `<0` bin remains structurally
empty on these suites and is reported at every `sims`.

## The s\* selection rule

**Target 0.55, window [0.4, 0.7], nearest wins.** Formally, over the swept
values whose at-par rate lands in `[0.4, 0.7]`, s\* is the one minimising
`|rate − 0.55|`.

Every branch gets its disposition, now:

| branch | disposition |
|---|---|
| **exactly one** swept value in the window | it is s\* |
| **several** in the window | nearest 0.55 wins |
| **a tie** on `\|rate − 0.55\|` | **the smaller `sims` wins** — it is cheaper, and cheapness is the co-benefit the ruling names |
| **none** in the window, but two adjacent swept values **straddle** it (one below 0.4, one above 0.7) | **bisect**: measure the integer midpoint of the bracket and re-apply this table. Repeat until a value lands in the window or the bracket is adjacent integers |
| bracket reaches **adjacent integers** without landing in the window | **the primary cannot be sited mid-scale.** A finding, filed with its numbers, requiring a ruling. **No post-hoc widening of the window** |
| **every** swept rate is **above 0.7**, including `sims = 6` | extend the sweep **downward** over `{1, 2, 3, 4}` and re-apply. `sims = 1` is the no-search limit — the network's raw policy — and if even that is above 0.7 the anchor cannot be de-saturated by budget, which is a finding, not a smaller window |
| **every** swept rate is **below 0.4**, including `sims = 48` | impossible against Part 0's measured 1193/1200 at `sims = 48`; if observed it means the sweep is not measuring what Part 0 measured, and the discrepancy is the finding |

The window and target are fixed here and are not re-openable by a measurement.

## P1's four-tuple — the slots fixed now, the values filled by the runs

| slot | value |
|---|---|
| metric | at-par rate on the frozen suites at s\*, **paired per problem**, model − anchor |
| **floor** | **the uniform stub at s\***, *measured by running it* — `search.uniform_stub`, no network, same sims, same suites. Rider (c): the floor is a run |
| **null** | **the anchor's own paired baseline** — the anchor against itself at s\*, which is **exactly 0 by the self-match identity** and is *run*, not assumed |
| **ceiling** | **1.0**, with `1.0 − anchor_rate` stated as the distance available |
| threshold | the paired-difference bootstrap's **95% CI excludes zero** and the mean is **positive** |
| measured | filled at each ladder pass (`--ladder-every 5`, ~20 iterations) |

## No-regress at the top of the budget

The loop may not buy low-budget competence by selling high-budget competence.
At `sims = 48`, both conditions, and the first is the binding one:

1. **Hard floor: at-par count ≥ 1188 of 1200.** Computed, not chosen: the anchor
   measured 1193/1200 = 0.994167; the binomial standard error at n = 1200 is
   `sqrt(0.994167 × 0.005833 / 1200) = 0.002192`; the lower end of a two-sided
   95% band is `0.994167 − 1.96 × 0.002192 = 0.989871`, i.e. **1187.8 → 1188**.
2. The paired-difference bootstrap at `sims = 48` must **not** have its 95% CI
   lying entirely below zero.

Plus gates 10b and 11 on the ruled wiring, as planned.

---

# Part 0c — the mid-strata suites

## What is minted

`runs/suites/solve_in_7.jsonl` … `solve_in_10.jsonl`, **200 problems each**,
`par_source = "scripted"`.

**Scripted par is a provisional floor and therefore beatable.** That is the
point, not a caveat: `EXACT_PAR_SOURCES = {"bfs"}`, so the `z = +1` tripwire does
**not** fire against a scripted label, the `+1` cell is **live**, and the par
game's escalation architecture finally has a frozen instrument denominated where
racing is possible. The existing suites cannot host it — nothing beats exact par,
which is F-20.

## The obligations, stated before minting

1. **Deeper emission.** New templates in `generator.py` reaching scripted par
   7–10. BFS labelling is not available here: `solve_both_sides_product` already
   costs ~4.5 s median at depth 6, and the branch factor makes depth ≥ 7
   unaffordable. This is *why* the label is scripted, and the reason is recorded
   rather than left as an omission.
2. **Stratum identity is the label.** A problem lands in `solve_in_k` because its
   scripted par **is** k, never because a template intended k. The chunk-5
   precedent: the template's intended depth is never the label.
3. **Censused at BOTH levels** against `train_100k` (problem) and `phase1_train`
   (state), through `pairedset.census`, which now refuses a single-level call.
   The decision rule is the one already fixed: **collisions dropped before the
   freeze**; contamination found after is a finding, never an edit.
4. **Frozen at birth, digested into ANCHORS**, and refused as a training source —
   `runs/suites` is already registered as an instrument.
5. **`scripted_par_delta` reported where BFS is affordable.** For any minted
   problem whose BFS par can be computed under a stated cap, the gap
   `scripted − bfs` is measured and reported. Where BFS is unaffordable the
   absence is recorded **with its reason**, never as a zero gap. The floor
   language is being paid for; this is the payment.

## Succession, declared now rather than when it is needed

**If the economy primary (P1) saturates, the mid-strata suite is M2's primary.**
Written into `PREREG-m1`'s own text at freeze time, before any pass runs — the
chess lesson (an anchor rung saturating while the real signal sat elsewhere)
arriving as a pre-registration instead of as a surprise. "Saturates" means the
same thing it means everywhere else here: the measured headroom falls below the
CI half-width of the test of record, computed the way `runs/chunk11_ceiling.json`
computes it.

## The rungs go informational

`greedy` and `sympy` against the model **at matched s\***, both currencies in
their own lanes, reported every ladder pass and gating nothing. A rung that
cannot move (sympy solved 251/251) is a reference point, not a criterion.

---

## Reported verbatim

The full sweep table (sims × at-par × over-par × capped × stuck × seconds); the
selection arithmetic showing every branch of the rule that was tested and why it
did or did not fire; s\* and its rate; the four-tuple's floor and null **as runs**;
the ceiling distance; the mint's per-stratum counts, censuses, digests, and
`scripted_par_delta` where computable, with its absence reasoned where not.

---

# Amendment P11B-A1 — 2026-08-15, two record items from the mint

**Written after the mint ran**, and it affects no threshold: both items are
transcriptions of arithmetic and sampling that were already fixed. Neither
changes a rule above.

## 1. The no-regress floor's arithmetic, restated on the page

Ordered so the floor is readable rather than inferable. It appears in the
no-regress section above and is repeated here in one place:

```
anchor at-par, measured (Part 0)    1193 / 1200 = 0.994167
binomial SE at n = 1200             sqrt(0.994167 × 0.005833 / 1200) = 0.002192
two-sided 95% lower bound           0.994167 − 1.96 × 0.002192 = 0.989871
as a count                          0.989871 × 1200 = 1187.8  →  1188
```

**1188 = 1193 minus a 95% binomial noise band**, rounded up to a whole problem.
The floor is the bottom of the band the anchor's own measurement could have
landed in by chance, so holding it means "not distinguishably worse than the
anchor", not "at least as good on every problem".

## 2. `scripted_in_9`'s sampling margin

Recorded for any future re-mint. From a 6,000-candidate pool the strata yielded:

| stratum | candidates yielded | frozen | margin |
|---|---|---|---|
| `scripted_in_7` | 2044 | 200 | 10.2× |
| `scripted_in_8` | 1849 | 200 | 9.2× |
| **`scripted_in_9`** | **251** | **200** | **1.26×** |
| `scripted_in_10` | 1707 | 200 | 8.5× |

**`scripted_in_9` is a near-census of its own yield: 200 of 251, 79.7%.** The
other three strata are samples; this one is very nearly the population the pool
produced. A re-mint must not assume diversity headroom exists at par 9 — it does
not, at this pool size and template set. `solve_mid_four_terms` lands on 10 far
more often than 9, and par 9 is the thin cell between two thick ones.

The census at par 9 therefore ran against 251 candidates rather than the 400 the
other strata got; it returned 0 collisions at both levels, as they all did.

---

# Amendment P11B-A2 — 2026-08-15, a fourth-decimal correction and a floor taxonomy

**Written after P11B-A1 and after the mint**, correcting an arithmetic slip in
P11B-A1 itself. It changes no threshold: the count of **1188 is unchanged**.

## 1. The standard error, recomputed

P11B-A1 and the no-regress section print `SE = 0.002192`. Recomputed:

```
p              = 1193 / 1200      = 0.994166667
p(1 − p) / n   = 0.994166667 × 0.005833333 / 1200
SE             = 0.002198353      (printed: 0.002192)
1.96 × SE      = 0.004308771
lower bound    = 0.989857895
as a count     = 1187.8295        →  1188
```

The printed SE was low by `6.4e-6`. **The lower bound and the floor of 1188 are
unchanged**, which is why this is a note rather than an amendment to the gate.
Caught by the principal recomputing rather than reading — the same move that
produced E5-1 and E5-2 on the same day.

## 2. Every no-regress floor states which kind it is

Countersigned into the record, because it will matter every time a floor is
quoted:

> A **noise-band floor is an indistinguishability gate.** Holding 1188 means
> **not distinguishably worse than the anchor** — it does not mean **at least as
> good on every problem**, and it does not mean **at least as good on average**.

These are three different gates and they license three different sentences. From
here, **every no-regress floor in this project names its kind** at the point it
is declared. The 1188 floor is an **indistinguishability** floor.

---

# Amendment P11B-A3 — 2026-08-16, the downward extension armed before it runs

**Written after Part-0b ran and after seeing its selection return
`s_star: null, needs: downward_extension`** — the measurement that forces this
amendment is already on the record at `runs/chunk11_part0b_sweep.json`. What was
known when it changed: all five swept budgets returned at-par in
`[0.9908, 0.9942]`, every one of them above the window's upper edge of 0.70, so
the `all_above_window` branch fired exactly as pre-registered.

## 1. What changes, and what does not

**One rule, two invocations.** The selection criterion is untouched: target 0.55,
window `[0.4, 0.7]`, nearest-to-target with ties broken toward smaller `sims`.
The **domain** extends downward-only over `(1, 2, 3, 4)` — forced by data, and
covering only points that carry no measurement. That is the amendment policy's
allowance exercised as designed, not a widened window and not a moved target.

**The m-clamp is unchanged and is stated per extended point.** The rule is
`m = min(cfg.search.gumbel_m, sims)` with `gumbel_m = 16`, i.e. `min(16, sims)`,
exactly as declared above and as recorded in the Part-0b protocol block. It is
**not** `min(5, sims)`. For every extended point the clamp binds trivially:

```
sims = 1  →  m = 1        sims = 3  →  m = 3
sims = 2  →  m = 2        sims = 4  →  m = 4
```

The values coincide with what a `min(5, sims)` clamp would give over this domain,
but the rule carried forward is the frozen one. No new clamp is introduced, which
is what a domain-only amendment requires.

## 2. The branch table, frozen before the extension runs

The rule must not meet its next result unarmed. Over the union of the five swept
and four extended points:

**(a) Any extended point lands in `[0.4, 0.7]`.** `s* =` the in-window point
nearest 0.55; ties toward smaller `sims`.

**(b) The window is straddled with a gap** — points above and below, none
within. `s* =` the point nearest 0.55 overall, and **the out-of-window fact is
stated on the row**, not silently carried.

**(c) `sims = 1` still exceeds 0.70.** The suite-economy primary is saturated
too, and **succession fires at iteration 0**: P1 becomes the scripted `{7, 8, 10}`
paired beat-par trajectory — the instrument Part-0d certified live on
2026-08-16. The succession lesson would then have completed its arrival entirely
pre-launch.

## 3. Known at the time of writing: the reference implementation diverges

Stated here because the policy asks what was already known when the amendment
changed, and because a rule that its own code contradicts is not armed:

`scripts/chunk11_part0b.py::select()` as committed at `46e1fdc` implements
**different behaviour on two of the three branches above**.

- On a straddle it returns `s_star: None` with `needs: bisection` (bisecting at
  the bracket midpoint), or `needs: ruling` when the bracket is adjacent
  integers. It does **not** return the nearest-0.55 point that **(b)** specifies.
- On all-above it re-fires `all_above_window` and asks for the same downward
  extension again — degenerate at `sims = 1`, which is the floor of the domain.
  It does not implement the succession of **(c)**.

**The table above governs; the code must be reconciled to it before any
selection is computed.** Only branch **(a)** is presently implemented as
frozen.

Two further mechanical facts about that script, recorded so the extension does
not damage the record:

- `main()` applies `select()` to **only the points named in `--sims`**, not to
  the union of both invocations. Branches **(b)** and **(c)** are union
  predicates and cannot be evaluated from a four-point invocation.
- `main()` writes unconditionally to `runs/chunk11_part0b_sweep.json`. A
  `--sims 1 2 3 4` invocation would **overwrite the five-point record**. The
  five-point record is preserved before the extension runs.

## 4. Rider (b)'s due diligence on the four byte-identical counts

Four points returned at-par `1189/1200` byte-identically (`sims` 6, 8, 12, 16),
which is the house's own trigger for asking whether the knob is connected.

**Item 1 — per-point cost. Answered from the record; the knob is live.**
Wall-clock is strictly monotone increasing in `sims`:

```
sims= 6   m= 6    452.38s
sims= 8   m= 8    550.76s     1.217x
sims=12   m=12    684.81s     1.243x
sims=16   m=16    776.47s     1.134x
sims=48   m=16   1475.88s     1.901x

sims=16 / sims=6 = 1.716x
```

`sims = 16` costs 1.716x `sims = 6`. The identical counts are therefore four
distinct and increasingly expensive computations returning the same answer, not
one computation reported four times. Cost is markedly **sublinear** in `sims`
(2.67x the budget for 1.72x the wall-clock), consistent with episodes
terminating sooner at higher budgets and with batch amortisation.

**Item 2 — the failing sets. Not answerable from existing artifacts.**
`IterationStats` accumulates the `steps_minus_par` histogram and episode
counters only; `run_iteration`'s fourth positional argument is
`ring: ReplayRing | None`, a replay sink, not a per-episode outcome recorder. No
committed artifact carries the identity of the 11 missing problems at any point.
Answering it requires a **separate diagnostic** that records per-episode
`(problem, steps - par)` at `sims = 6` and `sims = 16`, leaving the frozen
instrument untouched. Such a re-run is faithful rather than a fresh sample:
seeding is a per-episode, per-step fan-out from `seed = 0` over a fixed problem
order, so it reproduces the same episodes. Estimated cost `452 + 776 = 1229s`.

**This item is open. The primary must not be built on this axis until it
closes.**

---

# Amendment P11B-A4 — 2026-08-16, the reconciliation ruled: (b) withdrawn, (c) governs

**Two disclosures in the first sentence, because both bear on custody.** The
ruling below was formed *before* any extended rung had returned — it responds to
P11B-A3 §3's recorded divergence between table and code, not to a number. This
amendment is *transcribed* after `sims = 1` and `sims = 2` had both returned
`1176/1200 = 0.9800`, which is disclosed because the reader cannot otherwise tell
that the rule was armed blind. Nothing below was chosen with those two rungs in
view.

A3 recorded a contradiction between two authorities and explicitly declined to
resolve it. This resolves it, and it does not resolve uniformly in the table's
favour.

## 1. Branch (a) stands

Code and table already agreed. No change.

## 2. Branch (b) is **withdrawn** — the committed criterion governs

A3 §2(b) specified: on a straddle with a gap, `s* =` the point nearest 0.55
overall with the out-of-window fact stated on the row. That was called a
criterion that was untouched. **It was not.** Selecting a point outside the
declared window while unmeasured in-window points still exist is precisely the
fishing room the window was frozen to remove.

The committed `select()` is the better design and is retained unchanged:

- **Bisection into the gap is domain completion** — the same move as A3's
  downward extension, pointed inward instead of down.
- **`needs: ruling` at adjacent-integer exhaustion is the honest terminal**, because
  a human ruling on an exhausted domain carries no selection freedom. If that node
  ever fires, out-of-window-nearest becomes *one candidate decided then*, with the
  full bracket in hand — not a rule pre-committed to it now.

## 3. Branch (c) governs — and enters `select()` as code

The code's all-above node was degenerate by its own behaviour: at the floor of the
domain it requested a downward extension below `sims = 1`, which does not exist,
and it carried no successor concept because it predates Part-0d. **Completing a
rule at a node whose input did not yet exist is the amendment policy's exact
allowance**, exercised as designed.

Implemented at `scripts/chunk11_part0b.py`, both polarities, each with a test:

| condition | verdict |
|---|---|
| all-above **and** `min(sims) <= DOMAIN_FLOOR` (= 1) | `needs: succession` — P1 becomes the scripted `{7, 8, 10}` paired beat-par trajectory |
| all-above **and** `min(sims) > 1` | `needs: downward_extension`, as before |

Both are asserted in `tests/test_sweep_selection.py`. The second exists so the
succession clause cannot be a rule that always fires — which would make the
extension unreachable and A3's own domain move a fiction.

## 4. The union merge, hardened

Branches (b) and (c) are predicates over the **whole measured domain**, so a
selection computed from one invocation's points answers a different question than
the rule asks. `main()` previously did exactly that, and wrote unconditionally to
the canonical path.

- `merge_points()` unions by **rung key**. A rung measured twice must agree on
  every count; `seconds` is excluded, being wall-clock with no measurement
  content. **Disagreement raises `RungCollision`** — two measurements of one rung
  disagreeing is a finding demanding a diagnosis, never an overwrite deciding
  silently which one history keeps.
- `load_prior()` refuses a union across differing `config_fingerprint`, checked
  *before* any measurement runs so a mixed-protocol union fails in seconds rather
  than after the sweep is paid for.
- The record now carries `invocations.measured_this_invocation` and
  `carried_from_prior_record`, so a union is never mistaken for one sitting.

## 5. Recorded in passing: the tie-break's actual reach

The declared break — nearest 0.55, ties toward smaller `sims` — compares
`abs(rate - TARGET)` as floats. Two rates that are equidistant *in decimal* are
therefore not tied: `abs(0.50 - 0.55) = 0.05000000000000004` against
`abs(0.60 - 0.55) = 0.049999999999999996`, and the nearer-in-float point wins
outright. The break is **live for equal rates** — the tie this instrument
actually produces, four rungs having returned a byte-identical 1189 — and
**silent for decimal-symmetric ones**. Stated so the rule's reach is not
overstated later. This is a characterisation, not a defect, and no threshold
moves.

---

# Amendment P11B-A5 — 2026-08-16, the tie-break gains its secondary key

**No measurement this amendment affects exists.** It touches branch (a) alone,
which has never fired: every rung measured — 1, 2, 3, 4 at `0.9800` and 6, 8, 12,
16, 48 at `0.9908`–`0.9942` — sits above the window's upper edge, so no selection
has ever reached the in-window node. This is the same allowance A3 §2(c) used:
completing a rule at a node no data has reached.

## 1. What is amended

The declared break — nearest 0.55, ties toward smaller `sims` — gains an
**explicit secondary key: at every tie level, smaller `sims` wins**, stated as a
key rather than left to sort stability.

It is **economy-motivated**, and the motive is the primary's own axis: at equal
informativeness the cheaper rung serves the campaign. A rule whose whole purpose
is siting a budget should not be indifferent about budget when everything else
ties.

## 2. And the comparison becomes decimal-exact

A4 §5 recorded that the criterion compared `abs(rate - TARGET)` in binary float,
so decimal-symmetric rates were not ties. That record **stands untouched as the
description of what the code was**. This amendment records what it becomes:

```
before   abs(0.50 - 0.55) = 0.05000000000000004
         abs(0.60 - 0.55) = 0.049999999999999996     -> sims=8 wins outright
after    Decimal("0.50") and Decimal("0.60") are equidistant from
         Decimal("0.55")                             -> tie, sims=4 wins
```

Rates are counts over 1,200 rounded to six places and the target is 0.55; both
are exact decimals, so `Decimal(str(...))` removes float sensitivity from the
criterion entirely. The old behaviour selected on an artifact of binary
representation — not a threshold anyone declared.

**No threshold moves.** Target, window and the direction of the break are
unchanged. What changes is that the declared break now reaches the symmetric
pairs it always read as though it covered.

## 3. Implementation

`_distance()` and `_rank()` in `scripts/chunk11_part0b.py`, asserted in
`tests/test_sweep_selection.py` — including a test on the key itself, so a future
refactor that reintroduces float arithmetic fails at the key rather than inside a
selection nobody re-derives.

---

## Comparands

- `runs/chunk11_ceiling.json`
- `runs/chunk11_part0b_sweep.json`
- `runs/phase1/phase1.pt`
- `runs/suites`
- `runs/suites/solve_in_7.jsonl`
- `solve_in_10.jsonl`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
