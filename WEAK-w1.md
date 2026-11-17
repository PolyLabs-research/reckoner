# WEAK-w1 — a benchmark with headroom, and two things the search was not told

**Written 2026-11-16, before any arm has been run.** DRAFT until the principal
approves it; the approval is recorded at the foot of this page, and nothing
above it is edited afterwards. Amendments are appended, dated.

**Commissioned** by the principal's ruling on `REVIEW-2026-11-15.md`, verbatim:
*"agree on your proposals and order"*. This page is that review's step 1
("restore headroom … train weaker starting networks … make the loop extend them
to depths 4–6, where BFS grades every move exactly") and its step 3 ("let the
search use the steps head … as a one-lever round"), plus one lever the step-1
exploration surfaced.

---

## 1. Why this round exists

The Phase-1 anchor was supervised on problems of depth 1–6, the band the suites
measure, and it already plays 1,176 of 1,200 suite problems at par without
search. Nothing the loop does can be seen against that. A starting network
supervised only on shallow problems has never seen the early states of deep
ones, and every one of its moves on them can still be graded exactly, because
par on the suites is the BFS minimum.

So the question this round answers is the project's thesis in its smallest
form: **starting from a network that has seen no problem deeper than 3, does
the loop reach the deeper ones, and what in the search decides whether it
does?**

## 2. Exploration (not evidence)

AGENTS §7.1. The readings below chose this round's arms and predictions. They
are one run per cell unless two numbers are given, 100 problems per depth (40
for the search reading), a GPU, and a cut-down loop without the league or the
value switch (`scripts/explore_weak_start.py`). **This round measures all of it
again from scratch and inherits none of these numbers.**

Starting network supervised on depth ≤ 3; ten iterations of 200 episodes; at
par means the episode finished in the BFS-minimum number of steps.

| arm | no search, at par, depths 1–6 (of 600) | depth 5 / depth 6 (of 100) | with search, depth 5 / 6 (of 40) |
|---|---:|---:|---:|
| the starting network | 359 | 5 / 0 | 15 / 0 |
| loop as merged | 382, 408 | 37, 37 / 23, 17 | 14, 13 / 10, 8 |
| + an unsolved state at or past par is a loss | 440, 459 | 35, 34 / 7, 26 | 16, 16 / 4, 7 |
| the same at 4× the search budget | 515 | 56 / 60 | 25 / 26 |
| + the search reads its value off the steps head | 574, 577 | 92, 95 / 84, 84 | 40, 40 / 40, 40 |
| Phase-1 anchor (supervised on every depth) | 588 | 95 / 100 | 39 / 40 |

Three more things the exploration showed, each of which shaped the page:

- **The merged loop damages what the starting network already knew.** At par on
  depths 1–3 fell from 299 of 300 to 248 and 272. With the past-par rule it
  stayed at 299 in every run.
- **Repairing the training step made no visible difference** (the steps target
  restricted to solved episodes, the value loss off): 450 and 450 of 600
  against 440 and 459. The review expected more of this than it delivered, so
  it is not an arm here.
- **A network supervised only on depth ≤ 2** reached 554 of 600 with the
  steps-head value, from 217.

## 3. The benchmark

**The starting network.** Phase 1 exactly as the anchor was trained, except for
its diet: `scripts/train_phase1.py --name phase1_d3 --max-depth 3 --steps 3000
--seed 0`, on CPU. `--max-depth 3` keeps the 95,451 supervision rows that come
from problems of depth ≤ 3 and no row from a deeper problem, not even its last
three states (`SupervisionSet.up_to_depth`). It is registered by digest in
`runs/ANCHORS.sha256`.

**What it has and has not been given.** No BFS action label for any problem
deeper than 3. It does get, like every network in this project, each training
problem's **par**: the par game is the environment, and the loop's episodes are
scored against it.

**The settings every arm shares** (`configs/w1_a.yaml`; its difference from the
default is pinned by `test_the_w1_configs_are_the_brief`):

| key | default → here | why |
|---|---|---|
| `train.rehearsal_frac` | 0.65 → 0.0 | Rehearsal rows come from `phase1_train`, which holds BFS labels for the deep problems. |
| `campaign.iterations` | 20 → 10 | The exploration's curves had flattened by 10. |
| `campaign.episodes_per_iteration` | 400 → 200 | Half-size iterations. |
| `train.train_steps_per_iter`, `train.lr_warmup_steps` | 400, 200 → 200, 100 | The same schedule shape at half length. |
| `train.replay_capacity` | 500,000 → 60,000 | A run fills about a tenth of it; the default is 1.3 GiB per saved ring. |
| `ladder.ladder_every` | 5 → 99 | The campaign's cadence instruments carry floors sized for the Phase-1 anchor. This round reads its own (§5). |

Everything else is the default, so the loop is the campaign's: the league
draws par from snapshots for a fifth of the episodes, the value head trains
from the first step, and the value switch is consulted every iteration.

## 4. Arms

Each arm is the one before it plus **one config key**. Three seeds each
(42, 43, 44, through `stream_seed`), nine runs, all from the one starting
network, all through `campaign.run`.

| arm | config | the key it adds |
|---|---|---|
| A | `configs/w1_a.yaml` | none: the loop as merged |
| B | `configs/w1_b.yaml` | `search.past_par: loss` |
| V | `configs/w1_v.yaml` | `search.value_source: steps` |

**`search.past_par`.** Once an episode's step count reaches par with the
problem unsolved, its z is −1 whatever follows. The merged search hands such a
state to the evaluator anyway, and a silent value head answers 0. A solution
found past par is worth −1 exactly, so past par the merged search ranks every
unexplored line above solving. `loss` scores the state −1 and stops.
(`test_past_par_the_merged_search_ranks_an_unexplored_line_above_a_solution`
shows the ranking on a real search.)

**`search.value_source`.** With a silent value head every unproven state is
worth 0, so the search can refute a move only by exhausting everything beneath
it, which 48 simulations manage about two plies from the end. `steps` reads the
steps head instead: a state g steps into its episode with J predicted steps to
go is worth `clip(par − g − J, −1, 0)` against exact par (ceiling +1 against a
par that can be beaten). The steps head has been trained since Phase 1; the
value gate, which exists to wait for an untrained head, is not consulted.

Both keys default to the search M1 ran. With both at their defaults the ring-0
replay is byte for byte the 2026-11-11 ring (digest `4ff19a7d…`).

**On the two settled decisions this touches.** Plan §8 decision 1 fixes the
value head as win/draw/loss against par plus a steps regression, and decision 6
says par enters the label and never the search. Arm V keeps the head and the
par game and changes which output the search reads; par already reaches the
search through terminal scoring (F-13). The principal ruled this change in as a
registered one-lever round, which is what arm V against arm B is. **Neither
default moves on this page's results alone.**

## 5. What is read

The campaign's own no-regress instrument, unchanged: `ladderpass.run_pass` with
`ModelArm` at 48 simulations and at 1, over the six frozen exact-par suites
(200 problems each, 1,200), one `pair_scores` row per problem. It is read on:

- each run's final checkpoint (iteration 9), under its own arm's search;
- the starting network, under each arm's search;
- the Phase-1 anchor, under the merged search, as the supervised reference.

**At par** (z ≥ 0) is the reading. With one simulation it is the policy alone,
and it is exact: an episode ends at par only if every action in it was optimal.
This is the review's "is the chosen action optimal", scored over whole
episodes. Agreement with BFS's single label is not read: in one exploration
arm it fell to 13% on states six steps from solved while play at par on those
problems rose to 92%, because a deep state has many optimal actions and the loop
does not learn BFS's tie-break.

As in the campaign instrument, the win/draw/loss head contributes 0 to the
reading's search under arms A and B, whatever the switch did during the run.

## 6. Predictions and gates

Written before any arm runs. Counts are of the registered populations. Each
gate's four-tuple (AGENTS §5): the floor is 0 for all of them; the null is the
row named; the threshold is the prediction; measured is this round.

| # | prediction | null | exploration, same scale |
|---|---|---|---|
| P1 | The benchmark has headroom: the starting network is at par, without search, on **at most 200 of 600** problems of depth 4–6, and the Phase-1 reference on **at least 560**. | — | 120; 582 |
| P2 | Arm A damages the shallow depths: at par without search on depths 1–3 ends **at least 20 of 600 below** the starting network, on average over seeds. | the starting network | 102, 54 below |
| P3 | Arm B does not: **at most 6 of 600 below**, in every seed. | arm A | 0, 0 below |
| P4 | B − A, at par without search over all 1,200, paired by problem: **the interval excludes zero, above it.** | arm A | +116, +102 |
| P5 | Arm V reaches the deep problems: at par without search on depths 5–6 is **at least 300 of 400** in every seed, and V − B on those depths excludes zero, above it. | arm B | 352, 358 (B: 84, 120) |
| P6 | With search, arm V is at par on **at least 570 of 600** problems of depth 4–6 in every seed. | arm B | 600 (B: 300, 315) |
| P7 | Arm V ends within **60 of 1,200** of the Phase-1 reference, at par without search, in every seed. | the reference | 28, 22 short |

**The tests of record** are paired bootstraps (`ladder.paired_bootstrap`,
10,000 resamples) on the per-problem at-par indicator, averaged over the three
seeds: B − A over all depths, V − B over all depths, and V − B on depths 5–6.

**Where this round's protocol differs from the exploration's**, and so where a
prediction could fail for a reason other than the lever: the full driver (the
league's snapshot par on a fifth of the episodes, the value switch live), CPU
arithmetic on a different torch build, a starting network retrained on CPU,
200 problems per depth, and the unbatched instrument.

## 7. Branches, registered now

- **P3 and P4 hold** → the past-par rule is a defect fix with a measured
  effect. Its default is proposed to the principal for the next campaign.
- **P3 fails** → the forgetting has a second cause in the full loop. Reported;
  B's default stays `evaluate`.
- **P5 and P6 hold** → the steps-head value is the lever that lets the loop
  extend a weak start. The next campaign's PREREG is drafted on it, and the
  rounds the review listed after it (surprise-weighted sampling, reanalyse) are
  measured on this benchmark.
- **P5 fails with P3 holding** → the value lever does not survive the full
  loop. The first suspect is the league: snapshot par can be beaten, which
  lifts the value's ceiling to +1 on a fifth of the episodes. One lever, one
  follow-up round.
- **P1 fails** → the benchmark is not what the exploration saw, and nothing
  else on this page is read.

A sweep with a missing run or reading scores no prediction.

## 8. What this round does not decide

- No default changes here. `search.past_par` and `search.value_source` stay at
  the search M1 ran until the principal rules on this round's reading.
- Whether the campaign should start from a weak network. This is a benchmark
  for measuring levers. The Phase-1 anchor remains the campaign's start.
- The training-step repairs of the review's step 2. The exploration saw no
  effect from them here. They are not withdrawn; they are not tested.

## 9. Where it runs

A rented CPU pod (cpu5c, 32 vCPU), torch `+cpu` from `uv.lock`, 8 intra-op
threads per run as the campaign requires, five runs at a time, through
`cloud/launch_round.sh` (speed gate; results pulled every five minutes; the pod
deletes itself) and `cloud/run_weak_w1.sh`. No GPU: none is licensed for a
registered run. The record is `runs/w1/weak_w1_result.json`, with every run's
rows beside it.

---

**Three failed attempts at any gate → BLOCKED, never a weakened gate.**

---

## Comparands

- `runs/review_20261115/loss_probe_result.json`
- `runs/review_20261115/optimal_set_probe_result.json`
- `runs/ring0_replay/ring-0`
- `configs/w1_a.yaml`
- `configs/w1_b.yaml`
- `configs/w1_v.yaml`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*

---

## Approval (2026-11-16)

**Approved by the principal**, choosing *"Approve and launch (Recommended)"* on
the question: approve this page as committed at `dab241e` and launch its CPU
pod (cpu5c, 32 vCPU, $1.12/h with a $1.25/h ceiling, 6.5 h hard cap, worst case
$7.28). From here the page above is frozen; changes are appended below as dated
amendments.

---

## Amendment (2026-11-16, before any arm ran): three threads per run, nine runs side by side

§9 says "8 intra-op threads per run … five runs at a time". Worked through
before launch, that schedule does not fit the approved cap with margin: a run
is ten iterations of 200 CPU training steps, measured earlier on this pod type
at 1.3 to 2.3 s a step on 8 threads, so nine runs in two oversubscribed batches
could take up to 5.2 hours on top of the starting network's training and the
readings. A run cut off by the deadline is lost, because the pod has no volume.

**Changed:** every arm's config sets `campaign.intra_op_threads: 3`
(8 → 3), and all nine runs start together, 27 threads on 32 vCPUs. Each run's
reading follows it at once instead of waiting for the others.

**Not changed:** the starting network, the two levers, the seeds, the suites,
the readings, the predictions, the tests of record, and the cap. The thread
count is shared by all three arms, so no comparison on this page is between
runs at different thread counts. `test_the_w1_configs_are_the_brief` pins the
shared settings at eight keys.

**What it could change.** A different thread count can change the order of
floating-point sums in training, so these runs are not bit-identical to the
same runs at 8 threads. Nothing here compares against such runs. (On the search
path the thread count changed no bit: F-58's replay gate, 2 threads against 8.)

---

## Amendment (2026-11-16): the round ran once, complete — the readings

### The run

- **Launch.** Pod `4ikl9zs2ueotar` (cpu5c, 32 vCPU, AMD EPYC 9575F), 09:50 UTC.
  Speed gate 0.23 s. The launcher's next calls were refused by the RunPod API
  (HTTP 429: the account's per-minute limit, shared with other pods running on
  it), so it declared setup failed, and its terminate call was refused too,
  which left the pod up. The launcher was fixed (`ce6c212`: the pod's address is
  resolved once, terminate retries, a running pod can be adopted) and **the same
  pod was adopted four minutes after launch.** No arm had started; nothing was
  run twice.
- **Code and stack.** `ce6c212`, clean tree, torch 2.13.0+cpu, data matching
  ANCHORS.
- **The starting network.** 3,000 steps on CPU in 36 minutes; digest
  `a5464d86…`, registered in `runs/ANCHORS.sha256`.
- **The nine runs.** All ten iterations committed in every run, 80 to 96
  minutes each, `alarms {}` in all nine. **The value switch fired in none of
  them**, so arms A and B searched with a silent value head throughout, as the
  exploration's did.
- **Readings and record.** Collected on the pod at 12:28 UTC and pulled at
  12:33; the pod then deleted itself. A local re-run of the collector over the
  pulled rows reproduces the pod's record exactly. About 2 h 45 min of pod time,
  roughly $3.10.

### At par, by problem depth (of 200 each; 1,200 in all)

| | no search: depths 1–6 | total | with search: depths 1–6 | total |
|---|---|---:|---|---:|
| Phase-1 reference | 200 196 196 190 194 200 | 1176 | 200 200 200 199 194 200 | 1193 |
| starting network, A's search | 200 196 200 109 6 11 | 722 | 200 200 200 199 22 20 | 841 |
| starting network, B's search | (the same network) | 722 | 200 200 200 200 22 20 | 842 |
| starting network, V's search | (the same network) | 722 | 200 200 200 200 117 21 | 938 |
| **A** seed 42 | 178 186 155 170 30 10 | 729 | 178 190 155 170 30 16 | 739 |
| **A** seed 43 | 190 185 168 172 13 2 | 730 | 190 189 168 172 17 10 | 746 |
| **A** seed 44 | 196 194 179 189 33 4 | 795 | 196 198 179 189 36 11 | 809 |
| **B** seed 42 | 200 196 200 199 33 19 | 847 | 200 200 200 200 36 20 | 856 |
| **B** seed 43 | 200 196 199 199 34 24 | 852 | 200 200 199 200 35 25 | 859 |
| **B** seed 44 | 200 196 200 199 42 16 | 853 | 200 200 200 200 46 26 | 872 |
| **V** seed 42 | 200 196 200 199 194 186 | **1175** | 200 200 200 200 200 200 | **1200** |
| **V** seed 43 | 200 196 200 199 175 173 | **1143** | 200 200 200 200 200 200 | **1200** |
| **V** seed 44 | 200 196 200 199 189 172 | **1156** | 200 200 200 200 200 200 | **1200** |

The reference row is the instrument checking itself: 1,176 and 1,193 are the
Phase-1 anchor's recorded readings (RUNLOG-m1, F-20), reproduced here on
another host.

### The predictions (§6): all seven hold

| # | prediction | measured | |
|---|---|---|---|
| P1 | starting network at par, no search, depths 4–6: ≤ 200 of 600; reference ≥ 560 | 126; 584 | holds |
| P2 | arm A ends ≥ 20 of 600 below the starting network on depths 1–3, mean over seeds | 77, 53, 27 below (mean 52) | holds |
| P3 | arm B ≤ 6 of 600 below, every seed | 0, 1, 0 below | holds |
| P4 | B − A over all 1,200, paired: excludes zero, above | +0.083, CI [+0.072, +0.094] | holds |
| P5 | arm V at par, no search, depths 5–6: ≥ 300 of 400 every seed; V − B there excludes zero | 380, 348, 361; +0.768, CI [+0.738, +0.796] | holds |
| P6 | arm V with search, depths 4–6: ≥ 570 of 600 every seed | 600, 600, 600 | holds |
| P7 | arm V within 60 of 1,200 of the reference, no search, every seed | 1, 33, 20 short | holds |

The third test of record, V − B over all 1,200: +0.256, CI [+0.234, +0.279].
Every interval is a paired bootstrap over problems (10,000 resamples) of the
at-par indicator averaged over the three seeds; none is saturated.

### What the readings say

- **The benchmark has the headroom it was built for.** The starting network
  plays 126 of 600 deep problems at par; supervision on every depth gives 584.
- **The merged loop damages what the network knew, and its search cannot repair
  it.** Arm A loses 27 to 77 problems on depths 1–3, and its with-search column
  equals its no-search column there (178 and 178 at depth 1 in seed 42). A
  depth-1 problem is one move from solved. The merged search still cannot
  prefer the solving move to an unexplored one, because both read 0.
- **Scoring past-par states as losses removes the damage.** Arm B is within one
  problem of the starting network on depths 1–3 in every seed, and its self-play
  caps fall to 1–5 per iteration where arm A's rise to 16–23.
- **Reading the value off the steps head lets the loop reach the deep
  problems.** Without search arm V plays 348 to 380 of 400 depth-5 and depth-6
  problems at par, where arm B plays 52 to 58. With search it is at par on
  every one of the 1,200, in every seed. The supervised reference misses 7.
- **The steps head was worth reading before any loop ran.** The starting
  network under V's search is at par on 117 depth-5 problems against 22 under
  the others.

### Branches taken (§7)

- **P3 and P4 hold.** The past-par rule is a defect fix with a measured effect.
  Making `search.past_par: loss` the default is proposed to the principal for
  the next campaign.
- **P5 and P6 hold.** The steps-head value is the lever that lets the loop
  extend a weak start. The next campaign's PREREG is to be drafted on it, and
  the later rounds of the review are to be measured on this benchmark.

**No default has changed.** Both keys still default to the search M1 ran.

### Limits

- One benchmark: a depth ≤ 3 start, problems of depth ≤ 6, rule set v1, ten
  iterations of 200 episodes. What these levers do from the Phase-1 anchor is
  not measured here, and cannot be on these suites, which that anchor already
  saturates.
- Par is given to the search, as it always has been. Arm V uses it once more,
  to turn predicted steps into a value. On problems whose par is a loose
  estimate this arithmetic has a +1 ceiling that this round exercised only
  through the league's fifth of episodes.
- The steps head is trained on every ring row here, including unsolved
  episodes (the review's point 3). Arm V worked without repairing that.
  Whether repairing it adds anything is untested.
- Arm V's policy alone is still 1 to 33 problems short of the supervised
  reference. Its search is not.

## Comparands (added by this amendment)

The block above was frozen before the run. The artifacts this amendment rests
on are declared here.

- `runs/w1/weak_w1_result.json`
- `runs/phase1_d3/anchor.json`
- `runs/phase1_d3/provenance.json`
- `runs/ANCHORS.sha256`

---

## Exploration (not evidence), 2026-11-16, after the readings: starts with less supervision

AGENTS §7.1. Run locally on the GPU while the pod ran, with
`scripts/explore_weak_start.py` at `450ae8d`: the cut-down loop, 100 problems
per depth (40 for the search reading), one run per cell, fifteen iterations of
200 episodes. Both levers on, the steps target restricted to solved episodes,
the value loss off (the exploration's arm E). **No registered round has tested
any of this.** If the principal wants it tested, it is this page's successor: a
round whose starting network has less supervision, or none.

| starting network | no search, at par, depths 1–6 (of 600) | by depth | with search (of 240) |
|---|---:|---|---:|
| supervised on one-step problems only: at the start | 234 | 100 99 17 5 13 0 | 129 |
| the same after fifteen iterations | 590 | 100 99 98 95 99 99 | 240 |
| **no supervision at all** (random weights): at the start | 218 | 79 75 52 11 1 0 | 128 |
| the same after fifteen iterations | 535 | 100 100 96 97 90 52 | 240 |

- From one-step supervision the policy alone was within two problems of the
  fully supervised anchor (588 on this population) by iteration 7 and past it
  by iteration 9.
- From random weights, the search was at par on all 240 sampled problems at
  every reading from iteration 5 on. The policy alone stalled at about half of
  the depth-6 problems from iteration 3 on, in both arms tried. That stall is
  not understood.

---

## Ruling (2026-11-16)

Committed on receipt, before acting on it (AGENTS §7).

**What was put to the principal**, verbatim from the session report on this
round, under "Decisions for you":

> 1. **PRs.** Say the word and I'll open both, `foundation-round` first.
> 2. **Defaults.** I recommend making `past_par: loss` and `value_source: steps`
>    the defaults for future rounds, with the old search still available by key.
> 3. **Next round.** I'd register a from-scratch round (random and one-step
>    starts, about the cost of this one) to confirm the exploration and chase
>    the depth-6 stall. If it holds, Phase-1 supervision is optional, which
>    opens deeper problems and richer rule sets where no BFS labels exist. The
>    alternative is to go straight to a benchmark above depth 6.

**The principal, verbatim:**

> open prs
> agree with your next steps, proceed

Read as covering, and acted on as covering:

- **Item 1.** Both pull requests are open: #6 (`foundation-round`) and #7
  (`weak-w1`, stacked on it).
- **Item 2.** `search.past_par: loss` and `search.value_source: steps` become
  the defaults for rounds registered from here on. The search M1 ran stays
  available by setting both keys back. This is §7's first and third branches
  taken to their end. It lands as its own commit on its own branch, after this
  one, and moves the pinned fingerprints once, for both keys together.
- **Item 3.** A from-scratch round is to be registered: starting networks with
  one-step supervision and with none. The alternative named in item 3, going
  straight to a benchmark above depth 6, is not taken now.

What the ruling does not do:

- It approves no protocol, prediction, band or pod spend. The from-scratch
  round gets its own page and its own approval before anything runs.
- It does not re-read any earlier round under the new defaults. F-56 and F-57
  were measured under the search M1 ran and stay as measured.
- It does not change the campaign's starting network or its instrument floors.
  `NO_REGRESS` was derived under the old search. The next campaign's PREREG
  re-derives it.

---

## Amendment (2026-11-17): the ruling's item 2, carried out in part

**Done: `search.past_par` defaults to `loss`.** In the dataclass and in
`configs/default.yaml`. The pinned pair moved once:

| | campaign | eval |
|---|---|---|
| before | `16c1776ead71822f…` | `fb7ba396bee3e77e…` |
| now | `3f61d9402c153413…` | `cfa0d4086d448c64…` |

`FINGERPRINT_HISTORY` holds the earlier pair as its fifth entry, with the value
that moved. The test that walks the history backwards now undoes a moved
default as well as an added key, and still recovers every earlier pair down to
M1's.

**Held: `search.value_source` still defaults to `wdl`.** The ruling named both
keys. While carrying it out, the Phase-1 anchor was read under each search on
all ten suites (the exploration below). On the six exact-par suites the
steps-head value is the best search read. On the four scripted strata, where
par can be beaten, it is at par on 557 problems of 800 where the default it
would replace is at par on 788. This round could not have shown that: its
problems are of depth ≤ 6 at exact par, and its Limits and F-59's "registered,
not done" both name the suspect, the +1 ceiling against a par that is not exact.

A default is what the next campaign runs without choosing, and the next
campaign's primary population is those strata. So the second key waits for the
principal. The agent's recommendation is to leave it at `wdl` until a registered
round has read it there. That round is to be WEAK-w2, drafted on the branch
after this one. The ruling's own text (both keys, one move) is not what was
done, and this section is the record of the difference.

**The search every earlier round ran is one call away.**
`reckoner.config.with_m1_search(cfg)` sets both keys to the values M1 ran, and
at the defaults its result fingerprints at the pair pinned before the move. It
is used where a number must keep its meaning:

- **This round's arms.** `configs/w1_a.yaml` and `w1_b.yaml` now name both keys
  at the values their runs ran; `w1_v.yaml` always did. A test pins each file's
  fingerprint, at each seed, to the one recorded in that run's
  `iterations.jsonl`, and the baseline and reference readings likewise.
- **`scripts/weak_w1.py reference`**: "under the merged search", as registered.
- **`scripts/ring0_replay.py`**: M1's iteration 0, whose halt gate is that
  iteration's own counts.
- **Everything that plays `uniform_stub`.** The stub reports no steps-to-go and
  `'steps'` refuses an evaluator that reports none, so the stub belongs to the
  `'wdl'` search, and the numbers these scripts and tests carry were taken
  before the past-par rule existed: `chunk7_gate_table.py`,
  `chunk11_part0e.py`, `gate_phase1_search.py`, `f54_policy_target_probe.py`,
  `f54_box_iteration.py`, and by name where the probe must also run in older
  trees, `f53_ring_target_probe.py` and `f54_visits_arm_identity.py`. Ten test
  modules build the same config for the same reason.

**What runs the new default without being told:** the campaign driver,
`golden`, the campaign's instruments, and every script that builds `Config()`
and is not listed above. The Part-0 scripts that derive a campaign's baselines
are among them on purpose: the next campaign's baselines are to be read under
the search it will run.

**Tests no longer lean on the default.** The search, runner and evaluator tests
that exercise WEAK-w1's levers now name the search they run (the past-par rule
alone, and with the steps-head value), on a real network, so they hold whichever
way the second key is ruled.

### Gates

| gate | floor | null | threshold | measured |
|---|---|---|---|---|
| The history walks back to M1's pair, one recorded move at a time | 0 of 5 pairs recovered | a config with one other value moved: 0 of 5 (tested) | 5 of 5, both profiles | 5 of 5, both profiles |
| Each arm file fingerprints as its runs recorded | 0 of 9 runs | the files as they stood before this amendment, under the new default: arm A's and B's would not match | 9 of 9, and the 3 baselines and the reference | 9 of 9, 3 of 3, 1 of 1 |
| Ring-0 replay under `with_m1_search` | not reproduced | the same replay with no pin, under today's default: 400 / 400 solved, 87 pool-par, 1,274 rows, HALT | M1's own 398 / 400, 99 pool-par, 1,305 rows, and the ring arrays byte-identical to the tracked ring-0 | reproduced, byte-identical |
| `golden` under the new default | any check failing | not applicable: a liveness check | 15 of 15 checks | 15 of 15, 38.6 s |
| Whole suite, data present | any failure | 82 failures with both defaults moved and nothing else changed: 80 where the stub was refused, 2 that unpacked a pair | 0 failed, 0 skipped | see the commit: `runs/testcount.json` |

### What this does not do

- **`NO_REGRESS`, the watchlist's reference rows and the Part-0d paired
  baseline** were read under the search M1 ran. They are M1's instruments. The
  next campaign's PREREG re-derives them under the search it registers.
- **F-56 and F-57** are not re-read.
- **No floor is claimed for the stub under the past-par rule.** The stub can
  play it (the value source is still `'wdl'`), and no gate has been measured
  that way.

## Comparands (added by this amendment)

- `runs/ring0_replay/replay.json`
- `runs/w1/reference/reading_result.json`
- `runs/w1/baseline/A/reading_result.json`
- `runs/w1/baseline/B/reading_result.json`
- `runs/w1/baseline/V/reading_result.json`

---

## Exploration (not evidence), 2026-11-17: the Phase-1 anchor under each search, all ten suites

AGENTS §7.1. `scripts/explore_weak_start.py read`, on the GPU, batched, one run
per cell, 200 problems per suite, no training. **No registered round has read
any of this.** The round that will is WEAK-w2, drafted on the branch after
this one.

At par, of 1,200 on the six exact-par suites and of 800 on the four scripted
strata (depths 7 to 10). In brackets, how many of those 800 finished under par.

| search | exact, 48 sims | scripted, 48 sims | exact, 8 sims | scripted, 8 sims |
|---|---:|---:|---:|---:|
| the search M1 ran | 1,193 | 788 (265) | 1,189 | 788 (92) |
| + the past-par rule (today's default) | 1,197 | 788 (265) | 1,189 | 788 (92) |
| + the steps-head value, as arm V ran it | 1,200 | 557 (378) | 1,200 | 717 (206) |
| the same with an unproven state worth no more than par | 1,200 | 800 (268) | 1,200 | 799 (92) |

- **The past-par rule costs nothing anywhere it was read.** On the scripted
  strata at 48 simulations it also solves all 800, where the search M1 ran
  leaves 12 at the step cap.
- **The steps-head value as arm V ran it trades at-par for under-par on the
  scripted strata.** It finishes under par more often (378 against 265) and
  over par far more often (243 against 12).
- **Why, as far as this reading goes.** Against a par that can be beaten, an
  estimate may be worth up to +1. The steps head was trained on distances of 6
  and less. On a state 9 steps from solved it guesses low, the state looks
  under par, and that guess outranks a solution the tree has actually found at
  par.
- **The last row is a patch, not a library key.** It caps an estimate at 0
  against every par source. A solution found under par inside the tree still
  scores +1, because that is a proof. It is F-59's "registered, not done"
  item, read for the first time.
