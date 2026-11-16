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
