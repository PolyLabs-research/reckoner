# Findings

Measured facts that are not rounds. A round is a plan to test a change; a finding
is something already true that later chunks must not rediscover. Errata culture:
every defect named, including the trivial ones, and nothing fixed silently.

---

## F-01 — Scripted-solver suboptimality, first measured instance

**Found:** 2026-08-14, chunk 4 proofread (derivation 19 of `docs/derivations.md`).

The greedy scripted policy prefers `eval_add` over `eval_mul`, and on
`7 + (9 − 30) + 12 × 12` that costs a step: it evaluates the subtraction, folds
the two loose constants, then multiplies, then folds again — **4 steps against a
BFS-exact par of 3**. The optimal order evaluates the product first so a single
`eval_add` folds all three numerals at once.

**Why it matters beyond one derivation.** This is the first *measured* instance
of the gap that scripted par's "provisional floor" status (spec §3) is priced
for. It is small (one step at depth 3) and it is real, and until chunk 5 measures
the delta distribution it is the only data point.

**Consequence, already in chunk 5's brief:** on the depth ≤ 6 overlap where both
labels exist, compute BFS *and* scripted par and report the delta distribution —
free calibration of exactly how provisional the mid-strata floor is.

**Status:** recorded. The document now renders BFS-optimal derivations, so the
exhibit is preserved here rather than on the page.

---

## F-02 — Provenance acquired by silence

**Found:** 2026-08-14, chunk 4 proofread (derivations 33, 36, 39, 42, 45, 48).

Six derivations carried `par 2 (par_source=bfs)` beside a one-step solve, giving
`z = +1`. That row is impossible by construction: BFS-exact par **is** the
minimum step count in this rule system, so nothing can beat it.

**Root cause — neither of the two branches the review named.** BFS's terminal
test did not diverge from the episode's, and the par pipeline did not diverge
from BFS's output. **There was no BFS labeller at all.** Every par in the fixture
set was a hand-written literal, and `Problem.par_source` *defaulted* to `"bfs"`.
The document claimed exact provenance for numbers that had never been computed.

Audit of all fifty: **6 mislabelled, 44 correct by luck** — the hand-written
values happened to be right everywhere else, which is precisely why nothing
caught it.

**The structural fault, and the general lesson:** a provenance field whose
default is its *strongest* claim is not a provenance field. The most-trusted
value must be the one that costs something to say. Exactness is asserted, never
inherited.

**Fixes, all landed:**

| | |
|---|---|
| `Problem.par_source` default | `"bfs"` → `"unverified"`, validated against a known set |
| `EpisodeResult.__post_init__` | raises on `z > 0` with an exact `par_source` |
| `episode.bfs_solution` / `bfs_par` | a real labeller, calling `verify()` so a label and an outcome cannot disagree about what "solved" means |
| `scripts/render_derivations.py` | fixtures state *problems*; `label()` computes par and its true provenance |
| `docs/derivations.md` | regenerated: 0 × `z = +1`, 47 × `z = 0`, 3 × `z = −1` (the labelled ILLUSTRATIVE entries) |

**Status:** closed, with the invariant pinned. The reason it is pinned in the
result path rather than in a test is scale: chunk 5 mints 100K par labels and
1,200 suite pars, and a par off-by-one at that scale poisons the game's currency
at birth.

---

## F-03 — The pre-flight underestimated by 5×

**Found:** 2026-08-14, chunk 5 generation.

The stratified pre-flight projected **18.3 min** for 100K problems on 24
workers. Actual: **5,636.9 s — 94 min**, 5.1× over.

**Why.** Three compounding causes, none of them the stratification the
projection got right:

1. **Sample too small for a heavy tail.** 20 candidates per stratum, against a
   template (`solve_both_sides_product`) whose cost ranges 4.5 s median to 8.8 s
   worst. Twenty draws does not estimate the mean of that distribution.
2. **The projection priced labelling, not generation.** The driver emits ~1.6×
   the requested count and deduplicates, so the real work is ~1.6× the projected
   work before any retry.
3. **Retries are unpriced.** Strata that under-fill trigger another pass.

**The correction, for the next projection that matters:** sample per *template*
rather than per stratum, use the mean of ≥100 draws where the tail is heavy, and
multiply by the over-generation factor. The stratification instinct was right —
projecting from a flat average would have been worse — but a stratified estimate
built on 20 samples of a long-tailed distribution is still an estimate of the
wrong number.

**Not a blocker:** 94 min ran unattended and the artifacts are correct. Recorded
because the next pre-flight will be for something that cannot be re-run casually.

---

## F-04 — Depth histograms are outcomes, not requests

**Found:** 2026-08-14, chunk 5 generation.

The generation plan asks for equal counts per stratum. The training set came out
`{1: 20751, 2: 16582, 3: 14068, 4: 29036, 5: 16013, 6: 3550}` — depth 4 nearly
double its share, depth 6 at a fifth of it.

**This is the design working, not failing.** Difficulty is parameterised by
*measured* depth, so a candidate lands in the stratum BFS says it belongs to, not
the one its template was aimed at. `solve_two_terms` splits 4/21 across depths 3
and 4; `eval_deepest` splits 4/56 across 5 and 6. The plan is a request; the
histogram is the answer.

**Consequence for chunk 8:** the warm start sees this distribution, not a uniform
one, and depth 6 is thin (3.6%). Either the sampler rebalances at training time
or the generation plan over-requests the deep strata. Recorded here so it is a
decision rather than a surprise in a loss curve.

---

## F-05 — `add_both_sides` is the entire source of state growth

**Found:** 2026-08-14, chunk 6 sizing measurement.

Random legal play from the depth-5/6 suites, 24 steps, 1,600 walks:

| | tokens p99 | tokens p100 | sites p99 | sites p100 |
|---|---:|---:|---:|---:|
| with `add_both_sides` | 247 | 295 | 87 | 104 |
| **without** | **57** | **57** | **15** | **15** |

Without it, a reachable state never exceeds the scale of a *start* state
(p100 64 tokens / 17 sites). With it, states reach 5× the tokens and 7× the
sites — and since attention is O(L²), that is roughly a 25× cost multiplier on
the trunk, plus a policy head 7× wider.

**This is the third independent argument for ROUND-01**, and the first that
costs money rather than elegance:

1. reachability-redundant — par is identical without it (chunk 2)
2. an active trap — any policy that prefers it fails to terminate (chunk 4)
3. **it alone forces the model's sequence and action bounds** (here)

**What chunk 6 did with it:** nothing. The rule is in the pinned v1 set, so the
model is sized for the rule set that exists, not the one a round might produce —
the same discipline as computing par against the full set until ROUND-01 fires.
The measurement is recorded so the round's cost/benefit is a number rather than
a preference.

---

## F-06 — A search that did not search, and four gates that passed anyway

**Found:** 2026-08-14, while building chunk 8's depth-≤2 gate.

Chunk 7 shipped what its docstring called an array-tree MCTS. It expanded only
the root's children and then re-backed-up their already-computed values. On a
root with 6 legal actions at `m = 5`, `nodes` was pinned at 6 for every budget it
would ever be given:

    sims   6  16  31  48   ->  nodes  6  6  6  6      (pre-F-06)
    sims   6  16  31  48   ->  nodes  7 17 32 49      (fixed)

Forty-seven of forty-eight simulations did no work. It was a one-ply lookahead.

> **Erratum, 2026-08-14 (`ERRATA-chunk7.md` §3).** This finding first published
> the exhibit as `sims=48, m=5 -> nodes in tree: 2`. That number is real but
> under-specified to the point of being undemonstrative: 234 of the 1,200 frozen-
> suite problems have a single legal root action, and on 160 of them `nodes = 2`
> is the complete and correct tree under *both* searches, because the one action
> solves the problem. The exhibit above replaces it — flat `nodes` **while
> unexpanded actions remain** is the signature. The named-problem version of the
> original figure is a depth-2 single-action root: pre-F-06 `nodes 2, evals 49,
> terminal_solved 0` at `sims=48`; fixed, `nodes 3, evals 2, terminal_solved 1`.

**Every chunk-7 gate passed on it**, and each for a reason that had nothing to do
with the defect:

| gate | why it passed anyway |
|---|---|
| depth-1 suite 100% at m ≥ 5 | a depth-1 problem needs exactly one ply |
| budget identity | visits are counted whether or not they do work |
| sync vs batched, 16 cells | both paths were equally shallow |
| 2-seed + cross-process determinism | unaffected by depth |

That is the shape of the problem: the gates were chosen to catch specific
hazards — negation, budget drift, batching skew, hash-order leakage — and a
search that never descends is none of those. **No gate asked whether the search
searched.**

**Fixes:**

* Real descent. A simulation now walks from the chosen root action down through
  expanded nodes by the Gumbel-AZ non-root rule until it reaches an unexpanded
  action, expands it, evaluates and backs up. Measured after: `sims=48` on a
  branchy depth-5 SOLVE gives **49 nodes, max_depth 3**.
* `SearchStats` gains `max_depth`, so the shape of the tree is in every row.
  **`nodes` did not have to be added — it was already there**, populated by the
  chunk-7 code (`search.py:182,230` at `5b5fd61`), reading 6/6/6/6 in every stats
  row it ever wrote, asserted on by no gate. (Corrected 2026-08-14; the first
  publication of this finding said the field was gained. `ERRATA-chunk7.md` §3.)
* **The missing gate exists now** — `test_the_tree_deepens_past_one_ply` asserts
  more simulations build more tree, that the tree exceeds root-plus-children,
  and that node count scales with sims rather than with `m`.
* Batching moved to where it belongs: **across concurrent searches, not within
  one**. Pooling leaves inside one tree would change what the tree does, because
  simulation *k+1*'s selection depends on *k*'s backup — the old equivalence test
  was exact only because the old search had no selection to corrupt. Across
  trees there is no coupling, so parity stays an equality rather than a
  tolerance, and it matches AGENTS.md §8's prescribed shape.

**The general lesson, and it is not "add a test":** a gate suite assembled from
known hazards has a blind spot exactly the shape of *the component doing its job
at all*. Every chunk here has gates for how a thing can be subtly wrong; this is
the first case where the thing was grossly absent and every subtle-wrongness
check still went green. In force as house law, AGENTS.md §5, with the operational
rider that the **first** gate written for any component measures it doing its
central job.

**And one turn sharper, from the re-measurement (`ERRATA-chunk7.md` §2):** the
blind spot was not missing instrumentation. `SearchStats.nodes` was recorded in
every row chunk 7 ever wrote, it was correct, and it was flat at 6. No gate
asserted on it. A number nobody asserts on is not a gate — it is a comment that
happens to be computed.

**Where the re-measured evidence lives:** `ERRATA-chunk7.md` carries the full
chunk-7 gate table re-run under the fixed search, the pre-vs-post comparison that
settles which search produced the budget-identity figures (both — the quantity
does not discriminate), and the decomposed test delta.

---

## F-07 — Phase 1 was paying for Phase 2's envelope, 22× over

**Found:** 2026-08-14, chunk 8's required timing pre-flight, before any training
step was taken in earnest. Record: `runs/pilot_phase1_timing.json`.

`model.seq_len` is **512**, sized in chunk 6 from the *reachable*-state
distribution — correctly, because Phase-2 self-play reaches 295-token states
(F-05). `encode()` pads every state to it. But the Phase-1 supervision set's
`max_len` is **64**, so every batch was 8× wider than its widest content, and
attention is O(L²).

Measured at batch 128, same batch, same model, medians over 12 timed steps:

| | width | s/step | 10,000 steps |
|---|---:|---:|---:|
| as built | 512 | 23.8361 | **66.2 h** |
| cropped to content | 64 | 1.0922 | **3.0 h** |

**21.8×, and the outputs are bit-identical** — `torch.equal` on all three heads,
not `allclose`. Cropping the all-PAD tail changes nothing the network computes:
the trunk already masks PAD via `src_key_padding_mask`, `_masked_mean` pools over
`ne(PAD)`, and positions are indexed from zero, so rows `0..width-1` of the
position table are the same rows either way.

**Why the width is 64 is the interesting part, and it is three findings meeting.**
F-05: `add_both_sides` is the sole source of state growth. ROUND-01: optimal
derivations never apply it. Therefore every state on a BFS-optimal derivation —
which is every state in the supervision set — is bounded by the largest *start*
state. `max_len 64` is not a property of the sample; it is a consequence, and the
same mechanism that makes `seq_len 512` right for Phase 2 makes it 8× wasteful
for Phase 1.

**Do not "fix" this by lowering `model.seq_len`.** The envelope is right; the
width is a property of a *batch*, not of the model. `_crop_to_content` is in
`train.py`, gated by `test_cropping_padding_is_exact` on both polarities: the
crop must fire *and* the outputs must be bit-identical. A crop test that only
checked the outputs would go green on a crop that never cropped, which is F-06's
shape one layer down.

**Registered, not done (F-07):** the same crop is available to the search's evaluator
path, where batched leaves are also padded to 512 and reachable states are
usually far shorter. Not touched here — that is shipped chunk-7 code and this was
a chunk-8 pre-flight.

**What this says about the pre-flight itself.** F-03 recorded a projection that
was 5× optimistic. This one was 22× pessimistic before it was read, and the value
came from *splitting* the measurement rather than sharpening it: batch
construction is 1.2% of a step, the forward/backward is the rest. That split is
also the device answer — the tensor work is 98.8% of the step, which is exactly
what an accelerator addresses. AGENTS.md §8's "GPU idle while CPU-bound Python
saturates" is stated for *search* workloads and holds there; supervised Phase-1
training is not one, and the pilot distinguishes them by measurement rather than
by assumption.

---

## F-08 — The derivations walked through the instrument

**Found:** 2026-08-15, by amendment A1's census, before any training step.
Record: `runs/supervision_contamination.json`.

`train_100k`'s **problems** are disjoint from the frozen suites, and a test has
said so since chunk 5. Phase-1 supervision holds those problems' **derivation
states**, and an intermediate state of a deep problem can be, exactly, the start
state of a shallow suite problem. Solving `9x + (−28) = 44` passes through
`9x = 72`. Measured instances include `−4x = 28`, `−6x = −30`, `−7x = 14` — each
a `solve_in_1` problem verbatim, each sitting one step from the end of a par-2
derivation the model was about to be trained on.

**1,887 of 313,628 examples — 0.6017% — across 116 distinct states.**

| suite | colliding examples | | steps remaining | colliding examples |
|---|---:|---|---|---:|
| `solve_in_1` | 1,835 | | 1 | 1,835 |
| `solve_in_2` | 0 | | 2 | 0 |
| `solve_in_3` | 52 | | 3 | 52 |
| `solve_in_4` | 0 | | 4 | 0 |
| `solve_in_5` | 0 | | 5 | 0 |
| `solve_in_6` | 0 | | 6 | 0 |

**Those are the same table twice, and they agree exactly.** A state with *k*
steps remaining on a BFS-optimal path *is* a par-*k* problem, so it can only
collide with `solve_in_k`. The two counts were computed by different routes —
one keyed on suite membership, one on the `steps_remaining` array — and matching
1,835↔1,835 and 52↔52 is an independent confirmation that the keying is right,
not a restatement.

**Zero start states.** Every collision is an intermediate. The chunk-5
problem-level gate was correct and remains correct; what it could not do was
speak about states that did not exist when it ran.

**Why depth 1 dominates:** the depth-1 state space is tiny — `ax = b` and little
else — so 200 suite items catch 1.8% of 100,000 penultimate states. The rate
collapses with depth as the state space opens up.

**One measured fact that is not explained, recorded as unexplained.** 79,249
states have exactly 2 steps remaining and **none** of them matches any of the 200
`solve_in_2` problems, while depth 3 has 52 of 62,667. That is non-monotone and
there is no mechanism on offer for it. It is not a keying artifact: the weaker
goal-blind check (`dataset.py::expression_keys`'s definition) returns the
identical 1,835 and 52 with **zero** goal mismatches, so the goal dimension is
excluding nothing. Registered as an open question rather than rationalised.

**The removal, per A1's pre-stated rule** (≤1% → remove, re-digest, report):
1,887 removed, **311,741 remain**. Per-depth deltas — 0, −193, −475, −974, −245,
0 for depths 1–6 — sum to exactly 1,887. Depth 1 loses nothing because depth-1
problems contribute only their start state.

**A verification identity was deliberately broken here, and this is the notice.**
`313,628 = Σ(depth × count)` over F-04's histogram held to the digit, term by
term, and was the single number confirming one-example-per-derivation-step across
the whole set. It no longer holds, because a correctness fix removed rows from
the middle of derivations. The replacement is `311,741 = 313,628 − 1,887`, with
the per-depth deltas above. Without this note a later audit re-running the F-04
arithmetic finds a mismatch and reads it as a regression.

**The cost, stated rather than buried:** the removed states are overwhelmingly
penultimate states of exactly the shape the depth-≤2 solve gate measures. The
gate gets harder. That is the gate becoming honest, not the data becoming worse.

**The general lesson — and it is not "add a contamination test":** a derived
dataset inherits its parent's *provenance* but not its parent's *coverage*. The
derivation manufactures states that never existed as problems, so no inherited
property can speak about them. "Inherited status" answers *was my source clean?*
It can never answer *am I clean?* — and the two questions look identical right up
until a transform changes what the rows are. Where a dataset's rows are a
different **kind** of thing from its parent's, every guarantee must be re-earned
at the new granularity, not carried across.

The permanent gate is `test_no_supervision_state_is_a_suite_start_state`, with
`test_the_contamination_probe_can_find_a_collision` beside it — because a gate
that passes by finding nothing is indistinguishable from a probe that cannot find
anything (law 5 rider (a)). The census artifact is pinned to the digests it was
computed against, so it cannot outlive the bytes it describes.

---

## F-09 — The held-out set was 21% seen, and the split that was supposed to prevent it was working correctly

**Found:** 2026-08-15, immediately after F-08, by pointing the same census at the
train/eval boundary instead of at the suites. Record:
`runs/eval_independence.json`. **Open — the pre-stated rule says STOP.**

The Phase-1 held-out set is built from `eval_held_out`, 2,000 problems that a
chunk-5 test certifies disjoint from `train_100k`. Problem-level split hygiene,
done correctly, verified. Then the derivations were unrolled — and:

| | |
|---|---:|
| held-out supervision states | 6,570 |
| **also present in `phase1_train`** | **1,398 — 21.28%** |
| distinct shared states | 1,276 |
| by steps remaining | `{1: 906, 2: 288, 3: 196, 4: 8}` |

**This is F-08's mechanism aimed at something that matters more.** F-08 leaked an
instrument into training. This inflates a **DONE-WHEN metric**: the chunk-8 gate
is *held-out top-8 rule-site ≥ 0.90 on depth ≤ 3*, and depth ≤ 3 is exactly where
the overlap lives — 1,390 of the 1,398 shared states have three or fewer steps
remaining.

**Nothing was done wrong at the problem level.** Two disjoint problem sets can
still walk through the same intermediate state, because the state space at low
remaining-depth is small and both sets' derivations funnel into it. Disjoint
problems do not imply disjoint derivations, and no amount of care about the
former produces the latter.

> **CLOSED by ruling, 2026-08-15, recorded verbatim.** *The gate reads the
> unseen subset at the unchanged 0.90 threshold; both numbers publish with the
> inflation delta named; report unseen-≤3 n, and if n < 1,000,
> `eval_held_out_v2` is pre-authorized as a state-tested-at-birth frozen
> supplement, v1 untouched; option (c) rejected — funneling is structural,
> disjoint-by-construction selects unrepresentative problems and violates the
> frozen instrument.*
>
> **Applied.** Unseen-≤3 **n = 1,229**, which is ≥ 1,000, so `eval_held_out_v2`
> is **not triggered** and v1 stands unmodified. The published pair for gate 10
> is all-held-out **0.9782** and unseen **0.9699**; the **inflation delta is
> +0.0083** — the contaminated instrument reads 0.83 points high, and naming it
> is the point of publishing both. The held-out set keeps all 6,570 states; the
> exclusion is applied at *measurement* time, not by deleting data, so the same
> artifact serves both numbers forever.

**Not removed.** A1's threshold is pre-stated: ≤1% remove, >1% is structural and
is a joint ruling. 21.28% is emphatically structural, the census script encodes
the rule and refused, and the rule was fixed before the number was seen. The
options differ in what they do to the *instrument*, which is why this is not a
call to make alone:

1. **Deduplicate eval against train at state level.** 5,172 states survive, but
   they are systematically the deeper ones — the metric's depth mix shifts, and
   "depth ≤ 3" loses most of its mass.
2. **Report on the clean subset, keep the set whole.** The gate is declared on
   the 78.7% never seen in training; the contaminated number is reported beside
   it as the inflation measurement.
3. **Rebuild the held-out set to be state-disjoint by construction**, rejecting
   candidate problems whose derivations touch a training state.

**The lesson, and it is the sharp end of F-08's:** a split is only as fine as the
granularity of the thing it splits. `train_100k` and `eval_held_out` are disjoint
*as problem sets* and that guarantee is real — it simply does not survive a
transform that changes what a row is. **Every derived dataset re-opens every
question its parents had closed**, and the more useful the transform, the more
questions it re-opens.

---

## F-10 — The chunk-8 gate is passed by random initialisation

**Found:** 2026-08-15, immediately after the Phase-1 run returned `top8 = 1.0000`
at every depth in both the full and the unseen held-out sets. A number that
perfect is a claim about the metric, not about the model. Record:
`runs/gate_arithmetic_topk.json`.

The registered DONE-WHEN is *held-out top-8 rule-site ≥ 0.90 on depth ≤ 3*.
Top-k is ranked over the **legal** action set — an unmasked top-k would credit
actions the movegen refuses. So a state with ≤ k legal actions is a **certain
hit whatever the network outputs**, and the gate has a floor equal to the
fraction of such states.

| k | floor, depth ≤ 3 | untrained | trained | headroom | verdict |
|---:|---:|---:|---:|---:|---|
| 1 | 0.5241 | 0.6950 | **0.9782** | 0.4541 | discriminating |
| 2 | 0.7443 | 0.8681 | 1.0000 | 0.2557 | discriminating |
| 3 | 0.7575 | 0.9106 | 1.0000 | 0.2425 | discriminating |
| 5 | 0.8584 | 0.9914 | 1.0000 | 0.1416 | discriminating |
| **8** | **0.9897** | **1.0000** | 1.0000 | 0.0103 | **VACUOUS at 0.90** |

**The threshold sits below the floor.** 98.97% of depth ≤ 3 held-out states have
≤ 8 legal actions (94.05% over all depths; the legal-action distribution is
`{1: 1605, 2: 1066, 3: 975, 4: 573, 5: 791, 6: 580, 7: 484, 8: 105, 9: 391}`,
median 3, max 9). The gate cannot return less than 0.9897 for *any* network.

**Measured on both polarities, not argued.** An untrained network, random init,
seed 0, scores **top-8 = 1.0000 at every depth** — identical to the trained
model. The gate does not distinguish 5,000 steps of training from no training at
all.

**Why even the 391 nine-action states never miss**, which the floor alone does
not explain: the target's rank among 9 legal actions under the *untrained* model
is 2 in 381 of 391 cases and never worse than 4. The last-place exclusion that
top-8 performs therefore almost never lands on the target. The floor is 0.9897;
the achieved value is 1.0.

> **Erratum, 2026-08-15.** This finding first explained that row as "a random
> network already ranks the BFS-optimal action near the top." That sentence
> over-generalised a subpopulation into a claim about the whole metric, and the
> whole metric says otherwise: over all depth ≤ 3 states the untrained network
> scores **0.6950 against a uniform-random null of E[1/B] = 0.6803**. It is at
> chance, not near the top. The rank-2 concentration is real and confined to the
> 9-action states; the sentence generalising it was not measured.
>
> **Three hypotheses tested and rejected** (raised in review as "enumeration
> order appearing on both sides of the measurement"):
>
> | hypothesis | test | result |
> |---|---|---|
> | policy head is zero-initialised ⇒ constant logits | measure head init and logit spread | **rejected** — `rule_embedding` std 1.0030, logit spread over legal actions median 10.27, every value distinct on 1,512/1,512 multi-action rows |
> | ties ⇒ `argsort` falls back to index order | count exact ties | **rejected** — 0 rows with tied legal logits |
> | untrained ranking tracks enumeration order | top-scoring action == lowest-index legal action | **rejected** — 29.30%, near chance |
> | 391 rows are duplicate states, so it is one observation | count distinct `(identity_key, goal)` | **rejected** — 391 rows, **391 distinct states** |
>
> So the concentration is not a tie-break artifact and not over-counted
> evidence. **Tag: STANDING-UNEXPLAINED** — not superseded (the measurement holds
> as taken) and not shown to be an artifact (four candidate artifacts were tested
> and rejected). It stands open and measured: 381 of 391 distinct 9-action states
> put the BFS target at rank exactly 2 under a randomly initialised network. No
> mechanism is on offer.
>
> The cheap test that would discriminate, named but **not run** so the tag stays
> honest: re-measure the rank distribution across several init seeds. The
> untrained arm is a *single* random draw (seed 0), so the concentration may be
> one network's idiosyncratic interaction with site-position structure rather
> than anything general. Persisting across seeds ⇒ structural; varying ⇒ one
> draw. Until that is run, "standing-unexplained" is the whole claim. It does not affect the gate verdict — the floor and the
> untrained 1.0000 settle that independently — but it is the second unexplained
> regularity in this data (F-08's depth-2 zero is the first).
>
> **What review predicted and what happened.** The prediction was that the
> untrained baseline would fall toward E[1/B] once ties were removed. It is
> already there — 0.6950 against 0.6803 — because there were no ties to remove.
> The baseline was honest; the *sentence explaining it* was not. Item 4's
> single-BFS-path caveat is separately confirmed as measurably active: the
> target's position in legal enumeration order is `{1: 641, 2: 531, 3: 6, 4: 12,
> 5: 321, 6: 1}` over 1,512 sampled multi-action states — 42.4% at position 1
> against a uniform expectation near 25%. That is enumeration order on the
> *label* side, now measured rather than noted.
>
> **And one process note, because it is the same trap twice.** The first version
> of this diagnostic sampled `range(256)` — a prefix — and `phase1_eval` is laid
> out stratum by stratum, so all 256 rows were depth-1 with a single legal
> action and every statistic came back degenerate. That is exactly the mistake
> `build_phase1_data.py`'s `--limit` docstring records from F-03. A documented
> trap in this repository caught the person who had read the documentation.

**This is chunk 7's own instrument catching the opposite failure.** There, gate
arithmetic showed a 100% depth-1 gate was arithmetically *impossible* at m = 3
(P(sweep) = 1.337e-09). Here it shows a gate is arithmetically *guaranteed*.
Both are the same defect — a threshold placed without computing what the metric
can return — and only the first one announces itself by failing.

**The model is fine; the gate is broken.** top-1 on depth ≤ 3 moved from 0.6950
untrained to **0.9782** trained on the full held-out set, and **0.9699** on the
F-09 unseen subset. Both clear 0.90. Overall top-1 went 0.4791 → 0.9800. The
training worked, and it is the *gate* that failed to notice.

**Not repaired here.** The gate is a registered DONE-WHEN from
`experiment2_agent_plan.md`; replacing top-8 with top-1 is a *strengthening*, but
a registered gate is not mine to rewrite, and "the gate I passed was too easy so
I changed it" needs the same signature as "the gate was too hard". Both numbers
exist and both are recorded; the ruling is about the record, not about redoing
the work.

**The general lesson, and it is the third face of law 5.** Rider (a): the first
gate measures the component doing its central job. Rider (b): a number nobody
asserts on is not a gate. **Rider (c), proposed: a threshold nobody computed the
floor of is not a gate either — compute what the metric can return before
choosing where to draw the line.** A gate has two failure modes, unreachable and
unmissable, and the second one ships green.

---

## F-11 — A null baseline measured through a constant seed is not a null baseline

**Found:** 2026-08-15, building gate 11's stub-null row under rider (c). Record:
`runs/gate_phase1_search_m1.json`, `runs/gate_phase1_search_m5.json`.

The harness called `search(..., random.Random(0), ...)` — a **fresh** generator,
re-seeded to the same constant, for every search of every problem. The root
Gumbel draw is therefore a function of the *action count alone*, so every
5-action problem in a suite draws the same perturbation, considers the same
slots, and chooses the same one. Measured on six 5-action depth-1 problems:

```
fresh Random(0) per search        per-problem seed
  chosen slot 0  [8,8,0,0,0]        chosen slot 0  [8,8,0,0,0]
  chosen slot 0  [8,8,0,0,0]        chosen slot 3  [8,0,0,8,0]
  chosen slot 0  [8,8,0,0,0]        chosen slot 4  [8,0,0,0,8]
  chosen slot 0  [8,8,0,0,0]        chosen slot 2  [8,0,8,0,0]
  chosen slot 0  [8,8,0,0,0]        chosen slot 1  [0,8,0,8,0]
  chosen slot 0  [8,8,0,0,0]        chosen slot 1  [0,8,0,8,0]
```

Against flat stub priors this silently replaces the intended null — *uniform-random
action* — with a different one: *always the first legal action*. Same name,
different claim, and the substitute is **not** weaker. It is stronger, because
the first legal action is on an optimal path far more often than chance (F-10's
erratum measured the BFS target at enumeration position 1 in 42.4% of
multi-action states). The degenerate null therefore reads *high*, and a null that
reads high makes a gate look **more** vacuous than it is.

**What it cost, and what it corrected:**

| gate 11 config | null, constant seed | null, per-problem seed |
|---|---:|---:|
| 16 sims, m = 1 | 0.7450 | **0.7725** |
| 16 sims, m = 5 (registered) | **1.0000** | **0.9175** |

At m = 5 the difference is the whole verdict. Through the constant seed the null
read 1.0000 and gate 11 looked vacuous exactly like gate 10. Correctly measured
it reads 0.9175 against a 0.9500 threshold — **the gate discriminates**, narrowly,
and the first conclusion drawn from it was wrong.

**Caught by the inherited law, not by inspection.** *Instrument the trigger,
never trust identical numbers.* Two different `m` values returned byte-identical
rates — 160/200 and 138/200 at both m = 1 and m = 2. Nothing else was suspicious;
the numbers were plausible, the code read correctly, and the gate would have
shipped. The only signal was that two configurations which must differ did not.

**The general lesson:** rider (c) says the null is a *run*, not an estimate. This
adds the other half — **a null is a run over the randomness it claims to average
over.** A stub with flat priors has no signal of its own, so *all* of its
behaviour comes from the draw; seeding that draw to a constant does not make the
null reproducible, it makes it a different policy. Reproducibility belongs in the
*seed schedule* (per problem, per step, recorded), never in seeding every draw
identically.

Fixed: the search rng is seeded per `(problem index, step)`, and `verify`'s
equivalence rng is seeded separately so both arms are judged by the same draws.

**A design tension this exposed, recorded because both sides are house law.**
Item 9's gate arithmetic requires `m ≥ B_max = 5` so the winning action is
*certainly considered* — that is what makes a 100% gate reachable rather than
arithmetically impossible (chunk 7's lesson). Rider (c) requires the gate to
*discriminate* — measured value meaningfully above null. These pull opposite
ways, and the two measured points show it:

| m | null (stub) | threshold | measured | null-to-threshold margin |
|---:|---:|---:|---:|---:|
| 1 | 0.7725 | 0.95 | 1.0000 | 0.1775 |
| 5 (registered) | 0.9175 | 0.95 | 1.0000 | 0.0325 |

Raising `m` hands more of the work to the search and less to the prior, so the
null rises toward the threshold. At `m = 5` the margin is 0.0325 — the gate
still discriminates, but thinly, and a slightly better null would erase it. Two
points is not a trend and no monotonicity is claimed here; the *direction* is
measured and the mechanism is plain.

**Neither law is wrong.** They are answering different questions — *can this gate
be passed at all?* and *does passing it mean anything?* — and a gate needs both
answered. The practical consequence for later chunks: when a search budget makes
a gate reachable, check what it did to the null in the same breath, because the
same knob moves both.

---

## F-12 — Tier 1 held at every width; tier 2 was exceeded at every width, and the width stratification is not what caught it

**Found:** 2026-08-15, chunk 9 part 0b, against tolerances frozen at `abc765c`
while the venv was still CPU-only. Record: `runs/gpu_equivalence_smoke.json`.

**The gate passed.** CPU vs GPU (`torch 2.13.0+rocm7.2`, gfx1151), fp32, 512
states stratified across four width buckets:

| | |
|---|---:|
| legal-masked argmax identical | **512 / 512 (1.000000)** |
| legal-masked top-8 set identical | **512 / 512 (1.000000)** |

**The diagnostic tier was exceeded**, on the channel the declaration priced most
tightly:

| quantity | measured | declared bound | |
|---|---:|---:|---|
| policy logits | **3.721e-03** | 1e-3 | **EXCEEDED (3.7×)** |
| value probabilities | 3.779e-05 | 1e-4 | within |
| steps head | 3.204e-04 | 1e-3 | within |

Pre-declared disposition, applied without amendment: *"if tier 2 is exceeded
while tier 1 holds, that is a finding, not a failure — the number gets recorded
rather than the bound quietly widened."* **The bound is not widened.** 3.721e-03
against logits spanning ~10.3 is 3.6e-4 relative; the declared 1e-3 absolute
(~1e-4 relative) was an estimate of fp32 cross-device divergence for a 6-layer
transformer, and it was optimistic by 3.7×. The estimate was wrong; the gate was
not.

**And the part that corrects my own reasoning.** Amendment A1.2 added width
stratification on the argument that L = 64 and L = 300 exercise different kernel
paths, so equivalence proven narrow might not hold wide. Measured per bucket:

| width bucket | n | max abs Δ logits | argmax agreement |
|---|---:|---:|---:|
| 1–64 | 128 | 2.759e-03 | 1.000000 |
| 65–128 | 128 | **3.721e-03** | 1.000000 |
| 129–256 | 128 | 3.660e-03 | 1.000000 |
| ≥ 257 | 128 | 3.024e-03 | 1.000000 |

**Divergence does not track width.** Every bucket exceeds; the *narrowest*
exceeds at 2.759e-03; the maximum is in 65–128, not at the wide end. A
Phase-1-only sample would have exceeded the bound too, so **the width
stratification is not what surfaced this** — the tight bound is. A1.2 closed a
real blind spot in principle (a property proven for a set that is about to grow),
and it happens not to be the safeguard that mattered here. Saying which safeguard
actually caught a thing is worth more than claiming the credit for the one that
was argued for hardest.

**bf16, informational and not gated** (AGENTS.md §4.5 prescribes it for training):
argmax agreement **0.998047** — 511 of 512, one disagreement — at max abs Δ
**1.369e-01**. So bf16 is ~37× noisier than fp32 cross-device divergence and
still agrees on all but one decision. Recorded because "we did not test it" and
"we tested it and it was 0.998" are different states.

**The null.** `rule_embedding.weight[0] += 1.0` on the GPU copy: max abs Δ
**2.057e+01**, four orders above the 1e-3 detection bound — **DETECTED**. Its
argmax agreement was **0.314453**, recorded and not asserted per A1.1. Worth
noting against the original declaration: the enlarged perturbation moved 68% of
decisions, whereas the retired 1e-2 single-scalar version was the one at risk of
moving none. Enlarging it was the right call for a reason that turned out to be
demonstrable rather than merely prudent.

**What the gate now licenses.** Every chunk-8 number was measured on CPU. Tier 1
holding at 512/512 across all four width buckets means the device change is
invisible to every gate this project has — the decisions are identical, and the
margins are wide enough that 3.7e-3 of logit noise cannot reach them.

---

## F-13 — Two currencies in one loop: the tree scored solved-flat while training scored z-vs-par

**Found:** 2026-08-15, by a **passing** assertion in the ring's `root_q` sign
test. Ruled and fixed before the runner backed up anything.

`_leaf_outcome` returned **`1.0` for any solve**, regardless of step count. So a
depth-1 problem solved in one step — which ties BFS-exact par, and is therefore
`z = 0`, a **draw** — reported `root_value == 1.0`. The search was optimising
*solved-or-not*; the training target is *z against par*. Two currencies inside one
loop, which spec §6's one-currency clause forbids between the loop and the ladder
and forbids more strongly between the search and the reward.

**The consequence, stated as the mechanism rather than the symptom:** a flat `+1`
makes the searcher **indifferent between an over-par solve and an under-par one**.
It cannot race. The par game's entire premise dies at the backup rule, and every
test stays green, because nothing was asserting on the *scale* — only on the sign.

**Proven ancestor.** Chess's FINALE bug: several proven wins tied at `q̂ = +1` and
the engine played mate-in-5 over mate-in-2. Here it is reborn as
**par-indifference**, aimed at the one quantity this campaign measures.

**The fix (ruled):** the in-tree terminal value is z at the leaf.
`total_steps = steps_taken + tree.depth[leaf]`, scored against `problem.par`:
`+1` under, `0` at, `-1` over. The cap is `-1`, identically to going over par
(plan chunk 3), and `StateTooLarge` keeps its counted-terminal-loss `-1`. `search`
and `run_batched` take `steps_taken`, because a leaf's worth depends on the whole
line's length, not the tree-local depth. Everything needed was already in hand:
par rides on the problem, pool par resolves before the search starts. **A
value-function change, not a plumbing one.**

Re-pinned on three fixtures, exactly:

| fixture | `root_value` |
|---|---:|
| solve beats par (fixture par 5, solves in 1) | **+1.0** |
| solve **at** par (depth-1, par 1) | **0.0** |
| line runs over par (`steps_taken = par`) | **−1.0** |

**Two consequences worth their own lines.**

1. **A neutral evaluator now ties with an at-par solve.** Both read 0.0, so a
   solve no longer stands out *by value* against `uniform_stub`. The chunk-7
   depth-1 gate read `values.max() >= 1.0`; it is re-expressed as
   `stats.terminal_solved > 0`, which is what its arithmetic was always about —
   *was the winning action considered and found* — and is independent of the
   value scale. Both polarities still hold at m = 5 and m = 3.
2. **`root_value` was `max(the root's own evaluation, best line)`**, so a neutral
   root prior floored it at 0.0 and an all-losing root reported 0.0 rather than
   −1.0. Registered as an open question, then **ruled: the family identification
   was right, and it is fixed.** See the amendment below.

**Measurements this invalidates.** Anything whose numbers came from search
*choices* under the old scale must be re-measured: the chunk-7 gate table's
descent/determinism rows and chunk 8's gate-11 solve rates. The gates' *definitions*
are unaffected — gate 11 reads the checker, not a value — but the search now
prefers shorter lines, so the numbers are not carried over by assertion.

### Amendment, 2026-08-15 — own-eval scoped, and proofs propagate

Ruled after the fix above. Own-eval was doing two jobs and only one is legitimate:

* **Legitimate:** a proxy for actions *not yet tried*. While a node has unexpanded
  actions, its own estimate stands in for what those might yield, and taking part
  in the max is epistemically sound.
* **Illegitimate:** a permanent floor over *explored, proven* lines. Once a node's
  legal set is fully expanded, own-eval represents nothing, and the node is worth
  `max(children)` alone.

So: **own-eval participates in the backup only while unexpanded actions remain at
that node**, and **proofs propagate** — a terminal's z is a proof, and a fully
expanded node whose children are all proven is itself proven at `max(children)`.
That is the single-agent MCTS-Solver, one paragraph in a max tree.

**Why it was worth ruling rather than registering.** With a *trained* evaluator
the floor means `root_q` can never report a position as worse than the net already
believed — the net's belief floors its own MSE target. That is self-referential
optimism aimed squarely at the half of the blend the currency ruling exists to keep
from fighting the other half. Same clock as the currency: cheap before the first
row, expensive after.

**Same defect family, named:** chess's FINALE bug was proven evidence mishandled
at action *selection*; this is the same thing at value *aggregation* — a stale
prior outranking a proof. The project has now paid for that family twice.

**Consequences in the tests, both named rather than absorbed:**

* The over-par fixture is **sharpened**: it now uses the *neutral* stub at
  `m = 16`, so the −1 must emerge from proofs alone with no cooperation from the
  evaluator. It is therefore the regression test for the rule itself — if
  own-eval leaks back into a fully expanded node's max, that fixture goes
  green-to-red.
* `test_backup_never_lowers_a_value` is **removed, and its premise was the
  defect.** It asserted that a later worse line could never demote an earlier
  value. That monotonicity was a *consequence* of max-accumulating into a floor
  the node could not escape. Under the ruled semantics a proof is allowed to lower
  a node — and must be, or a trained net's optimism could never be contradicted by
  evidence. Replaced by
  `test_a_proof_may_lower_a_node_below_its_own_estimate`, which asserts the
  opposite property on purpose.

**The lesson, and it is about the test rather than the code:** this defect was
found by an assertion that **passed**. `root_value == 1.0` pinned the semantics
loudly enough that the semantics could be reviewed. An approximate pin — `> 0.5`,
or "near the win value" — would have been satisfied by both the right answer and
the wrong one, and the wrong one would have shipped. **Pin exactly; an exact pin
that is wrong is a question, and an approximate pin that is wrong is a silence.**
Fourth occasion a passing assertion has been the thing that exposed a defect.

---

## F-14 — The untrained value head was inert under solved-flat and is harmful under z

**Found:** 2026-08-15, re-running the F-13-invalidated measurements. Caught by the
**zero-value ablation**, which was a diagnostic nobody expected to move. Record:
`runs/gate_phase1_search.json` at git `8de6fb1`.

Gate 11 re-run under the ruled semantics (16 sims, m = 5, depth ≤ 2):

| arm | `solve_in_1` | `solve_in_2` | depth ≤ 2 |
|---|---:|---:|---:|
| **trained** (value head live) | 168/200 = 0.8400 | 200/200 | **0.9200** |
| stub null | 169/200 = 0.8450 | 154/200 | 0.8075 |
| **zero-value ablation** | **200/200** | **200/200** | **1.0000** |

**Gate 11 as registered: threshold 0.9500, measured 0.9200 — SHORT.** Not
weakened, not re-thresholded. And the trained arm is now *indistinguishable from
the null* on `solve_in_1` (0.8400 against 0.8450 — it is marginally **worse**).

**The mechanism.** The W/D/L head is trained at loss weight **0** by design
(spec §5): its output is an untrained head's noise. Under the old solved-flat
scale a terminal solve backed up `+1.0` and dominated any noise the head could
emit, so the head was invisible. Under the z scale an **at-par solve is 0.0**, and
noise around zero routinely exceeds it — so the search now prefers a
noisily-optimistic unexplored line over a **proven** at-par draw. Zeroing the head
removes the noise, every child ties at 0.0, and selection falls back to priors and
the Gumbel draw, which finds the solve.

**Gate 12's earlier reading was an artifact, and this is its erratum.** Chunk 8
reported the ablation as *inert* — 1.0000 against 1.0000 at every depth — and
concluded that the weight-0 head neither helped nor hurt. That conclusion was
true only of the solved-flat scale it was measured on. **The ablation was
measuring a quantity the scale had made unmeasurable.** It is now the arm that
separates a working search from a broken one by 8 points.

**This is not an argument against the currency ruling.** It is the ruling doing
what it was for: a flat `+1` was hiding the fact that an untrained value head
outranks proven draws, and the par game cannot race while that is true. The defect
was always there; the old scale made it invisible.

**Registered decision, not taken here.** Phase 2 wakes the W/D/L head on real z,
so the head is only noise at *iteration 0* — but iteration 0 is where self-play
data comes from, and a search that avoids proven draws generates a corpus that
teaches it to keep avoiding them. Options, none of them mine to pick: (a) force
`value = 0` in search until the head has trained on real z, with a config key and
a declared switch-over; (b) blend the head in by a schedule; (c) accept the
degradation for iteration 0 and measure how fast it clears. **Gate 11 stays SHORT
in the record until this is ruled.**

**And the small lesson beside the large one:** the chunk-7 gate table re-runs
*identically* — 90.0% / 100% / 100%, budget identity, determinism, StateTooLarge
27, descent 7/17/32/49, parity EXACT — once its depth-1 gate is read from
`terminal_solved` rather than from a value threshold. The chunk-7 gates were never
scale-dependent in what they *meant*; only in how one of them was *written*. Two
of my own scripts read the value threshold and only the test was fixed in the
first pass — a re-expression applied to a test and not to its script is the same
defect as a summary drifting from a derivation.

---

## F-15 — The loop had no training phase, and the one rule that would have trained it deadlocks

**Found:** 2026-08-15, answering a direct question from review: *has the training
phase ever executed inside the iteration loop?* Verified rather than recalled.

### Part 1 — it was never wired

| checked | result |
|---|---|
| anything reading the ring for training | **nothing** — every consumer is a writer (`runner`) or a persister (`resume`) |
| `train.value_q_mse_weight` read outside `config.py` | **nowhere** — the blend the brief calls "on from day one" has no implementation |
| what `train()` consumes | a `SupervisionSet` — Phase-1 memmap, not a ring |
| optimizers in the codebase | `train.py` (Phase-1) and the timing pilot. That is all |

So the ring collected experience nothing consumed, and Phase 2's loop **played
episodes and never learned from them**. It is the enrollment pattern at the loop's
core: `composition()` sat in a report as a number nobody asserted on, and here the
whole training phase sat in a chunk report as a claim nobody asserted on. "Plumbing
proven end to end" was true and was not the same sentence as "the loop trains".

**And a claim of mine was false when I made it.** A chunk-8 commit message says *"a
dead config key is the `batch_leaves` hazard and is not repeated."*
`value_q_mse_weight` was already dead at that moment, at the loop's core. The
hazard was repeated; I asserted it was not; the assertion was not checked.

### Part 2 — the F-14 wiring, taken literally, cannot work in Phase 2

F-14 ruled *one declaration, two consumers*: while the head is untrained-on-z, the
loss masks it **and** the search contributes zero. In Phase 1 that is exactly
right — spec §5 leaves W/D/L at weight 0 because imitation data has degenerate z.
In **Phase 2** it is a deadlock, and the arithmetic is one line:

```
loss weight = value_loss_weight × value_contribution(head) = 1.0 × 0.0 = 0.0
  → the head receives no gradient while untrusted
  → its z-accuracy never leaves chance
  → the switch criterion never clears
  → live never becomes True
```

**A ratchet that can never be pulled.** The head cannot learn the thing the
criterion tests for, because the criterion's own precondition switches off the
learning.

**The only non-deadlocking reading**, implemented and flagged rather than adopted
silently: the declaration governs **what the search trusts**, not **what the loss
teaches**. In Phase 2 the head trains on real z at full weight — that is how it
becomes trained — while the search contributes zero value until the criterion
clears. Phase 1's weight-0 masking stays as it is, because there the reason is
degenerate z rather than untrusted-ness.

That preserves everything F-14 was for: the search still never reads an opinion
nobody earned (noise-as-signal stays closed), iteration 0 still generates a clean
value-silent corpus, and the bootstrap argument still holds. What changes is that
the head is *taught* meanwhile, which is the only way the door it guards can ever
open.

**Registered for ruling.** The literal wording gates both consumers; the
implementation gates one. That is a deviation from a ruled decision, so it is
named here rather than absorbed — and the deadlock is the argument, not a
preference.

---

## F-16 — A commit message reported a test count it had not measured

**Found:** 2026-08-15, immediately after the commit landed, by reading the tool
output the commit had scrolled past.

Commit `1cd3233` reports:

> `make lint test: 786 passed in 152.6s (was 753; +33, 0 removed).`

The run in that same command reported **`772 passed in 152.52s`**.

| | claimed | actual |
|---|---|---|
| tests passing | 786 | **772** |
| delta from 753 | +33 | **+19** |
| removed | 0 | 0 (correct) |
| wall clock | 152.6s | 152.52s (correct) |

The +19 decomposes exactly: **17** in `tests/test_ladder.py`, **2** in
`tests/test_arms.py` (the branching-premise pair). Nothing is missing; the total
was simply wrong.

**Mechanism, which is the part worth keeping.** The commit message was written
into the *same shell invocation* as the test run — `make lint test && git commit
-m "…786 passed…"`. The number therefore had to be authored **before** the
measurement existed. There was no moment at which it could have been checked
without restructuring the command.

This is the four-tuple rule's own failure mode: **a measured value asserted
before it was measured**, in the sentence claiming compliance. Rider (c) says a
threshold nobody computed the floor of is not a gate; the same sentence read
forwards says a *measured* slot filled from expectation is not a measurement.
`make golden` and `make lint test` both compute the number honestly — the defect
is entirely in the reporting path, which had no such discipline because nobody
had thought of a commit message as a place where a number gets asserted.

**Standing correction to method, effective now:** the test count in a commit
message is pasted from a run that has already returned. Compose the message
*after* the run, never in the same command as it.

The commit stands unedited — verdicts do not retro-edit, and a commit message is
a verdict on what was done. This finding files beside it, and the correction is
also recorded in `8f30cb7`'s message so a reader following the history rather
than the findings file meets it too.

---

## F-17 — The ladder shipped a second identity normalizer for a job that has one

**Found:** 2026-08-15, while wiring `pair_scores` persistence, by reading
`dataset.problem_key`'s docstring for an unrelated reason.

`ladder.problem_key_of` was written as:

```python
return ",".join(str(t) for t in identity_key(problem.expr)) + f"|{problem.goal}"
```

That is `(identity_key(expr), goal)` — the **census** key. The project already
has **the** pairing/dedup key, `dataset.problem_key`, documented as such:
`encode_state(goal, expr, target)`, the canonical token sequence including the
goal prefix. The two are not interchangeable, and the difference has a direction:

| key | merges | correct for |
|---|---|---|
| `(identity_key(expr), goal)` | `3x + 6 = 21` with `6 + 3x = 21` | **contamination** — a model that saw one has effectively seen the other |
| `dataset.problem_key` | nothing that canonicalises apart | **pairing** — two rows of an instrument must never become one |

**The hazard, concretely.** `pair()` builds `{s.problem_key: s for s in scores_b}`.
Under the loose key, two distinct paired-set rows that canonicalise together
collapse to one dict entry; arm A's score for the first would be differenced
against arm B's score for the *second*, and the count check would still balance.
A silent mis-pairing, in the function whose entire job is to not do that.

**It was not producing a wrong number yet.** No paired set had been frozen, and
`generate.py` dedups suites on the strict key, so no existing file contains a
colliding pair. The defect is structural — a second normalizer that will drift —
not an active miscalculation, and saying otherwise would overstate it.

**What the test did.** `test_problem_keys_go_through_the_shared_normalizer`
asserted that distinct problems get distinct keys. That passes under *either*
key. The test **named** the law in its title and **checked** something weaker
than the law — rider (b)'s shape at the test layer: an assertion that happens to
be computed. It has been replaced by one that asserts the delegation itself.

**Fixed:** `problem_key_of` delegates to `dataset.problem_key` and only renders
it for the JSONL column. `pair()` additionally **refuses duplicate keys within an
arm**, so a collision from any future source is loud rather than silent.
`pairedset.freeze` refuses a duplicated problem at write time, which is the
earlier of the two places to catch it — a duplicate in the instrument mis-pairs
every pass ever run on it, not only the one that notices.

---

## F-18 — At the paired set's boundary, the problem-level census caught nothing and the state-level census caught eleven

**Measured:** 2026-08-15, `runs/paired_census.json`, seed 0, 400 candidates drawn
from `runs/data/eval_held_out` through `sample_indices`.

| level | reference | keys | candidates hit |
|---|---|---|---|
| problem-level | `runs/data/train_100k` | 100,000 | **0** |
| state-level | `runs/data/phase1_train` | 259,053 | **11** |
| **state-level beyond problem-level** | | | **11** — all of them |

**2.75% of candidates**, and the problem-level test would have passed every one.
`clean` = 389, which is the frozen paired set.

This is F-08's mechanism, re-measured at a new boundary and confirming L7 in the
direction L7 predicts: the rule "census the paired set against training" carried
across from the dataset boundary keeps its wording and loses its justification.
At the dataset boundary the problem-level question was the load-bearing one. Here
it answers *zero* and the state-level question does all the work — because
`eval_held_out` was **built** disjoint from `train_100k` at the problem level, and
nothing was ever built to make it disjoint from `phase1_train`'s *derivations*.

**Note what a single-level census would have reported:** "0 collisions against
training, clean." True, complete-sounding, and wrong about the thing that
matters. The count is not large; the point is that the level that caught
everything is the one a carried-over rule would have dropped.

`state_level_beyond_problem_level` is a **premise-dependent zero** in the schema
sense — it may legitimately read 0 on a future set, and 0 there means the
derivations happened not to pass through those states, not that the question was
skipped. It is reported explicitly for that reason.

---

## F-19 — The self-match null was denominated in solved-flat while the comparison it was a null for was denominated in z

**Found:** 2026-08-15, reading the first smoke pass's own output — S6 and S7 side
by side in `runs/ladder_smoke_result.json`. Both had already returned PASS.

`ladder.self_match` took a caller-supplied `play(problem, cfg, seed) -> (score,
steps)`. The smoke pass supplied:

```python
return float(result.solved) * 2 - 1, result.steps      # solved-flat, ±1
```

and the resulting `PairScore` was tagged `CURRENCY_Z`. So S6 — **the null run**,
the thing rider (c) requires to be a *run* rather than an estimate — measured
solved-vs-unsolved, while S7's metric was mean paired difference **in z**. The
null was in the wrong currency for the number it was the null for.

**Why nothing downstream could catch it.** Solved-flat scores land in {−1, +1}
and z lands in {−1, 0, +1}. Both are `CURRENCY_Z`-shaped, both pair, both
bootstrap, and both give exactly 0 under a deterministic self-match. Every
guard the currency ruling installed passed, correctly: the ruling separates
*z from solve-vs-budget*, and this was neither — it was a third denomination
with no name, inside the currency that has one. F-13's shape (the tree scored
solved-flat while training scored z-vs-par) reappearing one layer up.

**What it cost, measured.** The contrast case is the detector's own
non-vacuousness check, and it is where the error showed:

| self-play contrast, of 389 problems | non-zero differences |
|---|---|
| solved-flat (as run) | **1** |
| z (corrected) | **79** |

A contrast of 1/389 passes `any(d != 0)` and is a hair from vacuous — the
self-match null was very nearly a detector that reports zero for every
configuration, which is the exact failure its contrast case exists to exclude.

**Fixed structurally, not by correcting the call site.** `self_match` now takes
an arm's `play` directly and **computes z itself** from the problem's own par;
the caller no longer supplies a score and so cannot supply a currency. No
validation downstream could have distinguished the two denominations, so the
choice was removed rather than checked. A problem without par raises rather than
scoring 0 — absence does not become a number here either.

**Both verdicts stand.** The first smoke pass's S6 PASSed on the terms it was
written on; this finding files beside it with the superseded numbers, and the
re-run's S6 carries the corrected contrast of 79.

### F-18 — disposition, 2026-08-15: **REMOVED PRE-FREEZE**

Ruled by the principal on the two precedents; the preferred branch applies
because its condition holds. **`f0b10fd1…` is the post-removal digest**, and that
is verified rather than asserted from the code path: the frozen set was
re-censused at both levels *after* the freeze, and the digest on disk was compared
to the one in `ANCHORS`.

| | |
|---|---|
| digest on disk | `f0b10fd113e32013b1088590fea78cbb300dd5a7876fc08357a389e3addbe1ec` |
| digest in `runs/ANCHORS.sha256` | **identical** |
| problems | 389 |
| problem-level hits, post-freeze | **0** |
| state-level hits, post-freeze | **0** |

`runs/paired_census_postfreeze.json`. The eleven were dropped before the file
existed, so `smoke_v1` was **born clean** and the F-09 pattern does not attach:
there is no second number for a pass to report, because there is no contaminated
remainder inside the instrument.

The measured cost of cleaning was **11 rows of 400 (2.75%)**, which is what makes
this the right branch rather than the convenient one — frozen-with-annotation
buys nothing when removal is that cheap, and it would put a caveat on every
future pass forever to save eleven problems once.

### F-18 — structural completion

`pairedset.census` now **raises on an empty `problem_sources` or `state_sources`**.
A single-level census is not a smaller census; it answers a different question in
the voice of this one. The refusal quotes F-18's own numbers, so the next reader
meets the reason at the point of failure rather than in this file.

Both polarities are tested. There is no other paired-set census path: `census` is
the only entry point, and it is now total over the two levels.

### F-16 — repeat, 2026-08-15, and the rule replaced by a mechanism

**Written after the repeat, and it is a repeat of the thing this entry is about.**

The commit that filed F-18's disposition (`chunk 10 closes: …`) carries
`make lint test: 811 passed`. That number was written into a `git commit`
that ran *before* the `pytest` in the same shell block. It happened to be
correct — the run returned 811 — and the method was exactly the one F-16
forbids, **one commit after F-16 was accepted**.

The diagnosis the principal kept ("a number authored before its measurement
exists is a prediction wearing a report's clothes") was right and was not enough,
because the fix was a **rule of composition order** and the ergonomics push the
other way: message and run are convenient to send together, and a rule that
fights convenience loses to it every time the convenience recurs.

**So the number stops being authored.** `tests/conftest.py::pytest_sessionfinish`
writes `runs/testcount.json` at the end of every session — collected, passed,
failed, exit status — and a commit quotes or includes that file. Helpers prevent,
documentation warns; this had been documentation.

The record also carries **`whole_suite`**, because the obvious near-miss is a
targeted run's total reading exactly like a full-suite total. That is F-16's
failure in a new costume, and it is closed at the same time rather than left for
the next repeat to find.

---

## F-20 — The anchor is saturated against exact par, and so is every primary denominated in it

**Measured:** 2026-08-15, `runs/chunk11_part0_result.json` (Part 0, protocol
frozen at `7af32f4`) and `runs/chunk11_ceiling.json` (arithmetic derived from it,
nothing re-typed).

### M-A — the anchor's suite z-composition, 1,200 problems, eval protocol

| suite | solved | `<0` | `0` | `1` | `2`–`6+` | seconds |
|---|---|---|---|---|---|---|
| solve_in_1 | 200/200 | 0 | 200 | 0 | 0 | 25.04 |
| solve_in_2 | 200/200 | 0 | 200 | 0 | 0 | 79.00 |
| solve_in_3 | 200/200 | 0 | 200 | 0 | 0 | 182.32 |
| solve_in_4 | 200/200 | 0 | 199 | 1 | 0 | 290.38 |
| solve_in_5 | 200/200 | 0 | 194 | 6 | 0 | 405.16 |
| solve_in_6 | 200/200 | 0 | 200 | 0 | 0 | 517.69 |
| **pooled** | **1200/1200** | **0** | **1193** | **7** | **0** | **1499.65** |

0 capped, 0 stuck. **T = 7 over par (0.58%)**, all of them by exactly one step,
against a pre-registered threshold of **36**. The rule selects **rung
trajectory**.

The `<0` bin is 0 and is reported: definitional, not measured-and-absent.

### M-B — rung baselines on `smoke_v1` (389 problems, digest `f0b10fd1…`)

| | |
|---|---|
| model z histogram | **+1: 0, 0: 387, −1: 2** |
| model − greedy, mean | **0.228792**, CI **[0.187661, 0.272494]**, excludes zero |
| synthetic Elo, model vs greedy | 0.614396 |
| sympy (budget lane) | 251 played, **251 solved**, 138 declined, v1.14.0 |
| seconds by arm | model 246.434, greedy 0.352, sympy 0.114 |

### The part the pre-registered rule could not ask

The rule asked whether the **suite tail** could carry CI separation. It could not
ask the same question of the branch it falls back to, because the anchor's
paired-set z composition did not exist yet. M-B produced it, so the question is
now answerable — and the answer is that **the fallback is saturated too**:

```
mean model z now          = -2/389   = -0.005141
mean model z CEILING      =  0.0          (par is BFS-exact; +1 is impossible)
max possible movement     =  0.005141
CI half-width, model-greedy = 0.042417
movement / CI half-width  =  0.1212
```

Greedy is deterministic and fixed, so the paired difference moves **only** as the
model's z moves. Its entire remaining headroom is **12% of one CI half-width** —
an eighth of the distance it would have to travel to separate. The budget lane is
worse: sympy solves 251 of 251, so its score cannot move at all, and it cannot be
differenced against the z lane by construction.

**Every candidate primary denominated in z-against-exact-par is at or within
noise of its ceiling before iteration 0.** This is not a finding about the model.
It is the instrument's dynamic range being exhausted at the starting line: the
anchor is at par on 1,193/1,200 suite problems and 387/389 paired problems, and
nothing can score better than par where par is exact.

The principal's catch generalises further than it was aimed. "Beat-par rate
measures the impossible cell" was said of the *primary*; the same exactness makes
the **at-par** metric a ceiling too, because at-par is the top of the scale rather
than the middle of it. The chess succession lesson arrives here **by arithmetic
before the campaign**, which is what Part 0 was run to make possible.

### What has headroom, and what does not — for ruling, not for adoption

Not decided here. The measurement licenses these observations and no choice:

* **Search economy** has visible headroom and is already instrumented: the anchor
  reaches par at `sims = 48`. Whether a trained model reaches par at `sims = 6`
  is a question with a wide, movable range, and the wall clock says it matters —
  one eval pass over the suites costs **1499.65s on CPU**, so a 20-iteration
  campaign spends **≈ 8.3 hours** in eval alone at the current setting.
* **Harder problems** — the suites top out at par 6 with `step_cap = 24`. Nothing
  in the generator forbids deeper, and depth is where the tail lives (the only 7
  over-par problems sit at par 4–5).
* **Neither of the two currently frozen instruments** can carry a primary, and
  that is a statement about the instruments, not about what should replace them.

**Registered for ruling.** A new primary is a change to the campaign's criterion,
which is the principal's call, and PREREG-m1 does not freeze until it is made.

---

## F-21 — The scripted solver's calibration was a ≤6-band property, and both of its clauses die at stratum 7

**Measured:** 2026-08-15, `runs/chunk11_part0c_mint.json`, against the chunk-5
calibration in `runs/par_delta.json`.

L7 at the scripted solver's own boundary. The calibration was carried forward as a
sentence — *"never more than one step above exact"* — and the boundary is exactly
where that sentence had to be re-checked rather than re-applied.

### The chunk-5 calibration, quoted from the record rather than recalled

`runs/par_delta.json`, depths 1–6, 1,200 problems:

| gap `scripted − bfs` | count |
|---|---|
| 0 (optimal) | **1176** |
| 1 | **24** |
| ≥2 | **0** |

**1176/1200 = 98.0% optimal, maximum gap 1.** And by goal, the clause that matters
most here:

| goal | gap 0 | gap 1 |
|---|---|---|
| EVALUATE | 450 | 7 |
| SIMPLIFY | 235 | 17 |
| **SOLVE** | **491** | **0** |

**On SOLVE the scripted solver was 491/491 — perfectly optimal — through depth 6.**

### Stratum 7, 20-problem sample, BFS at cap 6

| gap | count |
|---|---|
| 0 (optimal) | **0** |
| 1 | **17** |
| 2 | **3** |

**0/20 optimal, maximum gap 2.** Every mid-strata problem is a SOLVE problem, so
the comparison is like-for-like and it is the sharpest possible form of the
boundary: **100% optimal at depth ≤6 → 0% optimal at 7.** Both clauses of the
carried sentence fail, and the "never more than one step" clause fails too.

### Why this is not a defect

The floor was always declared provisional — `solver.py` says so, `scripted_par`
is documented as "the floor this solver can certify", and `scripted_par_delta`
exists precisely to quantify the gap. This is **the first measurement of how
provisional**, taken at the first boundary where it mattered.

What would have been the defect is quoting the ≤6 calibration at stratum 7. The
sentence is now fenced: **the 98.0% / max-gap-1 / SOLVE-perfect calibration is a
property of depths 1–6 and does not survive the boundary.** Anyone citing it
outside that band is citing a number that has been measured false there.

The mid-strata suites are unaffected as instruments — a bigger gap means a
*more* beatable floor, which is what they were minted for. What changes is that
nobody may assume the floor is nearly tight.

---

## F-22 — The driver never left train mode, so every search it ran had dropout live

**Found:** 2026-08-17, from the resume gate. The gate compares a killed-and-resumed
run against an uninterrupted one on every row column except three wall-clock
fields, and it failed on columns that have nothing to do with resuming. The
suspect was the comparator; the cause was that **the driver was not deterministic
at all**, and two identical uninterrupted runs did not agree either.

### The mechanism

Three facts, each true on its own and harmless alone:

1. `load_checkpoint` returns a model in **train** mode. `nn.Module` defaults to
   `training=True` and nothing in the loader changes it.
2. `train_on_ring` calls `model.to(device).train()` and returns without
   restoring. It was written as a step, and it behaved as a mode switch.
3. `campaign.run` built its search evaluator from that same object.

So iteration 0 played the anchor in train mode because the loader left it there,
and every iteration after that played it in train mode because training put it
back. `cfg.model.dropout = 0.1`, and dropout in train mode is live: **the policy
priors were sampled through a stochastic forward pass in every search the driver
ever ran.**

### Measured

`entropy_prior_step1_*` from the driver at golden config, three fresh processes,
identical seeds and identical config:

| | `entropy_prior_step1_start` | `entropy_prior_step1_reached` |
|---|---|---|
| before, run 1 | 0.035472 | 0.034999 |
| before, run 2 | 0.038375 | 0.023913 |
| before, run 3 | 0.042216 | 0.034544 |
| **spread** | **0.006744 (19% of the low value)** | **0.011086 (46%)** |
| after, runs 1–3 | 0.035402 | 0.031519 |

`solved=12` and `pool_par_fraction=0.166667` in all six runs. The integer
counters were stable throughout, which is why this survived: **the columns that
moved were the float-derived ones, and the ones a person checks first were the
integers.**

### The funnel-baseline consequence, in full

`entropy_prior_step1_start` is not an incidental column. The funnel signature's
thresholds are expressed as a *fraction of* it, so a defect in that column moves
the measurement and the threshold together, in the same direction — which is the
form of error that a threshold check cannot report, because the check is
denominated in the thing that is wrong.

Had M1 run as written: iteration 0's funnel columns would have carried one
dropout sample, iteration 5's another, and the drift between them would have been
read as *the policy's prior distribution sharpening as the model learns* — the
exact phenomenon the funnel signature exists to detect. There is no column in
which that misreading would have looked anomalous.

**The registered baselines are not contaminated, and this was checked rather than
assumed.** Every `load_checkpoint` call site in `src/`, `scripts/` and `tests/`
was audited for the mode it measures in:

| script | mode at measurement | trains before measuring? |
|---|---|---|
| `chunk11_part0.py` | `.eval()` | no |
| `chunk11_part0b.py` | `.eval()` | no |
| `chunk11_part0d.py` | `.eval()` | no |
| `chunk11_misses_diagnostic.py` | `.eval()` | no |
| `chunk11_pilot_campaign_cost.py` | `.eval()` | trains **throwaway** models only |
| `shakedown.py` | `.eval()` | does not train |
| `campaign.py` (the driver) | **train** | **yes, and never restored** |

Every Part-0 number — the anchor's 1,200-problem baseline, `ANCHOR_BEAT = 101`,
both `NO_REGRESS` floors, the funnel thresholds themselves — was measured in eval
mode. The driver is the sole site that was wrong, and the driver is new in this
chunk: **it has never produced a registered number.** The blast radius is exactly
the driver's own development runs, all of which are superseded.

### The fix

The mode is asserted **where the evaluator is built**, not somewhere up the call
chain. `model_evaluator` raises `EvaluatorModeError` on a model in train mode,
because "we call `.eval()` somewhere" is precisely the claim that rots — it rots
by someone adding a training step between the `.eval()` and the search, which is
literally what happened here.

Train mode is confined to `train_on_ring`, which now captures the incoming mode
and restores it before returning. The driver calls `.eval()` at both checkpoint
load sites and after each training step; those are belt-and-braces, and the
evaluator's own guard is what makes them redundant rather than load-bearing.

Both polarities are tested, and a third test asserts that `dropout > 0` makes the
two modes actually differ — without it, a config that drifted to `dropout = 0`
would leave the guard passing while guarding nothing.

---

## F-23 — Resume rebuilt the pool with the anchor alone, restarting par escalation from rung zero

**Found:** 2026-08-17, by the resume gate's ring-content assertion — and only
*after* F-22 was fixed. While the driver was nondeterministic the gate failed on
ring content for a reason that explained itself, and a second cause underneath it
was indistinguishable from the first. Removing the nondeterminism is what made
the residual difference attributable.

### The mechanism

`CheckpointPool` lives in memory; its snapshots live on disk. `run` constructed a
fresh pool, seeded it with the anchor, and entered the loop — so a run resuming at
iteration *n* held **one member** where the uninterrupted run held *n* + 1. The
snapshots `snap-0.pt … snap-{n-1}.pt` were sitting in the run directory, written
by `enroll` before each commit, and nothing read them back.

Par escalation is the mechanism by which par rises with the model. A resumed
campaign silently restarted it from the anchor while the ring, the weights, the
value head and the row history all continued correctly — the run would have gone
on measuring improvement against a par that had quietly fallen back to where it
began.

### Measured

Killed at iteration 1, resumed, compared against uninterrupted. Three iterations,
twelve episodes each:

| | uninterrupted | killed and resumed |
|---|---|---|
| ring slots | 113 | 113 |
| slots identical | — | 104 / 113 |
| `par_source = bfs` | 95 | **96** |
| `par_source = pool` | 18 | **17** |
| rows, all columns ex-wall-clock | — | **identical** |

That last row is the finding's point. Every logged column agreed — `episodes`,
`episodes_solved`, `z_by_par_source`, `pool_par_fraction`, all seven provenance
fields — while the ring the model actually trains on differed. **No row-level
comparison at any strictness would have caught this**, because the pool's
composition is not in the schema: the only pool columns are `pool_refusals` and
`pool_par_fraction`. `CheckpointPool.composition()` exists, returns size and
member steps, and is logged by nothing.

At golden scale the divergence is one step in 113. At campaign scale — 20
iterations, 400 episodes each — a resume at iteration 15 discards fifteen rungs.

### The fix

`run` replays the original enrolment sequence on resume: the anchor, then
`snap-0` upward through `snap-{start-1}`. **Order, not just membership** —
`sample` is `rng.choice` over the members list, so the same set in a different
order draws a different snapshot.

A missing snapshot is a refusal, not a repair. `enroll` precedes the row and
`LATEST`, so every committed iteration has one; if it is gone the run's history
is not reconstructible, and continuing with a thinner pool would be this same
defect arriving by the path meant to prevent it.

**Registered as open:** the pool's composition should be a logged column. It is
load-bearing for M1-A2 §6's two-population split, and this finding is the
demonstration that its absence hides a real defect. That is a schema-era change
against a frozen page, so it is recorded here for ruling rather than taken.

---

## F-24 — The pre-flight was never called, and had been broken by an unrelated change nobody noticed

**Found:** 2026-08-17, while threading `pool_composition` through `iteration_row`
for M1-A3. `campaign.preflight` did not supply the new keyword — and did not
supply `evaluator_checkpoint_sha256` either, which became required at era 2, in
the previous session.

`grep` for callers returns three hits, and **all three are different functions
with the same name** in `chunk11_pilot_campaign_cost.py`, `generate.py` and
`chunk11_misses_diagnostic.py`. Nothing called this one:

```
TypeError: iteration_row() missing 2 required keyword-only arguments:
'evaluator_checkpoint_sha256' and 'pool_composition'
```

### Why it matters more than a missing call

D-A1 §3 registers the pre-flight as *every row class, at micro scale, before
iteration 0 spends anything* — a gate whose entire purpose is to catch a broken
output path before the campaign pays for it. It was broken by a signature change
made two sessions ago, and **that is the exact event it exists to catch, having
happened to the gate itself.** M1 would have run twenty iterations without it and
nobody would have learned that until reading this line.

Same species as F-23 (`composition()` logged by nothing) and the
`seed_pool_with_anchor` defect (honoured in one script and no driver): the
capability is built, documented and registered — and no consumer was ever wired.
The tell is identical in all three: *nothing referenced it*, and no test noticed,
because a test for an uncalled function is an uncalled test.

### Second instance, in the same commit

`scripts/shakedown.py` calls `iteration_row` too, and had the same missing
argument — so **shakedown has also been unrunnable since era 2**. It was
discovered only because the M1-A3 ruling required repointing it. Two of the three
`iteration_row` call sites in the repository were dead.

### The fix

`preflight` supplies both columns and is **called from `run` when `start == 0`**
— not on resume, where the output path has already written rows this run and
re-proving it would spend the anchor's time on a question the artifacts answer.
It costs 0.3s.

It now also writes a **switch row through the real criterion on the real micro
ring**, because "every row class" has to include the row class it was skipping:
the previous body constructed a `RunState` and `del`'d it, which exercises a
constructor and calls it coverage of an output path. An abstention there is a
pass — the switch row's whole registration is that abstentions write too.

### Registered, not taken: the pool's ordering key carries two denominations

Surfaced by M1-A3's new `order` view. `CheckpointPool` evicts by `Member.step`,
keeping the largest. The anchor's step is its **training-step count, 5000**;
campaign snapshots enrol with `step=n`, the **iteration index, 0–19**. Simulated
over the campaign's twenty iterations:

| | result |
|---|---|
| final pool | `[13, 14, 15, 16, 17, 18, 19, 5000]` |
| anchor retained as rung zero | yes |
| snapshot slots actually available | **7 of 8** |
| if the anchor's step were 3 instead | `[12 … 19]` — **anchor evicted at iteration 4** |

The behaviour is correct: rung zero is retained. But it is correct *because*
5000 > 20, not because anything decided it — and the anchor permanently occupies
one of the eight slots, so the effective snapshot depth is seven. `shakedown.py`
enrols at `5000 + n + 1`, which keeps its ordering sane by a constant chosen to
sit above the anchor's.

Correct by coincidence rather than by construction, which is the state this
project does not leave standing — but changing the eviction key changes par
escalation, and therefore M1's subject. **Recorded for ruling rather than taken.**

**Consequence for M2.** Rung-zero permanence should be a property of the pool's
design, not of `5000 > 20` — a pinned flag on the anchor member, or a two-key
sort that separates *is this the anchor* from *how recent is this snapshot*. The
present single key conflates them by conflating their denominations.

And the arithmetic that follows from it: because the anchor permanently occupies
one slot, **`pool_size: 8` buys the campaign an effective snapshot depth of
seven**, not eight. Whatever M2 sets that number to, it is buying one less than
it reads.


---

## F-25 — Every golden row recorded the campaign's fingerprint while running golden config

**Found:** 2026-08-17, reading `run` to confirm where the five-iteration
rehearsal's cadence fires. The row assembly passed `config_fingerprint=
CAMPAIGN_FINGERPRINT` — a module constant — and `run` is the composition
**shared by golden and the campaign**.

| | value |
|---|---|
| golden config's actual fingerprint | `fc75760f52fc9efd9d45…` |
| what golden's rows recorded | `ce41af96ee85f0a29c90…` |

The column's registered job is that *a run's rows can always be read against the
config that produced them*. Golden's rows named a config that had not produced
them, and no artifact carried the one that had. Two runs at two different configs
were indistinguishable by the field whose sole purpose is telling them apart.

### The shape

This is the stub-digest defect one level up, and the irony is exact: golden
already asserts that `evaluator_checkpoint_sha256` **moves**, on the reasoning
that *a config field records what was asked for and a digest records what ran*.
The row directly above that assertion was recording a constant. Golden checked
the model's provenance and never its own.

D-A2 §2 had already stated the correct rule — "rows carry their producing
profile's fingerprint" — so this is not a missing decision but an implementation
that did not match a decision already written down.

### The fix

`config_fingerprint(cfg)`: the config that ran. The campaign's rows still carry
the registered value, but they now carry it **because `run_campaign` asserted the
config at the door and the row records what that assertion licensed** — rather
than because the row quotes a constant back to itself. An assertion at the door
plus an honest recording downstream is a chain; a constant in the row is a
sentence about a check that happened somewhere else.

Golden gains both polarities: its rows must record golden config's fingerprint,
and that value must **not** be the campaign's.

---

## F-26 — The resume contract covered one log of three, and the gate that checks it compared the same one

**Found:** 2026-08-17, reading `run` to establish where the five-iteration
rehearsal's cadence fires — the same read that found F-25.

`resume` truncates `iterations.jsonl` to the committed prefix, and nothing else.
The switch row is written **before** `commit_iteration`, so it is provisional in
exactly the way the row is, and it was never truncated:

| run | `value_switch.jsonl` iterations |
|---|---|
| uninterrupted | `[0, 1, 2]` |
| killed at iteration 1, resumed | `[0, 1, 1, 2]` |

**F-13's duplication signature**, in a row class nothing compared. The value-head
switch log is how the criterion is audited after the fact; a duplicated
evaluation there is a second opinion the criterion never gave.

### The part that is about the gate, not the driver

`test_campaign_resume.py` opens by declaring that *the comparator names what it
ignores, per field* — and it did, exhaustively, for three wall-clock columns of
one file. It then ignored two entire files without mentioning them. Precision
about the wrong boundary reads as rigour, which is what made it survive: the
docstring's care is evidence of attention, and the attention stopped at the file
it had chosen.

The gate now iterates a named `LOGS` tuple, so adding a log without adding it
here is a visible omission. And because `instruments.jsonl` cannot exist at
golden config — `ladder_every = 99`, the cadence never fires — the gate asserts
**which** of those comparisons was real, rather than passing on two empty lists
and looking identical to one that compared something.

### Second defect, same read: the cadence unit was never durable

`run_instruments` returns the most expensive measurement the loop makes — 1,200
problems at two budgets, then 600 more for the primary — and it went into the
returned summary dict and nowhere else. **No artifact carried it.**

- A pod vanishing at iteration 12 lost iterations 4 and 9's instrument passes
  with nothing on disk to resume from, against the standing rule that volume +
  git + rsync'd artifacts must always suffice to continue elsewhere.
- M1-A2 §1 requires the funnel trigger's row to carry *the contemporaneous pooled
  beat-par delta*. The delta was in memory; the entropy columns were on disk. The
  requirement was **unsatisfiable from artifacts** — not violated, but
  unmeetable, which is the shape M1-A2 §1 and D-A2 §1 were both written about.

`instruments.jsonl` is now written with `fsync` at each cadence, provisional like
the row and truncated with it. Its truncation is **by the row's `iteration`, not
by position**: it carries one row per cadence, so a positional prefix of
`committed + 1` lines would keep a row for an iteration that never committed.
That distinction has its own test, with the positional answer named as the wrong
one it must not give.

---

## F-27 — The campaign's cost model is flat where the measured curve is not

**Measured:** 2026-08-17, the M1 dress rehearsal, `runs/m1_rehearsal` (recovered
to `runs/m1_rehearsal_recovered`). M1-A2 §2 prices self-play at **10.1 min per
iteration, flat, for all 20 iterations**, and the cadence unit at 109.5 min.

| iteration | self-play | train | capped | measured / modelled |
|---|---:|---:|---:|---:|
| 0 | 9.8 m | 8.8 m | 2 | 0.97× |
| 1 | 12.3 m | 9.4 m | 3 | 1.22× |
| 2 | 13.7 m | 9.6 m | 3 | 1.36× |
| 3 | **29.2 m** | 9.5 m | **43** | **2.89×** |
| 4 | *not measured — the row never committed (F-29)* | | | |
| **cadence unit** | **227 m** | | | **2.07×** |

**The mechanism is capped episodes.** A capped episode runs a search at every
step to the cap; a solved one stops early. Caps went 2 → 43 and self-play tripled
in the same iteration. Caps rise because par escalation works: pool par comes
from snapshots that are themselves improving, so the bar rises each iteration and
episodes that cannot clear it run to the cap. `z_by_par_source` shows the model
beating its own snapshots — `pool/+1` at 1, 7, 5, 12 — while `bfs/−1` climbs 4 →
76.

**What is NOT claimed.** Iteration 4's self-play was never recorded, and the
window between iteration 3's commit and `ckpt-4.pt` is 25.3 min covering
self-play *and* training — so iteration 4's self-play was plausibly ~16 min,
**below** iteration 3's 29.2. Four measured points with one spike, and a fifth
that is inferred and lower, do not establish a monotone curve. The honest
statement is that the flat 10.1 is wrong and the shape is not yet known.

The cadence factor is firmer: 227 minutes against 109.5, on fixed problem sets,
**measured end to end**. Four cadence units across 20 iterations is 7.3 h rather
than 3.7 h on that factor alone.

**Ruled: nothing changes.** M1-A2 §2's table is descriptive and no gate rests on
it. The analysis point stays hard at 20; no config moves, because config is
fingerprint. If wall-clock forces a stop, that is the prereg's existing
early-termination branch — reported as **resource, not results**, with the
primary evaluated at the last completed ladder pass.

**Registered watch item — `escalation-outruns-learner`:** pool par rising faster
than the model improves, driving caps toward dominance and starving the ring of
successes. Watch column: cap rate per iteration. No action mid-campaign.

---

## F-28 — The pool's two unavailability counters are separated, documented, and logged by nobody

**Found:** 2026-08-17, trying to answer whether iteration 0's
`pool_par_fraction = 0.2475` against a configured `par_from_pool_frac = 0.20`
was sampling or structure.

`PoolStats` counts `pool_par_unavailable_empty` and
`pool_par_unavailable_capped` **separately**, with the capped/stuck lesson
written into its docstring — *"they are different diseases … the fixes differ
(wait for snapshots vs raise the budget)"* — and exposes both through
`as_dict()`. `shakedown.py` logs that dict. **The campaign driver logs
`pool.stats.refusals` and nothing else.**

So the counter that answers the question never reaches an artifact, and the
in-memory value died with the run. Sixth instance of the species, and the second
of the *nothing compares it* variant — here sharpened to **nothing records it**:
the capability is built, the consumer interface is written, and one of two
callers is wired.

**Bounded in the meantime.** The pool was anchor-seeded at iteration 0, so
`_empty = 0` and any fallback there was `_capped`. Since a fallback can only move
par_source from `pool` to `bfs`, draws = 99 + fallbacks **≥ 99** — the fallback
can only push the fraction *down*, so the observation is bounded in the direction
that makes it more extreme, not less. Under Binomial(400, 0.2), 99 is +2.38σ,
P(X ≥ 99) = 0.0118.

**Fix:** both counters become iteration-row columns at era 4, batched with
anything else pending. **Approved and separate:** iteration 0's draw sequence is
fully determined — `training_problems(…, 400, seed=0)` against a pool of exactly
one member, `Random(13)` — so replaying `campaign_problems` offline answers the
0.2475 question *exactly* rather than by bound.

---

## F-29 — The first iteration in which nothing was absent, the row was refused

**Found:** 2026-08-17, by the dress rehearsal, at iteration 4, **after the
cadence unit had been paid for in full** — 3 h 47 m of measurement, then:

```
SchemaError: pool_par_fraction: named in 'absent' but present in the row
```

### The mechanism

`iteration_row` carried a default:

```python
"absent": absent or {
    "pool_par_fraction": "league.par_from_pool_frac not yet wired (chunk 9 part 2)",
    "ladder_pass": "not a ladder iteration",
},
```

`or` cannot distinguish *unspecified* from *specified as empty* — and here **the
empty case is the meaningful one**: it says every column has a value. The driver
computes absences into a dict and passed `absent=absent or None`, so an empty
dict became `None`, which the default then replaced with two fabricated claims
about columns the driver had just written.

The condition for an empty `absent` is precise: **the ladder must fire and the
pool must be non-empty** — a cadence iteration with live par escalation. That is
iteration 4, 9, 14, 19 of the campaign, and *no earlier iteration and no golden
run ever*: `golden_config` sets `ladder_every = 99`, which guarantees
`ladder_pass` is always absent and the default therefore always plausible.

**This was campaign-blocking, not merely a bug.** M1 would have run 4 iterations,
paid the first cadence unit, and died — and every resume would have re-run that
cadence and died again at the same place. The campaign could not have passed
iteration 4 by any number of retries.

### Two things are vindicated by it

**D-A2 §1.** The DONE-WHEN asked for a *three*-iteration rehearsal; D-A2 amended
it to five, on the arithmetic that three could never exercise a cadence unit. A
three-iteration rehearsal would have passed clean and M1 would have hit this at
hour four. **The amendment was written to fix an unsatisfiable gate and it caught
a campaign-blocking defect instead.**

**F-26.** The cadence unit's measurements survived the crash *only because*
`instruments.jsonl` had been made durable that morning. Under the previous code
the 3 h 47 m result lived in a dict inside the process that raised, and would
have been lost entirely.

### The fix

`iteration_row` no longer invents an absence. **Absence is a claim about the
caller's run, and the library cannot make it** — the fabricated text ("not yet
wired (chunk 9 part 2)") described a world that stopped existing two chunks
before it fired. An empty mapping now means what it says.

Three callers relied on the default and now declare their own: `shakedown.py`,
the driver's `preflight`, and `tests/test_runner.py`. That count is itself the
finding restated — the default was load-bearing for everyone and correct for
no one.

Both polarities are tested at the unit the defect lived in.

**Registered, not taken:** there is still **no end-to-end test of a cadence
iteration**, because `run_instruments` costs ~110 min at real scale and
`golden_config` deliberately switches the cadence off. That is the gap F-29 came
through, it is named here rather than left silent, and closing it needs a cheap
instrument population — a design change, not a fix.

---

## F-30 — The primary's test of record has been unsatisfiable since the freeze

**Found:** 2026-08-17, attempting to run it on the rehearsal's cadence unit.

PREREG-m1 §2 specifies the primary in a table:

| slot | value |
|---|---|
| pairing | **per problem**, against the anchor's Part-0d outcomes |
| test of record | **paired-difference bootstrap**, criterion **CI excludes zero** |

**Neither arm records per-problem outcomes.**

| arm | artifact | what it stores |
|---|---|---|
| baseline | `runs/chunk11_part0d_scripted_strata.json` | **3,391 bytes** — `at_par_z_0`, `beat_par_z_plus_1`, `mean_z`, `steps_minus_par` bins, per stratum. No problem identifiers anywhere in the file. |
| campaign | `instruments.jsonl` → `primary.per_stratum` | `{"beat": n, "of": 200}` |

A paired-difference bootstrap resamples *problems*, and needs each problem's
outcome under **both** arms. Two aggregate counts cannot be paired: 108 beats
against 101 beats says nothing about which problems moved, and the per-stratum
split — 7 losing 21 while 8 gains 24 — is exactly the pattern that pooling hides
and pairing exists to expose.

So the campaign's **primary** — the measurement M1 exists to make — cannot be
computed by its registered test from its own artifacts, and could not have been
since the page was frozen. This is a defect in the **analysis plan**, not the
code: §3 is meticulous about the *protocol* being identical across arms ("a
baseline measured under one protocol and a primary measured under another are not
paired, whatever the pairing code does") and never asks whether either arm
records what pairing consumes.

Same species as the schema default: specified, never wired end to end. The tell
here is that the specification is *more* careful than usual — the pairing row
names the exact artifact it pairs against — and the care went into which
comparison, never into whether the comparison is computable.

### Fix

Per-problem outcomes recorded on **both** arms:

* `run_instruments` records each problem's `(key, steps, par, z)` beside the
  aggregate, so the campaign arm carries pairing from its first cadence unit;
* **Part-0d re-runs deterministically** to capture the same for the anchor. Its
  protocol is printed verbatim in §3 and its inputs are frozen, so the re-run's
  aggregates must reproduce `101/600` and every per-stratum count exactly —
  **and that reproduction is the verification** that the per-problem capture is
  the same measurement rather than a new one. A mismatch is itself a finding.

---

## F-31 — The lever built so it would not have to be written under pressure was never connected

**Found:** 2026-08-17, designing the sweep that would have measured it.

`train.rehearsal_frac` is a fingerprinted config field. `rehearsal_split` is its
implementation. **`train_on_ring` never calls it.** Repo-wide, `rehearsal_split`
appears in exactly two places: its own definition, and four unit tests.

```
src/reckoner/train.py:370:  def rehearsal_split(...)          # the definition
tests/test_train.py:185     assert rehearsal_split(128, CFG) == (0, 128)
tests/test_train.py:193     assert rehearsal_split(128, cfg) == (32, 96)
tests/test_train.py:195     assert rehearsal_split(10, cfg) == (5, 5)
tests/test_train.py:203     with pytest.raises(...)
```

`train_on_ring` takes no supervision set and has no path that mixes supervised
batches. **Setting `rehearsal_frac = 0.25` would move the config fingerprint and
change nothing whatsoever.**

### Why this one escalates

The machinery was written *in anticipation*, and its own docstring says why:

> "Ported now and left inert, per the plan's 'the lever exists before it's
> needed': a rehearsal mechanism added *after* catastrophic forgetting appears is
> a mechanism written under pressure, against a run that is already degrading."

The anticipation was correct — F-27's diagnosis is catastrophic forgetting, top-1
0.9699 → 0.8942 in one iteration at 39 ring-epochs with no supervised anchor. The
lever was built for exactly this, and it is not attached to anything.

The docstring even names the adjacent trap — *"'dormant' means 'untested'"* — and
the tests answer it. **The unit is verified; the integration does not exist.** No
test asks whether the returned split reaches an optimiser, so four passing
assertions on a function nobody calls read as coverage.

**And the class was declared closed.** Chunk 8's commit: *"a dead config key is
the `batch_leaves` hazard and is not repeated."* At that moment
`value_q_mse_weight` was dead at the loop's core and `rehearsal_frac` was dead
too. A named class, declared closed, recurring twice underneath the declaration.

### The mechanism rung — ordered, not registered

Seven instances and a false closure is the ladder's terminal signal, so the
remedy is mechanical rather than another resolution:

> **A live-config test: every fingerprinted field either names the test that
> proves it changes observable behaviour, or fails.**

The ANCHORS adoption-scan pattern pointed at config. Decidable, cheap, and it
would have caught `batch_leaves`, `value_q_mse_weight` and `rehearsal_frac` at
the moment each was introduced. It is what makes "not repeated" a true sentence
rather than an intention.

### Governance

Wiring the rehearsal path is a **defect fix**, not an amendment: the spec already
specifies rehearsal and the implementation lacked it, so building it restores
specified behaviour. **Setting `f > 0` is the amendment** — that is the treatment
decision, and it lands as M1-A4 after the sweep, carrying measured numbers.

---

## F-32 — The campaign never evaluates the two gates whose regression §8 makes BLOCKED

**Found:** 2026-08-17, by the config census — `ladder.problems_per_pass` came back
with no consumer, which is only possible if no ladder pass runs.

PREREG-m1 §4.5 declares two gates on the ruled wiring:

| gate | threshold | anchor measured |
|---|---:|---:|
| **10b** top-1 rule-site, depth ≤ 3, F-09 unseen subset | 0.9000 | 0.9699 |
| **11** depth ≤ 2 solve rate, 16-sim, value-silent | 0.9500 | 1.0000 |

§8 declares: *"**Gate 10b or gate 11 regresses** on the ruled wiring. BLOCKED."*

`run_instruments` — the sole instrument seam, which every cadence measurement
goes through — computes **no-regress at both budgets and the primary. Nothing
else.** No gate 10b, no gate 11, no ladder pass. `ladderpass.run_pass` occurs in
`campaign.py` exactly once, inside a docstring.

**So two of §8's five BLOCKED branches cannot fire.** They are not failing to
trip; they are never evaluated.

### What that cost, measured

Today's manual diagnostic ran gate 10b across the rehearsal's checkpoints:

| checkpoint | top-1 | vs threshold 0.9000 |
|---|---:|---|
| `phase1.pt` | 0.9699 | +0.0699 |
| **`ckpt-0`** | **0.8942** | **−0.0058 — BREACHED** |
| `ckpt-4` | 0.8845 | −0.0155 |

**Gate 10b breaches at iteration 0**, after the first training step, ~20 minutes
into a run. The no-regress floors — which *are* evaluated — first fire at
iteration 4, after the 110-minute cadence unit, ~4 hours in. The campaign's
earliest and cheapest stop-branch is the one that is not wired, and the
expensive one is.

This also explains why the rehearsal's diagnosis needed a hand-written script:
the number that identifies the failure mode is one the campaign was specified to
compute and does not.

### Related, and cheaper to fix

* **`campaign.interop_threads`** is read by nothing. M1-A2 §4 classes it as
  **OBSERVED** — "licenses only 'this is what ran'" — so it is a legitimate
  provenance field rather than a pin. But `assert_threads` reports
  `torch.get_num_interop_threads()` and never compares it against the declared
  32, so the record's claim is unverified where one line would verify it.
* **`search.perspective`** is exempt on different grounds and the distinction
  matters for the registry: its legal range is exactly one value, enforced by
  `validate`, so varying it *within legal values* is impossible. A pinned
  invariant, vacuously behavioural. `rehearsal_frac` fails precisely where this
  passes — its legal range is `[0, 1)`, every value passes validation, and none
  of them changes anything.

That contrast is the two-tier rule's sharp edge: **the question is not whether a
field is read, but whether varying it across its legal range changes what runs.**

---

## F-33 — The cadence runs *beside* the ladder instead of *through* it, and that one fact is F-30, F-32 and the unread bootstrap

**Found:** 2026-08-17, from a single question — does `run_instruments` call
`ladderpass.run_pass`, or reimplement its job?

**It reimplements.** `run_instruments` calls `run_iteration` directly and folds
the outcome into `{"beat": n, "of": 200}`. `ladderpass.run_pass` is referenced in
`campaign.py` exactly once, inside a docstring.

### What the ladder already provides, and the driver rebuilt worse

| capability | `ladderpass` / `ladder` | `run_instruments` |
|---|---|---|
| per-problem records | `pair_scores.jsonl`, one row per (arm, problem), **appended as it happens** | aggregates only |
| paired-difference bootstrap | `ladder.py`, with `resamples`, `alpha`, CI | none |
| rung passes (arms + roles) | declared roles, refuses an undeclared one | none |
| resumability mid-pass | torn-tail repair, finished-unit skip | none |
| declared skips | counted per arm, keeps the paired set one set | n/a |
| currency safety | `ladder.pair` refuses across currencies | n/a |

`ladderpass`'s own module docstring names the exact defect the driver then
committed:

> "Every arm-problem outcome is appended **as it happens**. The alternative —
> accumulate in memory, write the aggregate at the end — loses two things that
> cannot be recovered … **The paired-difference bootstrap consumes *pairs*; a run
> that stored only means has thrown away the input to its own test of record.**"

`run_instruments` stores only means. Chunk 10 built the machinery and wrote down
why aggregates are insufficient; chunk 11's driver reimplemented the aggregate
path beside it.

### Three findings, one root cause

* **F-30** — the primary's paired bootstrap is uncomputable. *Because* `run_pass`
  is not called, so no per-problem rows exist.
* **F-32** — gates 10b and 11 are never evaluated, so two of §8's five BLOCKED
  branches cannot fire. *Because* no rung pass runs.
* **`ladder.bootstrap_resamples` unread** (config census). *Because* the
  bootstrap is never invoked.

Not three gaps to fill. **One routing error**, in the mono-instance family with
the second `SupervisionSet` and the second `model_evaluator` — and the same
signature: the second implementation is thinner than the first and drops exactly
the property the first was built to preserve.

**So the fix is routing, not adding.** F-30's per-problem recording arrives for
free, along with the rungs, the currency machinery, mid-pass resumability, and
the bootstrap. That also retires the "add per-problem output to
`run_instruments`" remedy F-30 proposed — which would have been a *third*
implementation of a job the repo already does properly once.

**Part-0d still needs its deterministic re-run**, since the baseline arm has no
per-problem records either and no routing change can retro-fit them.

### Related — two homes for one value, with a provenance edge

The census flagged `generator.train_set_size`, `suite_problems_per_depth`,
`suite_depths` and `max_bfs_depth` as unread. `scripts/generate.py` is why, and
the tell is inside one file:

```python
parser.add_argument("--seed",       default=CFG.seed)      # reads config
parser.add_argument("--train",      default=100_000)       # hardcoded literal
parser.add_argument("--suite-size", default=200)           # hardcoded literal
```

Each dataset's `meta.json` records a `config_fingerprint` — a claim that *this
config produced this data*. The generator fields in that config were never read.

**The values agree** — 100,000 against `train_set_size = 100000`, 200 against
`suite_problems_per_depth = 200` — so nothing on disk is false today. But the
agreement is coincidental rather than causal, and the fingerprint attests to an
authorship it does not have. Change either home and the recorded fingerprint
would testify to a config that did not shape the data.

Low priority, and **not** wire-by-default: these are unbacked keys, so the
disposition is delete-or-amend per the census rule. Recorded because a provenance
claim that is true by coincidence is one nobody will re-check.

---

## F-34 — The rung pass has no viable population, and both candidates are measured-saturated

**Found:** 2026-08-17, pricing the two dispositions for the cadence's missing
third part.

M1-A2 §2 prices the cadence unit in three terms. Two are implemented:
no-regress (1200) and the primary (600). The third — **model rung (200 @48) 5.0
min, greedy rung (200) 0.0 min** — corresponds to no code at all (F-33).

Building it needs a population and a budget, and **both referents have
evaporated**:

* PREREG calls the rungs *informational at matched `s*`* — and `s*` resolved to
  **null** when branch (c) fired. The budget clause survives its own referent.
* `ladder.problems_per_pass = 200` names a size and nothing else. It is also
  one of the census's dead keys.

### Both candidate populations are saturated, by measurement already on the record

**`runs/paired/smoke_v1.jsonl`** — 389 problems, depths 1–6, all-BFS par, born
clean of F-09, ANCHORS-registered at `f0b10fd1…`, purpose-built for the ladder.
Everything a rung population should be, except able to move. M-B:

| | |
|---|---|
| model z histogram | +1: **0**, 0: 387, −1: 2 |
| ceiling | **0.0** — par is BFS-exact, so `+1` is impossible by construction |
| max possible movement | 0.005141 |
| CI half-width (model − greedy) | 0.042417 |
| **movement / CI half-width** | **0.1212** |

Greedy is deterministic, so the paired difference moves only as the model's z
moves — and its entire remaining headroom is **12% of one CI half-width.**

**The suites** (`solve_in_1..6`) are saturated too: F-20, anchor 1193/1200 at
`sims = 48`.

**The scripted strata** `{7, 8, 10}` are the only unsaturated population — and
they are already the primary. A rung pass there would **duplicate the primary
rather than inform it.**

So there is no population on which a rung pass would read anything: two are at
their ceilings, and the third is the measurement the rung would be reporting
alongside.

### Disposition — (b), and the priced comparison

**(a) `smoke_v1` entire** — no subsample rule to invent, provenance impeccable,
cost affordable (389 problems unbatched ≈ 10–17 min against §2's 5.0 min for
200). **Rejected: the instrument cannot move.** A rung whose reading is pinned at
its ceiling by construction reports the ceiling, not the loop.

**(b) Drop the rungs from M1, reason stated, carry to M2.** **Taken.**

Nothing gates on their absence — PREREG calls them *informational* — so M1 loses
no analysis, and the alternative is minting an instrument mid-campaign on a
population measured unable to answer.

M2 is where a rung population gets minted deliberately, with headroom
demonstrated *before* it is frozen rather than discovered saturated afterwards —
which is the same lesson the succession to the scripted strata already taught
once.

**Not stubbed.** The rung path is left absent with this reason filed, because a
stub is a thing someone later mistakes for a specification.

---

## F-35 — The watchlist columns are specified "from data the pass already produces", and the pass produces no such data

**Found:** 2026-08-17, asked directly after F-33 — if the no-regress leg is
aggregate-only, what computes the watchlist?

PREREG-m1 §5 (M1-A2) freezes the 24-problem shared miss set and specifies two
columns:

```
family_remaining = |pass_miss_set ∩ frozen_24|      the family shrinking
novel_misses     = |pass_miss_set \ frozen_24|      a new family growing
```

> "Two columns, computed per ladder pass **from data the pass already
> produces**."

**Neither column exists.** Not in `ITERATION_FIELDS`, not in any code path,
repo-wide. And the quoted phrase is the defect in six words: the no-regress leg
emits `{at_par: 910, of: 1200}`, which **has no miss set**. There is no `∩` to
take.

### Why this one is worse than a missing column

§5 states the reason the pair exists:

> "The aggregate no-regress floor catches net regression and **cannot attribute
> it**. Together these two can, for free."

The page diagnoses the aggregate's blindness, prescribes the cure, calls it free
— and the implementation shipped the aggregate alone. **"For free" was the
assumption that did the damage**: it reads as *no extra measurement*, which was
true, and was taken to mean *no extra recording*, which was not.

The rehearsal is the demonstration. At-par fell 1193 → 910 and no artifact could
say whether the 283 new misses were the frozen family or a new one — which is
precisely the attribution §5 exists to provide, on precisely the event it was
written for. Diagnosing it needed hand-written scripts computing quantities the
campaign was specified to emit.

Sibling of **F-30**, same page, same shape: a specification meticulous about
*which* comparison and silent about whether its input is recorded.

### Fix

Per-problem recording on the no-regress leg, **required independent of execution
mode**: recording and batching are separate questions, and the batched/unbatched
ratio decides only how the pass runs. The frozen 24 are already keyed as
`problem_key` strings in `runs/chunk11_misses_diagnostic.json`
(`cross_reference_sims_1_vs_scripted.shared`), the same format `problem_key_of`
emits, so the intersection is computable the moment a miss set exists.

---

## F-36 — The driver ignores `league.snapshot_every` and agrees with it only because the default is 1

**Found:** 2026-08-18, building the live-config gate's exemption registry — the
census listed the field REPORTED, and the question "reported by whom, and who
acts on it?" had no good answer.

`scripts/shakedown.py` honours it:

```python
if (n + 1) % cfg.league.snapshot_every == 0:
    pool.enroll(...)
```

`campaign.py` does not — `pool.enroll(...)` runs unconditionally, every
iteration. The default is **1**, so the two agree, and every measurement on the
record is unaffected.

**Correct by coincidence, not by construction** — the fourth appearance of that
shape in this project, after the anchor's eviction (F-24), the generator's
CLI-shadowed dataset parameters (F-33), and `shakedown.py`'s duplicate evaluator.

Set `snapshot_every = 5` and the shakedown would enrol four fewer snapshots per
five iterations while the campaign enrolled all five — a fingerprinted field
steering one composition and not the other, changing pool growth and therefore
**par escalation**, which is the mechanism M1's primary is denominated against.

Registered in the live-config gate's registry rather than silently exempted,
because the honest entry is "this is not a clean exemption": the field is
*consumed*, just not by the composition that matters.

**Not fixed in this round.** Making the driver honour it changes enrolment
cadence at values other than 1 — behaviour, not restoration — and at the current
value it changes nothing at all. It batches into M1-A4 with the dead-key
dispositions, where the choice is: honour it in the driver, or delete it and let
unconditional enrolment be the declared behaviour.

---

## F-37 — Catastrophic forgetting saturates fast, which forecloses a whole class of remedy

**Measured:** 2026-08-21, the sweep verification (`runs/sweep_verify.json`).

The mechanism arm was built to test whether rehearsal works by *reducing
ring-epochs*. It also, without being designed for it, controlled the obvious
confound — and the control is what carries this finding.

| arm | ring-epochs | optimiser steps | gate-10b hits |
|---|---:|---:|---:|
| `f = 0.00`, 400 steps | 39.2 | 400 | **1099 / 1229** |
| `f = 0.00`, 200 steps | 19.6 | 200 | **1099 / 1229** |

**Identical hits from half the training.** The parameter digests *differ*
(`994fc3c3…` against `100da07f…`), so the steps argument took and the weights
genuinely moved — this is not a knob that failed to fire. Top-1 is simply
**insensitive** to halving the epochs in this range.

### What it forecloses

If 200 steps and 400 steps do identical harm, **forgetting is essentially
immediate rather than cumulative.** That rules out an entire family of otherwise
attractive designs:

* "run iteration 0 clean, turn rehearsal on afterwards" — by the time iteration 0
  finishes, the damage is already complete;
* "ramp `f` up as the ring grows" — the ramp arrives after the event it was
  meant to prevent.

**The supervised mixture has to be present from the first steps of iteration 0.**

It also suggests the damage may complete in far fewer than 200 steps, which is a
cheap probe if that ever matters — bisecting the step count against gate 10b
costs minutes, not the hours a full arm costs.

### And it kills the epoch hypothesis cleanly

At **matched** ring-epochs of 19.6, with the only difference being whether
supervision is present:

| | hits |
|---|---:|
| `f = 0.50`, 400 steps, supervision ON | **1180** |
| `f = 0.00`, 200 steps, supervision OFF | **1099** |

**Supervision buys +81 hits; epochs buy 0.** Epoch scaling drops as a lever;
rehearsal fraction is the lever. Registered as F-27's `escalation-outruns-learner`
watch item stays open but is not this.

### A note on how nearly this was misread

The verification script printed *"the 4dp match is coincidence, and the arm
stands"* — comparing only the rounded ratios. The **raw counts are also
identical**, so it is not a rounding artefact at all; it is insensitivity, which
is the finding. Reporting top-1 as a raw count beside the ratio is what made the
difference visible, and it was added precisely because an implausible
intermediate deserves checking rather than believing.

### Erratum (2026-11-10, F-41)

Both arms of this finding were measured while the ring's policy targets sat in the
wrong coordinates. At `f = 0.00`, ring rows carried about 7% of a policy gradient,
and more of that pointed at a wrong legal action than at the right one (F-41).
"Forgetting is essentially immediate" is therefore **unexplained under a defect**,
not established. It describes training on value and steps gradients plus an
`eval_add`-biased policy push, not training on search targets.

The 200/400-step pair is also not matched in learning rate:
- the 200-step arm is pure warmup and ends at the peak, 2.0e-4;
- the 400-step arm ends at 1.23e-8;
- their LR integrals are 0.0201 and 0.0402 (`lr_at`, `lr_warmup_steps 200`).

Re-measure under correct targets with an LR-controlled pair before relying on
either reading.

**Amendment (2026-11-10, F-41 review).** The sentence above saying more of the
surviving mass "pointed at a wrong legal action than at the right one" is
withdrawn. It was the probe's pooled uniform-stub reading over all ten suites. In
the stratum that matches ring-0 (BFS par 1–6), correct exceeds wrong-legal: 6.00%
against 4.61%, with 89.39% dropped. What stands is that roughly 90% of the target
was dropped. Also, ring-0 must be regenerated under the fixed writer before this
pair is re-run; `train_on_ring` now refuses the old ring.

---

## F-38 — A watchdog whose only output is a kill is indistinguishable from a healthy run

**Found:** 2026-08-22, when attempt 2's rehearsal ran to iteration 19 instead of
stopping at 4. The `stop_at_4` watchdog was never launched.

Attempt 2 launched **two** processes with authority over the run:

| process | authority | evidence it is alive |
|---|---|---|
| checkpoint watcher | kill on band breach | **reports at every checkpoint**, writes `checkpoint_watch.json` after each verdict |
| `stop_at_4` watchdog | kill at `LATEST = 4` | **none — its only output is the kill** |

The checkpoint watcher was *required* to have a liveness marker, on the reasoning
that *"silent absence should not be indistinguishable from silent vigilance"*.
That requirement was never extended to its sibling. **Set-growth again: a
property demanded of one member of a family and never asserted of the family.**

**The asymmetry is what hid it.** A watcher that reports every checkpoint is
self-evidencing — its silence is visible immediately. A watchdog that only acts
at a threshold looks *exactly* like a healthy run until the moment it fails to
act, and by then the thing it was guarding has already happened.

### The rule

> **Every process with authority over a run carries a liveness marker, and the
> launch verifies against a supervisor set declared UPSTREAM and FROZEN** — on
> the expectations page or beside the fingerprint in config.

A count is the part that closes it: markers make each process's *death*
detectable, and the count makes a process that was never *born* detectable —
which is the failure that actually occurred. No marker was missing; one was never
created.

**But the count must not be the launcher's own tally.** A launcher counting its
own launches is the self-referential check family — third member after the
`pgrep -f` self-match and `$?` read through a pipe — because **the forgetting
that omits a watchdog is the same forgetting that omits it from the count.** The
declared set lives upstream and frozen, and launch verifies against *that*. Then
a missing supervisor is a mismatch against a page committed before the run, which
is the mechanism that already catches everything else here.

### What it cost, and what it did not

Six hours and roughly $6. **It did not cost the rehearsal**, which is the part
worth being precise about: the rehearsal was never a separate program. It is the
campaign with an external stop — built as a watchdog rather than an
`iterations = 5` config precisely because `iterations` is fingerprinted, so a
config rehearsal would have run at a *different fingerprint from the thing it was
rehearsing*. Iterations 0–4 on the volume **are** the rehearsal: same program,
same seed, same fingerprint, same commit order.

A gate does two jobs — it **verifies**, and it **interlocks**. The omission
destroyed the interlock and left the verification untouched, and no re-run can
restore an interlock in front of a run that has already happened.

**The watcher that guarded measurement integrity was armed for all 20
iterations. The one that was forgotten guarded the budget.**

---

## F-39 — M1's gain is mostly faster-at-par, not better; and one stratum went backwards

**Measured:** 2026-08-22, from the campaign pass's own `pair_scores` step counts
against Part-0d's, on problems both arms solve. `solved := steps < step_cap (24)`.

The pooled primary separates — 101 → 209 of 600, mean difference **0.18**, CI
**[0.1367, 0.2217]**, excludes zero, unsaturated, 600 pairs. **The decomposition
is the finding.**

| stratum | anchor | campaign | mean diff | CI | separates | shorter / equal / longer | new solves |
|---|---:|---:|---:|---|:--:|---|---:|
| `scripted_in_7` | 43 | **35** | **−0.0400** | [−0.0900, +0.0050] | **no** | 8 / 174 / **18** | **0** |
| `scripted_in_8` | 26 | 85 | +0.2950 | [+0.2150, +0.3750] | yes | 71 / 111 / 15 | 1 |
| `scripted_in_10` | 32 | 89 | +0.2850 | [+0.2050, +0.3650] | yes | 70 / 104 / 17 | 5 |

**149 shorter against 50 longer**, and **zero anchor-only** — nothing the anchor
solved became unsolvable. So the gain is overwhelmingly **shorter paths to the
same solutions**: amortization against loose scripted par rather than capability
against tight par.

**The apparent exception dissolves on inspection.** Six problems (1 + 5) are
solved by the campaign and not by the anchor — but **all six solve at *exactly*
par** (8, 10, 10, 10, 10, 10). Beat-par requires `steps < par`, so every one
scores `z = 0` and **contributes nothing to the primary.** The measured gain is
therefore **100% shorter paths on problems both arms already solve**, with no
capability component in it at all.

Replaying the anchor on those six at rising budgets separates them further:

| | `sims = 1` | `sims = 48` | `sims = 192` |
|---|---|---|---|
| 3 of 6 | — | — | **solved, in 7 / 9 / 9 steps** |
| 3 of 6 | — | — | — |

**Three fall to more search**, so they are *ordering*: the solution was always
reachable and the campaign found it with a better prior — the same amortization
story in a different coat. Only three resist 4× the budget, and even those score
zero on the primary because they land on par.

Five of the six are in `scripted_in_10`, the loosest par in the set, where a solve
is reachable by a walk the anchor's ordering never took.

### Stratum 7 is the decomposition, not a blemish

The two strata that separate carry looser scripted par. The one that declines has
true par 5–6, where **beating par requires an actually shorter proof rather than
a tighter walk to the same place.** Its numbers say it directly: 8 shorter against
18 longer, and **zero** new solves.

The residual leans the same way — 7 misses at 48 sims and 25 at 1, unchanged from
the anchor's but for a single new one across twenty iterations.

### Reconciling two tables a careful reader will read as contradictory

**Zero anchor-only solves in the 600, and one novel miss in the 1,200.** These are
different populations measured by different instruments: the 600 are the scripted
strata `{7, 8, 10}` at `sims = 48` (the primary), and the 1,200 are the exact-par
suites `solve_in_1..6` at both budgets (the no-regress floors). The single novel
miss is a suite problem at `sims = 1`; it is not in the primary's population and
could not appear in its table. Nothing regressed on either instrument — one
problem moved on one of them.

**Consequence for M2**, registered before M2 designs itself around the +0.18: the
campaign made the model **faster at par**, not materially more capable. Those
license different next rounds, and the pooled number alone cannot tell them apart.

---

## F-40 — Registration must follow CITATION, not creation

**Found:** 2026-08-22. Attempt 1's ring became load-bearing the moment attempt 2's
expectations page named it the counterfactual arm — and it was never registered,
so it died with the pod and the comparison it was cited for is permanently
unavailable.

ANCHORS registers **at birth**, which is right for artifacts that are load-bearing
when written. It cannot cover an artifact that becomes load-bearing **later**, by
being cited: the citation arrives after the birth, and nothing re-examines the
registry when a page names a new comparand.

> **An artifact named as a comparand on any page is registered at the moment of
> that citation, not at its creation.**

Sibling of the retrofit census: the same shape, one layer out. What was measured
under one purpose is later relied on for another, and the discipline attached to
the first purpose does not follow.

**Standing action:** sweep the pages for anything cited as a comparand that is not
on the volume, and register what remains. What is already lost is recorded as
lost — attempt 1's ring, and with it the between-run par-escalation magnitude.

---

## F-41 — The ring trained the policy in the wrong coordinates, and the loss swallowed the evidence

**Found:** 2026-11-10, by a cross-repository comparison against TinyAlphaZero-v2,
whose `RootResult.improved_policy()` returns move coordinates and so never had
this defect. **Commissioned** by the principal's instruction, verbatim: *"yes,
write the finding and fix the indexing bug"*. **Instrument:**
`scripts/f41_ring_target_probe.py` at `1fc2473`, clean tree, torch 2.13.0+cpu,
`sims 48`, `gumbel_m 16`.

### The defect

Two coordinate systems meet at the ring, and the writer and reader used different
ones. Line numbers are at `1a434ac`:

* **The search** counts visits by **root slot**: `child_visits[root]` has one entry
  per position in `root_actions = legal_actions(expr)` (`search.py:422-431`).
* **The writer** stored those slot positions: `order = np.argsort(-result.visits)`,
  then `visit_actions=order` (`runner.py:309-314`).
* **The reader** expanded them as **dense action indices**, `rule * max_sites + site`:
  `dense[int(action)] = count / total` (`train.py:240-243`).

Slot *j* can therefore only ever reach cell *j*, which is `eval_add` at site *j*.
Usually that cell is not even legal, and the policy loss then makes the mass vanish:
the masked log-probability is `-inf` there, `nan_to_num` turns the product into
exactly 0 (`model.py:245-248`), and targets are not renormalised. The mass
contributes no gradient and raises nothing.

The search's **priors** were read correctly, through `action_index`
(`search.py:322-325`). Only the training target was wrong.

### Measured

| reading | coverage | correct | wrong legal cell | dropped (illegal cell) |
|---|---|---:|---:|---:|
| **Slot census**, policy-independent: slot *j* trained as cell *j* | all 2,000 frozen suite start states, 18,193 legal slots | **106** | **397** | **17,690** |
| the same, capped at the 16 slots the ring holds | 17,387 slots | 106 | 397 | 16,884 |
| **BFS-optimal first move**: where the move a good policy concentrates on lands | 1,200 states, `solve_in_1..6` | **102** | 0 | **1,098** |
| **End to end**, visit-mass weighted: real runner → ring → `ring_batch`, scored against each record's own `SearchResult` | 200 problems (20 per suite via `sample_indices`), 3,365 records, uniform stub | **2.58%** | **4.17%** | **93.25%** |

* Only **42 of 2,000** states have every slot land on its own cell.
* All 102 "correct" optimal moves are `eval_add`. Every optimal `combine_like_terms`
  (204), `eval_mul` (319), `eval_sub` (351), `sub_both_sides` (184) and
  `div_both_sides` (40) was dropped.
* Mean target mass on legal cells per ring row is **0.0675**, minimum 0.0.
* The target's argmax over legal cells is a max-visit action for **353 / 3,365** records.
* Coverage of the end-to-end reading:
  - records by par: 112 / 194 / 249 / 300 / 296 / 294 / 480 / 480 / 480 / 480 for pars 1–10;
  - by source: 1,445 `bfs`, 1,920 `scripted`;
  - n_legal histogram over the suite states: {1: 234, 2: 158, 3: 102, 4: 80, 5: 202, 6: 128, 8: 91, 9: 205, 12: 181, 15: 216, 18: 403}.

So a ring row carried about 7% of a policy gradient. Of that 7%, **more pointed at
the wrong legal action (4.17%) than at the right one (2.58%)**, and the wrong
action is always `eval_add` at a low site. Under a trained network, where visits
concentrate on one move, the target was **discarded entirely in 91.5% of states**:
every state whose best move is not `eval_add`.

### Why nothing caught it

* **Writer and reader were never tested together.**
  - The ring fixtures use synthetic `visit_actions = [5, 9]` on undecodable tokens
    (`test_replay.py:40`, `test_resume.py:38`) and never reach `ring_batch`.
  - `make_batch` is guarded (`test_train.py:107-121`), but that is the supervised
    path, which stores `action_index` correctly.
  - "Training on the ring moves the model" (`test_train.py:255-264`) passes on the
    value and steps gradients alone.
* **The types agreed and the coordinates did not.** Both sides held `int32`
  indices into *something*. This is the codec law's blind spot from the other
  side: a round trip proves two halves agree with each other, and here nothing
  even ran the round trip.
* **The mask was documented as a detector and acted as a silencer.** On *logits*,
  the policy loss's mask is the fabricated-target detector its docstring names. On
  *targets*, the same mask plus `nan_to_num` made illegal mass disappear. That is
  rider (b) in its purest form: the instrument existed, was populated, and could
  not speak.

### What it does and does not touch

**Stands.** Every *instrument* reading is a true measurement of its checkpoint,
because search, the ladder and the gates read priors through `action_index`. That
covers:
- P1: anchor 101/600 → campaign 209/600, CI [0.1367, 0.2217];
- the no-regress floors;
- top-1 0.9699 at every checkpoint;
- F-39's decomposition.

Phase 1 is untouched (`make_batch` one-hots the dataset's stored `action_index`).

**Compromised: the training-side conclusions, which now read as unexplained under
the defect, not false.**

* **F-37**, "forgetting is essentially immediate". Measured at `f = 0.00`, where
  every policy gradient came from ring rows: about 7% surviving mass, most of it
  pointed the wrong way.
* **AMENDMENT-m1a4 §5's `rehearsal_frac = 0.65`.** The screen compared arms whose
  ring policy signal was close to noise. Top-1 pinned at the anchor's 1192/1229 for
  `f ≥ 0.65` is what one would see if supervision were the only real policy signal
  in the batch, which under this defect it was.
* **Attempt 1's collapse**, 0.9699 → 0.8942 at `f = 0`.
* **What produced M1's +0.18.** The policy target the loop exists to learn from was
  absent. Candidates are the value and steps trunk gradients, the rehearsal rows,
  and the systematic push toward `eval_add`. This is an open question, not a finding.
* **Stale prose.** "Phase 2's improved-policy targets … wash it out"
  (`train.py:18-19`) could not have been true. The ring could wash nothing out.

### The fix

Lands in the next commit, tests written first:
- the writer stores `action_index` through one function;
- `RING_FORMAT` 2 redefines `visit_actions`, so format-1 records read as era-absent
  and are never reinterpreted;
- resume refuses a ring holding such records;
- the policy loss raises on target mass off the legal set, instead of dropping it.

The probe above is version-independent (it never assumes what `visit_actions`
means), and its post-fix reading is appended here when it exists.

**Not fixed here.** Even in correct coordinates, the target is the
Sequential-Halving visit share, not the Gumbel improved policy π′ the spec
promises. That is a registered one-lever change for later, not part of this
defect.

### Rule (proposed, not countersigned)

> **A coordinate conversion lives in exactly one function, and every writer/reader
> pair across a storage boundary is tested together on the real path, against
> ground truth captured from the producer.** A mask applied to *targets* raises; it
> never zeroes.

### Standing actions

1. **Re-measure under correct targets.** Re-run F-37's two `f = 0.00` arms and the
   `f` screen. These are minutes of compute, and F-37's 200/400-step pair should also
   control for warmup and cosine (its arms end at different LRs).
2. **M1's ring is repairable offline, if the volume survives.** Tokens are stored,
   and `legal_actions` order is part of the interface under `RULESET_VERSION 1`, so
   each slot maps back to its action exactly. Repair before any reuse rather than
   discard. The resume refusal deliberately blocks reusing it unrepaired.

### Amendment (2026-11-10, after independent review): four corrections, a defect in the fix, and the post-fix reading

The fix at `23939c6` was reviewed through four lenses: consumers, ring format and
resume, the loss detector, and the tests with this record. Each medium or high
finding then went to its own skeptic. Corrections are appended here, not edited
into the text above.

**1. The direction of the surviving mass was a pooled reading, and it does not
transfer to M1.**
- The 2.58 / 4.17 / 93.25 split pools all ten suites under the uniform stub, and
  1,920 of its 3,365 records are scripted pars 7–10.
- M1's ring and ring-0 were built on `train_100k`: BFS par 1–6 only.
- Re-measured by stratum, with the pre-fix writer imported from an export of
  `1fc2473` (the probe records the source tree it imported):

| stratum | records | correct | wrong legal | dropped |
|---|---:|---:|---:|---:|
| `bfs`, par 1–6 | 1,445 | **6.00%** | 4.61% | 89.39% |
| `scripted`, par 7–10 | 1,920 | 0.00% | 3.84% | 96.16% |

In the stratum that stands in for M1, **correct exceeds wrong-legal**. So "more
pointed at the wrong legal action than at the right one", and "most of it pointed
the wrong way", are **withdrawn as statements about M1, ring-0 and F-37's arms**.

What holds in every cut is that **89–96% of the mass was dropped**, and that what
survived included a systematic push toward `eval_add` at low sites. Both strata
are still uniform-stub readings. The split under the trained anchor is
unmeasured: the anchor is not in this clone.

**2. "Under a trained network … 91.5% of states" named the wrong mechanism.**
- At `sims 48`, `gumbel_m 16`, root visit shares are set by the Sequential
  Halving schedule, so "visits concentrate on one move" does not describe this
  search.
- The correct statement is network-independent. On **1,094 of the 1,200**
  `solve_in` start states (91.2%), plus 403 of the 800 `scripted_in` ones, **no
  storable slot lands on a legal cell at all**. The whole target is dropped
  whatever the network and whatever the budget.
- The 1,098 / 1,200 optimal-move reading stands as what it measured: where BFS's
  optimal first move lands.

**3. "Writer and reader were never tested together" was false.**
`a_filled_ring` → `train_on_ring` ran them together (`test_train.py:229-291`, which
needs `train_100k`). Nothing compared the target `ring_batch` built with the search
that produced it. That is rider (b), a mute instrument, not a missing one. The
rule's wording already asks for the comparison: "tested together … against ground
truth captured from the producer".

**4. The fix commit overstated its tests.** `23939c6` says every new test failed
pre-fix. The refusal-side tests did. The two accepting-polarity tests pass on both
sides, which is their job:
- `test_policy_loss_accepts_a_target_that_is_entirely_legal`;
- `test_resume_accepts_a_ring_written_in_the_current_format`.

**A defect in the fix, confirmed by three lenses and fixed at `d8ea7af`.**
- *What was wrong.* The refusal lived only in `resume`. `train_on_ring` on a
  pre-fix ring sat every record out and returned `steps == 0` silently.
  `sweep_rehearsal_frac.py` and `sweep_verify.py` load ring-0 directly, which is
  exactly what standing action 1 re-runs. `sweep_verify` would have printed
  identical digests and blamed the steps knob.
- *Now.* `train_on_ring` raises (`RingError`, F-41) before it draws from any RNG.
  `resume`'s check moved into a read-only `check_resumable`, which reads only the
  meta and the per-record format column. `scripts/campaign.py` calls it before it
  overwrites `campaign_summary.json`. That was the review's ordering finding: the
  refusal had arrived after a completed run's summary was already replaced.
- *Evidence.* The new refusal test fails against `23939c6` ("DID NOT RAISE
  RingError"), and its accepting partner passes there.

**Post-fix reading.** Same probe, `d8ea7af`, clean tree, same 200 problems and
3,365 records:

| | correct | wrong legal | dropped |
|---|---:|---:|---:|
| stored visits (48,618, the same total as pre-fix) | **48,618** | 0 | 0 |
| visit mass, both strata | **100%** | 0 | 0 |

- Target mass on legal cells is 1.0 in every record (mean and minimum).
- The target's argmax is a max-visit action in **3,365 / 3,365** records, against
  353 / 3,365 before.

**Suite in this cloud clone at `d8ea7af`.** 890 passed, 22 failed, 24 skipped.
The 22 failures are identical, by name, to the baseline's at `1a434ac` (869
passed). Every one needs gitignored data absent from a clone
(`runs/data/*/tokens.i32`, `runs/phase1/phase1.pt`), so the data-dependent ring
and supervision tests still need a run on the box that holds the data.

**Standing action 1, amended.** Regenerate ring-0 under the fixed writer
(`scripts/ring0_replay.py`) before re-running the sweep. `train_on_ring` now
refuses the old one by name.

**Standing action 3, new.** The policy loss now raises on supervision rows too.
`test_batch_targets_are_legal_actions` checks 64 sampled rows of `phase1_train`'s
311,741. A campaign samples far more of them, so a raise on a supervised batch
would be an older dataset defect surfacing, not this fix misfiring. A full-dataset
legality pass on the box would settle it in advance.

---

## F-42 — The policy target was the halving schedule's shape, not the improved policy

**Found:** 2026-11-10. It was a porting divergence surfaced by the
TinyAlphaZero-v2 comparison, and F-41 carried it as "not fixed here".

**Commissioned** by the principal, verbatim: *"now fix the Gumbel π' target"*. The
rollout was then chosen from three offered options, verbatim:

> *"π′ default, flag kept (Recommended)"* — *"Add train.policy_target with values
> 'improved' (new default) and 'visits' (kept as the control arm for a later
> A/B). default.yaml and both pinned fingerprints move once, with a dated
> amendment explaining why. The next campaign gets π′ without further edits."*

**Instrument:** `scripts/f42_policy_target_probe.py` at `0ec8e30`, clean tree,
torch 2.13.0+cpu, `sims 48`, `gumbel_m 16`. It ran 200 frozen suite start states
(20 per suite, via `sample_indices`) under three evaluators: the uniform stub, and a
fixed random-init network at value contribution 0.0 and at 1.0.

### The divergence

**The spec promises an improved-policy target** (spec §1 thesis and §5 Phase 2).
Gumbel MuZero's improved policy, and TinyAlphaZero-v2's training target
(`gumbel.py:241-256`), is

  π′ = softmax(logits + σ(completed Q)), over **every** legal root action,

with σ(q) = (c_visit + max N) · c_scale · q.

**What reckoner computes is not that.** `SearchResult.improved_policy()`
(`search.py:106-110`) returns `visits / visits.sum()`, the Sequential-Halving visit
share. The ring stores exactly that, and `ring_batch` expands it. It also feeds the
`entropy_target_*` columns (`runner.py:178`), whose schema text reads "H of the
**search-improved** target". The method's name claims the thing the spec promises;
its body computes another.

### Measured

| reading | result |
|---|---|
| **Schedule census.** Sorted root visit profile, grouped by actions visited, over 600 roots (200 states × 3 evaluators) | **exactly one profile per group, under every evaluator**, in all 11 groups: 1 → [48], 2 → [24, 24], 3 → [32, 8, 8], 4 → [18, 18, 6, 6], 5 → [20, 19, 3, 3, 3], 6 → [26, 8, 8, 2, 2, 2], 8 → [14, 14, 6, 6, 2×4], 9 → [17, 16, 5, 5, 1×5], 12 → [19, 7, 7, 3, 3, 3, 1×6], 15 → [20, 6, 6, 2×4, 1×8], 16 → [11, 11, 5, 5, 2×4, 1×8] |
| **Ceiling.** Largest share on one action | 32/48 = **0.667** on any multi-action root; a two-action root always teaches **24/24**. Mean top share **0.4519**, identical to four places under all three evaluators |
| **Proven-worse mass, start states.** Roots with two visited children proven by the search's terminal oracle at different z | **0 of 600**: start states almost never hold two depth-1 proofs, so this reading is empty, not reassuring |
| **Proven-worse mass, constructed.** All 40 five-action `solve_in_1` states, re-posed one step before the step cap with par = cap, so the solver is an exact at-par solve (z = 0) and the other four hit the cap (z = −1) | the search **chose the solver 40 / 40**. The target put **58.33% (28/48)** of its mass on proven losses **in every one**, 19/48 of it on a single losing action (visits [19, 3, 3, 3, 20]) |

So the target's **shape** is fixed by the schedule and the visited count. Only
**which action gets which share** carries information from the search, through
Sequential Halving's ranking (logits + Gumbel + σ(q)). Nothing in the target
records how much better an action proved: a proof of failure and a proof of
success that both survive a halving round are taught at the same share.

### What follows from it

* **The target-entropy columns were mute.** `entropy_target_*` is the entropy of
  a profile fixed by the visited-count mix. The target half of PREREG-m1 §6's
  funnel signature could move only when that mix moved (F-41 review, A1-02).
* **What π′ will and will not do in this domain (derived from the formula, not
  measured).** Against exact par, an at-par solve has z = 0, and while the value
  head is untrusted (`value_scale` 0) every unproven action also reads q = 0. So
  π′ departs from the prior only where the search holds a proof: z = −1 (over par,
  step cap, dead end) or z = +1 (beating non-exact par). Everywhere else π′ is the
  prior, and the policy gradient is about zero. That is Gumbel's intended
  behaviour, no improvement where search found none, and it replaces a target
  that sharpened the prior toward Gumbel-perturbed halving survivors whether or
  not the search had learned anything. Expect the policy signal to be sparse until
  the value head goes live.
* **M1's numbers are untouched.** They are instrument readings (F-41), and M1's
  training targets were already compromised by F-41's coordinate defect.

### The fix, as chosen

- `train.policy_target` is added: `improved` is the new default (π′, as specced),
  and `visits` is kept as the control arm for a later A/B.
- π′ is computed at the end of the search, where the tree is available, and stored
  in the ring.
- `default.yaml` and both pinned fingerprints move once, with dated notes.
- M1 is closed. The next campaign's PREREG registers the new fingerprint before
  launch.

The fix follows in later commits, tests first, and its post-fix reading is
appended here.
