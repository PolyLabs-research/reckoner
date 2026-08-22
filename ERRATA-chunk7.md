# Errata — chunk 7 (search core)

**Chunk 7's verdict is not reinterpreted here.** The gates as written were green;
the literal PASS stands. What this document records is that the suite had a hole
shaped like the component's purpose (`FINDINGS.md` F-06), that the descent gate
has joined the suite, and that the evidence base the PASS rested on was measured
through a search that has since been rewritten — so it has been re-measured.

Written 2026-08-14. Search under test: **post-F-06 (descending)**, HEAD of `main`
plus the working tree. Pre-F-06 comparisons are run from a worktree at `5b5fd61`.

---

## 1. The chunk-7 gate table, re-run under the fixed search

One table, verbatim, from `scripts/chunk7_gate_table.py` →
`runs/chunk7_gate_table.json`. Re-run twice; the JSON is byte-for-byte identical
across runs.

```
  CHUNK-7 GATE TABLE — re-run under the FIXED (descending) search

  budget identity
     sims   used  visits.sum  nodes  max_depth  identity
        6      6           6      7          2  HOLDS
       16     16          16     17          2  HOLDS
       31     31          31     32          3  HOLDS
       48     48          48     49          3  HOLDS

  2-seed eval-profile identity (root_noise=False): IDENTICAL  visits=[7, 6, 1, 1, 1, 0]
  cross-process identity (4 PYTHONHASHSEEDs): IDENTICAL  [[0, 3, 3, 20, 3, 19], [5, 2], 48, 49]

     m   depth-1 solved     rate
     3            180/200    90.0%
     5            200/200   100.0%
    16            200/200   100.0%

  StateTooLarge under descent (seq_len=26, sims=48): 27 counted, sims_used=48, visits.sum=48, nodes=43

  descent gate (branchy depth-5 SOLVE, 8 root actions)
     sims  nodes  max_depth  evals
        6      7          2      7
       16     17          2     17
       31     32          3     32
       48     49          3     49

  sequential/batched parity spot-check (4 searches, batch 3): EXACT
```

Row by row, against what each gate is now actually testing:

| gate | verdict | what changed under the fixed search |
|---|---|---|
| budget identity | HOLDS, 6/16/31/48 | the identity is unchanged — see §2, and that is the finding, not a coincidence |
| 2-seed determinism | IDENTICAL | **first genuine exercise.** Selection is now path-dependent; before, there was no selection to depend on anything |
| cross-process determinism | IDENTICAL, 4 hash seeds | same — and it now covers a descent order, not just a root ordering |
| depth-1 at m ≥ 5 | 90.0% / 100% / 100% | unchanged, and unchanged *for the right reason*: a depth-1 problem needs one ply either way. Both polarities still present |
| StateTooLarge under descent | 27 counted, budget intact | **first genuine exercise.** Previously "reachable" meant reachable by a search that never descended |
| descent gate (new) | nodes 7→49, max_depth 2→3 | the gate that did not exist |
| sequential/batched parity | EXACT | across trees, not within one |

Two rows deserve their numbers read rather than skimmed.

**StateTooLarge, at `model.seq_len = 26`, `sims = 48`:** 27 encode failures
counted, `sims_used = 48`, `visits.sum() = 48`, `nodes = 43`. The counted-terminal-loss
semantics is doing exactly what chunk 7 declared and could not previously
demonstrate: 27 of 48 simulations hit an oversized state deep in the tree, each
was charged as a terminal loss, the budget identity survived it, and the tree
still reached 43 nodes. Before the fix, nothing descended far enough for a state
to grow past the cap, so this path was declared and untested.

**The descent gate's `evals` column equals `nodes` at every budget** (7/17/32/49).
One evaluation per node created, no node evaluated twice — the property the old
search violated most loudly, and the cheapest single number to watch in future.

## 2. Erratum (b) resolved by measurement: which search produced the budget identity

The chunk-8-part-0 report gave budget-identity numbers without saying whether
they came from the old search or the new one. Post-F-06 that distinction is the
whole question, so it was measured rather than reasoned about. Same problem
(`3x + 6 = 21`, 6 legal root actions), same seed, same budgets, run from a
worktree at `5b5fd61`:

```
PRE-F-06 SEARCH (chunk 7, 5b5fd61)
 sims   used  visits.sum  nodes  evals  identity
    6      6           6      6      7  HOLDS
   16     16          16      6     17  HOLDS
   31     31          31      6     32  HOLDS
   48     48          48      6     49  HOLDS
```

**The budget-identity numbers are identical under both searches** — 6/16/31/48 →
6/16/31/48, HOLDS at every point, before and after. The report's ambiguity was
real and it resolves harmlessly, because the quantity does not discriminate: this
is precisely the mechanism F-06 named, that visits are counted whether or not the
simulation does work.

The column that discriminates is beside it. Under the old search `nodes` is
**6, 6, 6, 6** — flat across an 8× budget increase. The root has 6 legal actions
and `m = 5`, so the old search could create at most 5 children; `nodes` was pinned
at 6 for every budget it would ever be given. Under the fixed search the same
column reads **7, 17, 32, 49**.

**And `nodes` was already in `SearchStats` when chunk 7 shipped.** It was
recorded in every stats row, it was flat, and no gate asserted on it. That
sharpens F-06 by one turn: the blind spot was not missing instrumentation. The
instrument existed, was correct, and was un-asserted. A number nobody asserts on
is not a gate — it is a comment that happens to be computed.

## 3. Errata on F-06's own exhibit

Errata culture applies to the errata. Two claims in `FINDINGS.md` F-06 and in the
chunk-8-part-0 commit message do not survive re-measurement as written.

**(i) "`sims=48, m=5 -> nodes in tree: 2`" is under-specified to the point of
being undemonstrative.** The number is real, but it only shows a defect once the
problem is named. Census over all six frozen suites: **234 of 1,200 problems have
a single legal root action**, and on **160 of them — every B=1 root in
`solve_in_1` — `nodes = 2` is the complete and correct tree under *both*
searches**, because the one action solves the problem and a terminal leaf has
nothing below it. As published, the exhibit is indistinguishable from a correct
result.

The reproducible exhibit, with the problem named, is a **depth-2** single-action
root (74 such problems in `solve_in_2`):

```
                          nodes   evals   terminal_solved
  PRE-F-06   sims=6         2       7          0
  PRE-F-06   sims=48        2      49          0
  POST-F-06  sims=6         3       2          1
  POST-F-06  sims=48        3       2          1
```

The old search spent **49 evaluations re-evaluating the same single child 48
times and never reached the solve**. The fixed search descends one ply, finds the
terminal solve, and exhausts the subtree in **2 evaluations** — 24.5× fewer, at
8× the budget, with the answer found instead of missed. `nodes = 3` at both
budgets is correct here and not a repeat of the defect: the subtree is finite and
fully expanded, so more simulations cannot build more tree. The general signature
is `nodes` flat *while unexpanded actions remain*, which is why the branching case
(6 root actions, `nodes` pinned at 6 in §2) is the cleaner headline.

**(ii) "`SearchStats` gains `nodes` and `max_depth`" is wrong on `nodes`.** It
gained `max_depth`. `nodes` was present and populated in the chunk-7 code
(`search.py:182,230` at `5b5fd61`). Corrected claim: *`SearchStats` gains
`max_depth`; `nodes` already existed, read 6/6/6/6, and no gate looked at it.*

Both corrections are landed in `FINDINGS.md` F-06 and in the docstring of
`tests/test_search.py::test_the_tree_deepens_past_one_ply`, which repeated (i).

## 4. The test delta, decomposed — 540 = 538 + 2 conceals 18 added and 16 removed

`make lint test`: **540 passed in 127.51 s** (was 538). The net `+2` is the least
informative true statement available about a 244-insertion / 196-deletion rewrite
of `search.py`, so here is the decomposition — by collected test **ID**, not by
function name, so parametrisation is counted rather than eyeballed. Both sides
collected with the same pytest against their own tree (`5b5fd61` from a worktree,
HEAD from the working tree): 538 and 540 respectively, matching the reported
counts.

**Removed — 16 IDs, all one test's grid:**

| removed | cells | justification |
|---|---:|---|
| `test_search.py::test_sync_and_batched_agree_on_visits[m-sims]` | 16 | **Its subject was deleted, not its assertion weakened.** It compared `batch_leaves=1` against `batch_leaves=64` *within one tree*. Within-tree leaf pooling is now forbidden rather than parameterised: once selection is path-dependent, pooling leaves inside a tree changes what the tree does, so the comparison would be between two algorithms rather than two schedules. The old test was exact only because the old search had no selection to corrupt — it passed on a property the defect created. |

**Added — 18 IDs:**

| added | cells | what it buys |
|---|---:|---|
| `test_search.py::test_sequential_and_batched_agree_exactly[m-sims]` | 16 | the replacement claim: N searches through `run_batched` equal the same N run sequentially, **across** trees. Same m × sims grid, odd pairs included. No coupling to approximate away, so it stays an equality rather than becoming a tolerance |
| `test_search.py::test_the_tree_deepens_past_one_ply` | 1 | the gate F-06 was missing: more sims ⇒ more tree; tree exceeds root-plus-children; `max_depth ≥ 2`; node count scales with sims, not with `m` |
| `test_dataset.py::test_the_supervision_set_names_the_problem_set_it_inherits_from` | 1 | inherited contamination status is verified, not assumed — the recorded source digests must still equal `train_100k`'s digests on disk |

**Changes that moved no count at all**, listed because count-neutrality is exactly
what hides them:

- `test_sequential_and_batched_agree_exactly` **dropped one assertion** its
  predecessor carried, `sync.stats.batches >= batched.stats.batches`. Justified:
  `SearchStats.batches` no longer exists — batch accounting belongs to
  `run_batched`, which owns the pooling, not to a single tree's stats.
- It **gained one**, `a.stats.nodes == b.stats.nodes`. Tree shape is now part of
  what parity means; under the old search it would have been the constant 6.
- `test_no_suite_contamination` **narrowed**: datasets with
  `mode == "phase1_supervision"` are now skipped, because a supervision set holds
  states along derivations rather than problems and its rows are not comparable
  to a suite's. This is a real coverage reduction in a surviving test. It is
  compensated by the new inheritance test above and by nothing else — if that
  test is ever deleted, the supervision set becomes uncovered silently.
- Two `_backup` fixtures were rewritten for the new `_Tree` constructor
  (`tree.add(expr, parent, slot)` + `tree.open(...)`). Same assertions, same
  mixed three-level structure, same named wrong answers (−1.0 for negation,
  −0.0625 for mean backup).

**Ledger:** 538 − 16 + 18 = 540. One test removed with cause, one assertion
dropped with cause, one test narrowed with cause and a named compensator.

## 5. Records that were not records

Three gate records written by chunks 7–8 are matched by `runs/**` and are
therefore **untracked**, which is the exact failure `.gitignore`'s
MUST_REACH / MUST_IGNORE guard exists to prevent — and the guard did not catch
them because a rule that is never written is never tested:

- `runs/gate_arithmetic.json` — the B_max census the chunk-7 gate was declared on
- `runs/gate_arithmetic_d2.json` — the same for the depth-≤2 gate
- `runs/chunk7_gate_table.json` — this document's table

All three are cited as evidence by a report. Negations and MUST_REACH entries are
added in the same commit as this document.

## 6. Open, not closed: `search.batch_leaves` now names a discarded concept

`cfg.search.batch_leaves` (512, `[provisional — chunk 7]`) is still read — by
`run_batched`, where it sizes the pool **across concurrent searches**. Its name
means the thing chunk 8 part 0 forbade: pooling leaves within one tree. The key
therefore reads as a live setting for a discarded mechanism.

Renaming it is a config-key change, which AGENTS.md §7 makes a contract, so it is
recorded here rather than done: **recommendation — rename to
`search.batch_searches`, and flip the `[provisional — chunk 7]` tag at the same
time.** Awaiting the principal.

---

## Comparands

- `runs/chunk7_gate_table.json`
- `runs/gate_arithmetic.json`
- `runs/gate_arithmetic_d2.json`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
