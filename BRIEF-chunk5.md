# Chunk 5 brief — accumulated fold-ins

Chunk 5 is *generator, scripted solver, datasets, suites*. Its DONE-WHEN is in
`experiment2_agent_plan.md`. This file carries the constraints and instruments
added by review across chunks 2–4, so none of them has to be remembered.

**Blocked** until the chunk 4 proofread returns a verdict.

---

## Emission grammar — what a v1 problem may contain

These are closure constraints, not style. Each one names a construct the rule set
cannot reduce, so emitting it produces a problem that is unsolvable *by
construction* — and a suite of unsolvable problems looks exactly like a suite of
hard ones.

1. **No `DIV` in any emitted problem.** There is no `eval_div` in v1
   (`REGISTERED-ROUNDS.md` ROUND-02). Already enforced at the boundary:
   `Problem.__post_init__` rejects it. With the ban in place, no reachable v1
   state contains a `DIV`, which is the licence chunk 3's SIMPLIFY checker uses
   to compare by field equivalence alone.
2. **No variable-containing `SUB` subtree, any goal.** Same family as the `DIV`
   ban. `eval_sub` needs both operands numeric, the movers ignore `SUB`, and
   `combine_like_terms` cannot see through it — so `21 − 2x = 3` and `3x − x` are
   dead ends. **Not yet enforced in code; chunk 5 adds the check.**
3. **Numeric-only `SUB` must be emitted deliberately.** No rule *constructs* a
   `SUB`, so if the generator never emits one either, `eval_sub` becomes a dead
   rule whose soundness fuzz passes in a vacuum forever.
4. **Variable policy, stated per goal.** `y` and `z` exist and are reachable
   (`combine_like_terms` needs the unlike-variable case). State explicitly which
   goals may emit which variables, and record that **SOLVE-for-x with `y` present
   is out of v1 scope** — the rule set has no way to eliminate a second unknown.

## Par

5. **Par is computed, never asserted.** Use `episode.bfs_par` /
   `bfs_solution`, which call `verify()` — a labeller with its own terminal test
   is a second definition of "solved". See `FINDINGS.md` F-02 for what the
   alternative cost.
6. **`par_source` is explicit on every emitted problem.** The default is
   `unverified` and must stay that way; exactness is asserted, never inherited.
6a. **`unverified` may exist at construction but must never ship.** The dataset
   and suite writers assert `par_source != "unverified"` and `par is not None` on
   every row. The weakest honest state is allowed to *be*; it is not allowed to
   *leave*. A shipped row with an unlabelled par is an unlabelable problem
   wearing a label field, and every consumer downstream will read it as one.
6b. **Emitted problems carry `par >= 1`.** A par-0 problem is already in goal
   form — terminal at birth, nothing to learn from, and a free +0 in any average.
   `par=None` means unlabelled and `par=0` means solved-on-arrival; they are
   different facts and neither may stand in for the other.
7. **BFS-for-par runs the full v1 rule set**, including `add_both_sides`, until
   ROUND-01 fires and passes. Par is denominated in a rule system; computing it
   against a reduced set would make every recorded par a label for a system that
   does not exist.
8. **On the depth ≤ 6 overlap, compute both BFS and scripted par and report the
   delta distribution.** Free calibration of how provisional the mid-strata floor
   really is. `FINDINGS.md` F-01 is the only data point so far: one step at
   depth 3.
9. **`ruleset_version` and `vocab_version` in every dataset and suite
   `meta.json`**, beside the git SHA and config fingerprint. A par without its
   rule-system version is a number, not a label.

## Required pre-flight

**Before generation starts, project the total labelling cost.** Measure `label()`
on a sample, multiply by the 100K training count plus 1,200 suite problems, and
report the projection — the timing-slice discipline, applied where the real cost
now lives. BFS-exact par is not free: the chunk 4 document's fifty labels take
~3.4s, and the test suite went 44s → 73s the moment real labelling landed. That
is the first invoice, and it was for fifty problems.

If the projection is hours, that is a fact to decide against before generating,
not to discover at 40% completion. Record the measured per-problem cost by depth
— cost grows ~5× per ply at the measured branching, so the depth-6 stratum
dominates the bill and the projection must be *stratified*, not an average.

## Instruments

10. **Per-rule participation histogram over BFS-optimal derivations**, across the
    100K set. Three instruments in one: a rule-liveness audit (a rule at ~0% is
    dead weight regardless of its green fuzz), the suite-level evidence base
    ROUND-01 needs, and an early look at what the policy actually has to learn.
11. **Resolve `no_legal_actions`'s epistemic status.** Chunk 3 tests it directly
    but never reached it from a generated problem. Either prove the emission
    grammar makes dead ends unreachable — then the terminal reason gets a comment
    saying it is defensive, with the proof sketch — or exhibit one reachable
    instance. Status assigned, not left floating.
12. **Re-check chunk 7's gate arithmetic against chunk 5's real distribution.**
    The branching numbers in `benchmarks/results.jsonl` were measured on
    disclosed stand-in samplers, because no generator existed.

## Laws that bite here

13. **One formatter of states, ever.** Captions and labels describe, or they call
    `render_expr()`. No hand-formatted state anywhere.
14. **Every codec carries pinned absolute reference vectors** — round-trip gates
    are blind to symmetric bugs.
15. **A gate reports what it covered**, not only that it passed.
16. **Flipping a `[provisional — chunk 5]` config tag to decided-with-source is
    part of chunk 5's DONE-WHEN.** `generator.*` and `ladder.problems_per_pass`
    are the ones in scope.

---

## Comparands

- `meta.json`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
