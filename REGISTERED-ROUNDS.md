# Registered rounds

One-lever rounds that are **specified before they are run**, so the evidence
standard is fixed while nobody yet knows the answer. A round registered here is
not a plan to change something — it is a plan to *test* changing it, with the
kill condition written down first.

The plan's chunk 12 registers the programme-level rounds (Phase-3 adversarial
generator, grid-bit vs token embeddings, rule-set extensions, the FOL
specialist). This file registers the ones that came out of implementation, with
the measurements that motivated them.

**Namespace.** Registry entries are `ROUND-NN`. Review *rulings* are referred to
by date and topic, never by number — two R-prefixed numbering schemes in one
project is the same defect class as two spellings of one key, and it already
cost a paragraph of disambiguation once.

---

## ROUND-01 — Remove `add_both_sides`

**Registered:** 2026-08-14, from chunk 2.
**Status:** registered, not run. `add_both_sides` stays in rule set v1.

**Claim.** `add_both_sides` is *reachability-redundant*: it only ever grows both
sides of an equation, so no problem needs it and par is identical without it. It
is pure branching tax.

**Motivating measurement** (chunk 2, three problems, BFS both ways):

| problem | par with | par without | BFS states with | without |
|---|---|---|---|---|
| `3x + 6 = 21` | 3 | 3 | 243 | 35 |
| `2x − 4 = 10` | 3 | 3 | 243 | 35 |
| `5x + 3 = 2x + 18` | 5 | 5 | 9,378 | 167 |

**It is not merely redundant — it is an active trap** (chunk 4). A scripted
policy that merely *prefers* `add_both_sides` never terminates: the rule is
always legal on any equation and only ever grows the state, so preference alone
is an infinite loop. Chunk 4's first fixture draft produced six 12-step runaways
from exactly this, and the coverage manifest is what caught them — no amount of
reading the derivations would have. That promotes the rule from "costs branching"
to "costs branching *and* is a hazard for every scripted solver, curriculum
policy and heuristic opponent that will be written against this rule set."

**Why it is not simply removed.** Plan §8 decision 2 pins the rule set. Removal
is a spec change and therefore a registered round, not an edit.

**Suite-level evidence, measured (chunk 5).** Across **920 BFS-optimal
derivations** spanning all six suites, all three goals and depths 1–6,
`add_both_sides` appears in **0 of them — 0.0%**. Every other rule is live:

| rule | share of optimal derivations |
|---|---:|
| `eval_add` | 76.1% |
| `combine_like_terms` | 45.8% |
| `eval_mul` | 45.2% |
| `div_both_sides` | 37.2% |
| `sub_both_sides` | 28.3% |
| `eval_sub` | 21.7% |
| **`add_both_sides`** | **0.0%** |

One caveat that keeps this evidence and not proof: participation is measured on
the derivation BFS *returned*, which is one optimal derivation per problem, not
all of them. A rule at 0% here has not been shown unusable — only unused by the
paths BFS reached first. The round's par-with vs par-without comparison is still
the deciding test; this is the prior that makes it worth running.

**Evidence standard — three fixtures do not prove a universal.** The set-growth
law applies: the round's evidence is **par-with vs par-without over the full
chunk-5 suite**, BFS computed both ways on every problem. Prediction: identical
par on every problem. **Any single counterexample kills the round** — it would
mean `add_both_sides` is reachability-relevant and the claim above is false.

**Third argument, measured in chunk 6 (`FINDINGS.md` F-05).** It is the *sole*
driver of state growth: reachable states reach 295 tokens / 104 sites with it and
**57 / 15** without. Since it forces `seq_len` and `max_sites` roughly 5× and 7×
wider than the domain otherwise needs, and attention is O(L²), removing it is
worth roughly a 25× trunk-cost multiplier. The round is no longer only about
branching.

**It is the first argument that costs money rather than elegance.** The other
two are about correctness and ergonomics; this one is (295/57)² ≈ **26.8× of
attention cost** and 104/15 ≈ **6.9× of policy-head width**, paid on every
forward pass of every campaign.

**Declared dependency — and it has grown twice.** Removing the rule changes
branching, so this round must re-measure branching and re-derive chunk 7's
`(sims, m)`. It also changes the reachable-state extent, so it must re-run
`scripts/measure_state_extent.py` and re-decide `seq_len` / `max_sites` — which
makes it a **model-resizing** round with checkpoint-compatibility consequences:
every snapshot taken before it indexes a different action space, and
`load_checkpoint` will refuse them. This is no longer a one-line deletion under
any reading. It must (`scripts/measure_branching.py`) and re-derive chunk
7's `(sims, m)` gate arithmetic against the new numbers. It cannot be run as an
isolated deletion.

**Do not let the claim leak early.** BFS-for-par runs the **full v1 rule set**
until this round fires and passes. Par is denominated in a rule system; quietly
computing it against a reduced set would make every recorded par a label for a
system that does not exist.

---

## ROUND-02 — `eval_div` plus a structural `div_both_sides`

**Registered:** 2026-08-14, from chunk 2.
**Status:** registered, not run. Separate from ROUND-01; do not bundle them.

**Claim.** Adding `eval_div` (evaluate a `DIV` node with exact integer operands)
would let `div_both_sides` become structural like the other movers — emitting
`DIV(lhs, a)` / `DIV(rhs, a)` and deferring the arithmetic — restoring the
"structural rules move, `eval_*` rules compute" symmetry exactly.

**Why v1 does not do this.** With no `eval_div`, a deferred quotient is a dead
end: nothing in v1 reduces a `DIV` node. So `div_both_sides` computes `c // a`
inside its exactness guard, and it is the one asymmetric rule in the set.

**What the round would cost.** Three things it would break, all of which must be
re-established as part of the round rather than assumed:

1. **The v1 DIV-free invariant.** Today no rule constructs a `DIV`, so a
   `DIV`-free problem generates a `DIV`-free reachable state space
   (`test_no_rule_ever_constructs_a_div_node`). ROUND-02 breaks that by construction.
2. **The field-only licence that invariant buys.** Division is the only partial
   operation — it is what makes `eval_field` return `None`. Chunk 3's SIMPLIFY
   checker compares by field equivalence alone *because* no state contains a
   `DIV` and so no draw is ever undefined
   (`test_div_free_states_never_evaluate_to_undefined`). ROUND-02 reintroduces
   undefined draws, and the checker would have to skip-and-count them.
3. **Par everywhere.** Every solve gains a ply. `RULESET_VERSION` bumps, and
   every recorded par is invalidated.

**Evidence standard.** Par and branching re-measured over the full chunk-5
suite; the SIMPLIFY checker re-specified with skip-and-count and its own
anti-vacuity floors; the interpreter's rendering re-proofread. The round is
justified only if the granularity symmetry is worth a ply on every problem plus
a partial checker.

**Generator ban stands meanwhile.** Chunk 5's v1 generator must not emit `DIV`
in problems. The `DIV` token itself stays in the vocabulary — chunk 1's
id-stability principle: ids never move, so an unused token is cheaper than a
renumbering.

---

## ROUND-03 — Vendor the canonical base-625 glyph renderer

**Registered:** 2026-08-14, from chunk 4.
**Status:** registered, not run. The panel ships as a labelled local placeholder.

**Claim.** `interpreter.glyph_panel` renders a base-625 digit as a 2×2 grid of
base-5 cells (625 = 5⁴). That is a *reckoner-local* convention invented here,
not the convention the base-625 and Symbolic-Transformers projects use.

**Why it is a placeholder rather than a claim of reuse.** The upstream renderer
was not present on this machine when chunk 4 was built, and inventing a
convention while describing it as the existing one would have been the worse of
the two available failures. But the panel's *purpose* is continuity with the
real base-625 substrate, so an unlabelled local convention is misleading in a
subtler way — it looks like the system it is not. Every render therefore carries
its own status line, and the artifact carries its epistemic tag as always.

**It was deferred, not impossible.** The upstream repository is public. This
round names the deferral so it stops being invisible.

**Evidence standard.** Vendor (or depend on) the upstream renderer, then show
the two conventions side by side on the same numerals. If they agree, the local
one was right by luck and the status line comes off. If they disagree, every
glyph panel rendered before this round was a different notation than it
appeared to be, and `docs/derivations.md` is regenerated and re-proofread.

**Scope note.** The derivation renderer does not depend on the panel, so this
round cannot change any derivation text — only the panels beside it.

---

## Comparands

- `runs/rule_participation.json` — the participation evidence rounds are scored on
- `docs/derivations.md` — the derivation record rounds amend

*The round register records DECISIONS about the ruleset. Its comparands are the
evidence each round was decided on, not the ruleset itself.*

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
