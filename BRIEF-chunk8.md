# BRIEF-chunk8.md — Phase 1: supervised warm start (2026-08-14)
# Re-supplied verbatim 2026-08-15 after session crash; commit before executing further items.

[... items 1–14 exactly as delivered previously — Part 0: errata block,
countersign flips, timing slice as true pilot with CPU/GPU decision;
Training design: BFS-derivation supervision with tie-break caveat, W/D/L
head masked weight-0 with reason per spec §5, natural depth mix per F-04,
problem-level split hygiene + ANCHORS digest verification, inherited kit
with declared seed-repeat tolerance; Gates: B_max table before the
depth-≤2 declaration, top-8 ≥ 0.90 depth ≤ 3 stratified 1–6, depth-≤2
solve ≥ 0.95 on permanent value wiring with zero-value ablation as
diagnostic, loss-curve reproducibility; Close: solve_in_1..6 @ 16 sims
informational baselines, anchor by digest with full provenance meta.
Three failed attempts at any gate → BLOCKED, never a weakened gate.]

## Amendments (2026-08-15, pre-run)
A1. Supervision contamination census BEFORE training: exact
    (identity_key, goal) overlap between suite start states and the
    313,628 supervision states, counts per suite verbatim. Overlap ≤ 1%
    of examples → remove collisions, re-digest, report removed count.
    > 1% → STOP; joint ruling. Supervision datasets get a permanent
    state-level contamination test; the narrowed skip is reverted as a
    design regression.
A2. This run: CPU. ROCm swap = chunk 9 Part 0 (blessed stack, GPU/CPU
    equivalence smoke, box-occupancy declaration).
A3. search.batch_leaves → search.batch_searches, provisional tag flipped.
A4. Extension bound: 5,000 steps; one extension to 10,000 max; then
    BLOCKED-<date>-<topic>.md.

---

## Recording note — the state of this file, stated rather than implied

Committed 2026-08-15 exactly as re-supplied, including the bracketed line
standing in for items 1–14. **The itemised text of items 1–14 is not
recoverable**: the original brief was delivered in conversation and never
committed, and the session crash took it. What survives is the summary above
plus the four items whose execution left a record — item 1 (errata), item 2
(countersign), item 4 (Phase-1 supervision build) and item 9 (B_max on
`solve_in_2`), all reconstructible from `git log`. Item 3 is confirmed by ruling
to be the timing slice.

This note exists because absence carries a reason. A brief that reads as
complete when it is a summary would let a later reader believe a gate was
registered that nobody can now quote — which is the same defect class as F-02,
one layer up in the governance stack. If a disagreement later surfaces between
this summary and any item as originally written, that is a report line, not a
silent merge.

**The law this incident bought** is recorded in AGENTS.md §7: a brief is
committed verbatim on receipt, before any item executes. Governance text that
lives only in a conversation is destructible. This was learned once with the
gate artifact and re-learned here; it is not being learned a third time.

---

## Comparands

**None.** A build brief; its outputs are the chunk-8 gate's artifacts, declared
on `GATE-chunk8-VERDICT.md` where they are relied on. Empty by decision.

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
