# GATE — chunk 4 manual proofread

The chunk-4 gate is manual by design: no automated check can certify that the
rule named on a line matches the transformation shown. This file is its record.

## Artifact under gate

| | |
|---|---|
| document | `docs/derivations.md` |
| sha256 (as passed) | `b2b6c932a7236c8299a24dec72d08fc787d791a34ed83f0caaa38219540a22e0` |
| prior issue (failed) | `f6a10c094dfe13ee2c7b94518c38b86866e9d56fe7a3bc826eb0c1394a304a2b` |
| derivations | 50 |
| commit at pass | `deef969` |

## What the proofread certifies

1. Every line is followable without the code.
2. The rule named matches the transformation shown, all fifty times.
3. Base-625 numerals — multi-digit and negative especially — render correctly.
4. Nothing was prettified into unfaithfulness. `21 + (−6)` is the state, not a bug.
5. The three ILLUSTRATIVE labels are present and honest; version stamps are on
   the page; no caption restates an equation; the glyph panel is a labelled
   placeholder.

## First issue — FAIL

Two defect families, both recorded in `FINDINGS.md`:

* **F-01** — derivation 19 shipped a suboptimal exhibit, unlabelled, under a
  caption that miscounted its own steps.
* **F-02** — six derivations claimed BFS-exact provenance for hand-written par
  literals, producing `z = +1` rows that are impossible by construction.

Root cause of F-02 was neither branch the review's diagnostic named: there was no
BFS labeller at all, and `Problem.par_source` defaulted to `"bfs"`. Audit of all
fifty: 6 mislabelled, 44 correct by luck.

## Second issue — PASS

**Verdict, verbatim as given:** `verdict: PASS. continue`

**Basis: closed, permanently blank by the principal's election.** Two forms were
offered — "Proofread in full — PASS" and "Accepted on diff plus changed entries
— PASS" — and the principal elected to record neither. The line is therefore
**closed**, not pending: it will not be filled later, and the record should not
be read as claiming either basis. Absence carries a reason, and the reason here
is a decision rather than an omission.

Changed entries at the second issue: derivations 19, 33, 36, 39, 42, 45, 48,
plus the errata header. All other entries diff-verify against the prior digest.

## Countersign — closed

Countersign authority was delegated to the reviewer (AGENTS.md §0). The registry
is **five earned lines and one inherited corpus**, and the two are not the same
kind of thing. Earned lines are countersigned individually and dated; inherited
law arrived whole, before chunk 0, and is not countersignable because it was
never up for adoption.

1. Round-trip gates are blind to symmetric bugs; every codec carries pinned
   absolute reference vectors.
2. A gate must report what it covered, not only that it passed.
3. A provenance field whose default is its strongest claim is not a provenance
   field — the most-trusted value must be the one that costs something to say.
4. One formatter of states, ever — a caption describes, or it calls `render()`.

**Countersigned later (2026-08-14 / 15):**

5. A gate suite assembled from known hazards has a blind spot exactly the shape
   of the component doing its job at all. Rider (a): the first gate written for
   any component measures it doing its central job. Rider (b) *(2026-08-15)*: a
   number nobody asserts on is not a gate; it is a comment that happens to be
   computed.
6. A brief is committed verbatim on receipt, before any item executes
   *(2026-08-15)*.

**Not on this list — inherited, not earned:**

*One lever per round is an executable property, not a reviewed one.* Principal's
registry ruling, 2026-08-15: it predates the earned registry, so it is neither a
countersign nor a ruling. Two prior claims about it are corrected rather than
left to stand — a chunk-8 report called it countersigned (it was not), and a
chunk-8-part-1 correction called it a ratified ruling (also not). The reported
discrepancy in AGENTS.md §6 stands with it: the principle predates the registry,
but the phrase is not in §1's verbatim inherited-law block, and its pre-registry
sources are the spec's §2 lean and the plan's chunk 12.

All were already in force in the code and tests; the countersign settles the
record, not the behaviour.

## Reporting erratum

The chunk-6 report was ordered to open with this artifact's hash and did not.
The cause was a misreading, not an abstention: "the verdict artifact" was taken
to mean the document under review, and `docs/derivations.md`'s digest was given
instead. Had the artifact been correctly identified as missing, the report was
required to say so — absence carries a reason applies to ordered report items,
not only to schema fields.

---

## Comparands

- `docs/derivations.md` — **the artifact under gate**, pinned at
  `b2b6c932a7236c8299a24dec72d08fc787d791a34ed83f0caaa38219540a22e0`

*A `.md` artifact under a sha256 pin: cited with its digest on the page and
invisible to a path-extractor keyed to `runs/` and data extensions (F-44).*

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
