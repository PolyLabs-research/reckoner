# ERRATA-chunk5.md — corrections to a shipped chunk-5 report

Nothing is fixed silently. The commit stands unedited — verdicts do not retro-edit
— and each correction files beside it with the record that settles it.

---

## E5-1 — `4b93c25` states a percentage inconsistent with its own numerator

The chunk-5 commit message reads:

> `SCRIPTED PAR CALIBRATED: optimal on 1176/1200 (97.7%), never more than one step`

**1176/1200 = 98.0%, not 97.7%.** The count is right and the percentage is wrong;
`runs/par_delta.json` has held the correct numerator since the day it was written.

Found 2026-08-15, by quoting the record instead of the summary while writing F-21.
It survived the principal's own audit of that report — an internally inconsistent
pair inside one line, which is the kind of error a reader's eye completes rather
than checks, because the numerator looks authoritative and the percentage looks
like its restatement.

**The species:** a derived figure and its source, printed together, drifting apart.
Neither number is hard to verify; the pairing is what made verification feel
unnecessary. Same family as F-16 — a number that was not computed at the moment it
was written down — and the same remedy applies: quote the file.

## E5-2 — the by-goal split, back-solved and wrong in the ledger

The principal's ledger recorded the goal split by assuming **all 24 misses were
SIMPLIFY**, which back-solves to a SIMPLIFY population of **360**.
`runs/par_delta.json` says otherwise:

| goal | gap 0 | gap 1 | population |
|---|---|---|---|
| EVALUATE | 450 | **7** | 457 |
| SIMPLIFY | 235 | 17 | **252** |
| SOLVE | 491 | 0 | 491 |

SIMPLIFY's population is **252, not 360**, because **EVALUATE carried 7 of the 24
misses**. Corrected in the ledger by the principal on receipt.

**And it vindicates the suspicion it came from.** The ask at the time was:

> "100% on SOLVE, 93.3% on SIMPLIFY conspicuously omits EVALUATE"

That was right. EVALUATE was not perfect, and its 7 misses are **F-01's greedy
eval-order defect** — visible in the by-goal table the whole time, in a column the
summary did not carry. The suspicion was correct; only the arithmetic used to
chase it was off, and it was off because it was applied to a summary rather than
to the table.

**Summaries drifted twice; `runs/par_delta.json` caught both.** That is the
argument for records over reports, made twice in one file by two different
authors.

---

## Comparands

- `runs/par_delta.json`

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
