"""F-40's sweep, rebuilt on DECLARATION (F-44).

**The first build enumerated citations with a regex over backticked paths and
failed its own reference vector**: attempt 1's ring is cited in prose — *"attempt
1's ring died with the pod"* — and no path-extractor can see it. That failure is
recorded rather than fixed away, because it demonstrates page-first was
**necessary and insufficient**, which is stronger than either half.

**Extraction infers intent from form; declaration records it.** Every extractor
in this project has failed at the boundary of what its author remembered to
encode, and there is no fourth regex that changes that. So:

* **Pages DECLARE their comparands** in a ``## Comparands`` block. That block is
  the sweep's only authority.
* **The extractor survives as a LINT**, not an authority: it flags path-shaped
  strings a page uses and has not declared. Its job changed; it was not retired.
* **Its false negatives are named, not hidden.** A prose-only citation remains
  uncatchable by machine. That discipline is human and belongs in the
  page-writing habit: *if a sentence rests on an artifact, the artifact goes in
  the block.*

**The forcing function** is the completion criterion of the retrofit — a census
of the past, L7's sibling: **every committed page carries a block**, and an empty
one must say why (absence with a reason). A page with no block is a DEFECT, which
is what keeps ``## Comparands`` from becoming a heading nobody fills in.

**Registration is partitioned, never bulk.** Pinning something unverified pins
whatever state it happens to be in — including, for anything predating a
fingerprint move, a state that may not match what the citing page assumed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from reckoner.dataset import read_anchors

REPO = Path(__file__).resolve().parents[1]

#: Pages that CITE — the normative and narrative record. `FINDINGS.md` is
#: included (unlike in the config census, where writing about a defect would have
#: counted as specifying it): a finding citing an artifact as evidence relies on
#: it, and reliance is what F-40 attaches to.
PAGE_GLOBS = (
    "PREREG*.md",
    "BRIEF*.md",
    "REHEARSAL*.md",
    "SWEEP*.md",
    "RUNLOG*.md",
    "AMENDMENT*.md",
    "GATE*.md",
    "FINDINGS.md",
    "ERRATA*.md",
)

DECLARATION = re.compile(r"^## Comparands\s*$", re.M)

#: A declared line: ``- `path` — note`` or ``- **GONE:** ... `path` ...``
DECLARED_PATH = re.compile(r"^\s*-\s+.*?`([A-Za-z0-9_./*-]+)`", re.M)

#: LINT ONLY. Path-shaped strings in the body — no longer the authority.
LINTABLE = re.compile(
    r"`([A-Za-z0-9_./-]*runs/[A-Za-z0-9_./*-]+|[A-Za-z0-9_.-]+\.(?:pt|jsonl|json))`"
)

#: LINT-ONLY skips: patterns and examples the body mentions but does not rely on.
#: **Deliberately not applied to declarations.** A page's block is written by
#: hand, so everything in it is intentional — and the first thing this filter ate
#: was the declaration of attempt 1's ring, whose path is a glob. A filter
#: written for an extractor must not run over a declaration.
LINT_SKIP = ("*", "runs/m1/", "runs/<", "a_future", "example")

#: A declared entry naming a FAMILY (``ring-*``) rather than one file, resolved
#: as a glob. Resolving to nothing means GONE, which is the point.
GLOBBY = re.compile(r"[*?\[]")

#: A declared entry naming a record SHAPE (``runs/<name>/iterations.jsonl``),
#: which has no instance to check and is reported as such rather than as gone.
SHAPE = re.compile(r"<[a-z_]+>")

#: The fingerprint move M1-A4 landed. Anything cited only after it was cited
#: under the CURRENT config; anything cited before may rest on a superseded one.
LAST_FINGERPRINT_MOVE = "9f9691e"


def _declaration_span(text: str) -> tuple[int, int] | None:
    m = DECLARATION.search(text)
    if not m:
        return None
    return m.end(), len(text)


def declared() -> tuple[dict[str, set[str]], list[str]]:
    """``({artifact: {declaring pages}}, [pages with NO block])``."""
    out: dict[str, set[str]] = {}
    undeclared_pages: list[str] = []
    for page in _pages():
        text = page.read_text()
        span = _declaration_span(text)
        if span is None:
            undeclared_pages.append(page.name)
            continue
        for path in DECLARED_PATH.findall(text[span[0] : span[1]]):
            if "." not in path and "/" not in path:
                continue  # a prose token in the note, not a path
            out.setdefault(path, set()).add(page.name)
    return out, undeclared_pages


def lint_undeclared() -> dict[str, set[str]]:
    """Path-shaped strings a page USES in its body but has NOT declared.

    The extractor's new job. Not authority: a clean lint proves nothing about
    prose-only citations, and the docstring says so rather than the result
    implying otherwise.
    """
    out: dict[str, set[str]] = {}
    for page in _pages():
        text = page.read_text()
        span = _declaration_span(text)
        body, block = (text[: span[0]], text[span[0] :]) if span else (text, "")
        block_paths = set(DECLARED_PATH.findall(block))
        for match in LINTABLE.findall(body):
            if any(s in match for s in LINT_SKIP) or match in block_paths:
                continue
            out.setdefault(match, set()).add(page.name)
    return out


def _pages() -> list[Path]:
    seen: dict[str, Path] = {}
    for glob in PAGE_GLOBS:
        for page in REPO.glob(glob):
            seen[page.name] = page
    return [seen[k] for k in sorted(seen)]


def _cited_before_move(artifact: str, pages: set[str]) -> bool:
    """Was this artifact's citation present at or before the last fingerprint move?

    Answers from the RECORD — ``git log -S`` on the artifact string — rather than
    from the file's mtime, which says when bytes changed and nothing about when a
    page came to rest on them.

    **Searched across ALL paths, not just the declaring page.** The retrofit moved
    every citation into a block written today; keyed to the declaring page, every
    artifact would date from the retrofit commit and the partition would report
    one era for everything. What is being dated is when the RELIANCE entered the
    record, and that is the original citation wherever it first appeared.
    """
    try:
        out = subprocess.run(
            ["git", "log", "--format=%h", f"-S{artifact}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # unknown provenance is treated as superseded: the safe side
    commits = out.stdout.split()
    if not commits:
        return True
    oldest = commits[-1]
    anc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", oldest, LAST_FINGERPRINT_MOVE],
        cwd=REPO,
        capture_output=True,
    )
    return anc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "comparand_sweep.json")
    parser.add_argument(
        "--partition",
        action="store_true",
        help="classify unregistered comparands by fingerprint era (runs git log per artifact)",
    )
    args = parser.parse_args()

    registered = set(read_anchors())
    declarations, pages_without_block = declared()
    lint = lint_undeclared()
    pages = _pages()

    print(
        f"\n  COMPARAND SWEEP (declaration-based, F-44) — "
        f"{len(declarations)} artifacts declared across {len(pages)} pages\n"
    )

    # --- The forcing function -------------------------------------------------
    print("  RETROFIT COMPLETION — every committed page carries a block:")
    if pages_without_block:
        print(f"    DEFECT: {len(pages_without_block)} page(s) with NO ## Comparands block")
        for name in pages_without_block:
            print(f"      - {name}")
    else:
        print(f"    all {len(pages)} pages declare (empty blocks carry a reason)")

    # --- The declared comparands ---------------------------------------------
    gone, unregistered, ok, shapes = [], [], [], []
    for artifact, citing in sorted(declarations.items()):
        if SHAPE.search(artifact):
            shapes.append({"artifact": artifact, "pages": sorted(citing)})
            continue
        if GLOBBY.search(artifact):
            # A family. Resolving to nothing is GONE — this is the branch that
            # must return attempt 1's ring, and the sweep's reference vector.
            matches = sorted(REPO.glob(artifact))
            exists = bool(matches)
        else:
            matches = sorted(REPO.glob(f"**/{Path(artifact).name}"))
            exists = (REPO / artifact).exists() or bool(matches)
        covered = artifact in registered or any(
            str(m.relative_to(REPO)) in registered for m in matches
        )
        row = {
            "artifact": artifact,
            "pages": sorted(citing),
            "exists": exists,
            "registered": covered,
        }
        (ok if exists and covered else unregistered if exists else gone).append(row)

    print("\n  DECLARED COMPARANDS:")
    if shapes:
        print(f"    record SHAPES (no instance to check) : {len(shapes)}")
    print(f"    present, registered   : {len(ok)}")
    print(f"    present, UNREGISTERED : {len(unregistered)}")
    print(f"    DECLARED AND GONE     : {len(gone)}")

    if gone:
        print("\n  DECLARED AND GONE — a page rests on it and it does not exist:")
        for row in gone:
            print(f"    - {row['artifact']}   declared by {', '.join(row['pages'])}")

    # --- Partition, because bulk registration pins the unverified -------------
    if args.partition and unregistered:
        print("\n  PARTITION — registration disposition, never bulk:")
        clean, superseded = [], []
        for row in unregistered:
            tgt = superseded if _cited_before_move(row["artifact"], set(row["pages"])) else clean
            tgt.append(row)
            row["era"] = "superseded" if tgt is superseded else "current"
        print(f"    register cleanly (cited since {LAST_FINGERPRINT_MOVE}) : {len(clean)}")
        print(f"    --as-of treatment: digest AND fingerprint            : {len(superseded)}")
        for row in superseded[:12]:
            print(f"      ~ {row['artifact']}")

    # --- The lint, with its limit stated -------------------------------------
    print(f"\n  LINT — path-shaped strings used but not declared: {len(lint)}")
    for artifact, where in sorted(lint.items())[:15]:
        print(f"    ! {artifact}   in {', '.join(sorted(where)[:2])}")
    if len(lint) > 15:
        print(f"    ... and {len(lint) - 15} more")
    print(
        "\n  The lint's false negatives are NOT bounded by this number. A prose-only\n"
        "  citation is invisible to it by construction — the reason this sweep is\n"
        "  declaration-based at all. A clean lint means the declaration and the\n"
        "  path-shaped prose agree; it says nothing about the sentences."
    )

    args.out.write_text(
        json.dumps(
            {
                "basis": "declaration",
                "last_fingerprint_move": LAST_FINGERPRINT_MOVE,
                "pages_without_declaration": pages_without_block,
                "declared_and_gone": gone,
                "declared_and_unregistered": unregistered,
                "declared_and_covered": ok,
                "declared_record_shapes": shapes,
                "lint_undeclared_paths": {k: sorted(v) for k, v in sorted(lint.items())},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\n  wrote {args.out}")
    return 1 if pages_without_block or gone else 0


if __name__ == "__main__":
    raise SystemExit(main())
