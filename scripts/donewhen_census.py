"""F-48's census: every DONE-WHEN item, and whether anything could have failed it.

`BRIEF-m1-host.md` §0a required a pod-facts block committed. Its DONE-WHEN
asserts *"Pod-facts committed"*. No such file exists, and no page carries the
block. **The gate passed with nothing behind it.**

The file was the symptom. The finding is that **DONE-WHEN items are
self-attested** — every one is discharged by someone writing that it was, which
is the never-firable MUST_REACH one layer up: a check that cannot fail, because
satisfaction is asserted rather than demonstrated.

Two questions per item, and **the second is the real one**:

1. **Is there an artifact?** — does anything in the tree discharge it. Finds gaps.
2. **Would anything have NOTICED if there weren't?** — is the item connected to a
   test, a gate script, or a registered comparand. Finds the *class*.

An item can pass (1) and fail (2): the artifact exists because someone made it,
and nothing anywhere would have complained had they not. That is the population
this census exists to size, and it is expected to be most of them.

**This census does not adjudicate.** It classifies and counts; a human decides
what each unenforced item deserves. Third census of the past — after the config
census and record-coverage — and on the evidence of those two, the largest.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: DONE-WHEN blocks live in briefs; verdict pages record their discharge.
BRIEF_GLOBS = ("BRIEF*.md", "PREREG*.md")

HEADING = re.compile(r"^#+\s*DONE-WHEN.*$", re.M)
NEXT_HEADING = re.compile(r"^#+\s", re.M)
ITEM = re.compile(r"^\s*[-*]\s+(.+?)(?=\n\s*[-*]\s|\n\s*\n|\Z)", re.M | re.S)

#: Tokens naming a concrete, checkable artifact or mechanism.
ARTIFACTY = re.compile(
    r"`([A-Za-z0-9_./*<>-]+\.(?:py|json|jsonl|pt|md|lock|yaml|txt|i32|sha256)"
    r"|[A-Za-z0-9_./*-]*runs/[A-Za-z0-9_./*-]+)`"
)

#: An item that names a command or suite is enforced by running it. A COMMAND is
#: stronger enforcement than a file, so this is checked BEFORE artifacts —
#: the first build asked "any artifact?" first and short-circuited to
#: UNCHECKABLE, making the ENFORCED branch **never-firable**: a check that
#: cannot fire, inside the detector built to find checks that cannot fire.
ENFORCED_BY_COMMAND = re.compile(
    r"\b(make\s+\w+|pytest|golden|ruff|mypy|lint|suite|battery)\b", re.I
)

#: Named mechanisms that a test or gate script implements. An item saying "live-
#: config gate green" names something enforceable without naming a path, and a
#: path-only classifier would call it prose.
MECHANISMS = {
    "live-config gate": "config_census",
    "equivalence gate": "equivalence_gate",
    "ring-0 replay": "ring0_replay",
    "fingerprint refusal": "fingerprint",
    "interop pin": "interop_threads",
    "kill point": "resume",
    "kill-point": "resume",
    "resume": "resume",
    "inertness": "inert",
    "gradient-arrival": "rehearsal",
    "rehearsal path": "rehearsal",
    "rng non-perturbation": "getstate",
    "non-perturbation": "getstate",
    "pool composition": "pool_composition",
    "occupancy": "occupancy",
    "run_pass": "run_pass",
    "mono-instance": "mono",
    "elo": "elo",
    "self-match": "self_match",
}

#: The standing law line, repeated at the foot of every block. Not an item.
LAW = re.compile(r"Three failed attempts at any gate.*$", re.S)

#: **Granularity matters more than the classifier.** A bullet like "Pod-facts
#: committed; ANCHORS re-digest 0 mismatches; suite + battery + golden green"
#: carries FOUR claims. Classified whole, one enforced conjunct ("golden") makes
#: the bullet read ENFORCED — an always-firing bound that hides the unenforced
#: conjuncts, including the pod-facts clause that started F-48. Split first.
CLAUSE = re.compile(r"\s*;\s*")


def _blocks() -> list[tuple[str, str]]:
    """``[(page, done-when block text)]``."""
    out = []
    seen: dict[str, Path] = {}
    for glob in BRIEF_GLOBS:
        for page in REPO.glob(glob):
            seen[page.name] = page
    for name in sorted(seen):
        text = seen[name].read_text()
        for m in HEADING.finditer(text):
            rest = text[m.end() :]
            nxt = NEXT_HEADING.search(rest)
            out.append((name, rest[: nxt.start()] if nxt else rest))
    return out


def _grep(needle: str, *paths: str) -> list[str]:
    try:
        r = subprocess.run(
            ["git", "grep", "-l", "-F", "-i", needle, "--", *paths],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return r.stdout.split()


def _grep_repo(needle: str) -> str | None:
    """Would anything have noticed? Returns the enforcement ROUTE, or None.

    **`src/` is in scope, and excluding it was a defect.** This project's
    enforcement idiom is a *runtime refusal in the driver*, not only a test:
    RNG non-perturbation — one of the three conditions set for routing the
    cadence — is asserted at `campaign.py:336` as a `CampaignRefusal` and by
    nothing under `tests/`. A census scoped to tests would have reported that
    condition unchecked, which is false.

    **The limitation, named rather than hidden:** a mention in `src/` is not
    enforcement, so a source hit counts only if the file also carries the
    project's enforcement idiom (`CampaignRefusal` or `assert`). That is a
    proxy. It over-credits a file that refuses about something *else*, and the
    route is reported so a human can tell the difference.
    """
    if _grep(needle, "tests/"):
        return "test"
    if _grep(needle, "scripts/", "Makefile"):
        return "script"
    for hit in _grep(needle, "src/"):
        body = (REPO / hit).read_text()
        if "CampaignRefusal" in body or "assert " in body:
            return f"runtime:{Path(hit).name}"
    return None


def classify(item: str) -> dict:
    """Answer both questions for one DONE-WHEN item."""
    text = " ".join(LAW.sub("", item).split())
    artifacts = [a for a in ARTIFACTY.findall(item) if "*" not in a and "<" not in a]

    exists = [a for a in artifacts if (REPO / a).exists() or list(REPO.glob(f"**/{Path(a).name}"))]
    missing = [a for a in artifacts if a not in exists]

    # --- Question 2 FIRST. A command or an implemented mechanism enforces an
    # item whether or not it names a file, and asking Q1 first buries that.
    low = text.lower()
    by_command = bool(ENFORCED_BY_COMMAND.search(text))
    by_mechanism = [
        f"{t}({r})" for phrase, t in MECHANISMS.items() if phrase in low and (r := _grep_repo(t))
    ]
    by_artifact = [f"{a}({r})" for a in artifacts if (r := _grep_repo(a))]
    enforced = by_command or by_mechanism or by_artifact

    if enforced:
        verdict = "ENFORCED"
    elif missing:
        verdict = "NO ARTIFACT"
    elif artifacts:
        verdict = "UNENFORCED"
    else:
        verdict = "UNCHECKABLE"
    return {
        "item": text[:160],
        "artifacts": artifacts,
        "missing": missing,
        "enforced_by": (["command"] if by_command else []) + by_mechanism + by_artifact,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "donewhen_census.json")
    args = parser.parse_args()

    rows = []
    for page, block in _blocks():
        for m in ITEM.finditer(block):
            item = LAW.sub("", m.group(1)).strip()
            for clause in CLAUSE.split(item):
                clause = clause.strip()
                if len(clause) < 8:
                    continue
                row = classify(clause)
                row["page"] = page
                rows.append(row)

    order = ["NO ARTIFACT", "UNCHECKABLE", "UNENFORCED", "ENFORCED"]
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in order}

    print(
        f"\n  DONE-WHEN CENSUS (F-48) — {len(rows)} CLAUSES across "
        f"{len({r['page'] for r in rows})} pages\n"
    )
    print("  Q1 is there an artifact?   Q2 would anything have NOTICED if there weren't?\n")
    for v in order:
        note = {
            "NO ARTIFACT": "an item names a file that does not exist",
            "UNCHECKABLE": "prose only — no artifact named, so Q1 cannot be asked",
            "UNENFORCED": "artifact exists; NOTHING in tests/ or scripts/ names it",
            "ENFORCED": "a command, test or implemented mechanism would have failed",
        }[v]
        print(f"    {v:<12} {counts[v]:>3}   — {note}")

    for v in ("NO ARTIFACT", "UNENFORCED", "UNCHECKABLE"):
        sel = [r for r in rows if r["verdict"] == v]
        if not sel:
            continue
        print(f"\n  {v}:")
        for r in sel[:14]:
            tail = f"   [{', '.join(r['missing'] or r['artifacts'])}]" if r["artifacts"] else ""
            print(f"    - {r['page']}: {r['item'][:96]}{tail}")
        if len(sel) > 14:
            print(f"    ... and {len(sel) - 14} more")

    print(
        "\n  This census CLASSIFIES; it does not adjudicate. UNENFORCED is not a\n"
        "  defect list — it is the size of the population where an item was\n"
        "  discharged by assertion and nothing could have contradicted it."
    )
    args.out.write_text(json.dumps({"counts": counts, "items": rows}, indent=2) + "\n")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
