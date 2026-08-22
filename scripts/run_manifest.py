"""What a completed campaign run MUST contain, derived upstream of the run.

Per-file digests prove that what you pulled is **intact**. They cannot prove you
pulled **everything** — missing is invisible to an integrity check exactly as
cited-and-gone is invisible to a volume walk (F-40). So the expected list is
derived from the **must-reach contract** and the **expectations pages**, and the
tree is checked against that.

Same structural point as F-38's launcher counting its own launches: **the check
has to come from upstream of the thing it checks.** A manifest built by listing
the directory can only ever confirm the directory agrees with itself.

Anything deliberately left behind is DECLARED here, so an exclusion is a decision
on the page rather than a gap discovered later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Per-run record classes the must-track contract names (tests/test_gitignore_musttrack).
PER_RUN = ("iterations.jsonl", "value_switch.jsonl", "LATEST")
#: Named by the driver and by REHEARSAL-m1-attempt2.md.
PER_RUN_ALSO = ("instruments.jsonl", "campaign_summary.json", "checkpoint_watch.json")
#: Cadence iterations for a 20-iteration campaign at ladder_every = 5.
CADENCE = (4, 9, 14, 19)
#: Deliberately not pulled, with the reason. An exclusion is a decision, not a gap.
DECLARED_ABSENT = {
    "ckpt-1..18.pt": "intermediate checkpoints; ckpt-0 (the tripwire comparand) and "
    "ckpt-19 (the endpoint M2 warm-starts from) are retained",
    "snap-0..18.pt": "intermediate pool snapshots; snap-19 retained",
    "ring-17, ring-18": "superseded by ring-19, which carries the full accumulated ring",
    "state-17, state-18": "superseded by state-19",
}


def expected(iterations: int) -> list[str]:
    out = [
        *PER_RUN,
        *PER_RUN_ALSO,
        "state-19.json",
        "ring-19",
        "ckpt-0.pt",
        "ckpt-19.pt",
        "snap-19.pt",
    ]
    for index in CADENCE:
        for leg in ("no_regress", "primary"):
            out.append(f"ladder/{leg}/pass-{index:04d}/pair_scores.jsonl")
            out.append(f"ladder/{leg}/pass-{index:04d}/PASS-DONE")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    want = expected(args.iterations)
    missing = [rel for rel in want if not (args.run_dir / rel).exists()]

    print(f"\n  RUN MANIFEST — {args.run_dir}\n")
    print(f"    expected by contract + pages : {len(want)}")
    print(f"    present                      : {len(want) - len(missing)}")
    print(f"    MISSING                      : {len(missing)}")
    for rel in missing:
        print(f"      - {rel}")
    print("\n    declared absent, with reason:")
    for what, why in DECLARED_ABSENT.items():
        print(f"      {what}: {why}")

    (args.run_dir / "manifest.json").write_text(
        json.dumps(
            {"expected": want, "missing": missing, "declared_absent": DECLARED_ABSENT}, indent=2
        )
        + "\n"
    )
    if missing:
        print("\n  INCOMPLETE — the pull did not bring everything the contract names.")
        return 1
    print("\n  COMPLETE — every artifact the contract and the pages name is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
