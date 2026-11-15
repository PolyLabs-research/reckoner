"""Gate 14: anchor the Phase-1 checkpoint by digest, with full provenance.

An anchor is what every later pass is compared *against*, so it has to be
identifiable by something the comparison cannot silently change. The digest is
that something. Everything else here exists so a reader a month from now can
answer "what was this, exactly?" without trusting a filename:

* the checkpoint's sha256, appended to ``runs/ANCHORS.sha256``
* the config fingerprint, ruleset and vocab versions the checkpoint carries
* the **dataset digests it trained on**, verified against the datasets as they
  are on disk right now — an anchor whose training data has been rebuilt under
  it is not the anchor anyone thinks it is
* the gate records by digest, so the anchor points at the evidence rather than
  restating it

Refuses to write if the training data no longer matches what the run recorded.
That is the F-02 lesson at the anchor layer: a provenance field that cannot be
checked is a claim, and the trusted value is computed or absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reckoner.dataset import data_path, git_sha, sha256_file, write_record

REPO = Path(__file__).resolve().parents[1]

GATE_RECORDS = (
    "gate_arithmetic_topk.json",
    "gate_phase1_search_m1.json",
    "gate_phase1_search_m5.json",
    "supervision_contamination.json",
    "eval_independence.json",
    "pilot_phase1_timing.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "phase1")
    args = parser.parse_args()

    checkpoint = args.run / "phase1.pt"
    provenance = json.loads((args.run / "provenance.json").read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = state["meta"]

    # The anchor's training data must still be the data the run recorded.
    train_meta = json.loads((data_path("phase1_train") / "meta.json").read_text())
    if provenance["data_digests"] != train_meta["digests"]:
        raise SystemExit(
            "phase1_train has changed since the run: the anchor would claim a "
            "provenance it cannot support. Re-run, or anchor the run that matches."
        )

    digest = sha256_file(checkpoint)
    relative = str(checkpoint.resolve().relative_to(REPO))
    is_phase1 = relative == "runs/phase1/phase1.pt"
    anchor = {
        "name": args.run.name,
        "checkpoint_sha256": digest,
        "step": meta["step"],
        "config_fingerprint": meta["config_fingerprint"],
        "ruleset_version": meta["ruleset_version"],
        "vocab_version": meta["vocab_version"],
        "git_sha_at_anchor": git_sha(REPO),
        "git_sha_at_run": provenance["git_sha"],
        "trained_on": {
            "dataset": "phase1_train",
            "digests": provenance["data_digests"],
            "verified_against_disk": True,
            # None is the Phase-1 anchor's recipe: every row. A number is a weak
            # start, supervised on problems of at most that depth (WEAK-w1.md).
            "max_depth": provenance.get("max_depth"),
        },
        # The gate records are the Phase-1 anchor's own evidence. Another run has
        # none of them, and claiming them for it would be provenance by silence.
        "evidence": {
            name: sha256_file(REPO / "runs" / name)
            for name in GATE_RECORDS
            if is_phase1 and (REPO / "runs" / name).exists()
        },
        "evidence_missing": [
            name for name in GATE_RECORDS if is_phase1 and not (REPO / "runs" / name).exists()
        ],
    }
    if not is_phase1:
        anchor["evidence_absent"] = "the gate records belong to runs/phase1; this run has none"
    write_record(args.run / "anchor.json", anchor)

    anchors = REPO / "runs" / "ANCHORS.sha256"
    line = f"{digest}  {relative}\n"
    existing = anchors.read_text() if anchors.exists() else ""
    if line not in existing:
        kept = [
            row
            for row in existing.splitlines(keepends=True)
            if row.split(maxsplit=1)[-1].strip() != relative
        ]
        anchors.write_text("".join(sorted(kept + [line])))

    print("  PHASE-1 ANCHOR\n")
    print(f"  checkpoint : {digest}")
    print(f"  step       : {anchor['step']}")
    print(f"  config     : {anchor['config_fingerprint'][:16]}...")
    print(f"  ruleset/vocab: {anchor['ruleset_version']}/{anchor['vocab_version']}")
    print("  trained on : phase1_train, digests verified against disk")
    print(f"  evidence   : {len(anchor['evidence'])} records by digest")
    if anchor["evidence_missing"]:
        print(f"  MISSING    : {anchor['evidence_missing']}  (absence carries a reason)")
    print(f"\n  wrote {args.run / 'anchor.json'} and updated runs/ANCHORS.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
