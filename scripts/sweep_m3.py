"""The rehearsal sweep under 'improved', on a ring the network disagrees with.

Protocol: ``SWEEP-m3.md``, registered before any arm ran. SWEEP-m2 found that on
ring-0 the π′ target is the anchor's own policy on all but 57 of 1,305 rows, so
every 'improved' arm read the anchor's 1192 and the band could not tell one
``f`` from another (F-56). This sweep repeats the 'improved' arms on ring-V:
iteration 0 replayed with the anchor's value head trusted by the search
(``scripts/ring0_replay.py --value-scale 1.0``), where π′ departs from the
anchor's policy on a third of the states.

The arm machinery is SWEEP-m2's (``sweep_m2.run_arm``, the same scorer, the
same provenance), so the two sweeps differ in their ring and their arm table
only:

    python scripts/sweep_m3.py --list
    python scripts/sweep_m3.py --arm anchor
    python scripts/sweep_m3.py --arm 0          # writes runs/sweep_m3/improved/arm-00.json
    python scripts/sweep_m3.py --collect        # the table and the selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sweep_m2 import BAND_HITS, INSTRUMENT_N, REPO, run_anchor, run_arm

from reckoner.config import Config, validate

OUT = REPO / "runs" / "sweep_m3"
RING = REPO / "runs" / "ring_v" / "ring-0"
TARGET = "improved"
FRACS = (0.00, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75)
SEEDS = (0, 1, 2)

#: SWEEP-m3.md §4. Every candidate f at three seeds, so §6's seed requirement is
#: met in the same run instead of a follow-up round; then SWEEP-m2's mechanism
#: pair. Selection considers ONLY kind == "candidate".
ARMS = (
    tuple((frac, 400, 200, seed, "candidate", "") for frac in FRACS for seed in SEEDS)
    + (
        (0.00, 200, 200, 0, "mechanism", "the first 200 steps of the control, all warmup"),
        (0.00, 200, 100, 0, "mechanism", "the control's whole schedule in 200 steps"),
    )
    + tuple(
        (
            0.00,
            400,
            200,
            seed,
            "control",
            "policy target withheld: value and steps heads only",
            {"policy_loss_weight": 0.0},
        )
        for seed in (0, 1)
    )
)


def collect(out: Path) -> int:
    anchor_path = out / "anchor.json"
    if not anchor_path.exists():
        print("  no anchor.json: run --arm anchor first")
        return 1
    anchor = json.loads(anchor_path.read_text())
    rows = []
    for index in range(len(ARMS)):
        path = out / TARGET / f"arm-{index:02d}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    missing = len(ARMS) - len(rows)

    # §2: the smallest f at which EVERY seed holds the band.
    by_frac: dict[float, list[dict]] = {}
    for r in rows:
        if r["kind"] == "candidate":
            by_frac.setdefault(r["rehearsal_frac"], []).append(r)
    verdicts = {
        frac: len(group) == len(SEEDS) and all(r["holds_band"] for r in group)
        for frac, group in sorted(by_frac.items())
    }
    holding = [frac for frac, ok in verdicts.items() if ok]
    selected = holding[0] if holding and not missing else None

    print(f"\n  SWEEP-m3 — '{TARGET}' on ring-V   ({len(rows)}/{len(ARMS)} arms)")
    print(
        f"  band: hits >= {BAND_HITS}/{INSTRUMENT_N}   anchor here: {anchor['hits']}/{anchor['n']}\n"
    )
    print(f"    {'f':>5} {'seed 0':>8} {'seed 1':>8} {'seed 2':>8}  all hold")
    for frac in FRACS:
        group = {r["seed"]: r["hits"] for r in by_frac.get(frac, [])}
        cells = " ".join(f"{group.get(s, '-'):>8}" for s in SEEDS)
        print(f"    {frac:>5.2f} {cells}  {'YES' if verdicts.get(frac) else 'no'}")
    for r in rows:
        if r["kind"] == "mechanism":
            print(
                f"    mechanism: {r['steps']} steps, warmup {r['lr_warmup_steps']}: "
                f"{r['hits']}/{r['n']}"
            )
        elif r["kind"] == "control":
            print(f"    control, policy target withheld, seed {r['seed']}: {r['hits']}/{r['n']}")
    if missing:
        print(f"\n    INCOMPLETE: {missing} arm(s) missing; no selection is made")
    elif selected is None:
        print("\n    NO f holds the band on all three seeds (SWEEP-m3.md §5, second branch)")
    else:
        print(f"\n    smallest f holding the band on every seed: {selected:.2f}")

    report = {
        "protocol": "SWEEP-m3.md",
        "band_hits": BAND_HITS,
        "instrument_n": INSTRUMENT_N,
        "anchor": anchor,
        "target": TARGET,
        "arms": rows,
        "missing_arms": missing,
        "all_seeds_hold": {f"{frac:.2f}": ok for frac, ok in verdicts.items()},
        "selected": selected,
        "complete": missing == 0 and anchor["n"] == INSTRUMENT_N,
    }
    (out.parent / "sweep_m3.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n  written: {out.parent / 'sweep_m3.json'}   complete: {report['complete']}")
    return 0 if report["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", help="an index from --list, or 'anchor'")
    parser.add_argument("--ring", type=Path, default=RING)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    if args.list:
        for index, arm in enumerate(ARMS):
            print(index, *arm)
        return 0
    if args.collect:
        return collect(args.out_dir)
    if args.arm is None:
        parser.error("--arm is required to run an arm")

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    if args.arm == "anchor":
        path = args.out_dir / "anchor.json"
        row = run_anchor(cfg, args.ring)
    else:
        index = int(args.arm)
        path = args.out_dir / TARGET / f"arm-{index:02d}.json"
        row = run_arm(TARGET, index, cfg, args.ring, arms=ARMS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
