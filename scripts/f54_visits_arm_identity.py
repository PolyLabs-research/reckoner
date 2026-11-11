"""F-54's control-arm proof: ``train.policy_target = 'visits'`` IS pre-F-54 training.

The principal kept the visit share as the control arm for a later A/B. That arm
is only a control if it is the old behaviour, bit for bit, and that is a
derivation, not a claim. This script:

1. exports a base commit (default ``89b4095``, the last commit before F-54's
   code) with ``git archive`` into a scratch directory;
2. runs ONE fixture, in a fresh subprocess per tree, importing each tree's own
   ``src``: frozen suite problems (``sample_indices``), one ``run_iteration`` of
   self-play into a ``ReplayRing`` under the uniform stub, then ``train_on_ring``
   for 20 steps on a seeded random-init model at ``rehearsal_frac = 0.0`` and one
   torch thread;
3. compares digests of every ring array both trees store (``record_format``
   excluded by name: the format number is meant to differ), the trained
   ``state_dict``, the per-step losses, and the base's ``entropy_target_*``
   against the new ``entropy_visits_*`` (the same quantity under its new name).

It exits non-zero unless the new tree under ``'visits'`` matches the base on
every one, AND the new tree under ``'improved'`` produces different weights — the
second polarity, so a comparison that cannot see a difference does not pass.

Self-play runs under the uniform stub because a random-init network's episodes
are too slow on a 4-CPU box; the search's own bit-identity under a network is
tested in-process (``tests/test_improved_target.py``).

    python scripts/f54_visits_arm_identity.py --base 89b4095 --out /tmp/f54_identity.json

The default base was ``6a9b18f`` until the branch was rebased onto the local
record (2026-11-11). ``89b4095`` is the same design commit after the rebase;
``6a9b18f`` stays reachable under the tag ``cloud-f41-f42-original`` and can be
passed with ``--base``.

``--rehearsal-frac`` (added 2026-11-11, on the box that holds the data) repeats
the comparison with the supervised share wired in. The default stays 0.0, the
reading F-54 recorded. Above 0.0 ``train_on_ring`` opens ``phase1_train`` through
the anchor gate, which resolves against each tree's OWN root, so the exported
base tree is given a byte copy of ``runs/ANCHORS.sha256`` and
``runs/data/phase1_train`` (a symlink would resolve outside the tree and the
gate would refuse it). Each tree then reports the digest of the supervision
arrays it actually mapped, and the verdict requires them equal, plus
``rehearsal_batches == steps`` on both sides so an arm that silently trained no
supervised rows cannot pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reckoner.dataset import DATA_ROOT, anchored_data, anchors_path

REPO = Path(__file__).resolve().parents[1]

#: Runs inside each tree's interpreter. Version-independent: it sets the policy
#: target only where the field exists, and reads only what both trees have.
FIXTURE = r"""
import hashlib, json, random, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
import reckoner
from reckoner.config import Config
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.model import Reckoner
from reckoner.replay import ReplayRing
from reckoner.runner import run_iteration
from reckoner.search import uniform_stub
from reckoner.train import train_on_ring

suites, target, frac = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
cfg = Config()
cfg = replace(cfg, train=replace(cfg.train, rehearsal_frac=frac))
if hasattr(cfg.train, "policy_target"):
    cfg = replace(cfg, train=replace(cfg.train, policy_target=target))
problems = []
for path in sorted(suites.glob("*.jsonl")):
    rows = read_suite(path)
    problems += [suite_problem(rows[i]) for i in sample_indices(len(rows), 3, 0)]
ring = ReplayRing(20000, cfg)
stats = run_iteration(problems, uniform_stub(cfg), cfg, ring, sims=16, seed=0)
arrays = {}
for name in ReplayRing.ARRAYS:
    if name == "record_format":
        continue
    arrays[name] = hashlib.sha256(getattr(ring, name)[: len(ring)].tobytes()).hexdigest()
torch.manual_seed(7)
model = Reckoner(cfg)
train = train_on_ring(model, ring, cfg, steps=20, seed=3)
digest = hashlib.sha256()
for key, tensor in sorted(model.state_dict().items()):
    digest.update(key.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
ent = stats.entropies()
visits_entropy = {
    k.replace("entropy_visits", "VISIT_SHARE"): v for k, v in ent.items() if k.startswith("entropy_visits")
}
if not visits_entropy:  # the base tree: its target column WAS the visit share
    visits_entropy = {
        k.replace("entropy_target", "VISIT_SHARE"): v for k, v in ent.items() if k.startswith("entropy_target")
    }
supervision = None
if frac > 0.0:
    from reckoner.dataset import DATA_ROOT
    sup_dir = Path(reckoner.__file__).resolve().parents[2] / DATA_ROOT / "phase1_train"
    sup_digest = hashlib.sha256()
    for name in sorted(p.name for p in sup_dir.iterdir()):
        sup_digest.update(name.encode()); sup_digest.update((sup_dir / name).read_bytes())
    supervision = sup_digest.hexdigest()
print(json.dumps({
    "source": str(Path(reckoner.__file__).resolve().parent),
    "rehearsal_frac": cfg.train.rehearsal_frac,
    "rehearsal_batches": getattr(train, "rehearsal_batches", None),
    "supervision": supervision,
    "records": len(ring),
    "ring_arrays": arrays,
    "weights": digest.hexdigest(),
    "losses": train.losses,
    "steps": train.steps,
    "visit_share_entropy": visits_entropy,
}))
"""


def _run(tree_src: Path, suites: Path, target: str, frac: float) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", FIXTURE, str(suites), target, repr(frac)],
        env={**os.environ, "PYTHONPATH": str(tree_src), "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="89b4095")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rehearsal-frac",
        type=float,
        default=0.0,
        help="train.rehearsal_frac for all three runs; above 0.0 needs runs/data/phase1_train",
    )
    args = parser.parse_args()
    frac = args.rehearsal_frac

    suites = REPO / "runs" / "suites"
    with tempfile.TemporaryDirectory() as scratch:
        base_tree = Path(scratch) / "base"
        base_tree.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", args.base, "src"], capture_output=True, check=True
        )
        subprocess.run(["tar", "-x", "-C", str(base_tree)], input=archive.stdout, check=True)
        if frac > 0.0:
            anchors = anchors_path(REPO).relative_to(REPO)
            (base_tree / DATA_ROOT).mkdir(parents=True)
            shutil.copy2(REPO / anchors, base_tree / anchors)
            shutil.copytree(
                anchored_data("phase1_train", verify=True), base_tree / DATA_ROOT / "phase1_train"
            )
        base = _run(base_tree / "src", suites, "visits", frac)
    visits = _run(REPO / "src", suites, "visits", frac)
    improved = _run(REPO / "src", suites, "improved", frac)

    common = sorted(set(base["ring_arrays"]) & set(visits["ring_arrays"]))
    checks = {
        "same_records": base["records"] == visits["records"],
        "ring_arrays_identical": all(
            base["ring_arrays"][k] == visits["ring_arrays"][k] for k in common
        ),
        "weights_identical": base["weights"] == visits["weights"],
        "losses_identical": base["losses"] == visits["losses"],
        "visit_share_entropy_identical": base["visit_share_entropy"]
        == visits["visit_share_entropy"],
        "improved_changes_the_weights": improved["weights"] != visits["weights"],
    }
    if frac > 0.0:
        checks["same_supervision_bytes"] = (
            base["supervision"] is not None and base["supervision"] == visits["supervision"]
        )
        checks["rehearsal_ran_every_step"] = (
            base["rehearsal_batches"] == base["steps"] == visits["rehearsal_batches"] > 0
        )
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "base": args.base,
        "head": head,
        "common_ring_arrays": common,
        "records": visits["records"],
        "steps": visits["steps"],
        "rehearsal_frac": frac,
        "rehearsal_batches": {
            "base": base["rehearsal_batches"],
            "visits": visits["rehearsal_batches"],
            "improved": improved["rehearsal_batches"],
        },
        "supervision": visits["supervision"],
        "weights": {
            "base": base["weights"],
            "visits": visits["weights"],
            "improved": improved["weights"],
        },
        "checks": checks,
        "verdict": "IDENTICAL" if all(checks.values()) else "DIFFERS",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
