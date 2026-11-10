"""F-54's control-arm proof: ``train.policy_target = 'visits'`` IS pre-F-54 training.

The principal kept the visit share as the control arm for a later A/B. That arm
is only a control if it is the old behaviour, bit for bit, and that is a
derivation, not a claim. This script:

1. exports a base commit (default ``6a9b18f``, the last commit before F-54's
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

    python scripts/f54_visits_arm_identity.py --base 6a9b18f --out /tmp/f54_identity.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

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

suites, target = Path(sys.argv[1]), sys.argv[2]
cfg = Config()
cfg = replace(cfg, train=replace(cfg.train, rehearsal_frac=0.0))
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
print(json.dumps({
    "source": str(Path(reckoner.__file__).resolve().parent),
    "records": len(ring),
    "ring_arrays": arrays,
    "weights": digest.hexdigest(),
    "losses": train.losses,
    "steps": train.steps,
    "visit_share_entropy": visits_entropy,
}))
"""


def _run(tree_src: Path, suites: Path, target: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", FIXTURE, str(suites), target],
        env={**os.environ, "PYTHONPATH": str(tree_src), "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="6a9b18f")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    suites = REPO / "runs" / "suites"
    with tempfile.TemporaryDirectory() as scratch:
        base_tree = Path(scratch) / "base"
        base_tree.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", args.base, "src"], capture_output=True, check=True
        )
        subprocess.run(["tar", "-x", "-C", str(base_tree)], input=archive.stdout, check=True)
        base = _run(base_tree / "src", suites, "visits")
    visits = _run(REPO / "src", suites, "visits")
    improved = _run(REPO / "src", suites, "improved")

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
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "base": args.base,
        "head": head,
        "common_ring_arrays": common,
        "records": visits["records"],
        "steps": visits["steps"],
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
