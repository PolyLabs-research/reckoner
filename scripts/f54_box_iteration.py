"""F-54's box-side reading: one full iteration through ``train_on_ring`` with rehearsal on.

The cloud clone that built F-54 held no ``phase1_train``, so the supervised share
of the batch (``train.rehearsal_frac``, 0.65 by default) never met the code it
changed. This runs the campaign's iteration body at the DEFAULT config, with the
one substitution the handoff named, the uniform stub in self-play (a network in
search costs minutes per episode on CPU and is tested elsewhere):

1. ``episodes_per_iteration`` frozen suite problems, the same number from each
   suite via ``sample_indices``;
2. one ``run_iteration`` into a fresh ``ReplayRing`` under ``uniform_stub``;
3. ``train_on_ring`` on the Phase-1 anchor for ``train_steps_per_iter`` steps.

It reports the counts that say the path was exercised: rehearsal batches, steps,
skips, truncated rows, first and last losses. An ``IllegalTargetError`` (F-53's
refusal, which supervised rows now meet too) is caught, reported and exits
non-zero; nothing else is caught.

    python scripts/f54_box_iteration.py --out /tmp/f54_box_iteration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint
from reckoner.dataset import read_suite, sample_indices, suite_problem
from reckoner.model import IllegalTargetError, load_checkpoint
from reckoner.replay import ReplayRing
from reckoner.runner import run_iteration
from reckoner.search import uniform_stub
from reckoner.train import train_on_ring

REPO = Path(__file__).resolve().parents[1]


def _weights(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy-target", default=None, help="default: the config's")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cfg = Config()
    if args.policy_target is not None:
        cfg = replace(cfg, train=replace(cfg.train, policy_target=args.policy_target))

    paths = sorted((REPO / "runs" / "suites").glob("*.jsonl"))
    per_suite = cfg.campaign.episodes_per_iteration // len(paths)
    problems = []
    for path in paths:
        rows = read_suite(path)
        problems += [suite_problem(rows[i]) for i in sample_indices(len(rows), per_suite, 0)]

    ring = ReplayRing(cfg.train.replay_capacity, cfg)
    t0 = time.monotonic()
    stats = run_iteration(problems, uniform_stub(cfg), cfg, ring, seed=0)
    seconds_self_play = time.monotonic() - t0

    model, _ = load_checkpoint(REPO / "runs" / "phase1" / "phase1.pt", cfg)
    before = _weights(model)
    refusal = None
    train = None
    t0 = time.monotonic()
    try:
        train = train_on_ring(model, ring, cfg, steps=cfg.train.train_steps_per_iter, seed=0)
    except IllegalTargetError as exc:
        refusal = str(exc)
    seconds_train = time.monotonic() - t0

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "head": head,
        "torch": torch.__version__,
        "threads": args.threads,
        "config_fingerprint": config_fingerprint(cfg),
        "policy_target": cfg.train.policy_target,
        "rehearsal_frac": cfg.train.rehearsal_frac,
        "batch_size": cfg.train.batch_size,
        "problems": len(problems),
        "suites": len(paths),
        "episodes": stats.episodes,
        "episodes_solved": stats.episodes_solved,
        "ring_records": len(ring),
        "seconds_self_play": round(seconds_self_play, 1),
        "seconds_train": round(seconds_train, 1),
        "illegal_target_refusal": refusal,
        "train": None
        if train is None
        else {
            **{k: v for k, v in train.as_dict().items() if k != "losses"},
            "steps_requested": cfg.train.train_steps_per_iter,
            "first_loss": train.losses[0] if train.losses else None,
            "last_loss": train.losses[-1] if train.losses else None,
            "losses_finite": all(
                loss == loss and abs(loss) != float("inf") for loss in train.losses
            ),
        },
        "weights_before": before,
        "weights_after": _weights(model),
    }
    ok = (
        train is not None
        and train.steps == cfg.train.train_steps_per_iter
        and train.rehearsal_batches == train.steps
        and train.nan_skips == 0
        and report["weights_after"] != before
    )
    report["verdict"] = "RAN_CLEAN" if ok else "DID_NOT_RUN_CLEAN"
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
