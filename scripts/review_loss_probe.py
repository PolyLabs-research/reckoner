"""REVIEW-2026-11-15's first probe: what each loss term sends into the shared trunk.

At the Phase-1 anchor, on batches drawn from a ring, this reports for each term of
``train_on_ring``'s loss (policy toward π′, steps MSE, value cross-entropy, and
the weighted q term) its value and the norm of its gradient with respect to the
parameters every head shares: the embeddings, the transformer trunk and the
final norm. It also reports how far the steps head is from its ring target on
rows whose episode ended at par (z = 0) and on rows whose episode did not
(z = -1), and how many rows carry a π′ that differs from the anchor's own policy.

A probe for a review, not a registered reading: eight batches of 128 at one seed,
eval mode (no dropout), no claim beyond the numbers it prints.

    python scripts/review_loss_probe.py --out runs/review_20261115/loss_probe_result.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch
from sweep_m2 import ring_digest
from torch import nn

from reckoner.campaign import ANCHOR
from reckoner.config import Config
from reckoner.model import load_checkpoint, policy_loss, steps_loss
from reckoner.replay import ReplayRing
from reckoner.train import ring_batch

REPO = Path(__file__).resolve().parents[1]
RINGS = {
    "ring-0": REPO / "runs" / "ring0_replay" / "ring-0",
    "ring-V": REPO / "runs" / "ring_v" / "ring-0",
}
HEAD_PREFIXES = ("value_head", "steps_head", "rule_embedding", "site_projection")
BATCHES = 8


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def probe_ring(model, ring: ReplayRing, cfg: Config) -> dict:
    shared = [p for name, p in model.named_parameters() if not name.startswith(HEAD_PREFIXES)]

    def trunk_gradient(loss: torch.Tensor) -> float:
        grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g**2).sum() for g in grads if g is not None)))

    rng = random.Random(0)
    losses: dict[str, list[float]] = {}
    gradients: dict[str, list[float]] = {}
    steps_error: dict[str, list[float]] = {"z=0": [], "z=-1": []}
    steps_target: dict[str, list[float]] = {"z=0": [], "z=-1": []}
    for _ in range(BATCHES):
        indices = [rng.randrange(len(ring)) for _ in range(cfg.train.batch_size)]
        batch = ring_batch(ring, indices, cfg, target="improved")
        policy, value_logits, steps = model(batch.tokens, batch.site_positions)
        probs = torch.softmax(value_logits, dim=1)
        terms = {
            "policy": policy_loss(policy, batch.policy_target, batch.legal_mask),
            "steps_mse": steps_loss(steps, batch.steps_target, batch.solved_mask),
            "value_ce": nn.functional.cross_entropy(value_logits, batch.z_class),
            "q_mse_weighted": cfg.train.value_q_mse_weight
            * nn.functional.mse_loss(probs[:, 0] - probs[:, 2], batch.root_q),
        }
        for name, term in terms.items():
            losses.setdefault(name, []).append(float(term.detach()))
            gradients.setdefault(name, []).append(trunk_gradient(term))
        error = (steps - batch.steps_target).abs().detach()
        for label, cls in (("z=0", 1), ("z=-1", 2)):
            rows = batch.z_class == cls
            if bool(rows.any()):
                steps_error[label].append(float(error[rows].mean()))
                steps_target[label].append(float(batch.steps_target[rows].mean()))

    # How many rows teach the policy anything: π′ against the anchor's own policy.
    informative = disagree = 0
    for start in range(0, len(ring), 128):
        batch = ring_batch(
            ring, list(range(start, min(start + 128, len(ring)))), cfg, target="improved"
        )
        with torch.no_grad():
            policy, _, _ = model(batch.tokens, batch.site_positions)
        masked = policy.masked_fill(~batch.legal_mask, float("-inf"))
        log_prior = torch.log_softmax(masked, dim=-1).nan_to_num(neginf=0.0)
        target = batch.policy_target
        kl = torch.where(
            target > 0,
            target * (torch.log(target.clamp_min(1e-30)) - log_prior),
            torch.zeros_like(target),
        ).sum(-1)
        informative += int((kl > 0.01).sum())
        disagree += int((target.argmax(-1) != masked.argmax(-1)).sum())

    return {
        "rows": len(ring),
        "loss": {name: round(_mean(v), 4) for name, v in losses.items()},
        "trunk_gradient_norm": {name: round(_mean(v), 4) for name, v in gradients.items()},
        "steps_head_abs_error": {k: round(_mean(v), 3) for k, v in steps_error.items() if v},
        "steps_target_mean": {k: round(_mean(v), 3) for k, v in steps_target.items() if v},
        "rows_pi_prime_kl_above_0.01": informative,
        "rows_pi_prime_best_action_differs": disagree,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    model, _ = load_checkpoint(ANCHOR, cfg)
    model.eval()
    report = {
        "git_sha": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "batches": BATCHES,
        "batch_size": cfg.train.batch_size,
        "rings": {},
    }
    for name, path in RINGS.items():
        ring = ReplayRing.load(path, cfg)
        report["rings"][name] = {"ring_digest": ring_digest(path), **probe_ring(model, ring, cfg)}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
