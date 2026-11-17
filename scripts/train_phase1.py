"""Chunk 8: run the Phase-1 supervised warm start.

Thin argparse wrapper (AGENTS.md §6): builds a config, calls library code, owns
the terminal and the run directory. Every number it prints, it writes.

The run directory carries what makes the run interpretable on its own — the
resolved `config.yaml`, the git SHA, the `check_env.py` output, `metrics.jsonl`,
and a checkpoint whose meta can reproduce its own configuration.

**Extension bound (brief A4):** 5,000 steps, one extension to 10,000 maximum,
then `BLOCKED-<date>-<topic>.md`. `--steps` above 10,000 is refused here rather
than left to discipline, because "extend until it passes" is fishing and the
bound is what makes it not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint, save_config, validate
from reckoner.dataset import anchored_data, git_sha, write_record
from reckoner.model import Reckoner, save_checkpoint
from reckoner.train import SupervisionSet, make_batch, train

REPO = Path(__file__).resolve().parents[1]
MAX_STEPS = 10_000  # A4. Not a tunable.


def top_k_by_depth(
    model: Reckoner,
    data: SupervisionSet,
    cfg: Config,
    *,
    ks: tuple[int, ...] = (1, 8),
    batch_size: int = 256,
    exclude: set[int] | None = None,
    device: str = "cpu",
) -> dict:
    """Top-k (rule, site) accuracy, stratified by problem depth.

    ``exclude`` drops row indices from the measurement — used to report the gate
    on the subset of the held-out set that is genuinely unseen (`FINDINGS.md`
    F-09). Reported *beside* the full-set number, never instead of it.
    """
    model.eval()
    indices = [i for i in range(len(data)) if exclude is None or i not in exclude]
    hits = {k: {} for k in ks}
    totals: dict[int, int] = {}

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = make_batch(data, chunk, cfg)
            policy, _value, _steps = model(batch.tokens.to(device), batch.site_positions.to(device))
            # Illegal actions cannot be credited: a top-k that ranks an action the
            # movegen would refuse is measuring the wrong space.
            masked = policy.masked_fill(~batch.legal_mask.to(device), float("-inf"))
            target = batch.policy_target.argmax(dim=1).to(device)
            for k in ks:
                width = min(k, masked.shape[1])
                top = masked.topk(width, dim=1).indices
                correct = (top == target.unsqueeze(1)).any(dim=1)
                for depth, ok in zip(batch.depth.tolist(), correct.tolist(), strict=True):
                    hits[k][depth] = hits[k].get(depth, 0) + int(ok)
            for depth in batch.depth.tolist():
                totals[depth] = totals.get(depth, 0) + 1

    out = {"n": sum(totals.values()), "by_depth": {}, "overall": {}}
    for depth in sorted(totals):
        out["by_depth"][depth] = {
            "n": totals[depth],
            **{f"top{k}": round(hits[k].get(depth, 0) / totals[depth], 4) for k in ks},
        }
    for k in ks:
        out["overall"][f"top{k}"] = round(sum(hits[k].values()) / max(1, sum(totals.values())), 4)
    shallow = [d for d in totals if d <= 3]
    for k in ks:
        num = sum(hits[k].get(d, 0) for d in shallow)
        den = sum(totals[d] for d in shallow)
        out[f"depth_le_3_top{k}"] = round(num / den, 4) if den else None
    out["depth_le_3_n"] = sum(totals[d] for d in shallow)
    return out


def shared_state_indices(eval_path: Path) -> set[int]:
    """Rows of the held-out set that also occur in training (F-09), recomputed.

    Read from the census record's own definition rather than re-derived here, so
    one keying serves the gate, the census and this exclusion.
    """
    from reckoner.episode import decode_state
    from reckoner.expr import identity_key

    train_set = SupervisionSet(anchored_data("phase1_train"))
    seen = set()
    for i in range(len(train_set)):
        goal, _t, expr = decode_state(
            tuple(int(x) for x in train_set.tokens[i, : train_set.lengths[i]])
        )
        seen.add((identity_key(expr), goal))

    held = SupervisionSet(eval_path)
    out = set()
    for i in range(len(held)):
        goal, _t, expr = decode_state(tuple(int(x) for x in held.tokens[i, : held.lengths[i]]))
        if (identity_key(expr), goal) in seen:
            out.add(i)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="phase1")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=anchored_data("phase1_train"))
    parser.add_argument("--eval", type=Path, default=anchored_data("phase1_eval"))
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="train only on rows from problems of at most this depth: a weak starting "
        "network (WEAK-w1.md). Default: every row, which is the Phase-1 anchor's recipe.",
    )
    args = parser.parse_args()

    if args.steps > MAX_STEPS:
        print(
            f"  refused: --steps {args.steps} exceeds the A4 bound of {MAX_STEPS}. "
            f"Write BLOCKED-<date>-<topic>.md instead of extending again.",
            file=sys.stderr,
        )
        return 2

    cfg = Config()
    validate(cfg)  # strict, at load, not at first use

    run = REPO / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run / "config.yaml")
    env = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_env.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    (run / "check_env.txt").write_text(env.stdout + env.stderr)
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(REPO),
                "config_fingerprint": config_fingerprint(cfg),
                "steps_requested": args.steps,
                "seed": args.seed,
                "device": args.device,
                "data": str(args.data.name),
                "max_depth": args.max_depth,
                "data_digests": json.loads((args.data / "meta.json").read_text())["digests"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    data = SupervisionSet(args.data)
    if args.max_depth is not None:
        data = data.up_to_depth(args.max_depth)
    model = Reckoner(cfg)
    if args.init_weights is not None:
        state = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        print(f"  initialised from {args.init_weights}")

    print("  PHASE 1 — supervised warm start\n")
    print(f"  examples   : {len(data):,}")
    print(f"  steps      : {args.steps:,}  (A4 bound {MAX_STEPS:,})")
    print(f"  batch      : {cfg.train.batch_size}")
    print(
        f"  lr         : {cfg.train.lr}  {cfg.train.lr_schedule}, "
        f"warmup {cfg.train.lr_warmup_steps}"
    )
    print(f"  device     : {args.device}")
    print(f"  parameters : {sum(p.numel() for p in model.parameters()):,}\n")

    metrics = (run / "metrics.jsonl").open("a")
    started = time.perf_counter()

    def on_log(step: int, loss: float, lr: float) -> None:
        elapsed = time.perf_counter() - started
        row = {
            "step": step,
            "loss": round(loss, 5),
            "lr": round(lr, 8),
            "elapsed_s": round(elapsed, 1),
        }
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        print(f"    step {step:>6}  loss {loss:.4f}  lr {lr:.2e}  {elapsed / 60:.1f}m")

    # Mode audit: training must train, evaluation must not.
    assert model.training, "model is not in train mode at the start of training"
    stats = train(
        model,
        data,
        cfg,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        log_every=args.log_every,
        on_log=on_log,
    )
    wall = time.perf_counter() - started
    metrics.close()

    print(f"\n  trained {stats.steps:,} steps in {wall / 60:.1f}m")
    print(f"  nan skips  : {stats.nan_skips}")
    print(f"  encode skips: {stats.encode_skips}")
    print(f"  final loss : {stats.losses[-1]:.4f}")

    meta = save_checkpoint(run / "phase1.pt", model, cfg, stats.steps, **stats.as_dict())
    print(f"  checkpoint : {run / 'phase1.pt'}  ({meta['step']} steps)")

    # --- held-out metrics -------------------------------------------------
    result = {"train": stats.as_dict(), "wall_minutes": round(wall / 60, 2)}
    if (args.eval / "meta.json").exists():
        held = SupervisionSet(args.eval)
        print(f"\n  held-out top-k on {len(held):,} states — FULL SET (F-09: 21.3% seen)")
        full = top_k_by_depth(model, held, cfg, device=args.device)
        result["heldout_full"] = full
        print(f"    {'depth':>6} {'n':>7} {'top1':>8} {'top8':>8}")
        for depth, row in full["by_depth"].items():
            print(f"    {depth:>6} {row['n']:>7,} {row['top1']:>8.4f} {row['top8']:>8.4f}")
        print(f"    depth<=3 top8: {full['depth_le_3_top8']}  (n={full['depth_le_3_n']:,})")

        print("\n  computing the unseen-subset exclusion (F-09)...")
        shared = shared_state_indices(args.eval)
        clean = top_k_by_depth(model, held, cfg, device=args.device, exclude=shared)
        result["heldout_unseen"] = clean
        result["heldout_shared_excluded"] = len(shared)
        print(f"  held-out top-k on {clean['n']:,} UNSEEN states ({len(shared):,} excluded)")
        print(f"    {'depth':>6} {'n':>7} {'top1':>8} {'top8':>8}")
        for depth, row in clean["by_depth"].items():
            print(f"    {depth:>6} {row['n']:>7,} {row['top1']:>8.4f} {row['top8']:>8.4f}")
        print(f"    depth<=3 top8: {clean['depth_le_3_top8']}  (n={clean['depth_le_3_n']:,})")

    write_record(run / "phase1_result.json", result)
    print(f"\n  wrote {run / 'phase1_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
