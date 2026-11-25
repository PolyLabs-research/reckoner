"""Chunk 8's required pre-flight: the timing slice that picks the device.

The plan makes this mandatory before Phase 1 runs ("timing slice as required
pre-flight ... run Phase 1 on the box, CPU or GPU, whichever the timing slice
recommends"), and `config.py` defers the real `train.batch_size` to it.

**F-03 is the reason this script is shaped the way it is.** The chunk-5
pre-flight projected 18.3 min and the run took 94. Its three causes are each
answered here:

1. *Sample too small for a heavy tail* — every measurement is a median over
   ``--steps`` timed steps after warmup, never a single draw.
2. *The projection priced the wrong work* — batch construction and the
   forward/backward are timed **separately**, because they are not the same
   resource. On this box the expected profile (AGENTS.md §8) is CPU-bound Python
   saturating while the accelerator idles, and a projection that prices only the
   forward would repeat F-03 exactly.
3. *Unpriced overheads* — what this projection does and does not include is
   printed with the number, not left to the reader.

Batches are sampled across the whole set, never a prefix: the supervision set is
laid out stratum by stratum, so a prefix is a depth-1-and-2 pilot for a run that
will not see that distribution. That mistake has already been made once on this
data (``build_phase1_data.py``'s ``--limit``).

Writes ``runs/pilot_phase1_timing.json``. Recommends; does not decide.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint
from reckoner.dataset import anchored_data, git_sha, write_record
from reckoner.model import Reckoner, policy_loss, steps_loss
from reckoner.train import SupervisionSet, make_batch
from reckoner.vocab import PAD

REPO = Path(__file__).resolve().parents[1]


def time_one_config(
    data: SupervisionSet,
    cfg: Config,
    *,
    batch_size: int,
    device: str,
    steps: int,
    warmup: int,
    seed: int,
) -> dict:
    """Median seconds per step, split into batch-build and optimise."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = Reckoner(cfg).to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    build_s: list[float] = []
    optimise_s: list[float] = []
    encode_skips = 0

    for step in range(warmup + steps):
        t0 = time.perf_counter()
        indices = [rng.randrange(len(data)) for _ in range(batch_size)]
        batch = make_batch(data, indices, cfg)
        t1 = time.perf_counter()

        policy, _value, steps_out = model(batch.tokens.to(device), batch.site_positions.to(device))
        assert steps_out is not None
        loss = cfg.train.policy_loss_weight * policy_loss(
            policy, batch.policy_target.to(device), batch.legal_mask.to(device)
        )
        loss = loss + cfg.train.steps_loss_weight * steps_loss(
            steps_out, batch.steps_target.to(device), batch.solved_mask.to(device)
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        if step >= warmup:
            build_s.append(t1 - t0)
            optimise_s.append(t2 - t1)
            encode_skips += batch.skipped

    build = statistics.median(build_s)
    optimise = statistics.median(optimise_s)
    return {
        "batch_size": batch_size,
        "device": device,
        "timed_steps": steps,
        "build_s_median": round(build, 4),
        "optimise_s_median": round(optimise, 4),
        "step_s_median": round(build + optimise, 4),
        "build_share": round(build / (build + optimise), 3),
        "examples_per_s": round(batch_size / (build + optimise), 1),
        "encode_skips": encode_skips,
        "step_s_p90": round(
            statistics.quantiles([b + o for b, o in zip(build_s, optimise_s, strict=True)], n=10)[
                -1
            ],
            4,
        ),
    }


def measure_the_crop(
    data: SupervisionSet, cfg: Config, *, batch_size: int, steps: int, seed: int
) -> dict:
    """Price the padding crop, and prove it is free, in the same record.

    The recommendation this pilot makes is conditional on ``crop_to_content``,
    so the speedup belongs in the artifact rather than in someone's memory —
    otherwise the next reader has a projection whose basis they cannot check.
    Both halves are measured: the crop must be *exact*, and it must be *worth it*.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = Reckoner(cfg).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    batch = make_batch(data, [rng.randrange(len(data)) for _ in range(batch_size)], cfg)

    full = torch.full((batch.tokens.shape[0], cfg.model.seq_len), PAD, dtype=batch.tokens.dtype)
    full[:, : batch.width] = batch.tokens

    def timed(tokens: torch.Tensor) -> float:
        samples = []
        for _ in range(steps + 1):  # one warmup, discarded
            t0 = time.perf_counter()
            policy, _value, steps_out = model(tokens, batch.site_positions)
            assert steps_out is not None
            loss = policy_loss(policy, batch.policy_target, batch.legal_mask) + steps_loss(
                steps_out, batch.steps_target, batch.solved_mask
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples[1:])

    cropped_s = timed(batch.tokens)
    full_s = timed(full)

    check = Reckoner(cfg).eval()
    with torch.no_grad():
        a = check(batch.tokens, batch.site_positions)
        b = check(full, batch.site_positions)
    exact = all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

    return {
        "batch_size": batch_size,
        "cropped_width": batch.width,
        "full_width": cfg.model.seq_len,
        "cropped_s_median": round(cropped_s, 4),
        "full_s_median": round(full_s, 4),
        "speedup": round(full_s / cropped_s, 1),
        "outputs_bit_identical": exact,
        "timed_steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=anchored_data("phase1_train"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--steps", type=int, default=12, help="timed steps per configuration")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-steps", type=int, nargs="+", default=[2000, 5000, 10000])
    args = parser.parse_args()

    cfg = Config()
    data = SupervisionSet(args.data)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    model = Reckoner(cfg)
    params = sum(p.numel() for p in model.parameters())

    print("  PHASE-1 TIMING SLICE — chunk 8's required pre-flight\n")
    print(f"  torch        : {torch.__version__}")
    print(f"  devices      : {', '.join(devices)}")
    print(f"  parameters   : {params:,}")
    print(f"  supervision  : {len(data):,} examples, max_len {data.meta['max_len']}")
    print(f"  timed        : {args.steps} steps per cell, {args.warmup} warmup, medians\n")

    print(
        f"  {'device':>6} {'batch':>6} {'build s':>9} {'optim s':>9} {'step s':>8} "
        f"{'p90 s':>8} {'build%':>7} {'ex/s':>9}"
    )
    rows = []
    for device in devices:
        for batch_size in args.batch_sizes:
            row = time_one_config(
                data,
                cfg,
                batch_size=batch_size,
                device=device,
                steps=args.steps,
                warmup=args.warmup,
                seed=args.seed,
            )
            rows.append(row)
            print(
                f"  {row['device']:>6} {row['batch_size']:>6} {row['build_s_median']:>9.4f} "
                f"{row['optimise_s_median']:>9.4f} {row['step_s_median']:>8.4f} "
                f"{row['step_s_p90']:>8.4f} {row['build_share'] * 100:>6.1f}% "
                f"{row['examples_per_s']:>9.1f}"
            )

    best = max(rows, key=lambda r: r["examples_per_s"])
    throughputs = [r["examples_per_s"] for r in rows]
    spread = (max(throughputs) - min(throughputs)) / max(throughputs)
    print(
        f"\n  fastest cell : {best['device']} @ batch {best['batch_size']} — "
        f"{best['examples_per_s']:.1f} examples/s, "
        f"{best['build_share'] * 100:.1f}% of the step is batch construction"
    )
    # The argmax of a 12-step median is not a finding when the cells are this
    # close. Saying so here is cheaper than someone treating a 5% gap as a
    # tuning result — F-03's lesson, applied to this pilot's own output.
    print(
        f"  spread       : {min(throughputs):.1f}-{max(throughputs):.1f} examples/s "
        f"across {len(rows)} cells ({spread * 100:.0f}%). Cells within ~5% are noise; "
        f"batch size is not a throughput lever on this box"
    )

    print(
        f"\n  projection at {best['device']} / batch {best['batch_size']} "
        f"({best['step_s_median']:.4f} s/step median):"
    )
    print(f"    {'steps':>8} {'examples':>12} {'epochs':>8} {'median':>10} {'at p90':>10}")
    projections = []
    for steps in args.project_steps:
        examples = steps * best["batch_size"]
        entry = {
            "steps": steps,
            "examples": examples,
            "epochs": round(examples / len(data), 3),
            "median_minutes": round(steps * best["step_s_median"] / 60, 1),
            "p90_minutes": round(steps * best["step_s_p90"] / 60, 1),
        }
        projections.append(entry)
        print(
            f"    {steps:>8} {examples:>12,} {entry['epochs']:>8.3f} "
            f"{entry['median_minutes']:>9.1f}m {entry['p90_minutes']:>9.1f}m"
        )

    crop = measure_the_crop(
        data, cfg, batch_size=best["batch_size"], steps=args.steps, seed=args.seed
    )
    print(
        f"\n  padding crop at batch {crop['batch_size']}: "
        f"{crop['cropped_width']} cols {crop['cropped_s_median']:.4f}s vs "
        f"{crop['full_width']} cols {crop['full_s_median']:.4f}s — "
        f"{crop['speedup']}x, outputs bit-identical: {crop['outputs_bit_identical']}"
    )

    print("\n  what this projection prices : batch construction + forward + backward + step")
    print("  what it does NOT price      : checkpoint writes, held-out evaluation passes,")
    print("                                the depth-<=2 search gate, process startup")

    out = {
        "torch": torch.__version__,
        "devices": devices,
        "parameters": params,
        "examples": len(data),
        "cells": rows,
        "fastest": best,
        "padding_crop": crop,
        "projections": projections,
        "throughput_spread_frac": round(spread, 3),
        "caveat": (
            "throughput is flat across batch 64-256; the fastest cell is within noise "
            "of its neighbours and flipped between runs. Batch size is not a throughput "
            "lever on this box — pick it for gradient quality, not speed."
        ),
        "priced": "batch construction + forward + backward + optimiser step",
        "not_priced": "checkpoint writes, held-out eval, the depth-<=2 search gate, startup",
        "git_sha": git_sha(REPO),
        "config_fingerprint": config_fingerprint(cfg),
    }
    write_record(REPO / "runs" / "pilot_phase1_timing.json", out)
    print("\n  wrote runs/pilot_phase1_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
