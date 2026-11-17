"""F-58's benchmark: what padding every search leaf to ``model.seq_len`` cost.

Times the search evaluator on real search states (ring-V's, which include the
long ones) twice: as it now runs, cropped to the batch's content, and as it ran
until F-58, padded to the full envelope. The padded path is rebuilt here by hand,
because the library no longer has one. Both must return the same bits; the row
records that too.

Appends one row to ``benchmarks/results.jsonl`` (AGENTS §7).

    python scripts/bench_evaluator_crop.py --threads 2
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from reckoner.campaign import ANCHOR
from reckoner.config import Config
from reckoner.episode import Problem, decode_state
from reckoner.evaluate import model_evaluator
from reckoner.model import encode, load_checkpoint
from reckoner.replay import ReplayRing

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--states", type=int, default=192)
    parser.add_argument("--batch", type=int, default=48)
    args = parser.parse_args()

    cfg = Config()
    torch.set_num_threads(args.threads)
    model, _ = load_checkpoint(ANCHOR, cfg)
    model.eval()
    ring = ReplayRing.load(REPO / "runs" / "ring_v" / "ring-0", cfg)
    rng = random.Random(0)
    leaves = []
    for i in [rng.randrange(len(ring)) for _ in range(args.states)]:
        record = ring.get(i)
        goal, target, expr = decode_state(tuple(int(t) for t in record["tokens"]))
        leaves.append(
            (
                Problem(
                    goal=goal, expr=expr, par=int(record["par"]), target=target, par_source="bfs"
                ),
                expr,
            )
        )
    evaluator = model_evaluator(model, cfg, 1.0)

    cropped_s = padded_s = 0.0
    identical = 0
    widths = []
    for start in range(0, len(leaves), args.batch):
        chunk = leaves[start : start + args.batch]
        t0 = time.perf_counter()
        got = evaluator(chunk)
        cropped_s += time.perf_counter() - t0

        encoded = [encode(problem, expr, cfg) for problem, expr in chunk]
        tokens = torch.stack([e.tokens for e in encoded])
        sites = torch.stack([e.site_positions for e in encoded])
        widths.append(max(e.length for e in encoded))
        t0 = time.perf_counter()
        with torch.no_grad():
            policy, value_logits, _ = model(tokens, sites)
            probs = torch.softmax(value_logits, dim=1)
            expected = (probs[:, 0] - probs[:, 2]).tolist()
        padded_s += time.perf_counter() - t0
        for slot, (prior, value) in enumerate(got):
            same = np.array_equal(prior, policy[slot].numpy().astype(np.float32))
            identical += int(same and value == float(expected[slot]))

    row = {
        "bench": "evaluator_crop",
        "git_sha": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "threads": args.threads,
        "states": len(leaves),
        "batch": args.batch,
        "envelope_tokens": cfg.model.seq_len,
        "content_tokens_max": max(widths),
        "seconds_cropped": round(cropped_s, 3),
        "seconds_padded": round(padded_s, 3),
        "padded_over_cropped": round(padded_s / cropped_s, 1),
        "states_bit_identical": identical,
        "note": "cropped includes encode(); padded times the forward pass only, so the ratio is a floor",
    }
    with (REPO / "benchmarks" / "results.jsonl").open("a") as out:
        out.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
