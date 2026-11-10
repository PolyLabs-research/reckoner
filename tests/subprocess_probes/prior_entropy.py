"""Isolate the first-call variance in `entropy(result.root_priors)`.

Three arms, predictions registered before the run (see the resume gate's
finding):

1. ``double``  — two identical searches in ONE process. If the second differs
   from the first, the variance is per-call rather than first-call.
2. ``single``  — one search in a fresh process. Run twice, this should
   reproduce the observed run-to-run difference.
3. ``warmup``  — a discarded search first, then the measured one, in a fresh
   process. If the measured values now agree across processes, first-call state
   is proven and the cause is named.

The measured quantity is exactly the row's: ``entropy(result.root_priors)``,
the float-derived column that varied. ``visit_share()`` (then ``improved_policy()``) — visit counts,
integers — did not vary, which is the asymmetry this probe is chasing.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import random  # noqa: E402

import torch  # noqa: E402

from reckoner.config import Config, validate  # noqa: E402
from reckoner.dataset import read_suite, suite_problem  # noqa: E402
from reckoner.evaluate import model_evaluator  # noqa: E402
from reckoner.model import load_checkpoint  # noqa: E402
from reckoner.runner import entropy  # noqa: E402
from reckoner.search import search  # noqa: E402


def measure(model, cfg, problem) -> float:
    result = search(
        problem,
        problem.expr,
        model_evaluator(model, cfg, 0.0),
        cfg,
        random.Random(5),
        sims=cfg.search.sims,
        m=cfg.search.gumbel_m,
    )
    return entropy(result.root_priors)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)

    model, _ = load_checkpoint(REPO / "runs" / "phase1" / "phase1.pt", cfg)
    model.eval()
    rows = read_suite(REPO / "runs" / "suites" / "solve_in_3.jsonl")
    problem = suite_problem(rows[0])

    if mode == "double":
        print(f"{measure(model, cfg, problem):.17f} {measure(model, cfg, problem):.17f}")
    elif mode == "warmup":
        measure(model, cfg, problem)  # discarded
        print(f"{measure(model, cfg, problem):.17f}")
    else:
        print(f"{measure(model, cfg, problem):.17f}")
