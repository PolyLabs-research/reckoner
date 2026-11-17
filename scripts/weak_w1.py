"""WEAK-w1 — the weak-start benchmark's runs and readings. Protocol: ``WEAK-w1.md``.

Registered before any arm ran. Three arms, each one config key on from the last
(``configs/w1_a.yaml``, ``w1_b.yaml``, ``w1_v.yaml``), three seeds each, all from
one starting network supervised only on problems of depth <= 3:

    python scripts/weak_w1.py run --arm B --seed 43        # one loop run
    python scripts/weak_w1.py read --arm B --seed 43       # its final checkpoint, read
    python scripts/weak_w1.py baseline --arm B             # the weak start itself, under arm B's search
    python scripts/weak_w1.py reference                    # the Phase-1 anchor, under the merged search
    python scripts/weak_w1.py collect                      # runs/w1/weak_w1_result.json

**Nothing here is a second loop or a second instrument.** A run is
``campaign.run`` (the driver every campaign uses) at an arm's config. A reading is
the campaign's own no-regress pass: ``ladderpass.run_pass`` with ``ModelArm`` at
48 simulations and at 1, over the six frozen exact-par suites, one row per
problem. ``collect`` counts at-par rows and runs the paired bootstrap of record.

At par is the reading because it is exact: an episode ends at par only if every
action in it was optimal, and par on these suites is the BFS minimum.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.campaign import ANCHOR, SUITES, eval_profile, run
from reckoner.config import (
    Config,
    config_diff,
    config_fingerprint,
    load_config,
    validate,
    with_m1_search,
)
from reckoner.dataset import git_sha, read_suite, sha256_file, suite_problem, write_record
from reckoner.evaluate import ModelArm
from reckoner.ladder import paired_bootstrap
from reckoner.ladderpass import is_complete, read_pair_scores, run_pass
from reckoner.model import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs" / "w1"
WEAK_ANCHOR = REPO / "runs" / "phase1_d3" / "phase1.pt"
ARMS = {"A": "w1_a.yaml", "B": "w1_b.yaml", "V": "w1_v.yaml"}
SEEDS = (42, 43, 44)
#: The campaign instrument's two budgets: the full search, and the policy alone.
BUDGETS = (48, 1)
DEPTHS = (1, 2, 3, 4, 5, 6)
PER_SUITE = 200
READ_THREADS = 2

#: WEAK-w1.md §6, as counts. Scored by ``collect``; never edited after a run.
GATES = {
    "P1_anchor_deep_greedy_at_most": 200,  # of 600: the benchmark has headroom
    "P1_reference_deep_greedy_at_least": 560,  # of 600: and supervision closes it
    "P2_A_shallow_greedy_loss_at_least": 20,  # of 600, mean over seeds
    "P3_B_shallow_greedy_loss_at_most": 6,  # of 600, every seed
    "P5_V_far_greedy_at_least": 300,  # of 400 (depths 5-6), every seed
    "P6_V_deep_search_at_least": 570,  # of 600 (depths 4-6), every seed
    "P7_V_short_of_reference_at_most": 60,  # of 1,200, greedy, every seed
}


def arm_config(arm: str, seed: int = 42) -> Config:
    cfg = replace(load_config(REPO / "configs" / ARMS[arm]), seed=seed)
    validate(cfg)
    return cfg


def suite_problems() -> list:
    """The six exact-par suites, 200 each, in the campaign instrument's own order."""
    return [
        suite_problem(row)
        for depth in DEPTHS
        for row in read_suite(SUITES / f"solve_in_{depth}.jsonl")[:PER_SUITE]
    ]


def read_checkpoint(
    checkpoint: Path, cfg: Config, root: Path, index: int, *, role: str, note: str
) -> list[dict]:
    """The campaign's no-regress pass on one checkpoint, under ``cfg``'s search.

    Two threads: a pass is one small forward at a time, and several passes run
    side by side on the pod. The thread count does not change what is computed
    (F-58's replay gate), only how many passes fit on the machine.
    """
    torch.set_num_threads(READ_THREADS)
    ev = eval_profile(cfg)
    model, _ = load_checkpoint(checkpoint, ev)
    model.eval()
    arms = [
        ModelArm(model, name=f"model@{sims}", sims=sims, m=min(ev.search.gumbel_m, sims))
        for sims in BUDGETS
    ]
    if not is_complete(root, index):
        run_pass(
            root,
            index,
            arms,
            suite_problems(),
            ev,
            roles={arm.name: role for arm in arms},
            calibration_note=note,
            seed=0,
        )
    return read_pair_scores(root, index)


def provenance(cfg: Config, checkpoint: Path) -> dict:
    return {
        "git_sha": git_sha(REPO),
        "torch": torch.__version__,
        "config_fingerprint": config_fingerprint(cfg),
        "config_diff_from_default": {k: list(v) for k, v in config_diff(Config(), cfg).items()},
        "checkpoint": str(checkpoint.resolve().relative_to(REPO)),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


# ------------------------------------------------------------------ commands


def cmd_run(args: argparse.Namespace) -> int:
    cfg = arm_config(args.arm, args.seed)
    run_dir = ROOT / f"{args.arm}-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "status": "started",
        "arm": args.arm,
        "seed": args.seed,
        "anchor_sha256": sha256_file(WEAK_ANCHOR),
        "started_unix": time.time(),
        **provenance(cfg, WEAK_ANCHOR),
    }
    summary_path = run_dir / "campaign_summary.json"
    if not summary_path.exists():
        summary_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    summary = run(run_dir, cfg, run_name=f"w1-{args.arm}", anchor=WEAK_ANCHOR)
    done = identity | {"status": "complete", "seconds": round(time.perf_counter() - started, 1)}
    summary_path.write_text(json.dumps(done | summary, indent=2, sort_keys=True) + "\n")
    print(
        f"  w1 {args.arm} seed {args.seed}: committed {summary['committed']}, alarms {summary['alarms']}"
    )
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    cfg = arm_config(args.arm, args.seed)
    run_dir = ROOT / f"{args.arm}-{args.seed}"
    last = cfg.campaign.iterations - 1
    checkpoint = run_dir / f"ckpt-{last}.pt"
    rows = read_checkpoint(
        checkpoint,
        cfg,
        run_dir / "ladder" / "w1",
        last,
        role="subject",
        note=f"WEAK-w1 arm {args.arm} seed {args.seed}: the final checkpoint on the exact-par suites",
    )
    write_record(run_dir / "reading_result.json", provenance(cfg, checkpoint) | {"rows": len(rows)})
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    cfg = arm_config(args.arm)
    root = ROOT / "baseline" / args.arm
    rows = read_checkpoint(
        WEAK_ANCHOR,
        cfg,
        root,
        0,
        role="baseline",
        note=f"WEAK-w1: the weak starting network itself, under arm {args.arm}'s search",
    )
    write_record(root / "reading_result.json", provenance(cfg, WEAK_ANCHOR) | {"rows": len(rows)})
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    # "Under the merged search", as registered: the default of the day the round
    # ran, named outright since the defaults moved (2026-11-16).
    cfg = with_m1_search(Config())
    root = ROOT / "reference"
    rows = read_checkpoint(
        ANCHOR,
        cfg,
        root,
        0,
        role="baseline",
        note="WEAK-w1: the Phase-1 anchor (supervised on every depth), under the merged search",
    )
    write_record(root / "reading_result.json", provenance(cfg, ANCHOR) | {"rows": len(rows)})
    return 0


# ------------------------------------------------------------------- collect


def at_par_table(rows: list[dict]) -> dict:
    """``{budget: {depth: at-par count}}`` and the per-problem at-par indicator."""
    cap = Config().episode.step_cap
    counts = {str(sims): {str(d): 0 for d in DEPTHS} for sims in BUDGETS}
    solved = {str(sims): {str(d): 0 for d in DEPTHS} for sims in BUDGETS}
    by_problem: dict[str, dict[str, int]] = {str(sims): {} for sims in BUDGETS}
    depth_of: dict[str, str] = {}
    for row in rows:
        sims = row["arm"].split("@")[1]
        at_par = int(int(row["z"]) >= 0)
        counts[sims][str(row["par"])] += at_par
        solved[sims][str(row["par"])] += int(row["steps"] < cap)
        by_problem[sims][row["problem_key"]] = at_par
        depth_of[row["problem_key"]] = str(row["par"])
    return {"at_par": counts, "solved": solved, "by_problem": by_problem, "depth_of": depth_of}


def _sum(table: dict, sims: int, depths) -> int:
    return sum(table["at_par"][str(sims)][str(d)] for d in depths)


def _paired(tables_x: list[dict], tables_y: list[dict], sims: int, depths, resamples: int) -> dict:
    """X minus Y on the at-par indicator, per problem, averaged over the seeds."""
    keys = sorted(tables_x[0]["by_problem"][str(sims)])
    wanted = {str(d) for d in depths}
    problem_depth = tables_x[0]["depth_of"]
    differences = []
    for key in keys:
        if problem_depth[key] not in wanted:
            continue
        x = sum(t["by_problem"][str(sims)][key] for t in tables_x) / len(tables_x)
        y = sum(t["by_problem"][str(sims)][key] for t in tables_y) / len(tables_y)
        differences.append(x - y)
    return {"pairs": len(differences), **paired_bootstrap(differences, resamples=resamples, seed=0)}


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = Config()
    report: dict = {"protocol": "WEAK-w1.md", "gates": GATES, "arms": {}, "missing": []}

    def load(root: Path, index: int, label: str) -> dict | None:
        if not is_complete(root, index):
            report["missing"].append(label)
            return None
        return at_par_table(read_pair_scores(root, index))

    reference = load(ROOT / "reference", 0, "reference")
    baselines = {arm: load(ROOT / "baseline" / arm, 0, f"baseline {arm}") for arm in ARMS}
    finals: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        last = arm_config(arm).campaign.iterations - 1
        finals[arm] = {}
        for seed in SEEDS:
            table = load(ROOT / f"{arm}-{seed}" / "ladder" / "w1", last, f"{arm}-{seed}")
            if table is not None:
                finals[arm][seed] = table

    def public(table: dict | None) -> dict | None:
        return None if table is None else {k: table[k] for k in ("at_par", "solved")}

    report["reference"] = public(reference)
    for arm in ARMS:
        report["arms"][arm] = {
            "baseline": public(baselines[arm]),
            "final": {str(seed): public(t) for seed, t in finals[arm].items()},
        }
    complete = not report["missing"]
    report["complete"] = complete

    print("\n  WEAK-w1 — at par by problem depth 1..6 (of 200 each)\n")
    line = "    {:<22} greedy {}   search {}"

    def fmt(table: dict, sims: int) -> str:
        return " ".join(f"{table['at_par'][str(sims)][str(d)]:>3}" for d in DEPTHS)

    if reference:
        print(line.format("Phase-1 reference", fmt(reference, 1), fmt(reference, 48)))
    for arm in ARMS:
        if baselines[arm]:
            print(
                line.format(
                    f"weak start, {arm}'s search", fmt(baselines[arm], 1), fmt(baselines[arm], 48)
                )
            )
        for seed, table in finals[arm].items():
            print(line.format(f"arm {arm} seed {seed}", fmt(table, 1), fmt(table, 48)))

    if complete:
        shallow, deep, far, every = (1, 2, 3), (4, 5, 6), (5, 6), DEPTHS
        anchor = baselines["A"]
        a, b, v = (list(finals[arm].values()) for arm in ("A", "B", "V"))
        resamples = cfg.ladder.bootstrap_resamples
        loss_a = [_sum(anchor, 1, shallow) - _sum(t, 1, shallow) for t in a]
        loss_b = [_sum(anchor, 1, shallow) - _sum(t, 1, shallow) for t in b]
        verdicts = {
            "P1": _sum(anchor, 1, deep) <= GATES["P1_anchor_deep_greedy_at_most"]
            and _sum(reference, 1, deep) >= GATES["P1_reference_deep_greedy_at_least"],
            "P2": sum(loss_a) / len(loss_a) >= GATES["P2_A_shallow_greedy_loss_at_least"],
            "P3": max(loss_b) <= GATES["P3_B_shallow_greedy_loss_at_most"],
            "P5": min(_sum(t, 1, far) for t in v) >= GATES["P5_V_far_greedy_at_least"],
            "P6": min(_sum(t, 48, deep) for t in v) >= GATES["P6_V_deep_search_at_least"],
            "P7": max(_sum(reference, 1, every) - _sum(t, 1, every) for t in v)
            <= GATES["P7_V_short_of_reference_at_most"],
        }
        primaries = {
            "B_minus_A_greedy_all_depths": _paired(b, a, 1, every, resamples),
            "V_minus_B_greedy_all_depths": _paired(v, b, 1, every, resamples),
            "V_minus_B_greedy_depths_5_6": _paired(v, b, 1, far, resamples),
        }
        verdicts["P4"] = bool(
            primaries["B_minus_A_greedy_all_depths"]["excludes_zero"]
            and primaries["B_minus_A_greedy_all_depths"]["mean_difference"] > 0
        )
        verdicts["P5_paired"] = bool(
            primaries["V_minus_B_greedy_depths_5_6"]["excludes_zero"]
            and primaries["V_minus_B_greedy_depths_5_6"]["mean_difference"] > 0
        )
        report["readings"] = {
            "anchor_greedy_shallow_deep": [_sum(anchor, 1, shallow), _sum(anchor, 1, deep)],
            "reference_greedy_all": _sum(reference, 1, every),
            "A_shallow_loss_by_seed": loss_a,
            "B_shallow_loss_by_seed": loss_b,
            "greedy_all_by_seed": {
                arm: [_sum(t, 1, every) for t in finals[arm].values()] for arm in ARMS
            },
            "V_far_greedy_by_seed": [_sum(t, 1, far) for t in v],
            "V_deep_search_by_seed": [_sum(t, 48, deep) for t in v],
        }
        report["primaries"] = primaries
        report["predictions"] = verdicts
        print(
            "\n  predictions:",
            {k: ("holds" if ok else "FAILS") for k, ok in sorted(verdicts.items())},
        )
        for name, result in primaries.items():
            print(
                f"    {name}: mean {result['mean_difference']:+.4f}  CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  excludes zero: {result['excludes_zero']}"
            )
    else:
        print(f"\n  INCOMPLETE — missing: {report['missing']}. No prediction is scored.")

    write_record(ROOT / "weak_w1_result.json", report)
    print(f"\n  wrote {ROOT / 'weak_w1_result.json'}   complete: {complete}")
    return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, needs_seed in (
        ("run", cmd_run, True),
        ("read", cmd_read, True),
        ("baseline", cmd_baseline, False),
    ):
        p = sub.add_parser(name)
        p.add_argument("--arm", choices=sorted(ARMS), required=True)
        if needs_seed:
            p.add_argument("--seed", type=int, choices=SEEDS, required=True)
        p.set_defaults(handler=handler)
    sub.add_parser("reference").set_defaults(handler=cmd_reference)
    sub.add_parser("collect").set_defaults(handler=cmd_collect)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
