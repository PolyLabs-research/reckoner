"""The rehearsal-fraction sweep, re-run on a ring whose policy targets are real.

Protocol: ``SWEEP-m2.md``, registered before any arm ran. M1's sweep
(``sweep_rehearsal_frac.py``, ``sweep_verify.py``) is left as it was: it is the
record of a measurement taken while F-53 discarded 93% of the ring's policy
target, and its control value (0.8942) is a property of that defect.

Every arm trains from the Phase-1 anchor on the SAME fixed ring-0 and is scored
on gate 10b (top-1, depth <= 3, F-09 unseen subset), under one of the two policy
targets. One arm per process, so arms can run side by side without sharing an
RNG or a thread pool:

    python scripts/sweep_m2.py --list
    python scripts/sweep_m2.py --target visits --arm 0      # writes runs/sweep_m2/visits/arm-00.json
    python scripts/sweep_m2.py --collect                    # the tables and the selection

``--arm anchor`` scores the untrained anchor on this host: the reference every
delta is taken against is measured where the arms are, never carried over.
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

from reckoner.campaign import ANCHOR
from reckoner.config import POLICY_TARGETS, Config, config_fingerprint, validate
from reckoner.dataset import anchored_data, sha256_file
from reckoner.model import load_checkpoint
from reckoner.replay import ReplayRing
from reckoner.train import SupervisionSet, train_on_ring

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "sweep_m2"

REGISTERED_STEPS = 400
#: SWEEP-m2.md §3, carried from SWEEP-m1a4.md §3 (top-1 >= 0.968) and stated as
#: the count it means on the 1,229-row instrument.
BAND_HITS = 1190
INSTRUMENT_N = 1229

#: (rehearsal_frac, steps, lr_warmup_steps, seed, kind, note). SWEEP-m2.md §4.
#: Selection considers ONLY kind == "candidate". Every other arm is a diagnostic
#: and is excluded from candidacy by construction, so it can find an answer and
#: can never move the criterion.
ARMS = (
    (0.00, 400, 200, 0, "candidate", "control: a fresh measurement, not a reproduction target"),
    (0.10, 400, 200, 0, "candidate", ""),
    (0.15, 400, 200, 0, "candidate", ""),
    (0.25, 400, 200, 0, "candidate", ""),
    (0.35, 400, 200, 0, "candidate", ""),
    (0.50, 400, 200, 0, "candidate", ""),
    (0.65, 400, 200, 0, "candidate", "M1-A4's value"),
    (0.75, 400, 200, 0, "candidate", ""),
    (0.00, 400, 200, 1, "variance", "control, second seed"),
    (0.65, 400, 200, 1, "variance", "M1-A4's value, second seed"),
    (0.65, 400, 200, 2, "variance", "M1-A4's value, third seed"),
    (0.00, 200, 200, 0, "mechanism", "F-37's arm: the first 200 steps of the control, all warmup"),
    (0.00, 200, 100, 0, "mechanism", "LR-controlled: the control's whole schedule in 200 steps"),
)

#: SWEEP-m2 §6's follow-up arm set, registered by the page's 2026-11-12 amendment
#: before it ran: the selected 'visits' value, 0.15, at three seeds. Seed 0 is
#: arm 2 run again on a new host, so it must reproduce arm 2's weight digest
#: exactly: that is what makes the two new seeds comparable with the sweep.
FOLLOWUP_ARMS = (
    (0.15, 400, 200, 0, "reproduction", "arm 2 again: must reproduce its weight digest"),
    (0.15, 400, 200, 1, "variance", "selected f, second seed"),
    (0.15, 400, 200, 2, "variance", "selected f, third seed"),
)
FOLLOWUP_TARGET = "visits"
FOLLOWUP_REPRODUCES = 2


def weight_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def flat(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).double() for p in model.parameters()])


def ring_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.iterdir() if p.is_file()):
        digest.update(item.name.encode())
        digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def score(model, cfg: Config) -> dict:
    from train_phase1 import shared_state_indices, top_k_by_depth

    eval_path = anchored_data("phase1_eval", verify=True)
    held = SupervisionSet(eval_path)
    clean = top_k_by_depth(model, held, cfg, device="cpu", exclude=shared_state_indices(eval_path))
    top1, n = clean["depth_le_3_top1"], clean["depth_le_3_n"]
    return {"depth_le_3_top1": top1, "n": n, "hits": round(top1 * n)}


def provenance(cfg: Config, ring_path: Path) -> dict:
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "git_sha": sha,
        "tree_clean": not dirty,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "anchor_sha256": sha256_file(ANCHOR),
        "ring": str(ring_path.relative_to(REPO)),
        "ring_digest": ring_digest(ring_path),
        "base_config_fingerprint": config_fingerprint(cfg),
    }


def run_anchor(cfg: Config, ring_path: Path) -> dict:
    model, _ = load_checkpoint(ANCHOR, cfg)
    model.eval()
    return {"arm": "anchor", **score(model, cfg), **provenance(cfg, ring_path)}


def run_arm(target: str, index: int, cfg: Config, ring_path: Path, arms=ARMS) -> dict:
    """One arm. An arm is ``(f, steps, warmup, seed, kind, note)``, optionally with
    a seventh element: further ``train`` overrides, recorded in the row (SWEEP-m3's
    control arm withholds the policy target this way)."""
    frac, steps, warmup, seed, kind, note, *extra = arms[index]
    overrides = dict(extra[0]) if extra else {}
    arm_cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            rehearsal_frac=frac,
            lr_warmup_steps=warmup,
            policy_target=target,
            **overrides,
        ),
    )
    validate(arm_cfg)
    ring = ReplayRing.load(ring_path, arm_cfg)
    model, _ = load_checkpoint(ANCHOR, arm_cfg)
    anchor_flat = flat(model)
    started = time.monotonic()
    stats = train_on_ring(model, ring, arm_cfg, steps=steps, seed=seed)
    seconds = time.monotonic() - started
    model.eval()
    batch = arm_cfg.train.batch_size
    ring_rows = round(batch * (1 - frac))
    moved = float(torch.linalg.vector_norm(flat(model) - anchor_flat)) / float(
        torch.linalg.vector_norm(anchor_flat)
    )
    scored = score(model, arm_cfg)
    tail = stats.losses[-20:]
    return {
        "arm": index,
        "policy_target": target,
        "rehearsal_frac": frac,
        "steps": steps,
        "lr_warmup_steps": warmup,
        "seed": seed,
        "kind": kind,
        "note": note,
        "train_overrides": overrides,
        **scored,
        "holds_band": scored["hits"] >= BAND_HITS,
        "ring_rows": len(ring),
        "ring_epochs": round(steps * ring_rows / len(ring), 1),
        "value_head_examples_per_step": ring_rows,
        "steps_taken": stats.steps,
        "rehearsal_batches": stats.rehearsal_batches,
        "encode_skips": stats.encode_skips,
        "nan_skips": stats.nan_skips,
        "improved_truncated": stats.improved_truncated,
        "loss_first": stats.losses[0] if stats.losses else None,
        "loss_last20_mean": sum(tail) / len(tail) if tail else None,
        "weight_digest": weight_digest(model),
        "movement_vs_anchor": round(moved, 6),
        "seconds_train": round(seconds, 1),
        **provenance(arm_cfg, ring_path),
    }


def collect(out: Path) -> int:
    anchor_path = out / "anchor.json"
    if not anchor_path.exists():
        print("  no anchor.json: run --arm anchor first")
        return 1
    anchor = json.loads(anchor_path.read_text())
    report: dict = {
        "protocol": "SWEEP-m2.md",
        "band_hits": BAND_HITS,
        "instrument_n": INSTRUMENT_N,
        "anchor": anchor,
        "targets": {},
    }
    complete = anchor["n"] == INSTRUMENT_N
    for target in POLICY_TARGETS:
        rows = []
        for index in range(len(ARMS)):
            path = out / target / f"arm-{index:02d}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
        missing = len(ARMS) - len(rows)
        complete = complete and missing == 0
        candidates = sorted(
            (r for r in rows if r["kind"] == "candidate"), key=lambda r: r["rehearsal_frac"]
        )
        holding = [r for r in candidates if r["holds_band"]]
        selected = holding[0]["rehearsal_frac"] if holding and not missing else None
        report["targets"][target] = {"arms": rows, "missing_arms": missing, "selected": selected}

        print(f"\n  SWEEP-m2 — policy target '{target}'   ({len(rows)}/{len(ARMS)} arms)")
        print(
            f"  band: hits >= {BAND_HITS}/{INSTRUMENT_N}   anchor here: {anchor['hits']}/{anchor['n']}\n"
        )
        print(
            f"    {'f':>5} {'steps':>5} {'warm':>5} {'seed':>4} {'hits':>10} {'delta':>6} "
            f"{'ring-ep':>7} {'moved':>8} {'loss':>8}  band kind"
        )
        for r in rows:
            print(
                f"    {r['rehearsal_frac']:>5.2f} {r['steps']:>5} {r['lr_warmup_steps']:>5} "
                f"{r['seed']:>4} {r['hits']:>5}/{r['n']:<4} {r['hits'] - anchor['hits']:>+6} "
                f"{r['ring_epochs']:>7.1f} {r['movement_vs_anchor']:>8.5f} "
                f"{r['loss_last20_mean']:>8.3f}  {'YES' if r['holds_band'] else 'no':<4} {r['kind']}"
            )
        if missing:
            print(f"\n    INCOMPLETE: {missing} arm(s) missing; no selection is made")
        elif selected is None:
            print("\n    NO candidate holds the band (SWEEP-m2.md §5, second branch)")
        else:
            print(
                f"\n    smallest f holding the band: {selected:.2f}  (a candidate, not a decision)"
            )
    report["complete"] = complete
    report["followup"] = collect_followup(out, report)
    (out.parent / "sweep_m2.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n  written: {out.parent / 'sweep_m2.json'}   complete: {complete}")
    return 0 if complete else 1


def collect_followup(out: Path, report: dict) -> dict | None:
    """§6's verdict on the selected 'visits' value, once its seeds exist."""
    rows = []
    for index in range(len(FOLLOWUP_ARMS)):
        path = out / FOLLOWUP_TARGET / f"followup-{index:02d}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if not rows:
        return None
    original = next(
        (r for r in report["targets"][FOLLOWUP_TARGET]["arms"] if r["arm"] == FOLLOWUP_REPRODUCES),
        None,
    )
    repro = next((r for r in rows if r["kind"] == "reproduction"), None)
    reproduced = (
        repro is not None
        and original is not None
        and repro["weight_digest"] == original["weight_digest"]
        and repro["hits"] == original["hits"]
    )
    complete = len(rows) == len(FOLLOWUP_ARMS)
    holds = complete and reproduced and all(r["holds_band"] for r in rows)
    print(f"\n  SWEEP-m2 §6 follow-up — '{FOLLOWUP_TARGET}' at f = 0.15   ({len(rows)}/3 arms)\n")
    for r in rows:
        print(
            f"    seed {r['seed']}  {r['hits']:>5}/{r['n']:<4}  "
            f"{'YES' if r['holds_band'] else 'no':<4} {r['kind']:<12} {r['weight_digest'][:12]}"
        )
    if original is not None:
        print(f"    arm {FOLLOWUP_REPRODUCES} (the sweep's own)   {original['weight_digest'][:12]}")
    print(f"    reproduced: {reproduced}   all seeds hold: {holds}")
    return {
        "arms": rows,
        "reproduces_arm": FOLLOWUP_REPRODUCES,
        "reproduced": reproduced,
        "complete": complete,
        "selection_established": holds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", choices=POLICY_TARGETS)
    parser.add_argument("--arm", help="an index from --list, or 'anchor'")
    parser.add_argument("--ring", type=Path, default=REPO / "runs" / "ring0_replay" / "ring-0")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--followup", type=int, help="an index into FOLLOWUP_ARMS (§6)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    if args.list:
        for index, arm in enumerate(ARMS):
            print(index, *arm)
        return 0
    if args.collect:
        return collect(args.out_dir)

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    if args.arm == "anchor":
        path = args.out_dir / "anchor.json"
        row = run_anchor(cfg, args.ring)
    elif args.followup is not None:
        path = args.out_dir / FOLLOWUP_TARGET / f"followup-{args.followup:02d}.json"
        row = run_arm(FOLLOWUP_TARGET, args.followup, cfg, args.ring, arms=FOLLOWUP_ARMS)
        row["arm"] = f"followup-{args.followup}"
    else:
        if args.target is None or args.arm is None:
            parser.error("--target and --arm are both required to run an arm")
        index = int(args.arm)
        path = args.out_dir / args.target / f"arm-{index:02d}.json"
        row = run_arm(args.target, index, cfg, args.ring)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
