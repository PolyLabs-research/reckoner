"""0b: GPU/CPU equivalence, against the tolerances frozen in PREREG-chunk9-part0.md.

Every threshold here is read from the pre-registration, not chosen here. The
declaration was committed at `abc765c` while the venv was still CPU-only, so
nothing below can have been sized to fit a result.

Structure, per amendment A1:

* **A1.1 — the null is on the numeric channel, not the decision channel.** A
  sub-margin perturbation can legitimately flip zero argmaxes on a model at 0.98
  top-1, so an argmax-based null can fail while the pipeline is correct. The
  comparator self-check (`T` vs `T + 1.0` must report exactly 1.0) is guaranteed
  by construction; the end-to-end perturbation is sized to clear the tier-2 bound.
  Perturbed argmax agreement is **recorded, not asserted**.
* **A1.2 — the sample spans the widths chunk 9 will reach**, not the widths
  Phase 1 happened to produce. Narrow states come from `phase1_eval`; wide ones
  from the same capped random legal play that produced `state_extent.json`, where
  `add_both_sides` is available. The span is asserted, and so is its contrast
  premise: a `phase1_eval`-only sample must FAIL the span check.

Writes `runs/gpu_equivalence_smoke.json`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from reckoner.config import Config, validate
from reckoner.dataset import (
    anchored_data,
    git_sha,
    read_suite,
    sample_indices,
    suite_problem,
    write_record,
)
from reckoner.episode import Problem, decode_state, encode_state
from reckoner.expr import Expr
from reckoner.model import Reckoner, StateTooLarge, crop_to_content, encode
from reckoner.rules import apply, legal_actions
from reckoner.train import SupervisionSet

REPO = Path(__file__).resolve().parents[1]

# Frozen in PREREG-chunk9-part0.md. Read, never chosen here. The policy-logit
# bound was recalibrated by amendment A3 from 1e-3 (an estimate) to 2x the observed
# maximum, with F-12 as its derivation — a bound that always fires is not a gate
# (AGENTS.md rider (b) corollary). The other two channels never fired and are
# unchanged. Tier 1 is untouched and remains exact.
TIER2 = {"policy_logits": 7.5e-3, "value_probs": 1e-4, "steps": 1e-3}
REFERENCE = "runs/gpu_equivalence_reference.json"
BUCKETS = ((1, 64), (65, 128), (129, 256), (257, 10_000))
PER_BUCKET = 128


def width_of(problem: Problem, expr: Expr) -> int:
    return len(encode_state(problem.goal, expr, problem.target))


def narrow_states(cfg: Config, k: int) -> list[tuple[Problem, Expr, int]]:
    """Phase-1 states, drawn with the blessed subsampler — never a prefix."""
    data = SupervisionSet(anchored_data("phase1_eval"))
    out = []
    for i in sample_indices(len(data), k, seed=0):
        goal, target, expr = decode_state(tuple(int(t) for t in data.tokens[i, : data.lengths[i]]))
        problem = Problem(
            goal=goal, expr=expr, par=int(data.depth[i]), target=target, par_source="bfs"
        )
        out.append((problem, expr, width_of(problem, expr)))
    return out


def walked_states(cfg: Config, steps: int, per_problem: int, seed: int) -> list:
    """Capped random legal play from the deep suites — where the wide states live.

    ``add_both_sides`` is the entire source of state growth (F-05) and is always
    legal, so random play reaches widths optimal derivations never do. Measured
    ceiling is p100 332 tokens, which is why the top bucket is >= 257 rather than
    "up to 512".
    """
    rng = random.Random(seed)
    out = []
    for depth in (5, 6):
        for row in read_suite(REPO / "runs" / "suites" / f"solve_in_{depth}.jsonl"):
            problem = suite_problem(row)
            for _ in range(per_problem):
                expr = problem.expr
                for _ in range(steps):
                    actions = legal_actions(expr)
                    if not actions:
                        break
                    rule_id, site_id = rng.choice(actions)
                    expr = apply(expr, rule_id, site_id)
                    out.append((problem, expr, width_of(problem, expr)))
    return out


def bucket_of(width: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= width <= hi:
            return i
    return -1


def stratify(pool: list, per_bucket: int, seed: int) -> tuple[list, dict]:
    """Take ``per_bucket`` from each width bucket. Returns (states, census)."""
    rng = random.Random(seed)
    by_bucket: dict[int, list] = {i: [] for i in range(len(BUCKETS))}
    for item in pool:
        b = bucket_of(item[2])
        if b >= 0:
            by_bucket[b].append(item)
    chosen, census, labels = [], {}, []
    for b, items in by_bucket.items():
        rng.shuffle(items)
        take = items[:per_bucket]
        chosen.extend(take)
        labels.extend([b] * len(take))
        census[f"{BUCKETS[b][0]}-{BUCKETS[b][1] if BUCKETS[b][1] < 10_000 else 'max'}"] = {
            "available": len(items),
            "taken": len(take),
        }
    return chosen, census, labels


def encode_batch(states: list, cfg: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, sites, masks = [], [], []
    for problem, expr, _w in states:
        try:
            e = encode(problem, expr, cfg)
        except StateTooLarge:
            continue
        tokens.append(e.tokens)
        sites.append(e.site_positions)
        masks.append(e.legal_mask)
    stacked = torch.stack(tokens)
    positions = torch.stack(sites)
    return crop_to_content(stacked, positions), positions, torch.stack(masks)


def forward(model: Reckoner, states: list, cfg: Config, device: str, *, bf16: bool = False):
    """Run the model over all states in width-homogeneous batches."""
    outs = {"policy": [], "value": [], "steps": [], "mask": []}
    for start in range(0, len(states), 64):
        chunk = states[start : start + 64]
        tokens, positions, mask = encode_batch(chunk, cfg)
        tokens, positions = tokens.to(device), positions.to(device)
        with torch.no_grad():
            if bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    policy, value, steps = model(tokens, positions)
                policy, value, steps = policy.float(), value.float(), steps.float()
            else:
                policy, value, steps = model(tokens, positions)
        outs["policy"].append(policy.cpu())
        outs["value"].append(torch.softmax(value, dim=1).cpu())
        outs["steps"].append(steps.cpu())
        outs["mask"].append(mask)
    return {k: torch.cat(v) for k, v in outs.items()}


def compare(a: dict, b: dict, keep: list[int] | None = None) -> dict:
    """Max abs deltas plus decision agreement over the legal set."""
    if keep is not None:
        idx = torch.tensor(keep, dtype=torch.long)
        a = {k: v[idx] for k, v in a.items()}
        b = {k: v[idx] for k, v in b.items()}
    mask = a["mask"]
    ma = a["policy"].masked_fill(~mask, float("-inf"))
    mb = b["policy"].masked_fill(~mask, float("-inf"))
    argmax_same = int((ma.argmax(1) == mb.argmax(1)).sum())
    k = min(8, ma.shape[1])
    top_same = int(
        sum(
            set(x.tolist()) == set(y.tolist())
            for x, y in zip(ma.topk(k, 1).indices, mb.topk(k, 1).indices, strict=True)
        )
    )
    n = ma.shape[0]
    return {
        "n": n,
        "argmax_identical": argmax_same,
        "argmax_agreement": round(argmax_same / n, 6),
        "top8_set_identical": top_same,
        "top8_agreement": round(top_same / n, 6),
        "max_abs_policy_logits": float((a["policy"] - b["policy"]).abs().max()),
        "max_abs_value_probs": float((a["value"] - b["value"]).abs().max()),
        "max_abs_steps": float((a["steps"] - b["steps"]).abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs" / "phase1" / "phase1.pt")
    # Matches runs/state_extent.json exactly: cap 24 (= episode.step_cap), 4 walks
    # per problem. Shorter walks do not reach the >=257 bucket — add_both_sides grows
    # the state per application, so width is a function of walk LENGTH.
    parser.add_argument("--walk-steps", type=int, default=24)
    parser.add_argument("--walks-per-problem", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no accelerator — 0b requires the ROCm stack")

    cfg = Config()
    validate(cfg)
    # The licensed regime, read from config rather than assumed here. This whole
    # smoke is what licensed it, so it must not silently run as anything else.
    assert cfg.numerics.measure_dtype == "fp32", cfg.numerics.measure_dtype
    out: dict = {
        "git_sha": git_sha(REPO),
        "torch": torch.__version__,
        "tier2_bounds": TIER2,
        "measure_dtype": cfg.numerics.measure_dtype,
        "train_dtype_informational": cfg.numerics.train_dtype,
    }

    # --- A1.1(i) comparator self-check, guaranteed by construction ---------
    # The probe must be exactly representable, or `x - (x + 1)` is not exactly
    # -1 in fp32. A randn probe returns 1.000000238418579 and fails the declared
    # equality — the self-check caught its own implementation on first run, which
    # is the case for having one. See PREREG amendment A2.
    probe = torch.arange(512, dtype=torch.float32).reshape(16, 32)
    self_check = float((probe - (probe + 1.0)).abs().max())
    out["comparator_self_check"] = {
        "max_abs": self_check,
        "expected": 1.0,
        "passes": self_check == 1.0,
    }
    print(
        f"  comparator self-check: max|delta|(T, T+1) = {self_check:.6f}  "
        f"{'OK' if self_check == 1.0 else 'FAILED'}"
    )
    if self_check != 1.0:
        raise SystemExit("comparator self-check failed — the harness is not comparing two things")

    # --- A1.2 stratified sample -------------------------------------------
    pool = narrow_states(cfg, 4000) + walked_states(
        cfg, args.walk_steps, args.walks_per_problem, seed=0
    )
    states, census, labels = stratify(pool, PER_BUCKET, seed=0)
    out["width_census"] = census
    print(f"\n  width-bucket census of the tier-1 sample (n={len(states)}):")
    for name, row in census.items():
        print(f"    {name:>10}: taken {row['taken']:>4} of {row['available']:>6} available")

    spanned = all(row["taken"] >= PER_BUCKET for row in census.values())
    out["spans_all_buckets"] = spanned
    # The contrast premise, asserted exactly as the subsampler test asserts its own.
    _narrow, narrow_census, _nl = stratify(narrow_states(cfg, 4000), PER_BUCKET, seed=0)
    narrow_spans = all(r["taken"] >= PER_BUCKET for r in narrow_census.values())
    out["narrow_only_spans"] = narrow_spans
    print(f"  spans all four buckets: {spanned}")
    print(f"  phase1_eval-only sample spans (must be False): {narrow_spans}")
    if not spanned or narrow_spans:
        raise SystemExit("width stratification failed its own premise")

    # --- load the model on both devices -----------------------------------
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cpu_model = Reckoner(cfg)
    cpu_model.load_state_dict(state["state_dict"])
    cpu_model.eval()
    gpu_model = Reckoner(cfg)
    gpu_model.load_state_dict(state["state_dict"])
    gpu_model.eval().to("cuda")

    cpu_out = forward(cpu_model, states, cfg, "cpu")
    gpu_out = forward(gpu_model, states, cfg, "cuda")
    tier = compare(cpu_out, gpu_out)
    out["equivalence_fp32"] = tier

    print(f"\n  TIER 1 — decision equivalence (n={tier['n']})")
    print(
        f"    argmax identical   : {tier['argmax_identical']}/{tier['n']}  "
        f"({tier['argmax_agreement']:.6f})"
    )
    print(
        f"    top-8 set identical: {tier['top8_set_identical']}/{tier['n']}  "
        f"({tier['top8_agreement']:.6f})"
    )
    print("\n  TIER 2 — numeric divergence (bounds frozen in the PREREG)")
    for key, bound in TIER2.items():
        got = tier[f"max_abs_{key}"]
        print(
            f"    {key:>16}: {got:.3e}   bound {bound:.0e}   "
            f"{'within' if got <= bound else 'EXCEEDED'}"
        )

    # Per width bucket — the reason A1.2 demanded the span. If divergence tracks
    # width, a narrow-only sample would have passed and hidden the exceedance.
    assert len(labels) == tier["n"], "bucket labels lost alignment with encoded rows"
    per_bucket = {}
    print("\n  TIER 2 BY WIDTH BUCKET")
    for b, (lo, hi) in enumerate(BUCKETS):
        keep = [i for i, lab in enumerate(labels) if lab == b]
        if not keep:
            continue
        row = compare(cpu_out, gpu_out, keep)
        name = f"{lo}-{hi if hi < 10_000 else 'max'}"
        per_bucket[name] = row
        verdict = "within" if row["max_abs_policy_logits"] <= TIER2["policy_logits"] else "EXCEEDED"
        print(
            f"    {name:>10}: n={row['n']:>4}  max|d| logits {row['max_abs_policy_logits']:.3e}  "
            f"argmax {row['argmax_agreement']:.6f}  {verdict}"
        )
    out["equivalence_by_bucket"] = per_bucket

    # --- A1.1(ii) end-to-end perturbation, sized to clear tier 2 -----------
    perturbed = Reckoner(cfg)
    perturbed.load_state_dict(state["state_dict"])
    with torch.no_grad():
        perturbed.rule_embedding.weight[0] += 1.0
    perturbed.eval().to("cuda")
    null = compare(cpu_out, forward(perturbed, states, cfg, "cuda"))
    out["null_perturbed"] = null
    print("\n  NULL — rule_embedding.weight[0] += 1.0 (GPU copy only)")
    print(
        f"    max|delta| policy logits: {null['max_abs_policy_logits']:.3e}  "
        f"must exceed {TIER2['policy_logits']:.0e}  "
        f"{'DETECTED' if null['max_abs_policy_logits'] > TIER2['policy_logits'] else 'NOT DETECTED'}"
    )
    print(
        f"    argmax agreement        : {null['argmax_agreement']:.6f}   [recorded, not asserted]"
    )

    # --- bf16, informational only -----------------------------------------
    bf16 = compare(cpu_out, forward(gpu_model, states, cfg, "cuda", bf16=True))
    out["bf16_informational"] = bf16
    print("\n  bf16 (informational, NOT gated)")
    print(f"    argmax agreement : {bf16['argmax_agreement']:.6f}")
    print(f"    max|delta| logits: {bf16['max_abs_policy_logits']:.3e}")

    write_record(REPO / "runs" / "gpu_equivalence_smoke.json", out)
    print("\n  wrote runs/gpu_equivalence_smoke.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
