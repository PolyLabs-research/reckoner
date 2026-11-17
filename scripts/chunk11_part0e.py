"""Part 0e: the uniform-stub null on the successor strata {7, 8, 10}.

The four-tuple's last empty cell. Part-0d measured what the **anchor** does on the
scripted strata and certified three of them live; P1 will be stated as a
CI-separated improvement in pooled beat-par rate over that baseline. A baseline
without a null is a number with nothing underneath it: 101/600 = 0.1683 means one
thing if a policy-free search beats par at 0.02, and something very different if
it beats par at 0.15.

**Rider (c) does not exempt successors.** The null row is owed here exactly as it
was owed on the original axis, and nobody had computed it.

**The protocol is Part-0d's, inherited rather than restated** — sims 48, m 16,
root noise off, seed 0, the eval profile, value-silent — because the baseline's
protocol is the primary's protocol, and a null measured under a different one is
not a null for this baseline. The single deliberate difference is the evaluator:
:func:`reckoner.search.uniform_stub` in place of the model, which is what makes
this a null rather than a second measurement of the anchor.

`scripted_in_9` is excluded: Part-0d found it the sole stratum born saturated
from above (beat-par 164/200 = 0.82) and demoted it to informational.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from reckoner.config import Config, config_fingerprint, validate, with_m1_search
from reckoner.dataset import git_sha, read_suite, suite_problem, write_record
from reckoner.runner import run_iteration
from reckoner.search import uniform_stub

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
PART0D = REPO / "runs" / "chunk11_part0d_scripted_strata.json"

#: The successor set. Part-0d's ruling, 2026-08-16 — 9 demotes to informational.
STRATA = (7, 8, 10)

#: Inherited from Part-0d verbatim. Stated as constants so a reader can diff them
#: against that record rather than trusting a sentence that says "the same".
SIMS = 48
GUMBEL_M = 16
SEED = 0


def anchor_baseline() -> dict:
    """Part-0d's measured anchor outcomes on the successor strata."""
    if not PART0D.exists():
        return {}
    record = json.loads(PART0D.read_text())
    out = {}
    for k in STRATA:
        cell = record["per_stratum"][f"scripted_in_{k}"]
        out[f"scripted_in_{k}"] = {
            "beat": cell["beat_par_z_plus_1"],
            "problems": cell["problems"],
            "beat_par_rate": cell["beat_par_rate"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    # The stub reports no steps-to-go, so it plays the search that reads 'wdl'
    # (the default until 2026-11-16).
    cfg = with_m1_search(Config())
    validate(cfg)
    cfg = replace(cfg, search=replace(cfg.search, root_noise=False))
    evaluator = uniform_stub(cfg)

    started = time.perf_counter()
    per_stratum: dict[str, dict] = {}
    pooled_beat = pooled_n = 0

    for k in STRATA:
        problems = [suite_problem(r) for r in read_suite(SUITES / f"scripted_in_{k}.jsonl")]
        stats = run_iteration(problems, evaluator, cfg, None, sims=SIMS, m=GUMBEL_M, seed=SEED)
        stats.check_descent_identity()
        bins = dict(stats.steps_minus_par)
        n = stats.episodes
        beat = bins["<0"]
        at = bins["0"]
        over = sum(v for key, v in bins.items() if key not in ("<0", "0"))
        # Same accounting as Part-0d: unsolved episodes are z = -1 and sit outside
        # the solved histogram, so they are added rather than assumed away.
        unsolved = n - stats.episodes_solved
        mean_z = (beat - (over + unsolved)) / n
        per_stratum[f"scripted_in_{k}"] = {
            "par": k,
            "par_source": "scripted",
            "problems": n,
            "solved": stats.episodes_solved,
            "capped": stats.episodes_capped,
            "stuck": stats.episodes_stuck,
            "beat_par_z_plus_1": beat,
            "at_par_z_0": at,
            "over_par_or_unsolved_z_minus_1": over + unsolved,
            "beat_par_rate": round(beat / n, 6),
            "mean_z": round(mean_z, 6),
            "steps_minus_par": bins,
            "seconds": round(stats.seconds, 2),
        }
        pooled_beat += beat
        pooled_n += n
        print(
            f"    scripted_in_{k}: beat {beat:>3} at {at:>3} over/unsolved "
            f"{over + unsolved:>3} of {n}  mean z {mean_z:+.4f}  {stats.seconds:>7.1f}s",
            flush=True,
        )

    baseline = anchor_baseline()
    anchor_beat = sum(v["beat"] for v in baseline.values()) if baseline else None
    anchor_n = sum(v["problems"] for v in baseline.values()) if baseline else None

    comparison = {
        "null_pooled_beat_par": f"{pooled_beat}/{pooled_n}",
        "null_pooled_rate": round(pooled_beat / pooled_n, 6) if pooled_n else None,
        "anchor_pooled_beat_par": f"{anchor_beat}/{anchor_n}" if baseline else None,
        "anchor_pooled_rate": round(anchor_beat / anchor_n, 6) if baseline else None,
        "anchor_minus_null": (
            round(anchor_beat / anchor_n - pooled_beat / pooled_n, 6) if baseline else None
        ),
        "note": "the null row P1's baseline is read against; no CI is computed "
        "here — the paired-difference bootstrap belongs to the primary",
    }

    report = {
        "purpose": "the four-tuple's null row for the successor axis; "
        "rider (c) does not exempt successors",
        "git_sha": git_sha(REPO),
        "protocol": {
            "evaluator": "reckoner.search.uniform_stub (flat priors, neutral value)",
            "inherited_from": "runs/chunk11_part0d_scripted_strata.json",
            "sims": SIMS,
            "gumbel_m": GUMBEL_M,
            "root_noise": cfg.search.root_noise,
            "step_cap": cfg.episode.step_cap,
            "measure_dtype": cfg.numerics.measure_dtype,
            "seed": SEED,
            "config_fingerprint": config_fingerprint(cfg),
            "device": "cpu",
        },
        "strata": [f"scripted_in_{k}" for k in STRATA],
        "excluded": {
            "scripted_in_9": "born saturated from above in Part-0d "
            "(beat-par 164/200 = 0.82); informational only"
        },
        "per_stratum": per_stratum,
        "anchor_baseline_from_part0d": baseline,
        "comparison": comparison,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }
    write_record(REPO / "runs" / "chunk11_part0e_null.json", report)

    print("\n  NULL vs ANCHOR (pooled over {7, 8, 10})\n")
    print(json.dumps(comparison, indent=2))
    print(f"\n  wall clock {report['wall_clock_seconds']}s")
    print("  wrote runs/chunk11_part0e_null.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
