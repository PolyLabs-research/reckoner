"""M1's launcher. **The campaign, at the registered config, through its own door.**

There is exactly one behaviour here that is not `reckoner.campaign.run_campaign`:
choosing where the run directory lives. A path is not a behaviour — it names
where artifacts land, not what the loop does — and everything that *is* a
behaviour (extent, treatment size, thread pins, cadence) lives in
`CampaignConfig` under the fingerprint, which `run_campaign` asserts before it
spends anything.

So this script takes no `--iterations`, no `--episodes`, no `--sims`. If a number
here could change what M1 measures, it would be config that escaped the
fingerprint, which is the caller-choice hazard standing at the command line.

**The five-iteration dress rehearsal uses this same script.** It is not a
rehearsal mode — `campaign.iterations = 20` is under the fingerprint and setting
it to 5 moves the fingerprint to `7ba86c45…`, which this door refuses. The
rehearsal is the campaign's first five iterations, stopped externally once
`LATEST` reads 4, which the resume gate licenses: iteration 5's provisional
artifacts are exactly the debris `test_campaign_resume.py` kills into and
resumes from at both boundaries.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from reckoner.campaign import ANCHOR, CAMPAIGN_FINGERPRINT, run_campaign
from reckoner.config import Config, validate
from reckoner.dataset import git_sha
from reckoner.resume import check_resumable

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="where artifacts land")
    args = parser.parse_args()

    cfg = Config()
    validate(cfg)
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "campaign_summary.json"

    # Refuse BEFORE the first write below. The identity stub overwrites
    # campaign_summary.json, so a refusal that arrived later — inside
    # run_campaign's resume — would leave a completed run's summary replaced by
    # a "started" stub for a run that never started (F-41 review). Read-only.
    check_resumable(run_dir)

    # THE PILOT LAW, ON THIS SCRIPT'S OWN OUTPUT PATH (D-A1 §3). The loop's
    # pre-flight smokes the ROW classes; it cannot smoke the summary file this
    # script writes, because that file is this script's. Written now, at zero
    # cost, so a broken write path raises in a second rather than after hours.
    # The driver's own pre-flight covers everything downstream of here.
    identity = {
        "status": "started",
        "git_sha": git_sha(REPO),
        "config_fingerprint": CAMPAIGN_FINGERPRINT,
        "anchor": str(ANCHOR),
        "started_unix": time.time(),
    }
    summary_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    print(f"\n  M1 — {run_dir}")
    print(f"  git {identity['git_sha'][:12]}  config {CAMPAIGN_FINGERPRINT[:12]}\n")

    started = time.perf_counter()
    summary = run_campaign(run_dir, cfg, anchor=ANCHOR)
    elapsed = time.perf_counter() - started

    summary_path.write_text(
        json.dumps(
            identity | {"status": "complete", "seconds": round(elapsed, 1)} | summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\n  committed through iteration {summary['committed']} in {elapsed / 3600:.2f} h")
    print(f"  alarms: {summary['alarms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
