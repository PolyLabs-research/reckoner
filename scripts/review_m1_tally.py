"""REVIEW-2026-11-15's third probe: what M1's own tracked rows say about headroom.

Pure reader of committed artifacts, no model: how M1's self-play episodes ended,
whether its value switch ever fired, and how many of the 600 deep scripted
problems the anchor and M1's final checkpoint solve at all (``steps < step_cap``).

    python scripts/review_m1_tally.py --out runs/review_20261115/m1_tally_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reckoner.config import Config

REPO = Path(__file__).resolve().parents[1]
M1 = REPO / "runs" / "pod_pull_20260822" / "m1_attempt2"
ANCHOR_PASS = REPO / "runs" / "part0d_paired" / "pass-0000" / "pair_scores.jsonl"
FINAL_PASS = M1 / "ladder" / "primary" / "pass-0019" / "pair_scores.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cap = Config().episode.step_cap
    outcomes = {"+1": 0, "0": 0, "-1": 0}
    for row in _rows(M1 / "iterations.jsonl"):
        for cell in row["z_by_par_source"].values():
            for key, count in cell.items():
                outcomes[key] += count
    switch = _rows(M1 / "value_switch.jsonl")

    def solved(path: Path) -> dict:
        rows = _rows(path)
        return {
            "problems": len(rows),
            "solved": sum(r["steps"] < cap for r in rows),
            "mean_steps": round(sum(r["steps"] for r in rows) / len(rows), 2),
        }

    report = {
        "m1_self_play_episodes": sum(outcomes.values()),
        "m1_self_play_outcomes": outcomes,
        "m1_value_switch_rows": len(switch),
        "m1_value_switch_fired": sum(bool(r["fired"]) for r in switch),
        "m1_value_switch_abstained": sum(bool(r["abstained"]) for r in switch),
        "deep_scripted_anchor": solved(ANCHOR_PASS),
        "deep_scripted_m1_final": solved(FINAL_PASS),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
