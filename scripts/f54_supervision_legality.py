"""F-54's box-side reading: is every Phase-1 supervised target a legal action?

Since F-53, ``policy_loss`` raises ``IllegalTargetError`` where it used to drop
target mass under masked-off actions. Rehearsal (``train.rehearsal_frac > 0``)
feeds ``phase1_train`` rows through that same loss, and only 64 of its rows had
ever been checked against the legality oracle. A raise mid-campaign would be an
old dataset defect surfacing at the worst time, so this script settles it first:
it walks EVERY row of a supervision set through the same decode -> ``encode``
path ``make_batch`` uses and tests the stored action against the derived
``legal_mask``.

It reports what it covered, not only that it passed: rows read, rows that fail
to encode (``StateTooLarge`` — ``make_batch`` counts those and never crops), and
illegal targets, each by depth. Exits non-zero if any target is illegal.

    python scripts/f54_supervision_legality.py --data phase1_train --out /tmp/legality.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import anchored_data
from reckoner.episode import Problem, decode_state
from reckoner.model import StateTooLarge, encode
from reckoner.train import SupervisionSet

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="phase1_train")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    data = SupervisionSet(anchored_data(args.data, verify=True))
    n = len(data)
    rows_by_depth: Counter[int] = Counter()
    too_large_by_depth: Counter[int] = Counter()
    illegal_by_depth: Counter[int] = Counter()
    out_of_range = 0
    illegal_rows: list[int] = []
    n_actions = None
    started = time.monotonic()
    for i in range(n):
        depth = int(data.depth[i])
        rows_by_depth[depth] += 1
        seq = tuple(int(t) for t in data.tokens[i, : data.lengths[i]])
        goal, target, expr = decode_state(seq)
        problem = Problem(goal=goal, expr=expr, par=depth, target=target, par_source="bfs")
        try:
            state = encode(problem, expr, cfg)
        except StateTooLarge:
            too_large_by_depth[depth] += 1
            continue
        n_actions = int(state.legal_mask.shape[0])
        action = int(data.action[i])
        if not 0 <= action < n_actions:
            out_of_range += 1
            illegal_by_depth[depth] += 1
            illegal_rows.append(i)
        elif not bool(state.legal_mask[action]):
            illegal_by_depth[depth] += 1
            illegal_rows.append(i)

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    illegal = sum(illegal_by_depth.values())
    too_large = sum(too_large_by_depth.values())
    report = {
        "data": args.data,
        "head": head,
        "rows": n,
        "rows_checked": n - too_large,
        "rows_too_large": too_large,
        "illegal_targets": illegal,
        "illegal_out_of_range": out_of_range,
        "illegal_rows_first_50": illegal_rows[:50],
        "action_space": n_actions,
        "by_depth": {
            str(d): {
                "rows": rows_by_depth[d],
                "too_large": too_large_by_depth[d],
                "illegal": illegal_by_depth[d],
            }
            for d in sorted(rows_by_depth)
        },
        "seconds": round(time.monotonic() - started, 1),
        "verdict": "ALL_LEGAL" if illegal == 0 else "ILLEGAL_TARGETS",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0 if illegal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
