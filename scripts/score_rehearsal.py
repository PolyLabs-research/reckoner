"""Score the M1 dress rehearsal against `REHEARSAL-m1.md`, committed at `0d31989`.

**The predictions are transcribed here, not re-derived here.** Every check below
restates a line from the expectations page written before the run; where a
number appears it is the number that page committed to. A scorer that recomputed
its own expectations from the artifacts would agree with them by construction,
which is the whole failure pre-registration exists to prevent.

Reports on whatever is present, so it is useful mid-run. The overall verdict is
withheld until the run is complete — a partial run has not passed, and must not
be able to read as though it had.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reckoner.campaign import ANCHOR

# This script scores M1's rehearsal rows, so it checks them against M1's pair,
# not against whatever the campaign door pins today (F-42 moved that).
from reckoner.campaign import M1_CAMPAIGN_FINGERPRINT as CAMPAIGN_FINGERPRINT
from reckoner.campaign import M1_EVAL_FINGERPRINT as EVAL_FINGERPRINT
from reckoner.dataset import sha256_file
from reckoner.logschema import ITERATION_FIELDS, alarm_census, read_rows

REPO = Path(__file__).resolve().parents[1]

ITERATIONS = 5
CADENCE_AT = 4
FUNNEL = (
    "entropy_prior_step1_start",
    "entropy_prior_step1_reached",
    "entropy_target_step1_start",
    "entropy_target_step1_reached",
)


def rows_of(run: Path, name: str) -> list[dict]:
    path = run / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run: Path = args.run_dir

    rows = rows_of(run, "iterations.jsonl")
    switch = rows_of(run, "value_switch.jsonl")
    instruments = rows_of(run, "instruments.jsonl")
    latest = int((run / "LATEST").read_text().strip()) if (run / "LATEST").exists() else None
    complete = latest == ITERATIONS - 1

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # --- §3 artifacts ------------------------------------------------------
    check("LATEST == 4", latest == ITERATIONS - 1, f"LATEST={latest}")
    check(
        "iterations.jsonl carries 5 rows, 0..4",
        [r["iteration"] for r in rows] == list(range(ITERATIONS)),
        str([r["iteration"] for r in rows]),
    )
    check(
        "value_switch.jsonl carries 5 rows, 0..4",
        [r["iteration"] for r in switch] == list(range(ITERATIONS)),
        str([r["iteration"] for r in switch]),
    )
    check(
        "instruments.jsonl carries exactly 1 row, at iteration 4",
        [r["iteration"] for r in instruments] == [CADENCE_AT],
        str([r["iteration"] for r in instruments]),
    )
    for stem in ("ckpt", "snap"):
        found = sorted(p.name for p in run.glob(f"{stem}-*.pt"))
        check(f"{stem}-0..4.pt present", len(found) == ITERATIONS, f"{len(found)}: {found}")
    check(
        "state-4.json and ring-4 present",
        (run / "state-4.json").exists() and (run / "ring-4").is_dir(),
    )
    check("_preflight scratch removed itself", not (run / "_preflight").exists())

    # --- §4 every iteration row -------------------------------------------
    check("schema_era == 3 in every row", all(r["schema_era"] == 3 for r in rows))
    check(
        "config_fingerprint is the campaign's, in every row",
        {r["config_fingerprint"] for r in rows} == {CAMPAIGN_FINGERPRINT},
        str({r["config_fingerprint"][:12] for r in rows}),
    )
    sizes = [r["pool_composition"]["size"] for r in rows]
    check(
        "pool_composition size == n + 1",
        sizes == [n + 1 for n in range(len(rows))],
        f"{sizes} against {[n + 1 for n in range(len(rows))]}",
    )
    if len(rows) > CADENCE_AT:
        pc = rows[CADENCE_AT]["pool_composition"]
        check(
            "at iteration 4, order == [5000, 0, 1, 2, 3]",
            pc["order"] == [5000, 0, 1, 2, 3],
            str(pc["order"]),
        )
        check(
            "at iteration 4, steps == [0, 1, 2, 3, 5000]",
            pc["steps"] == [0, 1, 2, 3, 5000],
            str(pc["steps"]),
        )
        check(
            "and the two views DISAGREE — M1-A3 §3 observed",
            pc["order"] != pc["steps"],
            "identical views would mean the order column carries nothing",
        )
    check(
        "ladder_pass absent with its reason for iterations 0-3",
        all("ladder_pass" in (r.get("absent") or {}) for r in rows[:CADENCE_AT]),
        str([(r.get("absent") or {}).get("ladder_pass") for r in rows[:1]]),
    )
    if len(rows) > CADENCE_AT:
        check(
            "ladder_pass == 0 at iteration 4",
            rows[CADENCE_AT].get("ladder_pass") == 0,
            str(rows[CADENCE_AT].get("ladder_pass")),
        )
    fracs = [r.get("pool_par_fraction") for r in rows]
    check(
        "pool_par_fraction present in every row and <= 0.2",
        all(f is not None and f <= 0.2 for f in fracs),
        str(fracs),
    )
    check(
        "all four funnel columns present in every row",
        all(all(c in r for c in FUNNEL) for r in rows),
    )
    if rows:
        check(
            "iteration 0's evaluator digest is the anchor",
            rows[0]["evaluator_checkpoint_sha256"] == sha256_file(ANCHOR),
            rows[0]["evaluator_checkpoint_sha256"][:12],
        )
    digests = {r["evaluator_checkpoint_sha256"] for r in rows}
    check(
        "and the digest MOVES across the run",
        len(digests) >= 2,
        f"{len(digests)} distinct across {len(rows)} rows",
    )
    alarms = alarm_census(read_rows(run / "iterations.jsonl", ITERATION_FIELDS)) if rows else {}
    check("alarms == 0", not alarms, str(alarms))

    # --- §5 the switch log -------------------------------------------------
    check(
        "every switch row carries its outcome fields, abstentions included",
        all(
            all(k in r for k in ("abstained", "fired", "already_live", "clears", "n"))
            for r in switch
        ),
        f"{sum(bool(r.get('abstained')) for r in switch)} abstained of {len(switch)}",
    )

    # --- §6 the cadence unit ----------------------------------------------
    if instruments:
        inst = instruments[0]
        check(
            "the cadence ran the EVAL profile",
            inst.get("config_fingerprint") == EVAL_FINGERPRINT and inst.get("profile") == "eval",
            f"{str(inst.get('config_fingerprint'))[:12]} / {inst.get('profile')}",
        )
        for sims, floor in ((48, 1188), (1, 1167)):
            nr = inst.get(f"no_regress_sims_{sims}", {})
            check(
                f"no_regress @{sims}: of == 1200, floor == {floor}, recomputed == {floor}",
                nr.get("of") == 1200
                and nr.get("floor") == floor
                and nr.get("floor_recomputed") == floor,
                json.dumps(nr),
            )
        primary = inst.get("primary", {})
        check(
            "primary pooled over 600 against anchor_baseline 101/600",
            primary.get("anchor_baseline") == "101/600"
            and primary.get("pooled_beat_par", "").endswith("/600"),
            json.dumps({k: v for k, v in primary.items() if k != "per_stratum"}),
        )
        check(
            "per-stratum breakdown for 7, 8, 10",
            set(primary.get("per_stratum", {}))
            == {"scripted_in_7", "scripted_in_8", "scripted_in_10"},
            str(sorted(primary.get("per_stratum", {}))),
        )

    print(f"\n  M1 DRESS REHEARSAL — {run}\n")
    for name, ok, detail in checks:
        print(f"    {'ok  ' if ok else 'FAIL'} {name}{'  — ' + detail if detail else ''}")

    # --- recorded, NOT gating (§7) ----------------------------------------
    if instruments:
        inst = instruments[0]
        print("\n  RECORDED, NOT GATING — five iterations of twenty is not the campaign:\n")
        for sims in (48, 1):
            nr = inst.get(f"no_regress_sims_{sims}", {})
            held = "HELD" if nr.get("held") else "**BREACHED**"
            print(
                f"    no-regress @{sims:>2}: {nr.get('at_par')}/{nr.get('of')} "
                f"against floor {nr.get('floor')} — {held}"
            )
        primary = inst.get("primary", {})
        print(
            f"    primary      : {primary.get('pooled_beat_par')} "
            f"(anchor {primary.get('anchor_baseline')}), delta {primary.get('delta_vs_anchor')}"
        )
        for stratum, cell in sorted(primary.get("per_stratum", {}).items()):
            print(f"      {stratum:>16}: {cell.get('beat')}/{cell.get('of')}")
        if any(not inst.get(f"no_regress_sims_{s}", {}).get("held", True) for s in (48, 1)):
            print("\n    A BREACHED FLOOR IS CARRIED TO A RULING BEFORE M1 LAUNCHES (§7).")

    failed = [n for n, ok, _ in checks if not ok]
    if not complete:
        print(
            f"\n  INCOMPLETE — LATEST={latest}, {len(rows)} of {ITERATIONS} iterations. "
            "No verdict: a partial run has not passed."
        )
        return 2
    print(
        f"\n  {'PASS' if not failed else 'FAIL'} — {len(checks) - len(failed)}/{len(checks)} checks"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
