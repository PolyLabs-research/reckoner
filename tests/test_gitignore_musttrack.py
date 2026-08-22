"""The .gitignore must never swallow a record the project depends on.

This test is inherited, not invented. Four separate incidents across two
sessions of the chess campaign, each caught by hand after the fact:

  BLOCKED.md               ignored outright, contradicting the project's own rule
  bare `data/`             ignored runs/data/ as a *directory*, so git could not
                           descend to reach the meta.json negations
  benchmarks/results.jsonl ignored, making a dated SHA-stamped record unversioned
  runs/**                  swallowed runs/ANCHORS.sha256 on the commit that added it

Whack-a-mole was losing, so this closes the class. Three lists, three questions:

  MUST_TRACK   a record that exists now and must not be ignored
  MUST_REACH   a record that does NOT exist yet and must be reachable when it
               does — `git check-ignore` answers for hypothetical paths, so the
               rule is testable the day it is written, which is the only way to
               beat "we'll remember to add the negation later" (three of the
               four incidents above)
  MUST_IGNORE  the other polarity — a well-meant negation that starts tracking
               75 MB checkpoints is the same defect facing the other way

Add to the lists whenever a new kind of record starts mattering.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from reckoner.dataset import DATA_ROOT

REPO = Path(__file__).resolve().parents[1]

# Records that exist in the working tree today. A pattern matching nothing is
# itself a failure: a record that has vanished is as bad as one that is ignored.
MUST_TRACK: list[str] = [
    "experiment2_math_base625_spec.md",
    "experiment2_agent_plan.md",
    "AGENTS.md",
    "REGISTERED-ROUNDS.md",  # rounds specified before they are run
    "FINDINGS.md",  # measured facts later chunks must not rediscover
    # BRIEF-*.md as a glob, not one line per chunk: BRIEF-chunk8.md existed only
    # in a conversation until a crash took it, and a per-file list is a list
    # somebody has to remember to extend. The law is "committed on receipt"
    # (AGENTS.md §7); the pattern is what makes forgetting fail the build.
    "BRIEF-*.md",
    "GATE-*.md",  # manual-gate verdicts — the record a chunk closed on
    "ERRATA-*.md",  # corrections to a shipped chunk report, nothing fixed silently
    "RULING-*.md",  # reviewer rulings, committed verbatim on receipt before acting
    "Makefile",
    "pyproject.toml",
    "uv.lock",  # without it, "green in a clean clone" tests resolution luck
    "configs/*.yaml",
    "docs/*.md",  # generated reference (docs/vocab.md) — regenerable, but cited
    "src/reckoner/*.py",
    "tests/*.py",
    "tests/subprocess_probes/*.py",  # probes are files, not embedded strings
    "scripts/*.py",
    "runs/state_extent.json",  # decision-bearing: it sized seq_len and max_sites
]

# The chunk 0 list, verbatim from the plan, expanded to concrete example paths
# because `git check-ignore` answers about pathnames, not patterns. The pattern
# each one stands for is in the trailing comment.
MUST_REACH: list[str] = [
    "BLOCKED-2026-08-14-example.md",  # BLOCKED*.md — a halt is information
    "RUNLOG-m1.md",  # RUNLOG*.md — the campaign ledger
    "PREREG-m1.md",  # PREREG*.md — pre-registration, with its amendment header
    "runs/m1/iterations.jsonl",  # runs/*/iterations.jsonl — the loop's narrative
    "runs/m1/ladder.jsonl",  # runs/*/ladder.jsonl — the verdict
    "benchmarks/results.jsonl",  # dated, SHA-stamped bench record
    "runs/ANCHORS.sha256",  # dataset + suite digests
    # The ladder's instruments. Written the day the rule was, not the day the
    # first pass runs — `git check-ignore` answers for hypothetical paths, and
    # `pairedset.freeze` asserts its own trackedness before the bytes land.
    "runs/paired/smoke_v1.jsonl",  # runs/paired/*.jsonl — the frozen paired set
    "runs/testcount.json",  # written by the run itself, never typed (F-16)
    "runs/chunk11_part0_result.json",  # runs/chunk11_*.json — the headroom measurement
    "runs/chunk11_ceiling.json",  # the ceiling arithmetic derived from it
    "runs/paired_census.json",  # runs/paired_census*.json — the two-level censuses
    "runs/paired_census_postfreeze.json",  # the re-census proving it was born clean
    "runs/ladder_smoke_result.json",  # runs/**/*_result.json — the smoke verdicts
    # Beyond the plan's initial list, written now for the same reason: these are
    # named as deliverables by later chunks, and the negation is free today.
    # NAMED A PATH NOTHING WRITES. `ladder_pairscores.jsonl` has never been
    # emitted by any run; the code writes `ladder/<leg>/pass-NNNN/pair_scores.jsonl`.
    # This entry passed since the day it was written because MUST_REACH asserts a
    # hypothetical path WOULD be trackable, never that anything produces it — a
    # trigger with no arm that can fire. Kept, with the real class beside it, and
    # `test_emitted_records_are_trackable` now asserts the property over paths the
    # code ACTUALLY emits, which is the check a remembered name cannot pass.
    "runs/m1/ladder_pairscores.jsonl",  # historical: never written by any run
    "runs/m1/ladder/primary/pass-0000/pair_scores.jsonl",  # the real per-problem rows
    "runs/m1/ladder/primary/pass-0000/PASS-DONE",  # and the completion marker
    "runs/m1/instruments.jsonl",  # the cadence unit — durable since F-26
    "runs/m1/config.yaml",  # the RESOLVED config a run's rows are read against
    "runs/m1/provenance.json",  # git_sha + config fingerprint
    f"{DATA_ROOT}/train_100k/meta.json",  # dataset provenance sidecar
    "runs/m1/annotations.jsonl",  # why a timing row is anomalous
    # Chunk 5: the frozen instruments and the dataset provenance sidecars. The
    # suites ARE the measuring stick — a suite git silently dropped would make
    # every number measured against it unreproducible.
    "runs/suites/solve_in_3.jsonl",
    f"{DATA_ROOT}/train_100k/meta.json",
    f"{DATA_ROOT}/eval_held_out/meta.json",
    "runs/rule_participation.json",
    "runs/par_delta.json",
    # Chunks 7-8: gate records. These three EXISTED and were silently ignored —
    # the guard missed them because nobody wrote the rule (ERRATA-chunk7.md §5).
    # Listed as MUST_REACH rather than MUST_TRACK so the glob is tested for
    # names that do not exist yet, which is the point of this list.
    "runs/gate_arithmetic.json",
    "runs/gate_arithmetic_d2.json",
    "runs/chunk7_gate_table.json",
    "runs/gate_a_future_instrument.json",
    # Chunk 8: pre-flight projections, kept so they can be scored against the run
    # they projected. F-03 exists because one was not.
    "runs/pilot_phase1_timing.json",
    "runs/pilot_a_future_preflight.json",
    # A1: contamination censuses are standing evidence, not one-offs.
    "runs/supervision_contamination.json",
    "runs/eval_suite_contamination.json",
    "runs/eval_independence.json",
    "runs/a_future_set_contamination.json",
    # Chunk 8: what a run reported, and the environment it reported it from.
    "runs/phase1/phase1_result.json",
    "runs/m1/check_env.txt",
    "runs/phase1/anchor.json",  # the anchor every later pass is compared against
    "runs/m1/value_switch.jsonl",  # switch events, abstentions included
    "runs/m1/LATEST",  # the resume commit point
    "runs/m1/state-3.json",  # per-iteration resume state
    "runs/shakedown_result.json",  # the shakedown's verdicts survive the run dir
    "runs/gpu_equivalence_smoke.json",  # chunk 9 part 0b
    "runs/gpu_equivalence_reference.json",  # tier-2 drift reference (A3)
]

# The counterpart polarity: things that must STAY ignored.
MUST_IGNORE: list[str] = [
    "runs/m1/latest.pt",
    "runs/m1/snapshots/iter0005.pt",
    f"{DATA_ROOT}/train_100k/states.npy",
    f"{DATA_ROOT}/train_100k/tokens.i32",  # 100K x max_len int32 — build output, not a record
    ".venv/bin/python",
    "src/reckoner/__pycache__/config.cpython-312.pyc",
]


def _is_git_repo() -> bool:
    return (REPO / ".git").exists()


def _check_ignored(path: str) -> bool:
    """True if git ignores ``path``. Uses check-ignore's exit status (0 = ignored).

    ``git -C`` addresses the repo per-command: a ``cd`` persists for a whole
    shell invocation and has already mis-pointed three merge commits on this box.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", path],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _is_git_repo(), reason="not a git repository — nothing to check"
)


@pytest.mark.parametrize("pattern", MUST_TRACK)
def test_must_track_paths_are_not_ignored(pattern: str) -> None:
    matches = sorted(REPO.glob(pattern))
    assert matches, f"{pattern!r} matched no file — the record it names is missing"

    ignored = [
        str(p.relative_to(REPO)) for p in matches if _check_ignored(str(p.relative_to(REPO)))
    ]
    assert not ignored, (
        f"{pattern!r} is gitignored: {ignored}\n"
        "These are project records, not build output. Add a negation to .gitignore."
    )


@pytest.mark.parametrize("path", MUST_REACH)
def test_future_records_are_already_reachable(path: str) -> None:
    """The negation must be in place before the run that produces the record."""
    assert not _check_ignored(path), (
        f"{path!r} would be gitignored. A run that writes it would produce a record "
        "git silently drops — add a negation to .gitignore."
    )


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_must_ignore_paths_stay_ignored(path: str) -> None:
    """Guard the other direction: negations must not start tracking large blobs."""
    assert _check_ignored(path), (
        f"{path!r} is NOT ignored — a .gitignore negation has gone too wide. "
        "Checkpoints, datasets and the venv must never enter the repository."
    )


def test_the_guard_itself_detects_an_ignored_record() -> None:
    """Validate the detector on both polarities before trusting it.

    Every other test here asserts `_check_ignored` returns the answer we want.
    None of them would fail if `_check_ignored` were stuck returning False —
    which is exactly the shape of a guard that reports green forever. This pins
    both answers against paths whose status is not in question.
    """
    assert _check_ignored(".venv/lib/python3.12/site-packages/torch/__init__.py")
    assert not _check_ignored("pyproject.toml")


# --------------------------------------------- the paths the code ACTUALLY emits


#: Record classes whose real emitted paths must be trackable. Keyed by the class,
#: valued by a glob against a completed run — these are the names the CODE writes,
#: not the names a contract remembers.
EMITTED_RECORDS: dict[str, str] = {
    "per-problem ladder rows": "ladder/*/pass-*/pair_scores.jsonl",
    "pass completion markers": "ladder/*/pass-*/PASS-DONE",
    "cadence instruments": "instruments.jsonl",
    "iteration rows": "iterations.jsonl",
    "switch events": "value_switch.jsonl",
    "the commit marker": "LATEST",
    "resume state": "state-*.json",
    # --- classes a HEALTHY run never emits -----------------------------------
    # Listed deliberately. A run that completes cleanly produces none of these,
    # so a sample drawn from a healthy run cannot exercise them — and they are
    # precisely the records whose evidence matters most when something goes
    # wrong. Reported as UNCOVERED rather than passing silently, because a green
    # result over a sample that never contained the class is a green result about
    # nothing.
    "retained rings": "ring-*/meta.json",
    "the campaign summary": "campaign_summary.json",
    "checkpoint watch verdicts": "checkpoint_watch.json",
    "the run manifest": "manifest.json",
    # --- classes a healthy run genuinely CANNOT emit --------------------------
    # These have no instance in any clean run by construction, so the sample can
    # never exercise them and they must report UNCOVERED. Listing them is the
    # point: an unlisted class is invisible, a listed-and-uncovered one is a
    # named gap. First labelling of this block was wrong — `ring-*/meta.json` was
    # called post-kill debris and matched `ring-19`, a normal artifact, which made
    # the report read "UNCOVERED: none" while covering nothing abnormal at all.
    "orphan ring beyond LATEST (post-kill debris)": "ring-99/meta.json",
    "halt record": "BLOCKED-*.md",
    "torn-tail repair evidence": "*.repaired",
    "alarm census output": "alarms.json",
}


def test_emitted_records_are_trackable() -> None:
    """**The property, over the paths the code emits — not the names a list holds.**

    The contract named `runs/*/ladder_pairscores.jsonl`, which **nothing has ever
    written**. The code emits `ladder/<leg>/pass-NNNN/pair_scores.jsonl`. No
    negation matched it, so git silently dropped the artifact that makes P1's
    paired-difference bootstrap computable — the second time in this campaign the
    recording layer nearly ate the primary's evidence base (F-30 was the first).

    A per-name negation is the third patch in that family and a fourth path will
    diverge from a fifth pattern eventually. This asserts the property instead:
    **every path a completed run actually contains, for a must-track record
    class, must be reachable by git.** The filename mismatch could not have
    survived it, because it never consults a remembered name.

    Runs against the M1 artifacts on disk; skipped where they are absent, since
    the property needs real emitted paths rather than hypothetical ones — that
    hypothetical-only check is exactly what failed.
    """
    run = REPO / "runs" / "m1_final" / "m1_attempt2"
    if not run.exists():
        pytest.skip("no completed run on disk to enumerate")

    ignored: list[str] = []
    covered: dict[str, int] = {}
    for label, pattern in EMITTED_RECORDS.items():
        hits = sorted(run.glob(pattern))
        covered[label] = len(hits)
        for path in hits:
            rel = str(path.relative_to(REPO))
            if _check_ignored(rel):
                ignored.append(f"{label}: {rel}")

    uncovered = sorted(label for label, n in covered.items() if n == 0)
    (REPO / "runs" / "record_coverage.json").write_text(
        json.dumps(
            {
                "run": str(run.relative_to(REPO)),
                "covered": covered,
                "uncovered": uncovered,
                "ignored": ignored,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    assert sum(covered.values()), "the glob set matched nothing — the classes have moved"
    assert not ignored, (
        "paths a completed run actually emitted are gitignored:\n  "
        + "\n  ".join(ignored)
        + "\nA record git drops is a record that never existed."
    )


def test_the_trackability_property_reports_what_it_could_not_cover() -> None:
    """**Coverage is the coverage of the sample.** L2's business.

    `test_emitted_records_are_trackable` asserts over paths a completed run
    actually contains — the right subject — but a healthy run emits none of the
    abnormal-exit classes: post-kill debris, halt records, the BLOCKED path. Those
    would pass green **by never appearing**, and they are exactly the records
    whose evidence matters most when something goes wrong.

    So the uncovered set is written to `runs/record_coverage.json` and asserted
    to exist, making "73 green" mean something specific rather than something
    unbounded."""
    coverage = REPO / "runs" / "record_coverage.json"
    if not coverage.exists():
        pytest.skip("trackability property has not run (no completed run on disk)")
    report = json.loads(coverage.read_text())
    assert "uncovered" in report and "covered" in report, (
        "the coverage report does not name what it failed to reach, which is the "
        "only part that bounds the claim"
    )
