"""Crash-resume through both kill points, asserted against an uninterrupted run.

The claim is not "resume runs" — it is that a run killed at either point lands on
state **identical** to one that was never killed. Anything weaker permits a
duplicated replay corpus, which shows up as a mildly odd loss curve six
iterations later and is attributed to everything except the crash.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config
from reckoner.replay import ReplayRing
from reckoner.resume import (
    KEEP_RINGS,
    ResumeError,
    RunState,
    commit_iteration,
    latest_committed,
    prune_rings,
    resume,
)
from reckoner.valuegate import ValueHeadState

CFG = Config()


def a_record(par: int) -> dict:
    return {
        "tokens": np.array([1, 2, 3], dtype=np.int16),
        "site_positions": np.array([0, 1], dtype=np.int16),
        "visit_actions": np.array([5, 9], dtype=np.int32),
        "visit_counts": np.array([12, 4], dtype=np.int32),
        "root_q": 0.0,
        "z": 0,
        "par_source": "bfs",
        "par": par,
        "steps_remaining": 1,
        "depth": par,
        "goal": 16,
    }


class Harness:
    """A miniature loop: three iterations, each adding two ring rows and a log row."""

    def __init__(self, run: Path, *, kill_at: tuple[int, str] | None = None) -> None:
        self.run = run
        self.run.mkdir(parents=True, exist_ok=True)
        self.kill_at = kill_at

    def play(self, iterations: int = 3) -> None:
        start, ring, state = resume(self.run, CFG)
        if ring is None:
            ring = ReplayRing(64, CFG)
            state = RunState(iteration=0, value_head=ValueHeadState(), seed=7)

        for n in range(start, iterations):
            ring.append(**a_record(par=n + 1))
            ring.append(**a_record(par=n + 1))
            state = RunState(iteration=n, value_head=state.value_head, seed=state.seed)

            def write_row(n: int = n) -> None:
                with (self.run / "iterations.jsonl").open("a") as fh:
                    fh.write(json.dumps({"iteration": n}) + "\n")
                if self.kill_at == (n, "B"):
                    raise KeyboardInterrupt("killed after the row, before LATEST")

            if self.kill_at == (n, "A"):
                # Kill point A: ring and state on disk, row not yet written.
                ring.save(self.run / f"ring-{n}")
                (self.run / f"state-{n}.json").write_text(
                    json.dumps(state.as_dict(), indent=2, sort_keys=True) + "\n"
                )
                raise KeyboardInterrupt("killed after ring and state, before the row")

            commit_iteration(self.run, ring, state, write_row)


def final_state_partial(run: Path) -> tuple[int | None, list[str], list[str]]:
    """Everything on disk, for idempotence comparison — including nothing."""
    rows = (
        (run / "iterations.jsonl").read_text().splitlines()
        if (run / "iterations.jsonl").exists()
        else []
    )
    return (
        latest_committed(run),
        sorted(p.name for p in run.glob("ring-*")),
        rows,
    )


def final_state(run: Path) -> tuple[int, list[int], list[dict]]:
    committed = latest_committed(run)
    ring = ReplayRing.load(run / f"ring-{committed}", CFG)
    pars = [int(ring.get(slot)["par"]) for slot in range(len(ring))]
    rows = [json.loads(line) for line in (run / "iterations.jsonl").read_text().splitlines()]
    return committed, pars, rows


# ---------------------------------------------------------------------------
# The baseline the kills are compared against
# ---------------------------------------------------------------------------


def test_an_uninterrupted_run_commits_every_iteration(tmp_path: Path) -> None:
    Harness(tmp_path / "run").play()
    committed, pars, rows = final_state(tmp_path / "run")
    assert committed == 2
    assert pars == [1, 1, 2, 2, 3, 3]
    assert [r["iteration"] for r in rows] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Kill point A — after the ring and state, before the row
# ---------------------------------------------------------------------------


def test_kill_before_the_row_resumes_clean(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Harness(run, kill_at=(1, "A")).play()
    assert latest_committed(run) == 0, "iteration 1 must not be committed"

    Harness(run).play()
    committed, pars, rows = final_state(run)
    assert committed == 2
    assert pars == [1, 1, 2, 2, 3, 3], "the ring must not carry iteration 1 twice"
    assert [r["iteration"] for r in rows] == [0, 1, 2]


def test_kill_at_a_leaves_no_orphan_artifacts(tmp_path: Path) -> None:
    """A ring-7 beside a LATEST of 6 looks like history and is not."""
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Harness(run, kill_at=(1, "A")).play()
    assert (run / "ring-1").exists(), "the orphan exists before resume"
    resume(run, CFG)
    assert not (run / "ring-1").exists(), "resume must drop the uncommitted ring"
    assert not (run / "state-1.json").exists()


# ---------------------------------------------------------------------------
# Kill point B — after the row, before LATEST
# ---------------------------------------------------------------------------


def test_kill_after_the_row_resumes_clean(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Harness(run, kill_at=(1, "B")).play()
    assert latest_committed(run) == 0
    rows = (run / "iterations.jsonl").read_text().splitlines()
    assert len(rows) == 2, "the uncommitted row is on disk before resume"

    Harness(run).play()
    committed, pars, rows = final_state(run)
    assert committed == 2
    assert pars == [1, 1, 2, 2, 3, 3]
    assert [r["iteration"] for r in rows] == [0, 1, 2], "the uncommitted row must be truncated"


def test_the_uncommitted_row_is_truncated_not_kept(tmp_path: Path) -> None:
    """Leaving it would make the log claim an iteration the ring does not contain."""
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Harness(run, kill_at=(2, "B")).play()
    assert len((run / "iterations.jsonl").read_text().splitlines()) == 3
    resume(run, CFG)
    assert len((run / "iterations.jsonl").read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# Both kills land on the same state as no kill — the actual claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("point", ["A", "B"])
def test_a_killed_run_is_indistinguishable_from_one_that_never_stopped(
    tmp_path: Path, point: str
) -> None:
    clean = tmp_path / "clean"
    Harness(clean).play()

    killed = tmp_path / "killed"
    with pytest.raises(KeyboardInterrupt):
        Harness(killed, kill_at=(1, point)).play()
    Harness(killed).play()

    assert final_state(clean) == final_state(killed)


# ---------------------------------------------------------------------------
# The commit point's own properties
# ---------------------------------------------------------------------------


def test_a_fresh_run_starts_at_zero(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert resume(run, CFG) == (0, None, None)


def test_a_fresh_run_discards_a_stray_log(tmp_path: Path) -> None:
    """Rows without a commit are not history."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "iterations.jsonl").write_text('{"iteration": 0}\n')
    resume(run, CFG)
    assert not (run / "iterations.jsonl").exists()


def test_a_corrupt_marker_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "LATEST").write_text("nearly\n")
    with pytest.raises(ResumeError, match="not an iteration number"):
        latest_committed(run)


def test_the_value_head_declaration_survives_a_resume(tmp_path: Path) -> None:
    """The ratchet must not un-ratchet across a crash."""
    run = tmp_path / "run"
    run.mkdir()
    ring = ReplayRing(8, CFG)
    ring.append(**a_record(par=1))
    state = RunState(
        iteration=0,
        value_head=ValueHeadState(live=True, switched_at_iteration=0, switched_accuracy=0.8),
        seed=7,
    )
    commit_iteration(run, ring, state, lambda: None)
    _, _, restored = resume(run, CFG)
    assert restored is not None
    assert restored.value_head.live is True
    assert restored.value_head.switched_at_iteration == 0


# ---------------------------------------------------------------------------
# Ring retention — priced before it was chosen, and pruning cannot lose a commit
# ---------------------------------------------------------------------------


def test_only_the_last_k_rings_are_kept(tmp_path: Path) -> None:
    """Unpruned, a run costs 0.73 GiB per iteration — 36.3 GiB by iteration 50.

    Disk exhaustion mid-campaign is not how this policy should be discovered.
    """
    run = tmp_path / "run"
    Harness(run).play(iterations=6)
    rings = sorted(int(p.name.removeprefix("ring-")) for p in run.glob("ring-*"))
    assert rings == [3, 4, 5], f"expected the last {KEEP_RINGS}, got {rings}"
    assert not (run / "state-0.json").exists(), "state files prune with their rings"
    assert (run / "state-5.json").exists()


def test_pruning_never_removes_the_committed_iteration(tmp_path: Path) -> None:
    run = tmp_path / "run"
    Harness(run).play(iterations=6)
    committed = latest_committed(run)
    assert (run / f"ring-{committed}").exists()
    prune_rings(run)  # re-running must not touch it
    assert (run / f"ring-{committed}").exists()


def test_a_resume_still_works_after_pruning(tmp_path: Path) -> None:
    """The retention rule must not break the thing it shares a directory with."""
    run = tmp_path / "run"
    Harness(run).play(iterations=6)
    nxt, ring, state = resume(run, CFG)
    assert nxt == 6 and ring is not None and state is not None
    assert len(ring) == 12


# ---------------------------------------------------------------------------
# Repair idempotence — the one path that only runs after something went wrong
# ---------------------------------------------------------------------------


def test_repair_converges_when_re_run(tmp_path: Path) -> None:
    """A crash DURING the repair must converge, not compound.

    Every step is a drop-beyond-LATEST: truncation to a prefix is idempotent,
    removing an already-removed orphan is a no-op, and nothing here writes
    LATEST. Running it three times must equal running it once.
    """
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Harness(run, kill_at=(2, "B")).play()

    first = resume(run, CFG)
    snapshot = final_state_partial(run)
    for _ in range(2):
        again = resume(run, CFG)
        assert again[0] == first[0]
        assert final_state_partial(run) == snapshot


def test_repair_is_idempotent_at_kill_point_a_too(tmp_path: Path) -> None:
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Harness(run, kill_at=(1, "A")).play()
    first = resume(run, CFG)
    snapshot = final_state_partial(run)
    assert resume(run, CFG)[0] == first[0]
    assert final_state_partial(run) == snapshot


def test_repair_on_an_already_clean_run_changes_nothing(tmp_path: Path) -> None:
    """The accepting case: repair must be a no-op when there is nothing to repair."""
    run = tmp_path / "run"
    Harness(run).play()
    before = final_state(run)
    resume(run, CFG)
    assert final_state(run) == before


# ------------------------------------------------- every append-only log (F-26)


def test_resume_truncates_the_switch_log_not_only_the_iteration_log() -> None:
    """F-26. `iterations.jsonl` was truncated here and the other logs were not, so
    a killed-and-resumed run wrote its switch row twice for the redone iteration —
    F-13's duplication signature, in the row class nothing compared."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        Harness(run).play(iterations=2)  # commits 0 and 1
        (run / "value_switch.jsonl").write_text(
            "".join(json.dumps({"iteration": n}) + "\n" for n in (0, 1))
        )
        (run / "LATEST").write_text("0\n")  # as if iteration 1 never committed

        resume(run, CFG)

        kept = [json.loads(x) for x in (run / "value_switch.jsonl").read_text().splitlines() if x]
        assert [r["iteration"] for r in kept] == [0], "the uncommitted switch row survived"


def test_resume_truncates_the_instrument_log_by_iteration_not_by_position() -> None:
    """`instruments.jsonl` carries one row per CADENCE, not per iteration, so a
    positional prefix would be meaningless for it — iteration 4's instrument row
    is the FIRST line of the file, and dropping "everything after line 0" would
    keep a row for an iteration that never committed.

    This is the door-level cover for a case the campaign resume gate cannot
    reach: golden config sets `ladder_every = 99`, so its cadence never fires and
    no instrument row is ever written there.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        Harness(run).play(iterations=2)
        # Cadence rows for iterations 1 and 9, with the run committed only to 1.
        # A POSITIONAL prefix of `committed + 1 = 2` lines would keep BOTH — which
        # is precisely the wrong answer this truncation must not give.
        (run / "instruments.jsonl").write_text(
            "".join(json.dumps({"iteration": n, "primary": {}}) + "\n" for n in (1, 9))
        )
        (run / "LATEST").write_text("1\n")

        resume(run, CFG)

        kept = [json.loads(x) for x in (run / "instruments.jsonl").read_text().splitlines() if x]
        assert [r["iteration"] for r in kept] == [1], (
            "iteration 9's instrument row survived a commit that stopped at 1"
        )


def test_truncating_the_logs_is_idempotent() -> None:
    """The repair path only ever runs after something has already gone wrong, so a
    crash during the repair must converge rather than compound."""
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        Harness(run).play(iterations=2)
        (run / "value_switch.jsonl").write_text(
            "".join(json.dumps({"iteration": n}) + "\n" for n in (0, 1, 2))
        )
        (run / "LATEST").write_text("0\n")

        resume(run, CFG)
        once = (run / "value_switch.jsonl").read_text()
        resume(run, CFG)
        assert (run / "value_switch.jsonl").read_text() == once


# ---------------------------------------------------------------------------
# A ring whose policy targets predate F-41 is not resumed into
# ---------------------------------------------------------------------------


def _commit_one(run: Path, ring: ReplayRing) -> None:
    state = RunState(iteration=0, value_head=ValueHeadState(), seed=7)
    commit_iteration(run, ring, state, lambda: None)


def test_resume_refuses_a_ring_whose_visit_actions_predate_f41(tmp_path: Path) -> None:
    """Format-1 records hold root SLOT positions, not actions (F-41).

    ``ring_batch`` sits such a record out, which is right for a reader and silent
    for a campaign: a resume across the fix would quietly train its policy on the
    new rows alone, at an unchanged fingerprint, after iterations that trained on
    the defect. The campaign fingerprint cannot see this, so resume refuses.
    """
    ring = ReplayRing(8, CFG, ring_format=1)
    ring.append(**a_record(par=1))
    _commit_one(tmp_path, ring)
    with pytest.raises(ResumeError, match="F-41"):
        resume(tmp_path, CFG)


def test_resume_accepts_a_ring_written_in_the_current_format(tmp_path: Path) -> None:
    """The other polarity, so the refusal cannot be 'always refuse'."""
    ring = ReplayRing(8, CFG)
    ring.append(**a_record(par=1))
    _commit_one(tmp_path, ring)
    start, loaded, _state = resume(tmp_path, CFG)
    assert start == 1 and loaded is not None and len(loaded) == 1
