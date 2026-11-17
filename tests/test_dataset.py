"""Datasets, frozen suites, contamination, and reproducibility from seed.

The suites are the measuring stick. Every check here exists because a suite that
is wrong, or that leaked into training, does not fail loudly — it produces
plausible numbers for a year.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config
from reckoner.dataset import (
    DATA_ROOT,
    FIELDS,
    InstrumentAsTrainingSource,
    RecordWouldBeUntracked,
    assert_training_source,
    dataset_keys,
    problem_key,
    read_dataset,
    read_suite,
    sample_indices,
    sha256_file,
    source_role,
    state_keys,
    suite_problem,
    training_problems,
    write_dataset,
    write_record,
    write_suite,
)
from reckoner.episode import Problem, bfs_par, bfs_solution, decode_state
from reckoner.expr import eq, identity_key, mul, num, var
from reckoner.generator import check_emission
from reckoner.rules import RULESET_VERSION
from reckoner.vocab import GOAL_SOLVE, VAR_X, VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
DATA = REPO / DATA_ROOT
CFG = Config()
X = var(VAR_X)

DEPTHS = (1, 2, 3, 4, 5, 6)

pytestmark = pytest.mark.skipif(
    not (SUITES / "solve_in_1.jsonl").exists(),
    reason="suites not generated yet — run `python scripts/generate.py`",
)


def suite(depth: int) -> list[dict]:
    return read_suite(SUITES / f"solve_in_{depth}.jsonl")


# ---------------------------------------------------------------------------
# Suites: shape and provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", DEPTHS)
def test_suite_exists_and_is_the_declared_size(depth: int) -> None:
    rows = suite(depth)
    assert len(rows) == 200, f"solve_in_{depth} has {len(rows)} problems"


@pytest.mark.parametrize("depth", DEPTHS)
def test_every_suite_row_carries_its_versions(depth: int) -> None:
    for row in suite(depth):
        assert row["ruleset_version"] == RULESET_VERSION
        assert row["vocab_version"] == VOCAB_VERSION
        assert row["par_source"] == "bfs"
        assert row["par"] == row["depth"] == depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_every_suite_row_obeys_the_emission_grammar(depth: int) -> None:
    """The grammar is re-checked from disk, not trusted from generation."""
    for row in suite(depth):
        check_emission(suite_problem(row))


def test_suites_are_goal_diverse_where_the_rule_set_allows() -> None:
    """A stratum whose goal mix is an accident of which templates exist is not
    an instrument.

    The first generation produced `solve_in_2` with no SOLVE at all and
    `solve_in_6` with 200/200 EVALUATE — the deepest suite, the one that matters
    most for measuring *solving* rather than *computing*, contained no equations.
    Two templates were added to close both ends.
    """
    for depth in DEPTHS:
        goals = {row["goal"] for row in suite(depth)}
        assert GOAL_SOLVE in goals, f"solve_in_{depth} contains no SOLVE problem"
        assert len(goals) >= 2, f"solve_in_{depth} is a single-goal suite: {goals}"


# ---------------------------------------------------------------------------
# The depth label, re-verified independently of generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", DEPTHS)
def test_suite_depth_labels_are_reverified_by_bfs(depth: int) -> None:
    """**Chunk 5 gate.** Every label re-derived from disk, not trusted.

    Two-sided, because "a solution of length N exists" and "no shorter one
    exists" are different claims and only the pair means *minimum*:

      * a derivation of exactly ``depth`` steps exists, and
      * BFS at cap ``depth - 1`` finds nothing.
    """
    rows = suite(depth)
    # A *sample* here; the exhaustive pass lives in scripts/verify_suites.py and
    # its artifact is checked below. Depth-6 SOLVE labelling is ~4.5 s per
    # problem times three BFS runs, so verifying all 1,200 inline would make
    # `make test` a ten-minute command — and a ten-minute `make test` is a
    # command people stop running, which costs more than it buys.
    sample = rows[:25] if depth <= 4 else rows[:6]
    for row in sample:
        problem = suite_problem(row)
        assert bfs_par(problem, CFG) == depth, f"label {depth} disagrees with BFS"
        path = bfs_solution(problem, CFG)
        assert path is not None and len(path) == depth
        assert bfs_solution(problem, CFG, cap=depth - 1) is None, (
            f"a shorter derivation exists — {depth} is not the minimum"
        )


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (DATA / "train_100k").exists(), reason="training set not generated")
def test_no_suite_contamination() -> None:
    """**Chunk 5 gate.** No suite problem may appear in any training set.

    Ported with its config-field guard: the check is run against *every* dataset
    directory that exists, so a training set added later cannot quietly escape
    the test by not being named in it.
    """
    checked = []
    deferred = []
    suite_all: set[tuple[int, ...]] = set()
    for depth in DEPTHS:
        suite_all |= state_keys(suite(depth))

    for path in sorted(DATA.iterdir()):
        if not (path / "meta.json").exists():
            continue
        mode = json.loads((path / "meta.json").read_text())["mode"]
        if mode == "phase1_supervision":
            # A supervision set's rows are *states along derivations*, not
            # problems, so this check's key does not apply to them — but they are
            # NOT thereby exempt. They are handed to the state-level check below,
            # by name, and the handover is asserted. A1: a dataset that matches
            # neither check must fail the build, never fall through it.
            deferred.append(path.name)
            continue
        dataset = read_dataset(path)
        overlap = dataset_keys(dataset) & suite_all
        assert not overlap, f"{path.name} contains {len(overlap)} suite problems"
        checked.append(path.name)

    assert checked, "no dataset directories were checked — the guard is vacuous"
    assert "train_100k" in checked, f"the training set was not among {checked}"
    # The universal walk regains its universality: every dataset is covered by
    # exactly one check, and the set of deferrals is pinned rather than open.
    assert set(deferred) <= set(SUPERVISION_SETS), (
        f"{sorted(set(deferred) - set(SUPERVISION_SETS))} deferred to a state-level check "
        "that does not name them — a new dataset mode escaped both gates"
    )


@pytest.mark.skipif(
    not (DATA / "phase1_supervision_marker").exists() and not (DATA / "phase1_train").exists(),
    reason="phase-1 supervision not built",
)
def test_the_supervision_set_names_the_problem_set_it_inherits_from() -> None:
    """A derived set inherits contamination status — but only if it can prove it.

    The supervision set holds states along derivations of the training problems,
    so it is clean iff its source was. That is an inheritance, and an
    inheritance without a verified parent is an assumption: the recorded source
    digests must still equal the source's digests *now*.
    """
    meta = json.loads((DATA / "phase1_train" / "meta.json").read_text())
    assert meta["mode"] == "phase1_supervision"
    assert meta["source"].endswith("train_100k")
    source = json.loads((DATA / "train_100k" / "meta.json").read_text())
    assert meta["source_digests"] == source["digests"], (
        "the supervision set was built from a different train_100k than the one on "
        "disk — its inherited contamination status does not carry over"
    )
    assert meta["problems_dropped"] == 0, "some problems produced no derivation"
    assert meta["ruleset_version"] == RULESET_VERSION
    assert meta["vocab_version"] == VOCAB_VERSION


# ---------------------------------------------------------------------------
# A1 — the state-level check the inheritance could not perform
# ---------------------------------------------------------------------------


# Every supervision set and the census record that covers it. A set added here
# without a record fails the gate below; a set added to runs/data/ without being
# added here fails the universal walk above. Neither can be forgotten quietly.
SUPERVISION_SETS = {
    "phase1_train": "supervision_contamination",
    "phase1_eval": "eval_suite_contamination",
}

needs_supervision = pytest.mark.skipif(
    not (DATA / "phase1_train" / "meta.json").exists(), reason="phase-1 supervision not built"
)


def _suite_state_keys() -> set[tuple[tuple[int, ...], int]]:
    """``(identity_key(expr), goal)`` for every suite START state."""
    keys = set()
    for depth in DEPTHS:
        for row in suite(depth):
            goal, _target, expr = decode_state(tuple(row["tokens"]))
            keys.add((identity_key(expr), goal))
    return keys


def _supervision_keys(path: Path, indices: list[int]) -> list[tuple[tuple[int, ...], int]]:
    meta = json.loads((path / "meta.json").read_text())
    n, width = meta["n"], meta["max_len"]
    tokens = np.memmap(path / "tokens.i32", dtype=np.int32, mode="r", shape=(n, width))
    lengths = np.memmap(path / "lengths.i32", dtype=np.int32, mode="r", shape=(n,))
    out = []
    for i in indices:
        goal, _target, expr = decode_state(tuple(int(t) for t in tokens[i, : lengths[i]]))
        out.append((identity_key(expr), goal))
    return out


@needs_supervision
@pytest.mark.parametrize("name", sorted(SUPERVISION_SETS))
def test_no_supervision_state_is_a_suite_start_state(name: str) -> None:
    """**A1's permanent gate.** Inheritance covers provenance, not derivations.

    ``train_100k``'s *problems* are disjoint from the suites. Its *derivations*
    pass through states the suites also use as starts: solving ``9x + (-28) = 44``
    passes through ``9x = 72``, and ``-4x = 28`` is a ``solve_in_1`` problem
    verbatim. The unroll created a question the problem-level check could not
    ask and the inheritance check could not see. Measured before the fix: 1,887
    of 313,628 examples (0.6017%), 116 distinct states, **zero of them start
    states** — entirely intermediates (`FINDINGS.md` F-08).

    Structured like the BFS re-verification above, and for the same reason: the
    exhaustive pass is ``scripts/census_supervision_contamination.py``, whose
    artifact is checked here **and pinned to the bytes it censused**, plus a
    bounded direct sample so this test is not purely trusting a record. A full
    in-test pass is 311,741 decodes — a ``make test`` people stop running.
    """
    path = DATA / name
    if not (path / "meta.json").exists():
        pytest.skip(f"{name} not built")
    census = REPO / "runs" / f"{SUPERVISION_SETS[name]}.json"
    assert census.exists(), f"no census artifact for {name} — run the census script"
    record = json.loads(census.read_text())
    meta = json.loads((path / "meta.json").read_text())

    # Currency, not just content: a record that does not match the bytes on disk
    # is a record about a different dataset.
    assert record["dataset_digests"] == meta["digests"], (
        f"the census was computed against a different {name} than the one on disk — "
        "rebuild the census before trusting its zero"
    )
    assert record["examples"] == meta["n"]
    assert record["colliding_examples"] == 0, (
        f"{record['colliding_examples']} supervision states are suite start states"
    )

    # And a bounded live sample, so deleting the artifact check alone cannot hide
    # a fresh collision.
    suite_keys = _suite_state_keys()
    rng = random.Random(0)
    sample = sorted(rng.sample(range(meta["n"]), min(4000, meta["n"])))
    live = [k for k in _supervision_keys(path, sample) if k in suite_keys]
    assert not live, f"{len(live)} collisions in a {len(sample)}-row sample"


@needs_supervision
def test_the_contamination_probe_can_find_a_collision() -> None:
    """The other polarity, and here it is the one that matters.

    The gate above passes by finding nothing, which is indistinguishable from a
    probe that *cannot* find anything — law 5 rider (a). So the probe is run
    against a state known to be a suite start and must report it. Without this,
    a keying bug that returns the empty set forever would go green forever.
    """
    suite_keys = _suite_state_keys()
    assert suite_keys, "no suite keys built — the probe has nothing to detect"

    # Positive control: a real suite row, put through the same keying path the
    # gate uses, must be detected.
    row = suite(1)[0]
    goal, _target, expr = decode_state(tuple(row["tokens"]))
    assert (identity_key(expr), goal) in suite_keys, "the probe cannot see a known suite state"

    # Negative control: a state that is not a suite start must not be reported.
    assert (identity_key(eq(mul(num(97), X), num(9701))), goal) not in suite_keys


@pytest.mark.skipif(not (DATA / "eval_held_out").exists(), reason="eval set not generated")
def test_eval_is_disjoint_from_train() -> None:
    train = dataset_keys(read_dataset(DATA / "train_100k"))
    held = dataset_keys(read_dataset(DATA / "eval_held_out"))
    assert not (train & held), f"{len(train & held)} problems appear in both"


def test_suites_are_disjoint_from_each_other() -> None:
    """Different depths cannot share a problem: depth is a function of the state."""
    seen: dict[tuple[int, ...], int] = {}
    for depth in DEPTHS:
        for key in state_keys(suite(depth)):
            assert key not in seen, f"a problem is in both solve_in_{seen[key]} and _{depth}"
            seen[key] = depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_a_suite_has_no_internal_duplicates(depth: int) -> None:
    rows = suite(depth)
    assert len(state_keys(rows)) == len(rows), f"solve_in_{depth} repeats a problem"


def test_the_exhaustive_verification_artifact_covers_every_suite() -> None:
    """**Chunk 5 gate.** All 1,200 labels re-verified, and the record proves it.

    The artifact carries each suite's sha256 *as verified*, so a suite edited
    afterwards cannot inherit an old pass — the check is against the file that
    exists now, not the file that existed then.
    """
    path = REPO / "runs" / "suite_verification.json"
    assert path.exists(), "run `python scripts/verify_suites.py`"
    record = json.loads(path.read_text())

    assert not record["failures"], f"verification recorded failures: {record['failures'][:3]}"
    assert record["total"] == 1200, f"only {record['total']} problems verified"
    for depth in DEPTHS:
        name = f"solve_in_{depth}.jsonl"
        entry = record["suites"].get(name)
        assert entry, f"{name} was never verified"
        assert entry["verified"] == entry["problems"] == 200
        assert entry["sha256"] == sha256_file(SUITES / name), (
            f"{name} changed after it was verified — the pass does not carry over"
        )


# ---------------------------------------------------------------------------
# Reproducibility from seed
# ---------------------------------------------------------------------------


def test_suite_3_regenerates_byte_identical(tmp_path: Path) -> None:
    """**Chunk 5 gate.** The spot-check the plan names, in a fresh process.

    A frozen instrument that cannot be rebuilt from its seed is not frozen, it
    is merely old.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate.py"),
            "--suite-only",
            "3",
            "--out",
            str(tmp_path),
            "--workers",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rebuilt = tmp_path / "suites" / "solve_in_3.jsonl"
    original = SUITES / "solve_in_3.jsonl"
    assert sha256_file(rebuilt) == sha256_file(original), (
        "solve_in_3 did not regenerate byte-identically from its seed"
    )


# ---------------------------------------------------------------------------
# The memmap writer
# ---------------------------------------------------------------------------


def _labelled(par: int = 1) -> Problem:
    return Problem(
        goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X, par=par, par_source="bfs"
    )


def test_dataset_round_trips(tmp_path: Path) -> None:
    problems = [
        _labelled(),
        Problem(
            goal=GOAL_SOLVE,
            expr=eq(mul(num(4), X), num(20)),
            target=VAR_X,
            par=1,
            par_source="bfs",
        ),
    ]
    meta = write_dataset(tmp_path / "d", problems, CFG, mode="test", seed=1, repo=REPO)
    dataset = read_dataset(tmp_path / "d")
    assert len(dataset) == 2
    assert meta["ruleset_version"] == RULESET_VERSION
    assert meta["vocab_version"] == VOCAB_VERSION
    assert set(meta["digests"]) == set(FIELDS)
    for i, problem in enumerate(problems):
        assert dataset.state(i) == problem_key(problem)
        assert int(dataset.par[i]) == problem.par


def test_dataset_refuses_unshippable_rows(tmp_path: Path) -> None:
    """The shipping boundary, at the writer — the last place it can be enforced."""
    from reckoner.dataset import ShippingError

    with pytest.raises(ShippingError):
        write_dataset(
            tmp_path / "a",
            [Problem(goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X)],
            CFG,
            mode="test",
            seed=1,
            repo=REPO,
        )
    with pytest.raises(ShippingError, match="does not ship"):
        write_dataset(
            tmp_path / "b",
            [
                Problem(
                    goal=GOAL_SOLVE,
                    expr=eq(mul(num(3), X), num(15)),
                    target=VAR_X,
                    par=1,
                    par_source="unverified",
                )
            ],
            CFG,
            mode="test",
            seed=1,
            repo=REPO,
        )


def test_empty_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        write_dataset(tmp_path / "e", [], CFG, mode="test", seed=1, repo=REPO)


def test_suite_writer_refuses_unshippable_rows(tmp_path: Path) -> None:
    from reckoner.dataset import ShippingError

    with pytest.raises(ShippingError):
        write_suite(
            tmp_path / "s.jsonl",
            [Problem(goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X)],
        )


# ---------------------------------------------------------------------------
# ANCHORS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "runs" / "ANCHORS.sha256").exists(), reason="not generated")
def test_anchors_cover_every_artifact_and_still_match() -> None:
    """**Chunk 5 gate.** Digests on disk, and they are the digests of what is there."""
    lines = (REPO / "runs" / "ANCHORS.sha256").read_text().splitlines()
    entries = [line.split("  ", 1) for line in lines if line.strip()]
    assert entries, "ANCHORS.sha256 is empty"

    for digest, rel in entries:
        path = REPO / rel
        assert path.exists(), f"{rel} is anchored but missing"
        assert sha256_file(path) == digest, f"{rel} has changed since it was anchored"

    anchored = {rel for _, rel in entries}
    for depth in DEPTHS:
        assert f"runs/suites/solve_in_{depth}.jsonl" in anchored, f"suite {depth} unanchored"
    assert any(rel.endswith("meta.json") for rel in anchored), "no dataset meta.json anchored"


@pytest.mark.skipif(not (DATA / "train_100k").exists(), reason="training set not generated")
def test_training_set_meta_is_honest() -> None:
    meta = json.loads((DATA / "train_100k" / "meta.json").read_text())
    assert meta["par_source_histogram"] == {"bfs": meta["n"]}, "a non-BFS par shipped"
    assert set(meta["depth_histogram"]) <= {str(d) for d in DEPTHS} | set(DEPTHS)
    assert min(int(d) for d in meta["depth_histogram"]) >= 1, "a par-0 problem shipped"
    assert meta["git_sha"] != "unknown"
    assert len(meta["config_fingerprint"]) == 64


def test_the_generator_is_seeded_not_ambient() -> None:
    """Two plans from one seed are the same plan; from different seeds, not."""
    sys.path.insert(0, str(REPO / "scripts"))
    from generate import _plan

    assert _plan(60, 11) == _plan(60, 11)
    assert _plan(60, 11) != _plan(60, 12)
    random.seed(999)  # ambient state must not reach it
    assert _plan(60, 11) == _plan(60, 11)


# ---------------------------------------------------------------------------
# Records assert their own trackedness before the bytes land
# ---------------------------------------------------------------------------


def test_write_record_refuses_a_path_git_would_ignore(tmp_path: Path) -> None:
    """Both polarities, because this guard only earns its place if it can fire.

    Three records shipped untracked in chunk 8 and each was caught afterwards.
    The negation globs fixed the class; this fixes the mechanism. A guard that
    could never refuse would be a comment that happens to run.
    """
    ignored = REPO / "runs" / "phase1" / "phase1.pt"  # checkpoints stay ignored, by design
    with pytest.raises(RecordWouldBeUntracked):
        write_record(ignored.with_suffix(".pt"), {"never": "written"})
    assert not (REPO / "runs" / "phase1" / "phase1.pt.json").exists()


def test_write_record_writes_a_tracked_path(tmp_path: Path) -> None:
    """The accepting case: a path git will keep is written and round-trips."""
    target = REPO / "runs" / "gate_write_record_probe.json"
    try:
        write_record(target, {"probe": 1})
        assert json.loads(target.read_text()) == {"probe": 1}
    finally:
        target.unlink(missing_ok=True)


def test_write_record_skips_the_check_outside_a_repo(tmp_path: Path) -> None:
    """A clean checkout used as a library is not what this guards."""
    out = tmp_path / "nested" / "record.json"
    write_record(out, {"ok": True}, repo=tmp_path)
    assert json.loads(out.read_text()) == {"ok": True}


# ---------------------------------------------------------------------------
# The blessed subsampler — one implementation, never a prefix
# ---------------------------------------------------------------------------


def test_sample_indices_is_not_a_prefix() -> None:
    """The defect this exists to kill, asserted directly.

    Three defects came from ``range(k)`` on a stratum-ordered artifact. A sample
    that happens to be a prefix is the failure, so the test is not "it returns k
    indices" — it is "it does not return the first k".
    """
    picked = sample_indices(10_000, 50, seed=0)
    assert len(picked) == 50
    assert picked != list(range(50)), "the sampler returned a prefix"
    assert picked == sorted(picked), "indices must be sorted for sequential reads"
    assert len(set(picked)) == 50, "sampling must be without replacement"
    # It reaches the far end of the set, which a prefix never does.
    assert max(picked) > 5_000


def test_sample_indices_is_seed_stable_and_seed_sensitive() -> None:
    """Both polarities: reproducible on a seed, different across seeds."""
    assert sample_indices(1_000, 20, seed=7) == sample_indices(1_000, 20, seed=7)
    assert sample_indices(1_000, 20, seed=7) != sample_indices(1_000, 20, seed=8)


def test_sample_indices_returns_everything_when_asked_for_too_much() -> None:
    assert sample_indices(5, 5, seed=0) == [0, 1, 2, 3, 4]
    assert sample_indices(5, 99, seed=0) == [0, 1, 2, 3, 4]


@needs_supervision
def test_the_sampler_reaches_every_stratum() -> None:
    """The property that matters: a sample of a stratum-ordered set is mixed.

    ``phase1_train`` is laid out depth 1 first; a prefix of 2,000 rows is 2,000
    depth-1-and-2 states. This asserts the blessed sampler sees all six depths,
    which is the thing the three defects each failed to do.
    """
    meta = json.loads((DATA / "phase1_train" / "meta.json").read_text())
    n = meta["n"]
    depth = np.memmap(DATA / "phase1_train" / "depth.i32", dtype=np.int32, mode="r", shape=(n,))
    picked = sample_indices(n, 2_000, seed=0)
    assert {int(depth[i]) for i in picked} == {1, 2, 3, 4, 5, 6}
    assert {int(depth[i]) for i in range(2_000)} != {1, 2, 3, 4, 5, 6}, (
        "the prefix now spans all depths — this test's premise has changed"
    )


# ---------------------------------------------------------------------------
# The runtime boundary — every consumer re-opens the question its producers closed
# ---------------------------------------------------------------------------


def test_a_frozen_instrument_is_refused_as_an_episode_source() -> None:
    """The censuses guard datasets at BIRTH; nothing guarded the RUNTIME source.

    The chunk-9 shakedown drew from `solve_in_2` and did no harm — nothing
    trained — but demonstrated that the loop would happily consume its own
    measuring stick. A config slip in a real run trains on the suites and poisons
    every measurement downstream, including the ones used to detect it.
    """
    with pytest.raises(InstrumentAsTrainingSource, match="role 'instrument'"):
        assert_training_source(REPO / "runs" / "suites")


def test_the_held_out_set_is_an_instrument_too() -> None:
    """Held-out measures; it never trains. Same refusal, different artifact."""
    with pytest.raises(InstrumentAsTrainingSource):
        assert_training_source(DATA / "eval_held_out")


@pytest.mark.skipif(not (DATA / "train_100k").exists(), reason="training set not generated")
def test_a_training_source_is_accepted() -> None:
    """The other polarity — otherwise the guard would just mean 'refuse everything'."""
    assert_training_source(DATA / "train_100k")
    assert source_role(DATA / "train_100k") == "training"


def test_an_unclassified_artifact_refuses_rather_than_defaulting(tmp_path: Path) -> None:
    """F-02's shape at the runtime boundary: the permissive answer must not be free.

    Defaulting an unclassified artifact to "training" would hand it the trusted
    status nobody granted it.
    """
    stray = tmp_path / "mystery"
    stray.mkdir()
    with pytest.raises(InstrumentAsTrainingSource, match="declares no role"):
        source_role(stray)


def test_a_declared_role_in_meta_wins_over_the_registry(tmp_path: Path) -> None:
    """New artifacts carry their role natively; the registry is only for the
    frozen ones that cannot be rewritten without breaking their own digests."""
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "meta.json").write_text(json.dumps({"role": "instrument"}))
    assert source_role(fresh) == "instrument"
    with pytest.raises(InstrumentAsTrainingSource):
        assert_training_source(fresh)


@pytest.mark.skipif(not (DATA / "train_100k").exists(), reason="training set not generated")
def test_the_episode_loader_refuses_before_it_reads_anything() -> None:
    """The guard is the FIRST thing training_problems does, not a later check."""
    with pytest.raises(InstrumentAsTrainingSource):
        training_problems(DATA / "eval_held_out", 4, seed=0)
    problems = training_problems(DATA / "train_100k", 4, seed=0)
    assert len(problems) == 4
    assert all(p.par_source == "bfs" and p.par >= 1 for p in problems)


def test_the_strata_census_record_matches_the_frozen_suites() -> None:
    """F-60, pinned where a clean clone can check it: ``solve_in_6`` holds 95
    SOLVE problems, the census record says so, and it records none in the
    training set. The suites are tracked; the training set's arrays are not, so
    its side of the count is the record's."""
    import json
    from pathlib import Path

    from reckoner.dataset import read_suite
    from reckoner.vocab import GOAL_SOLVE

    repo = Path(__file__).resolve().parents[1]
    record = json.loads((repo / "runs" / "strata_census_result.json").read_text())
    rows = read_suite(repo / "runs" / "suites" / "solve_in_6.jsonl")
    solve = sum(row["goal"] == GOAL_SOLVE for row in rows)
    assert solve == 95 == record["suites"]["solve_in_6"]["by_par_and_goal"]["6"]["SOLVE"]
    assert record["suites"]["solve_in_6"]["both_sides_product_shape"] == 95
    train = record["datasets"]["train_100k"]
    assert "SOLVE" not in train["by_par_and_goal"]["6"]
    assert train["both_sides_product_shape"]["rows"] == 0
