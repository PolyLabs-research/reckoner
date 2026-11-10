"""The replay ring: capacity, absence semantics, era handling, and root_q's sign.

The sign test is the inherited p2_c obligation taken whole. Chess specified
`root_q` and never stored it, and the gap surfaced only when the z/q blend needed
the field — so here it is stored from field one AND its sign is proven before it
feeds anything. A value whose sign is wrong does not fail loudly; it trains
confidently in the wrong direction.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from reckoner.absence import AbsenceError
from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import Problem
from reckoner.replay import (
    IMPROVED_SLOTS,
    RING_FORMAT,
    Absent,
    ReplayRing,
    RingError,
    bytes_per_record,
)
from reckoner.search import search, uniform_stub

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()


def improved_fields(actions=(5, 9), probs=(0.75, 0.25), tail: float = 0.0) -> dict:
    """A format-3 record's π′ fields: ``IMPROVED_SLOTS`` wide, padded with zeros."""
    stored_actions = np.zeros(IMPROVED_SLOTS, dtype=np.int32)
    stored_probs = np.zeros(IMPROVED_SLOTS, dtype=np.float32)
    stored_actions[: len(actions)] = actions
    stored_probs[: len(probs)] = probs
    return {
        "improved_actions": stored_actions,
        "improved_probs": stored_probs,
        "improved_tail": tail,
    }


def a_record(**overrides) -> dict:
    record = {
        "tokens": np.array([1, 2, 3], dtype=np.int16),
        "site_positions": np.array([0, 1], dtype=np.int16),
        "visit_actions": np.array([5, 9], dtype=np.int32),
        "visit_counts": np.array([12, 4], dtype=np.int32),
        "root_q": 0.5,
        "z": 0,
        "par_source": "bfs",
        "par": 3,
        "steps_remaining": 2,
        "depth": 3,
        "goal": 16,
        **improved_fields(),
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# The arithmetic, asserted rather than quoted
# ---------------------------------------------------------------------------


def test_bytes_per_record_matches_the_declared_footprint() -> None:
    """The docstring's table and the code must not be two mental models.

    Fourth-instance discipline: a summary of a derivation drifts from it, so the
    number is computed and the declared value is checked against the computation.
    """
    assert bytes_per_record(CFG) == 2586  # 1558 before F-54's π′ fields


def test_the_declared_capacity_fits_the_declared_budget() -> None:
    """1.20 GiB at 500,000 against ~45 GiB realistically free since the
    2026-08-16 rebalance (AGENTS.md §3); 0.72 GiB before F-54."""
    gib = CFG.train.replay_capacity * bytes_per_record(CFG) / 1024**3
    assert gib == pytest.approx(1.20, abs=0.01)
    assert gib < 45.0


# ---------------------------------------------------------------------------
# Ring behaviour
# ---------------------------------------------------------------------------


def test_it_stores_and_returns_a_record() -> None:
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    got = ring.get(0)
    assert got["par_source"] == "bfs"
    assert got["root_q"] == pytest.approx(0.5)
    assert list(got["tokens"]) == [1, 2, 3]


def test_it_overwrites_oldest_first_and_length_saturates() -> None:
    ring = ReplayRing(3, CFG)
    for i in range(5):
        ring.append(**a_record(par=i))
    assert len(ring) == 3
    assert {int(ring.get(s)["par"]) for s in range(3)} == {2, 3, 4}


def test_sampling_uses_the_blessed_sampler_and_is_never_a_prefix() -> None:
    ring = ReplayRing(1000, CFG)
    for i in range(1000):
        ring.append(**a_record(par=i % 6 + 1))
    picked = ring.sample(50, seed=0)
    assert len(picked) == 50
    assert picked != list(range(50))


def test_reading_outside_the_valid_range_raises() -> None:
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    with pytest.raises(RingError, match="outside the 1 valid records"):
        ring.get(1)


# ---------------------------------------------------------------------------
# The third layer of the exact-par tripwire
# ---------------------------------------------------------------------------


def test_beating_exact_par_is_refused_at_the_ring_boundary() -> None:
    """EpisodeResult, logschema, and now here — checked wherever it crosses."""
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="impossible by construction"):
        ring.append(**a_record(par_source="bfs", z=1))


def test_beating_pool_par_is_allowed() -> None:
    """Pool par is not exact; beating it is the escalation mechanism."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record(par_source="pool", z=1))
    assert ring.get(0)["z"] == 1


def test_an_unknown_par_source_is_refused() -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="unknown par_source"):
        ring.append(**a_record(par_source="vibes"))


def test_an_out_of_range_z_is_refused() -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="z must be"):
        ring.append(**a_record(z=2))


# ---------------------------------------------------------------------------
# One absence semantics — and absence is never a zero
# ---------------------------------------------------------------------------


def test_an_undeclared_absence_is_refused() -> None:
    """A hole nobody can interpret is worse than a missing value."""
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="does not declare that it can be absent"):
        ring.append(**a_record(absent={"root_q": "search did not run"}))


def test_naming_an_unknown_field_absent_is_refused() -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="unknown field"):
        ring.append(**a_record(absent={"not_a_field": "because"}))


def test_era_absence_is_computed_not_stored() -> None:
    """A record written before a field existed cannot testify about it.

    The ring is written under format 0; every field arrived in format 1, so every
    field reads Absent with kind 'era' — computed from (record format, since).
    """
    ring = ReplayRing(4, CFG, ring_format=0)
    ring.append(**a_record())
    got = ring.get(0)
    assert isinstance(got["root_q"], Absent)
    assert got["root_q"].kind == "era"
    assert "predates_field" in got["root_q"].reason


def test_a_current_format_record_reports_no_era_absence() -> None:
    """The other polarity — otherwise era handling degenerates to 'always absent'."""
    ring = ReplayRing(4, CFG, ring_format=RING_FORMAT)
    ring.append(**a_record())
    assert not isinstance(ring.get(0)["root_q"], Absent)


def test_a_pre_f53_record_cannot_be_read_as_actions() -> None:
    """Format 1 stored root SLOT positions in ``visit_actions`` (F-53).

    A redefinition is an era boundary: the old values are not wrong-but-usable,
    they are in different coordinates, and the ring must refuse to hand them out
    as actions. Era-absent with the reason named, never the raw integers.
    """
    ring = ReplayRing(4, CFG, ring_format=1)
    ring.append(**a_record())
    got = ring.get(0)
    assert isinstance(got["visit_actions"], Absent)
    assert got["visit_actions"].kind == "era"
    assert "F-53" in got["visit_actions"].reason
    assert "slot" in got["visit_actions"].reason.lower()
    # Only the redefined field is affected; the rest of a format-1 record stands.
    assert not isinstance(got["visit_counts"], Absent)
    assert not isinstance(got["root_q"], Absent)


def test_a_current_format_record_hands_out_its_visit_actions() -> None:
    """The other polarity: without it, the refusal above could be 'always absent'."""
    assert RING_FORMAT >= 2
    ring = ReplayRing(4, CFG, ring_format=RING_FORMAT)
    ring.append(**a_record())
    assert ring.get(0)["visit_actions"].tolist()[:2] == [5, 9]
    assert ring.era_absent("visit_actions") == 0


def test_era_absent_counts_the_records_a_field_cannot_be_read_from() -> None:
    """Both formats in one ring: the count names exactly the old ones."""
    ring = ReplayRing(8, CFG, ring_format=1)
    ring.append(**a_record())
    ring.append(**a_record())
    ring.ring_format = RING_FORMAT
    ring.append(**a_record())
    assert ring.era_absent("visit_actions") == 2
    assert ring.era_absent("root_q") == 0


def test_ring_batch_sits_out_a_pre_f53_record() -> None:
    """A slot-indexed record must not become a policy target, wrong or dropped."""
    from reckoner.episode import encode_state
    from reckoner.generator import emit, label
    from reckoner.model import encode
    from reckoner.train import ring_batch

    problem = label(emit("solve_two_terms", random.Random(1)), CFG)
    assert problem is not None
    state = encode(problem, problem.expr, CFG)
    record = a_record(
        tokens=np.asarray(encode_state(problem.goal, problem.expr, problem.target), np.int16),
        site_positions=np.asarray(state.site_positions[: state.n_sites], np.int16),
        visit_actions=np.array([0], dtype=np.int32),
        visit_counts=np.array([1], dtype=np.int32),
    )
    old = ReplayRing(4, CFG, ring_format=1)
    old.append(**record)
    # The visit arm is the one F-53 is about; since F-54 it is no longer the default.
    assert ring_batch(old, [0], CFG, target="visits") is None

    new = ReplayRing(4, CFG)
    from reckoner.model import action_index
    from reckoner.rules import legal_actions

    legal = action_index(*legal_actions(problem.expr)[0], CFG.model.max_sites)
    new.append(**{**record, "visit_actions": np.array([legal], dtype=np.int32)})
    batch = ring_batch(new, [0], CFG, target="visits")
    assert batch is not None and batch.skipped == 0
    assert float(batch.policy_target[0, legal]) == 1.0


def test_an_absent_value_refuses_to_be_read_as_a_zero() -> None:
    """`Absent.__bool__` raises: truthiness is how an absence becomes a zero."""
    ring = ReplayRing(4, CFG, ring_format=0)
    ring.append(**a_record())
    absent = ring.get(0)["root_q"]
    with pytest.raises(AbsenceError, match="test for Absent"):
        bool(absent)


# ---------------------------------------------------------------------------
# root_q's SIGN and SCALE — re-pinned on the z currency (F-13)
# ---------------------------------------------------------------------------
#
# The first version of this test asserted root_value == 1.0 on a forced win and
# PASSED — which is how it exposed the defect. A forced win *at par* is z = 0, a
# draw, so a +1.0 reading meant the tree scored solved-flat while training scored
# z-vs-par: two currencies in one loop. The contrast that mattered was never
# win-versus-loss; it was at-par-versus-under-par.


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_root_q_is_plus_one_when_the_solve_beats_par() -> None:
    """Under par pays +1. Fixture par, since nothing beats real BFS par."""
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    base = suite_problem(row)
    under = Problem(goal=base.goal, expr=base.expr, par=5, target=base.target, par_source="pool")
    result = search(under, under.expr, uniform_stub(CFG), CFG, random.Random(0), sims=16, m=5)
    assert result.root_value == pytest.approx(1.0)


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_root_q_is_zero_when_the_solve_only_matches_par() -> None:
    """**The fixture the old test was missing.** At par is a DRAW, not a win.

    A depth-1 problem solved in one step ties BFS-exact par. Reading +1.0 here is
    the solved-flat defect; reading 0.0 is the par game.
    """
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    problem = suite_problem(row)
    result = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(0), sims=16, m=5)
    assert result.root_value == pytest.approx(0.0)


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_root_q_is_minus_one_when_the_line_runs_over_par() -> None:
    """Over par pays -1 — and with a NEUTRAL evaluator, so it comes from proofs.

    The sharper pin. Before the own-eval ruling this fixture needed a pessimistic
    evaluator, because `root_value` was `max(own evaluation, best line)` and a
    neutral 0.0 floored it. Now own-eval drops out of a fully expanded node, so
    the -1 must EMERGE FROM PROOFS ALONE with no cooperation from the evaluator.

    That makes this the regression test for the rule itself: if own-eval ever
    leaks back into a fully expanded node's max, this goes green-to-red.
    """
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    problem = suite_problem(row)
    result = search(
        problem,
        problem.expr,
        uniform_stub(CFG),  # neutral: value 0.0 everywhere
        CFG,
        random.Random(0),
        sims=16,
        m=16,  # every legal action considered, so the root can be fully expanded
        steps_taken=problem.par,  # the episode already spent its whole budget
    )
    assert result.root_value == pytest.approx(-1.0), (
        f"root_q is {result.root_value} with every line proven over par and a "
        "neutral evaluator — own-eval is flooring a fully expanded node"
    )


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_the_stored_root_q_is_the_searched_one() -> None:
    """The sign must survive the ring, not just the search."""
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    problem = suite_problem(row)
    result = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(0), sims=16, m=5)
    ring = ReplayRing(4, CFG)
    ring.append(**a_record(root_q=result.root_value, z=0, par_source="bfs"))
    assert ring.get(0)["root_q"] == pytest.approx(result.root_value)
    assert ring.get(0)["root_q"] == pytest.approx(0.0), "at par is a draw, and the ring stores it"


# ---------------------------------------------------------------------------
# The parked m -> 32 lever must trip loudly, not truncate
# ---------------------------------------------------------------------------


def test_raising_m_past_the_ring_slots_is_refused() -> None:
    """m is CONFIG; the visit layout is FORMAT. load()'s layout check cannot see
    a config that outgrew it, so the ring asserts at open."""
    cfg = Config()
    cfg.search.gumbel_m = 32
    with pytest.raises(RingError, match="exceeds the ring's 16 visit slots"):
        ReplayRing(4, cfg)


def test_m_at_the_slot_count_is_accepted() -> None:
    """The other polarity — the guard must not refuse the layout it was sized for."""
    cfg = Config()
    cfg.search.gumbel_m = 16
    assert ReplayRing(4, cfg).visit_actions.shape[1] == 16


# ---------------------------------------------------------------------------
# Persistence — the resume contract
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    ring = ReplayRing(8, CFG)
    for i in range(5):
        ring.append(**a_record(par=i + 1, root_q=i / 10))
    ring.save(tmp_path / "ring")
    back = ReplayRing.load(tmp_path / "ring", CFG)
    assert len(back) == 5
    assert back.cursor == ring.cursor
    for slot in range(5):
        assert back.get(slot)["par"] == ring.get(slot)["par"]
        assert back.get(slot)["root_q"] == pytest.approx(ring.get(slot)["root_q"])


def test_meta_is_written_last_so_its_presence_means_completeness(tmp_path: Path) -> None:
    """The write-ordering contract, asserted rather than described."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    meta = tmp_path / "ring" / "meta.json"
    assert meta.exists()
    newest = max((p.stat().st_mtime_ns for p in (tmp_path / "ring").iterdir()), default=0)
    assert meta.stat().st_mtime_ns == newest, "meta.json must be the last file written"


def test_loading_under_a_different_layout_is_refused(tmp_path: Path) -> None:
    """The record layout is denominated in these shapes; reinterpreting is silent."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    other = Config()
    other.model.max_sites = 64
    with pytest.raises(RingError, match="record layout is denominated"):
        ReplayRing.load(tmp_path / "ring", other)


def test_a_loaded_ring_stamps_new_records_with_the_writing_code_s_format(tmp_path: Path) -> None:
    """Per-record formats survive a load; NEW appends carry the current format.

    ``load`` used to rebuild the ring at the format in its meta, so every record
    appended after a resume was stamped with the OLD format. Harmless while only
    format 1 existed; the first bump (F-53) would have made every new, correct
    record read as era-absent. Both halves asserted.
    """
    old = ReplayRing(4, CFG, ring_format=1)
    old.append(**a_record())
    old.save(tmp_path / "ring")

    loaded = ReplayRing.load(tmp_path / "ring", CFG)
    loaded.append(**a_record())
    assert loaded.get(0)["ring_format"] == 1
    assert loaded.get(1)["ring_format"] == RING_FORMAT
    assert loaded.era_absent("visit_actions") == 1


def test_a_ring_from_a_newer_format_is_refused_at_load(tmp_path: Path) -> None:
    """Code cannot vouch for a meaning it predates."""
    import json

    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    meta_path = tmp_path / "ring" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["ring_format"] = RING_FORMAT + 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RingError, match="newer"):
        ReplayRing.load(tmp_path / "ring", CFG)


# ---------------------------------------------------------------------------
# F-54: ring format 3 — π′ fields, appended; old rings load era-absent
# ---------------------------------------------------------------------------


def test_the_pre_f54_field_positions_are_unchanged() -> None:
    """A field's position is its absent_mask bit. The π′ fields were APPENDED;
    an insertion would silently re-map every saved ring's absence bits."""
    from reckoner.replay import _FIELD_INDEX

    before = [
        "tokens",
        "site_positions",
        "length",
        "n_sites",
        "visit_actions",
        "visit_counts",
        "root_q",
        "z",
        "par_source",
        "par",
        "steps_remaining",
        "depth",
        "goal",
    ]
    assert [_FIELD_INDEX[name] for name in before] == list(range(len(before)))
    assert [_FIELD_INDEX[n] for n in ("improved_actions", "improved_probs", "improved_tail")] == [
        13,
        14,
        15,
    ]


def test_pi_prime_fields_round_trip_through_save_and_load(tmp_path: Path) -> None:
    ring = ReplayRing(4, CFG)
    ring.append(**a_record(**improved_fields(actions=(7, 3, 0), probs=(0.5, 0.25, 0.25))))
    ring.save(tmp_path / "ring")
    got = ReplayRing.load(tmp_path / "ring", CFG).get(0)
    assert got["improved_actions"].tolist()[:3] == [7, 3, 0]
    assert got["improved_probs"].tolist()[:3] == [0.5, 0.25, 0.25]
    assert got["improved_tail"] == 0.0


def _as_format_2(path: Path) -> None:
    """Rewrite a saved ring as a format-2 writer would have left it."""
    import json

    for name in ("improved_actions", "improved_probs", "improved_tail"):
        (path / f"{name}.bin").unlink()
    meta = json.loads((path / "meta.json").read_text())
    meta["ring_format"] = 2
    del meta["improved_slots"]
    (path / "meta.json").write_text(json.dumps(meta))


def test_a_format_2_ring_loads_with_its_pi_prime_era_absent(tmp_path: Path) -> None:
    old = ReplayRing(4, CFG, ring_format=2)
    old.append(**a_record())
    old.save(tmp_path / "ring")
    assert not (tmp_path / "ring" / "improved_probs.bin").exists(), (
        "a format-2 ring must be saved without the format-3 fields"
    )
    loaded = ReplayRing.load(tmp_path / "ring", CFG)
    got = loaded.get(0)
    assert isinstance(got["improved_probs"], Absent) and got["improved_probs"].kind == "era"
    assert not isinstance(got["visit_actions"], Absent), "format 2's visit actions stand"
    assert loaded.era_absent("improved_probs") == 1


def test_a_format_2_ring_carrying_format_3_files_is_refused(tmp_path: Path) -> None:
    import json

    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    meta = json.loads((tmp_path / "ring" / "meta.json").read_text())
    meta["ring_format"] = 2
    (tmp_path / "ring" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RingError, match="cannot contain improved"):
        ReplayRing.load(tmp_path / "ring", CFG)


def test_a_format_3_ring_missing_a_pi_prime_file_is_refused(tmp_path: Path) -> None:
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    (tmp_path / "ring" / "improved_tail.bin").unlink()
    with pytest.raises(RingError, match="missing improved_tail"):
        ReplayRing.load(tmp_path / "ring", CFG)


def test_a_ring_with_a_different_slot_count_is_refused(tmp_path: Path) -> None:
    import json

    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    meta = json.loads((tmp_path / "ring" / "meta.json").read_text())
    meta["improved_slots"] = 64
    (tmp_path / "ring" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RingError, match="π′ slots"):
        ReplayRing.load(tmp_path / "ring", CFG)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (improved_fields(probs=(0.75, 0.2)), "not 1"),
        (improved_fields(probs=(np.nan, 0.25)), "non-finite"),
        (improved_fields(probs=(1.25, -0.25)), "negative"),
        (improved_fields(tail=-0.1, probs=(0.75, 0.35)), "negative"),
        ({"improved_probs": np.ones(4, dtype=np.float32)}, "wide"),
    ],
)
def test_append_refuses_a_target_that_is_not_a_distribution(override: dict, match: str) -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match=match):
        ring.append(**a_record(**override))


def test_append_accepts_a_distribution_with_its_tail() -> None:
    """The other polarity: stored mass plus the recorded tail sums to one."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record(**improved_fields(probs=(0.75, 0.2), tail=0.05)))
    assert ring.get(0)["improved_tail"] == pytest.approx(0.05)
