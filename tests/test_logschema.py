"""The log schema, on both polarities.

Every check here has an accepting case beside its rejecting one. A schema
validator that only ever sees good rows is a validator nobody has shown can
refuse — rider (a): the first gate measures the component doing its central job,
and this component's central job is *saying no*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reckoner.logschema import (
    ITERATION_FIELDS,
    ROLES,
    SCHEMA_ERA,
    STEPS_MINUS_PAR_BINS,
    Field,
    SchemaError,
    alarm_census,
    append_row,
    describe,
    field_map,
    read_rows,
    validate_row,
)


def good_row(**overrides) -> dict:
    row = {
        "iteration": 0,
        "schema_era": SCHEMA_ERA,
        # D-A1 §1.1: the digest of the checkpoint the evaluator was built from.
        "evaluator_checkpoint_sha256": "d" * 64,
        # M1-A3 (era 3): the par-escalation pool as this iteration drew from it.
        "pool_composition": {
            "size": 2,
            "steps": [0, 5000],
            "order": [5000, 0],
            "value_head_live": [],
        },
        "run_name": "shakedown",
        "git_sha": "0" * 40,
        "config_fingerprint": "a" * 64,
        "ruleset_version": 1,
        "vocab_version": 1,
        "measure_dtype": "fp32",
        "train_dtype": "bf16",
        "solve_rate_by_depth": {"1": 1.0},
        "z_by_par_source": {"bfs": {"+1": 0, "0": 5, "-1": 1}},
        "steps_minus_par_histogram": dict.fromkeys(STEPS_MINUS_PAR_BINS, 0) | {"0": 5},
        "entropy_prior_step1_start": 1.2,
        "entropy_prior_step1_reached": 1.1,
        "entropy_target_step1_start": 0.8,
        "entropy_target_step1_reached": 0.7,
        # F-42 (era 4): the visit share's entropy under its own name, the KL that
        # moves while the value head is silent, and the π′ truncation counter.
        "entropy_visits_step1_start": 1.0,
        "entropy_visits_step1_reached": 0.9,
        "kl_improved_prior_step1_start": 0.4,
        "kl_improved_prior_step1_reached": 0.3,
        "ring_improved_truncated": 0,
        "episodes": 6,
        "episodes_solved": 5,
        "episodes_capped": 1,
        "episodes_stuck": 0,
        "episodes_conceded": 0,
        "search_nodes_total": 96,
        "search_evaluations_total": 96,
        "terminal_no_actions": 0,
        "state_too_large": 0,
        "nan_skips": 0,
        "pool_refusals": 0,
        "seconds_self_play": 1.0,
        "seconds_train": 2.0,
        "seconds_total": 3.0,
        "absent": {
            "pool_par_fraction": "league.par_from_pool_frac is 0 in the shakedown config",
            "ladder_pass": "not a ladder iteration",
        },
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The accepting case
# ---------------------------------------------------------------------------


def test_a_complete_row_validates() -> None:
    validate_row(good_row())


def test_every_field_declares_a_known_role() -> None:
    for f in ITERATION_FIELDS:
        assert f.role in ROLES, f"{f.name} has role {f.role!r}"


def test_the_schema_covers_every_obligation_the_brief_named() -> None:
    """The brief lists the columns by name; this asserts they exist.

    Not decoration: `logschema.py` from field one is only worth anything if the
    fields the brief registered are the fields it has, and the check belongs
    beside the schema rather than in a reviewer's memory.
    """
    names = set(field_map())
    for required in (
        "entropy_prior_step1_start",
        "entropy_prior_step1_reached",
        "entropy_target_step1_start",
        "entropy_target_step1_reached",
        "solve_rate_by_depth",
        "z_by_par_source",
        "steps_minus_par_histogram",
        "state_too_large",
        "nan_skips",
        "pool_refusals",
        "seconds_self_play",
        "seconds_train",
        "seconds_total",
    ):
        assert required in names, f"the brief registered {required} and it is missing"


# ---------------------------------------------------------------------------
# The rejecting cases — one per rule
# ---------------------------------------------------------------------------


def test_unknown_column_is_refused() -> None:
    with pytest.raises(SchemaError, match="unknown columns"):
        validate_row(good_row(solve_rate=0.9))


def test_missing_required_column_is_refused() -> None:
    row = good_row()
    del row["episodes"]
    with pytest.raises(SchemaError, match="required field missing"):
        validate_row(row)


def test_absent_optional_without_a_reason_is_refused() -> None:
    """The rule this module exists for."""
    row = good_row()
    del row["absent"]["ladder_pass"]
    with pytest.raises(SchemaError, match="absent without a reason"):
        validate_row(row)


def test_an_empty_reason_is_not_a_reason() -> None:
    row = good_row()
    row["absent"]["ladder_pass"] = "   "
    with pytest.raises(SchemaError, match="absent without a reason"):
        validate_row(row)


def test_null_is_refused_even_for_an_optional_field() -> None:
    """A null reads as zero to the next histogram that touches it."""
    row = good_row(pool_par_fraction=None)
    del row["absent"]["pool_par_fraction"]
    with pytest.raises(SchemaError, match="null is not how absence is expressed"):
        validate_row(row)


def test_absent_naming_a_present_column_is_refused() -> None:
    row = good_row(pool_par_fraction=0.2)
    with pytest.raises(SchemaError, match="named in 'absent' but present"):
        validate_row(row)


def test_absent_naming_an_unknown_column_is_refused() -> None:
    row = good_row()
    row["absent"]["not_a_column"] = "because"
    with pytest.raises(SchemaError, match="unknown column"):
        validate_row(row)


def test_wrong_type_is_refused() -> None:
    with pytest.raises(SchemaError, match="expected"):
        validate_row(good_row(episodes="six"))


# ---------------------------------------------------------------------------
# The Field constructor guards its own invariants
# ---------------------------------------------------------------------------


def test_optional_field_must_declare_an_absence_reason() -> None:
    with pytest.raises(SchemaError, match="MUST declare why it may be absent"):
        Field("x", int, "counter", "doc", required=False)


def test_required_field_may_not_declare_an_absence_reason() -> None:
    with pytest.raises(SchemaError, match="cannot declare an absence reason"):
        Field("x", int, "counter", "doc", absence="sometimes")


def test_role_outside_the_closed_vocabulary_is_refused() -> None:
    with pytest.raises(SchemaError, match="is not in"):
        Field("x", int, "vibes", "doc")


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "iterations.jsonl"
    append_row(path, good_row(iteration=0))
    append_row(path, good_row(iteration=1))
    rows = read_rows(path)
    assert [r["iteration"] for r in rows] == [0, 1]


def test_append_refuses_to_write_an_invalid_row(tmp_path: Path) -> None:
    """Validation at write time, not at analysis time.

    Writing first and validating later is how a run produces fifty iterations of
    unreadable log and finds out after the run is over.
    """
    path = tmp_path / "iterations.jsonl"
    with pytest.raises(SchemaError):
        append_row(path, good_row(episodes="six"))
    assert not path.exists(), "an invalid row must not reach the file"


def test_read_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "iterations.jsonl"
    append_row(path, good_row(iteration=0))
    with path.open("a") as fh:
        fh.write(json.dumps({"iteration": 1}) + "\n")
    with pytest.raises(SchemaError, match=r":2: "):
        read_rows(path)


def test_describe_renders_every_column(tmp_path: Path) -> None:
    text = describe()
    for f in ITERATION_FIELDS:
        assert f"`{f.name}`" in text


# ---------------------------------------------------------------------------
# The era boundary — both sides, because this is the one seam where "absence
# carries a reason" and "the row could not have known" collide
# ---------------------------------------------------------------------------

NEWER = ITERATION_FIELDS + (
    Field("added_later", int, "counter", "A column introduced in era 2.", since=2),
)


def test_an_old_era_row_may_omit_a_field_that_did_not_exist_yet() -> None:
    """Reading history must not require history to have predicted the future.

    The reason for this absence is *computed* from (row era, field.since), never
    asserted by the row — a row written under era 1 could not have explained the
    absence of a column nobody had named.
    """
    validate_row(good_row(schema_era=1), NEWER)


def test_a_current_era_row_may_not_omit_that_same_field() -> None:
    """The other side. Without it, era handling degenerates into 'stop checking'."""
    with pytest.raises(SchemaError, match="required field missing"):
        validate_row(good_row(schema_era=2), NEWER)


def test_the_era_message_names_both_numbers() -> None:
    with pytest.raises(SchemaError, match=r"row era 2 >= field since 2"):
        validate_row(good_row(schema_era=2), NEWER)


def test_a_field_present_before_its_era_is_still_accepted() -> None:
    """Writing a column early is not an error; omitting it late is."""
    validate_row(good_row(schema_era=1, added_later=3), NEWER)


# ---------------------------------------------------------------------------
# Pinned bins — schema, not config
# ---------------------------------------------------------------------------


def test_every_bin_is_present_in_every_row_including_the_zeros() -> None:
    row = good_row()
    assert set(row["steps_minus_par_histogram"]) == set(STEPS_MINUS_PAR_BINS)
    validate_row(row)


def test_a_missing_bin_is_refused_rather_than_read_as_zero() -> None:
    row = good_row()
    del row["steps_minus_par_histogram"]["6+"]
    with pytest.raises(SchemaError, match="bins must be exactly"):
        validate_row(row)


def test_an_invented_bin_is_refused() -> None:
    row = good_row()
    row["steps_minus_par_histogram"]["7"] = 1
    with pytest.raises(SchemaError, match="unexpected"):
        validate_row(row)


# ---------------------------------------------------------------------------
# The (exact-par, +1) tripwire — the second, independent layer
# ---------------------------------------------------------------------------


def test_the_impossible_cell_is_present_at_zero() -> None:
    validate_row(good_row())  # fixture carries {"bfs": {"+1": 0, ...}}


def test_an_absent_impossible_cell_is_refused() -> None:
    """A tripwire that can be omitted is a tripwire that cannot fire."""
    row = good_row()
    del row["z_by_par_source"]["bfs"]["+1"]
    with pytest.raises(SchemaError, match="explicit '\\+1' cell"):
        validate_row(row)


def test_beating_exact_par_is_refused_at_write_time() -> None:
    """F-02's shape, refused by a second module on a second code path."""
    row = good_row()
    row["z_by_par_source"]["bfs"]["+1"] = 1
    with pytest.raises(SchemaError, match="EXACT par"):
        validate_row(row)


def test_a_non_exact_source_may_legitimately_beat_its_par() -> None:
    """Pool par is not exact — beating it is the whole escalation mechanism, and
    widening the invariant to silence that is what BRIEF-chunk9 forbids."""
    row = good_row()
    row["z_by_par_source"]["pool"] = {"+1": 4, "0": 2, "-1": 1}
    validate_row(row)


# ---------------------------------------------------------------------------
# Splits sum, and the histogram's population is named rather than inferred
# ---------------------------------------------------------------------------


def test_outcomes_sum_to_episodes() -> None:
    validate_row(good_row())


def test_a_split_that_does_not_sum_is_refused() -> None:
    """A lost or double-counted episode looks like a plausible number otherwise."""
    with pytest.raises(SchemaError, match="outcomes do not sum"):
        validate_row(good_row(episodes_capped=2))


def test_the_histogram_population_is_solved_episodes_only() -> None:
    """Capped episodes must not be filed in any bin.

    Draw-inflation (the 0 bin rising) and timeout-composition (episodes_capped
    rising) are different diseases with different fixes. A histogram that pools
    them is two instruments sharing a name.
    """
    row = good_row()
    row["steps_minus_par_histogram"]["6+"] = 1  # the capped episode, mis-filed
    with pytest.raises(SchemaError, match="population is SOLVED episodes only"):
        validate_row(row)


def test_the_histogram_must_account_for_every_solved_episode() -> None:
    row = good_row()
    row["steps_minus_par_histogram"]["0"] = 4  # one solved episode unaccounted
    with pytest.raises(SchemaError, match="episodes_solved"):
        validate_row(row)


# ---------------------------------------------------------------------------
# Two species of zero, two dispositions
# ---------------------------------------------------------------------------


def test_a_definitional_zero_refuses_the_row(tmp_path: Path) -> None:
    """Nothing beats exact par by the meaning of the words.

    A nonzero reading could not have been evidence of anything except its own
    wrongness, so suppressing it loses nothing.
    """
    row = good_row()
    row["z_by_par_source"]["bfs"]["+1"] = 1
    path = tmp_path / "iterations.jsonl"
    with pytest.raises(SchemaError, match="EXACT par"):
        append_row(path, row)
    assert not path.exists(), "a definitionally impossible row must not land"


def test_a_premise_zero_writes_and_alarms(tmp_path: Path) -> None:
    """The row IS the evidence, so refusing it would suppress the observation.

    A nonzero `episodes_stuck` does not mean the counter lied — it means chunk
    5's reachability premise broke, and which premise broke is the finding.
    """
    row = good_row(episodes_solved=4, episodes_stuck=1)
    path = tmp_path / "iterations.jsonl"
    row["steps_minus_par_histogram"] = dict.fromkeys(STEPS_MINUS_PAR_BINS, 0) | {"0": 4}
    alarms = append_row(path, row)

    assert len(alarms) == 1
    assert alarms[0]["field"] == "episodes_stuck"
    assert alarms[0]["value"] == 1
    assert alarms[0]["expected"] == 0
    assert "ROUND-02" in alarms[0]["premises"], "the alarm must name candidate premises"

    written = read_rows(path)
    assert len(written) == 1, "the row must land — it is the evidence"
    assert written[0]["alarms"] == alarms, "the alarm travels IN the row"


def test_a_premise_zero_at_zero_raises_no_alarm() -> None:
    """The other polarity: the alarm must be capable of staying silent."""
    assert validate_row(good_row()) == []


def test_a_premise_zero_field_must_name_its_premises() -> None:
    with pytest.raises(SchemaError, match="MUST name its premises"):
        Field("x", int, "counter", "doc", zero_class="premise")


def test_only_a_premise_zero_may_name_premises() -> None:
    with pytest.raises(SchemaError, match="only a premise-dependent zero"):
        Field("x", int, "counter", "doc", zero_class="definitional", zero_premises="because")


def test_an_unknown_zero_species_is_refused() -> None:
    with pytest.raises(SchemaError, match="zero_class must be"):
        Field("x", int, "counter", "doc", zero_class="probably")


# ---------------------------------------------------------------------------
# The alarm census — the last leg: fired at write, carried in the row, surfaced
# at the run. Without this, an alarm written is an alarm stored.
# ---------------------------------------------------------------------------


def test_census_is_empty_when_nothing_fired(tmp_path: Path) -> None:
    """`{}` is the expectation line a run report asserts against: alarms: 0."""
    path = tmp_path / "iterations.jsonl"
    append_row(path, good_row(iteration=0))
    append_row(path, good_row(iteration=1))
    assert alarm_census(read_rows(path)) == {}


def test_census_counts_by_field_across_rows(tmp_path: Path) -> None:
    path = tmp_path / "iterations.jsonl"
    append_row(path, good_row(iteration=0))
    for i in (1, 2):
        row = good_row(iteration=i, episodes_solved=4, episodes_stuck=1)
        row["steps_minus_par_histogram"] = dict.fromkeys(STEPS_MINUS_PAR_BINS, 0) | {"0": 4}
        append_row(path, row)
    assert alarm_census(read_rows(path)) == {"episodes_stuck": 2}


def test_census_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    """The census reads what landed, not what a writer returned in memory."""
    path = tmp_path / "iterations.jsonl"
    row = good_row(episodes_solved=4, episodes_stuck=1)
    row["steps_minus_par_histogram"] = dict.fromkeys(STEPS_MINUS_PAR_BINS, 0) | {"0": 4}
    returned = append_row(path, row)
    from_disk = alarm_census(read_rows(path))
    assert from_disk == {"episodes_stuck": 1}
    assert len(returned) == 1


# ---------------------------------------------------------------------------
# F-42: era 4 — a redefined column and its new neighbours
# ---------------------------------------------------------------------------


def test_an_era_4_row_requires_the_f42_columns() -> None:
    row = good_row()
    del row["kl_improved_prior_step1_start"]
    with pytest.raises(SchemaError, match="kl_improved_prior_step1_start"):
        validate_row(row)


def test_an_era_3_row_validates_without_them() -> None:
    """The other polarity: M1's committed rows predate the columns, and their
    absence is computed from the era rather than asserted by the row."""
    row = good_row(schema_era=3)
    for name in (
        "entropy_visits_step1_start",
        "entropy_visits_step1_reached",
        "kl_improved_prior_step1_start",
        "kl_improved_prior_step1_reached",
        "ring_improved_truncated",
    ):
        del row[name]
    assert validate_row(row) == []


def test_the_redefined_target_entropy_says_what_it_used_to_mean() -> None:
    """A column redefined in place carries the reason its older values are not
    its quantity; "it did not exist yet" would be false (F-42)."""
    spec = field_map()["entropy_target_step1_start"]
    assert spec.since == 4 == SCHEMA_ERA
    assert spec.since_reason is not None and "visit share" in spec.since_reason


def test_a_nonzero_truncation_count_writes_and_alarms() -> None:
    alarms = validate_row(good_row(ring_improved_truncated=3))
    assert any(a["field"] == "ring_improved_truncated" and a["value"] == 3 for a in alarms)


def test_an_era_3_row_missing_the_redefined_column_is_refused() -> None:
    """The other polarity of era absence for a REDEFINED column: it existed in
    era 3 under its old meaning and was required then, so an era-3 row without
    it is missing a column — not excused by an era it did exist in."""
    row = good_row(schema_era=3)
    for name in (
        "entropy_visits_step1_start",
        "entropy_visits_step1_reached",
        "kl_improved_prior_step1_start",
        "kl_improved_prior_step1_reached",
        "ring_improved_truncated",
    ):
        del row[name]
    del row["entropy_target_step1_start"]
    with pytest.raises(SchemaError, match="entropy_target_step1_start"):
        validate_row(row)
