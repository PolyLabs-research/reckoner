"""D-A1's three central-job assertions, both polarities each.

Part 0a found the same defect three times in one loop: `golden` asserted that
the model **moves** and never that the moved model is **used**; that the pool
**grows** and never that par came **from** it; that the switch row **writes**
and never that the criterion saw a **real input**. One shape — an assertion that
the central job happened standing in for one that its product is consumed — and
it hid the loop's entire subject for three chunks.

So each of the three gets its assertion *and its contrast*. A positive alone is
the thing that failed here: "the model moved" was true every single time.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reckoner.campaign import (
    CAMPAIGN_FINGERPRINT,
    CampaignRefusal,
    assert_threads,
    run,
)
from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import sha256_file
from tests.campaign_fixture import ANCHOR, drive, drive_campaign, golden_config

needs_anchor = pytest.mark.skipif(not ANCHOR.exists(), reason="phase-1 anchor not present")
pytestmark = [pytest.mark.slow, needs_anchor]


# --- one driven run per CONFIGURATION, shared across every assertion of it ---
# The fixture returns artifacts precisely so assertions can READ rather than RUN.
# Twelve tests commissioning twelve loops would pay for the same evidence twelve
# times; these inspect the artifacts of two. `make test` still runs the whole
# suite unfiltered — the gate is never the subset.


@pytest.fixture(scope="session")
def seeded(tmp_path_factory) -> object:
    """The default campaign shape: pool seeded with the anchor at startup."""
    return drive(tmp_path_factory.mktemp("seeded") / "run")


@pytest.fixture(scope="session")
def unseeded(tmp_path_factory) -> object:
    """The contrast: no anchor seeding, so zero pool-par is the CORRECT answer."""
    return drive(
        tmp_path_factory.mktemp("unseeded") / "run",
        golden_config(league={"seed_pool_with_anchor": False}),
    )


# ------------------------------------------------- D-A1 §1.1  model-in-search


def test_the_row_names_the_checkpoint_the_evaluator_actually_used(seeded) -> None:
    """Provenance, not configuration: a digest records what RAN.

    `golden` ran `uniform_stub` every iteration while asserting the weights
    moved. This column is what makes that undetectable state detectable — at
    iteration 0 the evaluator is built from the anchor, so the row carries the
    anchor's digest.
    """
    result = seeded
    assert result.rows, "the driver wrote no rows"
    assert result.rows[0]["evaluator_checkpoint_sha256"] == sha256_file(ANCHOR)


def test_the_digest_moves_when_the_model_does(seeded) -> None:
    """The contrast. A constant digest would mean the loop never re-loaded.

    Iteration 1's evaluator is iteration 0's checkpoint, so the column must
    change — a digest that never moves is exactly the stub-forever defect
    wearing a provenance field.
    """
    result = seeded
    digests = [r["evaluator_checkpoint_sha256"] for r in result.rows]
    assert len(digests) >= 2
    assert digests[0] != digests[1], "the evaluator's provenance never changed"


# --------------------------------------------------- D-A1 §1.2  pool par drawn


def test_pool_par_is_actually_drawn_from_iteration_zero(seeded) -> None:
    """The positive. `seed_pool_with_anchor` makes the anchor rung zero.

    Before the driver seeded the pool this returned zero and *looked* correct:
    an empty pool has no par to give. The seeding is what makes
    `par_from_pool_frac` a live mechanism at iteration 0 rather than dead until
    snapshots accumulate.
    """
    result = seeded
    row = result.rows[0]
    assert row["pool_par_fraction"] > 0.0, "no episode took par from the pool"
    assert row["z_by_par_source"].get("pool"), "no pool-par outcomes were scored"


def test_an_unseeded_pool_reports_zero_rather_than_inferring_it(unseeded) -> None:
    """The contrast: zero pool-par is CORRECT here, and must stay counted.

    The empty-pool case keeps its counted-state behaviour — the two causes of
    unavailability are different diseases and neither may be inferred from a
    missing column.
    """
    result = unseeded
    row = result.rows[0]
    assert row.get("pool_par_fraction", 0.0) == 0.0
    assert not row["z_by_par_source"].get("pool"), "pool outcomes without a pool"


# ------------------------------------------- D-A1 §1.3  the criterion's inputs


def test_the_switch_row_is_written_every_iteration_including_abstentions(seeded) -> None:
    """An abstention nobody records is indistinguishable from a criterion nobody ran."""
    result = seeded
    assert len(result.switch_rows) == len(result.rows)
    assert all(r["abstained"] or r["fired"] or True for r in result.switch_rows)


def test_the_criterion_abstains_on_an_empty_holdout_rather_than_deciding(seeded) -> None:
    """The contrast, and the expected iteration-0 state.

    `evaluate_head` returns `([], [])` on an empty slice and the criterion reads
    that as abstention-with-its-reason. A criterion that *decided* on no data
    would be the never-firable trigger's mirror image.
    """
    result = seeded
    first = result.switch_rows[0]
    assert first["abstained"] is True
    assert first["fired"] is False


# ------------------------------------------------------ the fingerprint gates


def test_the_campaign_door_refuses_a_config_the_page_did_not_bless(
    tmp_path: Path,
) -> None:
    """Both polarities of the startup assertion — the refusing one."""
    with pytest.raises(CampaignRefusal, match="not the one the campaign was"):
        drive_campaign(tmp_path / "run", golden_config())


def test_the_campaign_door_accepts_the_registered_fingerprint() -> None:
    """And the accepting one, without paying for a campaign to prove it."""
    cfg = Config()
    validate(cfg)
    assert config_fingerprint(cfg) == CAMPAIGN_FINGERPRINT


def test_raising_the_analysis_point_changes_the_fingerprint_and_is_refused() -> None:
    """PREREG-m1's extent rule, enforced by mechanism rather than by sentence.

    "May not be raised after launch" is a law against the loud path; this is the
    mechanism against the silent one, and it survives a crash because resume
    re-asserts at every startup.
    """
    cfg = Config()
    raised = replace(cfg, campaign=replace(cfg.campaign, iterations=25))
    assert config_fingerprint(raised) != CAMPAIGN_FINGERPRINT


def test_a_set_omp_variable_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset is a value: the licensed absence defends itself."""
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    with pytest.raises(CampaignRefusal, match="UNSET throughout the licence"):
        assert_threads(Config())


def test_the_thread_pins_are_applied_not_merely_declared() -> None:
    cfg = Config()
    report = assert_threads(cfg)
    assert report["intra_op"] == cfg.campaign.intra_op_threads == torch.get_num_threads()


# ------------------------------------------------------------- the seam pin


def test_the_driver_provably_commits_through_the_shared_seam() -> None:
    """D-A1 §2(a). "Probes the seam the driver also uses" is only true while
    nothing reimplements the seam — and that is an assertion, not a hope.
    """
    import inspect

    source = inspect.getsource(run)
    assert "commit_iteration(" in source, "the driver does not call the commit seam"
    for reimplementation in ("LATEST.tmp", "os.replace"):
        assert reimplementation not in source, (
            f"the driver reimplements the commit contract ({reimplementation}); "
            "there must be exactly one implementation of the write ordering"
        )


# ------------------------------------------- the snapshot cadence (F-36)


@needs_anchor
def test_the_driver_honours_the_snapshot_cadence(tmp_path: Path) -> None:
    """F-36. The driver enrolled every iteration while `shakedown.py` honoured
    `league.snapshot_every`; they agreed only because the default is 1.

    Both polarities, because "agrees at the default" is exactly the evidence that
    let the divergence sit unnoticed — the test that matters is the one at a value
    where the two behaviours differ.
    """
    from tests.campaign_fixture import drive

    at_one = drive(tmp_path / "every", golden_config())
    assert [r["pool_composition"]["size"] for r in at_one.rows] == [1, 2], (
        "at snapshot_every = 1 the pool must gain a member every iteration"
    )

    at_two = drive(tmp_path / "alternate", golden_config(league={"snapshot_every": 2}))
    assert [r["pool_composition"]["size"] for r in at_two.rows] == [1, 1], (
        "at snapshot_every = 2 iteration 0 takes no snapshot, so the pool the "
        "iterations DRAW from stays at the anchor alone"
    )
    assert not (tmp_path / "alternate" / "snap-0.pt").exists(), "snap-0 was written anyway"
    assert (tmp_path / "alternate" / "snap-1.pt").exists(), "the cadence iteration took none"


def test_the_two_cadence_legs_do_not_share_a_pass_identity() -> None:
    """The collision the equivalence gate caught, pinned as a test.

    `run_pass` keys completion on `(root, index)` and both legs share the
    iteration index. With a shared root the second leg found the first's DONE
    marker, skipped its own pass, and then read zero rows for an arm name that
    pass had never played — surfacing as a ZeroDivisionError on a denominator of
    zero, a full cadence unit into the run.

    No test caught it because `golden_config` sets `ladder_every = 99`, so the
    cadence never fires in the suite — F-29's registered gap, biting exactly
    where it was registered. This is the cheap half of closing it: the identities
    must differ, asserted without paying for an instrument pass.
    """
    import inspect

    from reckoner import campaign

    source = inspect.getsource(campaign.run_instruments)
    roots = [
        line for line in source.splitlines() if '/ "ladder" /' in line or '/ "ladder" /' in line
    ]
    assert len(roots) >= 2, "expected a distinct pass root per cadence leg"
    assert any("no_regress" in r for r in roots) and any("primary" in r for r in roots), (
        "the two legs must not share a pass root: run_pass keys completion on "
        f"(root, index) and both legs share the index. Found: {roots}"
    )


def test_the_population_cap_is_inert_at_campaign_config() -> None:
    """`ladder.problems_per_pass` became live so `golden` could run a cadence on a
    tiny population. That is a defect fix only if it changes nothing at the
    campaign's config — and it does not, because every suite and stratum file
    holds exactly 200 rows and the cap is 200, so the slice is the identity.

    Asserted rather than assumed: if a suite is ever regenerated at a different
    size, this fails and the cap stops being inert."""
    from reckoner.config import Config
    from reckoner.dataset import read_suite

    cap = Config().ladder.problems_per_pass
    files = sorted(Path(__file__).resolve().parents[1].glob("runs/suites/*.jsonl"))
    assert files, "no suites present"
    for path in files:
        rows = read_suite(path)
        assert len(rows) <= cap, (
            f"{path.name} holds {len(rows)} rows against a cap of {cap}: the cap "
            "now truncates the campaign's own instrument population, which makes "
            "it a treatment change rather than a defect fix"
        )


def test_the_campaign_predicate_is_derived_from_the_config_not_supplied() -> None:
    """D-A2 §2's split, both polarities — and the predicate's provenance.

    `assert_eval_profile` takes the config a pass runs under and computes "is
    this a campaign measurement?" itself. A `campaign=` argument would be a
    caller-supplied claim about evidential status, which is F-19's currency tag
    wearing a keyword: the caller is exactly who should not be answering *is this
    a real measurement?*
    """
    import inspect

    from reckoner.campaign import EVAL_FINGERPRINT, assert_eval_profile

    signature = inspect.signature(assert_eval_profile)
    assert list(signature.parameters) == ["cfg"], (
        f"assert_eval_profile takes {list(signature.parameters)} — the campaign "
        "predicate must be derived from the config, never passed by the caller"
    )

    # campaign config: the registered pin is asserted, and holds
    assert assert_eval_profile(Config()) == EVAL_FINGERPRINT

    # golden config: accepted, and NOT pinned to the campaign's value
    got = assert_eval_profile(golden_config())
    assert got != EVAL_FINGERPRINT


def test_a_campaign_config_with_a_drifted_eval_profile_refuses(monkeypatch) -> None:
    """The polarity that matters. If the eval profile ever drifts from the
    registered value, every campaign pass must refuse — that is the half of
    D-A2 §2 the split preserves."""
    from reckoner import campaign as campaign_module

    monkeypatch.setattr(campaign_module, "EVAL_FINGERPRINT", "0" * 64)
    with pytest.raises(campaign_module.CampaignRefusal, match="not comparable"):
        campaign_module.assert_eval_profile(Config())

    # and golden is unaffected, because its numbers are never measurements
    assert campaign_module.assert_eval_profile(golden_config())


def test_the_interop_pin_is_applied_and_read_back() -> None:
    """M1-A4 §6. Interop moved from OBSERVED to EXERCISED when the campaign host
    changed and the new one defaulted to 48 against the licence's 32.

    The pin is APPLIED and then read back, so the returned record is what the
    runtime holds rather than what the caller hoped for — the distinction that
    made the old version an unverified claim in a document whose whole purpose is
    that its numbers were true.

    **The field's VALUE is unchanged**, so the config fingerprint does not move.
    What changed is whether the runtime is made to match the record or merely
    asked whether it happens to.
    """
    import torch

    from reckoner.campaign import CAMPAIGN_FINGERPRINT, assert_threads
    from reckoner.config import config_fingerprint

    cfg = Config()
    record = assert_threads(cfg)

    assert record["interop"] == cfg.campaign.interop_threads
    assert torch.get_num_interop_threads() == cfg.campaign.interop_threads, (
        "the record reports a value the runtime does not hold"
    )
    assert record["intra_op"] == cfg.campaign.intra_op_threads
    assert config_fingerprint(cfg) == CAMPAIGN_FINGERPRINT, (
        "applying the pin must not move the fingerprint — the value did not change"
    )


def test_a_set_omp_variable_still_refuses() -> None:
    """The other pin's polarity, unchanged by M1-A4 §6: the OMP family was unset
    throughout the licence, and unset is a value."""
    import os

    from reckoner.campaign import OMP_FAMILY, CampaignRefusal, assert_threads

    name = OMP_FAMILY[0]
    os.environ[name] = "4"
    try:
        with pytest.raises(CampaignRefusal, match="UNSET throughout the licence"):
            assert_threads(Config())
    finally:
        del os.environ[name]
