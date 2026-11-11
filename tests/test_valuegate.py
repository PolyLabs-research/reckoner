"""The value-head declaration, its two consumers, and the switch detector.

The detector gates automation, so it gets two-sided validation before live use —
the clamp-detector precedent. A criterion nobody has watched refuse is a
criterion that will fire on anything; a criterion nobody has watched fire is one
that has disabled the lever it guards. This is a **ratcheting** detector, so both
ends matter and the false-fire rate is priced.
"""

from __future__ import annotations

import random

import pytest

from reckoner.dataset import anchored_data
from reckoner.valuegate import (
    MARGIN,
    MIN_CLASS_SUPPORT,
    ValueGateError,
    ValueHeadState,
    balanced_accuracy,
    class_census,
    consider_switch,
    switch_criterion,
    value_contribution,
)

SUPPORT = MIN_CLASS_SUPPORT


def two_class(draws: int = SUPPORT * 3, losses: int = SUPPORT) -> list[int]:
    """A draw-heavy two-class slice — this project's predicted early regime."""
    return [0] * draws + [-1] * losses


# ---------------------------------------------------------------------------
# One declaration, two consumers
# ---------------------------------------------------------------------------


def test_an_untrained_head_contributes_nothing() -> None:
    assert value_contribution(ValueHeadState(live=False)) == 0.0


def test_a_trained_head_contributes_fully() -> None:
    assert value_contribution(ValueHeadState(live=True)) == 1.0


def test_both_consumers_read_the_same_number() -> None:
    """A config flag consulted by the loss and forgotten by the search is the
    two-gates-one-name hazard. There is one function; this pins that."""
    state = ValueHeadState(live=False)
    assert value_contribution(state) == value_contribution(state) == 0.0
    live = ValueHeadState(live=True)
    assert value_contribution(live) == value_contribution(live) == 1.0


def test_the_head_starts_untrusted() -> None:
    """Spec §5 leaves W/D/L at loss weight 0, so at iteration 0 it is noise."""
    assert ValueHeadState().live is False


# ---------------------------------------------------------------------------
# The four-tuple's slots keep their definitions
# ---------------------------------------------------------------------------


def test_floor_is_zero_and_marked_uninformative() -> None:
    """An accuracy has no structural minimum — an anti-correlated head scores
    below the trivial model. The slot is filled, not merged into the null."""
    result = switch_criterion(two_class(), [0] * (SUPPORT * 4))
    assert result["floor"] == 0.0
    assert result["floor_is_uninformative"] is True


def test_null_is_one_over_k_whatever_the_imbalance() -> None:
    """This is what makes the bar firable in a draw-heavy regime."""
    balanced = switch_criterion([0] * SUPPORT + [-1] * SUPPORT, [0] * (SUPPORT * 2))
    skewed = switch_criterion([0] * (SUPPORT * 9) + [-1] * SUPPORT, [0] * (SUPPORT * 10))
    assert balanced["null"] == pytest.approx(0.5)
    assert skewed["null"] == pytest.approx(0.5), "the null must not move with imbalance"
    assert balanced["threshold"] == skewed["threshold"] == pytest.approx(0.5 + MARGIN)


def test_the_class_census_is_recorded_so_k_is_data() -> None:
    result = switch_criterion(two_class(), [0] * (SUPPORT * 4))
    assert result["class_census"] == {-1: SUPPORT, 0: SUPPORT * 3}
    assert result["k_classes_with_support"] == 2


def test_a_third_class_moves_the_null_and_the_bar() -> None:
    """Once pool par exists, z can be +1 and K becomes 3."""
    labels = [0] * SUPPORT + [-1] * SUPPORT + [1] * SUPPORT
    result = switch_criterion(labels, [0] * (SUPPORT * 3))
    assert result["k_classes_with_support"] == 3
    assert result["null"] == pytest.approx(1 / 3)
    assert result["threshold"] == pytest.approx(1 / 3 + MARGIN)


def test_an_empty_held_out_set_is_refused() -> None:
    with pytest.raises(ValueGateError, match="cannot be evaluated on nothing"):
        class_census([])


def test_a_misaligned_pair_is_refused() -> None:
    with pytest.raises(ValueGateError, match="misaligned"):
        switch_criterion([0, 0], [0])


# ---------------------------------------------------------------------------
# The detector, both ends — it must fire, and it must refuse
# ---------------------------------------------------------------------------


def test_a_perfect_head_clears() -> None:
    labels = two_class()
    assert switch_criterion(labels, labels)["clears"] is True


def test_a_constant_head_scores_exactly_the_null_and_does_not_clear() -> None:
    """The collapsed-bar sentinel. A constant predictor ties 1/K by construction,
    so if this ever passes the bar has collapsed onto its null."""
    labels = two_class()
    result = switch_criterion(labels, [0] * len(labels))
    assert result["measured"] == pytest.approx(result["null"])
    assert result["clears"] is False


def test_a_random_head_does_not_clear() -> None:
    rng = random.Random(0)
    labels = two_class()
    guesses = [rng.choice([0, -1]) for _ in labels]
    assert switch_criterion(labels, guesses)["clears"] is False


def test_the_draw_heavy_regime_is_firable_at_all() -> None:
    """**The correction's whole point.** Under raw accuracy a 75/25 split demanded
    0.85 and a 98/2 split demanded 1.08 — unreachable, so the switch could never
    fire and the lever would be disabled by its own qualifying exam.

    Under balanced accuracy the same slice is cleared by a head that is merely
    *good*, not perfect: 90% recall on each class.
    """
    labels = two_class()
    rng = random.Random(1)
    predictions = [cls if rng.random() < 0.9 else (0 if cls == -1 else -1) for cls in labels]
    result = switch_criterion(labels, predictions)
    assert result["measured"] > result["threshold"]
    assert result["clears"] is True


# ---------------------------------------------------------------------------
# Abstention — "not evaluable yet" is not "evaluated and refused"
# ---------------------------------------------------------------------------


def test_a_thin_minority_class_abstains_rather_than_judging() -> None:
    """Balanced accuracy on two minority samples is not a measurement — minority
    recall can only be 0, 0.5 or 1."""
    labels = [0] * 500 + [-1] * 2
    result = switch_criterion(labels, labels)
    assert result["evaluable"] is False
    assert result["clears"] is False, "a perfect head must still not clear on 2 samples"
    assert result["smallest_class_support"] == 2


def test_abstention_is_recorded_distinctly_from_failure() -> None:
    labels = [0] * 500 + [-1] * 2
    state, event = consider_switch(ValueHeadState(), labels, labels, iteration=4)
    assert state.live is False
    assert event["abstained"] is True and event["fired"] is False

    enough = two_class()
    _, judged = consider_switch(ValueHeadState(), enough, [0] * len(enough), iteration=4)
    assert judged["abstained"] is False and judged["fired"] is False


def test_support_at_the_boundary_is_evaluable() -> None:
    labels = [0] * SUPPORT + [-1] * SUPPORT
    assert switch_criterion(labels, labels)["evaluable"] is True


# ---------------------------------------------------------------------------
# Ratcheting — fires once, never flickers
# ---------------------------------------------------------------------------


def test_it_fires_once_and_records_the_event() -> None:
    labels = two_class()
    state, event = consider_switch(ValueHeadState(), labels, labels, iteration=7)
    assert state.live is True
    assert state.switched_at_iteration == 7
    assert event["fired"] is True
    assert event["measured"] == pytest.approx(1.0)


def test_a_live_head_stays_live_even_if_the_criterion_would_now_fail() -> None:
    """Flickering would make consecutive iterations incomparable — the corpus each
    produces would be denominated in a different search."""
    labels = two_class()
    state, _ = consider_switch(ValueHeadState(), labels, labels, iteration=3)
    later, event = consider_switch(state, labels, [0] * len(labels), iteration=9)
    assert later.live is True
    assert later.switched_at_iteration == 3, "the ratchet must not re-date itself"
    assert event["fired"] is False and event["already_live"] is True


def test_a_failing_criterion_leaves_the_head_untrusted() -> None:
    labels = two_class()
    state, event = consider_switch(ValueHeadState(), labels, [0] * len(labels), iteration=1)
    assert state.live is False
    assert event["fired"] is False
    assert value_contribution(state) == 0.0


def test_balanced_accuracy_is_mean_per_class_recall() -> None:
    labels = [0, 0, 0, -1]
    assert balanced_accuracy(labels, [0, 0, 0, 0]) == pytest.approx(0.5)
    assert balanced_accuracy(labels, labels) == pytest.approx(1.0)
    assert balanced_accuracy(labels, [-1, -1, -1, -1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The K == 1 corner — found by the shakedown, not by review
# ---------------------------------------------------------------------------


def test_a_single_class_slice_abstains_with_no_variance_named() -> None:
    """Every episode drew. There is nothing for a head to know.

    Balanced accuracy is trivially 1.0 for a constant here and 1/K + margin is
    1.15 — unreachable. Refusing is correct; refusing *through an unreachable
    bar* would file as "evaluated and refused" when the truth is "there was
    nothing to evaluate", which is the never-firable shape in a new corner.
    """
    labels = [0] * (SUPPORT * 2)
    result = switch_criterion(labels, labels)
    assert result["k_classes_with_support"] == 1
    assert result["evaluable"] is False
    assert result["abstain_reason"].startswith("no_variance")
    assert result["clears"] is False


def test_the_two_abstention_causes_are_distinguishable() -> None:
    """Different diseases, different fixes: wait for the policy to start losing,
    versus wait for more episodes."""
    no_variance = switch_criterion([0] * 500, [0] * 500)
    thin = switch_criterion([0] * 500 + [-1] * 2, [0] * 502)
    assert no_variance["abstain_reason"].startswith("no_variance")
    assert thin["abstain_reason"].startswith("thin_minority")


def test_two_well_supported_classes_are_evaluable() -> None:
    """The accepting case — otherwise 'abstain' would just mean 'always'."""
    labels = two_class()
    result = switch_criterion(labels, labels)
    assert result["evaluable"] is True
    assert result["abstain_reason"] is None


# ---------------------------------------------------------------------------
# The evaluation slice — held out from training, in-distribution by design
# ---------------------------------------------------------------------------

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
needs_train_set = pytest.mark.skipif(
    not (anchored_data("train_100k")).exists(), reason="training set not generated"
)


def _filled_ring(cfg):
    from reckoner.dataset import anchored_data, training_problems
    from reckoner.replay import ReplayRing
    from reckoner.runner import run_iteration
    from reckoner.search import uniform_stub

    ring = ReplayRing(2048, cfg)
    problems = training_problems(anchored_data("train_100k"), 20, seed=0)
    run_iteration(problems, uniform_stub(cfg), cfg, ring, sims=8, m=5, seed=0)
    return ring


@needs_train_set
def test_the_holdout_is_deterministic_in_slot_and_seed() -> None:
    """A holdout that re-rolls is a holdout the model has seen."""
    from reckoner.config import Config

    cfg = Config()
    ring = _filled_ring(cfg)
    first = ring.holdout(0.1, seed=0)
    assert first == ring.holdout(0.1, seed=0)
    assert first != ring.holdout(0.1, seed=1)


@needs_train_set
def test_training_never_samples_a_held_out_row() -> None:
    """The head must not be graded on rows it fit — it would clear its own bar
    by memorising them."""
    import torch

    from reckoner.config import Config
    from reckoner.model import Reckoner
    from reckoner.train import train_on_ring

    cfg = Config()
    ring = _filled_ring(cfg)
    reserved = ring.holdout(cfg.train.ring_holdout_frac, seed=0)
    assert reserved, "the holdout is empty — the split is not splitting"

    torch.manual_seed(0)

    # A ring whose ONLY rows are held out must train zero steps.
    class OnlyHeldOut:
        def __init__(self, inner, reserved):
            self._inner = inner
            self._reserved = reserved

        def __len__(self):
            return len(self._inner)

        def holdout(self, frac, seed=0):
            return set(range(len(self._inner)))

        def get(self, i):
            return self._inner.get(i)

        def era_absent(self, name):
            # F-53/F-54: train_on_ring asks the ring for stale records before
            # anything else. Delegated, so the real ring's answer (none) is the
            # one that lets this reach the holdout logic under test.
            return self._inner.era_absent(name)

    stats = train_on_ring(Reckoner(cfg), OnlyHeldOut(ring, reserved), cfg, steps=3, seed=0)
    assert stats.steps == 0, "training consumed rows reserved for evaluation"


@needs_train_set
def test_the_head_is_evaluated_on_held_out_rows_in_distribution() -> None:
    """In-distribution is the POINT, not an oversight.

    F-09 was about a generalisation claim. This criterion asks whether the search
    should trust the head on the states the search actually visits — an
    in-distribution question by construction.
    """
    import torch

    from reckoner.config import Config
    from reckoner.model import Reckoner
    from reckoner.valuegate import evaluate_head

    cfg = Config()
    ring = _filled_ring(cfg)
    held = sorted(ring.holdout(cfg.train.ring_holdout_frac, seed=0))
    torch.manual_seed(0)
    labels, predictions = evaluate_head(Reckoner(cfg), ring, cfg, held)
    assert len(labels) == len(predictions) > 0
    assert all(z in (0, 1, 2) for z in labels), "labels are W/D/L class indices"


def test_evaluating_an_empty_slice_returns_nothing_rather_than_raising() -> None:
    from reckoner.config import Config
    from reckoner.model import Reckoner
    from reckoner.valuegate import evaluate_head

    assert evaluate_head(Reckoner(Config()), None, Config(), []) == ([], [])
