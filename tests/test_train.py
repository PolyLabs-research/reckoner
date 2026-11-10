"""Phase-1 warm start: the batch pipeline and its one optimisation.

The first gate here measures the component doing its central job (AGENTS.md §5,
the F-06 rider): the crop must *fire*, and it must change nothing. A crop test
that only checked the outputs would go green on a crop that never cropped —
which is F-06's shape exactly, one layer down.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reckoner.config import Config
from reckoner.dataset import anchored_data, data_path
from reckoner.model import Reckoner
from reckoner.train import (
    SupervisionSet,
    _crop_to_content,
    make_batch,
    rehearsal_split,
    train,
    train_on_ring,
)
from reckoner.valuegate import ValueHeadState
from reckoner.vocab import PAD

REPO = Path(__file__).resolve().parents[1]
DATA = anchored_data("phase1_train")
CFG = Config()

needs_data = pytest.mark.skipif(
    not (DATA / "meta.json").exists(), reason="phase-1 supervision not built"
)


# ---------------------------------------------------------------------------
# The crop: exact, and it must actually crop
# ---------------------------------------------------------------------------


@needs_data
def test_cropping_padding_is_exact() -> None:
    """Bit-identical outputs, and the crop demonstrably fired.

    ``encode`` pads to ``model.seq_len`` (512, sized for Phase-2 reachable
    states); Phase-1 supervision states top out at ``max_len 64``. Cropping the
    all-PAD tail is free because the trunk masks PAD, the pooling masks PAD, and
    positions are indexed from zero. "Free" is asserted here, not argued.
    """
    data = SupervisionSet(DATA)
    rng = random.Random(0)
    batch = make_batch(data, [rng.randrange(len(data)) for _ in range(16)], CFG)

    # Polarity one: the crop fired. Without this the equality below is vacuous.
    assert batch.width < CFG.model.seq_len, (
        f"crop did not fire: width {batch.width} == seq_len {CFG.model.seq_len}"
    )
    assert batch.tokens.shape[1] == batch.width

    padded = torch.full((batch.tokens.shape[0], CFG.model.seq_len), PAD, dtype=batch.tokens.dtype)
    padded[:, : batch.width] = batch.tokens

    model = Reckoner(CFG).eval()  # eval: dropout off, so this compares the maths
    with torch.no_grad():
        cropped_out = model(batch.tokens, batch.site_positions)
        full_out = model(padded, batch.site_positions)

    names = ("policy", "value", "steps")
    for name, cropped, full in zip(names, cropped_out, full_out, strict=True):
        assert torch.equal(cropped, full), f"{name} differs after cropping padding"


def test_crop_keeps_every_column_that_holds_content() -> None:
    """The other polarity: a batch that needs the full width keeps it."""
    tokens = torch.full((2, CFG.model.seq_len), PAD, dtype=torch.long)
    tokens[0, 0] = 5
    tokens[1, CFG.model.seq_len - 1] = 5  # content in the very last column
    sites = torch.zeros((2, CFG.model.max_sites), dtype=torch.long)
    assert _crop_to_content(tokens, sites).shape[1] == CFG.model.seq_len


def test_crop_never_cuts_below_a_site_position() -> None:
    """A site indexes into the trunk output; cropping past one would gather PAD.

    Sites cannot outrun their own tokens in practice, so this guards the
    invariant rather than an observed failure — which is the point of writing it
    while the invariant is cheap to state.
    """
    tokens = torch.full((1, CFG.model.seq_len), PAD, dtype=torch.long)
    tokens[0, :4] = 5
    sites = torch.zeros((1, CFG.model.max_sites), dtype=torch.long)
    sites[0, 0] = 300
    assert _crop_to_content(tokens, sites).shape[1] == 301


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


@needs_data
def test_batch_targets_are_legal_actions() -> None:
    """A one-hot policy target under a masked-off action trains toward nothing.

    ``policy_loss`` scores only legal actions, so a target on an illegal one was
    silently dropped rather than loudly rejected — the mask hid the defect it was
    meant to expose. This guarded the supervised path only; the ring path had the
    same hazard unguarded and lost 93% of its visit mass to it (F-41). Since F-41
    ``policy_loss`` itself raises on off-legal target mass; this remains as the
    check on the data as built, before any loss sees it.
    """
    data = SupervisionSet(DATA)
    rng = random.Random(1)
    batch = make_batch(data, [rng.randrange(len(data)) for _ in range(64)], CFG)
    chosen = batch.policy_target.argmax(dim=1)
    legal_at_target = batch.legal_mask.gather(1, chosen.unsqueeze(1)).squeeze(1)
    assert bool(legal_at_target.all()), (
        f"{int((~legal_at_target).sum())} of {len(chosen)} policy targets are illegal actions"
    )


@needs_data
def test_encode_failures_are_counted_never_cropped() -> None:
    """``StateTooLarge`` rows are skipped and counted, per chunk 6's semantics."""
    data = SupervisionSet(DATA)
    rng = random.Random(2)
    indices = [rng.randrange(len(data)) for _ in range(32)]
    batch = make_batch(data, indices, CFG)
    assert batch.skipped == len(indices) - int(batch.tokens.shape[0])


# ---------------------------------------------------------------------------
# Gate 13 — reproducible from seed, at a declared tolerance
# ---------------------------------------------------------------------------


@needs_data
def test_training_is_reproducible_from_seed() -> None:
    """**Declared tolerance: exact.** Not "close" — bit-identical loss curves.

    On CPU with every RNG seeded from config there is no source of nondeterminism
    to tolerate, so a tolerance would be a place for real drift to hide. If this
    ever needs one, that is a finding about the environment and gets written down
    rather than absorbed into an epsilon.

    Scope is declared too, because a gate must report what it covered: **5 steps
    at batch 16**, which exercises sampling, batch construction, the crop, both
    losses, clipping and the optimiser step. It does not cover long-run
    accumulation; the full run's reproducibility rests on this plus the seeded
    sampler, and a 2 x 83-minute repeat is not a `make test` cost.
    """
    data = SupervisionSet(DATA)
    cfg = Config()
    cfg.train.batch_size = 16

    def run() -> list[float]:
        torch.manual_seed(0)
        model = Reckoner(cfg)
        stats = train(model, data, cfg, steps=5, seed=0, log_every=0)
        return stats.losses

    first, second = run(), run()
    assert first == second, f"seed-identical runs diverged:\n  {first}\n  {second}"
    assert len(first) == 5
    # Both polarities: a different seed must NOT reproduce it, or the assertion
    # above would pass on a loop that ignores its inputs.
    torch.manual_seed(0)
    other = train(Reckoner(cfg), data, cfg, steps=5, seed=1, log_every=0).losses
    assert other != first, "a different seed produced an identical curve"


# ---------------------------------------------------------------------------
# Rehearsal — ported dormant, and "dormant" is asserted rather than assumed
# ---------------------------------------------------------------------------


def test_rehearsal_is_dormant_at_zero() -> None:
    """`rehearsal_frac = 0.0` must change nothing in the training path.

    "Dormant" untested means "untested" — the accepting case is that the split is
    a no-op, and it has to be asserted like any other.

    **This was `..._at_the_default` until M1-A4 §5 moved the default to 0.65.**
    The renaming matters: the property worth testing is that ZERO is inert, which
    is the governance condition behind F-31's wiring being a defect fix rather
    than a treatment change. Tying it to "the default" made it a test of a value
    that an amendment was always going to move, and it would have read as the
    lever breaking rather than as the default changing.
    """
    zero = replace(CFG, train=replace(CFG.train, rehearsal_frac=0.0))
    assert rehearsal_split(128, zero) == (0, 128)


def test_the_default_is_the_amended_value() -> None:
    """M1-A4 §5. Separate from the dormancy test above, because "zero is inert"
    and "the default is 0.65" are different claims and only one of them moves."""
    assert CFG.train.rehearsal_frac == 0.65
    assert rehearsal_split(128, CFG) == (83, 45)


def test_rehearsal_splits_when_armed() -> None:
    """The other polarity: the lever must do something when pulled, or it is not
    a lever that exists before it is needed — it is a key that does nothing."""
    cfg = Config()
    cfg.train.rehearsal_frac = 0.25
    assert rehearsal_split(128, cfg) == (32, 96)
    cfg.train.rehearsal_frac = 0.5
    assert rehearsal_split(10, cfg) == (5, 5)


def test_a_full_rehearsal_fraction_is_refused() -> None:
    """At 1.0 a Phase-2 iteration trains on no Phase-2 data, which is not rehearsal."""
    cfg = Config()
    cfg.train.rehearsal_frac = 1.0
    with pytest.raises(ValueError, match="not rehearsal"):
        rehearsal_split(128, cfg)


# ---------------------------------------------------------------------------
# Phase 2's training phase — the loop's central job, asserted
# ---------------------------------------------------------------------------


def a_filled_ring(cfg: Config, n: int = 8):
    from reckoner.dataset import anchored_data, training_problems
    from reckoner.replay import ReplayRing
    from reckoner.runner import run_iteration
    from reckoner.search import uniform_stub

    ring = ReplayRing(2048, cfg)
    problems = training_problems(anchored_data("train_100k"), n, seed=0)
    run_iteration(problems, uniform_stub(cfg), cfg, ring, sims=8, m=5, seed=0)
    return ring


needs_train_set = pytest.mark.skipif(
    not data_path("train_100k").exists(), reason="training set not generated"
)


@needs_train_set
def test_the_ring_carries_a_usable_state_not_just_an_outcome() -> None:
    """An earlier runner stored EMPTY token arrays, so every row was untrainable
    and `len(ring) > 0` still passed. Rider (a) at the ring boundary."""
    ring = a_filled_ring(CFG)
    record = ring.get(0)
    assert len(record["tokens"]) > 0
    assert record["n_sites"] > 0
    assert record["z"] in (-1, 0, 1)


@needs_train_set
def test_training_on_the_ring_moves_the_model() -> None:
    """The loop's central job. Without this, a loop that never learned would pass
    every plumbing check it had."""
    cfg = Config()
    ring = a_filled_ring(cfg)
    torch.manual_seed(0)
    model = Reckoner(cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stats = train_on_ring(model, ring, cfg, steps=2, seed=0)
    assert stats.steps == 2
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())


@needs_train_set
def test_the_value_head_trains_while_the_search_distrusts_it() -> None:
    """F-15: gating BOTH consumers by the declaration deadlocks the ratchet.

    No gradient means no accuracy means no switch means no gradient. The
    declaration governs what the search TRUSTS; the loss teaches regardless, or
    the door it guards can never open.
    """
    cfg = Config()
    ring = a_filled_ring(cfg)
    torch.manual_seed(0)
    model = Reckoner(cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_on_ring(model, ring, cfg, steps=2, seed=0, value_head=ValueHeadState(live=False))
    moved = [
        k
        for k in before
        if k.startswith("value_head") and not torch.equal(model.state_dict()[k], before[k])
    ]
    assert moved, "the W/D/L head received no gradient — the ratchet cannot ever be pulled"


@needs_train_set
def test_an_empty_ring_trains_nothing_rather_than_crashing() -> None:
    from reckoner.replay import ReplayRing

    cfg = Config()
    stats = train_on_ring(Reckoner(cfg), ReplayRing(16, cfg), cfg, steps=3, seed=0)
    assert stats.steps == 0


# ------------------------------------ the supervised gradient ARRIVES (F-31)


def _trained_weights(frac: float, steps: int = 4):
    """Train a fresh model on a fixed tiny ring at ``rehearsal_frac = frac``.

    Returns ``(weight_digest, stats)``. Same seeds throughout, so the ONLY thing
    that varies between calls is the rehearsal fraction.
    """
    import hashlib

    from reckoner.dataset import anchored_data, training_problems
    from reckoner.evaluate import model_evaluator
    from reckoner.replay import ReplayRing
    from reckoner.runner import run_iteration

    cfg = replace(CFG, train=replace(CFG.train, rehearsal_frac=frac))
    torch.manual_seed(11)
    scout = Reckoner(cfg)
    scout.eval()
    ring = ReplayRing(4096, cfg)
    problems = training_problems(anchored_data("train_100k"), 6, seed=3)
    run_iteration(problems, model_evaluator(scout, cfg, 0.0), cfg, ring, sims=4, m=4, seed=3)

    torch.manual_seed(99)
    model = Reckoner(cfg)
    stats = train_on_ring(model, ring, cfg, steps=steps, seed=5)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest(), stats


@needs_data
def test_rehearsal_is_bit_identically_inert_at_zero() -> None:
    """The governance condition (F-31). Wiring a dormant lever is a DEFECT FIX
    only if it changes nothing at the current fingerprint — and `rehearsal_frac`
    is 0.0 there. Proved by weight digest rather than argued.

    Two runs at 0.0 must agree with each other, which is also this probe's own
    reference-vector check: the first version of it hashed the checkpoint FILE,
    which embeds varying metadata, and was non-deterministic across identical
    code."""
    first, stats = _trained_weights(0.0)
    second, _ = _trained_weights(0.0)
    assert first == second, "the probe is not deterministic; it cannot judge inertness"
    assert stats.rehearsal_batches == 0, "a supervised batch was mixed at frac 0.0"


@needs_data
def test_the_supervised_gradient_actually_arrives() -> None:
    """The polarity the four `rehearsal_split` tests structurally cannot reach.

    They prove the split is COMPUTED. Nothing proved it is CONSUMED — which is
    exactly how a fully tested function stayed disconnected from the optimiser
    for two chunks (F-31). Passing unit tests on a function nobody calls read as
    coverage.

    So: same ring, same seeds, only the fraction differs. The weights must
    differ, because a supervised batch that changes no parameter is a supervised
    batch that was not trained on.
    """
    inert, inert_stats = _trained_weights(0.0)
    armed, armed_stats = _trained_weights(0.25)

    assert armed_stats.rehearsal_batches > 0, (
        "no batch carried a supervised share at frac 0.25 — the split is computed "
        "and discarded, which is the defect this test exists for"
    )
    assert inert_stats.rehearsal_batches == 0
    assert armed != inert, (
        "identical weights at frac 0.0 and 0.25: the supervised share reached no "
        "gradient. The lever moves the fingerprint and nothing else."
    )
