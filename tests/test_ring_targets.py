"""F-41: the ring's policy target is the distribution search actually produced.

The runner writes the ring and ``train.ring_batch`` reads it into the dense target
the policy loss consumes. The two ran together in tests (``a_filled_ring`` into
``train_on_ring``), but nothing compared the target ``ring_batch`` built with the
search that produced it — and they disagreed about what ``visit_actions`` holds: the runner stored root **slot** positions
(indices into ``legal_actions(expr)``) and ``ring_batch`` read them as **dense
action indices** (``rule * max_sites + site``). Slot j can only ever reach cell j,
which is ``eval_add`` at site j, so most stored visit mass landed on illegal cells
and the policy loss dropped it without a sound (`FINDINGS.md` F-41).

These tests run the pair end to end on the real path: real searches,
``run_iteration`` writing through ``runner._settle``, ``ring_batch`` reading.
Ground truth is each record's own ``SearchResult``, captured at ``_settle`` in the
order the original appends (the same capture as
``scripts/f41_ring_target_probe.py``). No gitignored data: the problems come from
the generator, so this runs in a clean clone.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from reckoner import runner
from reckoner.config import Config
from reckoner.generator import emit, label
from reckoner.model import N_RULES, IllegalTargetError, StateTooLarge, action_index, encode
from reckoner.replay import ReplayRing
from reckoner.search import SearchResult, uniform_stub
from reckoner.train import ring_batch

CFG = Config()
MAX_SITES = CFG.model.max_sites

#: (template, seed) pairs spanning all three goals and depths 2-4. Fixed, so the
#: fixture is the same states every run.
FIXTURE = (
    ("solve_two_terms", 1),
    ("solve_constant", 2),
    ("simplify_with_products", 3),
    ("simplify_two_vars", 4),
    ("eval_mixed", 5),
    ("eval_product", 6),
)


def generated_problems() -> list:
    problems = []
    for name, seed in FIXTURE:
        problem = label(emit(name, random.Random(seed)), CFG)
        assert problem is not None, f"{name}/{seed} did not label; pick another seed"
        problems.append(problem)
    return problems


def ring_with_truth(
    monkeypatch, *, ring_format: int | None = None
) -> tuple[ReplayRing, dict[int, SearchResult]]:
    """A ring filled by the real runner, and the SearchResult behind each record.

    ``ring_format`` stamps the records as an older writer would have, to build
    the stale ring a pre-fix run left on disk.
    """
    ring = (
        ReplayRing(4096, CFG)
        if ring_format is None
        else ReplayRing(4096, CFG, ring_format=ring_format)
    )
    truth: dict[int, SearchResult] = {}
    original = runner._settle

    def capturing(stats, ring_, e, trail, cfg_, *, solved, capped):
        kept = []
        for result, expr, _steps in trail[e.index]:
            try:
                encode(e.problem, expr, cfg_)
            except StateTooLarge:
                continue  # the runner skips these too, so they have no record
            kept.append(result)
        start = len(ring_)
        original(stats, ring_, e, trail, cfg_, solved=solved, capped=capped)
        assert len(ring_) - start == len(kept), "capture lost alignment with the appends"
        truth.update({start + k: result for k, result in enumerate(kept)})

    monkeypatch.setattr(runner, "_settle", capturing)
    runner.run_iteration(generated_problems(), uniform_stub(CFG), CFG, ring, sims=16, seed=0)
    return ring, truth


def expected_target(result: SearchResult, width: int) -> torch.Tensor:
    """The dense target the search's top-``width`` visits define, in action coordinates."""
    dense = torch.zeros(N_RULES * MAX_SITES)
    order = np.argsort(-result.visits)[:width]
    total = float(result.visits[order].sum())
    for slot in order:
        count = int(result.visits[slot])
        if count > 0:
            dense[action_index(*result.actions[int(slot)], MAX_SITES)] = float(count / total)
    return dense


def test_every_ring_target_is_the_search_distribution_in_action_coordinates(monkeypatch) -> None:
    """**The writer/reader pair doing its central job** (rider (a)).

    For every record, the dense target ``ring_batch`` builds must equal the
    search's own visit distribution placed on the cells of the actions it
    visited — exactly, cell for cell.

    It must also be able to FAIL for F-41. Under the slot reading, a record trains
    the right cells only if every visited slot j holds action ``(0, j)``. The
    fixture is required to contain records where that is false, and the count is
    published in the assertion so the gate says what it covered.

    Coverage when written (2026-11-10): 58 records, 55 of them discriminating;
    pars 1-4 (1 / 2 / 27 / 28); n_legal 1 to 52, so the ring's 16-slot cap is
    exercised. Against the pre-fix writer this test failed on record 0, whose one
    legal action ``(3, 0)`` belongs in cell 576 and was trained in cell 0.
    """
    ring, truth = ring_with_truth(monkeypatch)
    width = ring.visit_actions.shape[1]
    assert len(ring) == len(truth) >= 20, f"fixture produced only {len(ring)} records"

    discriminating = 0
    for i in range(len(ring)):
        result = truth[i]
        batch = ring_batch(ring, [i], CFG)
        assert batch is not None and batch.skipped == 0
        assert torch.equal(batch.policy_target[0], expected_target(result, width)), f"record {i}"

        visited = [int(s) for s in np.flatnonzero(result.visits)]
        discriminating += any(action_index(*result.actions[s], MAX_SITES) != s for s in visited)

    # The other polarity: on these records the slot reading trains other cells.
    assert discriminating >= 0.8 * len(ring), (
        f"only {discriminating}/{len(ring)} records separate action indices from slot "
        "positions; the fixture cannot tell the fix from F-41"
    )


def test_every_ring_target_is_entirely_legal(monkeypatch) -> None:
    """No stored visit mass lands outside the legal set, so none can be dropped."""
    ring, _truth = ring_with_truth(monkeypatch)
    batch = ring_batch(ring, list(range(len(ring))), CFG)
    assert batch is not None
    off_legal = batch.policy_target * (~batch.legal_mask)
    assert float(off_legal.sum()) == 0.0
    sums = batch.policy_target.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_the_slot_reading_is_refused_by_the_loss_not_absorbed(monkeypatch) -> None:
    """The F-41 target, rebuilt deliberately, must now raise rather than train.

    Builds what the HEAD writer produced (stored slot j trained as cell j) for the
    same records and feeds it to the policy loss with the true legal mask. Before
    F-41 the loss zeroed the illegal mass and carried on; now it refuses. This is
    the detector's live trigger, exercised on the defect it exists for.
    """
    from reckoner.model import policy_loss

    ring, truth = ring_with_truth(monkeypatch)
    width = ring.visit_actions.shape[1]
    indices = list(range(len(ring)))
    batch = ring_batch(ring, indices, CFG)
    assert batch is not None

    slot_targets = torch.zeros_like(batch.policy_target)
    for row, i in enumerate(indices):
        result = truth[i]
        order = np.argsort(-result.visits)[:width]
        total = float(result.visits[order].sum())
        for slot in order:
            if result.visits[slot] > 0:
                slot_targets[row, int(slot)] = float(result.visits[slot] / total)

    logits = torch.zeros_like(batch.policy_target)
    with pytest.raises(IllegalTargetError, match="F-41"):
        policy_loss(logits, slot_targets, batch.legal_mask)
    policy_loss(logits, batch.policy_target, batch.legal_mask)  # the fixed target passes


# ---------------------------------------------------------------------------
# A stale ring is refused where training happens, not only at resume
# ---------------------------------------------------------------------------


def _ring_only_cfg() -> Config:
    """No supervision rows, so no gitignored data: the ring is the only source."""
    from dataclasses import replace

    return replace(CFG, train=replace(CFG.train, rehearsal_frac=0.0))


def test_train_on_ring_refuses_a_ring_holding_pre_f41_records(monkeypatch) -> None:
    """Every record of a pre-fix ring is era-absent, so every batch was empty and
    ``train_on_ring`` returned ``steps == 0`` without a sound — the F-41 review's
    finding. The sweep scripts load such a ring directly, outside resume."""
    from reckoner.model import Reckoner
    from reckoner.replay import RingError
    from reckoner.train import train_on_ring

    ring, _truth = ring_with_truth(monkeypatch, ring_format=1)
    assert ring.era_absent("visit_actions") == len(ring) > 0
    torch.manual_seed(0)
    with pytest.raises(RingError, match="F-41"):
        train_on_ring(Reckoner(_ring_only_cfg()), ring, _ring_only_cfg(), steps=2, seed=0)


def test_train_on_ring_trains_every_step_on_a_current_ring(monkeypatch) -> None:
    """The other polarity: a ring written by the fixed runner trains every step,
    skips nothing, and moves the policy head itself."""
    from reckoner.model import Reckoner
    from reckoner.train import train_on_ring

    ring, _truth = ring_with_truth(monkeypatch)
    cfg = _ring_only_cfg()
    torch.manual_seed(0)
    model = Reckoner(cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stats = train_on_ring(model, ring, cfg, steps=2, seed=0)
    assert stats.steps == 2 and stats.encode_skips == 0
    policy_head = [k for k in before if k.startswith(("rule_embedding", "site_projection"))]
    assert policy_head and any(
        not torch.equal(model.state_dict()[k], before[k]) for k in policy_head
    )
