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
from reckoner.replay import VISIT_SLOTS, ReplayRing
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
        batch = ring_batch(ring, [i], CFG, target="visits")
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
    """No stored target mass lands outside the legal set, under either arm."""
    ring, _truth = ring_with_truth(monkeypatch)
    for target in ("visits", "improved"):
        batch = ring_batch(ring, list(range(len(ring))), CFG, target=target)
        assert batch is not None and batch.truncated == 0
        off_legal = batch.policy_target * (~batch.legal_mask)
        assert float(off_legal.sum()) == 0.0, target
        sums = batch.policy_target.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums)), target


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


# ---------------------------------------------------------------------------
# F-42: the 'improved' arm trains the producer's own π′, exactly
# ---------------------------------------------------------------------------


def _scatter_improved(result: SearchResult) -> torch.Tensor:
    dense = torch.zeros(N_RULES * MAX_SITES)
    for slot, p in enumerate(result.improved):
        if p > 0:
            dense[action_index(*result.actions[slot], MAX_SITES)] = float(p)
    return dense


def test_every_improved_target_is_the_producers_pi_prime_in_action_coordinates(
    monkeypatch,
) -> None:
    """F-41's rule applied to F-42: writer and reader on the real path, scored
    against ground truth captured from the producer. Bit-exact float32: the ring
    stores π′'s values unrenormalised and the reader scatters them unchanged."""
    ring, truth = ring_with_truth(monkeypatch)
    widths = []
    for i in range(len(ring)):
        result = truth[i]
        widths.append(len(result.actions))
        batch = ring_batch(ring, [i], CFG, target="improved")
        assert batch is not None and batch.skipped == 0 and batch.truncated == 0
        assert torch.equal(batch.policy_target[0], _scatter_improved(result)), f"record {i}"
    assert len(ring) >= 20 and max(widths) > VISIT_SLOTS, (
        f"coverage: {len(ring)} records, widest root {max(widths)} legal actions — the "
        "fixture must include roots wider than the visit layout, or π′'s extra width "
        "is untested"
    )


def test_policy_target_is_one_lever(monkeypatch) -> None:
    """The two arms read the same ring and differ in the policy target ONLY:
    tokens, legality, value labels and steps targets are identical, and the
    policy targets differ on most rows. Behavioural liveness for the key — the
    census cannot vouch for it (a bare-name collision with Batch.policy_target)."""
    ring, _truth = ring_with_truth(monkeypatch)
    rows = list(range(len(ring)))
    visits = ring_batch(ring, rows, CFG, target="visits")
    improved = ring_batch(ring, rows, CFG, target="improved")
    assert visits is not None and improved is not None
    for name in ("tokens", "site_positions", "legal_mask", "z_class", "root_q", "steps_target"):
        assert torch.equal(getattr(visits, name), getattr(improved, name)), name
    assert visits.skipped == improved.skipped
    differing = int((visits.policy_target != improved.policy_target).any(-1).sum())
    assert differing >= 0.5 * len(rows), f"only {differing}/{len(rows)} rows differ"


def _wide_problem():
    """A SOLVE state with 168 legal actions — wider than the ring's 160 π′ slots
    — that still encodes (167 sites). Constructed, so the cropped path is tested
    for real rather than skipped."""
    from reckoner.episode import Problem
    from reckoner.expr import add, eq, mul, num, var
    from reckoner.vocab import GOAL_SOLVE, VAR_X

    x = var(VAR_X)
    expr = eq(
        add(*[mul(num(i + 2), x) for i in range(41)], *[num(i + 1) for i in range(41)]), num(5)
    )
    return Problem(goal=GOAL_SOLVE, expr=expr, par=4, target=VAR_X, par_source="pool")


def _wide_ring(records: int = 24) -> tuple[ReplayRing, list[int], object]:
    """A ring whose records carry a CROPPED π′, written by the real writer.

    The state is real (168 legal actions, more than the ring's 160 slots, and it
    encodes). The search itself rarely spreads π′ that wide — on this state
    some children overflow the encoder and become proven losses, v_mix goes
    negative and π′ concentrates on the visited actions — so the root result
    handed to ``runner._settle`` carries the flattest π′ there is, uniform over
    every legal action. Everything downstream is the production path: the
    writer computes the tail and the counter, the ring stores them, and the
    reader decides on what was stored.
    """
    import random

    import numpy as np

    from reckoner.replay import IMPROVED_SLOTS, IMPROVED_TAIL_TOL
    from reckoner.rules import legal_actions
    from reckoner.search import SearchStats

    problem = _wide_problem()
    actions = legal_actions(problem.expr)
    n = len(actions)
    assert n > IMPROVED_SLOTS
    encode(problem, problem.expr, CFG)  # must encode: fails, not skips, if it stops
    result = SearchResult(
        np.ones(n, np.int32),
        np.zeros(n, np.float32),
        actions,
        actions[0],
        0.0,
        SearchStats(),
        root_priors=np.full(n, 1.0 / n),
        improved=np.full(n, 1.0 / n, np.float32),
    )
    ring = ReplayRing(256, CFG)
    stats = runner.IterationStats()
    live = runner._Live(problem=problem, expr=problem.expr, rng=random.Random(0), steps=records)
    trail = {0: [(result, problem.expr, step) for step in range(records)]}
    runner._settle(stats, ring, live, trail, CFG, solved=False, capped=True)
    wide = [i for i in range(len(ring)) if ring.get(i)["improved_tail"] > IMPROVED_TAIL_TOL]
    assert len(wide) == records, "every record must carry a dropped tail"
    assert ring.get(0)["improved_tail"] == pytest.approx((n - IMPROVED_SLOTS) / n, rel=1e-6)
    assert stats.ring_improved_truncated == records, "write-side count and stored tails disagree"
    return ring, wide, stats


def test_a_root_wider_than_the_ring_withholds_only_its_policy_target() -> None:
    """Counted, never cropped, never renormalised — and the row STAYS for the
    value and steps heads, so the policy lever cannot change who the value head
    learns from (F-42 review). The control arm trains every row's visit share."""
    ring, wide, _stats = _wide_ring()
    improved = ring_batch(ring, wide, CFG, target="improved")
    visits = ring_batch(ring, wide, CFG, target="visits")
    assert improved is not None and visits is not None
    assert improved.truncated == len(wide) and visits.truncated == 0
    assert improved.policy_rows is not None and not bool(improved.policy_rows.any())
    assert float(improved.policy_target.abs().sum()) == 0.0
    for name in ("tokens", "legal_mask", "z_class", "root_q", "steps_target"):
        assert torch.equal(getattr(improved, name), getattr(visits, name)), name


def test_a_ring_of_only_wide_rows_still_trains_the_value_head() -> None:
    """The silent zero-step run, closed: under 'improved' every policy target on
    this ring is withheld, and training still takes every step on value and
    steps, with the withheld rows counted as truncated, not as encode skips."""
    from reckoner.model import Reckoner
    from reckoner.train import train_on_ring

    ring, wide, _stats = _wide_ring()
    cfg = _ring_only_cfg()
    torch.manual_seed(0)
    model = Reckoner(cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stats = train_on_ring(model, ring, cfg, steps=2, seed=0)
    assert stats.steps == 2 and stats.encode_skips == 0 and stats.improved_truncated > 0
    moved = {
        k.split(".")[0] for k, v in model.state_dict().items() if not torch.equal(v, before[k])
    }
    assert "value_head" in moved and "steps_head" in moved


def test_the_configured_target_drives_ring_batch(monkeypatch) -> None:
    """Behavioural liveness of the key itself: with no ``target=`` the reader
    follows ``train.policy_target``, and the two settings produce different
    policy targets on the same rows."""
    from dataclasses import replace

    ring, _truth = ring_with_truth(monkeypatch)
    rows = list(range(len(ring)))
    for name in ("improved", "visits"):
        cfg = replace(CFG, train=replace(CFG.train, policy_target=name))
        implicit = ring_batch(ring, rows, cfg)
        explicit = ring_batch(ring, rows, CFG, target=name)
        assert implicit is not None and explicit is not None
        assert torch.equal(implicit.policy_target, explicit.policy_target), name
    assert not torch.equal(
        ring_batch(ring, rows, CFG, target="improved").policy_target,
        ring_batch(ring, rows, CFG, target="visits").policy_target,
    )


def test_the_value_switch_population_does_not_depend_on_the_policy_target() -> None:
    """``evaluate_head`` is pinned to the 'visits' row rule, so its labels and
    predictions are identical under either configured target, wide rows
    included."""
    from dataclasses import replace

    from reckoner.model import Reckoner
    from reckoner.valuegate import evaluate_head

    ring, wide, _stats = _wide_ring()
    torch.manual_seed(0)
    model = Reckoner(CFG)
    slots = list(range(len(ring)))
    readings = []
    for name in ("improved", "visits"):
        cfg = replace(CFG, train=replace(CFG.train, policy_target=name))
        readings.append(evaluate_head(model, ring, cfg, slots))
    assert readings[0] == readings[1]
    assert len(readings[0][0]) == len(ring), "the wide rows must be in the population"


def test_train_on_ring_refuses_a_pre_pi_prime_ring_under_improved(monkeypatch) -> None:
    """A format-2 ring has no π′: under 'improved' every row would sit out and
    the run would train nothing in silence. Refused before any RNG is drawn."""
    from reckoner.model import Reckoner
    from reckoner.replay import RingError
    from reckoner.train import train_on_ring

    ring, _truth = ring_with_truth(monkeypatch, ring_format=2)
    cfg = _ring_only_cfg()
    torch.manual_seed(123)
    model = Reckoner(cfg)
    state = torch.random.get_rng_state()
    with pytest.raises(RingError, match="F-42"):
        train_on_ring(model, ring, cfg, steps=2, seed=0)
    assert torch.equal(torch.random.get_rng_state(), state), "the refusal drew from the RNG"


def test_the_visits_arm_trains_on_a_pre_pi_prime_ring(monkeypatch) -> None:
    """The other polarity: the control arm reads the visit fields, which a
    format-2 ring has, and trains exactly as before."""
    from dataclasses import replace

    from reckoner.model import Reckoner
    from reckoner.train import train_on_ring

    ring, _truth = ring_with_truth(monkeypatch, ring_format=2)
    cfg = _ring_only_cfg()
    cfg = replace(cfg, train=replace(cfg.train, policy_target="visits"))
    torch.manual_seed(0)
    stats = train_on_ring(Reckoner(cfg), ring, cfg, steps=2, seed=0)
    assert stats.steps == 2


def test_a_corrupted_pi_prime_row_is_refused_on_the_read_path(monkeypatch) -> None:
    """The writer checks the mass; a row that fails it on the way back in was
    corrupted after writing and must raise, not train."""
    from reckoner.replay import RingError

    ring, _truth = ring_with_truth(monkeypatch)
    ring_batch(ring, [0], CFG, target="improved")  # intact: passes
    ring.improved_probs[0, 0] += 0.5
    with pytest.raises(RingError, match="not 1"):
        ring_batch(ring, [0], CFG, target="improved")
