"""Phase 1: supervised warm start.

Targets, per the chunk-8 brief:

* **policy** — the action the BFS-optimal derivation took, one-hot over the legal
  set, masked by ``legal_actions``.
* **steps** — remaining steps to the solve, regressed, masked to solved rows
  (every Phase-1 row is on a solved derivation, so the mask is all-true here and
  is kept anyway: the machinery must be exercised before Phase 2 relies on it).
* **W/D/L** — **loss weight 0, deliberately.** Spec §5 trains Phase-1 value on
  steps-to-solve. Imitation data has degenerate z: every row is on an optimal
  derivation, so every episode would be a draw, and training the head on that
  manufactures a confidently-wrong prior rather than an ignorant one.
  Untrained-but-uninformative beats miswired. Phase 2 wakes it.

The single-BFS-path caveat is recorded in the dataset's own meta: where several
optimal derivations exist, the network is taught the one BFS reached first.
This module once said Phase 2's targets "wash it out". Under F-41 they could not
(the ring trained the wrong coordinates). Under F-42 the target is π′, which
departs from the prior only where the search holds evidence, so it washes the
tie-break out only where proofs reach it. With rehearsal at 0.65 the supervised
one-hot is still most of every batch. Whether the imprint survives is measured,
not assumed (TAZ comparison, E-02/F-03).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reckoner.config import Config
from reckoner.dataset import anchored_data, anchored_path, sample_indices
from reckoner.episode import Problem, decode_state
from reckoner.model import N_RULES, Reckoner, StateTooLarge, encode, policy_loss, steps_loss
from reckoner.valuegate import ValueHeadState, value_contribution
from reckoner.vocab import PAD


@dataclass
class Batch:
    tokens: torch.Tensor
    site_positions: torch.Tensor
    legal_mask: torch.Tensor
    policy_target: torch.Tensor
    steps_target: torch.Tensor
    solved_mask: torch.Tensor
    depth: torch.Tensor
    skipped: int = 0
    width: int = 0  # columns after the crop; `tokens.shape[1]`, recorded not inferred
    #: Phase-2 only: the episode's real z as a W/D/L class, and the search's
    #: root_q. Absent for Phase-1 batches, where z is degenerate by construction.
    z_class: torch.Tensor | None = None
    root_q: torch.Tensor | None = None
    #: Phase-2 'improved' only: rows whose π′ dropped more than IMPROVED_TAIL_TOL
    #: beyond the ring's slots (F-42). Their POLICY target is withheld; they stay
    #: in the batch for the value and steps heads. Kept apart from ``skipped``,
    #: which counts encode and era sit-outs, which leave the batch entirely.
    truncated: int = 0
    #: Which rows carry a policy target, or ``None`` when every row does. Set
    #: only when a row's π′ was withheld, so the all-rows loss path — the
    #: control arm's, and 'improved' without truncation — is untouched.
    policy_rows: torch.Tensor | None = None


#: The ring fields each policy target is built from (F-42). The keys are exactly
#: ``config.POLICY_TARGETS``, and a test pins that.
TARGET_FIELDS: dict[str, tuple[str, ...]] = {
    "improved": ("improved_actions", "improved_probs", "improved_tail"),
    "visits": ("visit_actions", "visit_counts"),
}


@dataclass
class TrainStats:
    steps: int = 0
    examples: int = 0
    nan_skips: int = 0
    encode_skips: int = 0
    #: Steps whose batch carried a supervised share (F-31). Zero at
    #: ``rehearsal_frac = 0.0``, and that zero is the inertness claim made
    #: countable rather than asserted.
    rehearsal_batches: int = 0
    #: Ring rows that sat out under 'improved' because their π′ was cropped by
    #: the ring's slots (F-42). The write-side counter in the iteration row is
    #: the logged instrument; this is the train side's own count of the same.
    improved_truncated: int = 0
    losses: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "examples": self.examples,
            "rehearsal_batches": self.rehearsal_batches,
            "nan_skips": self.nan_skips,
            "encode_skips": self.encode_skips,
            "improved_truncated": self.improved_truncated,
            "final_loss": self.losses[-1] if self.losses else None,
        }


class SupervisionSet:
    """Mapped Phase-1 examples. Host RAM is the scarce pool; nothing is loaded."""

    def __init__(self, path: Path, repo: Path | None = None) -> None:
        # THE SECOND LOADER, gated. `read_dataset` governs the train_100k /
        # eval_held_out family and was structurally blind to this one, which
        # reads phase1_train and phase1_eval — the two datasets that carried no
        # registry entry at all until 2026-08-16. Both holes were the same shape
        # and they lined up, because SupervisionSet and those datasets are
        # pre-discipline contemporaries: born before registration-at-birth
        # existed, they travelled together under the old rules while every
        # mechanism built since covered the newer families.
        #
        # Hence the coverage-form corollary: a gate on births governs nothing
        # already alive, so a retrofit's completion criterion is a census of the
        # past, not merely a gate on the future.
        repo = repo or Path(__file__).resolve().parents[2]
        try:
            Path(path).resolve().relative_to(repo.resolve())
        except ValueError:
            pass  # a fixture outside the repo, per assert_tracked's precedent
        else:
            anchored_path(path, repo)
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        n, width = self.meta["n"], self.meta["max_len"]
        load = lambda name, shape: np.memmap(  # noqa: E731
            path / f"{name}.i32", dtype=np.int32, mode="r", shape=shape
        )
        self.tokens = load("tokens", (n, width))
        self.lengths = load("lengths", (n,))
        self.action = load("action", (n,))
        self.steps_remaining = load("steps_remaining", (n,))
        self.depth = load("depth", (n,))
        self.goal = load("goal", (n,))

    def sample(self, k: int, seed: int) -> list[int]:
        """Indices sampled across the whole set. Never a prefix — the supervision
        set is laid out stratum by stratum, so a prefix is a depth-1 pilot."""
        return sample_indices(len(self), k, seed)

    def __len__(self) -> int:
        return int(self.meta["n"])


def _crop_to_content(tokens: torch.Tensor, site_positions: torch.Tensor) -> torch.Tensor:
    """Drop trailing all-PAD columns. **Exact, not approximate** — and measured.

    ``encode`` pads every state to ``model.seq_len`` (512), sized in chunk 6 from
    the *reachable*-state distribution, because Phase 2 self-play can grow states
    past 295 tokens (F-05). Phase 1 supervision states are all on BFS-optimal
    derivations, and optimal derivations never apply ``add_both_sides``
    (ROUND-01), so the whole 313,628-example set has ``max_len 64``. Eight times
    the width, and attention is O(L²).

    Cropping changes nothing the network computes: the trunk already ignores PAD
    via ``src_key_padding_mask``, ``_masked_mean`` pools over ``ne(PAD)``, and
    positions are indexed from 0 by ``arange``, so rows 0..width-1 of the
    position table are the same rows either way. That equality is pinned as an
    exact tensor identity by ``test_cropping_padding_is_exact``, both polarities:
    the crop must fire *and* the outputs must be bit-identical.

    **Do not "fix" this by lowering ``model.seq_len``.** The envelope is sized
    for Phase 2, where the states really are that long; the width is a property
    of a batch, not of the model.
    """
    filled = tokens.ne(PAD).nonzero()
    # An all-PAD batch cannot occur (a state has at least a goal token), but a
    # width of 0 would be silently catastrophic, so the floor is stated.
    width = int(filled[:, 1].max()) + 1 if filled.numel() else 1
    width = max(width, int(site_positions.max()) + 1)
    return tokens[:, :width]


def make_batch(data: SupervisionSet, indices: list[int], cfg: Config) -> Batch:
    """Encode a batch. Site positions and legality are derived, never stored.

    Storing them would cost ~660 MB for a 313K set at `max_sites=192`; deriving
    them costs a decode plus one `legal_actions` call per row, and keeps the
    single legality oracle intact — a stored mask is a second copy that can
    disagree with the engine.
    """
    tokens, positions, masks, targets, steps, depths = [], [], [], [], [], []
    skipped = 0
    for i in indices:
        seq = tuple(int(t) for t in data.tokens[i, : data.lengths[i]])
        goal, target, expr = decode_state(seq)
        problem = Problem(
            goal=goal, expr=expr, par=int(data.depth[i]), target=target, par_source="bfs"
        )
        try:
            state = encode(problem, expr, cfg)
        except StateTooLarge:
            skipped += 1  # counted, never cropped
            continue
        one_hot = torch.zeros(N_RULES * cfg.model.max_sites)
        one_hot[int(data.action[i])] = 1.0
        tokens.append(state.tokens)
        positions.append(state.site_positions)
        masks.append(state.legal_mask)
        targets.append(one_hot)
        steps.append(float(data.steps_remaining[i]))
        depths.append(int(data.depth[i]))

    if not tokens:
        raise ValueError("every row in the batch failed to encode")
    steps_tensor = torch.tensor(steps)
    stacked = _crop_to_content(torch.stack(tokens), torch.stack(positions))
    return Batch(
        tokens=stacked,
        site_positions=torch.stack(positions),
        width=int(stacked.shape[1]),
        legal_mask=torch.stack(masks),
        policy_target=torch.stack(targets),
        steps_target=steps_tensor,
        solved_mask=torch.ones_like(steps_tensor, dtype=torch.bool),
        depth=torch.tensor(depths),
        skipped=skipped,
    )


#: z -> W/D/L class index. Column order is the head's and is part of the format:
#: 0 = under par (+1), 1 = equal (0), 2 = over par or capped (-1).
Z_TO_CLASS = {1: 0, 0: 1, -1: 2}


def ring_batch(ring, indices: list[int], cfg: Config, *, target: str | None = None) -> Batch | None:
    """Turn stored steps into a trainable batch. Legality is re-derived, never stored.

    **The policy target is ``train.policy_target`` (F-42)**, or ``target`` when a
    caller must pin one (the value switch pins ``'visits'`` so the policy lever
    cannot move its population):

    * ``'improved'`` — the Gumbel π′ the search stored, top ``IMPROVED_SLOTS`` by
      mass in dense action coordinates. A row whose dropped mass exceeds
      ``IMPROVED_TAIL_TOL`` has its POLICY target withheld (``policy_rows``),
      counted in ``truncated``; it is never cropped and never renormalised, and
      it still trains the value and steps heads, so the policy lever cannot
      change who the value head learns from. A row whose stored mass and tail
      do not sum to one is refused: that is a broken writer, not a row.
    * ``'visits'`` — the Sequential-Halving visit share, exactly as before F-42:
      the control arm.

    Both are stored **sparsely** and re-expanded into the dense layout the policy
    head emits. ``visit_actions`` are dense action indices (``runner.ring_visits``);
    records from ring_format 1 hold root slot positions there (F-41) and sit out.
    """
    from reckoner.absence import Absent
    from reckoner.replay import IMPROVED_MASS_TOL, IMPROVED_TAIL_TOL, RingError

    target = cfg.train.policy_target if target is None else target
    if target not in TARGET_FIELDS:
        raise ValueError(f"policy target {target!r} is not one of {sorted(TARGET_FIELDS)}")

    tokens, positions, masks, targets, steps, depths, zs, qs = [], [], [], [], [], [], [], []
    skipped = 0
    truncated = 0
    has_policy: list[bool] = []
    needed = ("tokens", *TARGET_FIELDS[target], "root_q", "z")
    for i in indices:
        record = ring.get(i)
        if any(isinstance(record[k], Absent) for k in needed):
            skipped += 1  # an era-absent field is not a zero; the row sits out
            continue
        if target == "improved":
            probs = np.asarray(record["improved_probs"], dtype=np.float64)
            tail = float(record["improved_tail"])
            total = float(probs.sum()) + tail
            if abs(total - 1.0) > IMPROVED_MASS_TOL:
                raise RingError(
                    f"ring record {i}: π′ stored mass plus tail is {total!r}, not 1 within "
                    f"{IMPROVED_MASS_TOL}. The writer checks this; a record that fails it "
                    "here was corrupted after writing."
                )
        withhold = target == "improved" and float(record["improved_tail"]) > IMPROVED_TAIL_TOL
        goal, target_var, expr = decode_state(tuple(int(t) for t in record["tokens"]))
        problem = Problem(
            goal=goal, expr=expr, par=int(record["par"]), target=target_var, par_source="bfs"
        )
        try:
            state = encode(problem, expr, cfg)
        except StateTooLarge:
            skipped += 1  # counted, never cropped
            continue

        dense = torch.zeros(N_RULES * cfg.model.max_sites)
        if withhold:
            truncated += 1  # policy target withheld: counted, never cropped or renormalised
        elif target == "improved":
            for action, p in zip(record["improved_actions"], record["improved_probs"], strict=True):
                if p > 0:
                    dense[int(action)] = float(p)
        else:
            visits = np.asarray(record["visit_counts"], dtype=np.float64)
            total = visits.sum()
            if total <= 0:
                skipped += 1
                continue
            for action, count in zip(record["visit_actions"], visits, strict=True):
                if count > 0:
                    dense[int(action)] = float(count / total)

        has_policy.append(not withhold)
        tokens.append(state.tokens)
        positions.append(state.site_positions)
        masks.append(state.legal_mask)
        targets.append(dense)
        steps.append(float(record["steps_remaining"]))
        depths.append(int(record["depth"]))
        zs.append(Z_TO_CLASS[int(record["z"])])
        qs.append(float(record["root_q"]))

    if not tokens:
        return None
    steps_tensor = torch.tensor(steps)
    stacked = _crop_to_content(torch.stack(tokens), torch.stack(positions))
    batch = Batch(
        tokens=stacked,
        site_positions=torch.stack(positions),
        width=int(stacked.shape[1]),
        legal_mask=torch.stack(masks),
        policy_target=torch.stack(targets),
        steps_target=steps_tensor,
        solved_mask=torch.ones_like(steps_tensor, dtype=torch.bool),
        depth=torch.tensor(depths),
        skipped=skipped,
        truncated=truncated,
        policy_rows=torch.tensor(has_policy) if truncated else None,
    )
    batch.z_class = torch.tensor(zs, dtype=torch.long)
    batch.root_q = torch.tensor(qs, dtype=torch.float32)
    return batch


def train_on_ring(
    model: Reckoner,
    ring,
    cfg: Config,
    *,
    steps: int,
    seed: int = 0,
    device: str = "cpu",
    value_head: ValueHeadState | None = None,
) -> TrainStats:
    """**Phase 2's training phase.** Ring -> batches -> blended loss -> step.

    The z/q blend, at last implemented: cross-entropy toward the episode's real
    ``z`` plus ``train.value_q_mse_weight`` times MSE between the head's expected
    z and the search's ``root_q``. The self-referential hazard that makes blending
    dangerous elsewhere is structurally absent here because the **checker**, not
    the model's own Q, decides solved — so a wrong Q cannot bootstrap a false
    solve. Stated here as well as in config so a later reader does not "fix" it by
    turning the blend off.

    **The value head trains at full weight even while the search distrusts it.**
    Gating both consumers by the declaration deadlocks the ratchet — no gradient,
    no accuracy, no switch, forever (`FINDINGS.md` F-15). The declaration governs
    what the search *trusts*; this governs what the head is *taught*.

    **A ring holding pre-F-41 records is refused before anything runs.**
    ``ring_batch`` sits such records out, so on an all-old ring every batch is
    empty and this used to return ``steps == 0`` with the skips filed as
    ``encode_skips`` — a sweep arm that trained nothing, reported as a
    measurement, and blamed by its caller on whatever knob it was varying.
    """
    from reckoner.replay import NO_IMPROVED_TARGET, STALE_TARGETS, RingError

    stale = ring.era_absent("visit_actions")
    if stale:
        raise RingError(
            f"{stale} of {len(ring)} ring records: {STALE_TARGETS}. Training would sit "
            "them out silently. Regenerate the ring (scripts/ring0_replay.py) or "
            "repair it before training on it."
        )
    # F-42: under 'improved', a record from before ring_format 3 has no π′ and
    # would sit out of every batch — the same silent zero-step run F-41's review
    # found, one format later. Refused here, before any RNG is drawn. Under
    # 'visits' such records train exactly as they always did.
    if cfg.train.policy_target == "improved":
        lacking = ring.era_absent("improved_probs")
        if lacking:
            raise RingError(
                f"{lacking} of {len(ring)} ring records: {NO_IMPROVED_TARGET}. "
                "train.policy_target is 'improved'; regenerate the ring, or train the "
                "'visits' control arm on it."
            )

    rng = random.Random(seed)
    torch.manual_seed(seed)
    was_training = model.training
    model.to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    stats = TrainStats()

    # The switch criterion's slice is held out from training: a head graded on
    # rows it fit would clear its own bar by memorising them.
    reserved = ring.holdout(cfg.train.ring_holdout_frac, seed=0)
    trainable = [i for i in range(len(ring)) if i not in reserved]

    # REHEARSAL (F-31). `rehearsal_split` has existed since chunk 9 and was called
    # by nothing, so `rehearsal_frac` moved the fingerprint and changed no
    # behaviour. This is the wire.
    #
    # EVERYTHING below is guarded on `n_sup` rather than folded into the common
    # path, because at f = 0.0 this function must be *bit-identically* what it was
    # before the wire existed: no supervision loaded, no extra draw from `rng`, no
    # arithmetic added to the loss. That inertness is what makes wiring a defect
    # fix rather than a treatment change, and it is proved by checkpoint digest
    # rather than argued.
    n_sup, n_ring = rehearsal_split(cfg.train.batch_size, cfg)
    supervision = SupervisionSet(anchored_data("phase1_train")) if n_sup else None
    # A SEPARATE GENERATOR for supervised draws, and it is the sweep's confound
    # rather than a nicety. Drawing them from `rng` means every extra supervised
    # index shifts the ring stream, so two arms differing in `f` also see
    # different RING sample sequences — `f` would change the mixture AND the
    # data, and no arm-to-arm comparison could separate them. The first sweep
    # carried exactly that flaw, and its low-`f` dip could not be told from
    # sampling noise.
    #
    # Seeded off `seed` so arms stay reproducible, and constructed only when
    # `n_sup` is nonzero so `f = 0.0` remains bit-identically inert.
    sup_rng = random.Random(seed * 7919 + 104729) if n_sup else None

    for step in range(steps):
        if not trainable:
            break
        indices = [
            trainable[rng.randrange(len(trainable))] for _ in range(min(n_ring, len(trainable)))
        ]
        batch = ring_batch(ring, indices, cfg)
        if batch is None:
            stats.encode_skips += len(indices)
            continue
        stats.encode_skips += batch.skipped
        stats.improved_truncated += batch.truncated

        lr = lr_at(step, steps, cfg)
        for group in optimiser.param_groups:
            group["lr"] = lr

        policy, value_logits, steps_out = model(
            batch.tokens.to(device), batch.site_positions.to(device)
        )
        rows = batch.policy_rows
        if rows is None:
            # Every row carries a policy target: the control arm's path, and
            # 'improved' without truncation — unchanged from before F-42.
            p_loss = policy_loss(
                policy, batch.policy_target.to(device), batch.legal_mask.to(device)
            )
        elif bool(rows.any()):
            keep = rows.to(device)
            p_loss = policy_loss(
                policy[keep],
                batch.policy_target.to(device)[keep],
                batch.legal_mask.to(device)[keep],
            )
        else:
            p_loss = policy.sum() * 0.0  # no row teaches the policy; value and steps still do
        loss = cfg.train.policy_loss_weight * p_loss
        assert steps_out is not None
        loss = loss + cfg.train.steps_loss_weight * steps_loss(
            steps_out, batch.steps_target.to(device), batch.solved_mask.to(device)
        )
        value_ce = nn.functional.cross_entropy(value_logits, batch.z_class.to(device))
        probs = torch.softmax(value_logits, dim=1)
        expected_z = probs[:, 0] - probs[:, 2]
        q_mse = nn.functional.mse_loss(expected_z, batch.root_q.to(device))
        loss = loss + cfg.train.value_loss_weight * (
            value_ce + cfg.train.value_q_mse_weight * q_mse
        )

        # The supervised share. Phase-1 rows carry a policy target and a steps
        # target and NO z — `Batch.z_class` is documented absent for them, "where
        # z is degenerate by construction" — so they contribute exactly the
        # supervision they hold, and the value head goes on learning only from
        # real episodes. Size-weighted so the pair reconstitutes one batch of
        # `batch_size` rather than becoming two full-weight batches.
        if n_sup and supervision is not None:
            sup_idx = [sup_rng.randrange(len(supervision)) for _ in range(n_sup)]
            try:
                sup = make_batch(supervision, sup_idx, cfg)
            except ValueError:
                sup = None
            if sup is not None:
                sup_policy, _sup_value, sup_steps_out = model(
                    sup.tokens.to(device), sup.site_positions.to(device)
                )
                sup_loss = cfg.train.policy_loss_weight * policy_loss(
                    sup_policy, sup.policy_target.to(device), sup.legal_mask.to(device)
                )
                assert sup_steps_out is not None
                sup_loss = sup_loss + cfg.train.steps_loss_weight * steps_loss(
                    sup_steps_out, sup.steps_target.to(device), sup.solved_mask.to(device)
                )
                total = n_ring + n_sup
                loss = (n_ring * loss + n_sup * sup_loss) / total
                stats.rehearsal_batches += 1

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not finite:
            stats.nan_skips += 1
            optimiser.zero_grad(set_to_none=True)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()

        stats.steps += 1
        stats.examples += int(batch.tokens.shape[0])
        stats.losses.append(float(loss.detach()))

        if stats.steps >= cfg.train.nan_abort_min_steps:
            rate = stats.nan_skips / stats.steps
            if rate > cfg.train.nan_abort_frac:
                raise RuntimeError(
                    f"non-finite gradients on {rate:.1%} of steps — a rate, not a transient"
                )
    # F-22: TRAIN MODE IS CONFINED HERE. Leaving it set is what put dropout
    # inside every subsequent search — the caller asked for a training step, not
    # for a permanently mutated model.
    model.train(was_training)
    return stats


def rehearsal_split(batch_size: int, cfg: Config) -> tuple[int, int]:
    """How many Phase-1 rows to mix into a Phase-2 batch. **Dormant at 0.0.**

    Ported now and left inert, per the plan's "the lever exists before it's
    needed": a rehearsal mechanism added *after* catastrophic forgetting appears
    is a mechanism written under pressure, against a run that is already
    degrading. At ``rehearsal_frac = 0.0`` this returns ``(0, batch_size)`` and
    nothing in the training path changes — which is the accepting case that has
    to be tested too, or "dormant" means "untested".

    Returns ``(from_supervision, from_replay)``.
    """
    frac = cfg.train.rehearsal_frac
    if not 0.0 <= frac < 1.0:
        raise ValueError(
            f"train.rehearsal_frac must be in [0, 1); got {frac}. At 1.0 a Phase-2 "
            "iteration trains on no Phase-2 data, which is not rehearsal."
        )
    supervision = int(round(batch_size * frac))
    return supervision, batch_size - supervision


def lr_at(step: int, total: int, cfg: Config) -> float:
    """Learning rate for ``step`` (0-based), per ``train.lr_schedule``.

    Warmup is linear and applies to both schedules: the first steps of a warm
    start see a policy head at chance over ~1,344 actions, and the gradient that
    produces is not the one the run should be steered by.
    """
    warmup = cfg.train.lr_warmup_steps
    if warmup and step < warmup:
        return cfg.train.lr * (step + 1) / warmup
    if cfg.train.lr_schedule == "constant":
        return cfg.train.lr
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup) / span)
    return cfg.train.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model: Reckoner,
    data: SupervisionSet,
    cfg: Config,
    *,
    steps: int,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 50,
    on_log: Callable[[int, float, float], None] | None = None,
    value_head: ValueHeadState | None = None,
) -> TrainStats:
    """Run the warm start. NaN-skip guard with a 1% abort, per the inherited kit.

    ``on_log(step, mean_loss_over_window, lr)`` is called every ``log_every`` steps.
    Library code does not print (AGENTS.md §6) — the caller in ``scripts/`` owns
    the terminal, and a training loop that printed would also be unusable from a
    dashboard or a test.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    stats = TrainStats()

    for step in range(steps):
        lr = lr_at(step, steps, cfg)
        for group in optimiser.param_groups:
            group["lr"] = lr
        indices = [rng.randrange(len(data)) for _ in range(cfg.train.batch_size)]
        batch = make_batch(data, indices, cfg)
        stats.encode_skips += batch.skipped

        policy, _value, steps_out = model(batch.tokens.to(device), batch.site_positions.to(device))
        loss = cfg.train.policy_loss_weight * policy_loss(
            policy, batch.policy_target.to(device), batch.legal_mask.to(device)
        )
        assert steps_out is not None
        loss = loss + cfg.train.steps_loss_weight * steps_loss(
            steps_out, batch.steps_target.to(device), batch.solved_mask.to(device)
        )
        # W/D/L head: loss weight is `value_loss_weight * value_contribution(state)`,
        # and in Phase 1 the state is untrusted so the product is 0 — the head's
        # parameters do not appear in this graph. THE SAME FUNCTION scales the
        # search's value contribution (F-14): one declaration, two consumers, so
        # a head nobody is training cannot be a head the search is trusting.
        _value_weight = cfg.train.value_loss_weight * value_contribution(
            value_head or ValueHeadState()
        )

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not finite:
            stats.nan_skips += 1
            optimiser.zero_grad(set_to_none=True)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()

        stats.steps += 1
        stats.examples += int(batch.tokens.shape[0])
        stats.losses.append(float(loss.detach()))

        if stats.steps >= cfg.train.nan_abort_min_steps:
            rate = stats.nan_skips / stats.steps
            if rate > cfg.train.nan_abort_frac:
                raise RuntimeError(
                    f"non-finite gradients on {rate:.1%} of steps "
                    f"({stats.nan_skips}/{stats.steps}) — a rate, not a transient"
                )
        if on_log is not None and log_every and (step + 1) % log_every == 0:
            window = stats.losses[-log_every:]
            on_log(step + 1, sum(window) / len(window), lr)

    return stats
