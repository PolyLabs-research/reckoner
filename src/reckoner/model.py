"""The network: compositional embeddings, a rule × site policy, W/D/L vs par.

Sizing came from measurement, and the measurement had to be of the right thing
-------------------------------------------------------------------------------
``seq_len`` and ``max_sites`` are set from the **reachable-state** distribution,
not from the dataset's start states. Episodes grow before they shrink —
``sub_both_sides`` adds a negation term that ``combine_like_terms`` only removes
a step later, and ``add_both_sides`` inflates both sides without bound — so the
longest state inside an episode strictly exceeds the longest problem. Measured
(``scripts/measure_state_extent.py``, ``runs/state_extent.json``):

    population              tokens p50 / p99 / p100     sites p50 / p99 / p100
    start states                33 /  64 /  64                10 / 17 /  17
    optimal derivations         25 /  64 /  64                 7 / 17 /  17
    random walks (cap 24)       64 / 216 / 332                19 / 76 / 121

Sizing from the dataset would have put ``seq_len`` at 64 and guaranteed overflow
the first time search explored. The chosen bounds carry ~1.5× headroom over the
measured p100 across all three populations.

Overflow is an error, never a crop
----------------------------------
A state that does not fit raises :class:`StateTooLarge`, and the episode runner
counts it and aborts that episode with a reason. **It is never silently
truncated.** A cropped state tensor is the fabricated-input twin of the
fabricated-target class the masked-loss test exists to catch: the network would
be scored on a position that does not exist, and nothing downstream could tell.

One legality oracle
-------------------
The policy mask is consumed from :func:`reckoner.rules.legal_actions`
byte-for-byte. **The model never re-derives legality**, and there is no second
implementation of it anywhere in this module — a mask that disagrees with the
movegen is a model trained to want illegal moves, and it fails silently because
the argmax is still legal most of the time. Fourth member of the mono-instance
family: one terminal test (``verify``), one formatter (``render_expr``), one
identity normalizer (``identity_key``), one legality source (``legal_actions``).

Action indexing
---------------
``action_index = rule_id * max_sites + site_id``. Fixed, because datasets store
it: chunk 9's replay ring writes improved-policy targets against this layout, and
a change silently relabels every stored target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from reckoner.config import Config, config_fingerprint
from reckoner.episode import Problem, encode_state
from reckoner.expr import Expr
from reckoner.rules import RULES, RULESET_VERSION, legal_actions, site_token_offsets
from reckoner.vocab import PAD, VOCAB_SIZE, VOCAB_VERSION

N_RULES = len(RULES)


class StateTooLarge(ValueError):
    """A state exceeds ``seq_len`` or ``max_sites``. Counted, never cropped."""


class IllegalTargetError(ValueError):
    """Policy-target mass on an action the movegen refuses. Refused, never dropped."""


# ---------------------------------------------------------------------------
# Encoding a state for the network
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodedState:
    """Everything the network needs about one position."""

    tokens: torch.Tensor  # (seq_len,) int64, PAD-filled
    length: int
    site_positions: torch.Tensor  # (max_sites,) int64, 0-filled past n_sites
    n_sites: int
    legal_mask: torch.Tensor  # (N_RULES * max_sites,) bool


def encode(problem: Problem, expr: Expr, cfg: Config) -> EncodedState:
    """Encode a live state. Raises :class:`StateTooLarge` rather than cropping."""
    seq = encode_state(problem.goal, expr, problem.target)
    # [goal, SEP] or [goal, target, SEP] — the expression starts after it.
    prefix = 2 if problem.target is None else 3
    seq_len, max_sites = cfg.model.seq_len, cfg.model.max_sites

    if len(seq) > seq_len:
        raise StateTooLarge(
            f"state is {len(seq)} tokens, seq_len is {seq_len}. This is counted and "
            "the episode aborts; it is never cropped — a truncated state is a "
            "position that does not exist being scored as if it did."
        )
    offsets = site_token_offsets(expr)
    if len(offsets) > max_sites:
        raise StateTooLarge(
            f"state has {len(offsets)} sites, max_sites is {max_sites}. Counted, "
            "not cropped: dropping sites would hide legal actions from the mask."
        )

    tokens = torch.full((seq_len,), PAD, dtype=torch.long)
    tokens[: len(seq)] = torch.tensor(seq, dtype=torch.long)

    positions = torch.zeros(max_sites, dtype=torch.long)
    positions[: len(offsets)] = torch.tensor([prefix + o for o in offsets], dtype=torch.long)

    # The mask is *consumed*, never re-derived. See the module docstring.
    mask = torch.zeros(N_RULES * max_sites, dtype=torch.bool)
    for rule_id, site_id in legal_actions(expr):
        mask[rule_id * max_sites + site_id] = True

    return EncodedState(tokens, len(seq), positions, len(offsets), mask)


def action_index(rule_id: int, site_id: int, max_sites: int) -> int:
    """``rule_id * max_sites + site_id``. Datasets store this; it does not move."""
    return rule_id * max_sites + site_id


def action_pair(index: int, max_sites: int) -> tuple[int, int]:
    return divmod(index, max_sites)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


class Reckoner(nn.Module):
    """Transformer trunk, factorised rule × site policy, W/D/L value, steps aux."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.max_sites = m.max_sites

        # Compositional embeddings: one row per vocabulary symbol, so a numeral
        # of any magnitude is a short run of digit rows rather than its own id.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, m.d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(m.seq_len, m.d_model)
        self.dropout = nn.Dropout(m.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=m.d_model,
            nhead=m.n_heads,
            dim_feedforward=m.d_ff,
            dropout=m.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # enable_nested_tensor is incompatible with norm_first and only emits a
        # warning; disabled explicitly so the test output stays readable.
        self.trunk = nn.TransformerEncoder(layer, num_layers=m.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(m.d_model)

        # Policy: bilinear over (rule, site), the (from, to) factorisation
        # transplanted. A rule vector meets a site vector in d_policy space.
        self.rule_embedding = nn.Embedding(N_RULES, m.d_policy)
        self.site_projection = nn.Linear(m.d_model, m.d_policy)

        self.value_head = nn.Linear(m.d_model, m.value_classes)
        self.steps_head = nn.Linear(m.d_model, 1) if m.steps_aux_head else None

    def forward(
        self, tokens: torch.Tensor, site_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(policy_logits, value_logits, steps)`` for a batch.

        ``policy_logits`` is ``(B, N_RULES * max_sites)`` in the fixed action
        layout. Masking is the caller's job and uses ``legal_mask`` — this
        module has no opinion about legality.
        """
        batch, seq = tokens.shape
        positions = torch.arange(seq, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        hidden = self.trunk(hidden, src_key_padding_mask=tokens.eq(PAD))
        hidden = self.norm(hidden)

        # A site's vector is the trunk state at the token its subtree starts on.
        index = site_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        site_hidden = hidden.gather(1, index)  # (B, max_sites, d_model)
        site_vectors = self.site_projection(site_hidden)  # (B, max_sites, d_policy)
        rule_vectors = self.rule_embedding.weight  # (N_RULES, d_policy)
        # (B, N_RULES, max_sites) -> flat, matching action_index's layout
        logits = torch.einsum("bsd,rd->brs", site_vectors, rule_vectors)
        policy_logits = logits.reshape(batch, N_RULES * self.max_sites)

        pooled = _masked_mean(hidden, tokens.ne(PAD))
        value_logits = self.value_head(pooled)
        steps = self.steps_head(pooled).squeeze(-1) if self.steps_head is not None else None
        return policy_logits, value_logits, steps

    # --- provenance -----------------------------------------------------

    def parameter_breakdown(self) -> dict[str, int]:
        """Params by component. A total alone cannot be audited against the spec.

        The embedding table is ~657 rows wide, so at this scale it is a
        meaningful slice rather than a rounding error, and the 2–7M envelope
        should be checkable from the parts.
        """
        groups = {
            "embeddings": [self.token_embedding, self.position_embedding],
            "trunk": [self.trunk, self.norm],
            "policy_head": [self.rule_embedding, self.site_projection],
            "value_head": [self.value_head],
            "steps_head": [self.steps_head] if self.steps_head is not None else [],
        }
        out = {
            name: sum(p.numel() for module in modules for p in module.parameters())
            for name, modules in groups.items()
        }
        out["total"] = sum(p.numel() for p in self.parameters())
        assert out["total"] == sum(v for k, v in out.items() if k != "total")
        return out


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Losses — every mask is a detector, and both are validated on both polarities
# ---------------------------------------------------------------------------


def policy_loss(
    policy_logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over the **legal** actions only.

    Logits under masked-off actions cannot affect the loss. That is the ported
    fabricated-target detector: without it a model can be trained toward an
    action the movegen would refuse, and the symptom is invisible because the
    argmax is legal most of the time anyway.

    **Target mass under masked-off actions is refused, not dropped (F-53).** The
    masked log-probability is ``-inf`` there and ``nan_to_num`` makes the product
    exactly 0, so an illegal target used to contribute nothing and say nothing.
    That is how a writer storing root slots instead of action indices sent 93% of
    the ring's visit mass into masked cells for all of M1 while every check
    passed: the mask on logits was a detector, and on targets it was a silencer.
    A target that names an illegal action has no correct reading, so it raises.
    """
    off_legal = (targets != 0) & ~legal_mask
    if bool(off_legal.any()):
        rows = int(off_legal.any(-1).sum())
        mass = float((targets * off_legal).sum())
        raise IllegalTargetError(
            f"{int(off_legal.sum())} policy-target entries in {rows} row(s) sit on "
            f"illegal actions (total mass {mass:.4g}). The loss would drop them "
            "silently; a target naming an action the movegen refuses is a writer bug "
            "(F-53), not data."
        )
    masked = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = torch.log_softmax(masked, dim=-1)
    log_probs = torch.nan_to_num(log_probs, neginf=0.0)  # rows with no legal action
    return -(targets * log_probs).sum(-1).mean()


def steps_loss(
    steps: torch.Tensor, targets: torch.Tensor, solved_mask: torch.Tensor
) -> torch.Tensor:
    """Steps-to-solve regression, **masked to solved episodes**.

    An unsolved episode has no steps-to-solve; regressing toward one is
    inventing a target. Same detector as the policy mask, on the other head —
    and it needs its own test, because a mask that is merely *declared* is not a
    mask.
    """
    weights = solved_mask.to(steps.dtype)
    error = (steps - targets) ** 2 * weights
    return error.sum() / weights.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def checkpoint_meta(
    cfg: Config, step: int, repo: Path | None = None, *, value_head: dict | None = None
) -> dict:
    """What a checkpoint must carry to be interpretable on its own.

    ``ruleset_version`` and ``vocab_version`` sit beside the config fingerprint
    because a checkpoint is denominated in a rule system exactly like a par is.
    **Registered for chunk 9:** ``CheckpointPool`` must refuse a snapshot whose
    versions differ from the running environment — a pool par re-solved under a
    different rule system is the par-provenance defect (F-02) reborn at the
    league layer, and the guard costs one line before the door exists.
    """
    from reckoner.dataset import git_sha

    return {
        "ruleset_version": RULESET_VERSION,
        "vocab_version": VOCAB_VERSION,
        "config_fingerprint": config_fingerprint(cfg),
        "git_sha": git_sha(repo or Path(__file__).resolve().parents[2]),
        "step": step,
        # **A snapshot's wiring is part of its identity.** Pool par is a par
        # re-solved BY THIS SNAPSHOT, so it must be solved under the declaration
        # the snapshot was trained and played under — not the running run's. A
        # pre-switch snapshot re-solved with post-switch wiring produces a par
        # that snapshot never achieved: a wrong number under a provenance tag
        # true in its own terms, which is F-02 one layer up.
        "value_head": value_head or {"live": False, "switched_at_iteration": None},
    }


def save_checkpoint(
    path: Path,
    model: Reckoner,
    cfg: Config,
    step: int,
    *,
    value_head: dict | None = None,
    **extra,
) -> dict:
    meta = checkpoint_meta(cfg, step, value_head=value_head)
    meta.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return meta


def load_checkpoint(
    path: Path, cfg: Config, *, strict_versions: bool = True
) -> tuple[Reckoner, dict]:
    """Load a checkpoint, refusing one denominated in a different system."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    if strict_versions:
        for field, current in (
            ("ruleset_version", RULESET_VERSION),
            ("vocab_version", VOCAB_VERSION),
        ):
            if meta.get(field) != current:
                raise ValueError(
                    f"checkpoint {field}={meta.get(field)} but this environment is "
                    f"{current}. Its weights index a different action or symbol space; "
                    "loading it would score a rule system that no longer exists."
                )
    model = Reckoner(cfg)
    model.load_state_dict(blob["state_dict"])
    return model, meta


def load_league_checkpoint(path: Path, cfg: Config) -> tuple[Reckoner, dict]:
    """Load a snapshot for the LEAGUE. **No escape hatch, by construction.**

    ``load_checkpoint`` takes ``strict_versions`` because offline inspection
    legitimately needs it. The league does not, and "must not pass False" is a
    convention until it is impossible — so the league path calls this, which has
    no such parameter to pass. Helpers prevent; conventions are remembered.
    """
    return load_checkpoint(path, cfg, strict_versions=True)


def meta_fingerprint(meta: dict) -> str:
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
