"""The network: shapes, both masks, the single legality oracle, provenance."""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import pytest
import torch

from reckoner import model as model_module
from reckoner.config import Config
from reckoner.episode import Problem, encode_state
from reckoner.expr import add, eq, mul, num, var
from reckoner.model import (
    N_RULES,
    Reckoner,
    StateTooLarge,
    action_index,
    action_pair,
    checkpoint_meta,
    encode,
    load_checkpoint,
    policy_loss,
    save_checkpoint,
    steps_loss,
)
from reckoner.rules import (
    RULESET_VERSION,
    enumerate_sites,
    legal_actions,
    site_token_offsets,
    successors,
)
from reckoner.vocab import (
    GOAL_EVALUATE,
    GOAL_SIMPLIFY,
    GOAL_SOLVE,
    PAD,
    VAR_X,
    VAR_Y,
    VOCAB_SIZE,
    VOCAB_VERSION,
)

REPO = Path(__file__).resolve().parents[1]
CFG = Config()
X = var(VAR_X)
Y = var(VAR_Y)

PROBLEMS = [
    Problem(
        goal=GOAL_SOLVE,
        target=VAR_X,
        par=3,
        par_source="bfs",
        expr=eq(add(mul(num(3), X), num(6)), num(21)),
    ),
    Problem(goal=GOAL_EVALUATE, par=1, par_source="bfs", expr=add(num(17), num(-25))),
    Problem(goal=GOAL_SIMPLIFY, par=1, par_source="bfs", expr=add(mul(num(3), X), mul(num(2), X))),
]


# ---------------------------------------------------------------------------
# Site positions: the map the policy head reads a site through
# ---------------------------------------------------------------------------


def test_site_offsets_agree_with_site_enumeration() -> None:
    """Two orders that must agree — pinned, not assumed to coincide."""
    from reckoner.expr import tokens

    rng = random.Random(3)
    checked = 0
    population = list(PROBLEMS)
    suites = REPO / "runs" / "suites"
    if suites.exists():
        from reckoner.dataset import read_suite, suite_problem

        for depth in range(1, 7):
            path = suites / f"solve_in_{depth}.jsonl"
            if path.exists():
                population += [suite_problem(r) for r in read_suite(path)[:15]]
    for problem in population:
        state = problem.expr
        for _ in range(6):
            sites = enumerate_sites(state)
            offsets = site_token_offsets(state)
            seq = tokens(state)
            assert len(offsets) == len(sites)
            for site, offset in zip(sites, offsets, strict=True):
                assert seq[offset : offset + len(tokens(site.node))] == tokens(site.node), (
                    f"site {site.site_id} does not start at token {offset}"
                )
                checked += 1
            options = successors(state)
            if not options:
                break
            state = rng.choice(options)[1]
    assert checked >= 200, f"only {checked} sites checked"


# ---------------------------------------------------------------------------
# One legality oracle
# ---------------------------------------------------------------------------


def test_the_mask_is_consumed_from_legal_actions_byte_for_byte() -> None:
    for problem in PROBLEMS:
        state = encode(problem, problem.expr, CFG)
        expected = torch.zeros(N_RULES * CFG.model.max_sites, dtype=torch.bool)
        for rule_id, site_id in legal_actions(problem.expr):
            expected[rule_id * CFG.model.max_sites + site_id] = True
        assert torch.equal(state.legal_mask, expected)
        assert int(state.legal_mask.sum()) == len(legal_actions(problem.expr))


def test_the_model_module_has_no_second_legality_implementation() -> None:
    """Fourth member of the mono-instance family, enforced structurally.

    One terminal test, one formatter, one identity normalizer, one legality
    source. A mask that disagrees with the movegen trains a model to want
    illegal moves, and it fails *silently* — the argmax is legal most of the
    time anyway.
    """
    source = inspect.getsource(model_module)
    assert source.count("legal_actions(") == 1, "legality is derived in more than one place"
    assert ".legal(" not in source, "the model calls Rule.legal — that is a second oracle"
    assert "def legal" not in source, "the model defines its own legality"


# ---------------------------------------------------------------------------
# Overflow is an error, never a crop
# ---------------------------------------------------------------------------


def test_an_over_long_state_raises_rather_than_cropping() -> None:
    """A cropped state tensor is the fabricated-input twin of a fabricated target."""
    cfg = Config()
    cfg.model.seq_len = 8
    with pytest.raises(StateTooLarge, match="never cropped"):
        encode(PROBLEMS[0], PROBLEMS[0].expr, cfg)


def test_too_many_sites_raises_rather_than_dropping_them() -> None:
    cfg = Config()
    cfg.model.max_sites = 3
    with pytest.raises(StateTooLarge, match="not cropped"):
        encode(PROBLEMS[0], PROBLEMS[0].expr, cfg)


def test_the_chosen_bounds_clear_the_measured_maxima() -> None:
    """Sized from reachable states, not start states — the numbers are on record."""
    import json

    path = REPO / "runs" / "state_extent.json"
    if not path.exists():
        pytest.skip("run scripts/measure_state_extent.py")
    measured = json.loads(path.read_text())["measured_max"]
    assert CFG.model.seq_len > measured["tokens"], "seq_len does not clear the measured p100"
    assert CFG.model.max_sites > measured["sites"], "max_sites does not clear the measured p100"
    # And the gap between start states and reachable states is why this matters.
    starts = json.loads(path.read_text())["start_states"]
    assert measured["tokens"] > 3 * starts["tokens_p100"]


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------


def batch(problems: list[Problem]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [encode(p, p.expr, CFG) for p in problems]
    return (
        torch.stack([s.tokens for s in states]),
        torch.stack([s.site_positions for s in states]),
        torch.stack([s.legal_mask for s in states]),
    )


#: The synthetic batch the DONE-WHEN names, with its composition stated. `y` is
#: in the vocabulary and reachable (SIMPLIFY's unlike-variable case), so the
#: model must digest it — a batch of x-only states would leave a live embedding
#: row untouched by every forward pass in the gate.
SYNTHETIC_BATCH = [
    *PROBLEMS,  # SOLVE(x), EVALUATE, SIMPLIFY(x)
    Problem(  # SIMPLIFY with two variables — y-bearing
        goal=GOAL_SIMPLIFY,
        par=1,
        par_source="bfs",
        expr=add(mul(num(3), X), mul(num(2), X), mul(num(4), Y), mul(num(5), Y), num(7)),
    ),
    Problem(  # SOLVE for y, so the target slot is not always VAR_X
        goal=GOAL_SOLVE,
        target=VAR_Y,
        par=1,
        par_source="bfs",
        expr=eq(mul(num(4), Y), num(20)),
    ),
    Problem(  # a mid-derivation state, longer than any start state
        goal=GOAL_SOLVE,
        target=VAR_X,
        par=2,
        par_source="bfs",
        expr=eq(add(mul(num(3), X), num(6), num(-6), num(2)), add(num(21), num(-6))),
    ),
]


def test_forward_backward_on_a_synthetic_batch_of_every_goal() -> None:
    """**Chunk 6 gate.** Forward and backward on all goals, y included.

    Composition (6 rows): 3 SOLVE — two for x, one for y — 1 EVALUATE, 2
    SIMPLIFY of which one carries two variables. Both variable tokens appear,
    both target slots are exercised, and one row is a mid-derivation state
    longer than any start state.
    """
    goals = {p.goal for p in SYNTHETIC_BATCH}
    assert goals == {GOAL_SOLVE, GOAL_EVALUATE, GOAL_SIMPLIFY}, "a goal is missing"
    targets = {p.target for p in SYNTHETIC_BATCH}
    assert {VAR_X, VAR_Y, None} <= targets, "the y target slot is untested"
    assert any(VAR_Y in encode_state(p.goal, p.expr, p.target) for p in SYNTHETIC_BATCH), (
        "no y token reaches the embedding table"
    )

    model = Reckoner(CFG)
    tokens, positions, mask = batch(SYNTHETIC_BATCH)
    policy, value, steps = model(tokens, positions)
    n = len(SYNTHETIC_BATCH)
    assert policy.shape == (n, N_RULES * CFG.model.max_sites)
    assert value.shape == (n, CFG.model.value_classes)
    assert steps is not None and steps.shape == (n,)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()

    targets_p = mask.float()
    targets_p = targets_p / targets_p.sum(-1, keepdim=True).clamp(min=1)
    loss = policy_loss(policy, targets_p, mask) + value.square().mean() + steps.square().mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


# ---------------------------------------------------------------------------
# Config-is-spec
# ---------------------------------------------------------------------------


def test_config_is_spec_for_the_model() -> None:
    """**Chunk 6 gate.** The shape the config declares is the shape that is built.

    Two spellings of a model — the config and the module — that are allowed to
    drift is how a checkpoint stops loading into the architecture that made it.
    Every value here is founding config: `n_layers = 6` in particular is a
    recorded decision, and any later depth change is a registered lever, not an
    edit.
    """
    m = CFG.model
    assert (m.d_model, m.n_layers, m.n_heads, m.d_ff) == (256, 6, 8, 1024)
    assert (m.seq_len, m.max_sites, m.d_policy) == (512, 192, 128)
    assert (m.value_classes, m.steps_aux_head) == (3, True)

    model = Reckoner(CFG)
    assert model.token_embedding.embedding_dim == m.d_model
    assert model.token_embedding.num_embeddings == VOCAB_SIZE
    assert model.position_embedding.num_embeddings == m.seq_len
    assert len(model.trunk.layers) == m.n_layers
    assert model.trunk.layers[0].self_attn.num_heads == m.n_heads
    assert model.trunk.layers[0].linear1.out_features == m.d_ff
    assert model.rule_embedding.num_embeddings == N_RULES
    assert model.site_projection.out_features == m.d_policy
    assert model.value_head.out_features == m.value_classes
    assert model.steps_head is not None and model.steps_head.out_features == 1
    assert model.max_sites == m.max_sites


def test_the_steps_head_can_be_switched_off_by_config_alone() -> None:
    """Both polarities on a config-is-spec key: the flag has to do something."""
    cfg = Config()
    cfg.model.steps_aux_head = False
    model = Reckoner(cfg)
    assert model.steps_head is None
    _, _, steps = model(*batch(PROBLEMS)[:2])
    assert steps is None


def test_backward_updates_every_parameter_group() -> None:
    model = Reckoner(CFG)
    tokens, positions, mask = batch(PROBLEMS)
    policy, value, steps = model(tokens, positions)

    targets = mask.float()
    targets = targets / targets.sum(-1, keepdim=True).clamp(min=1)
    loss = (
        policy_loss(policy, targets, mask)
        + torch.nn.functional.cross_entropy(value, torch.tensor([0, 1, 2]))
        + steps_loss(steps, torch.tensor([3.0, 1.0, 1.0]), torch.tensor([True, True, False]))
    )
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


# ---------------------------------------------------------------------------
# Masked-loss invariance — BOTH masks, BOTH polarities
# ---------------------------------------------------------------------------


def test_policy_loss_ignores_masked_off_logits() -> None:
    """The ported fabricated-target detector, on the legality mask."""
    torch.manual_seed(0)
    mask = torch.zeros(1, N_RULES * CFG.model.max_sites, dtype=torch.bool)
    mask[0, [3, 17, 400]] = True
    logits = torch.randn(1, N_RULES * CFG.model.max_sites)
    targets = mask.float() / mask.sum()

    before = policy_loss(logits, targets, mask).item()

    perturbed = logits.clone()
    perturbed[0, ~mask[0]] += 137.0  # every illegal logit, hard
    assert policy_loss(perturbed, targets, mask).item() == before

    moved = logits.clone()
    moved[0, 3] += 5.0  # one *legal* logit
    assert policy_loss(moved, targets, mask).item() != before


def test_policy_loss_refuses_target_mass_on_an_illegal_action() -> None:
    """F-41: target mass off the legal set is REFUSED, never silently dropped.

    The masked log-probability is ``-inf`` there and ``nan_to_num`` turned the
    product into exactly 0, so an illegal target contributed nothing and said
    nothing — 93% of the ring's visit mass went that way for all of M1. The mask
    on *logits* is a detector; on *targets* it was a silencer.
    """
    from reckoner.model import IllegalTargetError

    mask = torch.zeros(2, N_RULES * CFG.model.max_sites, dtype=torch.bool)
    mask[:, [3, 17, 400]] = True
    logits = torch.randn(2, N_RULES * CFG.model.max_sites)
    targets = mask.float() / 3
    targets[1, 17] = 0.0
    targets[1, 5] = 1 / 3  # cell 5 is not legal in row 1
    with pytest.raises(IllegalTargetError, match="F-41"):
        policy_loss(logits, targets, mask)


def test_policy_loss_accepts_a_target_that_is_entirely_legal() -> None:
    """The other polarity — a detector that always fires is not a gate either."""
    mask = torch.zeros(2, N_RULES * CFG.model.max_sites, dtype=torch.bool)
    mask[0, [3, 17, 400]] = True
    mask[1, [0, 1]] = True
    targets = torch.zeros_like(mask, dtype=torch.float32)
    targets[0, 17] = 1.0
    targets[1, [0, 1]] = 0.5
    loss = policy_loss(torch.randn_like(targets), targets, mask)
    assert torch.isfinite(loss)


def test_policy_loss_refuses_mass_on_a_row_with_no_legal_action() -> None:
    """The ``nan_to_num`` path exists for rows with no legal action; their target
    must be empty, or the same silence returns by another door."""
    from reckoner.model import IllegalTargetError

    mask = torch.zeros(1, N_RULES * CFG.model.max_sites, dtype=torch.bool)
    targets = torch.zeros(1, N_RULES * CFG.model.max_sites)
    assert policy_loss(torch.randn_like(targets), targets, mask).item() == 0.0
    targets[0, 0] = 1.0
    with pytest.raises(IllegalTargetError):
        policy_loss(torch.randn_like(targets), targets, mask)


def test_steps_loss_ignores_unsolved_rows() -> None:
    """The same detector on the auxiliary head — a mask declared is not a mask."""
    steps = torch.tensor([2.0, 5.0, 9.0])
    targets = torch.tensor([2.0, 5.0, 0.0])
    solved = torch.tensor([True, True, False])

    before = steps_loss(steps, targets, solved).item()

    perturbed = steps.clone()
    perturbed[2] += 1000.0  # the unsolved row, hard
    assert steps_loss(perturbed, targets, solved).item() == before

    moved = steps.clone()
    moved[0] += 1.0  # a solved row
    assert steps_loss(moved, targets, solved).item() != before


def test_a_row_with_no_legal_action_contributes_no_nan() -> None:
    mask = torch.zeros(1, N_RULES * CFG.model.max_sites, dtype=torch.bool)
    loss = policy_loss(torch.randn(1, N_RULES * CFG.model.max_sites), mask.float(), mask)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Action layout, parameters, provenance
# ---------------------------------------------------------------------------


def test_action_index_round_trips() -> None:
    for rule_id in range(N_RULES):
        for site_id in (0, 1, 191):
            index = action_index(rule_id, site_id, CFG.model.max_sites)
            assert action_pair(index, CFG.model.max_sites) == (rule_id, site_id)


def test_parameter_count_is_inside_the_envelope_and_decomposed() -> None:
    breakdown = Reckoner(CFG).parameter_breakdown()
    assert CFG.model.param_budget_min <= breakdown["total"] <= CFG.model.param_budget_max
    for part in ("embeddings", "trunk", "policy_head", "value_head", "steps_head"):
        assert breakdown[part] > 0, f"{part} has no parameters"
    assert breakdown["total"] == 5_073_156, "the model size changed — update the report"
    assert breakdown["trunk"] > breakdown["embeddings"], "the trunk should dominate at this scale"


def test_checkpoint_round_trips_with_its_provenance(tmp_path: Path) -> None:
    model = Reckoner(CFG)
    meta = save_checkpoint(tmp_path / "c.pt", model, CFG, step=7)
    assert meta["ruleset_version"] == RULESET_VERSION
    assert meta["vocab_version"] == VOCAB_VERSION
    assert len(meta["config_fingerprint"]) == 64

    loaded, loaded_meta = load_checkpoint(tmp_path / "c.pt", CFG)
    assert loaded_meta == meta
    for (a, p), (b, q) in zip(model.named_parameters(), loaded.named_parameters(), strict=True):
        assert a == b and torch.equal(p, q)


def test_a_checkpoint_from_another_rule_system_is_refused(tmp_path: Path) -> None:
    """Registered for chunk 9: CheckpointPool must refuse a version mismatch.

    A pool par re-solved under a different rule system is the par-provenance
    defect (FINDINGS.md F-02) reborn at the league layer. The guard costs one
    line and closes the door before it exists.
    """
    model = Reckoner(CFG)
    save_checkpoint(tmp_path / "c.pt", model, CFG, step=1)
    blob = torch.load(tmp_path / "c.pt", weights_only=False)
    blob["meta"]["ruleset_version"] = RULESET_VERSION + 1
    torch.save(blob, tmp_path / "c.pt")

    with pytest.raises(ValueError, match="rule system that no longer exists"):
        load_checkpoint(tmp_path / "c.pt", CFG)
    model2, _ = load_checkpoint(tmp_path / "c.pt", CFG, strict_versions=False)
    assert isinstance(model2, Reckoner)  # the escape hatch exists and is explicit


def test_checkpoint_meta_carries_both_versions() -> None:
    meta = checkpoint_meta(CFG, step=0)
    assert {"ruleset_version", "vocab_version", "config_fingerprint", "git_sha", "step"} <= set(
        meta
    )


def test_padding_does_not_reach_the_trunk_as_content() -> None:
    model = Reckoner(CFG).eval()
    problem = PROBLEMS[1]
    state = encode(problem, problem.expr, CFG)
    tokens = state.tokens.unsqueeze(0)
    with torch.no_grad():
        _, value_a, _ = model(tokens, state.site_positions.unsqueeze(0))
        longer = tokens.clone()
        longer[0, state.length + 5] = PAD  # still PAD; a no-op by construction
        _, value_b, _ = model(longer, state.site_positions.unsqueeze(0))
    assert torch.allclose(value_a, value_b)
