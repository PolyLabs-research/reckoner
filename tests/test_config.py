"""The config loader is a detector, so it is tested on both polarities.

Inherited law: every detector that gates automation is validated on both
polarities before live use. A strict loader that rejects everything passes a
"rejects unknown keys" test and is useless; a lenient one that accepts
everything passes a "loads the defaults" test and is worse than useless. Each
guard below therefore has a matching case that must still LOAD.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reckoner.config import (
    Config,
    EpisodeConfig,
    GeneratorConfig,
    LadderConfig,
    LeagueConfig,
    ModelConfig,
    ParConfig,
    SearchConfig,
    TrainConfig,
    _dataclass_to_dict,
    _flatten,
    config_diff,
    config_fingerprint,
    load_config,
    save_config,
    validate,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "configs" / "default.yaml"


def _write(tmp_path: Path, data: dict) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_default_config_instantiates() -> None:
    cfg = Config()
    assert isinstance(cfg.episode, EpisodeConfig)
    assert isinstance(cfg.par, ParConfig)
    assert isinstance(cfg.league, LeagueConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.search, SearchConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.generator, GeneratorConfig)
    assert isinstance(cfg.ladder, LadderConfig)


def test_nested_sections_are_reconstructed_not_left_as_dicts(tmp_path: Path) -> None:
    """The `from __future__ import annotations` trap.

    With raw string annotations, `dataclasses.is_dataclass(field.type)` is False
    for every nested section, so each one silently loads as a plain dict and the
    config that loads is not the config that was written. The failure is quiet:
    `cfg.search.sims` raises AttributeError only when search finally runs.
    """
    cfg = load_config(_write(tmp_path, {"search": {"sims": 16}}))
    assert isinstance(cfg.search, SearchConfig)
    assert not isinstance(cfg.search, dict)
    assert cfg.search.sims == 16
    assert cfg.search.gumbel_m == 16  # untouched sibling keeps its default


# ---------------------------------------------------------------------------
# Pinned values — the numbers the spec and plan actually fix
# ---------------------------------------------------------------------------


def test_pinned_episode_and_par_values() -> None:
    e, p, lg = EpisodeConfig(), ParConfig(), LeagueConfig()
    assert e.step_cap == 24  # plan chunk 3
    assert e.simplify_equiv_k == 32  # spec §3
    assert p.bfs_exact_max_depth == 6  # spec §3
    assert lg.par_from_pool_frac == 0.20  # amendment v1.1
    assert p.concede_enabled is False  # v1.1: implemented, default off


def test_pinned_value_head_shape_is_the_par_game_shape() -> None:
    """Amendment v1.1 reversed §8 decision 1: 3-class W/D/L vs par, not 2-class."""
    m = ModelConfig()
    assert m.value_classes == 3
    assert m.steps_aux_head is True
    assert (m.param_budget_min, m.param_budget_max) == (2_000_000, 7_000_000)


def test_pinned_single_agent_perspective() -> None:
    """Plan §8 decision 5 — no opponent in the tree, so backup does not negate."""
    assert SearchConfig().perspective == "single"


def test_pinned_train_values() -> None:
    t = TrainConfig()
    assert t.value_q_mse_weight == 0.5  # chunk 9: on by default
    assert t.rehearsal_frac == 0.65  # M1-A4 §5: the sweep's value, taken mechanically


def test_pinned_generator_and_ladder_values() -> None:
    g, ladder = GeneratorConfig(), LadderConfig()
    assert g.max_bfs_depth == 6
    assert g.train_set_size == 100_000
    assert g.suite_problems_per_depth == 200
    assert g.suite_depths == [1, 2, 3, 4, 5, 6]
    assert ladder.ladder_every == 5  # plan chunk 11


# ---------------------------------------------------------------------------
# configs/default.yaml is the same config, spelled twice — and must stay so
# ---------------------------------------------------------------------------


def test_default_yaml_exists_and_loads() -> None:
    assert DEFAULT_YAML.exists(), "configs/default.yaml is a chunk 0 deliverable"
    load_config(DEFAULT_YAML)


def test_default_yaml_matches_dataclass_defaults() -> None:
    """Config-is-spec, chunk 0 flavour.

    Two writable spellings of one number that are allowed to drift is how a run
    ends up honouring the copy nobody edited. A campaign that needs different
    values gets its own file in configs/; it does not edit default.yaml.
    """
    assert load_config(DEFAULT_YAML) == Config()


def test_default_yaml_is_complete() -> None:
    """Every field appears in the YAML, so nothing is settable only in code."""
    raw = yaml.safe_load(DEFAULT_YAML.read_text())
    expected = _dataclass_to_dict(Config())

    def walk(got: dict, want: dict, prefix: str = "") -> list[str]:
        missing = []
        for key, want_val in want.items():
            dotted = f"{prefix}{key}"
            if key not in got:
                missing.append(dotted)
            elif isinstance(want_val, dict):
                missing += walk(got[key], want_val, f"{dotted}.")
        return missing

    assert walk(raw, expected) == []


# ---------------------------------------------------------------------------
# Unknown keys are a hard error — both polarities
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown config key 'sedd'"):
        load_config(_write(tmp_path, {"sedd": 42}))


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unknown config key 'search\.simz'"):
        load_config(_write(tmp_path, {"search": {"simz": 16}}))


def test_near_miss_key_is_rejected_with_a_suggestion(tmp_path: Path) -> None:
    """The dangerous case: a plausible name that leaves a plausible default.

    `value_q_mse` next to `value_q_mse_weight`. Accepting it would leave the
    blend at 0.5 while reading, in the config file and in the RUNLOG, as a
    request to change it — the run looks fine and the number is wrong.
    """
    with pytest.raises(ValueError) as exc:
        load_config(_write(tmp_path, {"train": {"value_q_mse": 0.0}}))
    assert "train.value_q_mse" in str(exc.value)
    assert "value_q_mse_weight" in str(exc.value)


def test_known_keys_are_accepted(tmp_path: Path) -> None:
    """The other polarity: a loader that rejects everything is not strict, it is broken."""
    cfg = load_config(
        _write(
            tmp_path,
            {
                "seed": 7,
                "episode": {"step_cap": 32},
                "league": {"par_from_pool_frac": 0.5},
                "search": {"sims": 16, "gumbel_m": 5},
                "train": {"rehearsal_frac": 0.25},
            },
        )
    )
    assert cfg.seed == 7
    assert cfg.episode.step_cap == 32
    assert cfg.league.par_from_pool_frac == 0.5
    assert cfg.search.sims == 16
    assert cfg.search.gumbel_m == 5
    assert cfg.train.rehearsal_frac == 0.25


def test_empty_yaml_loads_as_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(str(path)) == Config()


def test_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(str(path))


# ---------------------------------------------------------------------------
# Overrides get the same strictness as files
# ---------------------------------------------------------------------------


def test_overrides_apply() -> None:
    cfg = load_config(overrides={"search.sims": 6, "seed": 99})
    assert cfg.search.sims == 6
    assert cfg.seed == 99
    assert cfg.search.gumbel_m == 16  # sibling untouched


def test_override_on_a_file_wins() -> None:
    cfg = load_config(DEFAULT_YAML, overrides={"episode.step_cap": 8})
    assert cfg.episode.step_cap == 8


def test_unknown_override_key_is_rejected() -> None:
    """A flag mistyped on the command line is the same failure as a key mistyped in YAML."""
    with pytest.raises(ValueError, match="unknown config key 'simz'"):
        load_config(overrides={"search.simz": 16})


def test_override_through_a_non_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a config section"):
        load_config(overrides={"seed.nested": 1})


# ---------------------------------------------------------------------------
# Validators — each on both polarities
# ---------------------------------------------------------------------------


def test_alternating_perspective_is_rejected(tmp_path: Path) -> None:
    """Plan §8 decision 5, enforced rather than remembered."""
    with pytest.raises(ValueError, match="no opponent in the tree"):
        load_config(_write(tmp_path, {"search": {"perspective": "alternating"}}))


def test_single_perspective_is_accepted(tmp_path: Path) -> None:
    assert (
        load_config(_write(tmp_path, {"search": {"perspective": "single"}})).search.perspective
        == "single"
    )


def test_two_class_value_head_is_rejected(tmp_path: Path) -> None:
    """A silent revert of amendment v1.1 back to the superseded §8 decision 1."""
    with pytest.raises(ValueError, match="3-class W/D/L vs par"):
        load_config(_write(tmp_path, {"model": {"value_classes": 2}}))


def test_three_class_value_head_is_accepted(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, {"model": {"value_classes": 3}})).model.value_classes == 3


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_pool_frac_is_rejected(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="from_pool_frac"):
        load_config(_write(tmp_path, {"league": {"par_from_pool_frac": bad}}))


@pytest.mark.parametrize("ok", [0.0, 0.2, 1.0])
def test_in_range_pool_frac_is_accepted(tmp_path: Path, ok: float) -> None:
    cfg = load_config(_write(tmp_path, {"league": {"par_from_pool_frac": ok}}))
    assert cfg.league.par_from_pool_frac == ok


def test_rehearsal_frac_of_one_is_rejected(tmp_path: Path) -> None:
    """rehearsal_frac == 1.0 means a Phase 2 iteration that trains on no Phase 2 data."""
    with pytest.raises(ValueError, match="rehearsal_frac"):
        load_config(_write(tmp_path, {"train": {"rehearsal_frac": 1.0}}))


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_step_cap_is_rejected(tmp_path: Path, bad: int) -> None:
    with pytest.raises(ValueError, match="step_cap"):
        load_config(_write(tmp_path, {"episode": {"step_cap": bad}}))


def test_zero_equiv_draws_is_rejected(tmp_path: Path) -> None:
    """k=0 draws is a SIMPLIFY checker that accepts everything."""
    with pytest.raises(ValueError, match="simplify_equiv_k"):
        load_config(_write(tmp_path, {"episode": {"simplify_equiv_k": 0}}))


def test_suite_depth_beyond_the_bfs_horizon_is_rejected(tmp_path: Path) -> None:
    """A frozen instrument whose depth label and par are unverifiable."""
    with pytest.raises(ValueError, match="BFS-verified"):
        load_config(_write(tmp_path, {"generator": {"suite_depths": [1, 7]}}))


def test_suite_depths_within_the_horizon_are_accepted(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, {"generator": {"suite_depths": [1, 2, 3]}}))
    assert cfg.generator.suite_depths == [1, 2, 3]


def test_inverted_param_budget_is_rejected() -> None:
    cfg = Config()
    cfg.model.param_budget_min = 9_000_000
    with pytest.raises(ValueError, match="param_budget_min"):
        validate(cfg)


# ---------------------------------------------------------------------------
# Round-trip and fingerprint
# ---------------------------------------------------------------------------


def test_yaml_round_trip(tmp_path: Path) -> None:
    path = str(tmp_path / "cfg.yaml")
    save_config(Config(), path)
    assert load_config(path) == Config()


def test_yaml_round_trip_modified(tmp_path: Path) -> None:
    cfg = Config()
    cfg.seed = 7
    cfg.episode.step_cap = 12
    cfg.search.sims = 31
    cfg.train.lr = 1e-3
    cfg.generator.suite_depths = [2, 4]

    path = str(tmp_path / "cfg.yaml")
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "dir" / "config.yaml"
    save_config(Config(), path)
    assert path.exists()


# ---------------------------------------------------------------------------
# One spelling per referent, and the lever-list enforcement it enables
# ---------------------------------------------------------------------------


def test_par_from_pool_frac_is_spelled_verbatim() -> None:
    """The key is spelled exactly as the plan, the amendment and the PREREGs spell it.

    A config key's name is not private to the config system. `grep
    par_from_pool_frac` has to hit documents, briefs, configs and code as one
    corpus, and two spellings of one referent is a defect class that has already
    cost this project once (`stockfish:0` vs `stockfish_s0`). The loader's
    near-miss suggestion guards config-space only; it cannot guard the half of
    the corpus written in English.
    """
    flat = _flatten(_dataclass_to_dict(Config()))
    assert "league.par_from_pool_frac" in flat
    assert not [k for k in flat if k.endswith("from_pool_frac") and "par_from_pool_frac" not in k]


def test_config_diff_is_empty_for_identical_configs() -> None:
    assert config_diff(Config(), Config()) == {}
    assert config_diff(Config(), load_config(DEFAULT_YAML)) == {}


def test_config_diff_names_every_changed_key_and_only_those() -> None:
    """The registered enforcement for one-lever-per-round.

    A campaign config's diff against the defaults IS its lever list, so a
    campaign test can assert the diff equals the set its PREREG registered. A
    second lever added quietly mid-run then fails the build, and fails it by
    name.
    """
    other = load_config(overrides={"search.sims": 16, "seed": 7})
    diff = config_diff(Config(), other)
    assert set(diff) == {"search.sims", "seed"}
    assert diff["search.sims"] == (48, 16)
    assert diff["seed"] == (42, 7)


def test_config_diff_reaches_every_section() -> None:
    """Both polarities: a differ blind to a section would report a clean lever list."""
    cfg = Config()
    cfg.episode.step_cap = 1
    cfg.par.concede_k = 99
    cfg.league.par_from_pool_frac = 0.0
    cfg.model.d_model = 8
    cfg.search.sims = 6
    cfg.train.lr = 1.0
    cfg.generator.train_set_size = 1
    cfg.ladder.ladder_every = 1
    assert set(config_diff(Config(), cfg)) == {
        "episode.step_cap",
        "par.concede_k",
        "league.par_from_pool_frac",
        "model.d_model",
        "search.sims",
        "train.lr",
        "generator.train_set_size",
        "ladder.ladder_every",
    }


def test_config_diff_sees_list_valued_keys() -> None:
    cfg = Config()
    cfg.generator.suite_depths = [1, 2, 3]
    assert config_diff(Config(), cfg) == {"generator.suite_depths": ([1, 2, 3, 4, 5, 6], [1, 2, 3])}


def test_fingerprint_is_stable_across_instances() -> None:
    fp = config_fingerprint(Config())
    assert fp == config_fingerprint(Config())
    assert len(fp) == 64


def test_fingerprint_changes_on_any_mutation() -> None:
    cfg = Config()
    cfg.search.sims = 16
    assert config_fingerprint(cfg) != config_fingerprint(Config())


def test_fingerprint_survives_a_round_trip(tmp_path: Path) -> None:
    """The fingerprint identifies the config, not the file that carried it."""
    path = str(tmp_path / "cfg.yaml")
    save_config(Config(), path)
    assert config_fingerprint(load_config(path)) == config_fingerprint(Config())


def test_policy_target_is_validated_both_ways() -> None:
    """F-54's key: the two named targets pass, anything else is refused by name."""
    from dataclasses import replace

    from reckoner.config import POLICY_TARGETS, validate

    cfg = Config()
    assert cfg.train.policy_target == "improved"
    for good in POLICY_TARGETS:
        validate(replace(cfg, train=replace(cfg.train, policy_target=good)))
    for bad in ("visit", "Improved", "", "pi"):
        with pytest.raises(ValueError, match="policy_target"):
            validate(replace(cfg, train=replace(cfg.train, policy_target=bad)))
