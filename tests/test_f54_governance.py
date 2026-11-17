"""F-54's governance, in a module that runs in a clean clone.

These first lived in ``test_campaign_driver.py``, whose module-wide
``needs_anchor`` mark skips everything when the gitignored Phase-1 anchor is
absent — so in every clean clone the proof that F-54 moved exactly one key was
skipped and reported as nothing at all (F-54 review). None of these needs the
anchor: they read the config, the constants and the record.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from reckoner.campaign import (
    CAMPAIGN_FINGERPRINT,
    DEVICE_ERA_FINGERPRINTS,
    EVAL_FINGERPRINT,
    FINGERPRINT_HISTORY,
    M1_CAMPAIGN_FINGERPRINT,
    M1_ERA_FINGERPRINTS,
    M1_EVAL_FINGERPRINT,
    describe_move,
    eval_profile,
)
from reckoner.config import (
    M1_SEARCH,
    Config,
    _dataclass_to_dict,
    config_diff,
    config_fingerprint,
    fingerprint_of,
    with_m1_search,
)

REPO = Path(__file__).resolve().parents[1]


def test_the_pinned_fingerprints_are_todays_config() -> None:
    """The door's two pins, asserted where a clean clone runs them."""
    assert config_fingerprint(Config()) == CAMPAIGN_FINGERPRINT
    assert config_fingerprint(eval_profile(Config())) == EVAL_FINGERPRINT


def _undo(tree: dict, era: dict) -> None:
    """Take one move back off a config tree: delete the key it added, or set the
    defaults it moved back to what they were."""
    if "then_added" in era:
        group, key = era["then_added"].split(".")
        del tree[group][key]
        return
    for dotted, (was, became) in era["then_moved"].items():
        group, key = dotted.split(".")
        assert tree[group][key] == became, f"{dotted} is not at the value the move left it"
        tree[group][key] = was


def test_every_pin_move_since_m1_was_exactly_what_its_entry_says() -> None:
    """Walk the history backwards. Starting from today's config, undoing each move,
    newest first (setting a moved default back, deleting an added key), recovers
    every pair the door ever pinned, exactly and for both profiles, down to M1's.
    So each move was what its entry says and nothing else, and no value has
    moved since M1 without an entry to show for it.

    F-54's own move is the second link, WEAK-w1's two levers are the next two, and
    the ruling that made the past-par rule the default is the last.
    """
    assert [era.get("then_added") or dict(era["then_moved"]) for era in FINGERPRINT_HISTORY] == [
        "campaign.device",
        "train.policy_target",
        "search.past_par",
        "search.value_source",
        {"search.past_par": ("evaluate", "loss")},
    ]
    assert all(("then_added" in era) != ("then_moved" in era) for era in FINGERPRINT_HISTORY)
    for cfg, now, profile in (
        (Config(), CAMPAIGN_FINGERPRINT, "campaign"),
        (eval_profile(Config()), EVAL_FINGERPRINT, "eval"),
    ):
        tree = _dataclass_to_dict(cfg)
        assert fingerprint_of(tree) == now
        for era in reversed(FINGERPRINT_HISTORY):
            _undo(tree, era)
            assert fingerprint_of(tree) == era[profile], (
                f"undoing '{describe_move(era)}' does not recover the {era['era']} {profile} pair"
            )
    assert DEVICE_ERA_FINGERPRINTS["campaign"] == FINGERPRINT_HISTORY[1]["campaign"]
    assert M1_ERA_FINGERPRINTS["eval"] == FINGERPRINT_HISTORY[0]["eval"]


def test_the_proof_would_catch_a_second_moved_key() -> None:
    """The other polarity, and one that can fail: had any other value moved beside
    the recorded moves, no link of the chain would be recovered."""
    cfg = Config()
    moved = replace(cfg, train=replace(cfg.train, lr=cfg.train.lr * 2))
    tree = _dataclass_to_dict(moved)
    for era in reversed(FINGERPRINT_HISTORY):
        _undo(tree, era)
        assert fingerprint_of(tree) != era["campaign"]


def test_the_past_par_rule_is_the_default_and_the_value_source_is_held() -> None:
    """The ruling of 2026-11-16 (WEAK-w1.md) named both keys. The past-par rule is
    the default. The steps-head value is held at the search M1 ran until the
    principal rules on the amendment of 2026-11-17. The search every earlier
    round ran is one call away, at the pair the door pinned before the move."""
    cfg = Config()
    assert (cfg.search.past_par, cfg.search.value_source) == ("loss", "wdl")
    old = with_m1_search(cfg)
    assert (old.search.past_par, old.search.value_source) == ("evaluate", "wdl")
    assert M1_SEARCH == {"past_par": "evaluate", "value_source": "wdl"}
    assert config_diff(old, cfg) == {"search.past_par": ("evaluate", "loss")}
    assert config_fingerprint(old) == FINGERPRINT_HISTORY[-1]["campaign"]
    assert config_fingerprint(eval_profile(old)) == FINGERPRINT_HISTORY[-1]["eval"]


def test_pulling_either_search_key_is_its_own_config() -> None:
    """Whatever a search key is set to other than today's default, the config is
    neither today's nor one the door pinned by accident: setting the past-par
    rule back is exactly the previous pair, and the steps-head value is a pair
    the door has never pinned."""
    cfg = Config()
    pinned = {CAMPAIGN_FINGERPRINT} | {era["campaign"] for era in FINGERPRINT_HISTORY}
    back = replace(cfg, search=replace(cfg.search, past_par="evaluate"))
    assert config_fingerprint(back) == FINGERPRINT_HISTORY[-1]["campaign"]
    steps = replace(cfg, search=replace(cfg.search, value_source="steps"))
    assert config_fingerprint(steps) not in pinned


def test_m1_literals_are_the_recorded_pair() -> None:
    """The M1 literals are quoted from the record, never recomputed."""
    page = (REPO / "PREREG-m1.md").read_text()
    assert M1_CAMPAIGN_FINGERPRINT[:16] in page
    assert M1_EVAL_FINGERPRINT[:16] in page
