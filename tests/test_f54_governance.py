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
    eval_profile,
)
from reckoner.config import Config, _dataclass_to_dict, config_fingerprint, fingerprint_of

REPO = Path(__file__).resolve().parents[1]


def test_the_pinned_fingerprints_are_todays_config() -> None:
    """The door's two pins, asserted where a clean clone runs them."""
    assert config_fingerprint(Config()) == CAMPAIGN_FINGERPRINT
    assert config_fingerprint(eval_profile(Config())) == EVAL_FINGERPRINT


def _delete(tree: dict, dotted: str) -> None:
    group, key = dotted.split(".")
    del tree[group][key]


def test_every_pin_move_since_m1_was_exactly_one_key() -> None:
    """Walk the history backwards. Starting from today's config, deleting the keys
    that were added, newest first, recovers every pair the door ever pinned,
    exactly and for both profiles, down to M1's. So each move was its key and
    nothing else, and no value has moved since M1 without a key to show for it.

    F-54's own move is the second link, and WEAK-w1's two levers are the last two.
    """
    assert [era["then_added"] for era in FINGERPRINT_HISTORY] == [
        "campaign.device",
        "train.policy_target",
        "search.past_par",
        "search.value_source",
    ]
    for cfg, now, profile in (
        (Config(), CAMPAIGN_FINGERPRINT, "campaign"),
        (eval_profile(Config()), EVAL_FINGERPRINT, "eval"),
    ):
        tree = _dataclass_to_dict(cfg)
        assert fingerprint_of(tree) == now
        for era in reversed(FINGERPRINT_HISTORY):
            _delete(tree, era["then_added"])
            assert fingerprint_of(tree) == era[profile], (
                f"deleting {era['then_added']} does not recover the {era['era']} {profile} pair"
            )
    assert DEVICE_ERA_FINGERPRINTS["campaign"] == FINGERPRINT_HISTORY[1]["campaign"]
    assert M1_ERA_FINGERPRINTS["eval"] == FINGERPRINT_HISTORY[0]["eval"]


def test_the_proof_would_catch_a_second_moved_key() -> None:
    """The other polarity, and one that can fail: had any other value moved beside
    the added keys, no link of the chain would be recovered."""
    cfg = Config()
    moved = replace(cfg, train=replace(cfg.train, lr=cfg.train.lr * 2))
    tree = _dataclass_to_dict(moved)
    for era in reversed(FINGERPRINT_HISTORY):
        _delete(tree, era["then_added"])
        assert fingerprint_of(tree) != era["campaign"]


def test_a_new_key_at_its_default_is_inert_until_it_is_set() -> None:
    """WEAK-w1's levers land at the search M1 ran. Setting either one moves the
    fingerprint, which is what makes pulling it a registered act."""
    cfg = Config()
    assert cfg.search.past_par == "evaluate" and cfg.search.value_source == "wdl"
    for change in ({"past_par": "loss"}, {"value_source": "steps"}):
        pulled = replace(cfg, search=replace(cfg.search, **change))
        assert config_fingerprint(pulled) != CAMPAIGN_FINGERPRINT


def test_m1_literals_are_the_recorded_pair() -> None:
    """The M1 literals are quoted from the record, never recomputed."""
    page = (REPO / "PREREG-m1.md").read_text()
    assert M1_CAMPAIGN_FINGERPRINT[:16] in page
    assert M1_EVAL_FINGERPRINT[:16] in page
