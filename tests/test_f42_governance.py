"""F-42's governance, in a module that runs in a clean clone.

These first lived in ``test_campaign_driver.py``, whose module-wide
``needs_anchor`` mark skips everything when the gitignored Phase-1 anchor is
absent — so in every clean clone the proof that F-42 moved exactly one key was
skipped and reported as nothing at all (F-42 review). None of these needs the
anchor: they read the config, the constants and the record.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from reckoner.campaign import (
    CAMPAIGN_FINGERPRINT,
    EVAL_FINGERPRINT,
    M1_CAMPAIGN_FINGERPRINT,
    M1_EVAL_FINGERPRINT,
    eval_profile,
)
from reckoner.config import Config, _dataclass_to_dict, config_fingerprint, fingerprint_of

REPO = Path(__file__).resolve().parents[1]


def test_the_pinned_fingerprints_are_todays_config() -> None:
    """The door's two pins, asserted where a clean clone runs them."""
    assert config_fingerprint(Config()) == CAMPAIGN_FINGERPRINT
    assert config_fingerprint(eval_profile(Config())) == EVAL_FINGERPRINT


def test_the_f42_move_changed_exactly_one_key() -> None:
    """Deleting ``train.policy_target`` from today's config recovers M1's pair
    exactly, for both profiles: nothing else moved under the same move."""
    for cfg, now, m1 in (
        (Config(), CAMPAIGN_FINGERPRINT, M1_CAMPAIGN_FINGERPRINT),
        (eval_profile(Config()), EVAL_FINGERPRINT, M1_EVAL_FINGERPRINT),
    ):
        tree = _dataclass_to_dict(cfg)
        assert fingerprint_of(tree) == now
        del tree["train"]["policy_target"]
        assert fingerprint_of(tree) == m1


def test_the_proof_would_catch_a_second_moved_key() -> None:
    """The other polarity, and one that can fail: had any second value moved
    beside the new key, deleting the key alone would NOT recover M1."""
    cfg = Config()
    moved = replace(cfg, train=replace(cfg.train, lr=cfg.train.lr * 2))
    tree = _dataclass_to_dict(moved)
    del tree["train"]["policy_target"]
    assert fingerprint_of(tree) != M1_CAMPAIGN_FINGERPRINT


def test_m1_literals_are_the_recorded_pair() -> None:
    """The M1 literals are quoted from the record, never recomputed."""
    page = (REPO / "PREREG-m1.md").read_text()
    assert M1_CAMPAIGN_FINGERPRINT[:16] in page
    assert M1_EVAL_FINGERPRINT[:16] in page
