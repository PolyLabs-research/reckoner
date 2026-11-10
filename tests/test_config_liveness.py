"""**The live-config gate.** Every fingerprinted field either changes what runs,
or names why it does not — with evidence.

Chunk 8 declared this class closed: *"a dead config key is the `batch_leaves`
hazard and is not repeated."* At that moment `value_q_mse_weight` was dead at the
loop's core and `rehearsal_frac` was dead too (F-31). A named class, declared
closed, recurring twice underneath the declaration — which is why the remedy is
mechanical rather than another resolution.

**The rule, in its adopted form:** the question is not whether a field is *read*,
but whether **varying it across its legal range changes what runs.**

That formulation replaced an earlier two-tier version, because the two-tier
version passes `rehearsal_frac`: it was read by `validate()` and copied into a
results JSON, and neither is acting on it. It also makes "observable only at
campaign scale" unsayable without a demonstration.

The worked pair, because it is the whole distinction:

* `search.perspective` — **guard-live.** Legal range is exactly one value,
  enforced by `validate` and asserted by a test, so varying it across its legal
  range is impossible and varying it beyond that fails the guard. It needed an
  exemption only while the census could not see guards.
* `train.rehearsal_frac` — **failed** (until this round wired it). Legal range
  `[0, 1)`, every value passes validation, and none of them changed anything:
  *vacuously inert.*

The registry has now pruned itself twice under real pressure: `rehearsal_frac`
left it when fix 1 wired the lever, and `league.snapshot_every` left it when
F-36's fix made the driver honour the cadence. Both times the staleness test
caught it rather than a reader noticing — which is the point of having the
second polarity at all.
"""

from __future__ import annotations

import ast
from dataclasses import replace

import pytest

from reckoner.config import Config, validate
from scripts.config_census import Walker, build, config_fields, reachable

#: Fields that do not change what runs, each with the reason and the evidence.
#: An entry is a claim, and the test below checks the claims it can check —
#: `search.perspective`'s refusal is executed, not asserted.
EXEMPT: dict[str, str] = {
    # --- spec-backed, blocked on work registered elsewhere --------------------
    "ladder.bootstrap_resamples": (
        "F-30 — the paired bootstrap needs per-problem outcomes on BOTH arms. Both "
        "now exist (the campaign arm from F-33's routing, the baseline arm from "
        "Part-0d's deterministic re-run), so this becomes live the moment the "
        "bootstrap is invoked on them. Dead until then, and no longer blocked."
    ),
}

# The registry has now pruned itself three times under real pressure, each time
# caught by the staleness test rather than by a reader: `rehearsal_frac` left when
# fix 1 wired the lever, `league.snapshot_every` when F-36 made the driver honour
# the cadence, and six more when the census learned to distinguish guard-live from
# dead. A registry nobody prunes becomes a place to hide things.


def _census() -> dict[str, str]:
    """``{qualified field: status}`` — the census, as the gate consumes it."""
    fields = config_fields()
    reads, script_reads, test_reads, calls, defined = build()
    live = reachable(calls, defined)
    out = {}
    for field, group in fields.items():
        scopes = reads.get(field, set())
        if scopes & live:
            status = "LIVE"
        elif test_reads.get(field):
            status = "GUARD"
        elif scopes:
            status = "DEAD"
        elif script_reads.get(field):
            status = "REPORTED"
        else:
            status = "UNREAD"
        out[f"{group}.{field}"] = status
    return out


def test_every_fingerprinted_field_is_live_or_registered() -> None:
    """The gate. A new dead key fails at introduction, which is the only version
    of "not repeated" that is true going forward rather than retroactively."""
    census = _census()
    unexplained = sorted(
        f for f, s in census.items() if s not in ("LIVE", "GUARD") and f not in EXEMPT
    )
    assert not unexplained, (
        "fingerprinted fields that change nothing and name no reason: "
        f"{unexplained}. Either wire the field, or register it in EXEMPT with the "
        "evidence — a config key that moves the fingerprint and moves nothing else "
        "is the class chunk 8 declared closed while two of them were live."
    )


def test_the_registry_does_not_outlive_its_entries() -> None:
    """The other polarity. An exemption for a field that has since been wired is
    a stale claim, and a registry nobody prunes becomes a place to hide things.

    This fired for real: the census's reference vector was `rehearsal_frac`, and
    wiring it in this round made the assertion fail on its own success."""
    census = _census()
    stale = sorted(f for f in EXEMPT if census.get(f) in ("LIVE", "GUARD"))
    assert not stale, f"registered as inert, but now live: {stale}. Remove the entry."


def test_a_pinned_invariant_refuses_variation() -> None:
    """`search.perspective`'s exemption, executed rather than asserted.

    Its legal range is one value, so it cannot vary within legality — which is
    what makes it vacuously behavioural rather than inert. The proof is that the
    only other value is refused."""
    cfg = Config()
    assert cfg.search.perspective == "single"
    with pytest.raises(ValueError, match="perspective"):
        validate(replace(cfg, search=replace(cfg.search, perspective="alternating")))


def test_the_detector_flags_a_synthetic_dead_key() -> None:
    """**The reference vector, synthetic on purpose.**

    This check used to run inside the census against `train.rehearsal_frac`, the
    field F-31 proved dead by hand. Then this round wired that field and the
    assertion fired on its own success: a detector validated against a live repo
    field is validated against a moving target.

    So the known positive is constructed here instead — a field read only inside
    a function nothing calls, which is exactly `rehearsal_split`'s shape and
    exactly what reference-counting would miss.
    """
    source = """
def unreachable_helper(cfg):
    return cfg.train.rehearsal_frac

def reachable_helper(cfg):
    return cfg.train.lr
"""
    walker = Walker(config_fields())
    walker.visit(ast.parse(source))
    assert walker.reads["rehearsal_frac"] == {"unreachable_helper"}
    assert walker.reads["lr"] == {"reachable_helper"}

    # `unreachable_helper` is called by nothing, so a reachability pass from any
    # entry point excludes it — while a grep for the field name would find it.
    live = reachable({}, {"unreachable_helper", "reachable_helper"})
    assert "unreachable_helper" not in live, (
        "the detector reached a function nothing calls; it is counting references "
        "rather than consumption, which is the failure that hid F-31"
    )


def test_the_detector_classifies_a_guard_only_field_as_live() -> None:
    """**The known-NEGATIVE reference vector**, and the fifth correction to this
    census's scope — every one of them found by use rather than by review.

    The census once reported eight fields dead and two were enforcing a guard:
    `test_model.py` asserts the built model's parameter count against
    `model.param_budget_min/max`. Excluding tests from *reachability* is right —
    four passing assertions on a function nobody calls are not evidence of
    consumption — but excluding them from *existence* is not, because varying a
    field a guard reads does change what happens: the guard fails.

    So the detector must classify a field read ONLY by a test as guard-live, and
    only a field that is neither runtime-live nor guard-live is a deletion
    candidate. A known-positive alone could never have caught this; it takes the
    negative too, which is the argument for both-polarity vectors in one line.
    """
    source = """
def test_the_model_fits_its_envelope(cfg):
    assert cfg.model.param_budget_min <= built <= cfg.model.param_budget_max
"""
    walker = Walker(config_fields())
    walker.visit(ast.parse(source))
    assert walker.reads["param_budget_min"] == {"test_the_model_fits_its_envelope"}
    assert walker.reads["param_budget_max"] == {"test_the_model_fits_its_envelope"}


def test_the_guarded_fields_are_reported_as_guard_live() -> None:
    """End to end on the real repo: `param_budget_min` is guard-live, not dead."""
    census = _census()
    assert census["model.param_budget_min"] == "GUARD", census["model.param_budget_min"]
    assert census["model.param_budget_max"] == "GUARD"
    assert census["model.param_budget_min"] not in ("DEAD", "UNREAD"), (
        "a field a guard test enforces is not a deletion candidate"
    )


# ------------------------------ the inertness prover's known positives


#: Fields unambiguously ON the measurement path. The prover MUST report each as
#: read inside the instrument seam's closure.
MEASUREMENT_FIELDS = ("sims", "gumbel_m", "step_cap", "root_noise", "problems_per_pass")


def test_the_inertness_prover_can_still_say_no() -> None:
    """**The known positives**, required before any inertness verdict is trusted.

    The prover decides whether a configuration change can have touched
    measurement — and therefore whether a three-hour equivalence gate must
    re-run. Its first verdict was `NOT inert`, produced by an **always-firing
    bound**: `git_sha` calls `subprocess.run`, the resolver bound that to
    `campaign.run`, and the whole training loop entered the measurement closure.
    Any `subprocess.run` anywhere would have produced that verdict for any field
    — a verdict independent of its input, which carries no information.

    The fix (module-qualified calls no longer bind to local names) was made in the
    direction that **saves three hours**, and the corrected tool is then used to
    justify not spending them. So it must be shown to still say *no*: a tool that
    can only say yes is the same broken instrument as one that could only say no.
    """
    from scripts.prove_measurement_inert import measurement_closure

    closure, reads = measurement_closure()
    for field in MEASUREMENT_FIELDS:
        inside = reads.get(field, set()) & closure
        assert inside, (
            f"{field!r} is on the measurement path and the prover cannot see it — "
            "the resolver has overshot from over-connection into under-connection, "
            "and every inertness verdict it has produced is void"
        )


def test_the_inertness_prover_still_says_yes_where_it_should() -> None:
    """The other polarity. `rehearsal_frac` is read only by `train_on_ring`, which
    the instrument seam never reaches — that is what licensed M1-A4's fingerprint
    move without re-running the gate."""
    from scripts.prove_measurement_inert import measurement_closure

    closure, reads = measurement_closure()
    assert not (reads.get("rehearsal_frac", set()) & closure), (
        "rehearsal_frac now reads inside the measurement closure; M1-A4's "
        "measurement-inertness claim no longer holds and the gate must re-run"
    )


def test_the_policy_target_is_never_read_on_the_measurement_path() -> None:
    """F-42's measurement-inertness: ``train.policy_target`` is read only by
    ``ring_batch`` and ``train_on_ring``, which the instrument seam never reaches.
    So the fingerprint move does not touch what the cadence measures."""
    from scripts.prove_measurement_inert import measurement_closure

    closure, reads = measurement_closure()
    assert not (reads.get("policy_target", set()) & closure), (
        "policy_target is now read inside the measurement closure; F-42's "
        "inertness claim no longer holds"
    )
