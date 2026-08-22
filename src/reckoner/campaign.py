"""The M1 campaign driver. **The** loop — `golden` is this at golden config.

Chunk 11's unbuilt half. `PREREG-m1.md` governs what it measures and refuses;
`BRIEF-chunk11-driver.md` composes it; D-A1 names the three parts that are built
rather than promoted; D-A2 fixes the rehearsal count and splits the fingerprint
assertion across its two consumers.

**One composition.** `golden.py` had the whole loop in test clothing — episodes,
ring, training, the switch row, the four-step commit — and the worst available
outcome was a fresh driver written beside it: two loops wearing one name, where
`golden` certifies a composition the campaign never runs. So the driver *is* the
loop and `golden` invokes it at golden config.

**Three parts are new, not promoted** (D-A1 §1). `golden` ran `uniform_stub`
every iteration, so the model→search→ring path — the loop's entire subject — had
never executed. Pool par had never been drawn. The criterion had never seen a
real input. Each now carries a central-job assertion in both polarities, because
an assertion that the central job *happened* standing in for one that its
**product is consumed** is the defect that hid all three.

**No behavioural flags.** Extent, treatment size and thread pins live in
`CampaignConfig`, under the fingerprint. A flag is unfingerprinted config — the
caller-choice hazard standing at the command line — and a campaign whose cadence
can be set at invocation is one that can silently diverge from its prereg.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import (
    anchored_data,
    git_sha,
    read_suite,
    sha256_file,
    suite_problem,
    training_problems,
)
from reckoner.episode import Problem
from reckoner.evaluate import ModelArm, assert_eval_mode, model_evaluator
from reckoner.gates import no_regress_floor
from reckoner.ladder import problem_key_of
from reckoner.ladderpass import PassPaths, is_complete, read_pair_scores, run_pass
from reckoner.logschema import (
    ITERATION_FIELDS,
    SCHEMA_ERA,
    VALUE_SWITCH_FIELDS,
    alarm_census,
    append_row,
    read_rows,
    switch_event_row,
)
from reckoner.model import load_checkpoint, save_checkpoint
from reckoner.pool import CheckpointPool, PoolError
from reckoner.replay import ReplayRing
from reckoner.resume import RunState, commit_iteration, latest_committed, resume
from reckoner.rules import RULESET_VERSION
from reckoner.runner import iteration_row, run_iteration
from reckoner.train import train_on_ring
from reckoner.valuegate import (
    ValueHeadState,
    consider_switch,
    evaluate_head,
    value_contribution,
)
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "runs" / "suites"

# --- the two fingerprints, recorded in PREREG-m1 §M1-A2 §1, MOVED by M1-A4 ----
# M1-A4 moved the campaign config ONCE, carrying every decision on that page:
# rehearsal_frac 0.0 -> 0.65, the two `ladder.sympy_*` deletions, and three role
# changes at unchanged value (interop EXERCISED, problems_per_pass as the
# population cap, snapshot_every honoured). The batching rule exists so a
# fingerprint moves once with a page explaining all of it.
#
#   was ce41af96ee85f0a2… / 314fbeb99b6640f6…  (through the rehearsal and
#                                               the equivalence gate)
# The driver runs TWO profiles and therefore makes TWO assertions, each pinned
# where its profile acts (D-A2 §2). A single startup check would compare one
# profile's value against a run that also uses the other.
CAMPAIGN_FINGERPRINT = "ccfec081f4eb16040c5587cc738e0f1c4ccbce0fcaad0e56eb3c55aa08ebb5f6"
EVAL_FINGERPRINT = "fd53b63f7502a4b106461ab1ce1f1fbab982e5d3df2ff5a07bc941c5760912de"

#: The values M1 ACTUALLY RAN UNDER, kept because they are quoted on pages and a
#: pin that moves must not silently rewrite what a completed run was measured at.
#: The pin above is what THIS code computes; the era below is what M1 recorded.
M1_ERA_FINGERPRINTS = {
    "campaign": "8443847bb8c41218dc93a13b4401929ef2151bec3e7947041cfed99ea9903c6d",
    "eval": "c8aa1fccd1c97dd84aa5d636356a7a54d08302146324bd9e43319f4dd2599e10",
    "_moved_by": "campaign.device, fingerprinted 2026-08-22 after F-51",
}

#: PREREG-m1 §4: both indistinguishability floors, on the 1,200-problem instrument.
NO_REGRESS = {48: 1188, 1: 1167}

#: PREREG-m1 §2: the primary's population.
SUCCESSOR_STRATA = (7, 8, 10)

#: PREREG-m1 §5: the 24 problems certified as the shared miss set of the anchor at
#: sims=1 and the scripted solver (intersection 24/24, Jaccard 1.0). FROZEN — the
#: watchlist observes against it and never re-derives it.
WATCHLIST = REPO / "runs" / "chunk11_misses_diagnostic.json"


def _enrols_at(iteration: int, cfg: Config) -> bool:
    """Whether iteration *n* takes a pool snapshot.

    One predicate, used by BOTH the enrolment site and resume's pool rebuild —
    because a rebuild that replayed a different cadence than the run would
    reconstruct a pool the run never had, which is F-23 wearing F-36's clothes.
    """
    return (iteration + 1) % cfg.league.snapshot_every == 0


def read_watchlist(instrument: dict) -> dict[str, dict]:
    """**The** reader for a pass record's watchlist: ``{budget: cell}``.

    Mono-instanced after the shape moved to a pair per leg and three consumers
    had to move with it — `campaign` wrote it, `golden` read it, and
    `equivalence_gate` did not, so the gate completed both measurements and then
    crashed in its reporting three hours in. One accessor means a future change
    breaks at the reader rather than at report time, which is the same move that
    mono-instancing `read_anchors` made — and that exposed a latent defect too.

    The partition is re-checked **here**, not only where the record is written: a
    reader that trusts its input is a reader that will one day report a silently
    emptied intersection as good news.
    """
    by_budget = instrument["watchlist"]["by_budget"]
    for budget, cell in by_budget.items():
        total = cell["family_remaining"] + cell["novel_misses"]
        if total != cell["pass_misses"]:
            raise CampaignRefusal(
                f"watchlist partition broken at sims={budget} on read: "
                f"{cell['family_remaining']} + {cell['novel_misses']} = {total}, but "
                f"the pass recorded {cell['pass_misses']} misses."
            )
    return by_budget


def watchlist_reference(instrument: dict) -> dict[str, list[int]]:
    """The anchor's own readings, carried on the record so a value has a SCALE."""
    return instrument["watchlist"]["anchor_reference"]


def frozen_watchlist() -> set[str]:
    """The frozen 24, as `problem_key` strings."""
    record = json.loads(WATCHLIST.read_text())
    return set(record["cross_reference_sims_1_vs_scripted"]["shared"])


#: The Phase-1 anchor: rung zero of the pool, and iteration 0's evaluator.
ANCHOR = REPO / "runs" / "phase1" / "phase1.pt"

#: PREREG-m1 §2.1: the anchor's measured baseline on {7, 8, 10}, from Part-0d —
#: 43 + 26 + 32 beats of 600. The primary is CI-separated improvement over this.
ANCHOR_BEAT = 101

#: The OMP family was UNSET throughout the licence. Unset is a value (M1-A2 §4).
OMP_FAMILY = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class CampaignRefusal(RuntimeError):
    """The campaign will not start, or will not continue, and says why."""


def golden_config(**overrides) -> Config:
    """Golden scale: small enough to be a liveness check, **real otherwise**.

    Lives in the package rather than in tests because it is a config, and
    `golden` is a production script. Scale is declared here rather than passed as
    flags — the no-behavioural-flags rule is about where a behaviour is declared,
    not about who is running.

    This is deliberately NOT the campaign's config: the campaign's door refuses
    it, which is that check working rather than failing.
    """
    cfg = Config()
    validate(cfg)
    cfg = replace(
        cfg,
        campaign=replace(cfg.campaign, iterations=2, episodes_per_iteration=12),
        train=replace(cfg.train, train_steps_per_iter=2),
        search=replace(cfg.search, sims=4, gumbel_m=4),
        # No cadence by DEFAULT, and a cadence available on request. The runs
        # whose assertions need one — `golden`'s path coverage and the resume
        # gate's three-log comparison — pass `cadence()` below; everything else
        # skips a measurement it never inspects. A suite people skip is a suite
        # that does not run, and that has beaten discipline more often here than
        # any missing check has.
        ladder=replace(cfg.ladder, ladder_every=99),
    )
    for group, changes in overrides.items():
        cfg = replace(cfg, **{group: replace(getattr(cfg, group), **changes)})
    return cfg


#: The cadence overrides for callers that assert on the cadence path. Named so
#: the two call sites cannot drift apart, and so a third one is a decision.
CADENCE = {"ladder": {"ladder_every": 2, "problems_per_pass": 2}}


# --------------------------------------------------------------------- profiles


def eval_profile(cfg: Config) -> Config:
    """The measurement profile: root noise off, everything else identical."""
    return replace(cfg, search=replace(cfg.search, root_noise=False))


def assert_campaign_profile(cfg: Config) -> str:
    """Once, at startup. The frozen page becomes executable authority."""
    got = config_fingerprint(cfg)
    if got != CAMPAIGN_FINGERPRINT:
        raise CampaignRefusal(
            f"campaign config fingerprint {got} != PREREG-m1's recorded "
            f"{CAMPAIGN_FINGERPRINT}. This config is not the one the campaign was "
            "registered against, so the run would not be the registered run."
        )
    return got


def assert_eval_profile(cfg: Config) -> str:
    """At **every** instrument pass, not once at startup (D-A2 §2).

    Two checks, and the split matters. The cadence measurements are comparable to
    their baselines only if they provably ran the eval profile, so **the profile
    must ACT** — that is verified on every pass, whoever is calling.

    The *registered* pin is additionally asserted when the pass is a campaign
    measurement. Golden runs this same seam as a liveness check (its numbers are
    never compared to a baseline), and pinning it to the campaign's fingerprint
    would make the shared composition unrunnable at golden config — which is
    precisely how the cadence path came to have no test at all, and how two
    campaign-blocking defects reached the pod.

    The ruling in D-A2 §2 is preserved rather than weakened: every campaign pass
    still asserts the registered value, at the boundary where the profile acts.

    **The predicate is DERIVED, never passed.** This takes the config the pass
    runs under and computes both the profile and "is this a campaign
    measurement?" itself, because *is this a real measurement* is exactly the
    question F-19 proved a caller should not be trusted to answer. A `campaign=`
    argument would be a caller-supplied claim about evidential status, which is
    the currency-tag defect wearing a keyword.
    """
    ev = eval_profile(cfg)
    campaign = config_fingerprint(cfg) == CAMPAIGN_FINGERPRINT
    if ev.search.root_noise:
        raise CampaignRefusal(
            "`eval_profile` returned a config with root noise ON, so an instrument "
            "pass would measure the self-play profile. The measurement is not "
            "comparable to any baseline, whoever is calling."
        )
    got = config_fingerprint(ev)
    if campaign and got != EVAL_FINGERPRINT:
        raise CampaignRefusal(
            f"eval profile fingerprint {got} != the pinned {EVAL_FINGERPRINT}; "
            f"this pass is not comparable to its baseline. (M1 ran under "
            f"{M1_ERA_FINGERPRINTS['eval']}; the pin moved when "
            f"{M1_ERA_FINGERPRINTS['_moved_by']}.)"
        )
    return got


def assert_device(cfg: Config) -> dict:
    """The declared device must be the device that runs, and it refuses.

    **The property that forbids the class**, where the class is *"ran somewhere
    nobody declared."* So the refusal is `declared != usable`, not
    `declared != best available`:

    * **declared unavailable → REFUSE.** A page saying `cuda` while the runtime
      offers only CPU produces numbers from a machine the page does not describe.
    * **GPU present, `cpu` declared → RECORDED, not refused.** That is M1: 9.6
      hours on a `+cpu` wheel with a card attached. It is a *cost* finding, and
      making it a refusal would forbid the one configuration M1 actually ran —
      a gate that fails its own history is a gate nobody keeps.

    The governance teeth are in the **fingerprint**, not here: `campaign.device`
    is fingerprinted, so moving to GPU mid-campaign moves the fingerprint and
    lands as an amendment. This assertion only guarantees the declaration is
    true, which is what makes the fingerprint mean anything.
    """
    declared = cfg.campaign.device
    build = "cuda" if torch.version.cuda else ("rocm" if torch.version.hip else "cpu")
    accelerator = torch.cuda.is_available()

    if declared != "cpu" and not accelerator:
        raise CampaignRefusal(
            f"campaign.device declares {declared!r} and the runtime offers no "
            f"accelerator (torch {torch.__version__}, build {build!r}). Execution "
            "environment is fingerprinted: a run on an undeclared device rescales "
            "every wall-clock number in the record and changes numerics under MCTS."
        )
    return {
        "declared": declared,
        "torch_version": torch.__version__,
        "torch_build": build,
        "accelerator_present": accelerator,
        # M1's cost, made visible wherever this record lands rather than
        # reconstructed from a wheel name two days later.
        "gpu_present_and_unused": bool(accelerator and declared == "cpu"),
    }


def assert_threads(cfg: Config) -> dict:
    """The pins, applied and verified — including the recorded absence.

    Both thread pins are now APPLIED and then read back, so the returned record
    is what the runtime holds rather than what the caller hoped for.
    """
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    present = {v: os.environ[v] for v in OMP_FAMILY if v in os.environ}
    if present:
        raise CampaignRefusal(
            f"the OMP family was UNSET throughout the licence and is set here: "
            f"{present}. Unset is a value; setting it is a gated configuration "
            "change whose reproduction gate re-runs, not a tidy-up."
        )
    # INTEROP IS NOW EXERCISED, NOT OBSERVED (M1-A4 §6, ruled 2026-08-19).
    #
    # M1-A2 §4 classed it OBSERVED — "licenses only 'this is what ran'" — because
    # it was constant at 32 across both hosts and undiscriminated. The campaign
    # host then changed and the new one defaults to 48, so the assertion refused,
    # correctly: a host that differs stands outside that evidence, not inside it.
    #
    # §4 names the disposition itself: "raising any of these is not a config
    # tweak. It is a new configuration whose reproduction gate re-runs." So the
    # pin is APPLIED here rather than merely checked, and the gate re-ran. The
    # FIELD'S VALUE IS UNCHANGED at 32, so the config fingerprint does not move —
    # what changed is whether the runtime is made to match the record or merely
    # asked whether it happens to.
    #
    # On the old 64-core host this is inert: ambient interop was already 32.
    wanted = cfg.campaign.interop_threads
    if torch.get_num_interop_threads() != wanted:
        try:
            torch.set_num_interop_threads(wanted)
        except RuntimeError as exc:  # set after the pool started — cannot be fixed here
            raise CampaignRefusal(
                f"interop threads are {torch.get_num_interop_threads()} and cannot be "
                f"set to the licensed {wanted}: {exc}. torch fixes the interop pool at "
                "first parallel work, so this must be set before anything runs — a "
                "process that has already measured under the wrong value cannot be "
                "corrected into the record."
            ) from exc
    observed_interop = torch.get_num_interop_threads()
    if observed_interop != wanted:
        raise CampaignRefusal(
            f"interop threads are {observed_interop} after being set to {wanted}. "
            "The pin did not take, and a pin that does not take is worse than none: "
            "it reports a value the runtime does not have."
        )
    return {
        "intra_op": torch.get_num_threads(),
        "interop": observed_interop,
        "omp_family": "unset",
    }


# ------------------------------------------------------------ the instrument seam


def _assert_rng_unmoved(entry: tuple) -> None:
    """The instrument passes must leave every generator where they found it.

    Measurement that advances a generator the loop later draws from couples the
    two silently: self-play's episodes would depend on whether a cadence ran, and
    the campaign's trajectory would differ between a run that measured and one
    that did not. Nothing in the artifacts would say so.

    This is the assertion that keeps "routing touches measurement only" true —
    the RNG is the one channel through which a read-only pass can reach the loop.
    """
    py_before, torch_before = entry
    if random.getstate() != py_before:
        raise CampaignRefusal(
            "an instrument pass advanced the global `random` generator. Measurement "
            "that moves a generator self-play draws from makes the trajectory depend "
            "on whether a cadence ran, which is a coupling no row would record."
        )
    if not torch.equal(torch.get_rng_state(), torch_before):
        raise CampaignRefusal(
            "an instrument pass advanced torch's global generator — same coupling, same silence."
        )


def run_instruments(
    model, cfg: Config, *, iteration: int, anchor_beat: int, run_dir: Path | None = None
) -> dict:
    """**The** instrument seam. Every cadence measurement goes through here.

    Mono-instance because the discipline requires it, and because Lever B —
    unit-parallel instrument passes — lands behind this signature later without
    the driver noticing. The seam still *delegates*; it is never bypassed.

    Asserts the eval fingerprint on entry: this is the boundary where the eval
    profile acts.

    **MEASUREMENT NEVER TOUCHES LOOP STATE** (F-33). The ring, the checkpoints and
    the pool are untouched here, so the campaign's trajectory is identical whether
    or not this runs — which is what makes routing the primary through the ladder
    a restoration rather than a treatment change. The one way that could be false
    is the RNG: a measurement that advanced a generator the loop later draws from
    would couple the two silently. So the generators are captured on entry and
    **verified unmoved on exit**, rather than assumed.
    """
    ev = eval_profile(cfg)
    fingerprint = assert_eval_profile(cfg)
    # F-22 at the seam every cadence measurement passes through. Named rather
    # than a bare `model_evaluator(...)` call kept for its side effect, because a
    # guard that reads as dead code is eventually deleted as dead code.
    assert_eval_mode(model, ev)
    out: dict = {"iteration": iteration, "config_fingerprint": fingerprint, "profile": "eval"}

    rng_entry = (random.getstate(), torch.get_rng_state().clone())

    # --- no-regress, both budgets, against the declared floors ---------------
    # THROUGH THE LADDER (F-35). This leg emitted {at_par, of} and nothing else,
    # so the watchlist PREREG §5 specifies — family_remaining and novel_misses,
    # "computed from data the pass already produces" — had no input: an aggregate
    # has no miss set, and there is no intersection to take. The rehearsal proved
    # the cost on the exact event §5 was written for: at-par 1193 -> 910, and no
    # artifact could say whether those misses were the frozen family or a new one.
    #
    # Recording is required independent of execution mode. Routing rather than
    # teaching `run_iteration` to emit rows keeps ONE implementation of
    # per-problem recording — F-33's lesson applied to F-33's own remedy.
    # THE POPULATION CAP, per file. `ladder.problems_per_pass` was one of the
    # census's dead keys; it is now the knob that lets `golden` exercise a cadence
    # iteration end to end. **Provably inert at campaign config**: every suite and
    # stratum file holds exactly 200 rows and the cap is 200, so the slice is the
    # identity — verified in test_campaign_driver, not assumed.
    cap = cfg.ladder.problems_per_pass
    suites = sorted(SUITES.glob("solve_in_*.jsonl"))
    instrument = [suite_problem(r) for p in suites for r in read_suite(p)[:cap]]
    budgets = sorted(NO_REGRESS, reverse=True)
    arms = [
        ModelArm(model, name=f"model@{sims}", sims=sims, m=min(ev.search.gumbel_m, sims))
        for sims in budgets
    ]
    # A PASS PER LEG, under distinct roots. The two legs share an iteration
    # index, and `run_pass` keys completion on (root, index) — so a shared root
    # meant the second leg found the first leg's DONE marker, skipped its own
    # pass, and read zero rows for an arm name that pass never played. Caught by
    # the equivalence gate rather than by a test, because `golden_config` sets
    # `ladder_every = 99` and no test exercises a cadence iteration (F-29's
    # registered gap, biting where it was registered).
    root = (run_dir or REPO / "runs") / "ladder" / "no_regress"

    # BOTH ARMS ARE SUBJECTS, and that is a refusal rather than a default. Each
    # floor is a ONE-SIDED comparison against a frozen constant — 1188 and 1167 —
    # not a paired comparison between budgets. Declaring one a baseline would
    # invite a bootstrap CI between 48 and 1 sims: a number nobody asked for,
    # wearing the ladder's authority.
    if not is_complete(root, iteration):
        run_pass(
            root,
            iteration,
            arms,
            instrument,
            ev,
            roles={a.name: "subject" for a in arms},
            calibration_note=(
                "exact-par suites; each budget is scored against its own frozen "
                "indistinguishability floor and NOT against the other budget"
            ),
            seed=0,
        )
    scored = read_pair_scores(root, iteration)

    misses: dict[int, set[str]] = {}
    for sims, arm in zip(budgets, arms, strict=True):
        rows = [r for r in scored if r["arm"] == arm.name]
        at_par = sum(int(r["z"]) == 0 for r in rows)
        misses[sims] = {r["problem_key"] for r in rows if int(r["z"]) != 0}
        floor = NO_REGRESS[sims]
        out[f"no_regress_sims_{sims}"] = {
            "at_par": at_par,
            "of": len(rows),
            "floor": floor,
            "held": at_par >= floor,
            "floor_kind": "indistinguishability",
            "licensed_sentence": "not below the anchor's own one-sided 95% band",
            "floor_recomputed": no_regress_floor(1193 if sims == 48 else 1176, 1200),
        }

    # THE WATCHLIST — A PAIR PER LEG, on the pass record (PREREG §5).
    #
    # Per pass, not at one budget: each leg has its own miss set, and the 48-sim
    # reading is the one commensurate with the primary's budget. Computing it only
    # at min(budgets) is what let a prediction registered at 48 sims be scored
    # against a 1-sim number.
    #
    # It rides on the PASS RECORD rather than as budget-named iteration columns,
    # because a budget encoded in a column name puts the budget in two homes — the
    # config and the schema — where they can drift. "Computed per pass" already
    # names the pass as the carrier, and the partition identity then holds where
    # its inputs live.
    #
    # Against the FROZEN reference, never re-derived: a family re-derived per pass
    # conflates "the family shrank" with "the family's definition moved".
    frozen = frozen_watchlist()
    watchlist: dict[str, dict] = {}
    for sims in budgets:
        missed = misses[sims]
        family, novel = len(missed & frozen), len(missed - frozen)
        if family + novel != len(missed):
            raise CampaignRefusal(
                f"watchlist partition broken at sims={sims}: {family} + {novel} != "
                f"{len(missed)}. Most likely the problem_key format drifted and the "
                "intersection emptied silently — the failure this identity exists to "
                "catch, and the field that has caused two incidents already."
            )
        watchlist[str(sims)] = {
            "family_remaining": family,
            "novel_misses": novel,
            "pass_misses": len(missed),
        }
    out["watchlist"] = {
        "by_budget": watchlist,
        "frozen_family_size": len(frozen),
        # The anchor's own readings, carried so the column has a SCALE. Without
        # them a campaign value of 18 means nothing: the anchor scores (24, 0) at
        # 1 sim — the family was defined from that miss set — and (7, 0) at 48,
        # its misses being a strict subset. Search rescues 17 of the 24 between
        # the budgets, which is what a campaign reading is really compared against.
        "anchor_reference": {"1": [24, 0], "48": [7, 0]},
    }

    # --- the primary: pooled beat-par on {7, 8, 10} --------------------------
    # THROUGH THE LADDER, not beside it (F-33). This used `run_iteration` and kept
    # `{beat, of}`, which cannot be paired — and PREREG §2 makes the test of record
    # a paired-difference bootstrap against Part-0d's per-problem outcomes. The
    # ladder already appends one row per (arm, problem) as it happens, precisely
    # because "a run that stored only means has thrown away the input to its own
    # test of record". Routing here rather than adding per-problem output to
    # `run_instruments` keeps one implementation instead of minting a third.
    arm = ModelArm(model, sims=48, m=16)
    problems, stratum_of = [], {}
    for k in SUCCESSOR_STRATA:
        for row in read_suite(SUITES / f"scripted_in_{k}.jsonl")[:cap]:
            problem = suite_problem(row)
            problems.append(problem)
            stratum_of[problem_key_of(problem)] = f"scripted_in_{k}"

    root = (run_dir or REPO / "runs") / "ladder" / "primary"
    if not is_complete(root, iteration):
        run_pass(
            root,
            iteration,
            [arm],
            problems,
            ev,
            roles={arm.name: "subject"},
            calibration_note=(
                "scripted par is a PROVISIONAL floor, not an exact minimum, so z = +1 "
                "is legal on these strata — which is what the instrument was minted "
                "to allow and what the exact-par suites cannot offer."
            ),
            seed=0,
        )
    # A completed pass is re-read rather than re-run: the measurement is
    # deterministic, and a resumed iteration should not pay for it twice.
    scores = [r for r in read_pair_scores(root, iteration) if r["arm"] == arm.name]

    per_stratum: dict[str, dict] = {
        f"scripted_in_{k}": {"beat": 0, "of": 0} for k in SUCCESSOR_STRATA
    }
    for row in scores:
        cell = per_stratum[stratum_of[row["problem_key"]]]
        cell["of"] += 1
        cell["beat"] += int(row["z"]) == 1
    pooled_beat = sum(c["beat"] for c in per_stratum.values())
    pooled_n = sum(c["of"] for c in per_stratum.values())
    out["pair_scores"] = str(PassPaths(root, iteration).scores)
    out["no_regress_pair_scores"] = str(
        PassPaths((run_dir or REPO / "runs") / "ladder" / "no_regress", iteration).scores
    )

    _assert_rng_unmoved(rng_entry)
    out["primary"] = {
        "pooled_beat_par": f"{pooled_beat}/{pooled_n}",
        "pooled_rate": round(pooled_beat / pooled_n, 6),
        "anchor_baseline": f"{anchor_beat}/{pooled_n}",
        "delta_vs_anchor": round((pooled_beat - anchor_beat) / pooled_n, 6),
        "per_stratum": per_stratum,
    }
    return out


# ---------------------------------------------------------------- the pre-flight


def preflight(cfg: Config, model, scratch: Path) -> None:
    """Every row class, at micro scale, before iteration 0 spends anything.

    D-A1 §3. Not a flag: a script that smokes its output path only when asked is
    a script that will be run without asking.
    """
    assert_device(cfg)
    started = time.perf_counter()
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    ring = ReplayRing(256, cfg)
    problems = training_problems(anchored_data("train_100k"), 2, seed=0)
    stats = run_iteration(
        problems, model_evaluator(model, cfg, 0.0), cfg, ring, sims=1, m=1, seed=0
    )
    pool = CheckpointPool(cfg)
    row = iteration_row(
        stats,
        iteration=0,
        run_name="preflight",
        git_sha=git_sha(REPO),
        config_fingerprint=config_fingerprint(cfg),
        cfg=cfg,
        ruleset_version=RULESET_VERSION,
        vocab_version=VOCAB_VERSION,
        schema_era=SCHEMA_ERA,
        evaluator_checkpoint_sha256="0" * 64,
        pool_composition=pool.composition(),
        # Declared, not inherited (F-29). The pre-flight's pool is empty by
        # construction and it runs no cadence, so both columns are genuinely
        # absent — and it says so itself rather than taking the library's word.
        absent={
            "pool_par_fraction": "the pre-flight's pool is empty by construction",
            "ladder_pass": "the pre-flight runs no ladder pass",
        },
    )
    append_row(scratch / "iterations.jsonl", row, ITERATION_FIELDS)

    # THE SWITCH ROW IS A ROW CLASS, so "every row class" has to mean it (F-24).
    # This previously built a `RunState` and `del`'d it, which exercised a
    # constructor and called it coverage of an output path.
    #
    # Through the REAL criterion on the REAL micro ring, not a synthesised event:
    # a pre-flight that hand-built the row it then wrote would prove the writer
    # works and nothing about the path that feeds it. An abstention here is a
    # pass, not a failure — the switch row's whole registration is that
    # abstentions write too.
    slots = sorted(ring.holdout(cfg.train.ring_holdout_frac, seed=0))
    labels, predictions = evaluate_head(model, ring, cfg, slots)
    _, event = consider_switch(ValueHeadState(), labels, predictions, iteration=0)
    append_row(
        scratch / "value_switch.jsonl",
        switch_event_row(event, schema_era=SCHEMA_ERA),
        VALUE_SWITCH_FIELDS,
    )

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"  pre-flight OK ({time.perf_counter() - started:.1f}s)")


# ------------------------------------------------------------------ episodes


def campaign_problems(
    cfg: Config, pool: CheckpointPool, model, k: int, seed: int
) -> tuple[list[Problem], int]:
    """Episode sources, with **pool par actually drawn** (D-A1 §1.2).

    `golden` enrolled snapshots and asserted the pool grows, and never once drew
    a par from it — so `par_from_pool_frac` had no exercise in any composition.
    Here it does: `league.par_from_pool_frac` of episodes take their par from a
    snapshot, which is the mechanism par escalation *is*.

    `training_problems` refuses a frozen instrument first thing, so the runtime
    boundary the contamination censuses could not reach is held here too.
    """
    problems = training_problems(anchored_data("train_100k"), k, seed=seed)
    rng = random.Random(seed * 7919 + 13)

    def factory(snapshot_model, value_scale):
        return model_evaluator(snapshot_model, cfg, value_scale)

    out, from_pool = [], 0
    for problem in problems:
        if rng.random() >= cfg.league.par_from_pool_frac:
            out.append(problem)
            continue
        drawn = pool.par_for_episode(problem, factory, rng)
        out.append(replace(problem, par=drawn.par, par_source=drawn.par_source))
        from_pool += drawn.par_source == "pool"
    return out, from_pool


# ----------------------------------------------------------------------- run


def run_campaign(run_dir: Path, cfg: Config, *, anchor: Path | None = None) -> dict:
    """The campaign's door. **This** is where the registered config is enforced.

    The assertion lives here rather than inside :func:`run` because `run` is the
    shared loop and `golden` invokes it at golden config *deliberately* — one
    composition, two callers. Putting the check inside the loop would force
    `golden` to either fail or route around it, and a routed-around check is the
    two-loops outcome arriving by the back door.

    So: the loop is shared, and the campaign's *entrypoint* is what refuses a
    config the frozen page did not bless.
    """
    assert_campaign_profile(cfg)
    return run(run_dir, cfg, run_name="m1", anchor=anchor)


def run(
    run_dir: Path,
    cfg: Config,
    *,
    run_name: str = "m1",
    anchor: Path | None = None,
    on_commit=None,
) -> dict:
    """**The** loop. `golden` is this at golden config; the campaign is
    :func:`run_campaign`, which asserts the registered fingerprint first.

    ``on_commit(iteration, phase)`` is called at ``"before_row"`` and
    ``"before_latest"`` — the two kill points the resume gate uses, and the only
    reason it exists. Same shape and same justification as
    :func:`ladderpass.run_pass`'s ``on_unit``: a real SIGKILL needs a real
    process, and a process cannot be killed at a boundary it does not announce.

    The iteration's artifacts commit through :func:`resume.commit_iteration` in
    the four-step order that module specifies — ring, state, row, then ``LATEST``
    by atomic rename. Only the rename commits; steps 1–3 are provisional, and a
    process killed anywhere in them leaves artifacts resume discards.
    """
    threads = assert_threads(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    anchor = anchor or (REPO / "runs" / "phase1" / "phase1.pt")
    start, ring, state = resume(run_dir, cfg)
    if ring is None:
        ring = ReplayRing(cfg.train.replay_capacity, cfg)
        state = RunState(iteration=0, value_head=ValueHeadState(), seed=cfg.seed)
    model, _ = load_checkpoint(anchor, cfg)
    model.eval()  # F-22: dropout must not be live inside a search

    # The evaluator's provenance: at iteration 0 the weights are the anchor's; on
    # resume they are the last committed checkpoint's. Either way the digest is
    # of the FILE the search evaluator was built from (D-A1 §1.1).
    evaluator_source = anchor
    if start > 0:
        resumed = run_dir / f"ckpt-{start - 1}.pt"
        if resumed.exists():
            model, _ = load_checkpoint(resumed, cfg)
            model.eval()  # F-22
            evaluator_source = resumed

    # SEED THE POOL WITH THE ANCHOR. `league.seed_pool_with_anchor` was honoured
    # in exactly one script and by no driver, so `par_from_pool_frac` would have
    # been dead through iteration 0 and thin after — the enrolment pattern one
    # level down: the class supports seeding, the config declares it, the
    # consumer never called it. The anchor is rung zero; without it there is
    # nothing to escalate FROM, and M1-A2 §6's two-population split would have
    # been 100/0 rather than 80/20 for the campaign's opening iterations.
    pool = CheckpointPool(cfg)
    if cfg.league.seed_pool_with_anchor and anchor.exists():
        pool.add(anchor)

    # AND REBUILD IT (F-23). The pool lives in memory; the snapshots live on
    # disk. A resumed run that only re-seeded the anchor would restart par
    # escalation from rung zero while the ring, the weights and the row history
    # all continued — the campaign would go on measuring against a par that
    # silently fell back to where it began, and nothing in the row would say so.
    #
    # Order is reconstructed, not just membership: `sample` is `rng.choice` over
    # `members`, so the same set in a different order draws a different snapshot.
    # Replaying the original add sequence — anchor, then snap-0 upward — makes
    # the list identical rather than equivalent.
    #
    # A missing snapshot is a REFUSAL. `enroll` precedes the row and LATEST, so
    # every committed iteration has one; if it is gone, this run's history is not
    # reconstructible, and continuing with a thinner pool would be the defect
    # arriving quietly by the path meant to repair it.
    for prior in range(start):
        if not _enrols_at(prior, cfg):
            continue  # that iteration took no snapshot, so there is none to replay
        snapshot = run_dir / f"snap-{prior}.pt"
        if not snapshot.exists():
            raise PoolError(
                f"resuming at iteration {start} needs {snapshot.name} to rebuild the pool, "
                "and it is absent. Iteration "
                f"{prior} committed, so it was written and has since been lost: the pool "
                "cannot be restored to what it was, and a resumed run with a thinner pool "
                "is not the run that was interrupted."
            )
        pool.add(snapshot)

    # THE PRE-FLIGHT, ACTUALLY CALLED (F-24). D-A1 §3 registered it, it was
    # built, and no caller ever ran it — so it had been broken by an unrelated
    # signature change and nothing noticed, which is the precise failure a
    # pre-flight exists to catch happening to the pre-flight itself.
    #
    # Only when starting fresh: on resume the output path has already written
    # rows this run, and re-proving it would spend the anchor's time on a
    # question the artifacts already answer.
    if start == 0:
        preflight(cfg, model, run_dir / "_preflight")

    summary: dict = {"run_name": run_name, "threads": threads, "iterations": []}

    for n in range(start, cfg.campaign.iterations):
        began = time.perf_counter()
        digest = sha256_file(evaluator_source)

        # --- self-play, MODEL-IN-SEARCH ------------------------------------
        scale = value_contribution(state.value_head)
        # M1-A3: the pool AS THIS ITERATION DREW FROM IT, captured before this
        # iteration's own enrolment. Reading it after `enroll` would log a pool
        # one rung deeper than the one that actually supplied par, which is an
        # off-by-one nobody could detect from the artifacts.
        composition = pool.composition()
        problems, from_pool = campaign_problems(
            cfg, pool, model, cfg.campaign.episodes_per_iteration, seed=n
        )
        t0 = time.perf_counter()
        stats = run_iteration(problems, model_evaluator(model, cfg, scale), cfg, ring, seed=n)
        seconds_self_play = time.perf_counter() - t0
        stats.check_descent_identity()

        # --- training -------------------------------------------------------
        t0 = time.perf_counter()
        train_stats = train_on_ring(
            model,
            ring,
            cfg,
            steps=cfg.train.train_steps_per_iter,
            seed=n,
            value_head=state.value_head,
        )
        seconds_train = time.perf_counter() - t0
        model.eval()  # F-22: belt and braces — train_on_ring restores, and the
        # evaluator's own guard would refuse if either of us were wrong.

        # --- checkpoint, then enrol: par escalates with the model ------------
        checkpoint = run_dir / f"ckpt-{n}.pt"
        save_checkpoint(checkpoint, model, cfg, n, value_head=state.value_head.as_dict())
        # F-36: the CADENCE IS HONOURED, not assumed. This enrolled every
        # iteration while `shakedown.py` honoured `league.snapshot_every`, and the
        # two agreed only because the default is 1 — a fingerprinted field
        # steering one composition and not the other. At 5 the shakedown would
        # enrol four fewer snapshots per five iterations than the campaign,
        # changing pool growth and therefore par escalation, which is the
        # mechanism the primary is denominated against.
        if _enrols_at(n, cfg):
            pool.enroll(model, n, state.value_head, run_dir / f"snap-{n}.pt")

        # --- the criterion, on REAL accrued holdout (D-A1 §1.3) -------------
        slots = sorted(ring.holdout(cfg.train.ring_holdout_frac, seed=0))
        labels, predictions = evaluate_head(model, ring, cfg, slots)
        head, event = consider_switch(state.value_head, labels, predictions, iteration=n)
        append_row(
            run_dir / "value_switch.jsonl",
            switch_event_row(event, schema_era=SCHEMA_ERA),
            VALUE_SWITCH_FIELDS,
        )

        # --- the cadence unit ------------------------------------------------
        absent: dict[str, str] = {}
        ladder_index = None
        if (n + 1) % cfg.ladder.ladder_every == 0:
            ladder_index = n // cfg.ladder.ladder_every
            measured = run_instruments(
                model, cfg, iteration=n, anchor_beat=ANCHOR_BEAT, run_dir=run_dir
            )
            summary.setdefault("instruments", []).append(measured)
            # PERSISTED, NOT MERELY RETURNED (F-26). The cadence unit is the most
            # expensive measurement the loop makes — 1,200 problems at two
            # budgets, then 600 more for the primary — and it lived only in the
            # returned dict. A pod that vanished at iteration 12 lost iterations
            # 4 and 9's instrument passes with nothing on disk to resume from,
            # against a standing rule that volume + git + artifacts must always
            # suffice to continue elsewhere.
            #
            # It also makes M1-A2 §1 satisfiable from artifacts: the funnel
            # trigger's row must carry the contemporaneous beat-par delta, and
            # the delta was in memory while the entropy columns were on disk.
            # Provisional like the row, and truncated with it on resume.
            with (run_dir / "instruments.jsonl").open("a") as handle:
                handle.write(json.dumps(measured, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        else:
            absent["ladder_pass"] = (
                "not a ladder iteration (ladder runs on ladder.ladder_every cadence)"
            )
        if from_pool == 0 and not pool.members:
            absent["pool_par_fraction"] = (
                "league.par_from_pool_frac is 0, or no snapshot has been taken yet"
            )

        # --- the row, then the four-step commit ------------------------------
        row = iteration_row(
            stats,
            iteration=n,
            run_name=run_name,
            git_sha=git_sha(REPO),
            # F-25: THE CONFIG THAT RAN, not the constant we hoped ran. This was
            # `CAMPAIGN_FINGERPRINT`, and `run` is shared with golden — so every
            # golden row claimed the campaign's fingerprint while executing
            # golden config, and the one column whose job is "read these rows
            # against the config that produced them" named a config that had not
            # produced them.
            #
            # The campaign's rows still carry the registered value, but they
            # carry it BECAUSE `run_campaign` asserted the config at the door and
            # this records what that assertion licensed — not because the row
            # quotes a constant back to itself.
            config_fingerprint=config_fingerprint(cfg),
            cfg=cfg,
            ruleset_version=RULESET_VERSION,
            vocab_version=VOCAB_VERSION,
            schema_era=SCHEMA_ERA,
            evaluator_checkpoint_sha256=digest,
            pool_composition=composition,
            seconds_train=seconds_train,
            absent=absent,
        )
        row["seconds_self_play"] = round(seconds_self_play, 3)
        row["seconds_total"] = round(time.perf_counter() - began, 3)
        row["nan_skips"] = train_stats.nan_skips
        row["pool_refusals"] = pool.stats.refusals
        if ladder_index is not None:
            row["ladder_pass"] = ladder_index
        if "pool_par_fraction" not in absent:
            row["pool_par_fraction"] = round(from_pool / max(1, stats.episodes), 6)

        state = RunState(iteration=n, value_head=head, seed=state.seed)

        def write_row(row=row, n=n):
            if on_commit is not None:
                on_commit(n, "before_row")
            append_row(run_dir / "iterations.jsonl", row, ITERATION_FIELDS)
            if on_commit is not None:
                on_commit(n, "before_latest")

        commit_iteration(run_dir, ring, state, write_row)
        evaluator_source = checkpoint
        summary["iterations"].append(
            {
                "iteration": n,
                "solved": f"{stats.episodes_solved}/{stats.episodes}",
                "pool_par": from_pool,
                "ring": len(ring),
                "switch": "fired"
                if event.get("fired")
                else ("abstained" if event.get("abstained") else "refused"),
                "seconds": round(time.perf_counter() - began, 1),
            }
        )
        print(
            f"  iter {n:>2}: {stats.episodes_solved}/{stats.episodes} solved, "
            f"pool-par {from_pool}, ring {len(ring)}, "
            f"{time.perf_counter() - began:.1f}s"
        )

    rows = read_rows(run_dir / "iterations.jsonl", ITERATION_FIELDS)
    summary["alarms"] = alarm_census(rows)
    summary["committed"] = latest_committed(run_dir)
    return summary
