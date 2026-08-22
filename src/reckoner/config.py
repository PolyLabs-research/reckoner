"""Single source of truth for all reckoner configuration.

Two properties this module exists to guarantee, both inherited law:

**1. Unknown keys are a hard error.** Silence on a mistyped key is the same
failure family as a wrong git SHA — the run looks fine and the number is wrong.
A near-miss on a real field is *worse* than a typo, because the default it
leaves in place is plausible: ``train.value_q_mse`` (not a field) sitting next
to ``train.value_q_mse_weight`` (a field) would read, in the config file and in
the RUNLOG, as a request that changed nothing.

**One spelling per referent.** A config key's name is not private to the config
system — it appears in the spec, the amendment, the PREREG and every RUNLOG that
cites it. ``grep par_from_pool_frac`` has to hit documents, briefs, configs and
code as one corpus, so a key is spelled here exactly as the prose spells it, and
the section it lives in is chosen to make that spelling read cleanly rather than
the spelling shortened to suit the section.

**2. Every value is traceable.** Each field below is tagged with where its value
comes from:

  ``[spec §N]``        pinned by ``experiment2_math_base625_spec.md``
  ``[plan chunk N]``   pinned by ``experiment2_agent_plan.md``
  ``[v1.1]``           pinned by the par-game amendment
  ``[provisional]``    a placeholder the named chunk owns and will set

A ``[provisional]`` number is not a decision. It exists so the config tree has a
shape to load, round-trip, and fingerprint from chunk 0 — the chunk that owns it
sets it and pins it with a guard test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class EpisodeConfig:
    """The environment contract: positions, the step cap, and the checker."""

    # [plan chunk 3] "Step cap (config, default 24)". Hitting the cap is a loss
    # (z = -1) under the par game, identically to going over par — the episode
    # does not get to end in a shrug.
    step_cap: int = 24

    # [spec §3] SIMPLIFY is verified by random-assignment equivalence, k=32 draws
    # over a prime field. EVALUATE checks by exact evaluation and SOLVE by
    # substitution; neither needs a draw count.
    simplify_equiv_k: int = 32

    # [plan chunk 2/3] "a prime field" is specified; *which* prime is not, so it
    # is a decision recorded here rather than buried in the checker. 2**31 - 1 is
    # a Mersenne prime and the largest that keeps a product of two residues
    # inside int64 without intermediate overflow ((2**31)**2 = 2**62 < 2**63),
    # which is what makes the equivalence fuzz safe to vectorise in numpy.
    equiv_prime: int = 2_147_483_647


@dataclass
class ParConfig:
    """The par game [v1.1] — the opponent enters the label, never the search."""

    # [spec §3] BFS-exact par is canonical for depth <= 6; scripted-solver par is
    # a provisional *floor* above that, and is provenance-tagged `par_source`.
    bfs_exact_max_depth: int = 6

    # [v1.1] The resign-vs-par analog: concede when the best solution found is
    # already >= par + k steps. "Implemented but default off, calibration
    # deferred to campaign evidence, per the resignation lesson." k is therefore
    # NOT a calibrated number — it is inert while ``concede_enabled`` is False,
    # and calibrating it is a campaign decision, not a default.
    concede_enabled: bool = False
    concede_k: int = 2


@dataclass
class LeagueConfig:
    """The snapshot league [v1.1, plan chunk 9] — where pool-derived par comes from.

    Chunk 9 ports ``CheckpointPool`` and fills this section out (pool size,
    snapshot cadence, opponent draw). It exists now because one key already
    belongs to it, and that key needed an owner that was not ``par``.
    """

    # [spec §3, v1.1] Spelled exactly as the plan, the amendment, and every
    # future PREREG and RUNLOG spell it. One spelling per referent: the string
    # lives in prose as well as in config, and a grep for `par_from_pool_frac`
    # has to hit documents, briefs, configs and code as one corpus. The loader's
    # near-miss suggestion only ever guarded config-space; it could not guard
    # the half of the corpus that is written in English.
    #
    # Fraction of training problems whose par comes from a pool snapshot's own
    # solution, solved fresh at episode time — so par escalates with the model.
    # This is the built-in half of the funnel treatment; the Phase-3 generator
    # is the other half.
    par_from_pool_frac: float = 0.20

    # [chunk 9] Pool membership and sampling, DECLARED rather than emergent.
    #
    # The Phase-1 anchor seeds the pool: without it `par_from_pool_frac` is dead
    # until enough snapshots accumulate, and the anchor is the natural rung-zero
    # member — it is the checkpoint every later pass is already compared against.
    # Declared either way rather than left to whichever happens first.
    seed_pool_with_anchor: bool = True
    # [provisional — chunk 9] Iterations between snapshots. Enrollment is what
    # makes this a league rather than a fixed opponent: par escalates because the
    # loop FEEDS the pool its own checkpoints. A pool nothing enrolls into is the
    # snapshot league in name only, and `par_from_pool_frac` would price a
    # mechanism that never engages.
    snapshot_every: int = 1
    # Most recent snapshots retained as pool members. Bounded because pool par is
    # re-solved at episode time, so membership is a cost as well as a population.
    pool_size: int = 8
    # How a member is drawn per episode. "uniform" over current members; the key
    # exists so a future recency-weighted policy is a one-lever round rather than
    # a silent change of what "pool par" means.
    pool_sample: str = "uniform"


@dataclass
class ModelConfig:
    """[plan chunk 6] owns every ``[provisional]`` number here.

    What *is* pinned: the parameter envelope and the head shapes. The value head
    is 3-class W/D/L vs par plus a steps-to-solve auxiliary regression — §8
    decision 1 was reversed by the par-game amendment, and a config that quietly
    sets ``value_classes: 2`` is reverting that decision without saying so, which
    is why ``validate()`` rejects it.
    """

    # [spec §4] "~2-7M parameters". Recorded as config so chunk 6's guard test
    # has a pinned envelope to assert the built model against, rather than a
    # number in prose.
    param_budget_min: int = 2_000_000
    param_budget_max: int = 7_000_000

    # [spec §4, v1.1] Value head = 3-class W/D/L vs par + steps-to-solve aux.
    value_classes: int = 3
    steps_aux_head: bool = True

    # [chunk 6, sized to the envelope] Trunk shape. n_layers is 6 rather than
    # the v2 family's 8 because 8 puts the total at ~6.6M — inside the 2-7M
    # envelope but pressed against its ceiling, leaving nothing for the head
    # growth a later round might want.
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    # [chunk 6, measured — runs/state_extent.json] Sized from the REACHABLE-state
    # distribution, not the dataset's start states. Episodes grow before they
    # shrink, so the longest state inside an episode strictly exceeds the longest
    # problem: start states reach 64 tokens / 17 sites, but 24-step random walks
    # reach 332 tokens / 121 sites. Sizing from the dataset would have set this
    # to 64 and guaranteed overflow the first time search explored.
    #
    # 512 is ~1.5x the measured p100. Overflow raises StateTooLarge and the
    # episode is aborted and counted — never cropped.
    seq_len: int = 512
    # [chunk 6, measured] Sites per state, same population, same headroom.
    # Action space is N_RULES * max_sites = 7 * 192 = 1344.
    max_sites: int = 192
    # [chunk 6] Bilinear projection dim for the factorized rule x site policy
    # head — the (from, to) factorization transplanted.
    d_policy: int = 128


@dataclass
class SearchConfig:
    """[plan chunk 7] the careful port. Array-tree MCTS + Gumbel + Seq. Halving."""

    # [plan §8 decision 5] THE porting hazard, and it is a config key so that a
    # run cannot silently be doing the wrong backup: "there is no opponent, so
    # backup does NOT negate per ply." ``validate()`` rejects any other value —
    # an alternating-perspective backup in this domain is not a variant, it is a
    # bug, and chunk 7 exists partly to prove the single-agent arithmetic.
    perspective: str = "single"

    # [provisional — chunk 7] Chunk 7's batched-equivalence parametrization is
    # m in {3, 5, 12, 16} x sims in {6, 16, 31, 48} "from day one" (the
    # odd-parity lesson, pre-paid). These defaults are the top of that grid; the
    # depth-1 gate's (sims, m) is set only after chunk 2's measured branching
    # factor makes the gate arithmetic checkable.
    sims: int = 48
    gumbel_m: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0

    # [chunk 8 — DECIDED, was `batch_leaves` provisional-chunk-7] How many
    # concurrent SEARCHES pool their leaf evaluations, never how many leaves are
    # pooled inside one tree. The rename is the whole point: within-tree pooling
    # was removed in chunk 8 part 0 because simulation k+1's selection depends on
    # k's backup, so pooling inside a tree changes what the tree does. A key
    # named for deleted semantics is a lie with a schema — the old name described
    # a mechanism this project has forbidden. Decided rather than provisional
    # because the quantity it now names is a batching width, set by the box, not
    # a search hyper-parameter awaiting a measurement.
    batch_searches: int = 512

    # [plan chunk 7] eval/self-play profiles kept. The default is the *self-play*
    # value, so anything that forgets to choose keeps generating diverse data
    # rather than silently freezing it. Suites and the ladder run root_noise off.
    root_noise: bool = True


@dataclass
class TrainConfig:
    """Optimizer and loss parameters [plan chunks 8, 9]."""

    # [plan chunk 9] "z/q value blend on by default (`value_q_mse_weight: 0.5`)".
    # Its self-referential hazard is *structurally absent here*: the checker, not
    # the model's own Q, is the source of solved/not-solved. Blending Q into the
    # value target cannot bootstrap a false solve, because solved is decided
    # outside the network. That is the reason this domain suits the method, and
    # it is why the weight is on from day one rather than earned by an ablation.
    value_q_mse_weight: float = 0.5

    # [plan chunk 9] "rehearsal machinery ported (dormant, `rehearsal_frac: 0.0`
    # default — the lever exists before it's needed)".
    # [M1-A4 §5] 0.65, taken MECHANICALLY by the rule frozen in SWEEP-m1a4.md:
    # the smallest f whose gate-10b top-1 holds the band, on the worst seed.
    # Three seeds at 0.65 all returned 0.9699 — the anchor's own count, 1192/1229
    # — with zero spread, so the estimator question declared before the seeds
    # existed turned out not to bite. Was 0.0 while `rehearsal_split` was
    # computed and discarded (F-31); the wire and the value are separate
    # decisions and were taken separately.
    rehearsal_frac: float = 0.65

    # [chunk 8 — DECIDED by the timing slice, was provisional 512] The pre-flight
    # measured 113.6 / 124.3 / 117.1 / 105.1 examples/s at batch 64 / 128 / 256 /
    # 512 (`runs/pilot_phase1_timing.json`). Throughput is flat across 64-256 and
    # falls at 512, so batch size is NOT a throughput lever on this box — which
    # means it is chosen for gradient quality at fixed wall-clock, and at fixed
    # wall-clock 128 buys 4x the optimiser steps of 512. 5,000 steps is 85.8 min
    # at 128 against 406 min at 512, for 2.04 epochs against 8.2.
    batch_size: int = 128
    lr: float = 2e-4

    # [plan chunk 8] "constant/cosine LR from config" — the schedule is a config
    # key, not a script flag, so a run's own resolved config.yaml states which
    # curve produced its loss trace. ``validate()`` rejects anything else.
    # [chunk 9] Fraction of ring rows reserved for evaluating the value head's
    # switch criterion. Held out from TRAINING, but deliberately IN-DISTRIBUTION:
    # see `valuegate.evaluate_head` for why that is the right shape here and not
    # the F-09 hazard repeated.
    ring_holdout_frac: float = 0.1
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_steps_per_iter: int = 400
    replay_capacity: int = 500_000

    # [provisional — chunk 6/8] Loss weights. ``steps`` is the auxiliary
    # regression, masked to solved episodes.
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    steps_loss_weight: float = 1.0

    # [plan chunk 8] NaN-skip guard, carried over verbatim: skip the update on a
    # non-finite gradient, count the skips, abort if they become a rate rather
    # than a transient.
    nan_abort_frac: float = 0.01
    nan_abort_min_steps: int = 20


@dataclass
class NumericsConfig:
    """[chunk 9 part 0] Which dtype produced a number — decided once, here.

    Two regimes, and they are not interchangeable:

    * ``measure_dtype`` — **fp32**. This is the regime the GPU/CPU equivalence
      gate licensed: tier 1 held at 512/512 argmax and top-8 across all four
      width buckets in fp32 (`FINDINGS.md` F-12). Every gate number, eval pass
      and ladder measurement runs here, so "which dtype produced this?" has one
      answer and it is the answer that was checked.
    * ``train_dtype`` — **bf16**, per AGENTS.md §4.5 (bf16 autocast, fp32 master
      weights). Training is guarded by the NaN-skip kit and its loss curve, not
      by cross-device identity, so it does not need the licensed regime and does
      need the throughput.

    bf16's only cross-device measurement is informational: 511/512 argmax
    agreement at max abs delta 1.369e-01. That is why it trains and does not
    measure. ``validate()`` rejects a measurement regime the equivalence gate
    never licensed.
    """

    measure_dtype: str = "fp32"
    train_dtype: str = "bf16"


@dataclass
class GeneratorConfig:
    """[plan chunk 5] the procedural problem generator and the frozen instruments."""

    # [plan chunk 5] Difficulty is parameterized by *verified minimum solution
    # depth* — BFS over the rule graph for depths <= 6. The depth label is
    # measured, not assumed, and the suites re-verify it independently.
    max_bfs_depth: int = 6

    # [plan chunk 5] "100K-problem training set + suites + eval sets on disk".
    train_set_size: int = 100_000

    # [plan chunk 5] Frozen instruments: `solve_in_N.jsonl` N = 1...6, 200 each,
    # BFS-verified, contamination-tested against every training set, and never
    # regenerated. Mate-in-N reborn [spec §6].
    suite_problems_per_depth: int = 200
    suite_depths: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])


@dataclass
class LadderConfig:
    """[plan chunk 10] the ladder — external, calibrated, impossible to flatter."""

    # [plan chunk 11] Campaign M1 runs at `--ladder-every 5` over ~20 iterations.
    ladder_every: int = 5

    # [inherited law] Paired-difference bootstrap is the test of record for
    # pass-vs-pass comparisons.
    bootstrap_resamples: int = 10_000

    # [provisional — chunk 10] Paired problem set size per ladder pass (the
    # frozen-openings-book analog). Since M1-A4 §4 it is also the per-file
    # population cap that lets `golden` exercise a cadence iteration — inert at
    # campaign scale, where every suite file holds exactly 200 rows.
    #
    # The sympy rung's budgets lived here and were DELETED by M1-A4 §4: they
    # configured a rung F-34 dropped from M1, and nothing read them. When the
    # sympy rung returns in M2 its budgets get declared at the site that consumes
    # them, rather than riding as dead weight through a campaign fingerprint.
    problems_per_pass: int = 200


@dataclass
class CampaignConfig:
    """[M1-A2] The campaign's extent, treatment size, and thread pins.

    These were absent, which is why the driver could not take its extent from
    fingerprinted config and honour the no-behavioural-flags rule at the same
    time. Adding them moves the fingerprint; M1-A2 records both eras.
    """

    # EXTENT, and it is hard rather than approximate. "~20 iterations" carried an
    # ambiguity the prereg could not afford: a campaign extended until the CI
    # excludes zero is p-hacking wearing patience. The analysis point is 20 and
    # MAY NOT BE RAISED AFTER LAUNCH; early termination happens only through the
    # prereg's BLOCKED and decision branches, with the primary then evaluated at
    # the last completed ladder pass and reported as exactly that.
    iterations: int = 20

    # TREATMENT SIZE, chosen-not-derived — see M1-A2 §3. No measured staleness
    # optimum exists; this is a judgement about reuse and criterion viability
    # whose consequences are computed on the page.
    episodes_per_iteration: int = 400

    # THREAD PINS, recording what the licence ran under rather than selecting new
    # values. Evidential classes are stated in M1-A2 §4: intra_op is EXERCISED
    # (defaults differed 16/32 across hosts and only the clamp made the licence's
    # parity possible); interop is OBSERVED (constant at 32 throughout,
    # undiscriminated, so it licenses only "this is what ran").
    intra_op_threads: int = 8
    interop_threads: int = 32

    # DEVICE IS FINGERPRINTED (ruled 2026-08-22, after F-51).
    #
    # M1 ran on a `+cpu` wheel for 9.6 hours and nobody could state that for two
    # days. The 62-field census covers config; the recording contract covers
    # artifacts; NOTHING covered the execution environment — so the one variable
    # that silently rescales every wall-clock number in the record was the one
    # variable no gate could see.
    #
    # Moving to a GPU build changes numerics (MCTS over a small net is exactly
    # where reduction-order differences show), so it is a treatment applied
    # mid-campaign, and it lands under the same rule as any fingerprinted change:
    # amendment, not convenience. Fingerprinting it is what makes that rule
    # enforceable rather than aspirational.
    device: str = "cpu"


@dataclass
class Config:
    """Root configuration container."""

    # [inherited law] One seed in config fans out to all RNGs; library code takes
    # explicit rng/Generator parameters, never global random state.
    seed: int = 42
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    par: ParConfig = field(default_factory=ParConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)
    campaign: CampaignConfig = field(default_factory=CampaignConfig)


# ---------------------------------------------------------------------------
# Validation — the laws a config file is not allowed to break
# ---------------------------------------------------------------------------


def validate(cfg: Config) -> None:
    """Reject configs that violate a pinned decision.

    Checked at load, not at first use: an overnight campaign that ran the wrong
    backup rule is not recoverable by noticing it in iteration 3's log.
    """
    if cfg.search.perspective != "single":
        raise ValueError(
            f"search.perspective must be 'single'; got {cfg.search.perspective!r}. "
            "This game has no opponent in the tree (plan §8 decision 5): the par "
            "opponent enters the label, never the search, so backup does NOT "
            "negate per ply. An alternating-perspective backup here is a bug, "
            "not a variant."
        )
    if cfg.model.value_classes != 3:
        raise ValueError(
            f"model.value_classes must be 3; got {cfg.model.value_classes}. "
            "The par game (amendment v1.1) reversed §8 decision 1: the value head "
            "is 3-class W/D/L vs par, not 2-class solved/timed-out. Changing it "
            "here would revert a recorded decision silently."
        )
    if cfg.model.param_budget_min > cfg.model.param_budget_max:
        raise ValueError(
            f"model.param_budget_min ({cfg.model.param_budget_min}) exceeds "
            f"param_budget_max ({cfg.model.param_budget_max})."
        )
    if cfg.league.pool_sample not in ("uniform",):
        raise ValueError(
            f"league.pool_sample must be 'uniform'; got {cfg.league.pool_sample!r}. "
            "A new sampling policy changes what 'pool par' means and is a "
            "one-lever round, not a config edit."
        )
    if cfg.league.pool_size < 1:
        raise ValueError(f"league.pool_size must be >= 1; got {cfg.league.pool_size}")
    if cfg.numerics.measure_dtype != "fp32":
        raise ValueError(
            f"numerics.measure_dtype must be 'fp32'; got {cfg.numerics.measure_dtype!r}. "
            "fp32 is the regime the chunk-9 equivalence gate licensed (tier 1 exact, "
            "512/512, all width buckets). bf16 has only an informational cross-device "
            "number (511/512). Measuring in an unlicensed regime would make every gate "
            "number's provenance a question nobody checked."
        )
    if cfg.numerics.train_dtype not in ("fp32", "bf16"):
        raise ValueError(
            f"numerics.train_dtype must be 'fp32' or 'bf16'; got {cfg.numerics.train_dtype!r}"
        )
    if cfg.train.lr_schedule not in ("constant", "cosine"):
        raise ValueError(
            f"train.lr_schedule must be 'constant' or 'cosine'; got "
            f"{cfg.train.lr_schedule!r}. The plan names those two; a third curve "
            "is a decision, not a typo, and belongs in a round."
        )
    if cfg.train.lr_warmup_steps < 0:
        raise ValueError(f"train.lr_warmup_steps must be >= 0; got {cfg.train.lr_warmup_steps}")
    if not 0.0 <= cfg.league.par_from_pool_frac <= 1.0:
        raise ValueError(
            f"league.par_from_pool_frac must be in [0, 1]; got {cfg.league.par_from_pool_frac}"
        )
    if not 0.0 <= cfg.train.rehearsal_frac < 1.0:
        raise ValueError(f"train.rehearsal_frac must be in [0, 1); got {cfg.train.rehearsal_frac}")
    if cfg.episode.step_cap < 1:
        raise ValueError(f"episode.step_cap must be >= 1; got {cfg.episode.step_cap}")
    if cfg.episode.simplify_equiv_k < 1:
        raise ValueError(
            f"episode.simplify_equiv_k must be >= 1; got {cfg.episode.simplify_equiv_k}"
        )

    # A suite depth beyond the BFS horizon cannot have a verified depth label or
    # a canonical par — both are BFS-exact by definition [plan chunk 5, v1.1] —
    # so the instrument would be frozen around an unverified number.
    over = [d for d in cfg.generator.suite_depths if d > cfg.generator.max_bfs_depth]
    if over:
        raise ValueError(
            f"generator.suite_depths {over} exceed max_bfs_depth "
            f"({cfg.generator.max_bfs_depth}). Suite depth labels and par are "
            "BFS-verified; beyond the BFS horizon neither can be canonical."
        )
    if any(d < 1 for d in cfg.generator.suite_depths):
        raise ValueError(
            f"generator.suite_depths must all be >= 1; got {cfg.generator.suite_depths}"
        )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts (handles nesting)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _from_dict(raw: dict, cls: type) -> Any:
    """Rebuild a dataclass tree from nested plain dicts.

    ``typing.get_type_hints`` is used rather than ``field.type`` because
    ``from __future__ import annotations`` makes every annotation a *string* —
    with the raw strings, no nested dataclass is ever recognised, every nested
    section silently falls through as a dict, and the config that loads is not
    the config that was written.
    """
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        nested = hints.get(f.name)
        if isinstance(value, dict) and dataclasses.is_dataclass(nested):
            value = _from_dict(value, nested)
        kwargs[f.name] = value
    return cls(**kwargs)


def _assert_known_keys(raw: dict, cls: type, source: str, prefix: str = "") -> None:
    """Reject config keys that do not exist on the dataclass tree."""
    valid = {f.name for f in fields(cls)}
    hints = typing.get_type_hints(cls)
    for key, value in raw.items():
        dotted = f"{prefix}{key}"
        if key not in valid:
            near = sorted(n for n in valid if n.startswith(key[:6]) or key.startswith(n[:6]))
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise ValueError(
                f"{source}: unknown config key {dotted!r} for {cls.__name__}.{hint} "
                f"Valid: {sorted(valid)}"
            )
        nested = hints.get(key)
        if isinstance(value, dict) and dataclasses.is_dataclass(nested):
            _assert_known_keys(value, nested, source, prefix=f"{dotted}.")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load a Config from YAML, optionally applying dot-notation overrides.

    ``path=None`` returns the defaults. ``overrides`` keys are dotted, e.g.
    ``{"search.sims": 16}``, and are checked against the tree exactly as file
    keys are — a CLI flag mistyped on the command line is the same failure as a
    YAML key mistyped in a file.
    """
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: config root must be a mapping, got {type(raw).__name__}")
        _assert_known_keys(raw, Config, str(path))
        cfg = _from_dict(raw, Config)
    else:
        cfg = Config()

    if overrides:
        cfg_dict = _dataclass_to_dict(cfg)
        for key, value in overrides.items():
            parts = key.split(".")
            target = cfg_dict
            for i, part in enumerate(parts[:-1]):
                if part not in target or not isinstance(target[part], dict):
                    raise ValueError(
                        f"override {key!r}: {'.'.join(parts[: i + 1])!r} is not a config section"
                    )
                target = target[part]
            if parts[-1] not in target:
                raise ValueError(f"override {key!r}: unknown config key {parts[-1]!r}")
            target[parts[-1]] = value
        _assert_known_keys(cfg_dict, Config, "overrides")
        cfg = _from_dict(cfg_dict, Config)

    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Serialise a Config to YAML. Round-trips exactly with load_config."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(_dataclass_to_dict(cfg), fh, default_flow_style=False, sort_keys=True)


def config_diff(base: Config, other: Config) -> dict[str, tuple[Any, Any]]:
    """Every key where ``other`` differs from ``base``, as ``{dotted: (base, other)}``.

    **The registered enforcement for one-lever-per-round.** ``configs/default.yaml``
    equals the dataclass defaults (a test pins that), so a campaign config's diff
    against the default *is* the campaign's lever list. When the first campaign
    file lands (chunk 8 or 11), its test is::

        assert set(config_diff(Config(), load_config("configs/m1.yaml"))) == LEVERS

    where ``LEVERS`` is the set the PREREG registered. That turns
    one-lever-per-round from a reviewed property into an executable one: a second
    lever added quietly during a run fails the build, and it fails it by *name*.

    Registered now, before the first campaign, so it is adopted by design rather
    than after an incident — which is the whole argument of the inherited-law
    block.
    """
    flat_base = _flatten(_dataclass_to_dict(base))
    flat_other = _flatten(_dataclass_to_dict(other))
    keys = sorted(set(flat_base) | set(flat_other))
    return {
        k: (flat_base.get(k), flat_other.get(k))
        for k in keys
        if flat_base.get(k) != flat_other.get(k)
    }


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Nested config dict -> ``{"section.key": value}``. Leaves stay whole."""
    flat: dict[str, Any] = {}
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def config_fingerprint(cfg: Config) -> str:
    """SHA-256 of the canonical (sorted-key) YAML representation.

    [inherited law] Goes into every checkpoint and dataset ``meta.json`` beside
    the git SHA, so a run's rows can always be read against the config that
    produced them.
    """
    canonical = yaml.safe_dump(_dataclass_to_dict(cfg), default_flow_style=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
