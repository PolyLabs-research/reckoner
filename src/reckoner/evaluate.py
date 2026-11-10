"""The one model evaluator, and the one model rung.

`model_evaluator` existed as a private helper inside `scripts/shakedown.py`. Two
scripts now need it, and two copies of a function that decides *how much the value
head is trusted* is exactly the kind of duplication that lets one copy drift into
trusting a head the other silences. One declaration, every consumer.

`ModelArm` makes the model a ladder rung with the same interface as
:class:`reckoner.arms.GreedyHeuristic` — it plays one problem to a terminal
outcome and returns an :class:`reckoner.arms.ArmResult` in ``CURRENCY_Z``. It is
rule-denominated: it acts in our rules, so its steps are steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np
import torch

from reckoner.arms import ArmError, ArmResult
from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.logschema import CURRENCY_Z
from reckoner.model import Reckoner, StateTooLarge, encode
from reckoner.rules import apply, legal_actions
from reckoner.search import search


class EvaluatorModeError(RuntimeError):
    """A search evaluator was built from a model that is still in train mode."""


def assert_eval_mode(model: Reckoner, cfg: Config) -> None:
    """Refuse a model in train mode. **Named so it is findable** (F-22).

    This began as the body of `model_evaluator`'s guard, and then as a bare
    `model_evaluator(...)` call at the campaign's instrument seam whose only
    purpose was raising. Ruff correctly flagged that as an unused expression, and
    a guard that looks like dead code is eventually deleted as dead code — by a
    reader with less context, or by a stricter lint configuration.

    Same reasoning as the seam pin: the protection must be findable by someone
    who does not already know it exists.
    """
    if model.training:
        raise EvaluatorModeError(
            "the search evaluator was built from a model in TRAIN mode, so dropout "
            f"(p={cfg.model.dropout}) is live inside every search. That is three "
            "defects at once: the priors become nondeterministic, the policy plays "
            "worse than its own weights, and entropy_prior_* — the column the "
            "funnel signature's thresholds are a fraction OF — measures dropout "
            "rather than the policy. Call model.eval() before building an evaluator."
        )


def model_evaluator(model: Reckoner, cfg: Config, value_scale: float):
    """The declaration applied: value contributes ``value_scale``, priors always.

    ``value_scale`` comes from :func:`reckoner.valuegate.value_contribution` and is
    **0.0 while the head is untrusted** — the search reads no opinion the head has
    not earned (F-14's noise-as-signal closure). The policy prior is unaffected,
    and it trains toward π′ from the first iteration (F-42). While the head is
    silent, π′ moves the prior only where the search holds a proof.
    """
    # F-22. THE MODE IS A PROPERTY OF THE EVALUATOR, so it is checked here rather
    # than trusted to a `.eval()` somewhere up the call chain. The campaign ran
    # every episode with dropout live: `load_checkpoint` returns a model in train
    # mode, `train_on_ring` re-asserts train mode, and nothing ever restored eval
    # — so 10% of activations were randomly zeroed inside every search. It
    # survived three chunks because `golden` played `uniform_stub`, which has no
    # network to drop out; the moment D-A1 §1.1 put the real model in the loop,
    # the mode became load-bearing and the first gate to compare two runs caught
    # it.
    assert_eval_mode(model, cfg)
    width = 7 * cfg.model.max_sites

    def evaluate(leaves):
        encoded, keep = [], []
        for i, (problem, expr) in enumerate(leaves):
            try:
                encoded.append(encode(problem, expr, cfg))
                keep.append(i)
            except StateTooLarge:
                continue
        out = [(np.zeros(width, dtype=np.float32), 0.0) for _ in leaves]
        if not encoded:
            return out
        with torch.no_grad():
            policy, value_logits, _ = model(
                torch.stack([e.tokens for e in encoded]),
                torch.stack([e.site_positions for e in encoded]),
            )
            probs = torch.softmax(value_logits, dim=1)
            expected = (probs[:, 0] - probs[:, 2]).tolist()
        for slot, i in enumerate(keep):
            out[i] = (policy[slot].numpy().astype(np.float32), value_scale * float(expected[slot]))
        return out

    return evaluate


@dataclass
class ModelArm:
    """The model as a ladder rung: search, act, repeat, until terminal.

    Unbatched by construction. `runner.run_iteration` pools leaf evaluations
    across concurrent searches and is far faster, but it reports an **aggregate**
    — and `pair_scores` needs one row per problem. Reaching for the fast path here
    would mean either changing an established interface or reconstructing
    per-problem outcomes from a histogram, which is un-aggregating something that
    was aggregated. So: the slow path, on the smaller set, deliberately.
    """

    model: Reckoner
    name: str = "model"
    currency: str = CURRENCY_Z
    nondeterministic: bool = False
    value_scale: float = 0.0
    sims: int | None = None
    m: int | None = None
    #: The eval profile is `root_noise=False`. Stated here rather than assumed
    #: from the config default, which is the SELF-PLAY value: anything that
    #: forgets to choose keeps generating diverse data rather than silently
    #: freezing it, so a measuring rung must choose explicitly.
    root_noise: bool = False

    def plays(self, problem: Problem) -> bool:
        return True

    def profile_config(self, cfg: Config) -> Config:
        """The config this rung actually runs under, with its profile applied.

        Exposed rather than applied privately so a report can state the protocol
        it measured under instead of the protocol it was handed.
        """
        if cfg.search.root_noise == self.root_noise:
            return cfg
        return replace(cfg, search=replace(cfg.search, root_noise=self.root_noise))

    def play(self, problem: Problem, cfg: Config, seed: int) -> ArmResult:
        cfg = self.profile_config(cfg)
        evaluator = model_evaluator(self.model, cfg, self.value_scale)
        rng = random.Random(seed)
        checker = random.Random(seed ^ 0x5EED)
        expr = problem.expr
        for step in range(cfg.episode.step_cap):
            if not legal_actions(expr):
                return ArmResult(False, step, CURRENCY_Z)
            result = search(
                problem, expr, evaluator, cfg, rng, sims=self.sims, m=self.m, steps_taken=step
            )
            if result.chosen is None:
                return ArmResult(False, step, CURRENCY_Z)
            expr = apply(expr, *result.chosen)
            if verify(problem, expr, cfg, checker):
                return ArmResult(True, step + 1, CURRENCY_Z)
        return ArmResult(False, cfg.episode.step_cap, CURRENCY_Z)

    def probe(self, problem: Problem, cfg: Config) -> None:
        """Construction gate: under the eval profile it is a function of the state.

        The same gate `GreedyHeuristic` gets, and it means more here — the search
        *has* an rng, and the claim is that with root noise off the rng cannot
        change the outcome. A rung whose repeatability nobody checked produces
        differences nobody can attribute (the skill-limiter lesson).
        """
        if self.root_noise:
            raise ArmError(
                f"{self.name} was probed for determinism with root_noise on. That is "
                "the self-play profile, where varying IS the correct behaviour — the "
                "probe would fail on a correct arm. Probe the eval profile."
            )
        a = self.play(problem, cfg, seed=0)
        b = self.play(problem, cfg, seed=99)
        if (a.solved, a.steps) != (b.solved, b.steps):
            raise ArmError(
                f"{self.name} declares itself deterministic under the eval profile and "
                f"is not: {(a.solved, a.steps)} vs {(b.solved, b.steps)}. With root "
                "noise off the search's rng must not reach the outcome."
            )


__all__ = ["ModelArm", "model_evaluator"]
