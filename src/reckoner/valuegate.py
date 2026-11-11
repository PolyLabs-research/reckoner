"""One declaration of whether the W/D/L head is trusted, and two consumers of it.

Noise-as-signal
---------------
This module exists because of a defect class that is the **evil twin of
absence-becomes-zero**. The system *declared* the value head untrained — loss
weight 0, by design, masked with a reason — and the search read that
declared-absent opinion as evidence anyway. Under the old solved-flat terminal
scale a solve backed up ``+1.0`` and drowned the noise; under the z currency an
at-par solve is ``0.0``, and noise around zero routinely beats it, so the search
preferred a noisily-optimistic unexplored line over a **proven** draw
(`FINDINGS.md` F-14). ``Absent.__bool__`` raises for exactly this reason one layer
down; the same principle applies to an opinion nobody has earned the right to
hold.

The structural fix is **one declaration, two consumers**: while the head is
untrained-on-z, the loss masks it *and* the search contributes zero value — both
read :func:`value_contribution` of the same state, so they cannot disagree. A
config flag consulted by one and forgotten by the other is the two-gates-one-name
hazard again.

The switch-over
---------------
Neither a guessed iteration nor a mid-run human call: a **pre-registered
criterion**, four-tupled per rider (c), on held-out z accuracy. It fires **once**
and **ratchets** — the head does not flicker between trusted and not — and it is
logged as an event row rather than inferred from a behaviour change.

The four-tuple's slots keep their countersigned definitions even when one carries
no information — otherwise the next metric that arrives, where they genuinely
differ, gets mis-filled by habit:

* **floor** — what the metric can return *regardless of skill*. For an accuracy
  it is **0**: nothing guarantees a minimum, because an anti-correlated head
  scores *below* the trivial model. The slot is filled and marked uninformative.
* **null** — the best *trivial* model, run. Here that is the constant predictor,
  and under balanced accuracy it scores exactly **1/K**.

An earlier draft merged the two and called majority-class accuracy "the floor".
The tell was in its own validation sentence — *"it ties the floor exactly"*
describes the **null** arm.

Why balanced accuracy, and not raw
-----------------------------------
Raw accuracy with a bar of ``max(0.60, null + margin)`` is **structurally
unfirable in this project's own predicted early regime.** Training problems are
100% ``par_source="bfs"`` and nothing beats exact par, so early z is two-class by
construction — ``{0, -1}``, with no ``+1`` until pool or scripted par exists. A
value-silent iteration-0 search that solves most problems at par (which the
chunk-8 baselines say it will) makes z draw-heavy: at 85/15 the bar would be
**0.95** held-out accuracy, and at 98/2 it would be **1.08**. Calling 1.08
"unreachable, correctly" was wrong — a bar the expected regime makes unreachable
means the switch never fires, value stays silent for the whole campaign, and the
lever the loop is named for is disabled by its own qualifying exam. **A
never-firable trigger is the always-firing detector's mirror image**; rider (b)'s
corollary has two ends and this criterion was sitting on the other one.

Balanced accuracy over the classes with support asks the thing worth asking —
*does the head know anything beyond the base rate* — without demanding
near-perfection on the majority to prove competence on the minority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Margin above the null (1/K) that the head must clear, in balanced accuracy.
#:
#: **Priced as the error rate of a one-way door**, because the switch ratchets and
#: a false fire is permanent. With ``MIN_CLASS_SUPPORT`` samples in the rarest
#: class, se(balanced accuracy) under the null is ``0.5 * sqrt(2 * 0.25 / n)``:
#:
#:     support  margin  se       z     P(fire | null)  over 20 iterations
#:      30      0.10    0.0645   1.55  6.07%           71.4%
#:      50      0.15    0.0500   3.00  0.135%           2.67%
#:     100      0.15    0.0354   4.24  0.001%           0.02%
#:
#: 0.10 at support 30 spuriously fires in **71% of 20-iteration campaigns**, which
#: is not a detector. 0.15 at support 100 is 0.02%. Abstention costs time; a false
#: fire costs the campaign, so the door is conservative.
MARGIN = 0.15

#: Minimum classes with support before the criterion will judge. With one class,
#: z did not vary and no head can distinguish itself from a constant — the
#: criterion abstains with that reason rather than through an unreachable bar.
MIN_CLASSES = 2

#: Minimum samples in the RAREST class before the criterion will judge at all.
#: Balanced accuracy on two minority samples is not a measurement — minority
#: recall can only be 0, 0.5 or 1. Below this the criterion ABSTAINS, which is
#: recorded distinctly from failing: "not evaluable yet" and "evaluated and
#: refused" are different states and only one of them is evidence.
MIN_CLASS_SUPPORT = 100


class ValueGateError(ValueError):
    """A misuse of the value-head declaration."""


@dataclass
class ValueHeadState:
    """Whether the W/D/L head has earned the right to be listened to.

    ``live`` is the single declaration. It starts False — spec §5 trains Phase-1
    value on steps-to-solve and leaves W/D/L at loss weight 0, so at iteration 0
    the head is noise — and it ratchets to True exactly once, when the criterion
    fires.
    """

    live: bool = False
    switched_at_iteration: int | None = None
    switched_accuracy: float | None = None
    switched_threshold: float | None = None

    def as_dict(self) -> dict:
        return {
            "live": self.live,
            "switched_at_iteration": self.switched_at_iteration,
            "switched_accuracy": self.switched_accuracy,
            "switched_threshold": self.switched_threshold,
        }


def value_contribution(state: ValueHeadState) -> float:
    """The multiplier BOTH consumers apply. 1.0 when trusted, 0.0 while not.

    * **search** scales the evaluator's value by this, so an untrained head
      contributes nothing and every unexplored child ties at 0.0 — selection then
      falls back to priors and the Gumbel draw, which is what found the solve in
      F-14's ablation arm.
    * **loss** scales ``train.value_loss_weight`` by this, so the head is not
      trained toward a target it is meanwhile being trusted for.

    One function, two call sites, one state. They cannot disagree.
    """
    return 1.0 if state.live else 0.0


def evaluate_head(model, ring, cfg, slots: Sequence[int]) -> tuple[list[int], list[int]]:
    """The criterion's evaluation slice: **held-out ring rows, in distribution.**

    Held out from *training* — the head must not be graded on rows it fit — but
    drawn from the same episode stream the loop produces, and that is deliberate
    rather than an oversight.

    **Why F-09's anxiety does not import here.** F-09 was about a *generalisation*
    claim: a held-out set that shared states with training inflated a metric
    meant to predict performance on unseen problems. This criterion asks a
    different question — *should the search trust this head on the states the
    search actually visits?* That is an **in-distribution** question by
    construction. Grading the head on out-of-distribution states would answer a
    question nobody is asking and would gate the switch on a capability the
    search never exercises.

    Returns ``(labels, predictions)`` — stored z against the head's argmax class,
    both in W/D/L class space.
    """
    import torch

    from reckoner.train import ring_batch

    if not slots:
        return [], []
    # The switch criterion's population is pinned to the pre-F-54 row rule. Under
    # 'improved' a row with a cropped π′ sits out of the batch, so reading the
    # configured target here would let the POLICY lever move the population the
    # VALUE switch is judged on. The value labels and inputs are the same either
    # way; only who is in the population could differ, and now cannot (F-54).
    batch = ring_batch(ring, list(slots), cfg, target="visits")
    if batch is None:
        return [], []
    model.eval()
    with torch.no_grad():
        _policy, value_logits, _steps = model(batch.tokens, batch.site_positions)
    predictions = value_logits.argmax(dim=1).tolist()
    labels = batch.z_class.tolist()
    assert len(labels) == len(predictions)
    return labels, predictions


def class_census(labels: Sequence[int]) -> dict[int, int]:
    """Support per class. Logged in the switch event so K is data, not a caption."""
    if not labels:
        raise ValueGateError("no held-out labels — the criterion cannot be evaluated on nothing")
    counts: dict[int, int] = {}
    for z in labels:
        counts[z] = counts.get(z, 0) + 1
    return dict(sorted(counts.items()))


def balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Mean per-class recall over the classes with support.

    A constant predictor scores exactly ``1/K`` here whatever the imbalance,
    which is what makes the bar firable in a draw-heavy regime.
    """
    census = class_census(labels)
    recalls = []
    for cls, support in census.items():
        hits = sum(1 for a, b in zip(labels, predictions, strict=True) if a == cls and b == cls)
        recalls.append(hits / support)
    return sum(recalls) / len(recalls)


def switch_criterion(labels: Sequence[int], predictions: Sequence[int]) -> dict:
    """The pre-registered switch bar, as a four-tuple.

    ``floor``       **0.0** — an accuracy has no structural minimum, because an
                    anti-correlated head scores below the trivial model. Filled
                    and marked uninformative rather than merged into the null.
    ``null``        ``1/K`` — the constant predictor's balanced accuracy, where K
                    is the number of classes WITH SUPPORT in this slice.
    ``threshold``   ``1/K + MARGIN``.
    ``measured``    the head's balanced accuracy.

    Returns the whole tuple plus the class census, always, so a row records what
    the bar *was* and what K it was judged against. Abstains (``evaluable``
    False) when the rarest class has fewer than ``MIN_CLASS_SUPPORT`` samples.
    """
    if len(labels) != len(predictions):
        raise ValueGateError(
            f"{len(labels)} labels against {len(predictions)} predictions — the "
            "criterion cannot be evaluated on a misaligned pair"
        )
    census = class_census(labels)
    k = len(census)
    smallest = min(census.values())
    null = 1.0 / k
    threshold = null + MARGIN
    measured = balanced_accuracy(labels, predictions)

    # TWO abstention causes, and the second was found by the shakedown itself.
    #
    # K == 1 means z had NO VARIANCE in this slice — every episode drew, or every
    # one lost. Balanced accuracy is then trivially 1.0 for a constant predictor,
    # and 1/K + MARGIN = 1.15 is unreachable. Refusing to fire is correct (a head
    # cannot demonstrate knowledge of a quantity that did not vary), but it must
    # refuse for the STATED reason rather than through a threshold that happens to
    # exceed 1.0 — otherwise it files as "evaluated and refused" when the truth is
    # "there was nothing to evaluate", and that is the never-firable shape wearing
    # a different corner.
    if k < 2:
        abstain = "no_variance: z took one value in this slice, so nothing distinguishes a head from a constant"
    elif smallest < MIN_CLASS_SUPPORT:
        abstain = f"thin_minority: rarest class has {smallest} samples, below {MIN_CLASS_SUPPORT}"
    else:
        abstain = None
    evaluable = abstain is None
    return {
        "metric": "held-out z balanced accuracy",
        "n": len(labels),
        "class_census": census,
        "k_classes_with_support": k,
        "smallest_class_support": smallest,
        "floor": 0.0,
        "floor_is_uninformative": True,
        "null": round(null, 6),
        "threshold": round(threshold, 6),
        "measured": round(measured, 6),
        "evaluable": evaluable,
        "abstain_reason": abstain,
        "clears": bool(evaluable and measured >= threshold),
    }


def consider_switch(
    state: ValueHeadState,
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    iteration: int,
) -> tuple[ValueHeadState, dict]:
    """Evaluate the criterion and ratchet if it clears. Returns (state, event).

    **Ratcheting is the point.** A head that flickers between trusted and not
    would make two consecutive iterations incomparable, and the corpus each
    produces would be denominated in different searches.
    """
    result = switch_criterion(labels, predictions)
    event = {"iteration": iteration, "already_live": state.live, **result}
    if not result["evaluable"]:
        # Abstaining is not failing. "Not evaluable yet" and "evaluated and
        # refused" are different states, and only one of them is evidence.
        event["fired"] = False
        event["abstained"] = True
        return state, event
    event["abstained"] = False
    if state.live or not result["clears"]:
        event["fired"] = False
        return state, event
    event["fired"] = True
    return (
        ValueHeadState(
            live=True,
            switched_at_iteration=iteration,
            switched_accuracy=result["measured"],
            switched_threshold=result["threshold"],
        ),
        event,
    )
