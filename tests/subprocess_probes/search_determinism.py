"""Cross-process determinism probe. Run as a FILE, never as an embedded string.

This lived as a quoted `-c` program inside two callers until a blind
`str.replace` matched *inside the literal* and broke both — twice in one session.
Documentation warns, methods help, **helpers prevent**: a program in its own file
has nothing for a text edit to corrupt silently, and it lints, imports and reads
like the code it is.

Prints one JSON line: the visit vector, the chosen action, sims used, and nodes.
The caller varies PYTHONHASHSEED and asserts every line is identical.
"""

import json
import random

from reckoner.config import Config, with_m1_search
from reckoner.episode import Problem
from reckoner.expr import add, eq, mul, num, var
from reckoner.search import search, uniform_stub
from reckoner.vocab import GOAL_SOLVE, VAR_X

if __name__ == "__main__":
    # The stub reports no steps-to-go, so it plays the search that reads 'wdl'.
    cfg = with_m1_search(Config())
    x = var(VAR_X)
    problem = Problem(
        goal=GOAL_SOLVE,
        target=VAR_X,
        par=3,
        par_source="bfs",
        expr=eq(add(mul(num(3), x), num(6)), num(21)),
    )
    result = search(problem, problem.expr, uniform_stub(cfg), cfg, random.Random(5), sims=48, m=5)
    print(
        json.dumps(
            [result.visits.tolist(), result.chosen, result.stats.sims_used, result.stats.nodes]
        )
    )
