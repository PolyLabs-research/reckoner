# AGENTS.md — reckoner: law, machine, and build conventions

Drop-in briefing for coding agents working on this repository. Read fully before
writing code, installing packages, or running anything GPU-related. When these
conventions conflict with an agent's defaults, these win.

Companion documents: `experiment2_math_base625_spec.md` (what is being built and
why) and `experiment2_agent_plan.md` (the chunk plan, with DONE-WHEN gates).

---

## 0. Countersign delegation

The principal (Tom) has delegated countersign authority for house-law additions
to the reviewer. Laws below marked **countersigned-under-delegation** carry that
authority and are in force. The delegation covers *recording* a law that the
work has already earned; it does not cover the manual gates themselves, which
remain the principal's and are recorded in `GATE-*.md`.

## 1. Inherited law

Reproduced **verbatim** from `experiment2_agent_plan.md`. It is in force from
chunk 0 — the point of writing it down before the first line of code is that it
is not adopted incident-by-incident, which is how it was learned the first time.

> **Inherited law (in force from chunk 0, not adopted incident-by-incident):**
> strict config loader with unknown-key hard errors; `.gitignore`
> MUST_REACH/MUST_IGNORE guard test; `logschema.py` as single schema definition
> with `role` fields and absence-carries-a-reason notes; `pair_scores` persisted
> from the first ladder row; PREREG files carry the amendment-policy header from
> day one; paired-difference bootstrap is the test of record for pass-vs-pass
> comparisons; every detector that gates automation is validated on both
> polarities before live use; instrument the trigger, never trust identical
> numbers; external processes are context managers; one shared identity
> normalizer for any dedup key; `git -C` in multi-command shells; waiters
> reference PIDs, never patterns; provenance (`git_sha` via
> `git -C <repo_root>`, config fingerprint) in every checkpoint and dataset
> `meta.json`.

And the execution law, also verbatim:

> Execute chunks in order, one chunk per session unless stated. Every chunk ends
> with its DONE-WHEN gates green, committed, **pushed**, and reported with
> verbatim numbers and a tree-state block (merged? pushed? test count?). If a
> gate can't pass after 3 distinct attempts: stop, write
> `BLOCKED-<date>-<topic>.md` (cause, attempts, options), commit it, halt.
> **Never weaken a gate to pass it.**

Where each clause lives in this repo, as of chunk 0:

| Clause | Enforced by |
|---|---|
| Strict config loader, unknown-key hard errors | `src/reckoner/config.py::_assert_known_keys`, `tests/test_config.py` |
| `.gitignore` MUST_REACH / MUST_IGNORE guard | `tests/test_gitignore_musttrack.py` |
| Detectors validated on both polarities | Every guard in `tests/` has its accepting case beside its rejecting one, including a self-check that the gitignore probe itself can return both answers |
| `git -C` in multi-command shells | `tests/test_gitignore_musttrack.py::_check_ignored` |
| Provenance in every checkpoint / `meta.json` | `config_fingerprint()` exists from chunk 0; the writers arrive in chunks 5–6 |
| `logschema.py`, `pair_scores`, PREREG headers, paired bootstrap | **Landed.** `src/reckoner/logschema.py` (chunk 9); `src/reckoner/ladderpass.py::run_pass` writes `pair_scores` from row one; `src/reckoner/ladder.py::paired_bootstrap` is the test of record; `PREREG-chunk9-part0.md`, `PREREG-chunk9-shakedown.md`, `PREREG-chunk10-smoke.md` carry the amendment header. The `.gitignore` negations predated all of them, which is the point of writing them first |
| Every no-regress floor names its KIND | Countersigned 2026-08-15 (`PREREG-chunk11-part0bc.md` P11B-A2). A **noise-band floor is an indistinguishability gate**: holding 1188/1200 means *not distinguishably worse than the anchor*, never *at least as good on every problem* and never *at least as good on average*. Three different gates, three different licensed sentences, so the kind is named where the floor is declared |
| Paired sets are instruments, not data | `src/reckoner/pairedset.py::freeze` (writes once, anchors, asserts its own trackedness), `::load` (re-digests on the read path), `dataset.SOURCE_ROLES["runs/paired"] = "instrument"` so the runtime guard refuses one as a training source |

## 2. Decisions already made (do not silently revisit)

From the plan's §8 block and amendment v1.1. Several are enforced by
`config.py::validate()` rather than left to memory:

1. **Value head = 3-class W/D/L vs par + steps-to-solve regression.** Amendment
   v1.1 reversed the original 2-class solved/timed-out decision. `validate()`
   rejects `model.value_classes != 3`.
2. **Rule set v1 is the minimal closed set:** `eval_add, eval_sub, eval_mul,
   combine_like_terms, add_both_sides, sub_both_sides, div_both_sides`, exact
   integer division only. No fractions, no distribution — extensions are later
   one-lever rounds.
3. **Goal representation = prefix tokens in the state** (`[GOAL_SOLVE, VAR, x,
   SEP, …]`). No architecture change.
4. **Phase-3 generator is a standalone model**, not weight-shared. Registered as
   the funnel treatment; not built in this plan.
5. **Single-agent search: there is no opponent, so backup does NOT negate per
   ply.** This is the sharpest porting hazard in the project. It is a config key
   (`search.perspective: single`) and `validate()` rejects anything else.
6. **The par opponent enters the label, never the search.** z ∈ {+1 under par, 0
   equal, −1 over par or step cap}.
7. **sympy is a ladder rung, never par** — unless a future chunk compiles its
   derivations into our rule vocabulary.

## 3. The machine

| Item | Value | Verified |
|------|-------|----------|
| Box | GMKtec NucBox EVO-X2 (mini PC) | — |
| APU | AMD Ryzen AI Max+ 395 "Strix Halo", 16C/32T | `lscpu` |
| GPU | Integrated Radeon 8060S — **gfx1151**, RDNA 3.5 | `rocm-smi` |
| VRAM | **64 GiB** dedicated (BIOS UMA carveout) | `mem_info_vram_total` → 68719476736 B, 2026-08-16 |
| Host RAM | **~62 GiB** visible to the OS; 8 GiB swap | `free -h`, 2026-08-16 |
| OS | Ubuntu 24.04.4 LTS, ROCm 7.1.0 userspace | `/opt/rocm/.info/version` |
| GPU stack | **ROCm — there is no NVIDIA hardware here. CUDA wheels will never work.** | — |

Implications you must internalize:

- PyTorch on ROCm masquerades as CUDA: `torch.cuda.is_available()`,
  `device="cuda"`, `torch.cuda.get_device_name()` are the correct API calls.
  Do not write ROCm-specific device strings.
- `nvidia-smi` does not exist. Use `rocminfo | grep gfx` (expect `gfx1151`) and
  `rocm-smi` for utilization.
- **The two memory pools have opposite pressure profiles, and the split was
  rebalanced on 2026-08-16.** GPU-side: 64 GiB, still enormous for this class of
  hardware, and a running local-LLM server can be holding most of it — check
  `rocm-smi` before assuming it is free. Host-side: ~62 GiB. Size replay buffers,
  dataset loads and worker pools against ~45 GiB of realistically free host
  memory, not against "128GB". Prefer streaming/memmap over loading datasets into
  host RAM; swap is 8 GiB, so an OOM-adjacent allocation still thrashes or dies
  rather than degrading gracefully.

  **Why it moved, because the reason outranks the number.** The carveout was
  96 GiB, leaving ~30 GiB host. On 2026-08-15 a CPU-only sweep drove host memory
  to 30.7 GiB with all 8 GiB of swap consumed; the reclaim thrash pushed package
  power from a 62 W baseline to **88 W** and Tctl from 96 °C to **101.2 °C**, and
  the box hard powered off — rails cut, no panic, no OOM in the log, nothing.
  The OOM killer had taken the editor and a browser process minutes earlier. The
  same sweep at 64/62 peaked at 14.8 GiB with **swap never touched**, ran
  143 minutes continuously, and recorded **zero** samples above 80 W. The steady
  97 °C clamp was survivable throughout; what was not survivable was the power
  excursion that swap thrash produced. **On this box a host-memory shortfall is a
  hardware-stability problem, not just an allocation problem** — it arrives as a
  power transient through the shared LPDDR5X controller. Size against host RAM
  accordingly, and do not raise the carveout back without a reason that outweighs
  that.
- gfx1151 support is recent (ROCm 7.1+). If GPU code misbehaves, suspect the
  wheel/stack version before the code. History on this box: the `amdgpu` driver
  was once blacklisted and silently forced CPU-only operation — if the GPU
  vanishes, check `lsmod | grep amdgpu` and `/etc/modprobe.d/` first.

### 3.1 Telemetry is standing equipment for campaign runs

**Any run expected to exceed ~30 minutes runs with the flight recorder on.** Not
optional, and not something to switch on after a failure — after a failure is
exactly when it cannot be switched on.

`pcwatch` (`~/.local/bin/pcwatch`, systemd user service `pcwatch.service`) samples
package power, Tctl, every temperature sensor, clocks, load, host memory, swap and
page-churn at **1 Hz**, and **fsyncs every row before taking the next sample**.
That last property is the whole point: on 2026-08-15 the box hard powered off and
journald lost its final minute to the page cache, leaving a crash with no
evidence. The recorder's log was intact to the second — 8,620 samples turned an
unexplained power-off into a one-line diagnosis, and the sizing rule in §3 above
is downstream of it.

`pcwatch-serve.service` publishes a read-only dashboard on `:18795`, reachable
over LAN or Tailscale, for watching a long run from elsewhere.

Both survive reboot. After any unexplained restart the **crashed** run is index 1,
because the recorder starts a fresh log on boot:

```
pcwatch --summary 1
```

Read it as: pinned near the clamp at the cut ⇒ thermal; moderate temperature with
a power spike in the final samples ⇒ delivery side. A campaign run that ends
without telemetry is a run whose failure mode cannot be diagnosed, and on this box
that has already cost a measurement.

## 4. PyTorch installation — the #1 recurring failure

A bare `pip install torch` pulls CUDA wheels from PyPI, which import fine and
then run CPU-only forever. This has burned real time more than once.

1. **This project is CPU-first through chunk 6.** The domain is CPU-heavy
   (pattern-matching movegen), GPU-light, and the model is 2–7M parameters.
   `pyproject.toml` pins the **CPU** index explicitly, so a bare `uv sync` can
   never resolve to CUDA:

   ```toml
   [tool.uv.sources]
   torch = { index = "pytorch-cpu" }

   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true
   ```

2. **ROCm variant (optional until chunk 7).** Change that URL to
   `https://download.pytorch.org/whl/rocm7.2` (any index ≥ rocm7.1 works for
   gfx1151; AMD's own `https://repo.amd.com/rocm/whl/gfx1151/` is the fallback)
   and rebuild the venv from scratch — `rm -rf .venv && make install`. Do not
   perform surgery on an existing venv.

3. **Verify after any environment change:** `make env` runs
   `scripts/check_env.py`, which prints the torch build, the device, and a
   matmul throughput number, and **exits non-zero on a CUDA wheel**.

4. No `HSA_OVERRIDE_GFX_VERSION` hacks. On ROCm 7.x gfx1151 is natively
   supported; needing the override means an old wheel snuck in.

5. GPU numerics, when the GPU is in play: bf16 autocast for training/inference,
   fp32 master weights. `torch.compile` is opt-in behind a flag, default off,
   and only after benchmarking eager vs compiled on this box.

## 5. Python tooling

- **`uv` for everything**: `uv venv`, `uv pip`, `uv lock`, `uv run`. Not conda,
  not poetry, not bare pip. Python 3.12.
- Projects are installable packages: `src/` layout, `pyproject.toml`, installed
  with `make install` (`uv sync --frozen --extra dev`). **`sys.path` mutation is
  banned** — if an import needs a path hack, the packaging is wrong; fix the
  packaging.
- **`uv.lock` is committed and installs are frozen.** Without it, a clean-clone
  gate tests today's resolution luck rather than reproducibility: "green in a
  clean clone" would mean "green against a different environment than the one
  the numbers were measured on". `make lint` runs `uv lock --check` so the lock
  and `pyproject.toml` cannot drift apart unnoticed; `make relock` is how
  dependencies change on purpose.
- Lint/format with `ruff` before every commit (`make lint`). Type-hint all
  public interfaces.
- Tests: `pytest`, with `hypothesis` for property-based testing of codecs,
  invariants and round-trips (chunk 1's parser/printer is the first customer).
  **All tests must pass on CPU only** — GPU availability is never a test
  dependency. GPU-dependent checks live in benchmark scripts.
- **Round-trip gates are blind to symmetric bugs; every codec carries pinned
  absolute reference vectors.** *(countersigned-under-delegation)* A codec whose encoder and decoder are wrong in
  the same direction round-trips perfectly forever. Measured, not argued: in
  chunk 1, flipping base-625 digits to LSB-first in *both* `to_digits` and
  `from_digits` survived all 200,000 round-trips and died only on the pinned
  table (`625 → [1, 0]`). A round-trip proves the two halves agree; only an
  absolute vector proves they agree with reality.
- **A gate must report what it covered, not only that it passed.** *(countersigned-under-delegation)*
- **One formatter of states, ever — a caption describes, or it calls
  `render_expr()`; there is no third path.** Hand-formatting a state outside the
  renderer is the renderer bug the renderer exists to prevent, recommitted one
  line above the fold, and no round-trip catches it because captions are prose.
  *(countersigned-under-delegation)* Publish the
  coverage distribution beside the result — a 200K-iteration loop concentrated
  at two depths is a narrower gate than its iteration count advertises, and the
  count alone will never say so.
- **A gate suite assembled from known hazards has a blind spot exactly the shape
  of the component doing its job at all.** Measured: chunk 7's search expanded
  only the root's children — 48 simulations, 2 nodes — and all four of its gates
  passed, each for a reason unrelated to the defect (`FINDINGS.md` F-06). Every
  gate asked how the search could be subtly wrong; none asked whether it
  searched.
  **Operational rider (a): the first gate written for any component measures it
  doing its central job.** For a search, that it searches. For a generator, that
  it generates the distribution it claims. For a checker, that it accepts a true
  answer and rejects a false one. The known-hazard gates come second, forever.
  **Operational rider (b): a number nobody asserts on is not a gate; it is a
  comment that happens to be computed.** Measured in the same incident:
  `SearchStats.nodes` existed in chunk 7, was populated correctly, was written
  into every stats row, and read 6, 6, 6, 6 across an 8× budget increase. The
  instrument was not missing — it was mute. Adding a field is not adding a gate.
  **Corollary (2026-08-15, countersigned): a bound that always fires is not a gate
  either.** Both are comments — one nobody reads, one nobody can act on. A
  detector that has triggered on every run since it was written has stopped
  carrying information, and the fix is to recalibrate it against the measured
  baseline rather than to keep recording the alarm. Measured: chunk 9's tier-2
  logit bound was declared at 1e-3 from an estimate and measured at 3.721e-3 on
  every width bucket (`FINDINGS.md` F-12); it was refused a widening under freeze,
  then recalibrated by ruling to 2× the observed maximum with the measurement as
  its derivation. **A detector's reference is the measured baseline, never an
  estimate the measurement has already falsified.**
  **Operational rider (c): a threshold nobody computed the floor of is not a gate
  either.** A gate has two failure modes — *unreachable* and *unmissable* — and
  only the first announces itself by failing; the second ships green. Measured:
  chunk 8's registered `top-8 ≥ 0.90 on depth ≤ 3` had a floor of 0.9897, because
  top-k ranks over the legal set and 98.97% of those states have ≤ 8 legal
  actions. A randomly initialised network scored 1.0000 (`FINDINGS.md` F-10).
  **Executable form: every gate declares a four-tuple — floor, null-model
  baseline, threshold, measured value — at declaration, not afterwards.** Chunk 7
  institutionalised this arithmetic for search budgets; it generalises to every
  metric this project will ever gate on. The null is a *run*, not an estimate:
  the uniform-prior stub at the same budget on the same problems.
  **Precedent under rider (c): a null computed with shared randomness is not a
  null.** All harness randomness is a per-problem (and per-step) derived seed
  fan-out, never one stream and never a constant re-seed. Measured: passing a
  fresh `random.Random(0)` to every search made the root Gumbel draw a function
  of the action *count*, so every 5-action problem chose the same slot — which
  silently replaced "uniform-random action" with "always the first legal action",
  a **stronger** null that made a discriminating gate read as vacuous
  (`FINDINGS.md` F-11). Reproducibility belongs in the seed *schedule*, recorded;
  seeding every draw identically is not reproducibility, it is a different
  policy. *(countersigned-under-delegation)*
- **A provenance field whose default is its strongest claim is not a provenance
  field — the most-trusted value must be the one that costs something to say.** *(countersigned-under-delegation)*
  "Fields carry their epistemic status" guards the *read* path; a default is a
  write nobody performs, and nothing guarded that. Measured: `Problem.par_source`
  defaulted to `"bfs"`, and fifty derivations shipped claiming BFS-exact
  provenance for hand-written literals (`FINDINGS.md` F-02). This generalises to
  every `*_source`, `*_asof` and verification flag this project will carry: **the
  trusted value is computed, or it is absent.** Defaults name the weakest honest
  state, never the strongest.
- The `Makefile` carries at least `make lint` and `make test`; `make bench`
  arrives with the first benchmark script (chunk 7).

## 6. Project conventions

- **One config system**: the nested dataclass tree in `src/reckoner/config.py`,
  loadable from YAML, with `config_fingerprint()` (sha256 of canonical YAML) and
  **unknown keys as a hard error**. CLI scripts are thin argparse wrappers that
  build a config and call library code. No config value duplicated between CLI
  defaults and code — `configs/default.yaml` must equal the dataclass defaults,
  and a test pins that identity. A campaign that needs different values gets its
  own file in `configs/`.
- **One lever per round is an executable property, not a reviewed one.**
  *(**Inherited, not earned** — principal's registry ruling, 2026-08-15. It
  predates the earned registry and is therefore neither a countersign nor a
  ruling: the earned lines are five, the inherited corpus is separate.
  **Reported discrepancy, not silently merged:** the phrase is not in §1's
  verbatim inherited-law block. Its pre-registry sources are
  `experiment2_math_base625_spec.md` §2 — "extend by round, one lever" — and
  `experiment2_agent_plan.md` chunk 12 — "grid-bit vs token embedding round
  (st2's question, one lever)". The principle predates the registry as ruled;
  the §1 block is not where it is written. A reader searching §1 for it should
  find this note rather than conclude the registry lies.)* Because
  `default.yaml` equals the defaults exactly, a campaign config's
  `config_diff(Config(), load_config("configs/<campaign>.yaml"))` **is** its
  lever list. Every campaign config ships with a
  `test_<campaign>_config_is_the_brief` asserting that diff equals the set its
  PREREG registered. A second lever added quietly mid-run then fails the build,
  and fails it by name. Registered here before the first campaign exists, so it
  is adopted by design rather than after an incident.
- Every config field is tagged with its provenance — `[spec §N]`,
  `[plan chunk N]`, `[v1.1]`, or `[provisional — chunk N]`. **A `[provisional]`
  number is not a decision**; the chunk that owns it sets it and pins it.
- **Flipping a `[provisional — chunk N]` tag to a decided one, with its source,
  is part of chunk N's DONE-WHEN.** The tags exist to tell a reader which
  numbers are load-bearing; a tag left stale after its chunk shipped tells them
  the opposite of the truth. The transition rule is what makes the tags guard
  the record and not just the reader.
- **Run artifacts** live in `runs/<name>/` (gitignored except the records named
  in `tests/test_gitignore_musttrack.py`). Every run directory gets the resolved
  `config.yaml`, the git SHA and the `check_env.py` output.
- **Checkpoints embed their provenance**: model state, optimizer state, full
  config dict, git SHA, step count. A checkpoint that cannot reproduce its own
  configuration is a bug.
- **Logging is JSONL**, one row per event/iteration, in the run directory.
  Dashboards and analysis are pure readers with zero imports from training code.
- **Determinism**: one seed in config fans out to all RNGs; library code takes
  explicit `rng`/`Generator` parameters, never global random state.
- **One subsampler, and raw prefix-slicing of a dataset is a review flag.**
  `dataset.sample_indices(total, k, seed)` — or `.sample(k, seed)` on a `Dataset`
  or `SupervisionSet` — is the only way to take "some rows". Every artifact here
  is laid out stratum by stratum, so `range(k)` is the whole of the shallowest
  stratum and none of the rest. Three defects, not one: F-03's pilot measured a
  distribution the real run would not see, `build_phase1_data.py` shipped a
  prefix-taking `--limit`, and F-10's tie-break diagnostic sampled `range(256)`
  and got 256 single-legal-action depth-1 states, degenerating every statistic.
  The first two were fixed by writing a warning; the third happened anyway, to
  someone who had read it. **Documentation warns, helpers prevent** — when a
  hazard recurs after being documented, the fix is a mono-instance, not a
  louder comment.
- **Large data**: `numpy.memmap` binary layouts with a small `meta.json`
  sidecar, over JSON/pickle/HDF5. On this box memmap is a memory-pressure tool,
  not a convenience: multi-GB datasets must be mapped or streamed, never fully
  loaded.
- Library code raises; only `scripts/` prints and exits. **No `print()` in
  `src/`.**

## 7. Working style

- **Work in chunks with objective gates.** A chunk is done when its DONE-WHEN
  commands pass — not when the code "looks right". One commit per chunk:
  `chunk NN: <summary>`. Push in the same chunk; the four-session push gap does
  not get re-learned.
- **Never weaken a gate to pass it.** Three distinct failed attempts ⇒ stop,
  write `BLOCKED-<date>-<topic>.md` (cause, attempts, options), commit, halt. A
  relaxed acceptance criterion is a silent lie discovered weeks later; a
  BLOCKED.md is useful information.
- **Briefs and reviewer rulings commit verbatim on receipt, before acting on
  them.** *(countersigned-under-delegation; scope extended in place 2026-08-15
  after a session boundary ate an issued F-09 ruling — the third instance)*
  Governance text that lives only in a
  conversation is destructible. Measured three times: the chunk-6 gate verdict existed
  only as a spoken "PASS. continue" until `GATE-chunk4-VERDICT.md` was written
  for it; the chunk-8 brief was never committed, so a session crash took items 3
  and 5–8 permanently (`BRIEF-chunk8.md` carries the summary that survived and
  says so); and the F-09 ruling was issued, lost to a session boundary, and had
  to be reissued. The commit precedes execution, not follows it: a brief
  committed after the work is a transcript, and a transcript cannot be the thing
  the work was checked against. **Amendments to a committed brief are appended to
  it, dated, never edited into the original text.** A ruling is committed into
  the artifact it rules on — F-09's ruling lives in F-09.
- **Interfaces are contracts.** Once a chunk's public interface (signatures,
  array layouts, file formats, constants, config keys) is established, do not
  change it without being asked — later chunks depend on it.
- **Correctness before speed, and profile before optimizing.** Benchmarks append
  to `benchmarks/results.jsonl` with the git SHA. Never trade a correctness test
  for throughput; never optimize an unmeasured bottleneck (`py-spy`).
- **Bias to minimal.** Small models, few dependencies, arrays over object
  graphs, one implementation of each concept. If a component exists twice, that
  is a defect. New dependencies need a reason; "it's popular" isn't one.
- **Prefer explicit structure in explanations**: precise interfaces, tables,
  worked numeric examples and named invariants over metaphors.
- **In multi-command shells, address git per-command with `git -C <path>`** — a
  `cd` persists for the whole invocation and has already mis-pointed three merge
  commits. The failure is silent.
- **A check must reference the thing checked — a PID, a path, a digest — never a
  pattern that can match the checker.** `while pgrep -f "train --games 20"`
  matches its own shell's command line and waits forever on itself, reporting
  "still running" for work that never started. It does not merely lose progress,
  it forges it.
- Ask before: deleting data, force-pushing, changing an established interface,
  adding a heavyweight dependency, or touching system config (drivers, kernel
  modules, `/etc`). Everything else: proceed and report.

### 7.1 The exploration lane

*(Ruled by the principal, 2026-11-15. Proposed in `REVIEW-2026-11-15.md` as "An
exploration lane. I'd allow cheap unregistered runs to find effects, with
registered runs to confirm them"; ruled, verbatim: "agree on your proposals and
order".)*

Until this ruling every question cost a registered page and usually a pod run,
so the project measured carefully and rarely. Two lanes now, and the line
between them is what keeps the second one honest.

- **What an exploration is.** A run made to find out whether an effect exists
  and roughly how big it is, before a page is written for it. It needs no
  registered page and no approval, **provided it costs nothing but this box's
  time.** Anything that spends money is approved first, as before.
- **Where it lives.** Scripts are `scripts/explore_*.py` and are committed, so
  an exploration can be re-run. Outputs go under `runs/explore/`, which git
  ignores, and are never registered in ANCHORS.
- **What it is not.** An exploration reading is never evidence.
  - It does not enter `FINDINGS.md`.
  - It does not move a default, a fingerprint, a gate, a band or a pin.
  - A registered page may mention it only in a section headed "Exploration (not
    evidence)", which names the registered round that will test it.
- **How it ends.** Either nothing was found, and the session report says so; or
  a registered round is drafted with the prediction the exploration suggested.
  **The registered round measures again from scratch.** It never inherits an
  exploration's numbers, because the exploration chose what to look at after
  seeing data, which is exactly what registration exists to prevent.
- **What does not relax.** Frozen instruments stay frozen: an exploration reads
  suites and paired sets and never writes, retunes or trains on them.
  `training_problems` refuses an instrument as a training source in both lanes.
  The machine rules in §3 apply in both lanes too: this box's memory and
  thermal limits do not know which lane a run is in.

## 8. Domain defaults

- Single-box, single-GPU. No distributed frameworks, no cloud assumptions.
- Throughput pattern: many CPU-side actors/workers feeding **one** process that
  owns the GPU with large batched forwards. Per-worker model copies fighting
  over the GPU is an anti-pattern here.
- The GPU sitting idle while CPU-bound Python saturates is the *expected*
  profile for small-model search workloads — confirm with profiling before
  "fixing" it. On this project it is the baseline assumption, not a surprise.
- Long runs must be resumable: killing the process at any point and restarting
  from the run directory must not corrupt state.

## 9. Quick reference

```bash
make install          # uv sync --frozen --extra dev (exactly what uv.lock pins)
make relock           # change dependencies on purpose
make lint test        # the chunk 0 gate (lint includes `uv lock --check`)
make env              # torch build / device / matmul TFLOPS; non-zero on a CUDA wheel

rocminfo | grep gfx   # expect gfx1151
rocm-smi              # GPU util / 64 GiB VRAM pool
free -h               # host RAM (~62 GiB — still the scarce one)

pcwatch --summary     # thermal/power flight recorder: peaks + the final samples
pcwatch --summary 1   # after an unexplained reboot, the run that DIED is [1]
pcwatch --list        # recorded runs, newest first

git log --branches --tags --not --remotes   # MUST be empty: nothing unpushed
```
