# BRIEF-m1-host.md — migrate the campaign to the RunPod instance (2026-08-16)
# The pod is the CAMPAIGN HOST. The local box remains dev + short runs.
# Development flow: build locally, push; pod pulls, verifies, runs.

## Part 0 — connect and inventory
0a. SSH in (details supplied by Tom). Record a pod-facts block committed
    to the repo: nvidia-smi (GPU, driver), torch.__version__ /
    torch.version.cuda / get_device_name(), lscpu (model, cores, clocks),
    RAM, disk layout (container vs /workspace), python version.
    Campaign numbers will cite this host; it gets the AGENTS.md treatment.
0b. Clone github.com/tripptytrip/reckoner at the pinned commit; assert
    HEAD == expected hash. Install from the lockfile. Assert sympy ==
    1.14.0 exactly — cas_version is part of the rung's identity; a
    different version is a different opponent and refuses.
0c. Transfer anything ANCHORS covers that isn't in git, then RE-DIGEST
    EVERY ANCHORS ENTRY ON THE POD. The digest check is the transfer-
    integrity gate: 0 mismatches or the sync failed. The Phase-1 anchor
    (45333caa…) must re-digest clean before anything runs.

## Licensing (nothing counts before these)
1. make lint test — full suite green on the pod, testcount.json
   whole_suite=true committed.
2. The GPU/CPU equivalence battery runs ON THE POD (CUDA is a new
   device; the gfx1151 license does not travel — L7 at the hardware
   boundary). Tier-1 + tier-2 verdicts written to runs/ and committed.
   measure_dtype=fp32 asserted via validate().
3. make golden on the pod — the loop end to end, CPU, within budget.

## Pilot (F-03: measured, not estimated)
4. One iteration at campaign config + one full ladder pass, wall-clock
   split recorded per phase. Note CPU utilization (GIL — informational,
   no parallelism levers now). These numbers feed M1-A2.

## M1-A2, then rehearsal, then campaign
5. Commit M1-A2 per the standing ruling: CampaignConfig fields,
   episodes_per_iter DERIVED on the page from pod-measured numbers
   (ring cap / consumption rate / wall-clock), old + new fingerprints,
   the exact yaml diff, analysis point = 20, no raise after launch.
6. Driver DONE-WHEN completes on the pod (golden-via-driver, both kill
   points, fingerprint refusal both polarities, 3-iteration dress
   rehearsal at campaign config — expectations committed first,
   deleted after recording).
7. Campaign launches in tmux. Pod telemetry sidecar: nvidia-smi +
   loadavg sampled to jsonl (thermals are the datacenter's problem now;
   the sidecar's job here is utilization and cost accounting).

## Operational rules
- Artifacts rsync back to the local box at every session end; commits
  and pushes happen ONLY locally. No tokens, no keys on the pod.
- STOP the pod when idle; NEVER Terminate (the volume disk dies with
  termination). Assume the pod can vanish: the volume + git + rsync'd
  artifacts must always be sufficient to resume elsewhere — that's
  what the four-step commit was built for.
- Frozen-instrument refusal, no behavioral flags, fingerprint
  assertion at startup: all travel with the code. The pod changes
  where the campaign runs, not what governs it.

## DONE-WHEN
- Pod-facts committed; ANCHORS re-digest 0 mismatches; suite + battery
  + golden green on pod; pilot table committed; M1-A2 committed.
- Then BRIEF-chunk11-driver's own DONE-WHEN, executed on this host.
Three failed attempts at any gate → BLOCKED, never a weakened gate.

---

## Comparands

- `runs/pod_facts.json` — the pod-facts block §0a requires and DONE-WHEN
  asserts. **NEVER PRODUCED, in any form** (F-45): no such file, and no page
  carries the block either. A DONE-WHEN item marked satisfied with nothing
  behind it.

*The host brief specifies a procedure, not a measurement; its only artifact is
the provenance record the procedure was supposed to produce.*

*Declared, not extracted (F-44). An artifact this page relies on belongs here;
a path-shaped string absent from this list is flagged by the sweep's lint.
Prose-only citations are uncatchable by machine — if a sentence rests on an
artifact, the artifact goes in this block.*
