#!/bin/bash
# WEAK-w1 on a CPU pod (protocol: WEAK-w1.md, registered and approved before it ran).
#
#   bash cloud/run_weak_w1.sh PAR        # PAR loop runs at a time, 8 threads each
#
# 1. the weak starting network (Phase 1 on problems of depth <= 3, CPU), registered
#    by digest, with the Phase-1 reference read alongside;
# 2. the weak start read under each arm's search, and the nine loop runs
#    (arms A, B, V x seeds 42, 43, 44) through campaign.run;
# 3. each run's final checkpoint read on the six suites; then the collector.
# Every stage writes as it goes, so the box can pull partial results.
set -uo pipefail
PAR=${1:?parallel runs}
cd /root/reckoner
P=.venv/bin/python
L=runs/w1/logs
mkdir -p $L
log() { echo "$(date -Is) $*" >> $L/run.log; }

log "stage 1: weak anchor and reference"
$P scripts/train_phase1.py --name phase1_d3 --max-depth 3 --steps 3000 --seed 0 > $L/anchor.log 2>&1 &
anchor=$!
$P scripts/weak_w1.py reference > $L/reference.log 2>&1 &
wait $anchor || { log "FAILED anchor training"; exit 1; }
$P scripts/anchor_phase1.py --run runs/phase1_d3 > $L/anchor_register.log 2>&1 || { log "FAILED anchor registration"; exit 1; }
log "anchor $(sha256sum runs/phase1_d3/phase1.pt | cut -c1-16)"

log "stage 2: baselines and runs"
for arm in A B V; do
    $P scripts/weak_w1.py baseline --arm $arm > $L/baseline-$arm.log 2>&1 || log "FAILED baseline $arm" &
done
for seed in 42 43 44; do for arm in A B V; do echo "$arm $seed"; done; done \
    | xargs -P "$PAR" -n 2 sh -c '.venv/bin/python scripts/weak_w1.py run --arm "$0" --seed "$1" > "runs/w1/logs/run-$0-$1.log" 2>&1 || echo "$(date -Is) FAILED run $0 $1" >> runs/w1/logs/run.log'
wait

log "stage 3: readings"
for seed in 42 43 44; do for arm in A B V; do echo "$arm $seed"; done; done \
    | xargs -P 9 -n 2 sh -c '.venv/bin/python scripts/weak_w1.py read --arm "$0" --seed "$1" > "runs/w1/logs/read-$0-$1.log" 2>&1 || echo "$(date -Is) FAILED read $0 $1" >> runs/w1/logs/run.log'
$P scripts/weak_w1.py collect > $L/collect.log 2>&1
log "done (collect exit $?)"
