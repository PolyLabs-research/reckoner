#!/bin/bash
# EXPLORATION (AGENTS §7.1), on a pod: the weak-start question of
# REVIEW-2026-11-15 step 1. Nothing this writes is evidence.
#
#   bash cloud/run_explore_weak.sh DEVICE      # cuda on a GPU pod, cpu otherwise
#
# 1. exact optimal action sets for 100 start states per suite depth (CPU workers)
#    and two weak anchors, supervised on problems of depth <= 2 and <= 3;
# 2. a reference reading of the full Phase-1 anchor (no training);
# 3. the loop from the depth<=3 anchor under arms A, B, C, D, and from the
#    depth<=2 anchor under B and D, ten iterations each, side by side.
set -uo pipefail
DEVICE=${1:-cpu}
cd /root/reckoner
P=.venv/bin/python
X=scripts/explore_weak_start.py
E=runs/explore/weak
mkdir -p $E/logs
CORES=$($P -c 'import os; print(len(os.sched_getaffinity(0)))')
WORKERS=$((CORES > 12 ? CORES - 6 : CORES / 2))
THREADS=$([ "$DEVICE" = cpu ] && echo 4 || echo 2)
echo "$(date -Is) device $DEVICE, $CORES cores, $WORKERS BFS workers, $THREADS threads per process" > $E/logs/run.log

$P $X optimal-sets --per-suite 100 --workers "$WORKERS" --out $E/optimal_sets.json > $E/logs/optimal_sets.log 2>&1 &
for d in 2 3; do
    $P $X train-anchor --max-depth $d --steps 3000 --device "$DEVICE" --threads "$THREADS" --out $E/anchor-d$d.pt > $E/logs/anchor-d$d.log 2>&1 &
done
wait
echo "$(date -Is) anchors and optimal sets done" >> $E/logs/run.log
for f in $E/optimal_sets.json $E/anchor-d2.pt $E/anchor-d3.pt; do
    [ -s "$f" ] || { echo "$(date -Is) MISSING $f" >> $E/logs/run.log; exit 1; }
done

loop() { # NAME ANCHOR ARM ITERATIONS
    $P $X loop --anchor "$2" --arm "$3" --iterations "$4" --device "$DEVICE" --threads "$THREADS" \
        --optimal-sets $E/optimal_sets.json --out-dir "$E/$1" > "$E/logs/$1.log" 2>&1 \
        || echo "$(date -Is) FAILED $1" >> $E/logs/run.log
}
loop full-ref runs/phase1/phase1.pt A 0 &
for arm in A B C D; do loop "d3-$arm" $E/anchor-d3.pt "$arm" 10 & done
for arm in B D; do loop "d2-$arm" $E/anchor-d2.pt "$arm" 10 & done
wait
$P $X summary --root $E > $E/summary.txt 2>&1
echo "$(date -Is) done" >> $E/logs/run.log
