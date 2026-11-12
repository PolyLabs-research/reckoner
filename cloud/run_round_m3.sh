#!/bin/bash
# SWEEP-m2 §6's follow-up seeds and SWEEP-m3, on one pod: both anchors first,
# then the 3 follow-up arms and the 26 SWEEP-m3 arms, PAR at a time, one
# process per arm, then SWEEP-m3's collector. SWEEP-m2's table is collected on
# the box, where its original 26 arm records live.
#
#   bash cloud/run_round_m3.sh PAR
set -uo pipefail
PAR=${1:?parallel arms}
cd /root/reckoner
P=.venv/bin/python
mkdir -p runs/sweep_m2/logs runs/sweep_m3/logs
$P scripts/sweep_m2.py --arm anchor > runs/sweep_m2/logs/anchor.log 2>&1 || exit 1
$P scripts/sweep_m3.py --arm anchor > runs/sweep_m3/logs/anchor.log 2>&1 || exit 1
N=$($P scripts/sweep_m3.py --list | wc -l)
{
    for i in 0 1 2; do echo "m2 $i"; done
    for ((i = 0; i < N; i++)); do echo "m3 $i"; done
} | xargs -P "$PAR" -n 2 sh -c '
    if [ "$0" = m2 ]; then
        .venv/bin/python scripts/sweep_m2.py --followup "$1" > "runs/sweep_m2/logs/followup-$1.log" 2>&1
    else
        .venv/bin/python scripts/sweep_m3.py --arm "$1" > "runs/sweep_m3/logs/arm-$1.log" 2>&1
    fi || echo "FAILED $0 $1" >> runs/FAILURES'
$P scripts/sweep_m3.py --collect > runs/sweep_m3/logs/collect.log 2>&1
