#!/bin/bash
# SWEEP-m2 on a pod: the anchor first, then all 26 arms, PAR at a time, one
# process per arm, then the collector. Each arm writes its own JSON as it ends,
# so the box can pull partial results while the rest run.
#
#   bash cloud/run_sweep_m2.sh PAR
set -uo pipefail
PAR=${1:?parallel arms}
cd /root/reckoner
P=.venv/bin/python
mkdir -p runs/sweep_m2/logs
$P scripts/sweep_m2.py --arm anchor > runs/sweep_m2/logs/anchor.log 2>&1 || exit 1
N=$($P scripts/sweep_m2.py --list | wc -l)
for target in visits improved; do
    for ((i = 0; i < N; i++)); do echo "$target $i"; done
done | xargs -P "$PAR" -n 2 sh -c '.venv/bin/python scripts/sweep_m2.py --target "$0" --arm "$1" > "runs/sweep_m2/logs/$0-$1.log" 2>&1 || echo "FAILED $0 $1" >> runs/sweep_m2/logs/FAILURES'
$P scripts/sweep_m2.py --collect > runs/sweep_m2/logs/collect.log 2>&1
