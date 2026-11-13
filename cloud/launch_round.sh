#!/bin/bash
# One step from launch to results on the box, so nothing can come between the
# job starting and the puller starting (SWEEP-m3, attempt 1's loss):
#
#   bash cloud/launch_round.sh TAG HOURS RUNNER OUTDIR [PAR]
#
# Launches a cpu5c / 32 vCPU pod (protocell's cpu_pod.py: $1.25/h ceiling, HOURS
# hard cap), runs the speed gate (terminates a host more than 2x slower than the
# box on a plain-Python loop), stages this checkout's branch and the data archive
# beside it (DATA, default ../stage.tar.zst), starts RUNNER under the job guard
# with its pull wait running to the deadline, then pulls runs/sweep_m* every 5
# minutes until the job is DONE, pulls the rest, and releases the pod.
set -uo pipefail
TAG=${1:?tag}; HOURS=${2:?hours}; RUNNER=${3:?runner}; OUT=${4:?outdir}; PAR=${5:-4}
DATA=${DATA:?set DATA to the data archive}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RP=("$HOME/Documents/dev/chess/.venv/bin/python" "$HOME/Documents/dev/chess/scripts/runpod_pod.py")
CPU=("$HOME/Documents/dev/protocell/.venv/bin/python" "$HOME/Documents/dev/protocell/cloud/cpu_pod.py")
GATE_S=0.48
log() { echo "$(date -Is) $*"; }
mkdir -p "$OUT"
cd "$REPO"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { log "refusing: uncommitted changes"; exit 1; }
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BUNDLE="$OUT/reckoner.bundle"
git bundle create "$BUNDLE" "$BRANCH" 2> /dev/null

line=$("${CPU[@]}" launch --tag "$TAG" --flavor cpu5c --vcpu 32 --hours "$HOURS" --max-rate 1.25 --yes 2>&1 | tee -a "$OUT/launch.log" | grep '^launched')
POD=$(echo "$line" | awk '{print $2}')
DEADLINE=$(echo "$line" | grep -o -- '-dl[0-9]*' | tr -dc 0-9)
[ -n "$POD" ] && [ -n "$DEADLINE" ] || { log "launch failed"; cat "$OUT/launch.log"; exit 1; }
log "launched $POD, deadline $(date -u -d @"$DEADLINE" +%H:%M:%S) UTC"
ssh_() { timeout "$1" "${RP[@]}" ssh "$POD" -- "${@:2}" < /dev/null; }

timeout 900 "${RP[@]}" wait-ssh "$POD" > /dev/null || { log "no ssh; terminating"; "${RP[@]}" terminate "$POD"; exit 1; }
gate=$(ssh_ 120 "python3 -u -c 'import time;t=time.perf_counter();s=0
for i in range(5_000_000): s+=i*i
print(round(time.perf_counter()-t,2))'")
log "speed gate ${gate}s (limit ${GATE_S}s)"
if [ -z "$gate" ] || awk -v g="$gate" -v l="$GATE_S" 'BEGIN{exit !(g > l)}'; then
    log "slow host; terminating"; "${RP[@]}" terminate "$POD"; exit 2
fi

ssh_ 60 mkdir -p /root/rk
"${RP[@]}" push "$POD" "$BUNDLE" /root/rk/reckoner.bundle > /dev/null
"${RP[@]}" push "$POD" "$DATA" /root/rk/data.tar.zst > /dev/null
"${RP[@]}" push "$POD" cloud/setup_pod.sh /root/rk/setup_pod.sh > /dev/null
ssh_ 1200 bash /root/rk/setup_pod.sh /root/rk/reckoner.bundle "$BRANCH" /root/rk/data.tar.zst 2>&1 | tail -5 | tee -a "$OUT/setup.log"
grep -q ' 0 dirty' "$OUT/setup.log" && grep -q 'passed' "$OUT/setup.log" || { log "setup failed; terminating"; "${RP[@]}" terminate "$POD"; exit 1; }

WAIT=$((DEADLINE - $(date +%s) - 300))
ssh_ 30 "cd /root/reckoner && nohup setsid bash cloud/pod_job.sh /root/work $WAIT -- bash $RUNNER $PAR > /dev/null 2>&1 < /dev/null & disown"
log "job started; the pod holds its results up to ${WAIT}s after the job ends, or the deadline"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    for d in sweep_m2 sweep_m3; do
        timeout 120 "${RP[@]}" pull "$POD" "/root/reckoner/runs/$d/" "$OUT/$d/" < /dev/null > /dev/null 2>&1
    done
    timeout 60 "${RP[@]}" pull "$POD" /root/work/ "$OUT/work/" < /dev/null > /dev/null 2>&1
    log "arms=$(find "$OUT"/sweep_m* -name '*.json' -path '*/[iv]*/*' 2> /dev/null | wc -l) done=$([ -e "$OUT/work/DONE" ] && echo yes || echo no)"
    if [ -e "$OUT/work/DONE" ]; then
        timeout 300 "${RP[@]}" pull "$POD" /root/reckoner/runs/ "$OUT/runs/" --exclude ring0_replay --exclude ring_v --exclude data --exclude phase1 < /dev/null > /dev/null 2>&1 \
            && ssh_ 60 touch /root/work/PULLED && log "PULLED; the pod deletes itself" && exit 0
    fi
    sleep 300
done
log "deadline reached without DONE"; exit 1
