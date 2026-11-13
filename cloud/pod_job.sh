#!/bin/bash
# Guard 1 of tinyalphazero.cloud ("job"), run on the pod (copied unchanged from the chess repo's cloud/pod_job.sh):
#
#   nohup setsid bash pod_job.sh WORKDIR PULL_WAIT_S -- COMMAND... > /dev/null 2>&1 &
#
# Runs COMMAND with its output in WORKDIR/job.log, then marks WORKDIR/DONE and waits up
# to PULL_WAIT_S seconds for this box to pull the results and touch WORKDIR/PULLED
# (the pod cannot push: the box is behind NAT). Then the pod deletes itself, pulled or
# not -- the wait is bounded so a dead puller cannot keep a pod billing. The deadline
# watchdog in the pod's start command remains the absolute cap either way.
set -u
WORK=$1
WAIT=$2
shift 2
[ "${1:-}" = "--" ] && shift
mkdir -p "$WORK"

# An SSH session does not inherit the container's environment -- the image's PIP_*
# settings, its CUDA PATH, RunPod's injected RUNPOD_* -- so adopt pid 1's.
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/1/environ
POD_ID=$RUNPOD_POD_ID
POD_KEY=$RUNPOD_API_KEY

log() { echo "$(date -Is) $*" >> "$WORK/lifecycle.log"; }

delete_self() {
    for _ in 1 2 3 4 5 6; do
        curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" \
            -H "Authorization: Bearer $POD_KEY" -A "taz-cloud/1.0 (+tinyalphazero)" \
            >> "$WORK/lifecycle.log" 2>&1 && return 0
        RUNPOD_API_KEY=$POD_KEY runpodctl remove pod "$POD_ID" >> "$WORK/lifecycle.log" 2>&1 && return 0
        sleep 20
    done
    return 1
}

log "job start: $*"
"$@" > "$WORK/job.log" 2>&1
echo $? > "$WORK/job.exit"
touch "$WORK/DONE"
log "job exit $(cat "$WORK/job.exit"); waiting up to ${WAIT}s for PULLED"

end=$(($(date +%s) + WAIT))
while [ ! -e "$WORK/PULLED" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 10; done
log "pulled=$([ -e "$WORK/PULLED" ] && echo yes || echo NO); deleting pod $POD_ID"
delete_self || log "SELF-DELETE FAILED; the deadline watchdog and the sweeper remain"
