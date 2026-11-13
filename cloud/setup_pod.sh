#!/bin/bash
# Prepare a fresh RunPod CPU pod (runpod/base:1.4.1-ubuntu2404) to run reckoner:
#
#   bash setup_pod.sh /root/rk/reckoner.bundle BRANCH /root/rk/data.tar.zst
#
# Clones the code from a git bundle pushed from the box (the pod never holds
# GitHub credentials), unpacks the gitignored data beside it, installs exactly
# what uv.lock pins with the CPU extra (AGENTS §4: ACCEL=cpu on the pod), checks
# the data against ANCHORS, and prints the stack and the CPU quota it landed on.
set -euo pipefail
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/1/environ
BUNDLE=${1:?bundle}
BRANCH=${2:?branch}
DATA=${3:?data archive}
need() { command -v "$1" > /dev/null || { DEBIAN_FRONTEND=noninteractive apt-get -qq update; DEBIAN_FRONTEND=noninteractive apt-get -qq install -y "$2" > /dev/null; }; }
need zstd zstd
need rsync rsync
need git git
cd /root
rm -rf reckoner
git clone -q -b "$BRANCH" "$BUNDLE" reckoner
cd reckoner
zstd -dc "$DATA" | tar -xf -
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
uv sync -q --frozen --extra cpu --extra dev
.venv/bin/python - << 'P'
from reckoner.dataset import anchored_data, anchored_path
for name in ("phase1_train", "phase1_eval"):
    anchored_data(name, verify=True)
anchored_path("runs/phase1/phase1.pt", verify=True)
print("data:  phase1_train, phase1_eval, phase1.pt match ANCHORS")
P
quota=$(cat /sys/fs/cgroup/cpu.max 2> /dev/null || echo "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)")
echo "code:  $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) dirty"
echo "stack: $(.venv/bin/python -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__)')"
echo "cpu:   $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)  quota '$quota'  RUNPOD_CPU_COUNT=${RUNPOD_CPU_COUNT:-?}  mem $(free -g | awk '/Mem/{print $2}')G"
echo "tests: $(.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_seed_replicates.py tests/test_f54_governance.py tests/test_config.py tests/test_anchored_path.py 2>&1 | tail -1)"
git checkout -q -- runs/testcount.json 2> /dev/null || true
