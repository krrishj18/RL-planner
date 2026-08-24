#!/usr/bin/env bash
# Submit an RL-planner job to OSMO.
#   osmo/submit.sh --job sweep --tag sweep_v2 --scenes synthetic \
#       --args "--scenes synthetic:0-200 --robots 3 --updates 300 --envs 32 --eval-episodes 16" \
#       [--pool dsta] [--cpu 24] [--gpu 1] [--mem 64Gi] [--timeout 12h] [--keep-alive] [--allow-cpu] [--dry-run]
# Ships a clean snapshot of this repo (no .venv/runs/data/.git) with `--rsync`; results land on
# airlab-storage at /volume4/dsta/rl-planner/<tag>/runs/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB=train; TAG=""; ARGS="--smoke --name smoke_osmo"; SCENES=synthetic; POOL=dsta; CPU=24; GPU=1; MEM=64Gi
TIMEOUT=12h; KEEP=false; ALLOW_CPU=false; DRY=""; NAS=/volume4/dsta/rl-planner; PRIORITY=NORMAL
REPO="${RLP_REPO_URL:-}"; BRANCH="${RLP_BRANCH:-main}"; FETCH=""; EXTRA_ENV=()
while [ $# -gt 0 ]; do case "$1" in
  --job) JOB=$2; shift 2;; --tag) TAG=$2; shift 2;; --args) ARGS=$2; shift 2;; --scenes) SCENES=$2; shift 2;;
  --pool) POOL=$2; shift 2;; --cpu) CPU=$2; shift 2;; --gpu) GPU=$2; shift 2;; --mem) MEM=$2; shift 2;;
  --timeout) TIMEOUT=$2; shift 2;; --nas) NAS=$2; shift 2;; --priority) PRIORITY=$2; shift 2;;
  --repo) REPO=$2; shift 2;; --fetch) FETCH=$2; shift 2;; --env) EXTRA_ENV+=("$2"); shift 2;; --branch) BRANCH=$2; shift 2;;
  --keep-alive) KEEP=true; shift;; --allow-cpu) ALLOW_CPU=true; shift;; --dry-run) DRY="--dry-run"; shift;;
  *) echo "unknown arg $1"; exit 1;; esac; done
[ -n "$TAG" ] || TAG="${JOB}_$(date -u +%Y%m%d_%H%M%S)"
RSYNC_OPT=()
if [ -z "$REPO" ]; then
  SHIP="/tmp/rlp_ship/RL-planner"
  rm -rf /tmp/rlp_ship && mkdir -p "$SHIP"
  rsync -a --exclude .venv --exclude runs --exclude data --exclude .git --exclude '__pycache__' \
        --exclude '.pytest_cache' --exclude 'zz_ship_complete' "$HERE/" "$SHIP/"
  date -u +%FT%TZ > "$SHIP/zz_ship_complete"
  echo "[submit] no --repo: shipping snapshot $(du -sh "$SHIP" | cut -f1) via --rsync (needs rsync enabled on the deployment)"
  RSYNC_OPT=(--rsync "$SHIP:/osmo/run/workspace/RL-planner")
else
  echo "[submit] pod will git clone $REPO @ $BRANCH"
  if [ -n "$(git -C "$HERE" status --porcelain 2>/dev/null)" ]; then echo "[submit] WARNING: uncommitted local changes will not be in the pod"; fi
fi
# resources/timeout are patched into a temp copy of the workflow (OSMO --set needs {{ }} fields)
WF=$(mktemp /tmp/rlp_wf_XXXX.yaml)
sed -e "s/^      cpu: .*/      cpu: $CPU/" -e "s/^      gpu: .*/      gpu: $GPU/" -e "s/^      memory: .*/      memory: $MEM/" \
    -e "s/^    exec_timeout: .*/    exec_timeout: $TIMEOUT/" -e "s/^  name: rlplanner-train/  name: rlp-$JOB-$TAG/" \
    "$HERE/osmo/workflow.yaml" > "$WF"
set -x
osmo workflow submit "$WF" --pool "$POOL" --priority "$PRIORITY" $DRY \
  "${RSYNC_OPT[@]}" \
  --set-env "RLP_REPO_URL=$REPO" "RLP_BRANCH=$BRANCH" "RLP_JOB=$JOB" "RLP_ARGS=$ARGS" "RLP_SCENES=$SCENES" "RLP_RUN_TAG=$TAG" "RLP_NAS_DEST=$NAS" \
            "RLP_KEEP_ALIVE=$KEEP" "RLP_ALLOW_CPU=$ALLOW_CPU" "RLP_FETCH=$FETCH" "${EXTRA_ENV[@]}"
set +x
echo "[submit] follow:  osmo workflow logs <id> -f      cancel: osmo workflow cancel <id>"
echo "[submit] results: ${NAS}/${TAG}/runs/ on airlab-storage"
