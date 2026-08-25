#!/usr/bin/env bash
# Pod entrypoint for RL-planner jobs on OSMO. Runs inside airstack-osmo-workspace
# (Ubuntu 24.04, python3.12, rsync, sshpass) as plain (non-DinD) task.
set -uo pipefail
log() { echo "[rlp] $(date -u +%H:%M:%S) $*"; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
JOB="${RLP_JOB:-train}"; ARGS="${RLP_ARGS:-}"; TAG="${RLP_RUN_TAG:-job}"
NAS_DEST="${RLP_NAS_DEST:-/volume4/dsta/rl-planner}"; SYNC_MIN="${RLP_SYNC_MIN:-10}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
mkdir -p runs data/scenes data/scenes_v2
JOBLOG="runs/osmo_${TAG}.log"
# no process substitution / background tee: anything holding the task's stdout after the
# job ends keeps the OSMO task RUNNING until exec_timeout
log() { echo "[rlp] $(date -u +%H:%M:%S) $*" | tee -a "$JOBLOG"; }
log "job=$JOB tag=$TAG args=[$ARGS] scenes=${RLP_SCENES:-synthetic} host=$(hostname) nproc=$(nproc)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || log "nvidia-smi unavailable"

# ---- NAS upload ---------------------------------------------------------------------------
NAS_OK=false
if [ -n "${AIRLAB_STORAGE_USER:-}" ] && [ -n "${AIRLAB_STORAGE_PASS:-}" ]; then
  command -v sshpass >/dev/null || { apt-get update -qq && apt-get install -y -qq sshpass rsync; }
  NAS_OK=true
fi
NAS_HOST="${AIRLAB_STORAGE_HOST:-airlab-storage.andrew.cmu.edu}"
STAGE=/tmp/rlp_stage; REL="$(basename "$NAS_DEST")/$TAG"
mkdir -p "$STAGE/$REL"; ln -sfn "$ROOT/runs" "$STAGE/$REL/runs"
nas_sync() {
  $NAS_OK || return 0
  local dest_parent; dest_parent="$(dirname "$NAS_DEST")"
  ( cd "$STAGE" && SSHPASS="$AIRLAB_STORAGE_PASS" sshpass -e rsync -rltzRL --partial --timeout=900 \
      --no-perms --no-owner --no-group --omit-dir-times \
      --exclude '*.pt.tmp' \
      -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
      "./$REL/runs/" "${AIRLAB_STORAGE_USER}@${NAS_HOST}:${dest_parent}/" ) \
    && log "synced runs/ -> ${NAS_HOST}:${dest_parent}/${REL}/runs/" \
    || log "WARN: NAS sync failed (rc=$?)"
}
$NAS_OK && log "NAS upload enabled -> ${NAS_HOST}:${NAS_DEST}/${TAG}/runs/" || log "NAS upload disabled (no airlab-storage credential)"

# ---- optional: pull a previous run's outputs back from the NAS (RLP_FETCH=tag[,tag2]) ------
if [ -n "${RLP_FETCH:-}" ] && $NAS_OK; then
  for t in ${RLP_FETCH//,/ }; do
    log "fetching ${NAS_DEST}/${t}/runs/ from the NAS"
    SSHPASS="$AIRLAB_STORAGE_PASS" sshpass -e rsync -rltz --partial --timeout=900 \
      -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
      "${AIRLAB_STORAGE_USER}@${NAS_HOST}:${NAS_DEST}/${t}/runs/" "runs/" \
      && log "fetched ${t}" || log "WARN: fetch of ${t} failed"
  done
fi

# ---- environment ------------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version
log "uv sync (torch cu128 from the lockfile)"
uv sync --frozen --extra dev 2>&1 | tail -3; [ "${PIPESTATUS[0]}" = 0 ] || { log "ERROR: uv sync failed"; nas_sync; exit 3; }
GPU_OK=$(uv run python -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
if [ "$GPU_OK" = "1" ]; then
  uv run python -c "import torch; print('[rlp] torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"
elif [ "${RLP_ALLOW_CPU:-false}" = "true" ]; then
  log "WARN: no CUDA device; continuing on CPU (RLP_ALLOW_CPU=true)"
else
  log "ERROR: torch.cuda.is_available() is False on this pod; set RLP_ALLOW_CPU=true to run on CPU"; nas_sync; exit 4
fi

# ---- scenes (deterministic; regenerated rather than shipped) ---------------------------------
case "${RLP_SCENES:-synthetic}" in
  *v1*) log "exporting v1 scenes"; uv run python scripts/export_scenes.py --preset earthquake tornado explosion --seeds 0:80 --region 400 400 --size-jitter 0.25 --casualties auto --bystanders auto --out data/scenes 2>&1 | tail -1 ;;
esac
case "${RLP_SCENES:-synthetic}" in
  *v2*) log "exporting v2 scenes"; uv run python scripts/export_scenes.py --pipeline v2 --locale downtown --disaster earthquake tornado explosion --severity-range ${RLP_V2_SEV:-0.5 1.0} --seeds "${RLP_V2_SEEDS:-0:60}" --region-range 500 1500 --size-jitter 0.25 --casualties auto --bystanders auto --out data/scenes_v2 2>&1 | tail -1 ;;
esac

# ---- job ----------------------------------------------------------------------------------
# worker count from the cgroup CPU quota (nproc reports the whole node on OSMO)
cpu_quota() {
  if [ -f /sys/fs/cgroup/cpu.max ]; then read -r q per < /sys/fs/cgroup/cpu.max; [ "$q" != "max" ] && { echo $(( q / per )); return; }; fi
  nproc
}
WORKERS="${RLP_WORKERS:-auto}"; [ "$WORKERS" = "auto" ] && WORKERS=$(( $(cpu_quota) - 2 )); [ "$WORKERS" -lt 1 ] && WORKERS=1
export RLP_WORKERS_RESOLVED="$WORKERS"
case "$JOB" in
  sweep|train|imitate) ARGS="$ARGS --workers $WORKERS" ;;
esac
# periodic sync: detached from the task's stdout pipe (a lingering child holding the
# pipe keeps the OSMO task RUNNING after the job exits), logs only to the job log
export NAS_OK STAGE REL NAS_HOST NAS_DEST JOBLOG AIRLAB_STORAGE_USER AIRLAB_STORAGE_PASS
setsid bash -c "$(declare -f log nas_sync); while true; do sleep $(( SYNC_MIN * 60 )); nas_sync; done" \
  </dev/null >/dev/null 2>&1 &
SYNC_PID=$!
if [ -f "scripts/${JOB}.sh" ]; then
  log "running: bash scripts/${JOB}.sh $ARGS  (workers=$WORKERS)"
  bash "scripts/${JOB}.sh" $ARGS 2>&1 | tee -a "$JOBLOG"; RC=${PIPESTATUS[0]}
else
  log "running: uv run python scripts/${JOB}.py $ARGS"
  uv run python "scripts/${JOB}.py" $ARGS 2>&1 | tee -a "$JOBLOG"; RC=${PIPESTATUS[0]}
fi
log "job exited rc=$RC"
kill -- -"$SYNC_PID" 2>/dev/null; kill "$SYNC_PID" 2>/dev/null
nas_sync
if [ "${RLP_KEEP_ALIVE:-false}" = "true" ]; then log "keep-alive: sleeping"; exec sleep infinity; fi
exit $RC
