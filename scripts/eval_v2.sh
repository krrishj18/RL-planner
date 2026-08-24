#!/usr/bin/env bash
# Evaluate a checkpoint and the heuristics on the v2 downtown set (area-scaled teams/horizons).
#   scripts/eval_v2.sh <ckpt> [episodes]
set -euo pipefail
CKPT="${1:?checkpoint path}"; EP="${2:-12}"; W="${RLP_WORKERS_RESOLVED:-10}"
for P in "$CKPT" ray_follower nearest_frontier random; do
  name=$(basename "${P%.pt}"); [ -f "$P" ] || name="$P"
  echo "=== $name on v2 heldout ==="
  uv run python scripts/eval_policy.py --policy "$P" --stochastic --scenes data/scenes_v2/*.json --split heldout \
    --robots auto --t-max auto --episodes "$EP" --workers "$W" --out "runs/eval_v2_${name}.csv" | tail -12
done
