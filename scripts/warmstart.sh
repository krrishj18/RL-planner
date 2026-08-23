#!/usr/bin/env bash
# Imitation warm-start then PPO fine-tune for one variant (GPU).
#   scripts/warmstart.sh <variant> [teacher] [iters] [steps] [ft_updates]
# Produces runs/ws_<variant>/{bc,ft}/ and a stochastic + greedy held-out eval of both.
set -euo pipefail
VARIANT="${1:-central_full}"; TEACHER="${2:-oracle_sweep}"; ITERS="${3:-6}"; STEPS="${4:-250}"; FT="${5:-200}"
W="${RLP_WORKERS_RESOLVED:-10}"; SCENES="${WS_SCENES:-synthetic:0-200}"; ROBOTS="${WS_ROBOTS:-3}"; EP="${WS_EVAL_EPISODES:-16}"
OUT="runs/ws_${VARIANT}"; mkdir -p "$OUT"
echo "[warmstart] variant=$VARIANT teacher=$TEACHER iters=$ITERS steps=$STEPS ft=$FT workers=$W"
uv run python -u scripts/imitate.py --name "ws_${VARIANT}/bc" --variant "$VARIANT" --teacher "$TEACHER" \
  --scenes $SCENES --robots "$ROBOTS" --iters "$ITERS" --steps "$STEPS" --envs 32 --workers "$W" \
  --epochs 3 --batch 64 --lr 5e-4 --max-gb 24 --eval-episodes "$EP" --eval-every 1 --device cuda --seed 0
uv run python -u scripts/train.py --name "ws_${VARIANT}/ft" --init-from "runs/ws_${VARIANT}/bc/bc.pt" \
  --scenes $SCENES --robots "$ROBOTS" --envs 32 --workers "$W" --rollout 64 --updates "$FT" \
  --lr 1e-4 --critic-warmup 10 --actor-warmup 20 --bc-kl 0.05 --eval-every 25 --eval-episodes "$EP" --device cuda --seed 0
for mode in "" "--stochastic"; do
  for ck in "runs/ws_${VARIANT}/bc/bc.pt" "runs/ws_${VARIANT}/ft/latest.pt"; do
    uv run python scripts/eval_policy.py --policy "$ck" $mode --variant "$VARIANT" --scenes $SCENES --split heldout \
      --episodes 24 --workers "$W" --out "${ck%.pt}_eval${mode:+_sampled}.csv" | tail -3
  done
done
echo "[warmstart] done -> $OUT"
