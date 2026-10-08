#!/usr/bin/env bash
# Retrain every main-paper model from scratch: 6 models x 3 seeds x 300 epochs.
#
# Roughly 25-30 min per run on an Apple M4 Max (MPS); 18 runs in total.
# Runs are independent, so set JOBS>1 to train several at once.
# Each run writes outputs/<model>/seed_<S>/{best.pt,accuracy_results.json,training_curves.png},
# which reproduce/evaluate_main.sh then picks up.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"0 5 10"}
EPOCHS=${EPOCHS:-300}
JOBS=${JOBS:-1}

runs=()
for s in $SEEDS; do
  runs+=("scripts/train_goal_predictor.py           --epochs $EPOCHS --seed $s --out outputs/goal_predictor")
  runs+=("scripts/train_context_goal_predictor.py   --epochs $EPOCHS --seed $s --out outputs/context_goal")
  runs+=("scripts/train_context_goal_predictor_aux.py --epochs $EPOCHS --seed $s --out outputs/context_goal_aux")
  runs+=("scripts/train_bdi_bottleneck.py           --epochs $EPOCHS --seed $s --out outputs/bdi_bottleneck")
  runs+=("scripts/train_bdi_bottleneck_ablation.py  --epochs $EPOCHS --seed $s --ablation no_desire")
  runs+=("scripts/train_bdi_bottleneck_ablation.py  --epochs $EPOCHS --seed $s --ablation no_belief")
done

mkdir -p logs
for cmd in "${runs[@]}"; do
  name=$(echo "$cmd" | awk '{print $1}' | xargs basename .py)
  tag="$name$(echo "$cmd" | grep -o -- '--ablation [a-z_]*' | tr -d ' -' || true)_seed$(echo "$cmd" | grep -o -- '--seed [0-9]*' | awk '{print $2}')"
  echo "== $tag"
  $PY $cmd > "logs/$tag.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
done
wait
echo "Done. Now run reproduce/evaluate_main.sh"
