#!/usr/bin/env bash
# Regenerate every main-paper figure and table from trained checkpoints.
#
# Expects the main-paper checkpoints under outputs/ (see README, "Checkpoints").
# Evaluation is deterministic: with the released checkpoints, every output
# below should be identical to the committed version (check with `git status`).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

# The released seed-0 checkpoints sit at outputs/<model>/ (the original seed-0 runs were
# launched without --seed); retrained ones from reproduce/train_main.sh sit in seed_0/.
seed0_dir() {
  if [ -f "outputs/$1/seed_0/best.pt" ]; then echo "outputs/$1/seed_0"; else echo "outputs/$1"; fi
}
seed_dirs() {
  echo "$(seed0_dir "$1") outputs/$1/seed_5 outputs/$1/seed_10"
}
seeds() {  # best.pt for the three paper seeds
  for d in $(seed_dirs "$1"); do echo -n "$d/best.pt "; done
}

echo "== Figure 2: goal inference (GoalPredictor, ContextGoalPredictor, BDIBottleneck, baselines)"
$PY scripts/eval_goal_inference.py --paper \
  --goal-ckpt $(seeds goal_predictor) \
  --ctx-ckpt  $(seeds context_goal) \
  --bnk-ckpt  $(seeds bdi_bottleneck)

echo "== Merge AuxContext into Figure 2, Table 1 (held-out beliefs) and Table 20 (10% increments)"
# AuxContext test/new-agent curves come from outputs/context_goal_aux/seed_*/accuracy_results.json;
# its held-out-belief columns come from results/goal_inference/aux_context_goal_quarters.csv,
# which is regenerated here when the AuxContext checkpoints are present.
if [ -f outputs/context_goal_aux/seed_0/best.pt ]; then
  $PY scripts/eval_aux_context_bh.py
fi
$PY scripts/make_goal_inference_figures.py --paper

echo "== Tables 21-22: McNemar significance"
$PY scripts/eval_mcnemar_goal_inference.py

echo "== Figure 6: ablations (NoDesire / NoBelief)"
$PY scripts/eval_ablations.py \
  --bdi       $(seed_dirs bdi_bottleneck) \
  --no-desire $(seed_dirs ablation_no_desire) \
  --no-belief $(seed_dirs ablation_no_belief)

echo "== Figure 3, Table 2, Figures 12-19: desire counterfactuals"
$PY scripts/eval_counterfactual_desire.py --paper

echo "== Figure 4, Table 3, Tables 14-15, Figure 9: belief counterfactuals"
$PY scripts/eval_counterfactual_belief.py --paper

echo "== Figure 8: belief-head sensitivity by evidence"
$PY scripts/eval_belief_evidence.py

echo "== Figure 7: distractor robustness"
$PY scripts/eval_distractor.py --paper

echo "Done. Compare against the paper's reference outputs with: git status results/"
