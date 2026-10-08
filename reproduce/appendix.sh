#!/usr/bin/env bash
# Appendix diagnostics: sparse/dense false-belief regimes, belief-channel diagnostics,
# same-budget robustness sweeps, and trajectory-level counterfactuals
# (paper Tables 12-13 and 16-19, plus the channel-use numbers in Appendix F).
#
# These experiments use their own 30-epoch model families, separate from the 300-epoch
# main-paper models. The robustness and model-planned commands match the recorded runs;
# the dense-split, base-family and per-stage diagnostic invocations are reconstructed
# from script defaults and run notes (marked "reconstructed" below).
#
# Run stage by stage: STAGES="data train diagnostics robustness aggregate trajectory tables"
# (default: all). "robustness" retrains the sweeps; "aggregate" only evaluates their checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
STAGES=${STAGES:-"data train diagnostics robustness aggregate trajectory tables"}
D=outputs/diagnostics
DENSE_SPLITS=$D/dense/splits
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has data; then
  echo "== Dense false-belief dataset (reconstructed: shortest-path routing, dense configs)"
  $PY scripts/run_simulation.py --belief-config config/belief_configs_dense.json \
    --router-k 1 --out $DENSE_SPLITS
fi

if has train; then
  echo "== Sparse 30-epoch model family, seeds 0/5/10 (reconstructed)"
  for s in 0 5 10; do
    $PY scripts/train_bdi_bottleneck.py            --epochs 30 --seed $s --out $D/sparse/bdi_bottleneck
    $PY scripts/train_context_goal_predictor.py    --epochs 30 --seed $s --out $D/sparse/context_goal
    $PY scripts/train_context_goal_predictor_aux.py --epochs 30 --seed $s --out $D/sparse/context_goal_aux
    $PY scripts/train_bdi_bottleneck_ablation.py   --epochs 30 --seed $s --ablation no_belief --out $D/sparse/no_belief
    $PY scripts/train_bdi_bottleneck_ablation.py   --epochs 30 --seed $s --ablation no_desire --out $D/sparse/no_desire
  done
  echo "== Dense 30-epoch models, seed 0 (reconstructed)"
  $PY scripts/train_bdi_bottleneck.py             --epochs 30 --seed 0 --splits $DENSE_SPLITS --out $D/dense/bdi_bottleneck
  $PY scripts/train_context_goal_predictor_aux.py --epochs 30 --seed 0 --splits $DENSE_SPLITS --out $D/dense/context_goal_aux_recovered
fi

if has diagnostics; then
  echo "== Belief diagnostics on the sparse seed-0 BDIBottleneck (Table 12, Appendix F channel use)"
  $PY scripts/diag_belief_attribution.py
  $PY scripts/diag_belief_channel_usage.py
  $PY scripts/diag_belief_dose_response.py
  $PY scripts/diag_belief_sensitivity.py
  $PY scripts/eval_aux_baseline.py
  echo "== Same diagnostics on the dense seed-0 BDIBottleneck (reconstructed)"
  DB=$D/dense/bdi_bottleneck/seed_0/best.pt
  $PY scripts/diag_dense_false_beliefs.py
  $PY scripts/diag_belief_attribution.py   --bnk-ckpt $DB --splits-dir $DENSE_SPLITS --out-dir $D/analyses/dense_current_belief_attribution
  $PY scripts/diag_belief_channel_usage.py --bnk-ckpt $DB --splits-dir $DENSE_SPLITS --out-dir $D/analyses/dense_belief_channel_usage
  $PY scripts/diag_belief_dose_response.py --bnk-ckpt $DB --splits-dir $DENSE_SPLITS --out-dir $D/analyses/dense_belief_dose_response
  $PY scripts/diag_belief_sensitivity.py   --bnk-ckpt $DB --splits-dir $DENSE_SPLITS --out-dir $D/analyses/dense_belief_sensitivity
  $PY scripts/eval_aux_baseline.py --bnk-ckpt $DB --splits-dir $DENSE_SPLITS \
    --aux-ckpt $D/dense/context_goal_aux_recovered/seed_0/best.pt --out-dir $D/analyses/dense_aux_supervised_baseline
fi

if has robustness; then
  echo "== Same-budget robustness sweeps (Table 13): sparse 4 configs x 5 seeds, dense 2 configs x 3 seeds"
  $PY scripts/robustness_sweep.py --phase sparse
  $PY scripts/robustness_sweep.py --phase dense --splits-dir $DENSE_SPLITS \
    --configs default_weighted balanced_focal --seeds 0 5 10
fi

if has aggregate; then
  echo "== Evaluate the robustness checkpoints (Table 13)"
  $PY scripts/robustness_aggregate.py --phase sparse
  $PY scripts/robustness_aggregate.py --phase dense --splits-dir $DENSE_SPLITS
fi

if has trajectory; then
  echo "== Simulator-resampled and model-planned counterfactual trajectories (Tables 16-19)"
  $PY scripts/cf_simulator_rollouts.py
  $PY scripts/cf_model_planned_paths.py
  # The dense runs were made in 120-record chunks per split, then merged. Each chunk restarts
  # the simulator RNG, so the same chunking is needed to reproduce the reported numbers.
  for script in cf_simulator_rollouts cf_model_planned_paths; do
    case $script in
      cf_simulator_rollouts)  out=$D/analyses/dense_true_counterfactual_rollouts ;;
      cf_model_planned_paths) out=$D/analyses/dense_model_planned_counterfactual_paths ;;
    esac
    chunks=$D/analyses/_chunks_$script   # outside $out: the merge skips files under $out
    for split in test test_new_agent; do
      for start in 0 120 240 360 480 600 720 840; do
        $PY scripts/$script.py --bnk-ckpt $D/dense/bdi_bottleneck/seed_0/best.pt \
          --splits-dir $DENSE_SPLITS --splits $split --router-k 1 \
          --record-start $start --record-limit 120 --chunk-label ${split}_${start} \
          --out-dir $chunks/${split}_${start}
      done
    done
    $PY scripts/$script.py --merge-chunks $chunks --out-dir $out
    rm -rf "$chunks"
  done
  $PY scripts/robustness_model_planned.py --phase sparse
  $PY scripts/robustness_model_planned.py --phase dense --splits-dir $DENSE_SPLITS --seeds 0 5 10
fi

if has tables; then
  echo "== Bootstrap CIs and the paper's trajectory-counterfactual tables"
  $PY scripts/cf_bootstrap_ci.py
  $PY scripts/make_trajectory_cf_tables.py
fi
echo "Done. Compare with: git status outputs/diagnostics results/trajectory_counterfactuals"
