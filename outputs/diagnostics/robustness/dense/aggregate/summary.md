# Robustness Summary

- Dense confirmation selected config: `balanced_focal` (within_goal_tolerance).

## Key Config Means

- AuxSupervisedContextGoalPredictor / balanced_focal / test: top1=0.3118±0.0079, closed_f1=0.5289±0.0022.
- BDIBottleneck / balanced_focal / test: top1=0.3069±0.0364, closed_f1=0.5247±0.0048.
- AuxSupervisedContextGoalPredictor / default_weighted / test: top1=0.3017±0.0095, closed_f1=0.5242±0.0017.
- BDIBottleneck / default_weighted / test: top1=0.3049±0.0644, closed_f1=0.5233±0.0027.

## Paired BDIBottleneck minus AuxContext

- balanced_focal / test: Δtop1=-0.0049±0.0398, Δclosed_f1=-0.0042±0.0069.
- default_weighted / test: Δtop1=0.0031±0.0569, Δclosed_f1=-0.0009±0.0031.

This summary is diagnostic and reports same-budget robustness, not tuned winner selection.
