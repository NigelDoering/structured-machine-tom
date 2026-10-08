# Robustness Summary

- Dense confirmation selected config: `balanced_focal` (within_goal_tolerance).

## Key Config Means

- AuxSupervisedContextGoalPredictor / balanced_bce / test: top1=0.2487±0.0192, closed_f1=0.3881±0.0012.
- BDIBottleneck / balanced_bce / test: top1=0.1737±0.0202, closed_f1=0.3854±0.0046.
- AuxSupervisedContextGoalPredictor / balanced_focal / test: top1=0.2479±0.0082, closed_f1=0.3887±0.0040.
- BDIBottleneck / balanced_focal / test: top1=0.1950±0.0522, closed_f1=0.3872±0.0077.
- AuxSupervisedContextGoalPredictor / default_weighted / test: top1=0.2525±0.0133, closed_f1=0.3992±0.0055.
- BDIBottleneck / default_weighted / test: top1=0.1958±0.0358, closed_f1=0.3918±0.0060.
- AuxSupervisedContextGoalPredictor / low_lr_weighted / test: top1=0.1562±0.0095, closed_f1=0.4018±0.0029.
- BDIBottleneck / low_lr_weighted / test: top1=0.1050±0.0237, closed_f1=0.4030±0.0031.

## Paired BDIBottleneck minus AuxContext

- balanced_bce / test: Δtop1=-0.0750±0.0039, Δclosed_f1=-0.0027±0.0043.
- balanced_focal / test: Δtop1=-0.0529±0.0546, Δclosed_f1=-0.0014±0.0093.
- default_weighted / test: Δtop1=-0.0567±0.0407, Δclosed_f1=-0.0074±0.0037.
- low_lr_weighted / test: Δtop1=-0.0512±0.0256, Δclosed_f1=0.0011±0.0044.

This summary is diagnostic and reports same-budget robustness, not tuned winner selection.
