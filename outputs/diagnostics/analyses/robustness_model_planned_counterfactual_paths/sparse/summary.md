# Robustness Model-Planned Counterfactual Path Summary

- Configs: default_weighted, balanced_focal.
- Per-seed metric rows: 240.
- Config aggregate rows: 48.

## Overall By Config

- `default_weighted`: simulator top-1=0.0876, simulator top-5=0.3441, goal-change vs original=0.8543, node-overlap original=0.2611.
- `balanced_focal`: simulator top-1=0.0767, simulator top-5=0.3255, goal-change vs original=0.8633, node-overlap original=0.2483.

These rows use frozen BDIBottleneck checkpoints only. AuxContext is excluded because it does not expose the same explicit bottleneck intervention interface.
