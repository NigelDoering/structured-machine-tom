# Robustness Model-Planned Counterfactual Path Summary

- Configs: default_weighted, balanced_focal.
- Per-seed metric rows: 144.
- Config aggregate rows: 48.

## Overall By Config

- `default_weighted`: simulator top-1=0.1378, simulator top-5=0.4065, goal-change vs original=0.8460, node-overlap original=0.3310.
- `balanced_focal`: simulator top-1=0.1455, simulator top-5=0.4183, goal-change vs original=0.8442, node-overlap original=0.3297.

These rows use frozen BDIBottleneck checkpoints only. AuxContext is excluded because it does not expose the same explicit bottleneck intervention interface.
