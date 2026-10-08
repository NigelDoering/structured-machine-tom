# Model-Planned Counterfactual Path Summary

- Model-planned path rows: 11520.
- Successful model-planned routes: 11520.
- Goal-change rate vs original goal: 0.8465.
- Goal-change rate vs normal model top-1: 0.4698.
- Mean node overlap with original future suffix: 0.2891.
- Mean edge overlap with original future suffix: 0.2479.
- Simulator comparison success rate: 1.0000.
- Simulator counterfactual goal top-1 agreement: 0.1005.
- Simulator counterfactual goal top-5 agreement: 0.3757.

This diagnostic freezes BDIBottleneck, edits belief/desire channels at an observed prefix, lets the model pick a counterfactual goal, and routes from the intervention node to that model-planned goal.
